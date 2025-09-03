# components/callbacks.py
import logging
import os
import wandb
import torch
import torch.distributed as dist
from time import perf_counter
from geqtrain.train.components.epoch_summary import EpochSummary
from geqtrain.utils import atomic_write_group
from geqtrain.train._key import TRAIN, VALIDATION, ABBREV
from geqtrain.utils.wandb import init_n_update_wandb, resume_wandb_run

class Callback:
    """Base class for callbacks."""
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_trainer_begin(self, **kwargs): pass
    def on_trainer_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_train_begin(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
    def on_validation_begin(self, **kwargs): pass
    def on_validation_end(self, **kwargs): pass

class Logger(Callback):
    """Handles all logging to the console and local files."""
    def __init__(self, wandb_is_on=False):
        self.wandb_is_on = wandb_is_on

    def on_trainer_begin(self):
        if self.trainer.dist.is_master:
            # Logic moved from Trainer.init_log()
            self.trainer.logger.info("! Starting training ...")
            if self.trainer.config.get('restart'):
                self.trainer.logger.info("! Resuming from checkpoint...")

    def on_trainer_end(self):
        if self.trainer.dist.is_master:
            # Logic moved from Trainer.final_log()
            self.trainer.logger.info(f"! Stop training: {self.trainer.stop_arg}")
            self.trainer.logger.info(f"Cumulative wall time: {self.trainer.cumulative_wall}")

    # == Timing Hooks ==
    def on_train_begin(self):
        if self.trainer.dist.is_master:
            self.trainer._phase_start_time = perf_counter()

    def on_train_end(self):
        if self.trainer.dist.is_master:
            self.trainer.train_wall = perf_counter() - self.trainer._phase_start_time

    def on_validation_begin(self):
        if self.trainer.dist.is_master:
            self.trainer._phase_start_time = perf_counter()

    def on_validation_end(self):
        if self.trainer.dist.is_master:
            self.trainer.validation_wall = perf_counter() - self.trainer._phase_start_time
            self.trainer.cumulative_wall += self.trainer.train_wall + self.trainer.validation_wall

    def on_epoch_end(self, summary: EpochSummary):
        if self.trainer.dist.is_master:
            self._log_epoch(summary)

    def on_batch_end(self):
        if self.trainer.dist.is_master:
            self._log_batch(self.trainer.batch_type)

    def _log_batch(self, batch_type):
        logger = self.trainer.logger
        batch_logger = logging.getLogger(self.trainer.batch_log[batch_type])

        mat_str = f"{self.trainer.iepoch+1:5d}, {self.trainer.ibatch+1:5d}"
        log_str = f" {self.trainer.iepoch+1:8d} {self.trainer.ibatch+1:8d}"
        header = "epoch, batch"
        log_header = " " + " ".join([f"{s:>8s}" for s in ["Epoch", "Batch"]])

        for name, value in self.trainer.batch_losses.items():
            mat_str += f", {value:16.5g}"
            header += f", {name:>12.12}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12.12}"

        metrics = self.trainer.metrics.flatten_metrics(metrics=self.trainer.batch_metrics, metrics_metadata=self.trainer.metrics_metadata)
        for key, value in metrics.items():
            mat_str += f", {value:16.5g}"
            header += f", {key:>12.12}"
            log_str += f" {value:12.3g}"
            log_header += f" {key:>12.12}"

        if self.trainer.ibatch == 0:
            batch_logger.info(header)
        batch_logger.info(mat_str)

        if self.trainer.ibatch == 0:
            logger.info(f"\n### {batch_type}")
            logger.info(log_header)
        if (self.trainer.ibatch + 1) % self.trainer.log_batch_freq == 0 or (self.trainer.ibatch + 1) == self.trainer.n_batches:
            logger.info(log_str)

    def _log_epoch(self, summary: EpochSummary):
        # 1. Get the flat dictionary containing all values
        mae_dict = summary.to_flat_dict()
        epoch = summary.epoch

        # Build console logs for each category separately
        log_headers_console = {}
        log_strs_console = {}

        categories = [TRAIN, VALIDATION] if epoch > 0 else [VALIDATION]
        for category in categories:
            wall = mae_dict["train_wall"] if category == TRAIN else mae_dict["validation_wall"]
            header = " ".join([f"{s:>12s}" for s in ["Epoch", "LR", "Wall"]])
            log_str = f"{epoch:12d} {mae_dict['LR']:12.3g} {wall:12.3f}"

            # 2. Get the specific list of loss keys from the summary
            for key in summary.get_loss_keys(category):
                full_key = f'{category}_{key}'
                header  += f" {key:>12.12}"
                log_str += f" {mae_dict.get(full_key, 0.0):12.3g}"

            # 3. Get the specific list of flattened metric keys from the summary
            for key in summary.get_flattened_metric_keys(category):
                header  += f" {key:>12.12}"
                log_str += f" {mae_dict.get(f'{category}_{key}', 'N.A.'):12.3g}"

            log_headers_console[category] = header
            log_strs_console[category] = log_str

        # Print to console
        header_to_print = log_headers_console[VALIDATION]
        self.trainer.logger.info("\n\n#################### " + header_to_print)
        if epoch > 0:
            self.trainer.logger.info("! Train              " + log_strs_console[TRAIN])
            self.trainer.logger.info("! Validation         " + log_strs_console[VALIDATION])
        else:
            self.trainer.logger.info("! Initial Validation " + log_strs_console[VALIDATION])
        self.trainer.logger.info(f"Cumulative wall time: {mae_dict['cumulative_wall']:.4f}s")

        # Save to CSV file
        epoch_logger = logging.getLogger(self.trainer.epoch_log)
        if epoch == 1 or (epoch == 0 and len(categories) > 0):
            epoch_logger.info(",".join(mae_dict.keys()))

        csv_values = [f"{v:.5g}" if isinstance(v, float) else str(v) for v in mae_dict.values()]
        epoch_logger.info(",".join(csv_values))

class WandbCallback(Callback):
    """Handles all logging and initialization for Weights & Biases."""
    def on_trainer_begin(self):
        """Initialize a new or resume an existing WandB run on the master rank only."""
        # This is the critical guard. Only the master process should interact with wandb.
        if not self.trainer.dist.is_master:
            # Force wandb to be disabled on all non-master processes
            os.environ["WANDB_MODE"] = "disabled"
            return

        # Use the provided helper functions from wandb.py for the master process
        if self.trainer.config.get('restart'):
            resume_wandb_run(self.trainer.config)
        else:
            # This function also updates the config with any sweeps
            updated_config = init_n_update_wandb(self.trainer.config)
            self.trainer.config.update(updated_config)

        if self.trainer.config.get("wandb_watch", False):
            wandb.watch(self.trainer.model, **self.trainer.config.get("wandb_watch_kwargs", {}))

    def on_epoch_end(self, summary: EpochSummary):
        """Log the epoch's metrics to WandB from the master rank only."""
        if self.trainer.dist.is_master and wandb.run is not None:
            wandb.log(summary.to_flat_dict())
            
            # Log each individual norm value for more detailed plots
            for norm in summary.grad_norms:
                wandb.log({"train_gradient_norm_step": norm})
            for norm_clip in summary.grad_norms_clip:
                wandb.log({"train_gradient_norm_clip_value_step": norm_clip})

            for key, norm_list in summary.node_feature_norms.items():
                for norm in norm_list:
                    wandb.log({f"train_{key}_step": norm.item()})

class CheckpointCallback(Callback):
    def on_epoch_end(self, summary: EpochSummary):
        if not self.trainer.dist.is_master: return
        
        current_metrics = summary.get_target_metric(self.trainer.metrics_key)
        if current_metrics is None:
            return
        
        with atomic_write_group():
            is_improved = (current_metrics < self.trainer.best_metrics if self.trainer.metric_criteria == 'decreasing' else current_metrics > self.trainer.best_metrics)
            if is_improved:
                self.trainer.best_metrics = current_metrics
                self.trainer.best_epoch = self.trainer.iepoch + 1
                self.trainer.checkpoint_handler.save_ema_model(self.trainer.best_model_path, blocking=False)
                self.trainer.logger.info(f"! Best model saved at epoch {self.trainer.best_epoch} with metric {self.trainer.best_metrics:.3f}")
            if (self.trainer.iepoch + 1) % self.trainer.log_epoch_freq == 0:
                self.trainer.checkpoint_handler.save(blocking=False)
            if self.trainer.save_checkpoint_freq > 0 and (self.trainer.iepoch + 1) % self.trainer.save_checkpoint_freq == 0:
                ckpt_path = self.trainer.output.generate_file(f"ckpt{self.trainer.iepoch+1}.pth")
                self.trainer.checkpoint_handler.save_ema_model(ckpt_path, blocking=False)

class EarlyStoppingCallback(Callback):
    """Handles early stopping with proper DDP synchronization."""
    def on_epoch_end(self, summary: EpochSummary):
        should_stop = False
        # 1. Only the master rank evaluates the stopping condition
        if self.trainer.dist.is_master:
            if self.trainer.early_stopping_conds is not None and summary is not None:
                is_over = self.trainer.iepoch >= self.trainer.warmup_epochs
                if is_over:
                    early_stop_triggered, args, _ = self.trainer.early_stopping_conds(summary.to_flat_dict())
                    if early_stop_triggered:
                        self.trainer.stop_arg = args
                        should_stop = True

            if self.trainer.iepoch + 1 >= self.trainer.max_epochs:
                self.trainer.stop_arg = "max epochs reached"
                should_stop = True

        # 2. Broadcast the decision from master to all other ranks
        if self.trainer.dist.is_distributed:
            # Create a tensor to hold the decision (1.0 for stop, 0.0 for continue)
            stop_tensor = torch.tensor([1.0 if should_stop else 0.0], device=self.trainer.dist.device)
            # Broadcast from rank 0 to all other processes
            dist.broadcast(stop_tensor, src=0)
            # All ranks now have the same decision
            if stop_tensor.item() == 1.0:
                self.trainer.should_stop = True
        else:
            # For single-GPU runs, just update the flag directly
            self.trainer.should_stop = should_stop