# components/callbacks.py
import logging
import os
import wandb
import torch
import torch.distributed as dist
from e3nn.o3 import Irreps
from time import perf_counter
import math

from geqtrain.data import AtomicDataDict
from geqtrain.nn.mace.irreps_tools import reshape_irreps
from geqtrain.train.components.epoch_summary import EpochSummary
from geqtrain.utils import atomic_write_group
from geqtrain.train._key import TRAIN, VALIDATION, ABBREV
from geqtrain.utils.wandb import init_n_update_wandb, resume_wandb_run
from geqtrain.train.grad_clipping_utils import gradient_clipping, Queue


class Callback:
    """Base class for callbacks."""
    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_trainer_begin(self, **kwargs): pass
    def on_trainer_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_after_backward(self, **kwargs): pass
    def on_before_step(self, **kwargs): pass
    def on_step_end(self, **kwargs): pass
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
    
    def on_epoch_end(self, summary: EpochSummary, run_validation: bool, **kwargs):
        if not run_validation:
            return
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

class ValidationBatchPredictionLogger(Callback):
    """Saves a single validation batch of predictions/targets to per-key CSVs."""
    def __init__(self):
        self._log_files = {}
        self._header_written = {}
        self._keys = []
        self._loss_func_by_key = {}
        self._capture_active = False
        self._pred_cache = {}
        self._ref_cache = {}
        self._destandardize_fields = {}

    def on_trainer_begin(self, **kwargs):
        if not self.trainer.dist.is_master:
            return
        loss = self.trainer.loss
        seen = set()
        for loss_key in loss.keys:
            clean_key = loss.remove_suffix(loss_key)
            if clean_key in seen:
                continue
            seen.add(clean_key)
            self._keys.append(clean_key)
            self._loss_func_by_key[clean_key] = loss.funcs[loss_key]
            safe_key = clean_key.replace("/", "_")
            log_path = self.trainer.output.open_logfile(
                f"pred_target_batch_{ABBREV[VALIDATION]}_{safe_key}.csv",
                propagate=False,
            )
            self._log_files[clean_key] = log_path
            self._header_written[clean_key] = False

        self._destandardize_fields = self.trainer.config.get('destandardize_fields', {})

    def on_validation_begin(self, **kwargs):
        if not self.trainer.dist.is_master:
            return
        self._capture_active = False
        self._pred_cache = {key: [] for key in self._keys}
        self._ref_cache = {key: [] for key in self._keys}

    def on_batch_begin(self, **kwargs):
        if not self.trainer.dist.is_master:
            return
        if self.trainer.batch_type != VALIDATION or self.trainer.ibatch != 0:
            return
        self._capture_active = True

    def on_step_end(self, batch_output=None, ref_data=None, **kwargs):
        if not self._capture_active or not self.trainer.dist.is_master:
            return
        if batch_output is None or ref_data is None:
            return
        for key in self._keys:
            pred_key, ref_key = self._extract_pred_ref(key, batch_output, ref_data)
            if pred_key is None or ref_key is None:
                continue
            self._pred_cache[key].append(pred_key.detach().cpu())
            self._ref_cache[key].append(ref_key.detach().cpu())

    def on_batch_end(self, **kwargs):
        if not self._capture_active or not self.trainer.dist.is_master:
            return
        if self.trainer.batch_type != VALIDATION or self.trainer.ibatch != 0:
            return
        self._capture_active = False
        self._flush_cache()

    def _extract_pred_ref(self, key, batch_output, ref_data):
        loss_func = self._loss_func_by_key.get(key)
        pred_key_name = key
        if loss_func is not None and hasattr(loss_func, "_get_pred_key_name"):
            pred_key_name = loss_func._get_pred_key_name(key)

        if pred_key_name not in batch_output or key not in ref_data:
            return None, None

        pred_copy = dict(batch_output)
        ref_copy = dict(ref_data)
        pred_copy[pred_key_name] = pred_copy[pred_key_name].detach().clone()
        ref_copy[key] = ref_copy[key].detach().clone()

        pred_key = pred_copy[pred_key_name]
        ref_key = ref_copy[key]

        if loss_func is not None and hasattr(loss_func, "_prepare_tensors"):
            pred_key, ref_key = loss_func._prepare_tensors(
                pred_copy,
                ref_copy,
                pred_key_name,
                key,
                mean=False,
                destandardize_fields=self._destandardize_fields,
            )

        if loss_func is not None and hasattr(loss_func, "_handle_supervision_shapes"):
            ref_key = loss_func._handle_supervision_shapes(pred_key, ref_key, pred_key_name, key)
        elif ref_key.shape != pred_key.shape:
            ref_key = ref_key.reshape(pred_key.shape)

        if loss_func is not None and hasattr(loss_func, "_apply_node_filter"):
            pred_key, ref_key = loss_func._apply_node_filter(pred_key, ref_key, ref_copy, key)

        return pred_key, ref_key

    def _flush_cache(self):
        epoch = self.trainer.iepoch + 1
        batch = self.trainer.ibatch + 1

        for key in self._keys:
            pred_chunks = self._pred_cache.get(key, [])
            ref_chunks = self._ref_cache.get(key, [])
            if not pred_chunks or not ref_chunks:
                continue

            pred = self._concat_chunks(pred_chunks)
            ref = self._concat_chunks(ref_chunks)
            pred_2d = self._to_2d(pred)
            ref_2d = self._to_2d(ref)

            if pred_2d.shape != ref_2d.shape:
                self.trainer.logger.warning(
                    f"Skipping CSV log for '{key}': prediction shape {pred_2d.shape} "
                    f"does not match target shape {ref_2d.shape}."
                )
                continue

            n_cols = pred_2d.shape[1]
            if n_cols == 1:
                pred_headers = ["pred"]
                ref_headers = ["target"]
            else:
                pred_headers = [f"pred_{i}" for i in range(n_cols)]
                ref_headers = [f"target_{i}" for i in range(n_cols)]

            if not self._header_written.get(key, False):
                header = ["epoch", "batch"] + pred_headers + ref_headers
                logging.getLogger(self._log_files[key]).info(",".join(header))
                self._header_written[key] = True

            logger = logging.getLogger(self._log_files[key])
            pred_rows = pred_2d.tolist()
            ref_rows = ref_2d.tolist()
            for pred_row, ref_row in zip(pred_rows, ref_rows):
                values = [epoch, batch] + pred_row + ref_row
                logger.info(self._format_csv_row(values))

    def _concat_chunks(self, tensors):
        prepared = []
        for tensor in tensors:
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            prepared.append(tensor)
        if len(prepared) == 1:
            return prepared[0]
        return torch.cat(prepared, dim=0)

    def _to_2d(self, tensor):
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            return tensor.unsqueeze(1)
        return tensor.reshape(tensor.shape[0], -1)

    def _format_csv_row(self, values):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.6g}")
            else:
                formatted.append(str(value))
        return ",".join(formatted)

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

class SanitizeGradCallback(Callback):
    """
    Turns NaN gradients into zeros before the optimizer step.
    This should be placed before any callback that uses gradients, like clipping.
    """
    def on_before_step(self, **kwargs):
        if not self.trainer.config.get('sanitize_gradients', False):
            return
        
        for p in self.trainer.model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, out=p.grad)

class GradientClippingCallback(Callback):
    """
    Handles gradient clipping before the optimizer step.
    Supports both 'fixed' value clipping and 'dynamic' clipping based on history.
    """
    def on_trainer_begin(self, **kwargs):
        """Initialize the queue for dynamic clipping if needed."""
        self.mode = self.trainer.config.get('gradient_clipping_mode', 'dynamic')
        self.max_gradient_norm = self.trainer.config.get('max_gradient_norm', math.inf)
        
        if self.mode == 'dynamic' and self.max_gradient_norm != math.inf:
            self.gradnorms_queue = Queue()
        else:
            self.gradnorms_queue = None

    def on_before_step(self, summary: EpochSummary, **kwargs):
        if self.max_gradient_norm == math.inf: return

        model_params = self.trainer.model.parameters()

        if self.mode == 'dynamic':
            grad_norm, max_grad_norm_val = gradient_clipping(
                self.trainer.model,
                self.gradnorms_queue,
                self.max_gradient_norm,
                self.trainer.dist.is_master,
            )
        elif self.mode == 'fixed':
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_params, 
                max_norm=self.max_gradient_norm, 
                norm_type=2.0
            )
            max_grad_norm_val = self.max_gradient_norm
        else:
            raise ValueError(f"Unknown gradient_clipping_mode: {self.mode}")

        if self.trainer.dist.is_master:
            summary.add_grad_norm(grad_norm.item(), float(max_grad_norm_val))

class CheckpointCallback(Callback):    
    def on_epoch_end(self, summary: EpochSummary, run_validation: bool, **kwargs):
        if not run_validation: return
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
    def on_epoch_end(self, summary: EpochSummary, run_validation: bool, **kwargs):
        should_stop = False
        if not run_validation:
            return
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

class GrokFastCallback(Callback):
    """Applies Grokfast's gradient filtering after the backward pass."""
    def on_trainer_begin(self, **kwargs):
        # Initialize the grads attribute on the trainer if needed
        if self.trainer.config.get('use_grokfast', False):
            if not hasattr(self.trainer, 'grads'):
                self.trainer.grads = None
    
    def on_after_backward(self, **kwargs):
        if self.trainer.config.get('use_grokfast', False):
            from geqtrain.utils import gradfilter_ema
            self.trainer.grads = gradfilter_ema(self.trainer.model, grads=self.trainer.grads)

class ActivationNormCallback(Callback):
    PARITY = {
         1: 'e',
        -1: 'o'
    }
    """Tracks the norm of node activations at the end of a training batch."""
    def on_step_end(self, batch_output, summary: EpochSummary, **kwargs):
        # This callback only runs during training and at the specified frequency
        if self.trainer.batch_type != TRAIN or batch_output is None or summary is None:
            return
            
        if not ((self.trainer.ibatch + 1) % self.trainer.log_batch_freq == 0 or (self.trainer.ibatch + 1) == self.trainer.n_batches):
            return

        with torch.no_grad():
            batch_node_norms = {}
            model = self.trainer.model
            # The model might be wrapped in DDP, access the module
            model_to_inspect = model.module if self.trainer.dist.is_distributed else model

            output_irreps = model_to_inspect.irreps_out.get(AtomicDataDict.NODE_FEATURES_KEY)
            if output_irreps is None:
                return # Nothing to do if the key is not in the output irreps

            node_reprs = batch_output[AtomicDataDict.NODE_FEATURES_KEY]

            # Correctly calculate split sizes and reshapers directly from the Irreps object
            split_sizes = [mul * ir.dim for mul, ir in output_irreps]
            reshapers = [reshape_irreps(Irreps(f"{mul}x{ir.l}{self.PARITY[ir.p]}")) for mul, ir in output_irreps]
            reprs = torch.split(node_reprs, split_sizes, dim=1)
            
            for i, (repr_chunk, (mul, ir)) in enumerate(zip(reprs, output_irreps)):
                norm = torch.mean(torch.linalg.vector_norm(reshapers[i](repr_chunk), dim=(-1)))
                batch_node_norms[f'node_repr_l_{ir.l}_p_{ir.p}_part_{i}_norm_mean'] = norm
            
            summary.add_node_feature_norms(batch_node_norms)
