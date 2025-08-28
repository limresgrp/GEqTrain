# components/loop.py
import torch
import contextlib

from .inference import run_inference
from geqtrain.train.grad_clipping_utils import gradient_clipping, Queue
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.train._key import TRAIN, VALIDATION


class TrainingLoop:
    """Manages the training and validation loops for each epoch."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.optim = trainer.optim
        self.loss_fn = trainer.loss
        self.metrics = trainer.metrics
        self.ema = trainer.ema
        self.dist = trainer.dist
        self.lr_sched = trainer.lr_sched
        self.warmup_sched = trainer.warmup_sched
        self.gradnorms_queue = Queue()

    def run_epoch(self, validation_only=False):
        """Runs a full training and validation epoch."""
        if not validation_only:
            self.run_phase(TRAIN)

        ema_cm = self.ema.average_parameters() if self.ema is not None else contextlib.nullcontext()
        with ema_cm:
            self.run_phase(VALIDATION)

        # Assemble the master epoch summary dict (`mae_dict`)
        self._build_epoch_summary()

        # Step the LR scheduler after the epoch summary is built
        self._lr_sched_step(batch_lvl=False)

    def _build_epoch_summary(self):
        """Assembles the `mae_dict` with all results from the epoch."""
        epoch = self.trainer.iepoch + 1
        train_wall = self.trainer.train_wall if epoch > 0 else 0.0
        
        self.trainer.mae_dict = {
            "epoch": epoch, "LR": self.trainer.optim.param_groups[0]["lr"],
            "train_wall": train_wall, "validation_wall": self.trainer.validation_wall,
            "cumulative_wall": self.trainer.cumulative_wall,
        }
        
        categories = [TRAIN, VALIDATION] if epoch > 0 else [VALIDATION]
        for category in categories:
            for key, value in self.trainer.loss_dict[category].items():
                self.trainer.mae_dict[f"{category}_{key}"] = value
            metrics = self.metrics.flatten_metrics(self.trainer.metrics_dict[category], self.trainer.metrics_metadata)
            for key, value in metrics.items():
                self.trainer.mae_dict[f"{category}_{key}"] = value

    def run_phase(self, phase: str):
        """Runs a single phase (training or validation)."""
        self.trainer._dispatch_callbacks(f'on_{phase}_begin')
        
        is_train = phase == TRAIN
        dataloader = self.trainer.dl_train if is_train else self.trainer.dl_val
        self.trainer.n_batches = len(dataloader)
        self.model.train(is_train)
        
        # Reset and move both trackers to the correct device
        self.loss_fn.reset()
        self.metrics.reset()
        self.loss_fn.to(self.dist.device)
        self.metrics.to(self.dist.device)
        
        if is_train:
            self.trainer.optim.zero_grad(set_to_none=True)
            self.trainer.accumulation_counter = 0

        for ibatch, batch in enumerate(dataloader):
            self.trainer.ibatch = ibatch
            self.trainer.batch_type = phase
            self.trainer._dispatch_callbacks('on_batch_begin')
            self._run_batch(batch, is_train=is_train)
            self.trainer._dispatch_callbacks('on_batch_end')
        
        # Get final results from each tracker
        self.trainer.loss_dict[phase] = self.loss_fn.current_result()
        self.trainer.metrics_dict[phase] = self.metrics.current_result()
        self.trainer._dispatch_callbacks(f'on_{phase}_end')

    def _run_batch(self, data, is_train: bool):
        """Runs a single batch with all original logic."""
        already_computed_nodes = None
        while True:
            out, ref_data, batch_chunk_center_nodes, _ = run_inference(
                model=self.model,
                data=data,
                device=self.dist.device,
                loss_fn=self.loss_fn,
                config=self.trainer.config.as_dict(),
                already_computed_nodes=already_computed_nodes,
                is_train=is_train,
            )
            loss, loss_contrib = self.loss_fn(pred=out, ref=ref_data)

            if is_train:
                accumulation_steps = self.trainer.config.get('accumulation_steps', 1)
                loss = loss / accumulation_steps
                loss.backward()
                self.trainer.accumulation_counter += 1

                if self.trainer.config.get('use_grokfast', False):
                    from geqtrain.utils import gradfilter_ema
                    self.trainer.grads = gradfilter_ema(self.model, grads=self.trainer.grads)
                
                if self.trainer.accumulation_counter == accumulation_steps:
                    max_grad_norm = self.trainer.config.get('max_gradient_norm')
                    if max_grad_norm is not None:
                        grad_norm, max_grad_norm_val = gradient_clipping(
                            self.model, self.gradnorms_queue, max_grad_norm, self.dist.is_master
                        )
                        if self.dist.is_master:
                            self.trainer.gradnorms.append(grad_norm.item())
                            self.trainer.gradnorms_clip.append(float(max_grad_norm_val))
                    
                    self.optim.step()
                    if self.ema:
                        self.ema.update()
                    
                    self._lr_sched_step(batch_lvl=True)
                    self.optim.zero_grad(set_to_none=True)
                    self.trainer.accumulation_counter = 0

            syncd_loss = self.dist.sync_tensor(loss.detach())
            syncd_loss_contrib = self.dist.sync_dict_of_tensors(loss_contrib)

            # The loss object's internal tracker is used to get per-batch results
            batch_losses = self.loss_fn.loss_stat(syncd_loss, syncd_loss_contrib)
            if self.dist.is_master:
                self.trainer.batch_losses = batch_losses
            
            with torch.no_grad():
                self.trainer.batch_metrics = self.metrics(pred=out, ref=ref_data)
            
            if not self.trainer.config.get('chunking', False):
                break
            
            already_computed_nodes = evaluate_end_chunking_condition(
                already_computed_nodes, batch_chunk_center_nodes, len(batch_chunk_center_nodes)
            )
            if already_computed_nodes is None:
                break

    def _lr_sched_step(self, batch_lvl: bool):
        """Handle LR scheduler steps for both warmup and main phases."""
        if self.lr_sched is None:
            return

        if batch_lvl:
            # Handle warmup for batch-level schedulers
            if self.warmup_sched and not self._is_warmup_period_over():
                with self.warmup_sched.dampening():
                    self.lr_sched.step()
            else:
                self._batch_lvl_lrscheduler_step()
        else: # Epoch level
            # We only step epoch-level schedulers after the warmup period is over
            if self._is_warmup_period_over():
                self._epoch_lvl_lrscheduler_step()
    
    def _is_warmup_period_over(self):
        """Check if the warmup period is finished."""
        if self.warmup_sched is None:
            return True
        warmup_steps = self.warmup_sched.warmup_params[0]["warmup_period"]
        return self.warmup_sched.last_step >= warmup_steps - 1

    def _batch_lvl_lrscheduler_step(self):
        """Step schedulers that update on every optimization step."""
        scheduler_name = self.trainer.config.get('lr_scheduler_name')
        if scheduler_name in ("CosineAnnealingLR", "CosineAnnealingWarmRestarts"):
            self.lr_sched.step()

    def _epoch_lvl_lrscheduler_step(self):
        """Step schedulers that update based on end-of-epoch metrics."""
        scheduler_name = self.trainer.config.get('lr_scheduler_name')
        if scheduler_name == "ReduceLROnPlateau":
            metric = self.trainer.mae_dict.get(self.trainer.metrics_key)
            if metric is not None:
                self.lr_sched.step(metrics=metric)