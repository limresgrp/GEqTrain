# components/loop.py
from collections import defaultdict
import math
import torch
import contextlib

from geqtrain.train.components.epoch_summary import EpochSummary

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

    def run_epoch(self, summary: EpochSummary, validation_only=False):
        """
        Runs a full training and validation epoch, populating the provided summary object.
        """
        if not validation_only:
            if self.trainer.train_sampler is not None:
                self.trainer.train_sampler.set_epoch(self.trainer.iepoch + 1)
            self.run_phase(TRAIN, summary)

        ema_cm = self.ema.average_parameters() if self.ema is not None else contextlib.nullcontext()
        with ema_cm:
            self.run_phase(VALIDATION, summary)

        # Step the LR scheduler after the epoch summary is built
        self._lr_sched_step(summary=summary, batch_lvl=False)

    def run_phase(self, phase: str, summary: EpochSummary):
        """Runs a single phase (training or validation)."""
        self.trainer._dispatch_callbacks(f'on_{phase}_begin')

        is_train = phase == TRAIN
        dataloader = self.trainer.dl_train if is_train else self.trainer.dl_val
        self.trainer.n_batches = len(dataloader)
        self.model.train(is_train)

        # Reset and move both trackers to the correct device
        self.loss_fn.reset()
        self.metrics.reset()
        self.node_features_norms = defaultdict(list)
        self.loss_fn.to(self.dist.device)
        self.metrics.to(self.dist.device)

        if is_train:
            self.trainer.optim.zero_grad(set_to_none=True)
            self.trainer.accumulation_counter = 0

        for ibatch, batch in enumerate(dataloader):
            self.trainer.ibatch = ibatch
            self.trainer.batch_type = phase
            self.trainer._dispatch_callbacks('on_batch_begin')
            self._run_batch(batch, is_train=is_train, summary=summary)
            self.trainer._dispatch_callbacks('on_batch_end')

        # Get final results and record them in the summary object
        loss_results    = self.loss_fn.current_result()
        metrics_results = self.metrics.current_result()
        summary.set_phase_results(phase, loss_results, metrics_results)

        self.trainer._dispatch_callbacks(f'on_{phase}_end')

    def _run_batch(self, data, is_train: bool, summary: EpochSummary):
        """
        Orchestrates batch processing, handling chunking if enabled.
        This method controls the looping logic.
        """
        if not self.trainer.config.get('chunking', False):
            # No chunking, just one step.
            self._process_step(data, is_train, summary)
        else:
            # Chunking is enabled, so we loop.
            already_computed_nodes = None
            while True:
                # Process one chunk of the batch
                center_nodes = self._process_step(
                    data, is_train, summary, already_computed_nodes
                )

                # Update the state for the next chunk
                already_computed_nodes = evaluate_end_chunking_condition(
                    already_computed_nodes, center_nodes, len(center_nodes)
                )

                if already_computed_nodes is None:
                    break # Finished all chunks for this batch

    def _process_step(self, data, is_train: bool, summary: EpochSummary, already_computed_nodes=None):
        """
        Performs the core computation for a single step (a chunk or a full batch).
        This method contains the inference and training logic.
        """
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
            self.trainer._dispatch_callbacks('on_after_backward')

            if self.trainer.accumulation_counter == accumulation_steps:
                grad_norm, max_grad_norm_val = gradient_clipping(
                    self.model,
                    self.gradnorms_queue,
                    self.trainer.config.get('max_gradient_norm', math.inf),
                    self.dist.is_master,
                )
                if self.dist.is_master:
                    summary.add_grad_norm(grad_norm.item(), float(max_grad_norm_val))

                self.optim.step()
                if self.ema: self.ema.update()
                self._lr_sched_step(summary=summary, batch_lvl=True)
                self.optim.zero_grad(set_to_none=True)
                self.trainer.accumulation_counter = 0

        syncd_loss = self.dist.sync_tensor(loss.detach())
        syncd_loss_contrib = self.dist.sync_dict_of_tensors(loss_contrib)
        batch_losses = self.loss_fn.loss_stat(syncd_loss, syncd_loss_contrib)
        if self.dist.is_master: self.trainer.batch_losses = batch_losses

        with torch.no_grad():
            batch_metrics = self.metrics(pred=out, ref=ref_data)
            syncd_batch_metrics = self.dist.sync_dict_of_tensors(batch_metrics)
            self.trainer.batch_metrics = syncd_batch_metrics

        # Dispatch the per-step hook for callbacks
        self.trainer._dispatch_callbacks('on_step_end', batch_output=out, summary=summary)

        # Return values needed by the chunking controller
        return batch_chunk_center_nodes

    def _lr_sched_step(self, summary: EpochSummary, batch_lvl: bool):
        """Handle LR scheduler steps for both warmup and main phases."""
        if self.lr_sched is None:
            return

        if batch_lvl:
            # Handle warmup for batch-level schedulers
            if self.warmup_sched and not self._is_warmup_period_over():
                with self.warmup_sched.dampening():
                    pass
            else:
                self._batch_lvl_lrscheduler_step()
        else: # Epoch level
            # We only step epoch-level schedulers after the warmup period is over
            if self._is_warmup_period_over():
                self._epoch_lvl_lrscheduler_step(summary)

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

    def _epoch_lvl_lrscheduler_step(self, summary: EpochSummary):
        """Step schedulers that update based on end-of-epoch metrics."""
        scheduler_name = self.trainer.config.get('lr_scheduler_name')
        if scheduler_name == "ReduceLROnPlateau":
            metric = summary.get_target_metric(self.trainer.metrics_key)
            if metric is not None:
                self.lr_sched.step(metrics=metric)