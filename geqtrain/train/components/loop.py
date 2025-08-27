# components/loop.py
import torch
import contextlib
from geqtrain.train.components.inference import run_inference
from geqtrain.train.grad_clipping_utils import gradient_clipping
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.train._key import TRAIN, VALIDATION


class TrainingLoop:
    """Manages the training and validation loops for each epoch."""
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.optim = trainer.optim
        self.loss_fn = trainer.loss
        self.loss_stat = trainer.loss_stat
        self.metrics = trainer.metrics
        self.ema = trainer.ema
        self.dist = trainer.dist

    def run_epoch(self, validation_only=False):
        """Runs a full training and validation epoch."""
        if not validation_only:
            self.run_phase(TRAIN)
        
        ema_cm = self.ema.average_parameters() if self.ema is not None else contextlib.nullcontext()
        with ema_cm:
            self.run_phase(VALIDATION)

    def run_phase(self, phase: str):
        """Runs a single phase (training or validation)."""
        self.trainer._dispatch_callbacks(f'on_{phase}_begin')
        
        is_train = phase == TRAIN
        dataloader = self.trainer.dl_train if is_train else self.trainer.dl_val
        self.trainer.n_batches = len(dataloader)  # Set n_batches for the current phase
        self.model.train(is_train)
        
        self.loss_stat.reset()
        self.metrics.reset()
        self.loss_stat.to(self.dist.device)
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
        
        self.trainer.loss_dict[phase] = self.loss_stat.current_result()
        self.trainer.metrics_dict[phase] = self.metrics.current_result()
        self.trainer._dispatch_callbacks(f'on_{phase}_end')
    
    def _run_batch(self, data, is_train: bool):
        """Runs a single batch with all original logic."""
        already_computed_nodes = None
        
        while True:
            cm = contextlib.nullcontext() if is_train else torch.no_grad()
            with cm:
                out, ref_data, batch_chunk_center_nodes, _ = run_inference(
                    model=self.model, data=data, device=self.dist.device,
                    output_keys=self.trainer.output_keys,
                    per_node_outputs_keys=self.trainer.per_node_outputs_keys,
                    already_computed_nodes=already_computed_nodes,
                    config=self.trainer.config,
                    chunking=self.trainer.chunking,
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
                            self.model, self.trainer.gradnorms_queue, max_grad_norm, self.dist.is_master
                        )
                        if self.dist.is_master:
                            self.trainer.gradnorms.append(grad_norm.item())
                            self.trainer.gradnorms_clip.append(float(max_grad_norm_val))
                    
                    self.optim.step()
                    if self.ema:
                        self.ema.update()
                    self.optim.zero_grad(set_to_none=True)
                    self.trainer.accumulation_counter = 0

            # --- Update and Sync Metrics/Losses ---
            syncd_loss = self.dist.sync_tensor(loss.detach())
            syncd_loss_contrib = self.dist.sync_dict_of_tensors(loss_contrib)
            
            # Call loss_stat and store the result for the logger
            # The call itself updates the epoch-level stats internally.
            batch_losses = self.loss_stat(syncd_loss, syncd_loss_contrib)
            if self.dist.is_master:
                self.trainer.batch_losses = batch_losses
            
            with torch.no_grad():
                self.trainer.batch_metrics = self.metrics(pred=out, ref=ref_data)
            
            # --- Evaluate chunking condition ---
            if not self.trainer.chunking:
                break 
            
            already_computed_nodes = evaluate_end_chunking_condition(
                already_computed_nodes, batch_chunk_center_nodes, len(batch_chunk_center_nodes)
            )
            if already_computed_nodes is None:
                break