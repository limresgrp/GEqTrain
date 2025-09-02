# trainer.py
import logging
from time import perf_counter
import torch
import torch.distributed as dist

from geqtrain.data.dataloader import DataLoader
from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.train._key import ABBREV, TRAIN, VALIDATION
from geqtrain.utils import Config, Output, load_callable
from geqtrain.model import model_from_config

# Import the new components
from .components.distributed import DistributedManager
from .components.callbacks import Logger, CheckpointCallback, EarlyStoppingCallback
from .components.setup import (setup_loss, setup_metrics, setup_optimizer,
                               setup_scheduler, setup_ema, set_seed, setup_early_stopping)
from .components.checkpointing import CheckpointHandler
from .components.loop import TrainingLoop

class Trainer:
    """
    Orchestrates the training process by delegating tasks to specialized components.

    Args:
        config (dict): A dictionary-like object containing the configuration.
        model (Grap, optional): An existing model. If None, a model is built from the config.
    """

    def __init__(self, config: dict, model: torch.nn.Module = None):
        # 1. Initial Setup
        self.config = Config.from_dict(config) if not isinstance(config, Config) else config
        self.dist = DistributedManager(config=self.config)

        # 2. Environment Setup (logging, output paths, seeds)
        self._initialize_env()
        self.checkpoint_handler = CheckpointHandler(trainer=self)

        trainer_state = None
        if self.config.get('restart'):
            # 3. Load the raw state dictionary FIRST.
            trainer_state = self.checkpoint_handler.load_raw_state_for_restart()
            
            # 4. Inject the essential dataset indices into the config.
            # This ensures the DatasetBuilder uses the correct, restored indices.
            logging.info("Updating config with train/validation indices from checkpoint.")
            self.config['train_idcs'] = trainer_state['train_idcs']
            self.config['val_idcs'] = trainer_state['val_idcs']

        # 5. Extract config parameters (now includes restored indices if restarting)
        self._extract_config_parameters()
        
        # 6. Initialize training state variables
        self._initialize_training_state()
        
        # 7. Handle fine-tuning (only if it's a new run, not a restart).
        if self.config.get('fine_tune') and not self.config.get('restart'):
            model, self.config = self.checkpoint_handler.load_model_for_resume()
        
        # 8. Setup dataset and dataloaders
        # This will now correctly use the restored indices on a restart.
        self._build_datasets()
        self._init_dataloaders()
        
        # 9. Setup model and training-related objects
        self._setup_model(model)
        self._setup_training_components()

        # 10. If restarting, APPLY the rest of the state to the components we just built.
        if self.config.get('restart') and trainer_state is not None:
             self.checkpoint_handler.apply_state_for_restart(trainer_state)
        
        # 11. Callbacks
        self._setup_callbacks()

    def _initialize_env(self):
        """Set up the environment: logging, output paths, and seeds."""
        if self.dist.is_master:
            self.output = Output.get_output(self.config)
            self.logfile = self.output.open_logfile("log", propagate=True)
            self.epoch_log = self.output.open_logfile("metrics_epoch.csv", propagate=False)
            self.batch_log = {
                TRAIN: self.output.open_logfile(f"metrics_batch_{ABBREV[TRAIN]}.csv", propagate=False),
                VALIDATION: self.output.open_logfile(f"metrics_batch_{ABBREV[VALIDATION]}.csv", propagate=False),
            }
        else:
            self.output = None; self.logfile = "dummy"

        self.logger = logging.getLogger(self.logfile)
        set_seed(self.config.get('seed'))
        self.dataset_rng = torch.Generator().manual_seed(self.config.get('dataset_seed'))

    def _extract_config_parameters(self):
        """Extract parameters from the config file to trainer attributes for easy access."""
        params_to_extract = [
            ("learning_rate", 1e-3), ("metrics_key", "validation_loss"), ("metric_criteria", "decreasing"),
            ("log_batch_freq", 10), ("log_epoch_freq", 1), ("save_checkpoint_freq", -1),
            ("max_epochs", 1000), ("warmup_epochs", 0), ("report_init_validation", True),
            ("use_ema", False), ("train_idcs", None), ("val_idcs", None), ("n_train", None),
            ("n_val", None), ("train_val_split", "random"), ("shuffle", True),
            ("metrics_metadata", {}),
        ]
        for param, default in params_to_extract:
            setattr(self, param, self.config.get(param, default))

    def _initialize_training_state(self):
        """Initialize counters, flags, and metrics for the training process."""
        self.iepoch = -1 if self.report_init_validation else 0
        self.best_metrics = float("inf") if self.metric_criteria == 'decreasing' else float('-inf')
        self.best_epoch, self.cumulative_wall = 0, 0
        self.should_stop, self.stop_arg = False, ""
        self.train_wall, self.validation_wall, self._phase_start_time = 0, 0, 0
        self.batch_losses, self.batch_metrics, self.loss_dict, self.metrics_dict, self.mae_dict = {}, {}, {}, {}, {}
        self.gradnorms, self.gradnorms_clip = [], []
        self.train_dset, self.val_dset = None, None
        self.train_idcs, self.val_idcs = None, None
        self.train_sampler, self.val_sampler = None, None

    def _build_datasets(self):
        """
        Builds datasets using a two-step, DDP-aware process:
        1. Master resolves indices using lightweight metadata.
        2. Master broadcasts indices.
        3. All processes build the final datasets from these indices.
        """
        builder = DatasetBuilder(self.config, self.dataset_rng, self.logger, self.output)
        
        # --- DDP Synchronization for Index Resolution ---
        # self.train_idcs is initialized from config in _initialize_training_state
        # If they are not specified, we must generate them.
        if self.train_idcs is None or self.val_idcs is None:
            
            # 1. Master process alone resolves the indices efficiently.
            if self.dist.is_master:
                train_idcs, val_idcs = builder.resolve_split_indices()
            else:
                train_idcs, val_idcs = None, None
            
            # 2. Synchronize and broadcast the indices to all processes.
            if self.dist.is_distributed:
                dist.barrier()
            
            self.train_idcs = self.dist.broadcast_object(train_idcs, src=0)
            self.val_idcs   = self.dist.broadcast_object(val_idcs, src=0)
        
        # --- Dataset Building on ALL processes ---
        # 3. Now that all processes have the SAME indices, they build the final datasets.
        #    This is where the full data is actually loaded.
        self.train_dset, self.val_dset = builder.build_datasets_from_indices(self.train_idcs, self.val_idcs)
        
        # Final barrier to ensure all dataset loading is complete before proceeding.
        if self.dist.is_distributed:
            dist.barrier()
        
        self.logger.info(f"Training data points: {len(self.train_dset)} | Validation data points: {len(self.val_dset)}")

    def _init_dataloaders(self):
        use_ensemble = self.config.get("dataset_mode") == "ensemble"
        self.train_sampler = self.dist.get_sampler(self.train_dset, self.shuffle, use_ensemble)
        self.val_sampler = self.dist.get_sampler(self.val_dset, False, use_ensemble)
        dl_kwargs = {"num_workers": self.config.get('dataloader_num_workers', 0), "pin_memory": self.dist.device.type == 'cuda', "generator": self.dataset_rng}
        self.dl_train = DataLoader(dataset=self.train_dset, batch_size=self.config.get('batch_size'), shuffle=(self.train_sampler is None and self.shuffle), sampler=self.train_sampler, **dl_kwargs)
        self.dl_val = DataLoader(dataset=self.val_dset, batch_size=self.config.get('validation_batch_size'), shuffle=False, sampler=self.val_sampler, **dl_kwargs)

    def _setup_model(self, model):
        if model is None:
            model, _ = model_from_config(config=self.config, initialize=True, dataset=self.train_dset)
        self.num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of trainable weights: {self.num_weights}")
        self.model = self.dist.wrap_model(model)

    def _setup_training_components(self):
        self.loss = setup_loss(self.config)
        self.metrics = setup_metrics(self.config)
        self.optim = setup_optimizer(self.model, self.config)
        self.lr_sched, self.warmup_sched = setup_scheduler(self.optim, self.config, len(self.dl_train))
        self.ema = setup_ema(self.model, self.config)
        self.early_stopping_conds = setup_early_stopping(self.config)

    def _setup_callbacks(self):
        callbacks = [Logger(), CheckpointCallback(), EarlyStoppingCallback()]
        if self.config.get('wandb'):
            from .components.callbacks import WandbCallback
            callbacks.insert(1, WandbCallback()) # Insert Wandb before checkpointing
        for cb_string in self.config.get('callbacks', []):
            callbacks.append(load_callable(cb_string)())
        for cb in callbacks:
            cb.set_trainer(self)
        self.callbacks = callbacks

    def _dispatch_callbacks(self, event: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, event)(**kwargs)

    def train(self):
        """Main training entry point."""
        self.training_loop = TrainingLoop(self)
        self._dispatch_callbacks('on_trainer_begin')
        self.wall = perf_counter()

        while not self.should_stop:
            self._dispatch_callbacks('on_epoch_begin')
            self.training_loop.run_epoch(validation_only=(self.iepoch == -1))
            self._dispatch_callbacks('on_epoch_end')
            self.iepoch += 1
        
        self._dispatch_callbacks('on_trainer_end')

    @property
    def best_model_path(self):
        return self.checkpoint_handler.best_model_path
    
    @property
    def last_model_path(self):
        return self.checkpoint_handler.last_model_path

    @property
    def trainer_save_path(self):
        return self.checkpoint_handler.trainer_save_path
