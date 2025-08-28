# trainer.py
import logging
from time import perf_counter
import torch
import torch.distributed as dist

from geqtrain.train.utils import instanciate_train_val_dsets
from geqtrain.train._key import ABBREV, TRAIN, VALIDATION
from geqtrain.train.components.splitter import DatasetSplitter
from geqtrain.utils import Config, Output, load_callable
from geqtrain.data import (
    DataLoader, InMemoryConcatDataset, LazyLoadingConcatDataset
)
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

        # 2. Environment Setup
        self._initialize_env()

        # 3. Core Components
        self.checkpoint_handler = CheckpointHandler(trainer=self)
        
        # 4. Extract config parameters
        self._extract_config_parameters()
        
        # 5. Initialize training state variables
        self._initialize_training_state()
        
        # 6. Handle fine-tuning, but only if it's a new run (not a restart).
        # A restart takes precedence and will load its own complete state later.
        if self.config.get('fine_tune') and not self.config.get('restart'):
            model, self.config = self.checkpoint_handler.load_model_for_resume()
        
        # 7. Setup dataset and dataloaders
        self.dataset_train, self.dataset_val = self._load_and_split_datasets()
        self._init_dataloaders()
        
        # 8. Setup model and training-related objects
        self.model = self._setup_model(model)
        self._setup_training_components()

        # 9. If restarting, load the state INTO the components we just built.
        if self.config.get('restart'):
             self.checkpoint_handler.load_for_restart()
        
        # 10. Callbacks
        self.callbacks = self._setup_callbacks()

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
            ("n_valid", None), ("train_val_split", "random"), ("shuffle", True),
            ("metrics_metadata", {}),
        ]
        for param, default in params_to_extract:
            setattr(self, param, self.config.get(param, default))

    def _initialize_training_state(self):
        """Initialize counters, flags, and metrics for the training process."""
        self.iepoch = -1 if self.report_init_validation else 0
        self.best_metrics = float("inf") if self.metric_criteria == 'decreasing' else float('-inf')
        self.best_epoch = 0; self.cumulative_wall = 0
        self.should_stop = False; self.stop_arg = ""
        self.train_wall = 0; self.validation_wall = 0; self._phase_start_time = 0
        self.batch_losses, self.batch_metrics, self.loss_dict, self.metrics_dict, self.mae_dict = {}, {}, {}, {}, {}
        self.gradnorms, self.gradnorms_clip = [], []

    def _load_and_split_datasets(self):
        """
        Loads and splits datasets, with a DDP guard to prevent data processing race conditions.
        """
        # In DDP, have workers wait until the master has finished processing the data.
        # This prevents multiple processes from trying to write the same cache file.
        # The master process (and single-GPU runs) will execute this and create the cache if needed.
        if self.dist.is_distributed and self.dist.is_master:
            train_dset, val_dset = instanciate_train_val_dsets(self.config)

        # When the master is done, it signals to the waiting workers that they can proceed.
        if self.dist.is_distributed:
            dist.barrier()
            if not self.dist.is_master:
                train_dset, val_dset = instanciate_train_val_dsets(self.config)
        
        # Now, all processes can safely proceed. The workers will load the cache the master created.
        splitter = DatasetSplitter(self)
        self.train_idcs, self.val_idcs = splitter.split(train_dset, val_dset)

        final_train_dset = self._index_dataset(train_dset, self.train_idcs)
        final_val_dset = self._index_dataset(val_dset if val_dset else train_dset, self.val_idcs)

        self.logger.info(f"Training data points: {len(final_train_dset)} | Validation data points: {len(final_val_dset)}")
        return final_train_dset, final_val_dset
    
    def _index_dataset(self, dataset, indices):
        """Selects a subset of a ConcatDataset using a list of lists of indices."""
        indexed_subdatasets = []
        if not isinstance(indices, list) or not all(isinstance(i, (list, torch.Tensor)) for i in indices):
            raise TypeError(f"indices must be a list of lists/tensors, but got {type(indices)}")
        
        for d, idcs in zip(dataset.datasets, indices):
            if len(idcs) > 0:
                if isinstance(dataset, InMemoryConcatDataset):
                    indexed_subdatasets.append(d.index_select(idcs))
                elif isinstance(dataset, LazyLoadingConcatDataset):
                    indexed_subdatasets.append(d[idcs].reshape(-1))
        
        if isinstance(dataset, InMemoryConcatDataset):
            return InMemoryConcatDataset(indexed_subdatasets)
        if isinstance(dataset, LazyLoadingConcatDataset):
            return dataset.from_indexed_dataset(indices)
        raise TypeError(f"Unsupported dataset type for indexing: {type(dataset)}")

    def _init_dataloaders(self):
        use_ensemble = self.config.get("dataset_mode") == "ensemble"
        train_sampler = self.dist.get_sampler(self.dataset_train, self.shuffle, use_ensemble)
        val_sampler = self.dist.get_sampler(self.dataset_val, False, use_ensemble)
        dl_kwargs = {"num_workers": self.config.get('dataloader_num_workers', 0), "pin_memory": self.dist.device.type == 'cuda', "generator": self.dataset_rng}
        self.dl_train = DataLoader(dataset=self.dataset_train, batch_size=self.config.get('batch_size'), shuffle=(train_sampler is None and self.shuffle), sampler=train_sampler, **dl_kwargs)
        self.dl_val = DataLoader(dataset=self.dataset_val, batch_size=self.config.get('validation_batch_size'), shuffle=False, sampler=val_sampler, **dl_kwargs)

    def _setup_model(self, model):
        if model is None:
            model, _ = model_from_config(config=self.config, initialize=True, dataset=self.dataset_train)
        self.num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of trainable weights: {self.num_weights}")
        return self.dist.wrap_model(model)

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
        return callbacks

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