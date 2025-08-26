# trainer.py

""" Adapted from https://github.com/mir-group/nequip """
import logging
from time import perf_counter
import torch

from geqtrain.nn._graph_mixin import GraphModuleMixin
from geqtrain.train._key import ABBREV, TRAIN, VALIDATION
from geqtrain.utils import Config, Output, load_callable
from geqtrain.data import (
    DataLoader, InMemoryConcatDataset, LazyLoadingConcatDataset
)
from geqtrain.model import model_from_config
from geqtrain.train.utils import instanciate_train_val_dsets

# Import the new components
from .components.distributed import DistributedManager
from .components.callbacks import Logger, CheckpointCallback, EarlyStoppingCallback
from .components.setup import (
    setup_loss, setup_metrics, setup_optimizer, setup_scheduler, setup_ema,
    parse_idcs_file, get_output_keys, set_seed, setup_early_stopping
)
from .components.checkpointing import CheckpointHandler
from .components.loop import TrainingLoop

class Trainer:
    """
    Orchestrates the training process by delegating tasks to specialized components.

    Args:
        config (dict): A dictionary-like object containing the configuration.
        model (Grap, optional): An existing model. If None, a model is built from the config.
    """

    def __init__(self, config: dict, model: GraphModuleMixin = None):
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
        
        # 6. Handle fine-tuning by loading an EXTERNAL model and updating the config
        #    This happens BEFORE the main model and dataset are built.
        if self.config.get('fine_tune'):
            model, self.config = self.checkpoint_handler.load_model_for_resume()
        
        # 7. Setup dataset and dataloaders
        self.dataset_train, self.dataset_val = self._load_and_split_datasets()
        self._init_dataloaders()
        
        # 8. Setup model and training-related objects (optimizer, scheduler, etc.)
        self.model = self._setup_model(model)
        self._setup_training_components()

        # 9. If restarting, load the state INTO the components we just built.
        #    This happens AFTER all components are initialized.
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
            ("metrics_metadata", {})
        ]
        for param, default in params_to_extract:
            setattr(self, param, self.config.get(param, default))

    def _initialize_training_state(self):
        """Initialize counters, flags, and metrics for the training process."""
        from geqtrain.train.grad_clipping_utils import Queue
        self.iepoch = -1 if self.report_init_validation else 0
        self.best_metrics = float("inf") if self.metric_criteria == 'decreasing' else float('-inf')
        self.best_epoch = 0; self.cumulative_wall = 0
        self.should_stop = False; self.stop_arg = ""
        self.train_wall = 0; self.validation_wall = 0; self._phase_start_time = 0
        self.batch_losses, self.batch_metrics, self.loss_dict, self.metrics_dict, self.mae_dict = {}, {}, {}, {}, {}
        self.gradnorms, self.gradnorms_clip = [], []
        self.gradnorms_queue = Queue()

    def _load_and_split_datasets(self):
        """Load raw datasets and apply train/validation splits."""
        train_dset, val_dset = instanciate_train_val_dsets(self.config)
        
        if isinstance(self.train_idcs, str): self.train_idcs = parse_idcs_file(self.train_idcs)
        if isinstance(self.val_idcs, str): self.val_idcs = parse_idcs_file(self.val_idcs)

        if self.train_idcs is None or self.val_idcs is None:
            self.train_idcs, self.val_idcs = self.split_dataset(train_dset, val_dset)

        final_train_dset = self.index_dataset(train_dset, self.train_idcs)
        final_val_dset = self.index_dataset(val_dset if val_dset else train_dset, self.val_idcs)

        self.logger.info(f"Training data points: {len(final_train_dset)} | Validation data points: {len(final_val_dset)}")
        return final_train_dset, final_val_dset

    def _init_dataloaders(self):
        use_ensemble = self.config.get("dataset_mode") == "ensemble"
        train_sampler = self.dist.get_sampler(self.dataset_train, self.shuffle, use_ensemble)
        val_sampler = self.dist.get_sampler(self.dataset_val, False, use_ensemble)
        dl_kwargs = {"num_workers": self.config.get('dataloader_num_workers', 0), "pin_memory": self.dist.device.type == 'cuda', "generator": self.dataset_rng}
        self.dl_train = DataLoader(dataset=self.dataset_train, batch_size=self.config.get('batch_size'), shuffle=(train_sampler is None and self.shuffle), sampler=train_sampler, **dl_kwargs)
        self.dl_val = DataLoader(dataset=self.dataset_val, batch_size=self.config.get('validation_batch_size'), shuffle=False, sampler=val_sampler, **dl_kwargs)

    def _setup_model(self, model):
        if model is None: model, _ = model_from_config(config=self.config, initialize=True, dataset=self.dataset_train)
        self.num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Number of trainable weights: {self.num_weights}")
        return self.dist.wrap_model(model)

    def _setup_training_components(self):
        self.loss, self.loss_stat = setup_loss(self.config)
        self.metrics = setup_metrics(self.config)
        self.optim = setup_optimizer(self.model, self.config)
        self.lr_sched, self.warmup_sched = setup_scheduler(self.optim, self.config, len(self.dl_train))
        self.ema = setup_ema(self.model, self.config)
        self.early_stopping_conds = setup_early_stopping(self.config)
        self.output_keys, self.per_node_outputs_keys = get_output_keys(self.loss)

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

    # In trainer.py, replace this method in the Trainer class

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
        
        # All cleanup is handled by the main script's finally block
        self._dispatch_callbacks('on_trainer_end')
        
    def _dispatch_callbacks(self, event: str, **kwargs):
        for cb in self.callbacks:
            getattr(cb, event)(**kwargs)

    @property
    def best_model_path(self):
        return self.checkpoint_handler.best_model_path
    
    @property
    def last_model_path(self):
        return self.checkpoint_handler.last_model_path

    @property
    def trainer_save_path(self):
        return self.checkpoint_handler.trainer_save_path

    def split_dataset(self, train_dset, val_dset=None):
        if self.n_train is not None and not isinstance(self.n_train, list): self.n_train = [self.n_train]
        if self.n_valid is not None and not isinstance(self.n_valid, list): self.n_valid = [self.n_valid]
        val_dset_provided = val_dset is not None
        def get_n_train_list():
            if self.n_train: return self.n_train
            if val_dset_provided: return train_dset.n_observations.tolist()
            if self.n_valid: return [n - v for n, v in zip(train_dset.n_observations, self.n_valid)]
            self.logger.warning("No 'n_train' or 'n_valid' provided; using default 80/20 split.")
            return (train_dset.n_observations * 0.8).astype(int).tolist()
        def get_n_valid_list():
            if self.n_valid: return self.n_valid
            source_dset = val_dset if val_dset_provided else train_dset
            if val_dset_provided: return source_dset.n_observations.tolist()
            return [n - t for n, t in zip(source_dset.n_observations, self.n_train)]
        self.n_train = get_n_train_list(); self.n_valid = get_n_valid_list()
        train_idcs, val_idcs = [], []
        for i, (n_obs, n_t) in enumerate(zip(train_dset.n_observations, self.n_train)):
            if n_t > n_obs: raise ValueError(f"n_train[{i}]={n_t} is > n_observations[{i}]={n_obs}")
            permutation = torch.randperm(n_obs, generator=self.dataset_rng)
            train_idcs.append(permutation[:n_t])
            if not val_dset_provided:
                n_v = self.n_valid[i]
                if n_t + n_v > n_obs: raise ValueError("n_train + n_valid > n_observations")
                val_idcs.append(permutation[n_t : n_t + n_v])
        if val_dset_provided:
            for i, (n_obs, n_v) in enumerate(zip(val_dset.n_observations, self.n_valid)):
                 if n_v > n_obs: raise ValueError(f"n_valid[{i}]={n_v} is > n_observations[{i}]={n_obs}")
                 permutation = torch.randperm(n_obs, generator=self.dataset_rng)
                 val_idcs.append(permutation[:n_v])
        return train_idcs, val_idcs

    def index_dataset(self, dataset, indices):
        indexed_subdatasets = []
        for d, idcs in zip(dataset.datasets, indices):
            if len(idcs) > 0: indexed_subdatasets.append(d.index_select(idcs))
        if isinstance(dataset, InMemoryConcatDataset): return InMemoryConcatDataset(indexed_subdatasets)
        elif isinstance(dataset, LazyLoadingConcatDataset): return dataset.from_indexed_dataset(indices)
        else: raise TypeError(f"Unsupported dataset type for indexing: {type(dataset)}")