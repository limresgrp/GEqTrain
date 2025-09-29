# trainer.py
import logging
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist

from geqtrain.data.dataloader import DataLoader
from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.train._key import ABBREV, TRAIN, VALIDATION
from geqtrain.train.components.epoch_summary import EpochSummary
from geqtrain.utils import Config, Output, load_callable, load_file
from geqtrain.utils._global_options import apply_global_config
from geqtrain.model import model_from_config

# Import the new components
from .components.distributed import DistributedManager
from .components.callbacks import (ActivationNormCallback, GrokFastCallback, Logger, 
                                   CheckpointCallback, EarlyStoppingCallback, 
                                   SanitizeGradCallback, GradientClippingCallback)
from .components.setup import (setup_loss, setup_metrics, setup_optimizer,
                               setup_scheduler, setup_ema, set_seed, setup_early_stopping)
from .components.checkpointing import CheckpointHandler
from .components.loop import TrainingLoop

def check_for_config_updates(new_config):
    """
    Load a saved config, merge new parameters, enforce current runtime settings,
    and return a clean, final Config object.
    """
    restart_file = f"{new_config['root']}/{new_config['run_name']}/trainer.pth"
    saved_state = load_file(
        filename=restart_file,
        supported_formats=dict(torch=["pt", "pth"]),
        enforced_format="torch",
        map_location="cpu",
        weights_only=False,
    )

    # Start with the dictionary from the saved run
    final_config_dict = saved_state['config'].as_dict()
    new_config_dict = new_config.as_dict()
    
    # 1. Update user-modifiable parameters
    modifiable_params = [
        "max_epochs", "learning_rate", "loss_coeffs", "metrics_components", "log_batch_freq",
        "use_ema", "wandb", "dataset_list", "validation_dataset_list", "test_dataset_list",
        "batch_size", "validation_batch_size", "dataloader_num_workers", "master_addr", "master_port",
        "device", "filepath", "ddp",
    ]
    logging.info("Checking for updated user-modifiable parameters...")
    for key in new_config_dict:
        if key in final_config_dict and new_config_dict[key] != final_config_dict[key]:
            if key in modifiable_params:
                logging.info(f'Updating parameter "{key}" from `{final_config_dict[key]}` to `{new_config_dict[key]}`')
                final_config_dict[key] = new_config_dict[key]
            else:
                raise ValueError(f'Parameter "{key}" is not user-modifiable and cannot be changed during restart.')

    # 2. Enforce critical runtime parameters from the new command
    # This ensures DDP status, device, etc., are always from the new command.
    runtime_params = ['ddp', 'device', 'find_unused_parameters']
    logging.info("Enforcing current runtime parameters...")
    for key in runtime_params:
        if key in new_config_dict and new_config_dict[key] is not None:
            if final_config_dict.get(key) != new_config_dict[key]:
                 logging.info(f"Overwriting runtime parameter '{key}' with new value '{new_config_dict[key]}'")
            final_config_dict[key] = new_config_dict[key]
        else:
            # If not specified in the new command, remove the old key
            if key in final_config_dict:
                logging.info(f"Removing stale runtime parameter '{key}' from loaded config.")
                final_config_dict.pop(key)
    
    # 3. Create a brand new, clean Config object from the final dictionary
    final_config = Config.from_dict(final_config_dict)
    final_config.filepath = new_config.filepath
            
    return final_config

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

        # 2. All ranks participate and synchronize on the final configuration.
        self.config = self._resolve_config_and_restart_status(self.config)
        apply_global_config(self.config)

        # 3. Environment Setup (logging, output paths, seeds)
        self._initialize_env()
        self.checkpoint_handler = CheckpointHandler(trainer=self)

        # 4. Load trainer state only if the synchronized config says to restart
        trainer_state = None
        if self.config.get('restart'):
            # 5. Load the raw state dictionary FIRST.
            trainer_state = self.checkpoint_handler.load_raw_state_for_restart()

            # 6. Inject the essential dataset indices into the config.
            # This ensures the DatasetBuilder uses the correct, restored indices.
            logging.info("Updating config with train/validation indices from checkpoint.")
            self.config['train_idcs'] = trainer_state['train_idcs']
            self.config['val_idcs'] = trainer_state['val_idcs']

        # 7. Extract config parameters (now includes restored indices if restarting)
        self._extract_config_parameters()

        # 8. Initialize training state variables
        self._initialize_training_state()

        # 9. Handle fine-tuning (only if it's a new run, not a restart).
        if self.config.get('fine_tune') and not self.config.get('restart'):
            model, self.config = self.checkpoint_handler.load_model_for_resume()

        # 10. Setup dataset and dataloaders
        # This will use the restored indices on a restart.
        self._build_datasets()
        self._init_dataloaders()

        # 11. Setup model and training-related objects
        self._setup_model(model)
        self._setup_training_components()

        # 12. If restarting, APPLY the rest of the state to the components we just built.
        if self.config.get('restart') and trainer_state is not None:
             self.checkpoint_handler.apply_state_for_restart(trainer_state)

        # 13. Callbacks
        self._setup_callbacks()

    def _resolve_config_and_restart_status(self, new_config: Config) -> Config:
        """
        Determines the definitive restart status and synchronizes the final config.
        Only the master rank checks the filesystem and determines the status.
        """
        # 1. Rank 0 checks the filesystem
        if self.dist.is_master:
            import os.path
            run_dir = f"{new_config.root}/{new_config.run_name}"
            is_restart = os.path.isdir(run_dir)
            
            if is_restart:
                logging.info(f"Master rank found existing run directory ({run_dir}), attempting to restart training.")
                final_config = check_for_config_updates(new_config)
            else:
                logging.info("Master rank starting a fresh training run.")
                final_config = new_config
            
            final_config['restart'] = is_restart
            
        else:
            # Non-master ranks wait to receive the final config
            is_restart = None
            final_config = None

        # 2. Synchronize (barrier for robustness, then broadcast)
        if self.dist.is_distributed:
            dist.barrier()

        # 3. Broadcast the final, resolved config object
        final_config = self.dist.broadcast_object(final_config, src=0)

        # The master's final_config is returned on Rank 0, 
        # the broadcasted final_config is returned on other ranks.
        return final_config

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
            config_path = self.output.generate_file("config.yaml")
            # Note: Must use the initial config's filepath to copy the *source* config.
            shutil.copyfile(Path(self.config.filepath).resolve(), config_path) 
            logging.info(f"Copied config file to {config_path}")
        else:
            self.output = None; self.logfile = "dummy"

        self.logger = logging.getLogger(self.logfile)
        set_seed(self.config.get('seed'))
        self.dataset_np_rng = np.random.default_rng(self.config.get('dataset_seed'))
        self.dataset_torch_rng = torch.Generator().manual_seed(self.config.get('dataset_seed'))

    def _extract_config_parameters(self):
        """Extract parameters from the config file to trainer attributes for easy access via self. """
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
        self.batch_losses, self.batch_metrics = {}, {}
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
        builder = DatasetBuilder(self.config, self.dataset_np_rng, self.logger, self.output)

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
        dl_kwargs = {"num_workers": self.config.get('dataloader_num_workers', 0), "pin_memory": self.dist.device.type == 'cuda', "generator": self.dataset_torch_rng}
        self.dl_train = DataLoader(dataset=self.train_dset, batch_size=self.config.get('batch_size'), shuffle=(self.train_sampler is None and self.shuffle), sampler=self.train_sampler, **dl_kwargs)
        self.dl_val = DataLoader(dataset=self.val_dset, batch_size=self.config.get('validation_batch_size'), shuffle=False, sampler=self.val_sampler, **dl_kwargs)

    def _setup_model(self, model):
        if model is None:
            model, _ = model_from_config(config=self.config, initialize=True, dataset=self.train_dset)
        self.num_weights = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.model = self.dist.wrap_model(model)
        if self.dist.is_master:
            self.logger.info(f"Number of trainable weights: {self.num_weights}")
            # self.logger.info(self.model)

    def _setup_training_components(self):
        self.loss = setup_loss(self.config)
        self.metrics = setup_metrics(self.config)
        self.optim = setup_optimizer(self.model, self.config)
        self.lr_sched, self.warmup_sched = setup_scheduler(self.optim, self.config, len(self.dl_train))
        self.ema = setup_ema(self.model, self.config)
        self.early_stopping_conds = setup_early_stopping(self.config)

    def _setup_callbacks(self):
        # Core callbacks
        callbacks = [Logger(), CheckpointCallback(), EarlyStoppingCallback()]

        # Optional integrations (e.g., Weights & Biases)
        if self.config.get('wandb'):
            from .components.callbacks import WandbCallback
            callbacks.insert(1, WandbCallback()) # Insert Wandb after Logger, before Checkpointing

        # Gradient processing callbacks (ORDER MATTERS for shared hooks)
        if self.config.get('sanitize_gradients', False):
            callbacks.append(SanitizeGradCallback())
        
        if self.config.get('use_grokfast', False):
            callbacks.append(GrokFastCallback())

        callbacks.append(GradientClippingCallback())

        # Other optional callbacks
        if self.config.get('track_activation_norms', True):
            callbacks.append(ActivationNormCallback())
        
        # User-defined external callbacks from config
        for cb_string in self.config.get('callbacks', []):
            callbacks.append(load_callable(cb_string)())
        
        # Register trainer with all callbacks
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
            # 1. Create the stateful summary object for the new epoch
            epoch_summary = EpochSummary(self)

            self._dispatch_callbacks('on_epoch_begin')
            
            # 2. Run epoch and pass the summary object to the loop to be populated
            self.training_loop.run_epoch(
                summary=epoch_summary,
                validation_only=(self.iepoch == -1)
            )
            
            # 3. Finalize the summary with end-of-epoch data (timings, LR)
            epoch_summary.finalize(self)
            
            # 4. Pass the completed summary to the callbacks
            self._dispatch_callbacks('on_epoch_end', summary=epoch_summary)
            
            # 5. Next epoch
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