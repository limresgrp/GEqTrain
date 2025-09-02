# components/checkpointing.py
import logging
from pathlib import Path
from os.path import isfile

import torch

from geqtrain.utils import load_file, save_file, atomic_write
from geqtrain.model import model_from_config
from geqtrain.utils.config import Config


class CheckpointHandler:
    """Handles saving and loading of trainer and model states."""
    def __init__(self, trainer):
        self.trainer = trainer
        self._init_paths()

    def _init_paths(self):
        # Master rank determines the paths
        if self.trainer.dist.is_master:
            output = self.trainer.output
            self.best_model_path = output.generate_file("best_model.pth")
            self.last_model_path = output.generate_file("last_model.pth")
            self.trainer_save_path = output.generate_file("trainer.pth")
        else:
            self.best_model_path, self.last_model_path, self.trainer_save_path = None, None, None

        # Broadcast the paths from master to all other ranks
        # Now all processes will have the correct path strings
        self.best_model_path = self.trainer.dist.broadcast_object(self.best_model_path)
        self.last_model_path = self.trainer.dist.broadcast_object(self.last_model_path)
        self.trainer_save_path = self.trainer.dist.broadcast_object(self.trainer_save_path)

    def save(self, blocking: bool = True):
        """Save the complete trainer state."""
        if not self.trainer.dist.is_master: return
        state = self._get_state_dict()
        save_file(item=state, supported_formats=dict(torch=["pt", "pth"]), filename=self.trainer_save_path, blocking=blocking)
        logging.debug(f"Saved trainer state to {self.trainer_save_path}")
        self.save_ema_model(self.last_model_path, blocking=blocking)

    def save_model(self, path, model_state_dict, blocking: bool = True):
        if not self.trainer.dist.is_master: return
        with atomic_write(path, blocking=blocking, binary=True) as f:
            torch.save(model_state_dict, f)

    def save_ema_model(self, path, blocking: bool = True):
        model_to_save = self.trainer.model.module if self.trainer.dist.is_distributed else self.trainer.model
        if self.trainer.ema is not None:
            with self.trainer.ema.average_parameters():
                state_dict = model_to_save.state_dict()
                self.save_model(path, state_dict, blocking)
        else:
            self.save_model(path, model_to_save.state_dict(), blocking)

    def _get_state_dict(self):
        """Collects the current state of the trainer."""
        model_to_save = self.trainer.model.module if self.trainer.dist.is_distributed else self.trainer.model
        return {
            'config': self.trainer.config,
            'model_state_dict': model_to_save.state_dict(),
            'optim_state_dict': self.trainer.optim.state_dict(),
            'sched_state_dict': self.trainer.lr_sched.state_dict() if self.trainer.lr_sched else None,
            'ema_state_dict': self.trainer.ema.state_dict() if self.trainer.ema else None,
            'iepoch': self.trainer.iepoch,
            'best_epoch': self.trainer.best_epoch,
            'best_metrics': self.trainer.best_metrics,
            'cumulative_wall': self.trainer.cumulative_wall,
            'train_idcs': self.trainer.train_idcs,
            'val_idcs': self.trainer.val_idcs
        }

    def load_model_for_resume(self):
        """Handles the logic for `fine_tune` by calling the shared static method."""
        config = self.trainer.config
        fine_tune_path_str = config.get("fine_tune")
        
        fine_tune_path = Path(fine_tune_path_str)
        traindir = fine_tune_path.parent if fine_tune_path.is_file() else fine_tune_path
        model_name = fine_tune_path.name if fine_tune_path.is_file() else "last_model.pth"
        
        model, updated_config = CheckpointHandler.load_model_from_training_session(
            traindir=traindir, 
            model_name=model_name, 
            device=self.trainer.dist.device,
        )
        
        updated_config = self._load_indices_from_run(updated_config, traindir)
        return model, updated_config
    
    def load_raw_state_for_restart(self):
        """Load the trainer state dictionary from the checkpoint file without applying it."""
        logging.info("Loading raw state from checkpoint for restart...")
        
        device = self.trainer.dist.device
        state = load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=self.trainer_save_path,
            enforced_format="torch",
            map_location=device,
            weights_only=False,
        )
        return state

    def apply_state_for_restart(self, state: dict):
        """Apply a loaded state dictionary to the trainer's components."""
        logging.info("Applying loaded state to trainer components...")
        
        # Load states into the already initialized components
        model_to_load = self.trainer.model.module if self.trainer.dist.is_distributed else self.trainer.model
        model_to_load.load_state_dict(state['model_state_dict'])
        self.trainer.optim.load_state_dict(state['optim_state_dict'])
        
        if self.trainer.lr_sched and state.get('sched_state_dict') is not None:
            self.trainer.lr_sched.load_state_dict(state['sched_state_dict'])
            
        if self.trainer.ema and state.get('ema_state_dict') is not None:
            self.trainer.ema.load_state_dict(state['ema_state_dict'])
        
        # Load trainer metadata
        self.trainer.iepoch = state['iepoch']
        self.trainer.best_epoch = state['best_epoch']
        self.trainer.best_metrics = state['best_metrics']
        self.trainer.cumulative_wall = state.get('cumulative_wall', 0)
        
        logging.info(f"Successfully applied state. Resuming from epoch {self.trainer.iepoch + 1}.")

    def _load_indices_from_run(self, config, traindir):
        """
        Load train/validation indices from a previous run's checkpoint and
        update the config to use them.
        """
        load_train = config.get('train_idcs') == 'load'
        load_val = config.get('val_idcs') == 'load'

        if load_train or load_val:
            original_trainer_path = traindir / "trainer.pth"
            assert isfile(original_trainer_path), f"trainer.pth not found in {traindir}, required for loading dataset indices."
            
            original_state = load_file(
                supported_formats=dict(torch=["pt", "pth"]),
                filename=str(original_trainer_path),
                enforced_format="torch",
                weights_only=False,
            )
            
            if load_train:
                assert 'train_idcs' in original_state, "Key 'train_idcs' not found in the original trainer state."
                # Update the config dict. This config will be passed to the DatasetBuilder.
                config['train_idcs'] = original_state['train_idcs']
                logging.info("Successfully loaded 'train_idcs' from previous run.")
            
            if load_val:
                assert 'val_idcs' in original_state, "Key 'val_idcs' not found in the original trainer state."
                # Update the config dict.
                config['val_idcs'] = original_state['val_idcs']
                logging.info("Successfully loaded 'val_idcs' from previous run.")
                
        return config
    
    @staticmethod
    def load_model_from_training_session(traindir, model_name="best_model.pth", device="cpu"):
        """
        Loads a model and its configuration from a training session directory.
        This is a static method and can be called without a Trainer instance.
        """
        traindir = Path(traindir)
        config = Config.from_file(traindir / "config.yaml")
        
        logging.info("Building model from training config...")
        model, _ = model_from_config(config=config, initialize=False)
        
        logging.info(f"Loading model state dict from {model_name}...")
        state_dict = torch.load(traindir / model_name, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, config