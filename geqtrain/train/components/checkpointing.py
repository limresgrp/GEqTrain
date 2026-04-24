# components/checkpointing.py
import logging
from pathlib import Path
from os.path import isfile
from typing import Dict, Tuple, Mapping

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

    def _resolve_fine_tune_checkpoint_path(self) -> Path:
        fine_tune_path_str = self.trainer.config.get("fine_tune")
        if fine_tune_path_str is None:
            raise ValueError("`fine_tune` is not set in config.")
        fine_tune_path = Path(str(fine_tune_path_str))
        return fine_tune_path if fine_tune_path.is_file() else fine_tune_path / "last_model.pth"

    @staticmethod
    def _unwrap_model_state_dict(raw_state):
        if isinstance(raw_state, Mapping) and "model_state_dict" in raw_state:
            return raw_state["model_state_dict"]
        return raw_state

    @staticmethod
    def _format_key_list(keys, max_items: int = 20) -> str:
        keys = sorted(list(keys))
        if len(keys) <= max_items:
            return ", ".join(keys)
        head = ", ".join(keys[:max_items])
        return f"{head}, ... (+{len(keys) - max_items} more)"

    def apply_fine_tune_weights(self):
        """
        Load fine-tune weights into the current model architecture defined by the current config.

        Behavior:
          - Unexpected checkpoint keys (present in checkpoint, absent in model):
              fail by default; can be ignored with `fine_tune_ignore_unexpected_keys: true`.
          - Missing model keys (present in model, absent in checkpoint):
              always allowed with warning.
          - Same-name keys with mismatched tensor shapes:
              fail.
        """
        checkpoint_path = self._resolve_fine_tune_checkpoint_path()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Fine-tune checkpoint not found: {checkpoint_path}")

        model_to_load = self.trainer.model.module if self.trainer.dist.is_distributed else self.trainer.model
        model_state = model_to_load.state_dict()
        device = self.trainer.dist.device
        raw_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_state = self._unwrap_model_state_dict(raw_state)
        if not isinstance(checkpoint_state, Mapping):
            raise TypeError(
                f"Unsupported fine-tune checkpoint format at {checkpoint_path}: "
                f"expected a mapping, got {type(checkpoint_state)}"
            )
        checkpoint_state = dict(checkpoint_state)

        # Handle common DDP prefix mismatch.
        if (
            len(checkpoint_state) > 0
            and all(k.startswith("module.") for k in checkpoint_state.keys())
            and not any(k.startswith("module.") for k in model_state.keys())
        ):
            checkpoint_state = {k[len("module."):]: v for k, v in checkpoint_state.items()}

        model_keys = set(model_state.keys())
        checkpoint_keys = set(checkpoint_state.keys())
        unexpected_keys = checkpoint_keys - model_keys
        missing_keys = model_keys - checkpoint_keys

        shared_keys = checkpoint_keys & model_keys
        shape_mismatch_keys = sorted(
            [k for k in shared_keys if checkpoint_state[k].shape != model_state[k].shape]
        )
        if shape_mismatch_keys:
            mismatch_details = ", ".join(
                f"{k}: ckpt{tuple(checkpoint_state[k].shape)} vs model{tuple(model_state[k].shape)}"
                for k in shape_mismatch_keys[:20]
            )
            if len(shape_mismatch_keys) > 20:
                mismatch_details += f", ... (+{len(shape_mismatch_keys) - 20} more)"
            raise ValueError(
                "Fine-tune checkpoint has same-name parameters with incompatible shapes: "
                f"{mismatch_details}"
            )

        ignore_unexpected = bool(self.trainer.config.get("fine_tune_ignore_unexpected_keys", False))
        if unexpected_keys and not ignore_unexpected:
            raise ValueError(
                "Fine-tune checkpoint contains keys not present in current model architecture: "
                f"{self._format_key_list(unexpected_keys)}. "
                "Set `fine_tune_ignore_unexpected_keys: true` to ignore these keys."
            )
        if unexpected_keys and ignore_unexpected:
            logging.warning(
                "Ignoring %d unexpected fine-tune checkpoint keys not present in current model: %s",
                len(unexpected_keys),
                self._format_key_list(unexpected_keys),
            )

        filtered_state = {k: v for k, v in checkpoint_state.items() if k in model_state}
        load_result = model_to_load.load_state_dict(filtered_state, strict=False)
        if len(load_result.unexpected_keys) > 0:
            raise RuntimeError(
                "Internal error: unexpected keys remained after fine-tune filtering: "
                f"{self._format_key_list(load_result.unexpected_keys)}"
            )
        if len(load_result.missing_keys) > 0:
            logging.warning(
                "Current model has %d parameters/buffers with no counterpart in fine-tune checkpoint. "
                "They keep their initialized values. Missing keys: %s",
                len(load_result.missing_keys),
                self._format_key_list(load_result.missing_keys),
            )
        elif missing_keys:
            # Fallback in case load_result format changes.
            logging.warning(
                "Current model has %d parameters/buffers with no counterpart in fine-tune checkpoint. "
                "They keep their initialized values. Missing keys: %s",
                len(missing_keys),
                self._format_key_list(missing_keys),
            )

        loaded_count = len(filtered_state)
        logging.info(
            "Loaded %d tensors from fine-tune checkpoint %s into current model.",
            loaded_count,
            checkpoint_path,
        )
    
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

        configured_group_lrs = self._get_configured_optimizer_lrs()

        # Load states into the already initialized components
        model_to_load = self.trainer.model.module if self.trainer.dist.is_distributed else self.trainer.model
        model_to_load.load_state_dict(state['model_state_dict'])
        self.trainer.optim.load_state_dict(state['optim_state_dict'])

        if self.trainer.lr_sched and state.get('sched_state_dict') is not None:
            self.trainer.lr_sched.load_state_dict(state['sched_state_dict'])

        if self.trainer.ema and state.get('ema_state_dict') is not None:
            self.trainer.ema.load_state_dict(state['ema_state_dict'])

        lr_overridden = self._is_learning_rate_overridden(state)
        if lr_overridden:
            target_group_lrs = configured_group_lrs
            self._set_optimizer_group_lrs(target_group_lrs)
            self._sync_scheduler_lrs(target_group_lrs)
            logging.info(
                "Detected updated `learning_rate` in restart config; warmup target LR "
                "will use the configured optimizer LR values."
            )
        else:
            target_group_lrs = [group["lr"] for group in self.trainer.optim.param_groups]

        if self.trainer.warmup_sched is not None:
            self._reset_warmup_scheduler(target_group_lrs)
            logging.info(
                "Restart warmup reset from step 0 with target LRs: %s",
                [float(lr) for lr in target_group_lrs],
            )

        # Load trainer metadata
        self.trainer.iepoch = state['iepoch']
        self.trainer.best_epoch = state['best_epoch']
        self.trainer.best_metrics = state['best_metrics']
        self.trainer.cumulative_wall = state.get('cumulative_wall', 0)
        
        logging.info(f"Successfully applied state. Resuming from epoch {self.trainer.iepoch + 1}.")

    def _get_configured_optimizer_lrs(self):
        if self.trainer.warmup_sched is not None and hasattr(self.trainer.warmup_sched, "lrs"):
            return [float(lr) for lr in self.trainer.warmup_sched.lrs]
        return [float(group["lr"]) for group in self.trainer.optim.param_groups]

    def _set_optimizer_group_lrs(self, lrs):
        if len(lrs) != len(self.trainer.optim.param_groups):
            raise ValueError(
                f"Cannot assign {len(lrs)} learning rates to {len(self.trainer.optim.param_groups)} optimizer groups."
            )
        for group, lr in zip(self.trainer.optim.param_groups, lrs):
            lr = float(lr)
            group["lr"] = lr
            if "initial_lr" in group:
                group["initial_lr"] = lr

    def _sync_scheduler_lrs(self, lrs):
        lr_sched = self.trainer.lr_sched
        if lr_sched is None:
            return
        if hasattr(lr_sched, "base_lrs"):
            lr_sched.base_lrs = [float(lr) for lr in lrs]
        if hasattr(lr_sched, "_last_lr"):
            lr_sched._last_lr = [float(lr) for lr in lrs]

    def _reset_warmup_scheduler(self, target_lrs):
        warmup_sched = self.trainer.warmup_sched
        if warmup_sched is None:
            return
        self._set_optimizer_group_lrs(target_lrs)
        warmup_sched.lrs = [float(lr) for lr in target_lrs]
        warmup_sched.last_step = -1
        # Re-apply the first warmup dampening factor so the next optimizer step uses warm-start LR.
        warmup_sched.dampen()

    def _is_learning_rate_overridden(self, state: dict) -> bool:
        checkpoint_config = state.get("config", None)
        if checkpoint_config is None:
            return False
        saved_lr = checkpoint_config.get("learning_rate", None)
        current_lr = self.trainer.config.get("learning_rate", None)
        return current_lr != saved_lr

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
        file_config = Config.from_file(traindir / "config.yaml")
        config = file_config
        trainer_path = traindir / "trainer.pth"
        if trainer_path.is_file():
            trainer_state = torch.load(trainer_path, map_location="cpu", weights_only=False)
            trainer_config = trainer_state.get("config")
            if trainer_config is not None:
                config = Config(trainer_config)
                for key, value in file_config.items():
                    if key not in config:
                        config[key] = value
                logging.info("Loaded model config from trainer.pth and backfilled missing keys from config.yaml.")
        
        logging.info("Building model from training config...")
        model, _ = model_from_config(config=config, initialize=False)
        
        logging.info(f"Loading model state dict from {model_name}...")
        state_dict = torch.load(traindir / model_name, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, config
    
    @staticmethod
    def load_model(model_path_str: str, device="cpu") -> Tuple[torch.jit.ScriptModule, Config, Dict]:
        """
        Loads a model, intelligently handling both deployed models and training checkpoints.

        Returns:
            A tuple containing:
            - The loaded PyTorch model (torch.jit.ScriptModule).
            - The model's configuration (Config object).
            - A dictionary of metadata (empty if loaded from training).
        """
        from geqtrain.utils.deploy import _ALL_METADATA_KEYS, CONFIG_KEY
        metadata = {k: "" for k in _ALL_METADATA_KEYS}
        try:
            # First, try to load as a deployed model
            import yaml
            from geqtrain.utils.deploy import load_deployed_model
            model, metadata = load_deployed_model(model_path_str, device=device, freeze=False, extra_metadata=metadata)
            # The config is stored as a YAML string within the metadata
            model_config_str = metadata.get(CONFIG_KEY, "{}")
            model_config = Config(yaml.safe_load(model_config_str))
            logging.info("Successfully loaded deployed model and its metadata.")
        except (ValueError, RuntimeError, FileNotFoundError): 
            # If it fails, fall back to loading from a training session directory
            logging.warning("Could not load as a deployed model. Falling back to loading from a training session.")
            from geqtrain.train.components.checkpointing import CheckpointHandler
            model_path = Path(model_path_str)
            model, model_config = CheckpointHandler.load_model_from_training_session(
                traindir=model_path.parent, 
                model_name=model_path.name, 
                device=device,
            )
            metadata = {} # No metadata available for simple training checkpoints
        return model, model_config, metadata
