# geqtrain/train/components/epoch_summary.py
from collections import defaultdict
from typing import Dict, Union, List
import torch

from geqtrain.train._key import TRAIN, VALIDATION

class EpochSummary:
    """
    A stateful builder that accumulates all results and metadata from a single training epoch.
    """
    def __init__(self, trainer):
        # Initialize with static information and empty containers for results
        self.epoch = trainer.iepoch + 1
        self._metrics_computer = trainer.metrics
        self._metrics_metadata = trainer.metrics_metadata

        # Containers for accumulating data
        self.loss_dict = {}
        self.metrics_dict = {}
        self.grad_norms = []
        self.grad_norms_clip = []
        self.node_feature_norms = defaultdict(list)
        
        # Placeholders for final data
        self.lr = None
        self.train_wall = 0.0
        self.validation_wall = 0.0
        self.cumulative_wall = 0.0

        # Lazy-generated flat dictionary
        self._flat_dict = None

    def set_phase_results(self, phase: str, loss_results: dict, metrics_results: dict):
        """Records the final aggregated results for a training or validation phase."""
        self.loss_dict[phase] = loss_results
        self.metrics_dict[phase] = metrics_results

    def add_grad_norm(self, norm: float, norm_clip: float):
        """Adds a gradient norm measurement from one optimization step."""
        self.grad_norms.append(norm)
        self.grad_norms_clip.append(norm_clip)

    def add_node_feature_norms(self, norms_dict: dict):
        """Adds node feature norm measurements from a batch."""
        for key, norm in norms_dict.items():
            self.node_feature_norms[key].append(norm)

    def finalize(self, trainer):
        """Finalizes the summary by adding data available only at the end of the epoch."""
        self.lr = trainer.optim.param_groups[0]["lr"]
        self.train_wall = trainer.train_wall if self.epoch > 0 else 0.0
        self.validation_wall = trainer.validation_wall
        self.cumulative_wall = trainer.cumulative_wall
        # Invalidate cached dictionary
        self._flat_dict = None
    
    def get_loss_keys(self, phase: str) -> list:
        """Returns the original loss keys for a given phase."""
        return list(self.loss_dict.get(phase, {}).keys())
    
    def get_flattened_metric_keys(self, phase: str) -> list:
        """
        Returns the flattened metric keys for a given phase,
        matching the keys used in to_flat_dict.
        """
        metrics_for_phase = self.metrics_dict.get(phase, {})
        if not metrics_for_phase:
            return []
        
        # We don't need the values, just the keys from the flattening process
        flattened_metrics = self._metrics_computer.flatten_metrics(
            metrics_for_phase, self._metrics_metadata
        )
        return list(flattened_metrics.keys())

    def get_target_metric(self, key: Union[str, List[str]]) -> float:
        """Calculates the target metric used for checkpointing and schedulers."""
        # Ensure the dict is generated with final data
        flat_dict = self.to_flat_dict()
        if isinstance(key, (list, tuple)):
            values = torch.tensor([flat_dict[metric] for metric in key], dtype=torch.float32)
            return values.mean().item()
        return flat_dict.get(key)

    def to_flat_dict(self) -> Dict[str, Union[int, float]]:
        """Generates and caches a flat dictionary of all epoch results for logging."""
        if self._flat_dict is not None:
            return self._flat_dict

        # This method now uses the state that has been built up over the epoch
        # (The logic inside this method is identical to the previous version)
        flat_dict = {
            "epoch": self.epoch, "LR": self.lr,
            "train_wall": self.train_wall, "validation_wall": self.validation_wall,
            "cumulative_wall": self.cumulative_wall,
        }
        categories = [TRAIN, VALIDATION] if self.epoch > 0 else [VALIDATION]
        for category in categories:
            if category in self.loss_dict:
                for key, value in self.loss_dict[category].items():
                    flat_dict[f"{category}_{key}"] = value
            if category in self.metrics_dict:
                metrics = self._metrics_computer.flatten_metrics(self.metrics_dict[category], self._metrics_metadata)
                for key, value in metrics.items():
                    flat_dict[f"{category}_{key}"] = value
        if self.grad_norms:
            flat_dict['train_grad_norm_mean'] = torch.tensor(self.grad_norms).mean().item()
        if self.node_feature_norms:
             for key, norms in self.node_feature_norms.items():
                if norms:
                    flat_dict[f'train_{key}_norm'] = torch.stack(norms).mean().item()
        self._flat_dict = flat_dict
        return self._flat_dict