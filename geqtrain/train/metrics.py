""" Adapted from https://github.com/mir-group/nequip
"""
import inspect
from typing import Dict, List, Union

import torch

from geqtrain.data import AtomicDataDict
from geqtrain.train.loss import Loss
from geqtrain.utils.torch_runstats._runstats import RunningStats, Reduction
from ._key import ABBREV


class Metrics(Loss):
    """
    Computes running statistics for various metrics (MAE, RMSE, etc.) over the course of training.
    This class correctly inherits from `Loss` to reuse its component parsing logic.
    """
    def __init__(self, components: Union[str, List[str], List[dict]]):
        # Initialize the parent Loss class to parse components and populate
        # self.keys, self.funcs, etc.
        super().__init__(components)

        # Initialize state specific to Metrics
        self.running_stats: Dict[str, RunningStats] = {}
        self.params: Dict[str, dict] = {}
        
        # Configure RunningStats for each metric component found by the parent
        for key in self.keys:
            func_params = self.func_params.get(key, {})
            
            # Set defaults for metric-specific options
            func_params.setdefault("PerSpecies", False)
            func_params.setdefault("PerTarget", False)
            func_params.setdefault("functional", "L1Loss")
            
            reduction_str = func_params.get("reduction")
            if reduction_str is None and hasattr(self.funcs[key], "reduction"):
                reduction_str = self.funcs[key].reduction
            
            reductions = {'mean': Reduction.MEAN, 'rms': Reduction.RMS, 'latest': Reduction.LATEST}
            reduction = reductions.get(reduction_str, Reduction.MEAN)
            
            self.params[key] = {'reduction': reduction, **func_params}

    def _init_runstat(self, key: str, error: torch.Tensor) -> RunningStats:
        """Initialize a RunningStats counter based on error shape and config."""
        params = self.params[key]
        # Filter kwargs to only those accepted by RunningStats constructor
        init_kwargs = {k: v for k, v in params.items() if k in inspect.signature(RunningStats).parameters}
        
        init_kwargs.setdefault("dim", error.shape[1:])
        
        if "reduce_dims" not in init_kwargs and not params.get("report_per_component", False):
            init_kwargs["reduce_dims"] = tuple(range(len(error.shape) - 1))
            
        # Correctly instantiate first, then move to device
        rs = RunningStats(**init_kwargs)
        rs.to(error.device)
        return rs

    def _prepare_accumulation_params(self, key: str, error: torch.Tensor, ref: dict) -> dict:
        """Prepare parameters for `accumulate_batch`, handling PerSpecies and PerTarget logic."""
        params = self.params[key]
        accum_params = {}
        
        if params.get("PerSpecies"):
            node_types = ref[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
            center_nodes = torch.unique(ref[AtomicDataDict.EDGE_INDEX_KEY][0])
            if len(error) != len(center_nodes):
                error = error[center_nodes]
            accum_params["accumulate_by"] = node_types[center_nodes]
            
        if params.get("PerTarget"):
            num_rows, num_targets = error.shape
            accum_by = accum_params.get("accumulate_by", torch.zeros(num_rows, device=error.device, dtype=torch.long))
            per_target_accum_by = accum_by * num_targets + torch.arange(num_targets, device=error.device).unsqueeze(0)
            accum_params["accumulate_by"] = per_target_accum_by.flatten()
            
        return accum_params

    def __call__(self, pred: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Update metrics with the current batch and return per-batch values."""
        metrics = {}
        for key in self.keys:
            clean_key = self.remove_suffix(key)
            error = self.funcs[key](pred=pred, ref=ref, key=clean_key, mean=False)
            if error.dim() > 1 and not self.params[key].get("PerTarget"):
                 error = error.mean(dim=tuple(range(1, error.dim())))

            # Lazily initialize the RunningStats object on the first batch
            if key not in self.running_stats:
                self.running_stats[key] = self._init_runstat(key, error)
            
            stat = self.running_stats[key]
            accum_params = self._prepare_accumulation_params(key, error, ref)
            
            metrics[key] = stat.accumulate_batch(error, **accum_params)
            
        return metrics

    def reset(self):
        """Reset all running statistics."""
        for stat in self.running_stats.values():
            stat.reset()

    def to(self, device):
        """Move all running statistics to a specified device."""
        for stat in self.running_stats.values():
            stat.to(device=device)

    def current_result(self):
        """Get the current accumulated results for the epoch."""
        return {key: stat.current_result() for key, stat in self.running_stats.items()}

    def flatten_metrics(self, metrics: Dict[str, torch.Tensor], metrics_metadata: Dict[str, List[str]] = None) -> Dict[str, float]:
        """Flattens the metrics dictionary for easy logging and reporting."""
        metrics_metadata = metrics_metadata or {}
        type_names = metrics_metadata.get('type_names')
        target_names = metrics_metadata.get('target_names')
        flat_dict = {}

        for key, value in metrics.items():
            params = self.params[key]
            reduction = params['reduction']
            
            key_clean = self.remove_suffix(key)
            metric_name = ABBREV.get(key_clean, key_clean)
            loss_name = str(self.funcs[key])
            metric_key = f"{metric_name}_{loss_name}_{reduction.name.lower()}"

            if params.get("PerSpecies"):
                for idx, value_row in enumerate(value):
                    species_name = type_names[idx] if type_names and idx < len(type_names) else f"type_{idx}"
                    base_key = f"{species_name}_{metric_key}"
                    if params.get("PerTarget"):
                        for target_idx, item in enumerate(value_row):
                            target_name = target_names[target_idx] if target_names and target_idx < len(target_names) else f"target_{target_idx}"
                            flat_dict[f"{base_key}_{target_name}"] = item.item()
                    else:
                        flat_dict[base_key] = value_row.item()
            elif params.get("PerTarget"):
                for target_idx, item in enumerate(value):
                    target_name = target_names[target_idx] if target_names and target_idx < len(target_names) else f"target_{target_idx}"
                    flat_dict[f"{metric_key}_{target_name}"] = item.item()
            else:
                flat_dict[metric_key] = value.item()
                
        return flat_dict