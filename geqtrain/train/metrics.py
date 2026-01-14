# geqtrain/train/metrics.py

import inspect
from typing import Dict, List, Union

import torch
import torch.distributed as dist

from geqtrain.data import AtomicDataDict
from geqtrain.train.loss import Loss
from geqtrain.utils.torch_runstats._runstats import RunningStats, Reduction
from ._key import ABBREV
from ._loss import StatefulMetric
from .loss import Loss

class _Metric:
    """Internal helper class to manage the state and logic for a single metric."""
    def __init__(self, func: callable, params: dict):
        self.func = func
        self.params = params
        self.accumulator: Union[RunningStats, StatefulMetric, None] = None

        # If the metric is stateful, it acts as its own accumulator
        if isinstance(self.func, StatefulMetric):
            self.accumulator = self.func

    def accumulate(self, pred: dict, ref: dict, key: str, destandardize_fields: dict) -> torch.Tensor:
        """Calculates and accumulates the metric for the current batch."""
        if isinstance(self.accumulator, StatefulMetric):
            self.accumulator.update(pred, ref, key)
            return self.accumulator.compute()  # Return partial result for batch logs
        
        # --- Logic for stateless (RunningStats) metrics ---
        error = self.func(pred=pred, ref=ref, key=key, mean=False, destandardize_fields=destandardize_fields)
        
        # If per-target metrics are not requested, average over the feature dimension
        if error.dim() > 1 and not self.params.get("PerTarget"):
            error = error.mean(dim=tuple(range(1, error.dim())))
        
        # Lazily initialize the RunningStats accumulator on the first batch
        if self.accumulator is None:
            self._init_runstat(error)

        accum_params = self._prepare_accumulation_params(error, ref)
        return self.accumulator.accumulate_batch(error, **accum_params)

    def get_final_result(self) -> torch.Tensor:
        if isinstance(self.accumulator, StatefulMetric):
            return self.accumulator.compute()
        return self.accumulator.current_result() if self.accumulator else None

    def reset(self):
        if self.accumulator:
            self.accumulator.reset()

    def _init_runstat(self, error: torch.Tensor):
        """Initializes the RunningStats accumulator."""
        init_kwargs = {k: v for k, v in self.params.items() if k in inspect.signature(RunningStats).parameters}
        init_kwargs.setdefault("dim", error.shape[1:])
        
        # Default to reducing over all component dimensions if not otherwise specified
        if "reduce_dims" not in init_kwargs and not self.params.get("report_per_component", False):
            init_kwargs["reduce_dims"] = tuple(range(len(error.shape) - 1))
            
        self.accumulator = RunningStats(**init_kwargs)
        self.accumulator.to(error.device)

    def _prepare_accumulation_params(self, error: torch.Tensor, ref: dict) -> dict:
        """Prepares the `accumulate_by` tensor for PerSpecies or PerTarget logic."""
        accum_params = {}
        if self.params.get("PerSpecies"):
            node_types = ref[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
            center_nodes_idx = ref[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
            # This logic assumes the error is per-node. A check might be needed.
            accum_params["accumulate_by"] = node_types[center_nodes_idx]

        if self.params.get("PerTarget"):
            num_rows, num_targets = error.shape
            accum_by = accum_params.get("accumulate_by", torch.zeros(num_rows, device=error.device, dtype=torch.long))
            per_target_accum_by = accum_by * num_targets + torch.arange(num_targets, device=error.device).unsqueeze(0)
            accum_params["accumulate_by"] = per_target_accum_by.flatten()
            
        return accum_params


class Metrics(Loss):
    def __init__(self, components: Union[str, List[str], List[dict]], destandardize_fields: dict = {}):
        super().__init__(components)
        self.destandardize_fields = destandardize_fields
        self.metrics: Dict[str, _Metric] = {}
        
        for key in self.keys:
            func = self.funcs[key]
            params: dict = self.func_params.get(key, {})
            if hasattr(func, "extra_params"):
                params.update(func.extra_params)
            
            # Set defaults and process reduction parameter for stateless metrics
            if not isinstance(func, StatefulMetric):
                params.setdefault("PerSpecies", False)
                params.setdefault("PerTarget", False)
                reduction_str = params.get("reduction", "mean")
                reductions = {'mean': Reduction.MEAN, 'rms': Reduction.RMS}
                params['reduction'] = reductions.get(reduction_str, Reduction.MEAN)
            
            self.metrics[key] = _Metric(func, params)

    def __call__(self, pred: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_metrics = {}
        for key, metric_handler in self.metrics.items():
            clean_key = self.remove_suffix(key)
            batch_metrics[key] = metric_handler.accumulate(pred, ref, clean_key, self.destandardize_fields)
        return batch_metrics

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def current_result(self, dist_manager=None) -> Dict[str, torch.Tensor]:
        if dist_manager is not None and dist_manager.is_distributed:
            self._sync_running_stats(dist_manager)

        final_metrics = {}
        for key, metric_handler in self.metrics.items():
            result = metric_handler.get_final_result()
            if result is not None:
                if dist_manager is not None and dist_manager.is_distributed and isinstance(metric_handler.accumulator, StatefulMetric):
                    result = dist_manager.sync_tensor(result)
                final_metrics[key] = result
        return final_metrics

    def flatten_metrics(self, metrics: Dict[str, torch.Tensor], metrics_metadata: Dict[str, List[str]] = None) -> Dict[str, float]:
        """Flattens the metrics dictionary for easy logging and reporting."""
        metrics_metadata = metrics_metadata or {}
        type_names = metrics_metadata.get('type_names')
        target_names = metrics_metadata.get('target_names')
        flat_dict = {}

        for key, value in metrics.items():
            handler = self.metrics[key]
            params = handler.params
            
            key_clean = self.remove_suffix(key)
            metric_name = ABBREV.get(key_clean, key_clean)
            loss_name = str(handler.func)
            reduction_name = params.get('reduction', Reduction.MEAN).name.lower()
            metric_key = f"{metric_name}_{loss_name}_{reduction_name}"

            # This complex formatting logic remains, as it's required for detailed logging
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

    def _sync_running_stats(self, dist_manager):
        device = dist_manager.device
        for metric_handler in self.metrics.values():
            accumulator = metric_handler.accumulator
            if not isinstance(accumulator, RunningStats):
                continue

            local_bins = torch.tensor([accumulator.n_bins], device=device, dtype=torch.long)
            dist.all_reduce(local_bins, op=dist.ReduceOp.MAX)
            max_bins = int(local_bins.item())

            if accumulator.n_bins < max_bins:
                pad = max_bins - accumulator.n_bins
                accumulator._state = torch.cat(
                    (
                        accumulator._state,
                        accumulator._state.new_zeros((pad,) + accumulator._state.shape[1:]),
                    ),
                    dim=0,
                )
                accumulator._n = torch.cat(
                    (
                        accumulator._n,
                        accumulator._n.new_zeros((pad,) + accumulator._n.shape[1:]),
                    ),
                    dim=0,
                )
                accumulator._n_bins = max_bins

            counts = accumulator._n.to(dtype=accumulator._state.dtype)
            sums = accumulator._state * counts

            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

            accumulator._state = torch.where(
                counts > 0,
                sums / counts,
                accumulator._state.new_zeros(accumulator._state.shape),
            )
            accumulator._n = counts.round().to(dtype=accumulator._n.dtype)
