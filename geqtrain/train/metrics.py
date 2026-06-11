# geqtrain/train/metrics.py

import inspect
from typing import Dict, List, Union

import torch
import torch.distributed as dist

from geqtrain.data import AtomicDataDict
from geqtrain.data import _NODE_FIELDS
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
        self.node_type_indices = self._resolve_node_type_indices()
        self.node_mask_field = self.params.pop("node_mask_field", self.params.pop("node_mask_key", None))

        # If the metric is stateful, it acts as its own accumulator
        if isinstance(self.func, StatefulMetric):
            self.accumulator = self.func

    def _resolve_node_type_indices(self):
        node_type_indices = self.params.pop("node_type_indices", None)
        node_type_names = self.params.pop("node_type_names", None)
        type_names = self.params.pop("type_names", None)

        if node_type_indices is not None and node_type_names is not None:
            raise ValueError("Specify only one of `node_type_indices` or `node_type_names`.")

        if node_type_indices is not None:
            if isinstance(node_type_indices, (int, str)):
                node_type_indices = [node_type_indices]
            return torch.tensor([int(v) for v in node_type_indices], dtype=torch.long)

        if node_type_names is not None:
            if type_names is None:
                raise ValueError(
                    "`node_type_names` was provided for a metric, but no `type_names` list was found in the metric parameters. "
                    "Add `type_names` next to the metric entry or use `node_type_indices` instead."
                )
            if isinstance(node_type_names, str):
                node_type_names = [node_type_names]
            if isinstance(type_names, str):
                type_names = [type_names]
            type_name_to_idx = {str(name): idx for idx, name in enumerate(type_names)}
            missing = [name for name in node_type_names if str(name) not in type_name_to_idx]
            if missing:
                raise ValueError(
                    f"Unknown node type names in metric filter: {missing}. Available type names: {list(type_name_to_idx.keys())}"
                )
            return torch.tensor([type_name_to_idx[str(name)] for name in node_type_names], dtype=torch.long)

        return None

    def _resolve_node_mask(self, pred: dict, ref: dict):
        if self.node_mask_field is None:
            return None

        mask_source = pred if self.node_mask_field in pred else ref
        if self.node_mask_field not in mask_source:
            return None

        node_mask = mask_source[self.node_mask_field]
        if not torch.is_tensor(node_mask):
            node_mask = torch.as_tensor(node_mask)
        node_mask = node_mask.to(dtype=torch.bool)
        if node_mask.ndim > 1:
            node_mask = node_mask.squeeze(-1)
        return node_mask

    def _apply_node_filter(self, pred: dict, ref: dict, key: str):
        species_mask = None
        if self.node_type_indices is not None and (AtomicDataDict.NODE_TYPE_KEY in pred or AtomicDataDict.NODE_TYPE_KEY in ref):
            node_type_source = pred if AtomicDataDict.NODE_TYPE_KEY in pred else ref
            node_types = node_type_source[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
            species_mask = torch.isin(node_types, self.node_type_indices.to(node_types.device))

        node_mask = self._resolve_node_mask(pred, ref)
        combined_mask = None
        if species_mask is not None:
            combined_mask = species_mask
        if node_mask is not None:
            combined_mask = node_mask if combined_mask is None else (combined_mask & node_mask)

        if combined_mask is not None:
            if key in _NODE_FIELDS or (key in pred and pred[key].ndim > 0 and pred[key].shape[0] == combined_mask.shape[0]):
                pred = dict(pred)
                pred[key] = pred[key][combined_mask]
            if key in _NODE_FIELDS or (key in ref and ref[key].ndim > 0 and ref[key].shape[0] == combined_mask.shape[0]):
                ref = dict(ref)
                ref[key] = ref[key][combined_mask]
        return pred, ref, combined_mask

    def accumulate(self, pred: dict, ref: dict, key: str, normalization_fields: dict) -> torch.Tensor:
        """Calculates and accumulates the metric for the current batch."""
        pred, ref, node_mask = self._apply_node_filter(pred, ref, key)
        if isinstance(self.accumulator, StatefulMetric):
            self.accumulator.update(pred, ref, key)
            return self.accumulator.compute()  # Return partial result for batch logs
        
        # --- Logic for stateless (RunningStats) metrics ---
        error = self.func(
            pred=pred,
            ref=ref,
            key=key,
            mean=False,
            normalization_fields=normalization_fields,
        )
        
        # If per-target metrics are not requested, average over the feature dimension
        if error.dim() > 1 and not self.params.get("PerTarget"):
            error = error.mean(dim=tuple(range(1, error.dim())))
        
        # Lazily initialize the RunningStats accumulator on the first batch
        if self.accumulator is None:
            self._init_runstat(error)

        accum_params = self._prepare_accumulation_params(error, ref, node_mask)
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

    def _prepare_accumulation_params(self, error: torch.Tensor, ref: dict, node_mask: torch.Tensor = None) -> dict:
        """Prepares the `accumulate_by` tensor for PerSpecies or PerTarget logic."""
        accum_params = {}
        if self.params.get("PerSpecies"):
            node_types = ref[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
            if node_mask is not None and node_mask.shape[0] == node_types.shape[0]:
                # `node_mask` already encodes the final node selection applied to
                # pred/ref, so using it keeps the accumulation bins aligned with the
                # filtered error tensor.
                accum_params["accumulate_by"] = node_types[node_mask]
            else:
                center_nodes_idx = ref[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
                accum_params["accumulate_by"] = node_types[center_nodes_idx]

        if self.params.get("PerTarget"):
            num_rows, num_targets = error.shape
            accum_by = accum_params.get("accumulate_by", torch.zeros(num_rows, device=error.device, dtype=torch.long))
            per_target_accum_by = accum_by * num_targets + torch.arange(num_targets, device=error.device).unsqueeze(0)
            accum_params["accumulate_by"] = per_target_accum_by.flatten()
            
        return accum_params


class Metrics(Loss):
    def __init__(
        self,
        components: Union[str, List[str], List[dict]],
        normalization_fields: dict = None,
    ):
        super().__init__(components)
        self.normalization_fields = {} if normalization_fields is None else normalization_fields
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
            batch_metrics[key] = metric_handler.accumulate(
                pred,
                ref,
                clean_key,
                self.normalization_fields,
            )
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
            if params.get("PerSpecies") or handler.node_type_indices is not None:
                species_indices = (
                    handler.node_type_indices.tolist()
                    if handler.node_type_indices is not None
                    else list(range(len(value)))
                )
                if len(value) == len(species_indices):
                    species_pairs = list(enumerate(species_indices))
                    value_lookup = lambda local_idx, species_idx: value[local_idx]
                else:
                    species_pairs = [(species_idx, species_idx) for species_idx in species_indices if species_idx < len(value)]
                    value_lookup = lambda local_idx, species_idx: value[species_idx]

                for local_idx, species_idx in species_pairs:
                    species_name = type_names[species_idx] if type_names and species_idx < len(type_names) else f"type_{species_idx}"
                    value_row = value_lookup(local_idx, species_idx)
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
