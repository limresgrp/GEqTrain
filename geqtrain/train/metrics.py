# geqtrain/train/metrics.py

import inspect
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from e3nn.o3 import Irreps

from geqtrain.utils.torch_runstats._runstats import RunningStats, Reduction
from ._key import ABBREV
from ._loss import LossWrapper, PreparedTarget, StatefulMetric, prepare_target, resolve_node_type_indices
from .loss import Loss

class _Metric:
    """Internal helper class to manage the state and logic for a single metric."""
    def __init__(
        self,
        func: callable,
        params: dict,
        *,
        aggregate_ensemble: bool = False,
        ensemble_aggregation: str = "mean",
    ):
        self.func = func
        self.params = params
        self.accumulator: Union[RunningStats, StatefulMetric, None] = None
        self.node_type_indices = self._resolve_node_type_indices()
        self.node_mask_field = self.params.pop("node_mask_field", self.params.pop("node_mask_key", None))
        self.node_level_filter = self.params.pop(
            "node_level_filter",
            getattr(self.func, "node_level_filter", "auto"),
        )
        self.aggregate_ensemble = aggregate_ensemble
        self.ensemble_aggregation = ensemble_aggregation

        # If the metric is stateful, it acts as its own accumulator
        if isinstance(self.func, StatefulMetric):
            self.accumulator = self.func

    def _resolve_node_type_indices(self):
        out = resolve_node_type_indices(self.params, "metric")
        self.params.pop("node_type_indices", None)
        self.params.pop("node_type_names", None)
        self.params.pop("type_names", None)
        return out

    def accumulate(
        self,
        pred: dict,
        ref: dict,
        key: str,
        normalization_fields: dict,
        feature_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Calculates and accumulates the metric for the current batch."""
        pred_key_name = key
        if isinstance(self.func, LossWrapper):
            pred_key_name = self.func._get_pred_key_name(key)

        if pred_key_name not in pred or key not in ref:
            return None

        pred_key = pred[pred_key_name]
        ref_key = ref[key]
        if isinstance(self.func, LossWrapper):
            self.func._initialize_supervision_weights(pred_key.device, pred_key.dtype)
            ref_key = self.func._handle_supervision_shapes(pred_key, ref_key, pred_key_name, key)
            ignore_nan = self.func.ignore_nan
        else:
            if ref_key.shape != pred_key.shape:
                try:
                    ref_key = ref_key.reshape(pred_key.shape)
                except Exception:
                    pass
            ignore_nan = bool(self.params.get("ignore_nan", False))

        prepared = prepare_target(
            pred=pred,
            ref=ref,
            key=key,
            pred_key_name=pred_key_name,
            pred_key=pred_key,
            ref_key=ref_key,
            node_type_indices=self.node_type_indices,
            node_mask_field=self.node_mask_field,
            node_level_filter=self.node_level_filter,
            ignore_nan=ignore_nan,
            denormalize=True,
            normalization_fields=normalization_fields,
            aggregate_ensemble=self.aggregate_ensemble,
            ensemble_aggregation=self.ensemble_aggregation,
        )

        if feature_indices is not None:
            if prepared.pred_key.ndim == 0:
                return None
            indices = torch.as_tensor(feature_indices, dtype=torch.long, device=prepared.pred_key.device)
            pred_key = prepared.pred_key.index_select(-1, indices)
            ref_key = prepared.ref_key.index_select(-1, indices)
            prepared_pred = dict(prepared.pred)
            prepared_ref = dict(prepared.ref)
            prepared_pred[pred_key_name] = pred_key
            prepared_ref[key] = ref_key
            prepared = PreparedTarget(
                pred=prepared_pred,
                ref=prepared_ref,
                pred_key=pred_key,
                ref_key=ref_key,
                node_types=prepared.node_types,
                mask=prepared.mask,
            )

        if isinstance(self.accumulator, StatefulMetric):
            self.accumulator.update(prepared.pred, prepared.ref, key)
            return self.accumulator.compute()  # Return partial result for batch logs
        
        # --- Logic for stateless (RunningStats) metrics ---
        if isinstance(self.func, LossWrapper):
            error = self.func._calculate_loss(prepared.pred_key, prepared.ref_key, mean=False)
        else:
            error = self.func(
                pred=prepared.pred,
                ref=prepared.ref,
                key=key,
                mean=False,
                normalization_fields={},
            )
        
        # If per-target metrics are not requested, average over the feature dimension
        if error.dim() > 1 and not self.params.get("PerTarget"):
            error = error.mean(dim=tuple(range(1, error.dim())))

        # If filtering removed all relevant samples from this chunk/batch, do not
        # update the running statistics. This happens with chunked validation when
        # a chunk contains no atoms matching the requested species/mask filter.
        if error.numel() == 0:
            return None
        
        # Lazily initialize the RunningStats accumulator on the first batch
        if self.accumulator is None:
            self._init_runstat(error)

        accum_params = self._prepare_accumulation_params(error, prepared)
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

    def _prepare_accumulation_params(self, error: torch.Tensor, prepared: PreparedTarget) -> dict:
        """Prepares the `accumulate_by` tensor for PerSpecies or PerTarget logic."""
        accum_params = {}
        if self.params.get("PerSpecies"):
            if prepared.node_types is not None:
                accum_params["accumulate_by"] = prepared.node_types.to(error.device)

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
        target_irreps: dict = None,
        dataset_mode: str = "single",
        ensemble_loss_on_aggregate: bool = True,
        ensemble_aggregation: str = "mean",
    ):
        super().__init__(
            components,
            dataset_mode=dataset_mode,
            ensemble_loss_on_aggregate=ensemble_loss_on_aggregate,
            ensemble_aggregation=ensemble_aggregation,
        )
        self.normalization_fields = {} if normalization_fields is None else normalization_fields
        self.target_irreps = self._resolve_target_irreps(target_irreps or {})
        self.enable_irrep_breakdown = False
        self.metrics: Dict[str, _Metric] = {}
        self.base_keys = list(self.keys)
        self.irrep_metric_specs: Dict[str, dict] = {}
        
        for key in self.base_keys:
            func = self.funcs[key]
            params: dict = dict(self.func_params.get(key, {}))
            if hasattr(func, "extra_params"):
                params.update(dict(func.extra_params))
            
            # Set defaults and process reduction parameter for stateless metrics
            if not isinstance(func, StatefulMetric):
                params.setdefault("PerSpecies", False)
                params.setdefault("PerTarget", False)
                reduction_str = params.get("reduction", "mean")
                reductions = {'mean': Reduction.MEAN, 'rms': Reduction.RMS}
                params['reduction'] = reductions.get(reduction_str, Reduction.MEAN)
            
            self.metrics[key] = _Metric(
                func,
                dict(params),
                aggregate_ensemble=self.aggregate_ensemble,
                ensemble_aggregation=self.ensemble_aggregation,
            )
            self._register_irrep_breakdown_metrics(key, func, params)

    def __call__(self, pred: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_metrics = {}
        for key in self.base_keys:
            metric_handler = self.metrics[key]
            clean_key = self.get_target_key(key)
            value = metric_handler.accumulate(
                pred,
                ref,
                clean_key,
                self.normalization_fields,
            )
            if value is not None:
                batch_metrics[key] = value

        if self.enable_irrep_breakdown:
            for key, spec in self.irrep_metric_specs.items():
                metric_handler = self.metrics[key]
                value = metric_handler.accumulate(
                    pred,
                    ref,
                    spec["target_key"],
                    self.normalization_fields,
                    feature_indices=spec["feature_indices"],
                )
                if value is not None:
                    batch_metrics[key] = value
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
            if value is None or (torch.is_tensor(value) and value.numel() == 0):
                continue
            handler = self.metrics[key]
            params = handler.params
            
            key_clean = self.remove_suffix(key) if key in self.irrep_metric_specs else self.get_target_key(key)
            metric_name = ABBREV.get(key_clean, key_clean)
            loss_name = str(handler.func)
            reduction_name = params.get('reduction', Reduction.MEAN).name.lower()
            metric_key = f"{metric_name}_{loss_name}_{reduction_name}"
            counts = None
            if isinstance(handler.accumulator, RunningStats) and hasattr(handler.accumulator, "_n"):
                counts = handler.accumulator._n

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
                    bin_idx = local_idx if len(value) == len(species_indices) else species_idx
                    if counts is not None and torch.is_tensor(counts) and bin_idx < counts.shape[0]:
                        if torch.as_tensor(counts[bin_idx]).sum().item() == 0:
                            continue
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

    def _resolve_target_irreps(self, target_irreps: dict) -> Dict[str, Irreps]:
        resolved = {}
        candidates = dict(target_irreps)
        for key, spec in self.normalization_fields.items():
            if key not in candidates and isinstance(spec, dict) and spec.get("irreps") is not None:
                candidates[key] = spec.get("irreps")

        for key, value in candidates.items():
            try:
                resolved[str(key)] = Irreps(value)
            except Exception:
                continue
        return resolved

    def _register_irrep_breakdown_metrics(self, key: str, func: callable, base_params: dict):
        if isinstance(func, StatefulMetric):
            return

        target_key = self.get_target_key(key)
        irreps = self.target_irreps.get(target_key)
        if irreps is None:
            return

        groups = self._irrep_feature_groups(irreps)
        if len(groups) <= 1:
            return

        suffix = key[len(self.remove_suffix(key)) :]
        for label, indices in groups.items():
            if len(indices) == irreps.dim:
                continue
            metric_key = f"{self.remove_suffix(key)}_{label}{suffix}"
            disambiguator = 1
            while metric_key in self.metrics:
                metric_key = f"{self.remove_suffix(key)}_{label}_{disambiguator}{suffix}"
                disambiguator += 1
            self.target_keys[metric_key] = target_key
            self.metrics[metric_key] = _Metric(
                func,
                dict(base_params),
                aggregate_ensemble=self.aggregate_ensemble,
                ensemble_aggregation=self.ensemble_aggregation,
            )
            self.irrep_metric_specs[metric_key] = {
                "target_key": target_key,
                "feature_indices": indices,
            }

    @staticmethod
    def _irrep_feature_groups(irreps: Irreps) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        for mul_ir, slc in zip(irreps, irreps.slices()):
            _, ir = mul_ir
            parity = "e" if ir.p == 1 else "o"
            label = f"l{ir.l}{parity}"
            groups.setdefault(label, []).extend(range(slc.start, slc.stop))
        return groups

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
