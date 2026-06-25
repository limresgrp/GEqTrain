import importlib
import inspect
import re
from typing import Union, List, Dict

import torch
from geqtrain.train._loss import LossWrapper, StatefulMetric, graph_zero_like, prepare_target, resolve_node_type_indices
from geqtrain.train.utils import parse_loss_metrics_dict
from ._key import ABBREV

from geqtrain.utils.torch_runstats._runstats import RunningStats, Reduction


def _instantiate_from_path(path: str):
    """Dynamically imports and returns a class from a full path string."""
    try:
        module_path, class_name = path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import class '{path}'. Reason: {e}")


class Loss:
    """
    A self-contained class that computes the training loss and tracks its statistics.
    """
    def __init__(self, components: Union[str, List[str], List[dict]]):
        self.keys: List[str] = []
        self.coeffs: Dict[str, torch.Tensor] = {}
        self.funcs: Dict[str, torch.nn.Module] = {}
        self.func_params: Dict[str, dict] = {}
        self.target_filters: Dict[str, dict] = {}
        self.target_keys: Dict[str, str] = {}
        self.key_pattern = r"\_\d+"

        self._parse_components_from_yaml(components)
        
        # The Loss class owns its own statistics tracker.
        self.loss_stat = LossStat(self)

    def __call__(self, pred: dict, ref: dict, normalization_fields: dict = None, **kwargs):
        """Computes the total weighted loss and contributions from each component."""
        if normalization_fields is None:
            normalization_fields = {}
        total_loss = 0.0
        contributions = {}
        for key in self.keys:
            clean_key = self.get_target_key(key)
            try:
                func = self.funcs[key]
                prepared_pred, prepared_ref = pred, ref
                prepared = None
                pred_key_name = clean_key
                if isinstance(func, LossWrapper):
                    pred_key_name = func._get_pred_key_name(clean_key)
                if pred_key_name in pred and clean_key in ref:
                    pred_key = pred[pred_key_name]
                    ref_key = ref[clean_key]
                    if isinstance(func, LossWrapper):
                        func._initialize_supervision_weights(pred_key.device, pred_key.dtype)
                        ref_key = func._handle_supervision_shapes(pred_key, ref_key, pred_key_name, clean_key)
                    target_filter = self.target_filters.get(key, {})
                    prepared = prepare_target(
                        pred=pred,
                        ref=ref,
                        key=clean_key,
                        pred_key_name=pred_key_name,
                        pred_key=pred_key,
                        ref_key=ref_key,
                        node_type_indices=target_filter.get("node_type_indices"),
                        node_mask_field=target_filter.get("node_mask_field"),
                        node_level_filter=target_filter.get("node_level_filter", "auto"),
                        ignore_nan=target_filter.get("ignore_nan", False),
                        denormalize=False,
                    )
                    prepared_pred, prepared_ref = prepared.pred, prepared.ref

                if prepared is not None and prepared.pred_key.numel() == 0:
                    loss_val = graph_zero_like(prepared.pred_key)
                else:
                    call_kwargs = dict(
                        pred=prepared_pred,
                        ref=prepared_ref,
                        key=clean_key,
                        mean=True,
                        normalization_fields=normalization_fields,
                        **kwargs,
                    )
                    if isinstance(func, LossWrapper):
                        call_kwargs["skip_target_filter"] = True
                    loss_val = func(**call_kwargs)
                contributions[key] = loss_val.detach()
                total_loss += self.coeffs[key].to(loss_val.device) * loss_val
            except Exception as e:
                raise RuntimeError(f"Error computing loss for key '{clean_key}': {e}") from e

        return total_loss, contributions

    def _parse_components_from_yaml(self, components):
        if components is None:
            return
        if isinstance(components, str):
            self.register_coeffs_and_loss(key=components, coeff=1.0, func="MSELoss", func_params={})
        elif isinstance(components, list):
            for elem in components:
                if isinstance(elem, str):
                    self.register_coeffs_and_loss(key=elem, coeff=1.0, func="MSELoss", func_params={})
                elif isinstance(elem, dict):
                    for key, coeff, func, func_params in parse_loss_metrics_dict(elem):
                        self.register_coeffs_and_loss(key=key, coeff=coeff, func=func, func_params=func_params)
                else:
                    raise NotImplementedError(f"loss_coeffs can only a list of str or dict. got {type(components)}")
        else:
            raise NotImplementedError(f"loss_coeffs can only be str, list[str] or list[dict]. got {type(components)}")

    def register_coeffs_and_loss(self, key: str, coeff: float, func: str, func_params: dict = None):
        target_key = key
        func_params = {} if func_params is None else dict(func_params)
        display_name = func_params.pop("name", func_params.pop("loss_name", None))
        key = self.suffix_key(str(display_name) if display_name is not None else target_key)
        self.keys.append(key)
        self.target_keys[key] = target_key
        self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.float32)

        instance = None
        # 1. Check for a standard torch.nn loss without relying on exceptions.
        torch_cls = getattr(torch.nn, func, None) if isinstance(func, str) else None
        is_torch_loss = inspect.isclass(torch_cls) and issubclass(torch_cls, torch.nn.modules.loss._Loss)
        if is_torch_loss:
            instance = LossWrapper(func_name=func, params=dict(func_params))
        else:
            # 2. If it's not a torch.nn loss, treat it as a custom one
            if callable(func) and not isinstance(func, str):
                loss_class = func
            else:
                try:
                    from . import _loss
                    # Try loading from our custom _loss.py module first
                    loss_class = getattr(_loss, func)
                except Exception:
                    # If not found locally, assume it's a full path to a user's class
                    loss_class = _instantiate_from_path(func)
            # Instantiate the custom class. We assume a constructor that accepts params.
            instance = loss_class(**func_params)

        if instance is None:
            raise NotImplementedError(f"Could not instantiate loss/metric function '{func}'")

        self.funcs[key] = instance
        self.func_params[key] = func_params
        self.target_filters[key] = self._extract_target_filter(func_params)

    def _extract_target_filter(self, func_params: dict) -> dict:
        params = {} if func_params is None else dict(func_params)
        return {
            "node_type_indices": resolve_node_type_indices(params, "loss"),
            "node_mask_field": params.get("node_mask_field", params.get("node_mask_key", None)),
            "node_level_filter": params.get("node_level_filter", "auto"),
            "ignore_nan": bool(params.get("ignore_nan", False)),
        }

    def suffix_key(self, key):
        suffix_id = 0
        key = self.add_suffix(key, suffix_id)
        while key in self.keys:
            key = self.remove_suffix(key)
            key = self.add_suffix(key, suffix_id)
            suffix_id += 1
        return key

    def remove_suffix(self, key):
        return re.sub(self.key_pattern, '', key)

    def get_target_key(self, key):
        return self.target_keys.get(key, self.remove_suffix(key))

    def add_suffix(self, key: str, suffix_id: int):
        if re.search(self.key_pattern, key):
            raise AssertionError(f"Loss name must not contain '_[$int]' in name: {key}")
        return f"{key}_{str(suffix_id)}"

    # --- Methods delegated to the internal LossStat ---
    def reset(self):
        """Resets all stateful loss functions and the statistics tracker."""
        self.loss_stat.reset()
        for key in self.keys:
            if isinstance(self.funcs[key], StatefulMetric):
                self.funcs[key].reset()

    def to(self, device):
        """Moves the statistics tracker to the specified device."""
        self.loss_stat.to(device)

    def current_result(self) -> Dict[str, float]:
        """Gets the current accumulated results for the epoch."""
        return self.loss_stat.current_result()

class LossStat:
    """Accumulates loss values. Used internally by the Loss class."""
    def __init__(self, loss_instance: Loss):
        self.loss_stat = {"total": RunningStats(reduction=Reduction.MEAN, dim=tuple())}
        self.ignore_nan = {key: getattr(func, "ignore_nan", False) for key, func in loss_instance.funcs.items()}

    def __call__(self, loss: torch.Tensor, loss_contrib: Dict[str, torch.Tensor]):
        """Update stats and return per-batch values."""
        results = {"loss": self.loss_stat["total"].accumulate_batch(loss).item()}
        for k, v in loss_contrib.items():
            if k not in self.loss_stat:
                device = v.device
                self.loss_stat[k] = RunningStats(
                    dim=tuple(), reduction=Reduction.MEAN, ignore_nan=self.ignore_nan.get(k, False)
                ).to("cpu" if device == -1 else device)
            
            results["loss_" + ABBREV.get(k, k)] = self.loss_stat[k].accumulate_batch(v).item()
        return results

    def reset(self):
        for v in self.loss_stat.values(): v.reset()

    def to(self, device):
        for v in self.loss_stat.values():
            v.to(device=device)
        return self

    def current_result(self):
        results = {"loss_" + ABBREV.get(k, k): v.current_result().item() for k, v in self.loss_stat.items() if k != "total"}
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results
