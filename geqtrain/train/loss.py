import importlib
import re
from typing import Union, List, Dict

import torch
from geqtrain.train._loss import StatefulMetric
from geqtrain.train.utils import parse_loss_metrics_dict
from ._key import ABBREV

from geqtrain.utils.torch_runstats._runstats import RunningStats, Reduction
from geqtrain.train import LossWrapper


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
        self.key_pattern = r"\_\d+"

        self._parse_components_from_yaml(components)
        
        # The Loss class owns its own statistics tracker.
        self.loss_stat = LossStat(self)

    def __call__(self, pred: dict, ref: dict, **kwargs):
        """Computes the total weighted loss and contributions from each component."""
        total_loss = 0.0
        contributions = {}
        for key in self.keys:
            clean_key = self.remove_suffix(key)
            try:
                loss_val = self.funcs[key](pred=pred, ref=ref, key=clean_key, mean=True, **kwargs)
                contributions[key] = loss_val
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

    def register_coeffs_and_loss(self, key: str, coeff: float, func: str, func_params: dict = {}):
        key = self.suffix_key(key)
        self.keys.append(key)
        self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.float32)

        instance = None
        # 1. Try to instantiate as a standard torch.nn loss via the wrapper
        try:
            # The LossWrapper's __init__ will fail if `func` is not in `torch.nn`
            instance = LossWrapper(func_name=func, params=func_params)
        except NameError:
            # 2. If it's not a torch.nn loss, treat it as a custom one
            try:
                from . import _loss
                # Try loading from our custom _loss.py module first
                loss_class = getattr(_loss, func)
            except NameError:
                # If not found locally, assume it's a full path to a user's class
                loss_class = _instantiate_from_path(func)
            
            # Instantiate the custom class. We assume a constructor that accepts params.
            instance = loss_class(**func_params)

        if instance is None:
            raise NotImplementedError(f"Could not instantiate loss/metric function '{func}'")

        self.funcs[key] = instance
        self.func_params[key] = func_params

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
        for v in self.loss_stat.values(): v.to(device=device)

    def current_result(self):
        results = {"loss_" + ABBREV.get(k, k): v.current_result().item() for k, v in self.loss_stat.items() if k != "total"}
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results