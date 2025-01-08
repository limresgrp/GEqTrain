""" Adapted from https://github.com/mir-group/nequip
"""

import inspect
import torch.nn

from typing import Dict
from importlib import import_module
from geqtrain.utils import instantiate_from_cls_name


class SimpleLoss:
    """
    wrapper to torch.nn loss function; computes weighted loss function

    if "ignore_nan" not provided in yaml then NaNs will propagate as normal.

    Args:

    func_name (str): any loss function defined in torch.nn that
        takes "reduction=none" as init argument, uses prediction tensor,
        and reference tensor for its call functions, and outputs a vector
        with the same shape as pred/ref
    params (str): arguments needed to initialize the function above

    Return:

    if mean is True, return a scalar; else return the error matrix of each entry
    """

    def __init__(
        self,
        func_name: str,
        params: dict = {},
    ):

        self.func_name = func_name
        for key, value in params.items():
            setattr(self, key, value)

        # instanciates torch.nn loss func
        self.func, _ = instantiate_from_cls_name(
            torch.nn,
            class_name=func_name,
            prefix="",
            positional_args=dict(reduction="none"),
            optional_args=params,
            all_args={},
        )

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter = self.prepare(pred, ref, key, **kwargs)

        loss = self.func(pred_key, ref_key) * not_nan_filter
        if mean:
            return loss.sum() / not_nan_filter.sum()
        loss[~not_nan_filter.bool()] = torch.nan
        return loss
    
    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        ref_key = ref.get(key, None)
        assert isinstance(ref_key, torch.Tensor)
        not_nan_filter = self._get_not_nan(ref_key, key)

        pred_key = pred.get(key, None)
        assert isinstance(pred_key, torch.Tensor)
        pred_key = pred_key.view_as(ref_key)

        return torch.nan_to_num(pred_key, nan=0.), torch.nan_to_num(ref_key, nan=0.), not_nan_filter
    
    def _get_not_nan(self, ref_key: torch.Tensor, key: str):
        self._check_nan(ref_key, key)
        return (ref_key == ref_key).int()
    
    def _check_nan(self, ref_key: torch.Tensor, key: str):
        has_nan = torch.isnan(ref_key.sum())
        if has_nan and not (hasattr(self, "ignore_nan") and self.ignore_nan):
            raise Exception(f"Target field '{key}' has nan values."
                             "\nIf this is intended, set 'ignore_nan' to true in config file for this loss.")


def instantiate_loss_function(name: str, params: Dict):
    """
    Search for loss functions in this module
    instanciates the loss obj
    If the name starts with PerSpecies, MSELoss return the PerSpeciesLoss instance
    """

    if isinstance(name, str): # name is function
        try:
            module_name = ".".join(name.split(".")[:-1])
            class_name  = ".".join(name.split(".")[-1:])
            functional = params.get("functional", "L1Loss")
            return getattr(import_module(module_name), class_name)(functional, params) # func_name, params of SimpleLoss ctor
        except Exception:
            return SimpleLoss(name, params)
    elif inspect.isclass(name):
        return SimpleLoss(name, params)
    elif callable(name):
        return name
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")