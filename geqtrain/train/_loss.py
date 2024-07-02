""" Adapted from https://github.com/mir-group/nequip
"""

import logging
import inspect
import torch.nn

from typing import Dict
from importlib import import_module
from torch_scatter import scatter, scatter_mean
from geqtrain.utils import instantiate_from_cls_name
from geqtrain.data import AtomicDataDict


class SimpleLoss:
    """wrapper to compute weighted loss function

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

        for key, value in params.items():
            setattr(self, key, value)

        func, _ = instantiate_from_cls_name(
            torch.nn,
            class_name=func_name,
            prefix="",
            positional_args=dict(reduction="none"),
            optional_args=params,
            all_args={},

        )

        self.func_name = func_name
        self.func = func

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str, # first row of each element listed under loss_coeffs:
        mean: bool = True,
    ):
        pred_key, ref_key, has_nan, not_zeroes = self.prepare(pred, ref, key)

        if has_nan:
            not_nan_zeroes = (ref_key == ref_key).int() * (pred_key == pred_key).int() * not_zeroes
            loss = self.func(torch.nan_to_num(pred_key, nan=0.), torch.nan_to_num(ref_key, nan=0.)) * not_nan_zeroes
            if mean:
                return loss.sum() / not_nan_zeroes.sum()
            else:
                loss[~not_nan_zeroes.bool()] = torch.nan
                return loss
        else:
            loss = self.func(pred_key, ref_key) * not_zeroes
            if mean:
                return loss.mean(dim=-1).sum() / not_zeroes.sum()
            else:
                return loss

    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
    ):
        ref_key = ref.get(key, None)
        assert isinstance(ref_key, torch.Tensor)
        pred_key = pred.get(key, None)
        assert isinstance(pred_key, torch.Tensor)
        pred_key = pred_key.view_as(ref_key)

        has_nan = torch.isnan(ref_key.sum()) or torch.isnan(pred_key.sum())
        if has_nan and not (hasattr(self, "ignore_nan") and self.ignore_nan):
            raise Exception(f"Either the predicted or true property '{key}' has nan values. "
                             "If this is intended, set 'ignore_nan' to true in config file for this loss.")

        if hasattr(self, "ignore_zeroes") and self.ignore_zeroes:
            not_zeroes = (~torch.all(ref_key == 0., dim=-1)).int() if len(ref_key.shape) > 1 else (ref_key != 0)
        else:
            not_zeroes = torch.ones(*ref_key.shape[:max(1, len(ref_key.shape)-1)], device=ref_key.device).int()
        not_zeroes = not_zeroes.reshape(*([-1] + [1] * (len(pred_key.shape)-1)))
        return pred_key, ref_key, has_nan, not_zeroes

class PerLabelLoss(SimpleLoss):

    def __init__(self, func_name: str, params: dict = ...):
        self.instanciated = True
        super().__init__(func_name, params)


    def __call__(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str, # first row of each element listed under loss_coeffs:
        mean: bool = True,
    ):
        pred_key, ref_key, has_nan, not_zeroes = self.prepare(pred, ref, key)
        loss = self.func(pred_key, ref_key)

        if mean:
            return loss.mean()
        return loss.mean(dim=0)




class PerSpeciesLoss(SimpleLoss):
    """Compute loss for each species and average among the same species
    before summing them up.

    Args same as SimpleLoss
    """

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
    ):
        if not mean:
            raise NotImplementedError("Cannot handle this yet")

        ref_key = ref.get(key, torch.zeros_like(pred[key], device=pred[key].device)) # get tensor assiciated to the key in predictions else return zeros
        has_nan = self.ignore_nan and torch.isnan(ref_key.sum()) # bool that defines wheter predictions have nans

        if has_nan:
            not_nan = (ref_key == ref_key).int()
            per_node_loss = (
                self.func(pred[key], torch.nan_to_num(ref_key, nan=0.0)) * not_nan
            )
        else:
            per_node_loss = self.func(pred[key], ref_key) # call to loss func

        reduce_dims = tuple(i + 1 for i in range(len(per_node_loss.shape) - 1))

        spe_idx = pred[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
        if has_nan:
            if len(reduce_dims) > 0:
                per_node_loss = per_node_loss.sum(dim=reduce_dims)
            assert per_node_loss.ndim == 1

            per_species_loss = scatter(per_node_loss, spe_idx, dim=0)

            assert per_species_loss.ndim == 1  # [type]

            N = scatter(not_nan, spe_idx, dim=0)
            N = N.sum(reduce_dims)
            N = N.reciprocal()
            N_species = ((N == N).int()).sum()
            assert N.ndim == 1  # [type]

            per_species_loss = (per_species_loss * N).sum() / N_species

            return per_species_loss

        else:

            if len(reduce_dims) > 0:
                per_node_loss = per_node_loss.mean(dim=reduce_dims)
            assert per_node_loss.ndim == 1

            # offset species index by 1 to use 0 for nan
            _, inverse_species_index = torch.unique(spe_idx, return_inverse=True)

            per_species_loss = scatter_mean(per_node_loss, inverse_species_index, dim=0)
            assert per_species_loss.ndim == 1  # [type]

            return per_species_loss.mean()


def find_loss_function(name: str, params):
    """
    Search for loss functions in this module     instanciates the loss obj

    If the name starts with PerSpecies, rMSELosseturn the PerSpeciesLoss instance
    """

    wrapper_list = dict(
        perspecies=PerSpeciesLoss,
    )

    if isinstance(name, str): #  name is funct
        for key in wrapper_list:
            if name.lower().startswith(key):
                logging.debug(f"create loss instance {wrapper_list[key]}")
                return wrapper_list[key](name[len(key) :], params)
        try:
            module_name = ".".join(name.split(".")[:-1])
            class_name  = ".".join(name.split(".")[-1:])
            functional = params.get("functional", "MSELoss")
            return getattr(import_module(module_name), class_name)(functional, params) # func_name, params of SimpleLoss ctor
        except Exception:
            return SimpleLoss(name, params)
    elif inspect.isclass(name):
        return SimpleLoss(name, params)
    elif callable(name):
        return name
    else:
        raise NotImplementedError(f"{name} Loss is not implemented")