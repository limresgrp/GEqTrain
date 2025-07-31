""" Adapted from https://github.com/mir-group/nequip
"""

import inspect
import torch.nn

from typing import Dict
from importlib import import_module
from geqtrain.utils.torch_runstats._runstats import Reduction
from geqtrain.data import AtomicDataDict
from geqtrain.utils import instantiate_from_cls_name


class SimpleLoss:
    """
    wrapper to torch.nn loss function; computes weighted loss function
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

        self.extra_params = {}

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key = self.prepare(pred, ref, key, **kwargs)
        try:
            loss = self.func(pred_key, ref_key)
        except:
            ref_key = ref_key.squeeze().long()
            if pred_key.shape[0] == 1: # in case bs == 1
                loss = self.func(pred_key.squeeze(), ref_key)
            else:
                loss = self.func(pred_key, ref_key)  # if ref_key.dim() > 1 in CrossEntropyLoss

        return loss.mean() if mean else loss

    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key = pred.get(key, None) # None is allowed just to make sure that the assert below is raised properly if needed
        assert isinstance(pred_key, torch.Tensor), f"Expected prediction tensor for pred key {key}, found {type(pred_key)}"

        # the below creates a target tensor filled with reference_value if provided in params dict provided in __init__
        if hasattr(self, "reference_value"):
            try:
                fill_value = float(self.reference_value)
                ref_key = torch.full_like(pred_key, fill_value)
            except ValueError:
                raise ValueError(f"Invalid fill value for ref: {self.ref}. It must be convertible to a float.")
        else:
            ref_key = ref.get(key, None)
        assert isinstance(ref_key,  torch.Tensor), f"Expected prediction tensor for ref key {key}, found {type(ref_key)}"

        return pred_key, ref_key

    def __str__(self):
        return self.func_name

class SimpleLossWithNaNsFilter(SimpleLoss):
    """
    as above but remove nans from target/prediction
    """
    def __init__(self, func_name, params = {}):
        super().__init__(func_name, params)

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, not_nan_pred_filter, ref_key, not_nan_ref_filter = self.prepare(pred, ref, key, **kwargs)
        not_nan_filter = not_nan_pred_filter * not_nan_ref_filter

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
        pred_key, ref_key = super().prepare(pred, ref, key, **kwargs)
        not_nan_pred_filter = self._get_not_nan(pred_key, key)
        not_nan_ref_filter  = self._get_not_nan(ref_key, key)
        return torch.nan_to_num(pred_key, nan=0.), not_nan_pred_filter, torch.nan_to_num(ref_key, nan=0.), not_nan_ref_filter

    def _get_not_nan(self, ref_key: torch.Tensor, key: str):
        self._check_nan(ref_key, key)
        return (ref_key == ref_key).int()

    def _check_nan(self, ref_key: torch.Tensor, key: str):
        has_nan = torch.isnan(ref_key.sum())
        if has_nan and not (hasattr(self, "ignore_nan") and self.ignore_nan):
            raise Exception(f"Target field '{key}' has nan values."
                             "\nIf this is intended, set 'ignore_nan' to true in config file for this loss.")


class SimpleNodeLoss(SimpleLossWithNaNsFilter):
    """
    as above but removes all nodes that are not edge-centers (i.e. keeps only nodes for which we have a non-nan prediction)
    """
    def __init__(self, func_name, params = ...):
        super().__init__(func_name, params)

    def __call__(
        self,
        pred: dict,
        ref : dict,
        key : str ,
        mean: bool = True,
        **kwargs,
    ):
        pred_key, ref_key, not_nan_filter, center_nodes_filter = self.prepare(pred, ref, key, **kwargs)

        loss = self.func(pred_key, ref_key) * not_nan_filter
        if mean:
            return loss.sum() / not_nan_filter.sum() # + penalty
        loss[~not_nan_filter.bool()] = torch.nan
        return loss

    def prepare(
        self,
        pred: Dict,
        ref:  Dict,
        key:  str,
        **kwargs,
    ):
        pred_key, not_nan_pred_filter, ref_key, not_nan_ref_filter = super().prepare(pred, ref, key, **kwargs)
        center_nodes_filter = pred.get(AtomicDataDict.EDGE_INDEX_KEY)[0].unique()
        num_atoms = len(pred[AtomicDataDict.POSITIONS_KEY])
        if len(pred_key) == num_atoms:
            pred_key = pred_key[center_nodes_filter]
            not_nan_pred_filter = not_nan_pred_filter[center_nodes_filter]

        if len(ref_key) == num_atoms:
            ref_key  = ref_key [center_nodes_filter]
            not_nan_ref_filter = not_nan_ref_filter[center_nodes_filter]

        not_nan_filter = not_nan_pred_filter * not_nan_ref_filter

        return pred_key, ref_key, not_nan_filter, center_nodes_filter


class RMSDLoss(SimpleLossWithNaNsFilter):

    def __init__(self, func_name: str, params: dict = ...):
        super().__init__('MSELoss', params)
        self.reduction = Reduction.RMS

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        if mean:
            raise Exception("RMSDLoss should be used only as a Metric. Please use MSELoss instead.")
        
        pred_key, not_nan_pred_filter, ref_key, not_nan_ref_filter = self.prepare(pred, ref, key, **kwargs)
        not_nan_filter = not_nan_pred_filter * not_nan_ref_filter
        loss = torch.sum(self.func(pred_key, ref_key) * not_nan_filter, dim=-1)

        # The accumulate_batch() method used by metrics first squares the loss, then computes the average and then extracts the root.
        # Thus, we need to pass the sqrt(loss) to obtain the RMSD as output.
        loss[~not_nan_filter[:, 0].bool()] = torch.nan
        return torch.sqrt(loss)
    
    def __str__(self):
        return "RMSD"


class FocalLossBinaryAccuracy(SimpleLoss):
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        '''
        alpha is a number between 0 and 1
        If alpha is 0.25, the loss for positive examples (target is 1) is multiplied by 0.25,
        and the loss for negative examples (target is 0) is multiplied by 0.75 (since 1-0.25=0.75).
        Effect: less weight to positive class and more weight to negative class, useful when the positive class is over-represented.

        gamma: purpose: focus more on hard-to-classify examples by reducing the relative loss for well-classified examples.
        higher gamma higher focus to hard-to-classify examples i.e. examples on which the net is not so confident.
        scalses up the loss value when the net is not confident in the correct class
        '''

        super().__init__("BCEWithLogitsLoss", params)
        self.alpha: float = params.get('alpha', 0.85)
        self.gamma: float = params.get('gamma', 2)
        assert 0 < self.alpha < 1

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        logits, target = super().prepare(pred, ref, key, **kwargs)
        bce_loss = self.func(logits, target)
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean() if mean else loss
    
    def __str__(self):
        return "FocalLossBinaryAccuracy"


class BinaryAUROCMetric:
    def __init__(
        self,
        func_name: str,
        params: dict = {},
        **kwargs,
    ):
        from torcheval.metrics import BinaryAUROC
        self.metric = BinaryAUROC(**params)
        self.extra_params = {"reduction": "latest"}

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        **kwargs,
    ):
        if mean:
            raise(f"{__class__.__name__} cannot be used as loss function for training")

        logits = pred[key].squeeze(-1)
        target = ref[key].squeeze(-1)

        if target.dim() == 0: # if bs = 1
            target = target.unsqueeze(0)
            logits = logits.unsqueeze(0)

        self.metric.update(logits, target)
        rocauc = self.metric.compute()
        return rocauc.to(logits.device)
    
    def reset(self):
        self.metric.reset()
    
    def __str__(self):
        return "BinaryAUROC"


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
    raise NotImplementedError(f"{name} Loss is not implemented")