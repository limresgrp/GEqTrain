import re
from typing import Union, List, Dict

import torch.nn

from geqtrain.train.utils import parse_dict
from ._loss import instantiate_loss_function
from ._key import ABBREV

from torch_runstats import RunningStats, Reduction


class Loss:
    """
    assemble loss function based on key(s) and coefficient(s)

    Args:
        coeffs (dict, str): keys with coefficient and loss function name

    Example input dictionaries

    ```python
    'total_energy'
    ['total_energy', 'forces']
    {'total_energy': 1.0}
    {'total_energy': (1.0)}
    {'total_energy': (1.0, 'MSELoss'), 'forces': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, user_define_callables), 'force': (1.0, 'L1Loss', param_dict)}
    {'total_energy': (1.0, 'MSELoss'),
     'force': (1.0, 'Weighted_L1Loss', param_dict)}
    ```

    The loss function can be a loss class name that is exactly the same (case sensitive) to the ones defined in torch.nn.
    It can also be a user define class type that
        - takes "reduction=none" as init argument
        - uses prediction tensor and reference tensor for its call functions,
        - outputs a vector with the same shape as pred/ref
    """

    def __init__(
        self,
        components: Union[str, List[str], List[dict]],
    ):
        self.keys: List = [] # loss names
        self.coeffs: Dict  = {} # coefficients to weight losses
        self.funcs: Dict = {} # call key-associated (custom) callable defined in _loss.py. Classes in _loss.py acts as wrapper of torch.nn loss func (to provide further options)
        self.func_params = {}

        self._parse_components_from_yaml(components)

    def _parse_components_from_yaml(self, components):
        if isinstance(components, str):
            self.register_coeffs_and_loss(key=components, coeff=1.0, func="MSELoss", func_params={})
        elif isinstance(components, list):
            for elem in components:
                if isinstance(elem, str):
                    self.register_coeffs_and_loss(key=elem, coeff=1.0, func="MSELoss", func_params={})
                elif isinstance(elem, dict):
                    for key, coeff, func, func_params in parse_dict(elem):
                        self.register_coeffs_and_loss(key=key, coeff=coeff, func=func, func_params=func_params)
                else:
                    raise NotImplementedError(
                        f"loss_coeffs can only a list of str or dict. got {type(components)}"
                    )
        else:
            raise NotImplementedError(
                f"loss_coeffs can only be str, list[str] or list[dict]. got {type(components)}"
            )

    def __call__(self, pred: dict, ref: dict):
        '''
        returns:
        total loss for this batch
        hash map of non-weighted contributions to loss in this batch
        '''
        loss = 0.0
        contrib = {} # hash map of non-weighted contributions to loss in this batch
        for key in self.keys: # for k in "losses-keys" (i.e. the losses names as listed in yaml) that have to be evaluated
            _loss = self.funcs[key]( # call key-associated (custom) callable defined in _loss.py
                pred=pred,
                ref=ref,
                key=self.remove_suffix(key),
                mean=True,
            )
            contrib[key] = _loss
            loss += self.coeffs[key] * _loss # total_loss += weight_i * loss_i

        return loss, contrib

    def register_coeffs_and_loss(self, key: str, coeff: float, func: str, func_params: dict = {}):
        '''
        given the loss-func-name given in yaml, it registers the associated loss-coeff in self.coeffs[key] dict
        where key is the loss-func-name given in yaml
        it also stores the associated callable loss function in self.funcs[key]

        Args:
        func_params: Dict dictionary of kwarded args to be passed
        '''
        key = self.suffix_key(key)
        self.keys.append(key)
        self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.float32)
        self.funcs[key] = instantiate_loss_function(func, func_params)
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
        return re.sub('_suffix_\d+', '', key)

    def add_suffix(self, key: str, suffix_id: int):
        return f"{key}_suffix_{str(suffix_id)}"



class LossStat:
    """
    The class that accumulate the loss function values over all batches
    for each loss component.

    Args:

    loss_instance: the instance of Loss instancitaed by the trainer

    """

    def __init__(self, loss_instance=None):
        self.loss_stat = {
            "total": RunningStats(
                dim=tuple(), reduction=Reduction.MEAN, ignore_nan=False
            )
        }
        # if the wrapper of the torch.nn does not have the "ignore_nan" field set (i.e. not provided in yaml then it is False)
        self.ignore_nan = {}
        if loss_instance is not None:
            for key, func in loss_instance.funcs.items():
                self.ignore_nan[key] = (
                    func.ignore_nan if hasattr(func, "ignore_nan") else False
                )

    def __call__(self, loss, loss_contrib):
        """
        Args:

        loss (torch.Tensor): the value of the total loss function for the current batch
        loss (Dict(torch.Tensor)): the dictionary which contain the loss components
        """

        results = {}

        results["loss"] = self.loss_stat["total"].accumulate_batch(loss).item()

        # go through each component
        for k, v in loss_contrib.items():

            # initialize for the 1st batch
            if k not in self.loss_stat:
                self.loss_stat[k] = RunningStats(
                    dim=tuple(),
                    reduction=Reduction.MEAN,
                    ignore_nan=self.ignore_nan.get(k, False),
                )
                device = v.get_device()
                self.loss_stat[k].to(device="cpu" if device == -1 else device)

            results["loss_" + ABBREV.get(k, k)] = (
                self.loss_stat[k].accumulate_batch(v).item()
            )
        return results

    def reset(self):
        """
        Reset all the counters to zero
        """

        for v in self.loss_stat.values():
            v.reset()

    def to(self, device):
        for v in self.loss_stat.values():
            v.to(device=device)

    def current_result(self):
        results = {
            "loss_" + ABBREV.get(k, k): v.current_result().item()
            for k, v in self.loss_stat.items()
            if k != "total"
        }
        results["loss"] = self.loss_stat["total"].current_result().item()
        return results
