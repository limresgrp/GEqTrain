import re
from typing import Union, List

import torch.nn

from geqtrain.train.utils import parse_dict
from ._loss import find_loss_function
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
        coeffs: Union[dict, str, List[str]],
        coeff_schedule: str = "constant",
    ):

        self.coeff_schedule = coeff_schedule
        self.coeffs = {}
        self.funcs = {}
        self.keys = []

        if isinstance(coeffs, str):
            self.register_coeffs(key=coeffs, coeff=1.0, func="MSELoss", func_params={})
        elif isinstance(coeffs, list):
            for elem in coeffs:
                if isinstance(elem, str):
                    self.register_coeffs(key=elem, coeff=1.0, func="MSELoss", func_params={})
                elif isinstance(elem, dict):
                    for key, coeff, func, func_params in parse_dict(elem):
                        self.register_coeffs(key=key, coeff=coeff, func=func, func_params=func_params)
                else:
                    raise NotImplementedError(
                        f"loss_coeffs can only a list of str or dict. got {type(coeffs)}"
                    )
        elif isinstance(coeffs, dict):
            for key, coeff, func, func_params in parse_dict(coeffs):
                parse_dict(coeffs)
        else:
            raise NotImplementedError(
                f"loss_coeffs can only be str, list and dict. got {type(coeffs)}"
            )

        for key, coeff in self.coeffs.items():
            self.coeffs[key] = torch.as_tensor(coeff, dtype=torch.get_default_dtype())
            self.keys += [key]

    def __call__(self, pred: dict, ref: dict):

        loss = 0.0
        contrib = {}
        for key in self.keys: # for k in losses keys that have to be evaluated
            _loss = self.funcs[key]( # call its associated func
                pred=pred,
                ref=ref,
                key= self.remove_suffix(key),
                mean=True,
            )
            contrib[key] = _loss
            loss = loss + self.coeffs[key] * _loss # total_loss += weight_i * loss_i

        return loss, contrib

    def register_coeffs(self, key: str, coeff: float, func: str, func_params: dict = {}):
        key = self.suffix_key(key)
        self.coeffs[key] = coeff
        self.funcs[key] = find_loss_function(func, func_params)

    def suffix_key(self, key):
        suffix_id = 0
        key = self.add_suffix(key, suffix_id)
        while key in self.coeffs.keys():
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

    keys (null): redundant argument

    """

    def __init__(self, loss_instance=None):
        self.loss_stat = {
            "total": RunningStats(
                dim=tuple(), reduction=Reduction.MEAN, ignore_nan=False
            )
        }
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
