""" Adapted from https://github.com/mir-group/nequip
"""

from copy import deepcopy
from hashlib import sha1
from typing import Union, Sequence, Tuple

import yaml

import torch

from geqtrain.data import AtomicDataDict
from geqtrain.train.utils import parse_dict
from torch_runstats import RunningStats, Reduction

from ._loss import find_loss_function
from ._key import ABBREV


class Metrics:
    """Computes all mae, rmse needed for report

    Args:
        components: a list or a tuples of definition.

    Example:

    ```
    components = [(key1, "rmse"), (key2, "mse")]
    ```

    You can also offer more details with a dictionary. The keys can be any keys for RunningStats or

    report_per_component (bool): if True, report the mean on each component (equivalent to mean(axis=0) in numpy),
                                 otherwise, take the mean across batch and all components for vector data.
    functional: the function to compute the error. It has to be exactly the same as the one defined in torch.nn.
                Callables are also allowed.
                default: "L1Loss"
    PerSpecies: whether to compute the estimation for each species or not

    the keys are case-sensitive.


    ```
    components = (
        (
            AtomicDataDict.FORCE_KEY,
            "rmse",
            {"PerSpecies": True, "functional": "L1Loss", "dim": 3},
        ),
        (AtomicDataDict.FORCE_KEY, "mae", {"dim": 3}),
    )
    ```

    """

    def __init__(
        self, components: Sequence[Union[Tuple[str, str], Tuple[str, str, dict]]]
    ):

        self.running_stats = {}
        self.params = {}
        self.funcs = {}
        self.kwargs = {}
        for component in components:
            if isinstance(component, dict):
                for key, _, func, func_params in parse_dict(component):

                    func_params["PerSpecies"] = func_params.get("PerSpecies", False)
                    func_params["PerNode"] = func_params.get("PerNode", False)
                    func_params["functional"] = func_params.get("functional", "L1Loss")

                    param_hash = Metrics.hash_component(component)

                    # default is to flatten the array
                    if key not in self.running_stats:
                        self.running_stats[key] = {}
                        self.funcs[key] = {}
                        self.kwargs[key] = {}
                        self.params[key] = {}

                    # store for initialization
                    kwargs = deepcopy(func_params)
                    kwargs.pop("PerSpecies")
                    kwargs.pop("PerNode")
                    kwargs.pop("functional")

                    # by default, report a scalar that is mae and rmse over all component
                    loss_func = find_loss_function(func_params["functional"], func_params)
                    self.funcs[key][param_hash] = loss_func
                    reduction = getattr(loss_func, "reduction", Reduction.MEAN)
                    self.kwargs[key][param_hash] = dict(reduction=reduction)
                    self.kwargs[key][param_hash].update(kwargs)
                    self.params[key][param_hash] = (reduction, func_params)
                    

    def init_runstat(self, params, error: torch.Tensor):
        """
        Initialize Runstat Counter based on the shape of the error matrix

        Args:
        params (dict): dictionary of additional arguments
        error (torch.Tensor): error matrix
        """

        kwargs = deepcopy(params)
        # automatically define the dimensionality
        if "dim" not in kwargs:
            kwargs["dim"] = error.shape[1:]

        if "reduce_dims" not in kwargs:
            if not kwargs.pop("report_per_component", False):
                kwargs["reduce_dims"] = tuple(range(len(error.shape) - 1))

        rs = RunningStats(**kwargs)
        rs.to(device=error.device)
        return rs

    @staticmethod
    def hash_component(component):
        buffer = yaml.dump(component).encode("ascii")
        return sha1(buffer).hexdigest()

    def __call__(self, pred: dict, ref: dict):

        metrics = {}
        N = None
        for key in self.funcs.keys():
            for param_hash, kwargs in self.kwargs[key].items():
                func = self.funcs[key][param_hash]
                error = func(
                    pred=pred,
                    ref=ref,
                    key=key,
                    mean=False,
                )

                _, params = self.params[key][param_hash]
                per_species = params["PerSpecies"]
                per_node = params["PerNode"]

                # initialize the internal run_stat base on the error shape
                if param_hash not in self.running_stats[key]:
                    self.running_stats[key][param_hash] = self.init_runstat(
                        params=kwargs, error=error
                    )

                stat = self.running_stats[key][param_hash]

                params = {}
                if per_species:
                    params = {
                        "accumulate_by": pred[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
                    }
                if per_node:
                    if N is None:
                        N = torch.bincount(ref[AtomicDataDict.BATCH_KEY]).unsqueeze(-1)
                    error_N = error / N
                else:
                    error_N = error

                error_N[error_N == 0.] = torch.nan # Turn 0. errors to nan
                if stat.dim == () and not per_species:
                    metrics[(key, param_hash)] = stat.accumulate_batch(
                        error_N.flatten(), **params
                    )
                else:
                    metrics[(key, param_hash)] = stat.accumulate_batch(
                        error_N, **params
                    )

        return metrics

    def reset(self):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.reset()

    def to(self, device):
        for stats in self.running_stats.values():
            for stat in stats.values():
                stat.to(device=device)

    def current_result(self):

        metrics = {}
        for key, stats in self.running_stats.items():
            for reduction, stat in stats.items():
                metrics[(key, reduction)] = stat.current_result()
        return metrics

    def flatten_metrics(self, metrics, type_names=None):

        flat_dict = {}
        skip_keys = []
        for k, value in metrics.items():

            key, param_hash = k
            reduction, params = self.params[key][param_hash]

            short_name = ABBREV.get(key, key)
            if hasattr(self.funcs[key][param_hash], "get_name"):
                short_name = self.funcs[key][param_hash].get_name(short_name)

            per_node = params["PerNode"]
            suffix = "/N" if per_node else ""
            item_name = f"{short_name}{suffix}_{reduction}"

            stat = self.running_stats[key][param_hash]
            per_species = params["PerSpecies"]

            if per_species:
                if stat.output_dim == tuple():
                    if type_names is None:
                        type_names = [i for i in range(len(value))]
                    for id_ele, v in enumerate(value):
                        if type_names is not None:
                            flat_dict[f"{type_names[id_ele]}_{item_name}"] = v.item()
                        else:
                            flat_dict[f"{id_ele}_{item_name}"] = v.item()

                    flat_dict[f"psavg_{item_name}"] = value.mean().item()
                else:
                    for id_ele, vec in enumerate(value):
                        ele = type_names[id_ele]
                        for idx, v in enumerate(vec):
                            name = f"{ele}_{item_name}_{idx}"
                            flat_dict[name] = v.item()
                            skip_keys.append(name)

            else:
                if stat.output_dim == tuple():
                    # a scalar
                    flat_dict[item_name] = value.item()
                else:
                    # a vector
                    for idx, v in enumerate(value.flatten()):
                        flat_dict[f"{item_name}_{idx}"] = v.item()
        return flat_dict, skip_keys
