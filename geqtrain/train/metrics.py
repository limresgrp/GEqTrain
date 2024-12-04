""" Adapted from https://github.com/mir-group/nequip
"""

from copy import deepcopy
from hashlib import sha1
from typing import Dict, List, Union

import yaml

import torch

from geqtrain.data import AtomicDataDict
from geqtrain.train.loss import Loss
from geqtrain.train.utils import parse_dict
from torch_runstats import RunningStats, Reduction

from ._loss import instantiate_loss_function
from ._key import ABBREV


class Metrics(Loss):
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
        self,
        components: Union[str, List[str], List[dict]],
    ):

        super(Metrics, self).__init__(components)

        self.running_stats: Dict[str, RunningStats] = {}
        self.params = {}
        self.kwargs = {}
        for key in self.keys:
            func_params: Dict = self.func_params.get(key, {})
            func_params["PerSpecies"] = func_params.get("PerSpecies", False)
            func_params["PerLabel"]   = func_params.get("PerLabel", False)
            func_params["PerNode"]    = func_params.get("PerNode", False)
            func_params["functional"] = func_params.get("functional", "L1Loss")
            func_params["reduction"]    = func_params.get("reduction", "mean")

            # default is to flatten the array
            if key not in self.running_stats:
                self.kwargs[key] = {}
                self.params[key] = {}

            reductions = {
                'mean': Reduction.MEAN,
                'rms': Reduction.RMS,
            }
            reduction = reductions[func_params.get('reduction')]
            self.kwargs[key] = dict(reduction=reduction)

            # store for initialization
            kwargs = deepcopy(func_params)
            kwargs.pop("PerSpecies")
            kwargs.pop("PerLabel")
            kwargs.pop("PerNode")
            kwargs.pop("functional")
            kwargs.pop("reduction")
            
            self.kwargs[key].update(kwargs)
            self.params[key] = (reduction.value, func_params)


    def init_runstat(self, params: Dict, error: torch.Tensor) -> RunningStats:
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
    
    @property
    def clean_keys(self):
        for key in self.keys:
            yield self.remove_suffix(key)

    def __call__(
        self,
        pred: Dict[str, torch.Tensor],
        ref:  Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        metrics = {}
        N = None
        for key in self.keys:
            clean_key = self.remove_suffix(key)
            error: torch.Tensor = self.funcs[key]( # call key-associated (custom) callable defined in _loss.py
                pred=pred,
                ref=ref,
                key=clean_key,
                mean=False,
            )

            _, params = self.params[key]
            per_species = params["PerSpecies"]
            per_node    = params["PerNode"]
            per_label   = params["PerLabel"]

            # initialize the internal run_stat base on the error shape
            if key not in self.running_stats:
                self.running_stats[key] = self.init_runstat(
                    params=self.kwargs[key], error=error
                )

            stat: RunningStats = self.running_stats[key]

            params = {}
            if per_species:
                num_pred_nodes = len(error)
                node_idcs = ref[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
                num_nudes = len(node_idcs)
                if num_pred_nodes == num_nudes: # All nodes have a prediction
                    params = {
                        "accumulate_by": node_idcs
                    }
                else: # Only nodes for which target is not NaN were predicted
                    params = {
                        "accumulate_by": node_idcs[torch.unique(ref[AtomicDataDict.EDGE_INDEX_KEY][0])]
                    }

            elif per_label:
                params = {
                    "accumulate_by": torch.cat(len(ref[clean_key]) * [
                        torch.arange(ref[clean_key].shape[-1], device=error.device)
                    ])
                }
            if per_node:
                if N is None:
                    N = torch.bincount(ref[AtomicDataDict.BATCH_KEY]).unsqueeze(-1)
                error_N = error / N

            else:
                error_N = error

            if stat.dim == () and not per_species:
                metrics[key] = stat.accumulate_batch(
                    error_N.flatten(), **params
                )
            else:
                metrics[key] = stat.accumulate_batch(
                    error_N, **params
                )

        return metrics

    def reset(self):
        for stat in self.running_stats.values():
            stat.reset()

    def to(self, device):
        for stat in self.running_stats.values():
            stat.to(device=device)

    def current_result(self):

        metrics = {}
        for key, stat in self.running_stats.items():
            metrics[key] = stat.current_result()
        return metrics

    def flatten_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        metrics_metadata: Dict[str, List[str]]=None,
    ):

        '''
        Flatten the metrics dictionary into a single dictionary
        This is used to convert the metrics dictionary into a format that is easy to understand and analyze
        It also allows for easy plotting of the metrics

        The flatten_metrics function is designed to convert a nested dictionary of metrics into a flat dictionary format, making it easier to access and use the metrics. Here's a breakdown of its components:
        Parameters:
        metrics: A dictionary containing computed metrics, where keys are tuples of (key, param_hash) and values are the corresponding metric values.
        metrics_metadata: An optional dictionary that can contain additional information, such as type_names and target_names, which are used to label the metrics.

        Returns:
        flat_dict: A flat dictionary containing the metrics, with keys and values flattened for easy access.
        skip_keys: A list of keys that were skipped during the flattening process.
        '''

        type_names = metrics_metadata.get('type_names', [])
        target_names = metrics_metadata.get('target_names', [])

        flat_dict = {}
        skip_keys = []
        for key, value in metrics.items():
            reduction, params = self.params[key]

            key_clean = self.remove_suffix(key)
            short_name = ABBREV.get(key_clean, key_clean)
            if hasattr(self.funcs[key], "get_name"):
                short_name = self.funcs[key].get_name(short_name)

            per_node = params["PerNode"]
            suffix = "/N" if per_node else ""
            loss_name = self.funcs[key].func_name
            item_name = f"{short_name}{suffix}"

            stat = self.running_stats[key]
            per_species = params["PerSpecies"]
            per_label = params["PerLabel"]

            if per_species:
                if stat.output_dim == tuple():
                    if type_names is None:
                        type_names = [i for i in range(len(value))]
                    for id_ele, v in enumerate(value):
                        if type_names is not None:
                            flat_dict[f"{type_names[id_ele]}_{item_name}.{reduction}"] = v.item()
                        else:
                            flat_dict[f"{id_ele}_{item_name}.{reduction}"] = v.item()

                    flat_dict[f"psavg_{item_name}"] = value.mean().item()
                else:
                    for id_ele, vec in enumerate(value):
                        ele = type_names[id_ele]
                        for idx, v in enumerate(vec):
                            name = f"{ele}_{item_name}_{idx}"
                            flat_dict[name] = v.item()
                            skip_keys.append(name)
            elif per_label:
                if stat.output_dim == tuple():
                    if not target_names:
                        target_names = [i for i in range(len(value))]
                    for id_ele, v in enumerate(value):
                        if target_names is not None:
                            flat_dict[f"{target_names[id_ele]}.{loss_name}.{reduction}"] = v.item()
                        else:
                            flat_dict[f"{item_name}.{loss_name}.{reduction}"] = v.item()
                else:
                    for id_ele, vec in enumerate(value):
                        ele = type_names[id_ele]
                        for idx, v in enumerate(vec):
                            name = f"{ele}.{item_name}.{idx}"
                            flat_dict[name] = v.item()
                            skip_keys.append(name)
            else:
                if stat.output_dim == tuple():
                    # a scalar
                    flat_dict[item_name] = value.item()
                else:
                    # a vector
                    for idx, v in enumerate(value.flatten()):
                        flat_dict[f"{item_name}.{idx}"] = v.item()
        return flat_dict, skip_keys
