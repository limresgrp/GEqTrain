""" Adapted from https://github.com/mir-group/nequip
"""

from copy import deepcopy
from hashlib import sha1
import inspect
from typing import Dict, List, Union

import yaml

import torch

from geqtrain.data import AtomicDataDict
from geqtrain.train.loss import Loss
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
            func_params["PerTarget"]   = func_params.get("PerTarget", False)
            func_params["functional"] = func_params.get("functional", "L1Loss")
            func_params["reduction"]    = func_params.get("reduction", "mean")

            # default is to flatten the array
            if key not in self.running_stats:
                self.kwargs[key] = {}
                self.params[key] = {}

            reductions = {
                'mean': Reduction.MEAN,
                'rms' : Reduction.RMS,
            }
            reduction = reductions[func_params.get('reduction')]
            self.kwargs[key] = dict(reduction=reduction)

            # store for initialization
            kwargs = deepcopy(func_params)
            kwargs.pop("PerSpecies")
            kwargs.pop("PerTarget")
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

        # Inspect the function's signature
        sig = inspect.signature(RunningStats)
        # Filter kwargs based on the function's parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        rs = RunningStats(**filtered_kwargs)
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
            per_target   = params["PerTarget"]

            # initialize the internal run_stat base on the error shape
            if key not in self.running_stats:
                self.running_stats[key] = self.init_runstat(params=self.kwargs[key], error=error.flatten())

            stat: RunningStats = self.running_stats[key]
            params = {}
            if per_species:
                node_idcs = ref[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
                not_nan_ref_nodes_fltr = torch.unique(ref[AtomicDataDict.EDGE_INDEX_KEY][0])
                if len(error) != len(not_nan_ref_nodes_fltr):
                    error = error[not_nan_ref_nodes_fltr]
                params["accumulate_by"] = node_idcs[not_nan_ref_nodes_fltr]
            if per_target:
                num_rows, num_targets = error.size()
                accumulate_by = params.get("accumulate_by", torch.zeros(num_rows, device=error.device))
                per_target_accumulate_by = []
                for acc_by_idx in accumulate_by:
                    for target in torch.arange(num_targets, device=error.device):
                        per_target_accumulate_by.append(acc_by_idx * num_targets + target)
                accumulate_by = torch.tensor(per_target_accumulate_by, device=error.device).long()
                params["accumulate_by"] = accumulate_by

            metrics[key] = stat.accumulate_batch(error.flatten(), **params)

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
        '''

        type_names   = metrics_metadata.get('type_names')
        target_names = metrics_metadata.get('target_names')

        flat_dict = {}
        for key, value in metrics.items():
            reduction, params = self.params[key]

            key_clean = self.remove_suffix(key)
            metric_name = ABBREV.get(key_clean, key_clean)
            loss_name = self.funcs[key].func_name
            metric_key = f"{metric_name}_{loss_name}_{reduction}"

            per_species = params["PerSpecies"]
            per_target   = params["PerTarget"]

            if per_species:
                for idx, value_row in enumerate(value):
                    type_name = type_names[idx]
                    flat_dict[f"{type_name}_{metric_key}"] = value_row
            else:
                flat_dict[metric_key] = value

            if per_target:
                for flat_key in list(flat_dict.keys()):
                    if flat_key.endswith(metric_key):
                        flat_value = flat_dict.pop(flat_key)
                        if target_names is None:
                            target_names = [f"target_{i}" for i in range(len(flat_value))]
                        for target_name, value_item in zip(target_names, flat_value):
                            flat_dict[f"{flat_key}_{target_name}"] = value_item

        return {k: v.item() for k,v in flat_dict.items()}