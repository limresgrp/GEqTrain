# components/early_stopping.py
""" Adapted from https://github.com/mir-group/nequip """

from collections import OrderedDict
from copy import deepcopy
from typing import Mapping

class EarlyStopping:
    """
    Early stop conditions for metrics that should decrease or increase.

    The patience condition checks if a metric has not improved for a given
    number of epochs. "Improvement" is defined based on the metric's criteria.

    Args:
        lower_bounds (dict): Stop if a metric goes below this value.
        upper_bounds (dict): Stop if a metric goes above this value.
        patiences (dict): The number of epochs to wait for an improvement.
        criteria (dict): Maps metric keys to 'increasing' or 'decreasing'.
                         Defaults to 'decreasing' if a key is not specified.
        delta (dict): The minimum change to qualify as an improvement.
        cumulative_delta (bool): If True, the best value is only updated
                                 if the improvement is greater than delta.
    """
    def __init__(
        self,
        lower_bounds: dict = {},
        upper_bounds: dict = {},
        patiences: dict = {},
        criteria: dict = {},
        delta: dict = {},
        cumulative_delta: bool = False,
    ):
        self.patiences = deepcopy(patiences)
        self.criteria = deepcopy(criteria)
        self.lower_bounds = deepcopy(lower_bounds)
        self.upper_bounds = deepcopy(upper_bounds)
        self.cumulative_delta = cumulative_delta

        self.delta = {}
        self.counters = {}
        self.bests = {}  # Renamed from minimums for generality

        for key, pat in self.patiences.items():
            self.patiences[key] = int(pat)
            self.counters[key] = 0
            self.bests[key] = None
            self.delta[key] = delta.get(key, 0.0)

            # Set default criterion if not provided
            self.criteria.setdefault(key, 'decreasing')
            if self.criteria[key] not in ['decreasing', 'increasing']:
                raise ValueError(f"Criterion for {key} must be 'decreasing' or 'increasing'.")
            if pat < 1:
                raise ValueError(f"Patience for {key} should be a positive integer.")
            if self.delta[key] < 0.0:
                raise ValueError("Delta should not be a negative number.")

    def __call__(self, metrics) -> None:
        stop = False
        stop_args = "Early stopping:"
        debug_args = None

        # Check patience condition for each metric
        for key, pat in self.patiences.items():
            if key not in metrics:
                continue
                
            value = metrics[key]
            best_so_far = self.bests[key]
            delta = self.delta[key]
            criterion = self.criteria[key]

            if best_so_far is None:
                self.bests[key] = value
                continue

            # Determine if the metric has improved
            has_improved = False
            if criterion == 'decreasing':
                if value < (best_so_far - delta):
                    has_improved = True
            else: # increasing
                if value > (best_so_far + delta):
                    has_improved = True
            
            if has_improved:
                self.bests[key] = value
                self.counters[key] = 0
            else:
                self.counters[key] += 1
                debug_args = f"EarlyStopping patience for '{key}': {self.counters[key]}/{pat}"
                if self.counters[key] >= pat:
                    stop_args += f" '{key}' has not improved for {pat} epochs."
                    stop = True
        
        # Check boundary conditions
        for key, bound in self.lower_bounds.items():
            if key in metrics and metrics[key] < bound:
                stop_args += f" '{key}' ({metrics[key]}) is smaller than lower bound ({bound})."
                stop = True

        for key, bound in self.upper_bounds.items():
            if key in metrics and metrics[key] > bound:
                stop_args += f" '{key}' ({metrics[key]}) is larger than upper bound ({bound})."
                stop = True

        return stop, stop_args, debug_args

    def state_dict(self) -> "OrderedDict[dict, dict]":
        return OrderedDict([("counters", self.counters), ("bests", self.bests)])

    def load_state_dict(self, state_dict: Mapping) -> None:
        self.counters = state_dict["counters"]
        # Support for old checkpoints
        if "minimums" in state_dict:
            self.bests = state_dict["minimums"]
        else:
            self.bests = state_dict["bests"]