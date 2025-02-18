""" Adapted from https://github.com/mir-group/nequip
"""

from typing import List
from functools import reduce

import torch

from geqtrain.utils.torch_geometric import Batch, Data


class Collater(object):
    """Collate a list of ``AtomicData``.

    callable

    Args:
        fixed_fields: which fields are fixed fields
        exclude_keys: keys to ignore in the input, not copying to the output
    """

    def __init__(self, exclude_keys: List[str] = []):
        self._exclude_keys = set(exclude_keys)

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        # Allow to merge ensemble graphs into a batch.
        # Groups graphs by ensemble and adds a mapping tensor for tracking.
        batch_ensemble_index = []  # Tracks which molecule each graph belongs to        
        for graph in batch:
            batch_ensemble_index.append(graph.ensemble_index)

        batch_graphs = Batch.from_data_list(batch, exclude_keys=self._exclude_keys.union(["ensemble_index"]))
        _, batch_graphs.ensemble_index = torch.unique(torch.tensor(batch_ensemble_index, dtype=torch.long), return_inverse=True)

        return batch_graphs

    def __call__(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        return self.collate(batch)

    @property
    def exclude_keys(self):
        return list(self._exclude_keys)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: List[str] = [],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(exclude_keys=exclude_keys),
            **kwargs,
        )
