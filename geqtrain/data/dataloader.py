""" Adapted from https://github.com/mir-group/nequip
"""

import torch
from typing import List
from geqtrain.data import AtomicDataDict
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

    @staticmethod
    def _is_optional_mask_key(key: str) -> bool:
        return "mask" in str(key).lower()

    def _fill_missing_optional_masks(self, batch: List[Data]) -> List[Data]:
        """Treat a missing mask field as an all-true mask for that graph.

        Some datasets provide node masks only for validation/test targets. When
        such graphs are batched with graphs that do not carry the mask, the
        intended semantics are "no extra mask" for the missing graph, not a
        collation failure.
        """
        all_keys = set().union(*(set(graph.keys) for graph in batch)) - self._exclude_keys
        optional_mask_keys = [key for key in all_keys if self._is_optional_mask_key(key)]
        if not optional_mask_keys:
            return batch

        for key in optional_mask_keys:
            present = [graph[key] for graph in batch if key in graph and graph[key] is not None]
            if not present or not all(torch.is_tensor(item) for item in present):
                continue
            exemplar = present[0]
            if exemplar.ndim == 0:
                continue
            for graph in batch:
                if key in graph and graph[key] is not None:
                    continue
                if graph.num_nodes is None:
                    continue
                shape = (graph.num_nodes,) + tuple(exemplar.shape[1:])
                graph[key] = torch.ones(shape, dtype=exemplar.dtype, device=exemplar.device)
        return batch

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        batch = self._fill_missing_optional_masks(batch)

        # Allow to merge ensemble graphs into a batch.
        # Groups graphs by ensemble and adds a mapping tensor for tracking.
        batch_ensemble_index = []  # Tracks which molecule each graph belongs to
        for graph in batch:
            batch_ensemble_index.append(graph.ensemble_index)

        batch_graphs = Batch.from_data_list(batch, exclude_keys=self._exclude_keys.union([AtomicDataDict.ENSEMBLE_INDEX_KEY]))
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
        batch_sampler=None,
        exclude_keys: List[str] = [],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        if batch_sampler is not None:
            super(DataLoader, self).__init__(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=Collater(exclude_keys=exclude_keys),
                **kwargs,
            )
        else:
            super(DataLoader, self).__init__(
                dataset,
                batch_size,
                shuffle,
                collate_fn=Collater(exclude_keys=exclude_keys),
                **kwargs,
            )
