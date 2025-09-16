from typing import List, Optional

from collections.abc import Sequence

import torch
import numpy as np
from torch import Tensor

from .data import Data
from .dataset import IndexType


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, ptr=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Data
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_nodes_list__ = None
        self.__num_graphs__ = None

    @classmethod
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""

        keys = list(set(data_list[0].keys) - set(exclude_keys))
        assert "batch" not in keys and "ptr" not in keys

        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != "__" and key[-2:] != "__":
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ["batch"]:
            batch[key] = []
        batch["ptr"] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, Tensor) and item.dim() == 0:
                    cat_dim = None
                cat_dims[key] = cat_dim

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device

                batch[key].append(item)  # Append item to the attribute list.

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f"{key}_{j}_batch"
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size,), i, dtype=torch.long, device=device)
                            )
                    else:
                        tmp = f"{key}_batch"
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size,), i, dtype=torch.long, device=device)
                        )

            if hasattr(data, "__num_nodes__"):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long, device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            if any(x is None for x in items):
                raise ValueError(f"Found a `None` in the provided data objects for batching in key `{key}`")
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, Tensor):
                try:
                    batch[key] = torch.cat(items, cat_dim)
                except:
                    items = [tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor for tensor in items]
                    batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)
        
        return batch.contiguous()

    def get_example(self, idx: int) -> Data:
        r"""Reconstructs the :class:`torch_geometric.data.Data` object at index
        :obj:`idx` from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                (
                    "Cannot reconstruct data list from batch because the batch "
                    "object was not created using `Batch.from_data_list()`."
                )
            )

        data = self.__data_class__()
        idx = self.num_graphs + idx if idx < 0 else idx

        for key in self.__slices__.keys():
            item = self[key]
            if self.__cat_dims__[key] is None:
                item = item[idx]
            else:
                if isinstance(item, Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item.narrow(dim, start, end - start)
                else:
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item[start:end]
                    item = item[0] if len(item) == 1 else item

            cum = self.__cumsum__[key][idx]
            if isinstance(item, Tensor):
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, (int, float)):
                item = item - cum

            data[key] = item

        if self.__num_nodes_list__[idx] is not None:
            data.num_nodes = self.__num_nodes_list__[idx]

        return data

    def index_select(self, idx: IndexType) -> List[Data]:
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])
        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            idx = idx.flatten().tolist()
        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()
        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            idx = idx.flatten().tolist()
        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0].flatten().tolist()
        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass
        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')"
            )
        return [self.get_example(i) for i in idx]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(Batch, self).__getitem__(idx)
        elif isinstance(idx, (int, np.integer)):
            return self.get_example(idx)
        else:
            return self.index_select(idx)

    def to_data_list(self) -> List[Data]:
        return [self.get_example(i) for i in range(self.num_graphs)]

    def subgraph(self, subset: torch.Tensor, edge_index: Optional[Tensor] = None, ignore_keys: List[str] = None) -> "Batch":
        """
        Returns a new Batch object containing only the nodes in `subset`
        and the edges between them.
        """
        from geqtrain.data import _EDGE_FIELDS, _GRAPH_FIELDS, _NODE_FIELDS
        
        ignore_keys = ignore_keys or []

        if subset.dtype != torch.bool:
            node_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=subset.device)
            node_mask[subset] = True
            subset = node_mask

        current_edge_index = edge_index if edge_index is not None else self.edge_index        
        edge_set = set(map(tuple, current_edge_index.t().tolist()))
        edge_mask = torch.tensor(
            [tuple(e) in edge_set for e in self.edge_index.t().tolist()],
            dtype=torch.bool,
            device=self.edge_index.device,
        )

        node_map = torch.full((self.num_nodes,), -1, dtype=torch.long, device=subset.device)
        node_map[subset] = torch.arange(subset.sum(), device=subset.device)

        new_data_kwargs = {}
        for key, value in self:
            if key in ['batch', 'ptr'] or key in ignore_keys: continue
            
            if key == 'edge_index':
                new_data_kwargs[key] = node_map[value[:, edge_mask]]
            elif torch.is_tensor(value) and value.size(0) == self.num_nodes:
                new_data_kwargs[key] = value[subset]
            elif torch.is_tensor(value) and value.size(0) == self.num_edges:
                new_data_kwargs[key] = value[edge_mask]
            else:
                new_data_kwargs[key] = value

        new_batch_tensor = self.batch[subset]
        new_ptr = torch.cat([torch.tensor([0], device=new_batch_tensor.device), torch.cumsum(torch.bincount(new_batch_tensor), 0)])
        unique_graphs, new_graph_node_counts = torch.unique(new_batch_tensor, return_counts=True)
        
        if len(unique_graphs) == 0: return None

        new_slices, new_cumsum = {}, {}
        graph_map = torch.full((self.num_graphs,), -1, dtype=torch.long, device=new_batch_tensor.device)
        graph_map[unique_graphs] = torch.arange(len(unique_graphs), device=new_batch_tensor.device)
        node_increments = torch.cat([torch.tensor([0], device=self.batch.device), torch.cumsum(new_graph_node_counts, 0)])
        
        all_keys = set(self.__slices__.keys())
        for key, value in self:
             if key.endswith('_slices'):
                 all_keys.add(key.replace('_slices', ''))

        for key in all_keys:
            if key not in self.__slices__: continue

            item_sizes = torch.tensor(self.__slices__[key], device=self.batch.device).diff()
            
            if key in _NODE_FIELDS or key == 'batch':
                new_item_sizes = new_graph_node_counts
                new_cumsum[key] = [0] * (len(unique_graphs) + 1)
            elif key in _EDGE_FIELDS or key == 'edge_index':
                edge_batch = self.batch[self.edge_index[0, edge_mask]]
                new_item_sizes = torch.bincount(graph_map[edge_batch], minlength=len(unique_graphs))
                new_cumsum[key] = node_increments.tolist()
            elif key in _GRAPH_FIELDS:
                new_item_sizes = torch.ones(len(unique_graphs), dtype=torch.long, device=self.batch.device)
                new_cumsum[key] = [0] * (len(unique_graphs) + 1)
            else:
                new_item_sizes = item_sizes[unique_graphs]
                new_cumsum[key] = [0] * (len(unique_graphs) + 1)
                slice_key = f"{key}_slices"
                if slice_key in self:
                    new_slice_tensor = torch.cat([
                        torch.tensor([0], device=self.batch.device),
                        torch.cumsum(new_item_sizes, 0)
                    ])
                    new_data_kwargs[slice_key] = new_slice_tensor

            new_slices[key] = torch.cat([torch.tensor([0], device=self.batch.device), torch.cumsum(new_item_sizes, 0)]).tolist()

        new_batch_obj = Batch(batch=new_batch_tensor, ptr=new_ptr, **new_data_kwargs)
        new_batch_obj.__slices__ = new_slices
        new_batch_obj.__cumsum__ = new_cumsum
        new_batch_obj.__num_graphs__ = len(unique_graphs)
        new_batch_obj.__cat_dims__ = self.__cat_dims__
        if self.__num_nodes_list__ is not None:
            new_batch_obj.__num_nodes_list__ = new_graph_node_counts.tolist()
        
        return new_batch_obj

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        if self.__num_graphs__ is not None:
            return self.__num_graphs__
        elif self.ptr is not None:
            return self.ptr.numel() - 1
        elif self.batch is not None:
            return int(self.batch.max()) + 1
        else:
            raise ValueError

