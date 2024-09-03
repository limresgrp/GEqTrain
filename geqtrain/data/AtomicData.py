""" Adapted from https://github.com/mir-group/nequip

Original authors: Albert Musaelian
"""

from typing import Union, Dict, Set, Sequence
from collections.abc import Mapping

import numpy as np

import torch
import e3nn.o3

from . import AtomicDataDict
from geqtrain.utils.torch_geometric import Data


# There is no built-in way to check if a Tensor is of an integer type
_TORCH_INTEGER_DTYPES = (torch.int, torch.long)

_DEFAULT_LONG_FIELDS: Set[str] = {
    AtomicDataDict.NODE_TYPE_KEY,
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.EDGE_TYPE_KEY,
    AtomicDataDict.BATCH_KEY,
}
_DEFAULT_NODE_FIELDS: Set[str] = {
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.NODE_FEATURES_KEY,
    AtomicDataDict.NODE_ATTRS_KEY,
    AtomicDataDict.NODE_TYPE_KEY,
    AtomicDataDict.NODE_OUTPUT_KEY,
    AtomicDataDict.BATCH_KEY,
}
_DEFAULT_EDGE_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_VECTORS_KEY,
    AtomicDataDict.EDGE_LENGTH_KEY,
    AtomicDataDict.EDGE_FEATURES_KEY,
    AtomicDataDict.EDGE_ATTRS_KEY,
    AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
    AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
    AtomicDataDict.EDGE_TYPE_KEY,
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
}
_DEFAULT_GRAPH_FIELDS: Set[str] = {
    AtomicDataDict.GRAPH_ATTRS_KEY,
    AtomicDataDict.GRAPH_OUTPUT_KEY,
}

_NODE_FIELDS:  Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS:  Set[str] = set(_DEFAULT_EDGE_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_LONG_FIELDS:  Set[str] = set(_DEFAULT_LONG_FIELDS)


def register_fields(
    node_fields:  Sequence[str] = [],
    edge_fields:  Sequence[str] = [],
    graph_fields: Sequence[str] = [],
    long_fields:  Sequence[str] = [],
) -> None:
    r"""

    Called during instantiation of the dataset using the cfg,
    updates global dicts:
    - _NODE_FIELDS
    - _EDGE_FIELDS
    - _GRAPH_FIELDS
    - _LONG_FIELDS
    that are used to parse the yaml and thus the data from source.


    Register fields as being per-node, per-edge, or per-graph.
    with this function we can register custom keys in the AtomicData/AtomicDataDict
    register as key in the AtomicDataDict the values of the following yaml keys:
        - node_fields
        - edge_fields
        - graph_fields
        - long_fields
    """
    node_fields:  set = set(node_fields)
    edge_fields:  set = set(edge_fields)
    graph_fields: set = set(graph_fields)
    allfields = node_fields.union(edge_fields, graph_fields)
    assert len(allfields) == len(node_fields) + len(edge_fields) + len(graph_fields)

    _NODE_FIELDS.update(node_fields)
    _EDGE_FIELDS.update(edge_fields)
    _GRAPH_FIELDS.update(graph_fields)
    _LONG_FIELDS.update(long_fields)
    if len(set.union(_NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS)) < (len(_NODE_FIELDS) + len(_EDGE_FIELDS) + len(_GRAPH_FIELDS)):
        raise ValueError("At least one key was registered as more than one of node, edge, or graph!")


def _process_dict(kwargs, ignore_fields=[]):
    """Convert a dict of data into correct dtypes/shapes according to key"""
    # Deal with _some_ dtype issues
    for k, v in kwargs.items():
        if k in ignore_fields:
            continue

        if k in _LONG_FIELDS:
            # Any property used as an index must be long (or byte or bool, but those are not relevant for atomic scale systems)
            # int32 would pass later checks, but is actually disallowed by torch
            kwargs[k] = torch.as_tensor(v, dtype=torch.long)
        elif isinstance(v, bool):
            kwargs[k] = torch.as_tensor(v)
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                kwargs[k] = torch.as_tensor(v)
        elif isinstance(v, list):
            ele_dtype = np.array(v).dtype
            if np.issubdtype(ele_dtype, np.floating):
                kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                kwargs[k] = torch.as_tensor(v)
        elif np.issubdtype(type(v), np.floating):
            # Force scalars to be tensors with a data dimension
            # This makes them play well with irreps
            kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
        elif np.issubdtype(type(v), int):
            # Force scalars to be tensors with a data dimension
            # This makes them play well with irreps
            kwargs[k] = torch.as_tensor(v, dtype=torch.long)
        elif isinstance(v, torch.Tensor) and len(v.shape) == 0:
            # ^ this tensor is a scalar; we need to give it
            # a data dimension to play nice with irreps
            kwargs[k] = v

    if AtomicDataDict.BATCH_KEY in kwargs:
        num_frames = kwargs[AtomicDataDict.BATCH_KEY].max() + 1
    else:
        num_frames = 1

    for k, v in kwargs.items():
        if k in ignore_fields: # check if k from red_kwords from yaml has to be ingnored
            continue

        # check if it must be added the batch size, nb: bs always = 1 when reading data
        if len(v.shape) == 0:
            kwargs[k] = v.unsqueeze(-1)
            v = kwargs[k]

        # check if it must be added the batch size, nb: bs always = 1 when reading data
        if k in set.union(_NODE_FIELDS, _EDGE_FIELDS) and len(v.shape) == 1:
            kwargs[k] = v.unsqueeze(-1)
            v = kwargs[k]

        # consistency checks
        if (
            k in _NODE_FIELDS
            and AtomicDataDict.POSITIONS_KEY in kwargs
            and v.shape[0] != kwargs[AtomicDataDict.POSITIONS_KEY].shape[0]
        ):
            raise ValueError(
                f"{k} is a node field but has the wrong dimension {v.shape}"
            )
        elif (
            k in _EDGE_FIELDS
            and AtomicDataDict.EDGE_INDEX_KEY in kwargs
            and v.shape[0] != kwargs[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        ):
            raise ValueError(
                f"{k} is a edge field but has the wrong dimension {v.shape}, \
                    (first dimension should be {kwargs[AtomicDataDict.EDGE_INDEX_KEY].shape[1]})"
            )
        elif k in _GRAPH_FIELDS:
            if num_frames > 1 and v.shape[0] != num_frames:
                raise ValueError(f"Wrong shape for graph property {k}")


class AtomicData(Data):
    """A neighbor graph for points in real space.

    For typical cases ``from_points`` should be used to construct a AtomicData.

    Args:
        pos (Tensor [n_nodes, 3]): Positions of the nodes.
        edge_index (LongTensor [2, n_edges]): ``edge_index[0]`` is the per-edge
                 index of the source node and ``edge_index[1]`` is the target node.
        node_features (Tensor [n_atom, ...]): the input features of the nodes, optional
        node_attrs (Tensor [n_atom, ...]): the attributes of the nodes, for instance the atom type, optional
        batch (Tensor [n_atom]): the graph to which the node belongs, optional
        atom_type (Tensor [n_atom]): optional.
        **kwargs: other data, optional.
    """

    def __init__(
        self,
        irreps: Dict[str, e3nn.o3.Irreps] = {},
        _validate: bool = True,
        **kwargs
    ):

        # empty init needed by get_example
        if len(kwargs) == 0 and len(irreps) == 0:
            super().__init__()
            return

        # Check the keys
        if _validate:
            AtomicDataDict.validate_keys(kwargs)
            _process_dict(kwargs)

        super().__init__(num_nodes=len(kwargs["pos"]), **kwargs)

        if _validate:
            # Validate shapes
            assert self.pos.dim() == 2 and self.pos.shape[1] == 3
            assert self.edge_index.dim() == 2 and self.edge_index.shape[0] == 2
            if AtomicDataDict.NODE_ATTRS_KEY in self and self.node_attrs is not None:
                assert self.node_attrs.shape[0] == self.num_nodes
                #! assert self.node_attrs.dtype == self.pos.dtype
            if AtomicDataDict.NODE_FEATURES_KEY in self and self.node_features is not None:
                assert self.node_features.shape[0] == self.num_nodes
                assert self.node_features.dtype == self.pos.dtype
            if AtomicDataDict.NODE_TYPE_KEY in self and self.node_types is not None:
                assert self.node_types.dtype in _TORCH_INTEGER_DTYPES
            if AtomicDataDict.BATCH_KEY in self and self.batch is not None:
                assert self.batch.dim() == 2 and self.batch.shape[0] == self.num_nodes
            if AtomicDataDict.CELL_KEY in self and self.cell is not None:
                assert self.cell.shape == (3, 3)

            # Validate irreps
            # __*__ is the only way to hide from torch_geometric
            self.__irreps__ = AtomicDataDict._fix_irreps_dict(irreps)
            for field, irreps in self.__irreps__:
                if irreps is not None:
                    assert self[field].shape[-1] == irreps.dim

    @classmethod
    def from_points(
        cls,
        pos=None,
        r_max: float = None,
        **kwargs,
    ):
        """Build neighbor graph from points.

        Args:
            pos (np.ndarray/torch.Tensor shape [N, 3]): node positions. If Tensor, must be on the CPU.
            r_max (float): neighbor cutoff radius.
            **kwargs (optional): other fields to add. Keys listed in ``AtomicDataDict.*_KEY` will be treated specially.
        """
        if pos is None or r_max is None:
            raise ValueError("pos and r_max must be given.")

        pos = torch.as_tensor(pos, dtype=torch.get_default_dtype())
        edge_index = kwargs.get(AtomicDataDict.EDGE_INDEX_KEY, None)

        if edge_index is None:
            edge_index = neighbor_list(pos=pos, r_max=r_max)

        return cls(pos=pos, edge_index=edge_index, **kwargs)

    @staticmethod
    def to_AtomicDataDict(
        data: Union[Data, Mapping], exclude_keys=tuple()
    ) -> AtomicDataDict.Type:
        if isinstance(data, Data):
            keys = data.keys
        elif isinstance(data, Mapping):
            keys = data.keys()
        else:
            raise ValueError(f"Invalid data `{repr(data)}`")

        return {
            k: data[k]
            for k in keys
            if (
                k not in exclude_keys
                and data[k] is not None
                and isinstance(data[k], torch.Tensor)
            )
        }

    @classmethod
    def from_AtomicDataDict(cls, data: AtomicDataDict.Type):
        # it's an AtomicDataDict, so don't validate-- assume valid:
        return cls(_validate=False, **data)

    @property
    def irreps(self):
        return self.__irreps__

    def __cat_dim__(self, key, value):
        if key == AtomicDataDict.EDGE_INDEX_KEY:
            return 1  # always cat in the edge dimension
        elif key in _GRAPH_FIELDS:
            # graph-level properties and so need a new batch dimension
            return None
        else:
            return 0  # cat along node/edge dimension

def neighbor_list(
    pos: torch.Tensor,
    r_max: float,
):
    """Create neighbor list (``edge_index``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
    """

    dist_matrix = torch.norm(pos[:, None, ...] - pos[None, ...], dim=-1).fill_diagonal_(torch.inf)
    return torch.argwhere(dist_matrix <= r_max).T.long().to(device=pos.device)