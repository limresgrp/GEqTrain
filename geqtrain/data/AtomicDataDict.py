""" Adapted from https://github.com/mir-group/nequip

These TorchScript functions operate on ``Dict[str, torch.Tensor]`` representations
of the ``AtomicData`` class which are produced by ``AtomicData.to_AtomicDataDict()``.

Original authors: Albert Musaelian
"""

from typing import Dict, Any

import torch
import torch.jit

from e3nn import o3

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys

# Define a type alias
Type = Dict[str, torch.Tensor]


def validate_keys(keys, graph_required=True):
    # Validate combinations
    if graph_required:
        if not (_keys.POSITIONS_KEY in keys and _keys.EDGE_INDEX_KEY in keys):
            raise KeyError("At least pos and edge_index must be supplied")


_SPECIAL_IRREPS = [None]


def _fix_irreps_dict(d: Dict[str, Any]):
    return {k: (i if i in _SPECIAL_IRREPS else o3.Irreps(i)) for k, i in d.items()}


def _irreps_compatible(ir1: Dict[str, o3.Irreps], ir2: Dict[str, o3.Irreps]):
    return all(ir1[k] == ir2[k] for k in ir1 if k in ir2)


@torch.jit.script
def with_edge_vectors(data: Type, with_lengths: bool = True) -> Type:
    """Compute the edge displacement vectors for a graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    if _keys.EDGE_VECTORS_KEY not in data:
        pos = data[_keys.POSITIONS_KEY]
        edge_index_src = data[_keys.EDGE_INDEX_KEY][0]
        edge_index_trg = data[_keys.EDGE_INDEX_KEY][1]
        data[_keys.EDGE_VECTORS_KEY] = pos[edge_index_trg] - pos[edge_index_src]
        
    if with_lengths and _keys.EDGE_LENGTH_KEY not in data:
        data[_keys.EDGE_LENGTH_KEY] = torch.linalg.norm(
            data[_keys.EDGE_VECTORS_KEY], dim=-1
        )
    
    return data

@torch.jit.script
def with_batch(data: Type) -> Type:
    """Get batch Tensor.

    If this AtomicDataPrimitive has no ``batch``, one of all zeros will be
    allocated and returned.
    """
    if _keys.BATCH_KEY in data:
        return data
    
    pos = data[_keys.POSITIONS_KEY]
    batch = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    data[_keys.BATCH_KEY] = batch
    return data
