"""Edge geometry modules and compatibility exports.

`BaseEdgeEmbedding` and `BaseEdgeEqEmbedding` now live in
`geqtrain.nn.embeddings.edge`.
"""

from typing import Optional, Union

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from .cutoffs import PolynomialCutoff
from .radial_basis import BesselBasis
from .embeddings.edge import BaseEdgeEmbedding, BaseEdgeEqEmbedding


def get_irreps_edge_sh(
    l_max: Optional[int] = None,
    parity: Optional[str] = None,
    irreps_edge_sh: Optional[str] = None,
):
    if l_max is not None or parity is not None:
        assert l_max is not None
        assert parity is not None
        assert l_max >= 0
        assert parity in ("o3_full", "so3")
        irreps_edge_sh_computed = repr(
            o3.Irreps.spherical_harmonics(l_max, p=(1 if parity == "so3" else -1))
        )
        if irreps_edge_sh is not None:
            assert irreps_edge_sh_computed == irreps_edge_sh
        return irreps_edge_sh_computed
    assert irreps_edge_sh is not None
    return irreps_edge_sh


@compile_mode("script")
class SphericalHarmonicEdgeAngularAttrs(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Optional[Union[int, str, o3.Irreps]],
        l_max: Optional[int] = None,
        parity: Optional[str] = None,
        edge_sh_normalize: bool = True,
        edge_sh_normalization: str = "norm",
        out_field: str = AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.out_field = out_field

        irreps_edge_sh = get_irreps_edge_sh(
            l_max=l_max,
            parity=parity,
            irreps_edge_sh=irreps_edge_sh,
        )
        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh,
            edge_sh_normalize,
            edge_sh_normalization,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.out_field not in data:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
            data[self.out_field] = self.sh(data[AtomicDataDict.EDGE_VECTORS_KEY])
        return data


@compile_mode("script")
class BasisEdgeRadialAttrs(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_RADIAL_EMB_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.out_field not in data:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
            edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
            data[self.out_field] = self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        return data


__all__ = [
    "get_irreps_edge_sh",
    "SphericalHarmonicEdgeAngularAttrs",
    "BasisEdgeRadialAttrs",
    "BaseEdgeEmbedding",
    "BaseEdgeEqEmbedding",
]
