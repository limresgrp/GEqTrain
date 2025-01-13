import torch
from typing import Union

from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from .radial_basis import BesselBasis
from .cutoffs import TanhCutoff


@compile_mode("script")
class SphericalHarmonicEdgeAngularAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        edge_sh_normalization (str): the normalization scheme to use
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalize: bool = True,
        edge_sh_normalization: str = "norm", # with norm each l has norm=1
        irreps_in = None,
        out_field: str = AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.out_field not in data:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
            edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
            edge_sh = self.sh(edge_vec)
            data[self.out_field] = edge_sh
        return data


@compile_mode("script")
class BasisEdgeRadialAttrs(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=TanhCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
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
            edge_length_embedded = self.basis(edge_length) * self.cutoff(edge_length)[:, None]
            data[self.out_field] = edge_length_embedded
        return data


@compile_mode("script")
class EdgeRadialAttrsEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        input_field: str = AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
        out_field: str = "EmbeddedRadialBasis",
    ):
        super().__init__()
        self.input_field = input_field
        self.out_field = out_field
        self.lin_proj = torch.nn.Linear(in_dim, out_dim, bias=True)
        # init ensures to out a 0 vect s.t. sigmoid outs .5 forall edges at start of training
        torch.nn.init.xavier_uniform_(self.lin_proj.weight,  gain=torch.nn.init.calculate_gain("sigmoid"))
        with torch.no_grad(): self.lin_proj.bias.fill_(0.5)

    def forward(self, data: AtomicDataDict.Type, latents:torch.Tensor) -> torch.Tensor:
        # return 2 * latents * self.lin_proj(data[self.input_field]).sigmoid() # lin proj out squishes variance to .5 -> double it to get std=1
        return latents * self.lin_proj(data[self.input_field]).sigmoid() # lin proj out squishes variance to .5 -> double it to get std=1
