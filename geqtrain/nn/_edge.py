import torch
from typing import Union

from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn._embedding import BaseEmbedding
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
        edge_sh_normalization (str): the normalization scheme to use. 'component': each l has norm=sqrt(2l+1) | 'norm': each l has norm=1
        out_field (str, default: AtomicDataDict.EDGE_SPHARMS_EMB_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalize: bool = True,
        edge_sh_normalization: str = "component",
        out_field: str = AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
        irreps_in = None,
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
            data[self.out_field] = self.sh(data[AtomicDataDict.EDGE_VECTORS_KEY])
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


@compile_mode("script")
class BaseEdgeEmbedding(BaseEmbedding):

    def __init__(self, latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.has_edge_attr  = False
        self.proj_to_latent = False

        node_irreps = self.irreps_in[self.node_field]
        edge_radial_irreps = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        edge_dim = 2 * node_irreps.dim + edge_radial_irreps.dim
        out_dim = edge_dim
        
        if self.edge_field in self.irreps_in:
            self.has_edge_attr = True
            edge_dim += self.irreps_in[self.edge_field].dim

        if latent_dim is not None:
            self.proj_to_latent = True
            self.register_parameter("linear_proj", torch.nn.Parameter(torch.randn(edge_dim, latent_dim)))
            torch.nn.init.xavier_uniform_(self.linear_proj)
            out_dim = latent_dim

        self.out_irreps = o3.Irreps(f"{out_dim}x0e")
    
    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_center, edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_attr               = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        node_attr               = data[self.node_field]

        edge_attrs_to_cat = [node_attr[edge_center], node_attr[edge_neigh], edge_attr]
        if self.has_edge_attr:
            edge_attrs_to_cat.append(data.pop(self.edge_field))
        edge_attrs_emb = torch.cat(edge_attrs_to_cat, dim=-1)
        if self.proj_to_latent:
            edge_attrs_emb = edge_attrs_emb @ self.linear_proj
        return edge_attrs_emb


@compile_mode("script")
class BaseEdgeEqEmbedding(BaseEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_node_eq_attr = False
        self.has_edge_eq_attr = False

        # 1. Create the Irreps object for the naively concatenated features
        unsorted_irreps_list = list(self.irreps_in[AtomicDataDict.EDGE_SPHARMS_EMB_KEY])
        if self.edge_eq_field in self.irreps_in:
            self.has_edge_eq_attr = True
            # 1.1 Add edge_eq_input_attr to the Irreps object
            unsorted_irreps_list += list(self.irreps_in[self.edge_eq_field])

        if self.node_eq_field in self.irreps_in: # concat node_eq_attr of each atom pair to spharms
            self.has_node_eq_attr = True
            # 1.2 Add pairs of node_eq_input_attr to the Irreps object
            unsorted_irreps_list += list(self.node_eq_irreps_out) + list(self.node_eq_irreps_out)
            
        if self.has_edge_eq_attr or self.has_node_eq_attr:
            unsorted_irreps = o3.Irreps(unsorted_irreps_list)
            # 2. Get the sorted irreps and the BLOCK permutation
            sorted_irreps, p_blocks, _ = unsorted_irreps.sort()
            # 3. Get the dimensions of each block in the ORIGINAL unsorted order,
            #    correctly accounting for multiplicity.
            dims = torch.tensor([mul * ir.dim for mul, ir in unsorted_irreps])
            # 4. Get the starting indices (offsets) of each block in the original tensor.
            offsets = torch.cumsum(torch.cat((torch.tensor([0]), dims[:-1])), dim=0)
            # 5. Compute the inverse of the block permutation (argsort).
            #    This tells us which original block should go into each new position.
            arg_p_blocks = sorted(range(len(p_blocks)), key=p_blocks.__getitem__)
            # 6. Build the full element-wise permutation using the inverse block permutation.
            p_elements = torch.cat([torch.arange(dims[i]) + offsets[i] for i in arg_p_blocks])
            # 7. Store the final, correct, element-wise permutation as a buffer
            self.register_buffer('concatenation_permutation', p_elements)
            # 8. Store the final sorted irreps description for later use
            edge_eq_irreps = sorted_irreps
        else:
            # If there are no node features to concatenate, no permutation is needed.
            self.concatenation_permutation = None
            edge_eq_irreps = o3.Irreps(unsorted_irreps_list)
        self.out_irreps = edge_eq_irreps
    
    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_eq_attr = data[AtomicDataDict.EDGE_SPHARMS_EMB_KEY]
        if self.has_edge_eq_attr:
            # 1.1 Perform the naive concatenation.
            edge_eq_field = self.edge_eq_field
            assert isinstance(edge_eq_field, str)
            edge_eq_attr = torch.cat([edge_eq_attr, data.pop(edge_eq_field)])
        if self.has_node_eq_attr:
            # 1.2 Perform the naive concatenation.
            edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
            edge_neigh  = data[AtomicDataDict.EDGE_INDEX_KEY][1]
            node_eq_field = self.node_eq_field
            assert isinstance(node_eq_field, str)
            node_eq_attr            = data[node_eq_field]
            edge_eq_attr = torch.cat([
                edge_eq_attr, 
                node_eq_attr[edge_center], 
                node_eq_attr[edge_neigh]
            ], dim=-1)
        if self.has_edge_eq_attr or self.has_node_eq_attr:
            # 2. Apply the pre-computed element-wise permutation.
            #    This `edge_attr` tensor now correctly corresponds to `input_edge_eq_irreps`.
            edge_eq_attr = edge_eq_attr[:, self.concatenation_permutation]
        return edge_eq_attr