import math
import functools
import torch
import wandb

from typing import Optional, List, Tuple, Union
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from einops.layers.torch import Rearrange

from e3nn import o3
from e3nn.util.jit import compile_mode

from .cutoffs import TanhCutoff
from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    GraphModuleMixin,
    SO3_LayerNorm,
    ScalarMLPFunction,
)
from geqtrain.nn.allegro import (
    Linear,
    Contracter,
    MakeWeightedChannels,
)
from geqtrain.utils.tp_utils import SCALAR, tp_path_exists
from geqtrain.nn.cutoffs import tanh_cutoff
from geqtrain.nn._film import FiLMFunction

from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.nn.AdaLN import AdaLN


def apply_residual_stream(skip_residual: bool, latents, new_latents, this_layer_update_coeff: Optional[torch.Tensor], active_edges):
    if skip_residual:
        # Normal (non-residual) update
        # index_copy replaces, unlike index_add
        return torch.index_copy(latents, 0, active_edges, new_latents)

    assert this_layer_update_coeff is not None
    # At init, we assume new and old to be approximately uncorrelated
    # Thus their variances add
    # we always want the latent space to be normalized to variance = 1.0,
    # because it is critical for learnability. Still, we want to preserve
    # the _relative_ magnitudes of the current latent and the residual update
    # to be controled by `this_layer_update_coeff`
    # Solving the simple system for the two coefficients:
    #   a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
    # gives
    #   a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
    # rsqrt is reciprocal sqrt
    coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
    coefficient_new = this_layer_update_coeff * coefficient_old
    # Residual update
    # Note that it only runs when there are latents to resnet with, so not at the first layer
    # index_add adds only to the edges for which we have something to contribute
    return torch.index_add(coefficient_old * latents, 0, active_edges, coefficient_new * new_latents)


@compile_mode("script")
class GotenInteractionModule(GraphModuleMixin, torch.nn.Module):
    '''
    '''
    num_layers: int
    env_embed_multiplicity: int
    out_field: str
    def __init__(
        self,
        # required params
        num_layers: int,
        # optional params
        out_irreps_node: Optional[Union[o3.Irreps, str]] = None,
        out_irreps_edge: Optional[Union[o3.Irreps, str]] = None,
        # cutoff
        # cutoff=TanhCutoff,
        # cutoff_kwargs={},
        # alias:
        out_field_node        = AtomicDataDict.NODE_FEATURES_KEY,
        out_field_edge        = AtomicDataDict.EDGE_FEATURES_KEY,
        # hyperparams:
        latent_dim:           int = 64,
        eq_multiplicity:      int = 8,
        head_dim:             int = 16,
        product_correlation:  int  = 2,
        # MLP parameters:
        latent                 = ScalarMLPFunction,
        latent_kwargs          = {},
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        assert (num_layers >= 1)
        # save parameters
        self.out_field_node       = out_field_node
        self.out_field_edge       = out_field_edge
        self.latent_dim           = latent_dim

        # self.cutoff = cutoff(**cutoff_kwargs)
        
        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_ATTRS_KEY,
                AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
        ])
        # initialize scalar functions
        latent = functools.partial(latent, **latent_kwargs)

        # Embed to the spharm * it as mul
        spharms_irreps = self.irreps_in[AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY]
        assert all(mul == 1 for mul, _ in spharms_irreps)

        node_eq_irreps = o3.Irreps([(eq_multiplicity, ir) for _, ir in spharms_irreps])
        assert (node_eq_irreps[0].ir == SCALAR), "node_eq_irreps must start with scalars"

        # if not out_irreps is specified, default to hidden irreps with degree of spharms and multiplicity of latent
        if out_irreps_node is None:
            out_irreps_node = o3.Irreps([(self.latent_dim, ir) for _, ir in spharms_irreps])
        else:
            out_irreps_node = out_irreps_node if isinstance(out_irreps_node, o3.Irreps) else o3.Irreps(out_irreps_node)
        if out_irreps_edge is None:
            out_irreps_edge = o3.Irreps([(self.latent_dim, (0, 1))])
        else:
            out_irreps_edge = out_irreps_edge if isinstance(out_irreps_edge, o3.Irreps) else o3.Irreps(out_irreps_edge)

        self.out_muls_node = [mul for mul, _ in out_irreps_node]
        self.out_muls_edge = [mul for mul, _ in out_irreps_edge]

        # irreps
        input_node_irreps = irreps_in[AtomicDataDict.NODE_ATTRS_KEY]
        input_edge_irreps = irreps_in[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY]
        if AtomicDataDict.EDGE_FEATURES_KEY in irreps_in:
            input_edge_irreps = input_edge_irreps + irreps_in[AtomicDataDict.EDGE_FEATURES_KEY]

        # init h
        self.W_node = torch.nn.Parameter(torch.randn(input_node_irreps.dim, self.latent_dim))
        self.W_center_node = torch.nn.Parameter(torch.randn(input_node_irreps.dim, self.latent_dim))
        self.W_concat_node = torch.nn.Parameter(torch.randn(2 * self.latent_dim, self.latent_dim))
        edge_attr_dim = self.irreps_in[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY].dim
        self.W_edge = torch.nn.Parameter(torch.randn(edge_attr_dim, self.latent_dim))
        self.node_norm = torch.nn.LayerNorm(self.latent_dim)
        self.edge_norm = torch.nn.LayerNorm(self.latent_dim)

        # init t_ij
        self.t_ij_cat_to_t_ij = latent(
            mlp_input_dimension=2*self.latent_dim + input_edge_irreps.dim,
            mlp_output_dimension=self.latent_dim,
        )

        # init X
        self.env_weighter = MakeWeightedChannels(irreps_in=spharms_irreps, multiplicity_out=eq_multiplicity)
        generate_n_weights = (self.env_weighter.weight_numel)

        self.t_ij_emb0 = latent(
            mlp_input_dimension=self.latent_dim,
            mlp_output_dimension=generate_n_weights,
        )

        self.h_j_emb0 = latent(
            mlp_input_dimension=self.latent_dim,
            mlp_output_dimension=generate_n_weights,
        )

        self.node_eq_norm = SO3_LayerNorm(node_eq_irreps)

        # layers
        self.layers = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                GotenInteractionLayer(
                    spharms_irreps,
                    eq_multiplicity,
                    head_dim,
                    latent_dim,
                    latent,
                )
            )
        
        # final
        self.linear = Linear(
            irreps_in=node_eq_irreps,
            irreps_out=out_irreps_node,
            internal_weights=False,
        )
        final_generate_n_weights = self.linear.weight_numel
        self.h_emb = latent(
            mlp_input_dimension=self.latent_dim,
            mlp_output_dimension=final_generate_n_weights,
        )

        self.t_ij_emb = latent(
            mlp_input_dimension=self.latent_dim,
            mlp_output_dimension=out_irreps_edge.dim,
        )

        self.reshape_node = inverse_reshape_irreps(out_irreps_node)

        # - End build modules - #
        
        self.irreps_out.update({
            self.out_field_node: out_irreps_node,
            self.out_field_edge: out_irreps_edge,
        })
    
    def init_features(self, data, edge_center, edge_neighbor, phi_ij, spharms):
        # node scalars
        node_attr = data[AtomicDataDict.NODE_ATTRS_KEY]
        num_nodes: int = node_attr.shape[0]
        
        proj_node = torch.einsum('ni,id -> nd', node_attr, self.W_node)
        proj_edge = proj_node[edge_neighbor]
        proj_radial = torch.einsum('ej,jd -> ed', phi_ij, self.W_edge)
        m_node = scatter(proj_edge * proj_radial, edge_center, dim=0, dim_size=num_nodes)
        proj_center_node = torch.einsum('ni,id -> nd', node_attr, self.W_center_node)
        h = self.node_norm(torch.einsum('nk,kd -> nd', torch.cat([proj_center_node, m_node], dim=-1), self.W_concat_node))

        # edge scalars
        h_i = h[edge_center]
        h_j = h[edge_neighbor]
        t_ij_cat = torch.cat([h_i, h_j, phi_ij], dim=-1)
        if AtomicDataDict.EDGE_FEATURES_KEY in data:
            t_ij_cat = torch.cat([t_ij_cat, data[AtomicDataDict.EDGE_FEATURES_KEY]], dim=-1)
        t_ij = self.edge_norm(self.t_ij_cat_to_t_ij(t_ij_cat))

        # node equivariants
        env_ij_scalar0 = self.t_ij_emb0(t_ij)
        env_j_scalar0  = self.h_j_emb0(h_j)
        env_ij_w0      = env_ij_scalar0 * env_j_scalar0
        X_ij           = self.env_weighter(spharms, env_ij_w0)
        X              = self.node_eq_norm(scatter(X_ij, edge_center, dim=0, dim_size=num_nodes))

        return num_nodes, h, X, t_ij

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center, edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_length       = data[AtomicDataDict.EDGE_LENGTH_KEY]
        phi_ij            = data[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY]
        spharms           = data[AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY]

        num_nodes, h, X, t_ij = self.init_features(data, edge_center, edge_neighbor, phi_ij, spharms)

        for layer in self.layers:
            h, X, t_ij = layer(h, X, t_ij, edge_center, edge_neighbor, phi_ij, spharms, num_nodes)

        # --- final layer --- #
        w = self.h_emb(h)
        data[self.out_field_node] = self.reshape_node(self.linear(X, w))
        data[self.out_field_edge] = self.t_ij_emb(t_ij)

        return data


@compile_mode("script")
class GotenInteractionLayer(torch.nn.Module):
    def __init__(
        self,
        spharms_irreps: o3.Irreps,
        eq_multiplicity: int,
        head_dim: int,
        latent_dim: int,
        latent,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # irreps
        eq_irreps = o3.Irreps([(eq_multiplicity, ir) for _, ir in spharms_irreps])

        # modules
        generate_n_weights = latent_dim

        # t_ij
        self.env_weighter = MakeWeightedChannels(irreps_in=spharms_irreps, multiplicity_out=eq_multiplicity)
        generate_n_weights += self.env_weighter.weight_numel

        # X
        self.linear = Linear(
            irreps_in=eq_irreps,
            irreps_out=eq_irreps,
            internal_weights=False,
        )
        generate_n_weights += self.linear.weight_numel
        self.node_eq_norm = SO3_LayerNorm(eq_irreps)

        # h
        self.W_rs = torch.nn.Parameter(torch.randn(latent_dim, generate_n_weights))
        self.h_j_emb = latent(
            mlp_input_dimension=latent_dim,
            mlp_output_dimension=generate_n_weights,
        )
        self.node_norm = torch.nn.LayerNorm(self.latent_dim)

        # attention
        self.W_query      = torch.nn.Parameter(torch.randn(latent_dim, eq_multiplicity * head_dim))
        self.W_key        = torch.nn.Parameter(torch.randn(latent_dim, eq_multiplicity * head_dim))
        self.rearrange_qk = Rearrange('e (m h) -> e m h', m=eq_multiplicity, h=head_dim)
        self.isqrtd       = math.isqrt(head_dim)

    def apply_attention(self, x, h_j, t_ij, edge_center) -> torch.Tensor:
        Q = torch.einsum('ed,dw -> ew', t_ij, self.W_query)
        Q = self.rearrange_qk(Q)

        K = torch.einsum('ed,dw -> ew', h_j , self.W_key)
        K = self.rearrange_qk(K)

        W = torch.einsum('emh,emh -> em', Q, K) * self.isqrtd
        return torch.einsum('emd,em->emd', x, scatter_softmax(W, edge_center, dim=0))

    def apply_mace(self, local_env_per_active_atom, node_invariants, active_node_centers) -> torch.Tensor:
        # Asserts needed for JIT
        assert self.product is not None
        assert self.reshape_in_module is not None
        expanded_features_per_active_atom: torch.Tensor = self.product(
            node_feats=local_env_per_active_atom,
            node_attrs=node_invariants[active_node_centers],
        )
        # updated local_env_per_active_atom
        return self.reshape_in_module(expanded_features_per_active_atom)

    def forward(self, h, X, t_ij, edge_center, edge_neighbor, phi_ij, spharms, num_nodes):
        h_j = h[edge_neighbor]

        env_ij_scalar = torch.einsum('ed,dw -> ew', t_ij, self.W_rs)
        env_j_scalar  = self.h_j_emb(h_j)
        env_ij_w      = env_ij_scalar * env_j_scalar

        # ...
        w_index = 0
        delta_h_j = env_ij_w.narrow(-1, w_index, self.latent_dim) # (dim, start, length)
        delta_h   = scatter(delta_h_j, edge_center, dim=0, dim_size=num_nodes)
        h         = self.node_norm(h + delta_h)
        # h_j = h[edge_neighbor] ? Optional. Check if it works better

        delta_spharm_ij_w = env_ij_w.narrow(-1, w_index, self.env_weighter.weight_numel) # (dim, start, length)
        w_index += self.env_weighter.weight_numel
        delta_spharm_ij = self.env_weighter(spharms, delta_spharm_ij_w)

        delta_X_ij_w = env_ij_w.narrow(-1, w_index, self.linear.weight_numel) # (dim, start, length)
        w_index += self.linear.weight_numel
        X_ij = X[edge_neighbor]
        delta_eq_ij = self.linear(X_ij, delta_X_ij_w)

        delta_X_ij = self.apply_attention(delta_spharm_ij + delta_eq_ij, h_j, t_ij, edge_center)
        delta_X = scatter(delta_X_ij, edge_center, dim=0, dim_size=num_nodes)
        X = self.node_eq_norm(X + delta_X)

        return h, X, t_ij # + delta_t_ij