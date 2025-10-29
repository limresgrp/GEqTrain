import math
import functools
import torch

from typing import Callable, List, Optional, Tuple, Union

from geqtrain.nn._embedding import BaseEmbedding
from geqtrain.utils._model_utils import process_out_irreps
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_softmax

from e3nn import o3
from e3nn.util.jit import compile_mode

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
from geqtrain.utils.so3 import PSEUDO_SCALAR, SCALAR
from geqtrain.nn.mace.irreps_tools import inverse_reshape_irreps


def apply_residual_stream(x, x_new, update_coeff):
    # At init, we assume new and old to be approximately uncorrelated
    # Thus their variances add
    # we always want the latent space to be normalized to variance = 1.0,
    # because it is critical for learnability. Still, we want to preserve
    # the _relative_ magnitudes of the current latent and the residual update
    # to be controled by `update_coeff`
    # Solving the simple system for the two coefficients:
    #   a^2 + b^2 = 1  (variances add)   &    a * update_coeff = b
    # gives
    #   a = 1 / sqrt(1 + update_coeff^2)  &  b = update_coeff / sqrt(1 + update_coeff^2)
    # rsqrt is reciprocal sqrt
    coefficient_old = torch.rsqrt(update_coeff.square() + 1)
    coefficient_new = update_coeff * coefficient_old
    # Residual update
    return coefficient_old * x + coefficient_new * x_new


# ==============================================================================
# 1. NODE EMBEDDING
# ==============================================================================
@compile_mode("script")
class GotenNodeEmbedding(BaseEmbedding):
    """
    Initializes scalar node features based on atom types and their local environment.
    This version removes the final LayerNorm, deferring normalization to subsequent layers.
    """
    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        node_irreps = self.irreps_in[self.node_field]
        node_dim = node_irreps.dim
        edge_attr_dim = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY].dim
        
        self.W_neighbor_embed = torch.nn.Parameter(torch.randn(node_dim, latent_dim))
        self.W_center_embed = torch.nn.Parameter(torch.randn(node_dim, latent_dim))
        self.W_radial_embed = torch.nn.Parameter(torch.randn(edge_attr_dim, latent_dim))
        self.W_update = torch.nn.Parameter(torch.randn(2 * latent_dim, latent_dim))
        self.norm = torch.nn.LayerNorm(latent_dim)

        self.out_irreps = o3.Irreps(f'{latent_dim}x0e')

    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_attr = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        
        node_field = self.node_field
        assert isinstance(node_field, str)
        node_attr = data[node_field]
        num_nodes = len(node_attr)

        neighbor_embeddings = torch.einsum('ni,id -> nd', node_attr, self.W_neighbor_embed)
        radial_features = torch.einsum('ej,jd -> ed', edge_attr, self.W_radial_embed)
        messages_source_features = neighbor_embeddings[edge_neighbor]
        messages = messages_source_features * radial_features
        aggregated_messages = scatter_sum(messages, edge_center, dim=0, dim_size=num_nodes)
        center_node_features = torch.einsum('ni,id -> nd', node_attr, self.W_center_embed)
        concatenated_features = torch.cat([center_node_features, aggregated_messages], dim=-1)
    
        # Apply final update and layer normalization
        updated_features = torch.einsum('nk,kd -> nd', concatenated_features, self.W_update)
        return self.norm(updated_features)

# ==============================================================================
# 2. EDGE EMBEDDING
# ==============================================================================
@compile_mode("script")
class GotenEdgeEmbedding(BaseEmbedding):
    """
    Initializes scalar edge features (`t_ij`) based on node features and radial embeddings.
    This version implements the original GotenNet EdgeInit logic: (h_i + h_j) * W_erp(phi_ij).
    Inherits from BaseEmbedding to be compatible with the EmbeddingAttrs class.
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # We expect self.node_field to be populated by node embedding
        # and EDGE_SPHARMS_EMB_KEY to be populated by BasisEdgeRadialAttrs
        node_irreps = self.irreps_in[self.node_field]
        input_edge_irreps = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        
        # Linear layer to project radial basis functions, similar to W_erp in the paper.
        self.W_erp = torch.nn.Linear(input_edge_irreps.dim, node_irreps.dim)
        
        self.out_irreps = node_irreps
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_erp.weight)

    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_center   = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        phi_ij        = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        h             = data[AtomicDataDict.NODE_ATTRS_KEY]

        # Gather features from center and neighbor nodes
        h_i = h[edge_center]
        h_j = h[edge_neighbor]
        
        # Project the radial basis functions
        radial_proj = self.W_erp(phi_ij)
        
        # Compute t_ij by combining node and edge features
        t_ij = (h_i + h_j) * radial_proj
        
        # Return the tensor directly, as per BaseEmbedding contract
        return t_ij

# ==============================================================================
# 3. INTERACTION MODULE
# ==============================================================================

@compile_mode("script")
class GotenInteractionModule(GraphModuleMixin, torch.nn.Module):
    """
    Refactored GotenNet interaction module.
    Initializes steerable features (X) and then performs iterative updates.
    """
    def __init__(
        self,
        num_layers: int,
        out_irreps_node: Optional[Union[o3.Irreps, str]] = None,
        output_ls:  Optional[List[int]] = None,
        output_mul: Optional[Union[str, int]] = None,
        out_field_node: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field_edge: str = AtomicDataDict.EDGE_FEATURES_KEY,
        latent_dim: int = 64,
        eq_multiplicity: int = 8,
        use_attention: bool = False,
        head_dim: int = 16,
        latent: Callable = ScalarMLPFunction,
        latent_kwargs: dict = {},
        irreps_in=None,
    ):
        super().__init__()
        assert num_layers >= 1

        self.out_field_node = out_field_node
        self.out_field_edge = out_field_edge
        self.latent_dim = latent_dim

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_ATTRS_KEY,      # 'h' from embeddings
                AtomicDataDict.EDGE_ATTRS_KEY,      # 't_ij' from embeddings
                AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
            ]
        )
        
        latent = functools.partial(latent, **latent_kwargs)
        spharms_irreps = self.irreps_in[AtomicDataDict.EDGE_SPHARMS_EMB_KEY]
        assert all(mul == 1 for mul, _ in spharms_irreps)
        
        node_eq_irreps = o3.Irreps([(eq_multiplicity, ir) for _, ir in spharms_irreps])
        assert node_eq_irreps[0].ir == SCALAR, "node_eq_irreps must start with scalars"

        # === Process output irreps using the same logic as InteractionModule ===
        out_irreps_node = process_out_irreps(
            out_irreps=out_irreps_node,
            output_ls=output_ls,
            output_mul=output_mul,
            default_irreps=node_eq_irreps,
        )
        self.out_multiplicity = out_irreps_node[0].mul
        self.out_feat_elems = sum(irr.ir.dim for irr in out_irreps_node)
        
        # Split out_irreps into scalar and equivariant parts
        self.has_scalar_output = any(ir.l == 0 for _, ir in out_irreps_node)
        self.has_equivariant_output = any(ir.l > 0 for _, ir in out_irreps_node)
        self.out_n_scalars = 0
        if self.has_scalar_output:
            self.out_n_scalars = (out_irreps_node.count(SCALAR) + out_irreps_node.count(PSEUDO_SCALAR)) // self.out_multiplicity

        eq_out_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps_node if ir.l > 0])
        # === End of irreps processing ===

        # Define projection layers to map initial h and t_ij to latent_dim.
        # This allows embedding layers to have variable output dimensions.
        node_attr_dim = self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY].dim
        edge_attr_dim = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY].dim
        
        self.h_proj = latent(mlp_input_dimension=node_attr_dim, mlp_output_dimension=self.latent_dim)
        self.t_ij_proj = latent(mlp_input_dimension=edge_attr_dim, mlp_output_dimension=self.latent_dim)

        # === Initialize weights for X (steerable features) ===
        self.env_weighter = MakeWeightedChannels(irreps_in=spharms_irreps, multiplicity_out=eq_multiplicity)
        generate_n_weights = self.env_weighter.weight_numel
        # The MLPs below take latent_dim as input
        self.t_ij_to_weights_init = latent(mlp_input_dimension=self.latent_dim, mlp_output_dimension=generate_n_weights)
        self.h_j_to_weights_init = latent(mlp_input_dimension=self.latent_dim, mlp_output_dimension=generate_n_weights)
        self.init_weights_norm = torch.nn.LayerNorm(generate_n_weights)
        self.node_eq_norm = SO3_LayerNorm(node_eq_irreps)

        # === Interaction Layers ===
        self.layers = torch.nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(GotenInteractionLayer(spharms_irreps, eq_multiplicity, use_attention, head_dim, latent_dim, latent))
        self.update_coeffs = torch.nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))

        # === Final Projection ===
        self.final_linear, self.eq_w_mlp, self.final_scalar_mlp, self.final_norm = None, None, None, None
        
        # Projection for equivariant part (l>0)
        if self.has_equivariant_output:
            self.final_linear = Linear(irreps_in=node_eq_irreps, irreps_out=eq_out_irreps, internal_weights=False)
            self.eq_w_mlp = latent(mlp_input_dimension=self.latent_dim, mlp_output_dimension=self.final_linear.weight_numel)
            self.final_norm = SO3_LayerNorm(eq_out_irreps)

        # Projection for scalar part (l=0)
        if self.has_scalar_output:
            self.final_scalar_mlp = latent(
                mlp_input_dimension=self.latent_dim,
                mlp_output_dimension=self.out_multiplicity * self.out_n_scalars,
            )

        self.reshape_node = inverse_reshape_irreps(out_irreps_node)
        self.irreps_out.update({self.out_field_node: out_irreps_node})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center   = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        spharms       = data[AtomicDataDict.EDGE_SPHARMS_EMB_KEY]
        h             = data[AtomicDataDict.NODE_ATTRS_KEY]
        t_ij          = data[AtomicDataDict.EDGE_ATTRS_KEY]
        num_nodes: int = h.shape[0]

        # Apply the initial projection to ensure h and t_ij have the correct
        # latent_dim before entering the first interaction layer.
        h = self.h_proj(h)
        t_ij = self.t_ij_proj(t_ij)

        # === Initialize X (Steerable Features) using h and t_ij ===
        h_j = h[edge_neighbor]
        # Now w_from_t and w_from_h correctly receive tensors of shape latent_dim
        w_from_t = self.t_ij_to_weights_init(t_ij)
        w_from_h = self.h_j_to_weights_init(h_j)
        env_weights = self.init_weights_norm(w_from_t * w_from_h)
        X_ij = self.env_weighter(spharms, env_weights)
        X = self.node_eq_norm(scatter_sum(X_ij, edge_center, dim=0, dim_size=num_nodes) + 1.e-10)

        # === Interaction Loop ===
        update_coeffs = torch.sigmoid(self.update_coeffs)
        for i, layer in enumerate(self.layers):
            # Pass the single update_coeff for this layer
            h, X, t_ij = layer(h, X, t_ij, edge_center, edge_neighbor, spharms, update_coeffs[i], num_nodes)

        # === Final Projection ===
        # Initialize an empty tensor to store the final combined output features
        final_output = torch.zeros(
            (num_nodes, self.out_multiplicity, self.out_feat_elems),
            dtype=h.dtype,
            device=h.device
        )

        # 1. Compute and place equivariant features (l>0)
        if self.has_equivariant_output:
            assert self.final_linear is not None and self.eq_w_mlp is not None and self.final_norm is not None
            eq_w = self.eq_w_mlp(h)
            eq_features = self.final_linear(X, eq_w)
            eq_features = self.final_norm(eq_features)
            # Place into the latter part of the output tensor
            final_output[..., self.out_n_scalars:] = eq_features

        # 2. Compute and place scalar features (l=0)
        if self.has_scalar_output:
            assert self.final_scalar_mlp is not None
            scalar_features = self.final_scalar_mlp(h)
            # Place into the beginning of the output tensor
            final_output[..., :self.out_n_scalars] = scalar_features.unsqueeze(-1)
        
        # Reshape and store the final result
        data[self.out_field_node] = self.reshape_node(final_output)
        data[self.out_field_edge] = t_ij

        return data

# ==============================================================================
# INTERACTION LAYER
# ==============================================================================
@compile_mode("script")
class GotenInteractionLayer(torch.nn.Module):
    def __init__(
        self,
        spharms_irreps: o3.Irreps,
        eq_multiplicity: int,
        use_attention: bool,
        head_dim: int,
        latent_dim: int,
        latent,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.num_heads = latent_dim // head_dim
        assert self.num_heads * head_dim == latent_dim, "latent_dim must be divisible by head_dim"

        eq_irreps = o3.Irreps([(eq_multiplicity, ir) for _, ir in spharms_irreps])

        # === Weights for Spatial Filter Message ===
        self.W_rs = torch.nn.Linear(latent_dim, latent_dim)
        self.gamma_s = latent(mlp_input_dimension=latent_dim, mlp_output_dimension=latent_dim)

        # === Weights for Attention Message (sea_ij) ===
        if self.use_attention:
            self.W_q = torch.nn.Linear(latent_dim, latent_dim)
            self.W_k = torch.nn.Linear(latent_dim, latent_dim)
            self.W_re = torch.nn.Linear(latent_dim, latent_dim, bias=False)
            self.gamma_v = latent(mlp_input_dimension=latent_dim, mlp_output_dimension=latent_dim)

        # MLP to expand the combined message before splitting
        self.gamma_split = latent(
            mlp_input_dimension=latent_dim, 
            mlp_output_dimension=3 * latent_dim
        )
        # MLPs to map scalar gates to the multiplicity dimension of equivariant features
        self.mlp_d = latent(mlp_input_dimension=latent_dim, mlp_output_dimension=eq_multiplicity)
        self.mlp_t = latent(mlp_input_dimension=latent_dim, mlp_output_dimension=eq_multiplicity)

        # === Weights for Combined Message Processing ===
        self.env_weighter = MakeWeightedChannels(irreps_in=spharms_irreps, multiplicity_out=eq_multiplicity)
        self.linear_X = Linear(irreps_in=eq_irreps, irreps_out=eq_irreps, internal_weights=False)
        self.message_to_weights = latent(
            mlp_input_dimension=latent_dim, 
            mlp_output_dimension=latent_dim + self.env_weighter.weight_numel + self.linear_X.weight_numel
        )
        
        # === Normalization Layers ===
        self.node_norm = torch.nn.LayerNorm(self.latent_dim)
        self.node_eq_norm = SO3_LayerNorm(eq_irreps)

        # === Weights for Hierarchical Tensor Refinement (HTR) ===
        self.W_to_query = Linear(irreps_in=eq_irreps, irreps_out=eq_irreps, internal_weights=True, shared_weights=True)
        self.W_to_key = Linear(irreps_in=eq_irreps, irreps_out=eq_irreps, internal_weights=True, shared_weights=True)
        instr = [(i_1, i_2, i_1) for i_1, (_, ir_1) in enumerate(eq_irreps) for i_2, (_, ir_2) in enumerate(eq_irreps) if SCALAR in ir_1 * ir_2]
        tp_out_irreps = o3.Irreps([(eq_multiplicity, SCALAR) for _ in instr])
        self.tp = Contracter(irreps_in1=eq_irreps, irreps_in2=eq_irreps, irreps_out=tp_out_irreps, instructions=instr, connection_mode="uuu", has_weight=False)
        self.gamma_w = latent(mlp_input_dimension=tp_out_irreps.dim, mlp_output_dimension=latent_dim)
        self.gamma_t = latent(mlp_input_dimension=latent_dim, mlp_output_dimension=latent_dim)

        # === EQFF Layer ===
        self.eqff = EQFF(latent_dim=latent_dim, eq_multiplicity=eq_multiplicity, lmax=max(eq_irreps.ls), latent=latent)
    
    @staticmethod
    def vector_rejection(features: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """
        Computes the component of `features` orthogonal to `directions`.
        It operates on tensors with shape [Edges, Multiplicity, Irrep_Dim].
        """
        # directions shape: [E, D_geom], features shape: [E, M, D_geom]
        # Unsqueeze directions to allow broadcasting over the multiplicity dimension
        directions_unsqueezed = directions.unsqueeze(1)  # -> [E, 1, D_geom]
        
        # Scalar projection (dot product)
        # (features * directions) -> [E, M, D_geom]
        # .sum() -> [E, M, 1]
        proj_scalar = (features * directions_unsqueezed).sum(dim=-1, keepdim=True)
        
        # Subtract the vector projection to get the rejection
        return features - proj_scalar * directions_unsqueezed

    def forward(self, h, X, t_ij, edge_center, edge_neighbor, spharms, update_coeff, num_nodes: int):
        
        # === 1. HTR: Update edge scalars t_ij first, now with vector rejection ===
        X_q_full = self.W_to_query(X[edge_center])
        X_k_full = self.W_to_key(X[edge_neighbor])

        # Apply vector rejection before the tensor product
        X_q = self.vector_rejection(X_q_full, spharms)
        X_k = self.vector_rejection(X_k_full, -spharms) # Use negative direction for neighbor

        w_ij = self.tp(X_q, X_k).reshape(len(X_q), -1)
        delta_t_ij = self.gamma_w(w_ij) * self.gamma_t(t_ij)
        t_ij = apply_residual_stream(t_ij, delta_t_ij, update_coeff)
        
        # === 2. GATA: Create and combine messages ===
        h_i = h[edge_center]
        h_j = h[edge_neighbor]
        
        spatial_message = self.W_rs(t_ij) * self.gamma_s(h_j)
        
        if self.use_attention:
            q_i = self.W_q(h_i).view(-1, self.num_heads, self.head_dim)
            k_j = self.W_k(h_j).view(-1, self.num_heads, self.head_dim)
            v_j = self.gamma_v(h_j)
            
            t_ij_attn = torch.nn.functional.silu(self.W_re(t_ij))
            t_ij_attn = t_ij_attn.view(-1, self.num_heads, self.head_dim)
            
            alpha_ij = (q_i * k_j * t_ij_attn).sum(dim=-1) / math.sqrt(self.head_dim)
            attn_weights = scatter_softmax(alpha_ij, edge_center, dim=0)
            
            sea_ij = (attn_weights.view(-1, self.num_heads, 1) * v_j.view(-1, self.num_heads, self.head_dim)).view(-1, self.latent_dim)
            
            combined_message = sea_ij + spatial_message
        else:
            combined_message = spatial_message
            
        # === 3. Update h and X using original "split-and-gate" GATA logic ===
        
        # Expand the message, then split into direct gates for h, spharms, and X
        gates = self.gamma_split(combined_message)
        o_s, o_d, o_t = torch.split(gates, self.latent_dim, dim=-1)

        # Update h using the scalar gate o_s
        delta_h = self.node_norm(scatter_sum(o_s, edge_center, dim=0, dim_size=num_nodes))
        h = apply_residual_stream(h, delta_h, update_coeff)

        # Map scalar gates o_d and o_t to the multiplicity dimension
        gate_d = self.mlp_d(o_d) # -> [E, M]
        gate_t = self.mlp_t(o_t) # -> [E, M]
        
        # Apply gates to update X
        # gate.unsqueeze(-1) gives shape [E, M, 1] for broadcasting
        # spharms.unsqueeze(1) gives shape [E, 1, D_geom] for broadcasting
        delta_X_d_ij = gate_d.unsqueeze(-1) * spharms.unsqueeze(1)
        delta_X_t_ij = gate_t.unsqueeze(-1) * X[edge_neighbor]
        
        delta_X_ij = delta_X_d_ij + delta_X_t_ij
        delta_X = self.node_eq_norm(scatter_sum(delta_X_ij, edge_center, dim=0, dim_size=num_nodes))
        X = apply_residual_stream(X, delta_X, update_coeff)

        # === 4. EQFF: Final refinement ===
        h, X = self.eqff(h, X)
        
        return h, X, t_ij

@compile_mode("script")
class EQFF(torch.nn.Module):
    """
    Equivariant Feed-Forward (EQFF) Network for mixing atom features.
    Correctly handles channel mixing for steerable features.
    """
    def __init__(self, latent_dim: int, eq_multiplicity: int, lmax: int, latent: Callable, epsilon: float = 1e-8):
        super(EQFF, self).__init__()
        self.lmax = lmax
        self.latent_dim = latent_dim
        self.eq_multiplicity = eq_multiplicity
        self.epsilon = epsilon
        
        # MLP to generate gates from scalar features (h) and the norm of equivariant features (X)
        self.gamma_m = latent(
            mlp_input_dimension=latent_dim + eq_multiplicity, 
            mlp_output_dimension=latent_dim + eq_multiplicity
        )
        
        # Linear layer for channel mixing of equivariant features
        self.W_vu = torch.nn.Parameter(torch.randn(eq_multiplicity, eq_multiplicity) / math.sqrt(eq_multiplicity))

    def forward(self, h: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X has shape [n_nodes, eq_multiplicity, total_l_dim]
        
        # 1. Apply channel-mixing linear layer to X
        # 'nmi,mj -> nji' -> for each node, mix the multiplicity channels
        X_p = torch.einsum('nmi,mj -> nji', X, self.W_vu)
        
        # 2. Compute invariant norm over the geometric dimension
        # Result shape: [n_nodes, eq_multiplicity]
        X_pn_norm = torch.linalg.vector_norm(X_p, dim=-1, keepdim=False)
        
        # 3. Create context vector for the MLP
        # Concatenate scalar features h with the norm of X
        ctx = torch.cat([h, X_pn_norm], dim=-1)
        
        # 4. Generate gates m1 and m2
        gates = self.gamma_m(ctx)
        m1, m2 = torch.split(gates, [self.latent_dim, self.eq_multiplicity], dim=-1)
        
        # 5. Apply gates to update h and X
        # Update h with m1
        h = h + m1
        
        # Update X with m2. Unsqueeze m2 to broadcast over the geometric dimension.
        # m2 shape: [n_nodes, eq_multiplicity] -> [n_nodes, eq_multiplicity, 1]
        dX_intra = m2.unsqueeze(-1) * X_p
        X = X + dX_intra
        
        return h, X