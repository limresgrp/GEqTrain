import math
import torch
import wandb

from typing import Any, Dict, Optional, List, Tuple, Union
from geqtrain.utils._model_utils import process_out_irreps
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_softmax
from einops.layers.torch import Rearrange

from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import (
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
    _EXTRA_FIELDS,
    _FIXED_FIELDS,
)
from geqtrain.nn import (
    GraphModuleMixin,
    SO3_LayerNorm,
    ScalarMLPFunction,
)
from geqtrain.nn.allegro import (
    Contracter,
    MakeWeightedChannels,
)
from geqtrain.utils.tp_utils import PSEUDO_SCALAR, SCALAR, tp_path_exists
from geqtrain.nn._equivariant_scalar_mlp import EquivariantScalarMLP
from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps

# Helper functions
def log_feature_on_wandb(name: str, t: torch.Tensor, train: bool):
    if not torch.jit.is_scripting() and wandb.run is not None:
        s = "train" if train else "eval"
        try:
            wandb.log({
                f"activations_dists/{s}/{name}.mean": t.mean().item(),
                f"activations_dists/{s}/{name}.std":  t.std().item(),
            })
        except RuntimeError as e:
            print(f"[WandB log error] Skipped logging {name}: {e}")

def prepare_conditioning_tensors(
    data: AtomicDataDict.Type,
    conditioning_fields: List[str],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepares conditioning tensors for node and edge operations.

    Args:
        data (AtomicDataDict.Type): The input data dictionary.
        conditioning_fields (List[str]): A list of field names to use for conditioning.
        node_fields (List[str]): A list of keys that are considered node fields.
        edge_fields (List[str]): A list of keys that are considered edge fields.

    Returns:
        A tuple containing:
        - node_conditioning_tensor (Optional[torch.Tensor]): Concatenated tensor of node-level conditioning fields.
        - edge_conditioning_tensor (Optional[torch.Tensor]): Concatenated tensor of edge-level conditioning fields.
    """
    node_cond_tensors = []
    edge_cond_tensors = []

    if not conditioning_fields:
        return None, None

    edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY][1]
    num_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    num_edges = edge_center.shape[0]
    radial_emb = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
    edge_cond_tensors.append(radial_emb) # edge conditioning fields need to have information on radial distance

    for field in conditioning_fields:
        # Not jittable if uncommented
        # if field not in irreps_in:
        #     raise ValueError(f"Conditioning field '{field}' not found in irreps_in.")
        
        # cond_irreps = irreps_in[field]
        # if not all(ir.l == 0 for _, ir in cond_irreps):
        #     raise ValueError(f"Conditioning field '{field}' must have scalar (l=0) irreps, but got {cond_irreps}.")

        tensor = data[field]

        if len(tensor) == num_nodes:
            node_cond_tensors.append(tensor)
            edge_cond_tensors.append(tensor[edge_center])
            edge_cond_tensors.append(tensor[edge_neigh])
        elif len(tensor) == num_edges:
            edge_cond_tensors.append(tensor)

    node_conditioning = torch.cat(node_cond_tensors, dim=-1) if node_cond_tensors else None
    edge_conditioning = torch.cat(edge_cond_tensors, dim=-1) if edge_cond_tensors else None
    
    return node_conditioning, edge_conditioning

def apply_residual_stream(latents, new_latents, this_layer_update_coeff: Optional[torch.Tensor], active_edges):
    if this_layer_update_coeff is None:
        # This happens on the first layer, where the dimensions of `latents` (from initial edge attributes)
        # and `new_latents` (from the first MLP) are different. We just take the new latents.
        return new_latents
    
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
    
    # We update only the active edges.
    # To do this, we scale the whole latent tensor, then add the new latents only for the active edges.
    # This is more efficient than creating a zero tensor and using index_add for the old latents.
    updated_latents = coefficient_old * latents
    updated_latents.index_add_(0, active_edges, coefficient_new * new_latents)
    return updated_latents

def build_tps_irreps_list(num_layers: int, eq_latent_irreps: o3.Irreps, final_out_irreps: o3.Irreps,) -> Tuple[List[o3.Irreps], List[o3.Irreps]]:
        """Pre-computes the irreps for each layer's tensor product."""
        
        # Build up the irreps for the iterated TPs layer by layer
        tps_irreps = [eq_latent_irreps]
        arg_irreps = eq_latent_irreps
        for i in range(num_layers):
            ir_out = final_out_irreps if i == num_layers - 1 else eq_latent_irreps
            # Prune impossible paths
            ir_out = o3.Irreps([(mul, ir) for mul, ir in ir_out if tp_path_exists(arg_irreps, eq_latent_irreps, ir)])
            arg_irreps = ir_out
            tps_irreps.append(ir_out)
            
        # Prune unneeded paths backwards from the output
        new_tps_irreps = [tps_irreps[-1]]
        temp_out_irreps = tps_irreps[-1]
        for current_arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in current_arg_irreps:
                # Check if this irrep `arg_ir` can produce any of the irreps `temp_out_irreps`
                # that we need for the next step in the backward pass.
                if any(out_ir in arg_ir * env_ir for _, env_ir in eq_latent_irreps for _, out_ir in temp_out_irreps):
                    new_arg_irreps.append((mul, arg_ir))
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            temp_out_irreps = new_arg_irreps
        
        tps_irreps = list(reversed(new_tps_irreps))
        
        return tps_irreps[:-1], tps_irreps[1:]


@compile_mode("script")
class InteractionModule(GraphModuleMixin, torch.nn.Module):
    conditioning_fields: List[str]

    def __init__(
        self,
        num_layers: int,
        out_irreps: Optional[Union[o3.Irreps, str]] = None,
        output_ls: Optional[List[int]] = None,
        output_mul: Optional[Union[str, int]] = None,
        node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
        edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
        edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
        out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        latent_dim: int = 256,
        eq_latent_multiplicity: int = 16,
        use_attention: bool = False,
        head_dim: int = 16,
        use_mace_product: bool = False,
        product_correlation: int = 2,
        irreps_in = None,
        conditioning_fields: Optional[List[str]] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_invariant_field = node_invariant_field
        self.edge_invariant_field = edge_invariant_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field = out_field
        self.latent_dim = latent_dim
        self.conditioning_fields = conditioning_fields if conditioning_fields is not None else []
        self.debug = debug

        # --- Irreps Initialization and Validation ---
        required_irreps = [self.node_invariant_field, self.edge_invariant_field, self.edge_equivariant_field]
        required_irreps.extend(self.conditioning_fields)
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=required_irreps
        )

        # --- Conditioning Tensor Dimension Calculation ---
        self.node_conditioning_dim = 0
        self.edge_conditioning_dim = 0
        for field in self.conditioning_fields:
            dim = self.irreps_in[field].dim
            if field in _NODE_FIELDS:
                self.node_conditioning_dim += dim
                self.edge_conditioning_dim += 2 * dim
            elif field in _EDGE_FIELDS:
                self.edge_conditioning_dim += dim

        # Keep only l>0 from the input equivariant field
        full_input_edge_eq_irreps = self.irreps_in[self.edge_equivariant_field]
        self.input_edge_eq_l0_dim = sum(mul * ir.dim for mul, ir in full_input_edge_eq_irreps if ir.l == 0)
        input_edge_eq_irreps_l_gt_0 = o3.Irreps([(mul, ir) for mul, ir in full_input_edge_eq_irreps if ir.l > 0])

        eq_latent_irreps = o3.Irreps([(eq_latent_multiplicity, ir) for _, ir in input_edge_eq_irreps_l_gt_0])
        eq_only_latent_irreps = o3.Irreps([(mul, ir) for mul, ir in eq_latent_irreps if ir.l > 0])
        # Since we only use l>0 from input, the latent space will also only have l>0
        assert all(ir.l > 0 for _, ir in eq_latent_irreps), "eq_latent_irreps should only contain l>0 components"

        final_out_irreps, _, _ = process_out_irreps(
            out_irreps=out_irreps, output_ls=output_ls, output_mul=output_mul,
            latent_dim=latent_dim, edge_attrs_irreps=full_input_edge_eq_irreps,
        )
        combined_edge_inv_dim = 2 * self.irreps_in[self.node_invariant_field].dim + self.irreps_in[self.edge_invariant_field].dim

        tp_out_irreps = o3.Irreps([(eq_latent_multiplicity, ir) for _, ir in final_out_irreps])
        tps_irreps_in, tps_irreps_out = build_tps_irreps_list(num_layers, eq_latent_irreps, tp_out_irreps)
        self.env_weighter = MakeWeightedChannels(irreps_in=input_edge_eq_irreps_l_gt_0, multiplicity_out=eq_latent_multiplicity)

        # === Initial Equivariant Latent Embedding ===
        self.eq_latent_init_mlp = ScalarMLPFunction(combined_edge_inv_dim, [], self.env_weighter.weight_numel)
        
        self._latent_resnet_update_params = torch.nn.Parameter(torch.zeros(self.num_layers - 1))
        self.interaction_layers = torch.nn.ModuleList()
        final_latent_dim = self.latent_dim
        
        for i in range(self.num_layers):
            is_last_layer = (i == self.num_layers - 1)
            
            # The first layer's scalar latent MLP takes the initial edge attributes as input.
            # Subsequent layers take the output of the previous layer.
            latent_in_dim = combined_edge_inv_dim if i == 0 else self.latent_dim

            layer = InteractionLayer(
                latent_dim=self.latent_dim,
                latent_in_dim=latent_in_dim,
                eq_latent_multiplicity=eq_latent_multiplicity,
                use_attention=use_attention,
                head_dim=head_dim,
                use_mace_product=use_mace_product,
                product_correlation=product_correlation,
                tps_irreps_in=tps_irreps_in[i],
                tps_irreps_out=tps_irreps_out[i],
                eq_latent_irreps=eq_only_latent_irreps,
                linear_out_irreps=final_out_irreps if is_last_layer else eq_only_latent_irreps,
                irreps_in=self.irreps_in,
                node_invariant_field=self.node_invariant_field,
                edge_invariant_field=self.edge_invariant_field,
                edge_equivariant_field=self.edge_equivariant_field,
                env_weighter=self.env_weighter,
                is_last_layer=is_last_layer,
                node_conditioning_dim=self.node_conditioning_dim,
                edge_conditioning_dim=self.edge_conditioning_dim,
                combined_edge_inv_dim=combined_edge_inv_dim,
                debug=self.debug,
            )
            self.interaction_layers.append(layer)
            final_latent_dim += eq_latent_multiplicity * layer.tp_n_scalar_out
        
        self.irreps_out[self.out_field] = final_out_irreps

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        active_edges = torch.arange(data[self.edge_invariant_field].shape[0], device=data[AtomicDataDict.EDGE_INDEX_KEY].device)
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Pre-compute combined edge invariants since they are static across layers
        node_invariants = data[self.node_invariant_field]
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_inv_features = data[self.edge_invariant_field]
        edge_attrs = torch.cat([node_invariants[edge_center], node_invariants[edge_neigh], edge_inv_features], dim=-1)

        # Prepare conditioning tensors
        node_conditioning, edge_conditioning = prepare_conditioning_tensors(
            data=data,
            conditioning_fields=self.conditioning_fields,
        )

        # === Initialize Latents ===
        # 1. Scalar latent is set by first mlp
        scalar_latent = edge_attrs
        # 2. Equivariant latent is initialized from spherical harmonics
        # We use only the l>0 part of the edge equivariant field
        edge_equivariant_features_l_gt_0 = data[self.edge_equivariant_field].narrow(-1, self.input_edge_eq_l0_dim, data[self.edge_equivariant_field].shape[-1] - self.input_edge_eq_l0_dim)
        init_eq_weights = self.eq_latent_init_mlp(edge_attrs)
        eq_latent = self.env_weighter(edge_equivariant_features_l_gt_0, init_eq_weights)

        for i, layer in enumerate(self.interaction_layers):
            scalar_latent, eq_latent = layer(
                data=data,
                edge_equivariant_features=edge_equivariant_features_l_gt_0,
                scalar_latent=scalar_latent,
                eq_latent=eq_latent,
                edge_attrs=edge_attrs,
                active_edges=active_edges,
                node_conditioning=node_conditioning,
                edge_conditioning=edge_conditioning,
                this_layer_update_coeff=layer_update_coefficients[i - 1] if i > 0 else None,
            )

            if self.debug and wandb.run is not None:
                log_feature_on_wandb(f"InteractionLayer/{i}.scalar_latent", scalar_latent, self.training)
                log_feature_on_wandb(f"InteractionLayer/{i}.eq_latent", eq_latent, self.training)

        if eq_latent is not None:
            final_output_features = torch.cat([scalar_latent, eq_latent], dim=-1)
        else:
            final_output_features = scalar_latent
        data[self.out_field] = final_output_features

        if self.debug and wandb.run is not None:
            log_feature_on_wandb(f"{str(self)}.out_features.scalar", scalar_latent, self.training)
            if eq_latent is not None: log_feature_on_wandb(f"{self.name}.out_features.vectorial", eq_latent, self.training)
        
        return data


@compile_mode("script")
class InteractionLayer(torch.nn.Module):
    """
    Refactored InteractionLayer.
    - Decoupled from InteractionModule; all dependencies are injected.
    - `__init__` is streamlined and groups related module constructions.
    - Forward pass is clarified and returns the correct signature.
    """
    def __init__(
        self,
        latent_dim: int,
        latent_in_dim: int,
        eq_latent_multiplicity: int,
        head_dim: int,
        use_attention: bool,
        use_mace_product: bool,
        product_correlation: int,
        tps_irreps_in: o3.Irreps,
        tps_irreps_out: o3.Irreps,
        eq_latent_irreps: o3.Irreps,
        linear_out_irreps: o3.Irreps,
        irreps_in: dict,
        node_invariant_field: str,
        edge_invariant_field: str,
        edge_equivariant_field: str,
        combined_edge_inv_dim: int,
        env_weighter: MakeWeightedChannels,
        node_conditioning_dim: int,
        edge_conditioning_dim: int,
        is_last_layer: bool,
        debug: bool,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.eq_latent_multiplicity = eq_latent_multiplicity
        self.use_attention = use_attention
        self.use_mace_product = use_mace_product
        self.debug = debug
        self._env_weighter = env_weighter
        self.node_invariant_field = node_invariant_field
        self.edge_equivariant_field = edge_equivariant_field
        self.combined_edge_inv_dim = combined_edge_inv_dim

        # === Define dimensions and irreps ===
        node_inv_dim = irreps_in[node_invariant_field].dim
        edge_inv_dim = irreps_in[edge_invariant_field].dim
        
        edge_mlp_kwargs = {"conditioning_dim": edge_conditioning_dim, 'latent_kwargs': {'mlp_latent_dimensions': [128, 128]}}
        node_mlp_kwargs = {"conditioning_dim": node_conditioning_dim, 'latent_kwargs': {'mlp_latent_dimensions': [128, 128]}}

        # === Scalar latent processing modules ===
        self.latent_mlp = EquivariantScalarMLP(in_dim=latent_in_dim, out_dim=self.latent_dim, **edge_mlp_kwargs)
        self.latent_mlp_norm = torch.nn.LayerNorm(self.latent_dim)

        # === Environment embedding modules ===
        # MLP to generate weights for env_weighter
        self.env_embed_mlp = ScalarMLPFunction(self.latent_dim, [], env_weighter.weight_numel)

        self.node_env_norm = SO3_LayerNorm(eq_latent_irreps)
        self.env_linear = EquivariantScalarMLP(
            in_irreps=eq_latent_irreps,
            out_irreps=eq_latent_irreps,
            reshape_in=False,
            reshape_back=False,
            **node_mlp_kwargs
        )

        # === Attention modules ===
        if self.use_attention:
            self.isqrtd = math.isqrt(head_dim)
            # Use combined_edge_inv_dim for the MLP that processes original edge attributes
            self.edge_attr_to_query = ScalarMLPFunction(self.combined_edge_inv_dim, [], eq_latent_multiplicity * head_dim)
            self.latent_to_key = ScalarMLPFunction(self.latent_dim, [], eq_latent_multiplicity * head_dim)
            self.rearrange_qk = Rearrange('e (m d) -> e m d', m=eq_latent_multiplicity, d=head_dim)
        
        # === MACE product modules ===
        self.reshape_in_module = None
        if self.use_mace_product:
            self.product = EquivariantProductBasisBlock(
                node_feats_irreps=eq_latent_irreps,
                target_irreps=eq_latent_irreps,
                correlation=product_correlation,
                num_elements=node_inv_dim,
            )
            self.reshape_in_module = reshape_irreps(eq_latent_irreps)

        # === Tensor Product modules ===
        self.edge_env_norm = SO3_LayerNorm(eq_latent_irreps)
        self.eq_features_norm = SO3_LayerNorm(eq_latent_irreps)

        self.tp, self.tp_n_scalar_out, full_out_irreps = self._build_tp(tps_irreps_in, tps_irreps_out, eq_latent_irreps)
        self.tp_norm = SO3_LayerNorm(full_out_irreps)
        self.rearrange_scalars = Rearrange('e m s -> e (m s)')

        # === Final Scalar Projection === #
        scalar_in_dim_for_linear = self.latent_dim + self.tp_n_scalar_out
        self.scalar_latent_projection_mlp = EquivariantScalarMLP(
            in_dim=scalar_in_dim_for_linear,
            out_dim=self.latent_dim,
            **edge_mlp_kwargs
        )

        # === Final Equivariant Projection === #
        # The input to this linear layer is already the 'equivariants' part from tp_out
        # So its in_irreps should reflect only the l>0 part of full_out_irreps
        equiv_in_irreps_for_linear = o3.Irreps([(mul, ir) for mul, ir in full_out_irreps if ir.l > 0])
        # The output of this linear layer should be the l>0 part of linear_out_irreps
        equiv_out_irreps_for_linear = o3.Irreps([(mul, ir) for mul, ir in linear_out_irreps if ir.l > 0])
        self.equivariant_linear_out = EquivariantScalarMLP(
            in_irreps=equiv_in_irreps_for_linear,
            out_irreps=equiv_out_irreps_for_linear,
            reshape_in=False, # equiv_in_irreps_for_linear is already reshaped if needed by TP
            reshape_back=is_last_layer, # Only reshape back if it's the very last layer output
            **edge_mlp_kwargs
        ) if len(equiv_in_irreps_for_linear) > 0 else None

    def _build_tp(self, tps_irreps_in, tps_irreps_out, eq_latent_irreps):
        tmp_i_out = 0
        tp_n_scalar_outs = 0
        instr = []
        full_out_irreps_list = []
        for i_out, (_, ir_out) in enumerate(tps_irreps_out):
            for i_1, (_, ir_1) in enumerate(tps_irreps_in):
                for i_2, (mul, ir_2) in enumerate(eq_latent_irreps):
                    if ir_out in ir_1 * ir_2:
                        if ir_out == SCALAR or ir_out == PSEUDO_SCALAR:
                            tp_n_scalar_outs += 1
                        instr.append((i_1, i_2, tmp_i_out))
                        full_out_irreps_list.append((mul, ir_out))
                        tmp_i_out += 1
        full_out_irreps = o3.Irreps(full_out_irreps_list)
        
        tp = Contracter(
            irreps_in1=tps_irreps_in, irreps_in2=eq_latent_irreps, irreps_out=full_out_irreps,
            instructions=instr, connection_mode="uuu", shared_weights=False, has_weight=False, normalization='component',
        )
        
        return tp, tp_n_scalar_outs, full_out_irreps
        
    def forward(
        self,
        data: AtomicDataDict.Type,
        edge_equivariant_features: torch.Tensor,
        scalar_latent: torch.Tensor,
        eq_latent: Optional[torch.Tensor],
        edge_attrs: torch.Tensor,
        active_edges: torch.Tensor,
        node_conditioning: Optional[torch.Tensor],
        edge_conditioning: Optional[torch.Tensor],
        this_layer_update_coeff: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # === Prepare inputs ===
        node_invariants = data[self.node_invariant_field]
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_nodes = node_invariants.shape[0]

        # === 1. Update scalar latents (residual stream) ===
        new_latents = self.latent_mlp(scalar_latent, edge_conditioning)
        new_latents = self.latent_mlp_norm(new_latents)
        scalar_latent = apply_residual_stream(scalar_latent, new_latents, this_layer_update_coeff, active_edges)

        # === 2. Build local environment embedding ===
        # Create weighted edge attributes for environment pooling
        weights = self.env_embed_mlp(scalar_latent)
        env_edges = self._env_weighter(edge_equivariant_features, weights)
        
        if self.use_attention:
            Q = self.rearrange_qk(self.edge_attr_to_query(edge_attrs))
            K = self.rearrange_qk(self.latent_to_key(scalar_latent))
            W = torch.einsum('emd,emd -> em', Q, K) * self.isqrtd
            attn_softmax = scatter_softmax(W, edge_center, dim=0)
            env_edges = torch.einsum('...d, ... -> ...d', env_edges, attn_softmax)
        
        local_env = scatter_sum(env_edges, edge_center, dim=0, dim_size=num_nodes)
        local_env = self.node_env_norm(local_env)
        local_env = self.env_linear(local_env, node_conditioning)

        if self.use_mace_product:
            assert self.reshape_in_module is not None
            local_env = self.reshape_in_module(self.product(node_feats=local_env, node_attrs=node_invariants, sc=None))
        
        local_env_per_edge = local_env[edge_center]
        
        # === 3. Interact via Tensor Product ===
        local_env_per_edge = self.edge_env_norm(local_env_per_edge)
        assert eq_latent is not None
        eq_latent = self.eq_features_norm(eq_latent)

        tp_out = self.tp(eq_latent, local_env_per_edge)
        tp_out = self.tp_norm(tp_out)
        
        # === 4. Extract new features ===
        scalars, equivariants = torch.split(tp_out, [self.tp_n_scalar_out, tp_out.shape[-1] - self.tp_n_scalar_out], dim=-1)
        
        # New scalar latents for next layer are concatenation of old ones and new scalars from TP
        concatenated_scalars = torch.cat([scalar_latent, self.rearrange_scalars(scalars)], dim=-1)
        updated_scalar_latent = self.scalar_latent_projection_mlp(concatenated_scalars, edge_conditioning)

        # New equivariant features for next layer
        if self.equivariant_linear_out is not None:
            updated_eq_latent = self.equivariant_linear_out(equivariants, edge_conditioning)
            return updated_scalar_latent, updated_eq_latent
        return updated_scalar_latent, None
