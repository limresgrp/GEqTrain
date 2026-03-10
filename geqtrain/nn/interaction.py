import math
import torch
from dataclasses import dataclass

from typing import Optional, List, Tuple, Union
from geqtrain.utils._model_utils import build_concatenation_permutation, prepare_conditioning_tensors, process_out_irreps
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_softmax
from einops.layers.torch import Rearrange
from geqtrain.nn import SO3_Linear
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import (
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)
from geqtrain.nn import (
    GraphModuleMixin,
    SO3_LayerNorm,
    ScalarMLPFunction,
)
from geqtrain.nn.allegro import Contracter
from geqtrain.utils.so3 import split_irreps, tp_path_exists
from geqtrain.nn._equivariant_scalar_mlp import EquivariantScalarMLP
from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps

def apply_residual_stream(
    latents: torch.Tensor,
    new_latents: torch.Tensor,
    residual_update_coeff: Optional[torch.Tensor],
) -> torch.Tensor:
    if residual_update_coeff is None:
        # This happens on the first layer, where the dimensions of `latents` (from initial edge attributes)
        # and `new_latents` (from the first MLP) are different. We just take the new latents.
        return new_latents
    
    # At init, we assume new and old to be approximately uncorrelated
    # Thus their variances add
    # we always want the latent space to be normalized to variance = 1.0,
    # because it is critical for learnability. Still, we want to preserve
    # the _relative_ magnitudes of the current latent and the residual update
    # to be controled by `residual_update_coeff`
    # Solving the simple system for the two coefficients:
    #   a^2 + b^2 = 1  (variances add)   &    a * residual_update_coeff = b
    # gives
    #   a = 1 / sqrt(1 + residual_update_coeff^2)  &  b = residual_update_coeff / sqrt(1 + residual_update_coeff^2)
    # rsqrt is reciprocal sqrt
    coefficient_old = torch.rsqrt(residual_update_coeff.square() + 1)
    coefficient_new = residual_update_coeff * coefficient_old
    return coefficient_old * latents + coefficient_new * new_latents


def build_tps_irreps_list(
    num_layers: int,
    tp_in_irreps: o3.Irreps,
    tp_out_irreps: o3.Irreps,
    tp_latent_irreps: o3.Irreps
) -> Tuple[List[o3.Irreps], List[o3.Irreps]]:
        """Pre-computes the irreps for each layer's tensor product."""
        
        # Build up the irreps for the iterated TPs layer by layer
        tps_irreps = [tp_in_irreps]
        arg_irreps = tp_in_irreps
        for i in range(num_layers):
            ir_out = tp_out_irreps if i == num_layers - 1 else tp_latent_irreps
            # Prune impossible paths
            ir_out = o3.Irreps([(mul, ir) for mul, ir in ir_out if tp_path_exists(arg_irreps, tp_latent_irreps, ir)])
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
                if any(out_ir in arg_ir * env_ir for _, env_ir in tp_latent_irreps for _, out_ir in temp_out_irreps):
                    new_arg_irreps.append((mul, arg_ir))
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            temp_out_irreps = new_arg_irreps
        
        tps_irreps = list(reversed(new_tps_irreps))
        
        return tps_irreps[:-1], tps_irreps[1:]


@dataclass
class InteractionLayerConfig:
    """Configuration for an InteractionLayer."""
    latent_module: torch.nn.Module
    latent_module_kwargs: dict
    latent_dim: int
    eq_latent_multiplicity: int
    use_attention: bool
    attention_head_dim: int
    use_mace_product: bool
    product_correlation: int
    tp_irreps_in: o3.Irreps
    tp_irreps_out: o3.Irreps
    equiv_latent_irreps: o3.Irreps
    last_layer_equiv_latent_irreps: o3.Irreps
    node_invariant_field: str
    edge_conditioning_dim: int
    is_last_layer: bool
    irreps_in: dict
    attention_logit_clip: float
    use_equivariant_residual: bool


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
        node_equivariant_field=AtomicDataDict.NODE_EQ_ATTRS_KEY,
        edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
        edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
        edge_spharm_emb_field=AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
        edge_radial_emb_field=AtomicDataDict.EDGE_RADIAL_EMB_KEY,
        out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        latent_module=ScalarMLPFunction,
        latent_module_kwargs: Optional[dict] = None,
        latent_dim: int = 256,
        eq_latent_multiplicity: int = 16,
        use_attention: bool = False,
        attention_head_dim: int = 64,
        attention_logit_clip: float = 0.0,
        use_mace_product: bool = False,
        product_correlation: int = 2,
        residual_update_max: float = 1.0,
        use_equivariant_residual: bool = True,
        conditioning_fields: Optional[List[str]] = None,
        irreps_in = None,
    ):
        super().__init__()
        self.num_layers             = num_layers
        self.node_invariant_field   = node_invariant_field
        self.node_equivariant_field = node_equivariant_field
        self.edge_radial_emb_field  = edge_radial_emb_field
        self.edge_invariant_field   = edge_invariant_field
        self.edge_spharm_emb_field  = edge_spharm_emb_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field              = out_field
        if latent_module_kwargs is None:
            latent_module_kwargs = {"mlp_latent_dimensions": [128, 128], "mlp_nonlinearity": "silu"}
        
        self.conditioning_fields = conditioning_fields if conditioning_fields is not None else []
        if residual_update_max <= 0.0:
            raise ValueError("`residual_update_max` must be > 0.")
        if attention_logit_clip < 0.0:
            raise ValueError("`attention_logit_clip` must be >= 0.")
        self.residual_update_max = float(residual_update_max)
        self.attention_logit_clip = float(attention_logit_clip)
        self.use_equivariant_residual = bool(use_equivariant_residual)
        
        # --- Irreps Initialization and Validation ---
        required_irreps = [
            self.node_invariant_field, self.edge_radial_emb_field,
            self.edge_spharm_emb_field
        ] + self.conditioning_fields
        self._init_irreps(
            required_irreps_in=required_irreps,
            irreps_in=irreps_in,
        )

        # Initialize and validate all input fields
        self._init_input_field('node_invariant_field', 'invariant')
        self._init_input_field('node_equivariant_field', 'equivariant')
        self._init_input_field('edge_invariant_field', 'invariant')
        self._init_input_field('edge_equivariant_field', 'equivariant')

        # Also init radial and spharm fields
        self._init_input_field('edge_radial_emb_field', 'invariant')
        self._init_input_field('edge_spharm_emb_field', 'equivariant')

        # --- Conditioning Tensor Dimension Calculation ---
        self.edge_conditioning_dim = 0
        for field in self.conditioning_fields:
            dim = self.irreps_in[field].dim
            if field in _NODE_FIELDS:
                self.edge_conditioning_dim += 2 * dim
            elif field in _EDGE_FIELDS:
                self.edge_conditioning_dim += dim

        # --- Define network architecture based on final irreps ---
        # 1. Build list of irreps for initial equivariant latent
        eq_irreps_to_concat = [self.edge_spharm_emb_field_irreps]
        if self.has_edge_equivariant_field_input:
            eq_irreps_to_concat.append(self.edge_equivariant_field_irreps)
        if self.has_node_equivariant_field_input:
            eq_irreps_to_concat.append(self.node_equivariant_field_irreps) # for src
            eq_irreps_to_concat.append(self.node_equivariant_field_irreps) # for target

        permutation, initial_equiv_latent_irreps = build_concatenation_permutation(eq_irreps_to_concat)
        self.register_buffer('eq_concat_permutation', permutation)

        # The equivariant latent will have these irreps
        equiv_latent_irreps = o3.Irreps([(eq_latent_multiplicity, ir) for _, ir in initial_equiv_latent_irreps])

        # 2. Build list of dimensions for initial scalar features
        initial_scalar_latent_dim = self.edge_radial_emb_field_irreps.dim
        if self.has_edge_invariant_field_input: initial_scalar_latent_dim += self.edge_invariant_field_irreps.dim
        if self.has_node_invariant_field_input: initial_scalar_latent_dim += 2 * self.node_invariant_field_irreps.dim

        # Determine the final output irreps for the entire module
        default_out_irreps = o3.Irreps([
            (latent_dim, ir) if ir.l == 0 else (eq_latent_multiplicity, ir)
            for _, ir in initial_equiv_latent_irreps]
        )
        final_out_irreps = process_out_irreps(
            out_irreps=out_irreps,
            output_ls=output_ls,
            output_mul=output_mul,
            default_irreps=default_out_irreps,
        )
        # The tensor product takes the l>0 eq_latent and the l>=0 env_nodes
        tp_recurrent_irreps = equiv_latent_irreps
        tp_out_irreps = o3.Irreps([(eq_latent_multiplicity, ir) for _, ir in final_out_irreps])
        # On last layer, the equiv_latent should be of only equivariant features, as scalar ones are not used and would be discarded
        _, last_layer_equiv_latent_irreps = split_irreps(o3.Irreps([(mul, ir) for mul, ir in equiv_latent_irreps if ir in tp_out_irreps]))

        tps_irreps_in, tps_irreps_out = build_tps_irreps_list(num_layers, tp_recurrent_irreps, tp_out_irreps, tp_recurrent_irreps)

        # === Initial Latent Projection ===
        # This MLP will take the initial scalar and equivariant features and project them
        # into the initial latent space for the interaction layers.
        self.initial_latent_generator = EquivariantScalarMLP(
            in_irreps=(initial_scalar_latent_dim, initial_equiv_latent_irreps),
            out_irreps=(latent_dim, equiv_latent_irreps),
            latent_module=latent_module,
            latent_kwargs=latent_module_kwargs,
            equiv_linear_module=SO3_Linear,
            output_shape_spec="channel_wise",
        )

        self._latent_resnet_update_params = torch.nn.Parameter(torch.full((self.num_layers - 1,), -4.0))
        self.interaction_layers = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            is_last_layer = (i == self.num_layers - 1)

            layer_config = InteractionLayerConfig(
                latent_module = latent_module,
                latent_module_kwargs = latent_module_kwargs,
                latent_dim = latent_dim,
                eq_latent_multiplicity = eq_latent_multiplicity,
                use_attention = use_attention,
                attention_head_dim = attention_head_dim,
                use_mace_product = use_mace_product and not is_last_layer,
                product_correlation = product_correlation,
                tp_irreps_in = tps_irreps_in[i],
                tp_irreps_out = tps_irreps_out[i],
                equiv_latent_irreps = equiv_latent_irreps,
                last_layer_equiv_latent_irreps = last_layer_equiv_latent_irreps,
                irreps_in = self.irreps_in,
                node_invariant_field = self.node_invariant_field,
                is_last_layer = is_last_layer,
                edge_conditioning_dim = self.edge_conditioning_dim,
                attention_logit_clip = self.attention_logit_clip,
                use_equivariant_residual = self.use_equivariant_residual,
            )

            self.interaction_layers.append(InteractionLayer(layer_config))

        # Final projection to the desired output irreps
        self.final_projection = EquivariantScalarMLP(
            in_irreps=(latent_dim, last_layer_equiv_latent_irreps), # Input from the recurrent latent state
            out_irreps=final_out_irreps, # Desired final output irreps
            latent_module=latent_module,
            latent_kwargs=latent_module_kwargs,
            equiv_linear_module=SO3_Linear,
            output_shape_spec="flat", # The out_field expects a single flat tensor
            conditioning_dim=self.edge_conditioning_dim, # Pass edge conditioning to the final projection
        )

        # The output field will contain the final equivariant features
        self.irreps_out[self.out_field] = final_out_irreps

    def _init_input_field(self, field_name: str, field_type: str):
        """
        Initializes and validates an input field from `irreps_in`.
        Sets `self.has_<field_name>_input` and `self.<field_name>_irreps`.
        """
        field_key = getattr(self, field_name)
        has_input_attr = f"has_{field_name}_input"
        irreps_attr = f"{field_name}_irreps"

        if field_key is not None and field_key in self.irreps_in:
            irreps = self.irreps_in[field_key]
            if irreps.dim > 0:
                if field_type == 'invariant':
                    if not all(ir.l == 0 for _, ir in irreps):
                        raise ValueError(f"Field '{field_key}' is specified as invariant but contains non-scalar irreps: {irreps}")
                    setattr(self, has_input_attr, True)
                    setattr(self, irreps_attr, irreps)
                    return
                elif field_type == 'equivariant':
                    setattr(self, has_input_attr, True)
                    setattr(self, irreps_attr, irreps)
                    return
                else:
                    raise ValueError(f"Invalid field_type '{field_type}'")
        # Field is not present or not specified
        setattr(self, has_input_attr, False)
        setattr(self, irreps_attr, o3.Irreps(""))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid() * self.residual_update_max

        # === 1. Prepare Initial Scalar and Equivariant Latents ===
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        # Build initial scalar latent
        scalar_input_parts = [data[self.edge_radial_emb_field]]
        if self.has_edge_invariant_field_input:
            scalar_input_parts.append(data[self.edge_invariant_field])
        if self.has_node_invariant_field_input:
            node_invariants = data[self.node_invariant_field]
            scalar_input_parts.append(node_invariants[edge_src])
            scalar_input_parts.append(node_invariants[edge_dst])
        initial_scalar_latent = torch.cat(scalar_input_parts, dim=-1)

        # Build initial equivariant latent
        equiv_input_parts = [data[self.edge_spharm_emb_field]]
        if self.has_edge_equivariant_field_input:
            equiv_input_parts.append(data[self.edge_equivariant_field])
        if self.has_node_equivariant_field_input:
            node_equivariants = data[self.node_equivariant_field]
            equiv_input_parts.append(node_equivariants[edge_src])
            equiv_input_parts.append(node_equivariants[edge_dst])
        
        initial_equiv_latent = torch.cat(equiv_input_parts, dim=-1)
        if self.eq_concat_permutation is not None:
            initial_equiv_latent = initial_equiv_latent[:, self.eq_concat_permutation]

        # === 2. Generate Initial Latent State ===
        init_out = self.initial_latent_generator(
            (initial_scalar_latent, initial_equiv_latent),
        )
        if torch.jit.isinstance(init_out, Tuple[torch.Tensor, torch.Tensor]):
            scalar_state, equiv_state = init_out
        else:
            raise RuntimeError("initial_latent_generator must return scalar and equivariant tensors.")

        # Prepare conditioning tensors for interaction layers
        _, edge_conditioning = prepare_conditioning_tensors(
            data=data,
            conditioning_fields=self.conditioning_fields,
        )

        scalar_state, equiv_state = self._run_interaction_stack(
            data=data,
            scalar_state=scalar_state,
            equiv_state=equiv_state,
            edge_conditioning=edge_conditioning,
            layer_update_coefficients=layer_update_coefficients,
        )

        final_out = self.final_projection((scalar_state, equiv_state))
        if torch.jit.isinstance(final_out, torch.Tensor):
            data[self.out_field] = final_out
        else:
            raise RuntimeError("final_projection must return a Tensor.")
        return data

    def _run_interaction_stack(
        self,
        data: AtomicDataDict.Type,
        scalar_state: torch.Tensor,
        equiv_state: Optional[torch.Tensor],
        edge_conditioning: Optional[torch.Tensor],
        layer_update_coefficients: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s = scalar_state
        e = equiv_state
        for i, layer in enumerate(self.interaction_layers):
            residual_update_coeff = layer_update_coefficients[i - 1] if i > 0 else None
            s, e = layer(
                data=data,
                scalar_state=s,
                equiv_state=e,
                edge_conditioning=edge_conditioning,
                residual_update_coeff=residual_update_coeff,
            )
        return s, e


@compile_mode("script")
class InteractionLayer(torch.nn.Module):
    """
    Refactored InteractionLayer.
    - Decoupled from InteractionModule; all dependencies are injected.
    - `__init__` is streamlined and groups related module constructions.
    - Forward pass is clarified and returns the correct signature.
    """
    __constants__ = ["node_invariant_field", "use_attention", "use_mace_product", "attention_logit_clip", "use_equivariant_residual"]
    def __init__(
        self, config: InteractionLayerConfig
    ):
        super().__init__()
        self.node_invariant_field = config.node_invariant_field
        self.use_attention = config.use_attention
        self.use_mace_product = config.use_mace_product
        self.attention_logit_clip = float(config.attention_logit_clip)
        self.use_equivariant_residual = bool(config.use_equivariant_residual)

        # === Define dimensions and irreps ===
        node_inv_dim = config.irreps_in[config.node_invariant_field].dim
        
        edge_mlp_kwargs = {
            "conditioning_dim": config.edge_conditioning_dim,
            'latent_kwargs': config.latent_module_kwargs
        }

        # === Environment embedding modules ===
        self.env_embed_mlp = EquivariantScalarMLP(
            in_irreps=(config.latent_dim, config.equiv_latent_irreps),
            out_irreps=config.equiv_latent_irreps,
            **edge_mlp_kwargs
        )
        self.node_env_norm = SO3_LayerNorm(config.equiv_latent_irreps)

        # === Attention modules ===
        self.node_attr_to_query = None
        self.latent_to_key = None
        if config.use_attention:
            self.inv_sqrtd = 1. / math.sqrt(config.attention_head_dim)
            self.node_attr_to_query = config.latent_module(node_inv_dim, [], config.eq_latent_multiplicity * config.attention_head_dim)
            self.latent_to_key = config.latent_module(config.latent_dim, [], config.eq_latent_multiplicity * config.attention_head_dim)
            self.rearrange_qk = Rearrange('e (m d) -> e m d', m=config.eq_latent_multiplicity, d=config.attention_head_dim)
        
        # === MACE product modules ===
        self.node_inv_to_product_mlp = None
        self.reshape_in_module = None
        if config.use_mace_product:
            # Project node invariants to match the multiplicity of the equivariant features
            self.node_inv_to_product_mlp = config.latent_module(
                mlp_input_dimension=node_inv_dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=config.eq_latent_multiplicity
            )

            self.product = EquivariantProductBasisBlock(
                node_feats_irreps=config.equiv_latent_irreps,
                target_irreps=config.equiv_latent_irreps,
                correlation=config.product_correlation,
                num_elements=node_inv_dim,
            )
            self.reshape_in_module = reshape_irreps(config.equiv_latent_irreps)

        # === Tensor Product modules ===
        # Build the tensor product for this layer
        # The output of the TP will be projected to the new scalar and equivariant latents.
        instr = []
        full_out_irreps_list: List[Tuple[int, o3.Irrep]] = []
        i_out_running = 0
        for _, ir_out in config.tp_irreps_out:
            for i_1, (_, ir_1) in enumerate(config.tp_irreps_in): # this is eq_latent (l>0)
                for i_2, (mul, ir_2) in enumerate(config.equiv_latent_irreps): # this is env_nodes (l>=0)
                    if ir_out in ir_1 * ir_2:
                        instr.append((i_1, i_2, i_out_running))
                        full_out_irreps_list.append((mul, ir_out))
                        i_out_running += 1
        tp_out_irreps = o3.Irreps(full_out_irreps_list)
        self.tp = Contracter(
            irreps_in1=config.tp_irreps_in, irreps_in2=config.equiv_latent_irreps, irreps_out=tp_out_irreps,
            instructions=instr, connection_mode="uuu", shared_weights=False, has_weight=False, normalization='component',
        )
        self.tp_norm = SO3_LayerNorm(tp_out_irreps)

        # === Final Scalar+Equivariant Projection === #
        self.projection = EquivariantScalarMLP(
            in_irreps=tp_out_irreps,
            out_irreps=(config.latent_dim, config.last_layer_equiv_latent_irreps if config.is_last_layer else config.equiv_latent_irreps),
            output_shape_spec="flat" if config.is_last_layer else "channel_wise", # Only flatten if it's the very last layer output
            **edge_mlp_kwargs
        )
        
    def forward(
        self,
        data: AtomicDataDict.Type,
        scalar_state: torch.Tensor,
        equiv_state: Optional[torch.Tensor],
        edge_conditioning: Optional[torch.Tensor],
        residual_update_coeff: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # === Prepare inputs ===
        node_attrs = data[self.node_invariant_field]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_nodes = node_attrs.shape[0]
        if equiv_state is None:
            raise RuntimeError("InteractionLayer expected an equivariant latent tensor but received None.")
        equiv_state_tensor = equiv_state

        # === 1. Build local environment embedding ===
        env_edges_raw = self.env_embed_mlp((scalar_state, equiv_state_tensor))
        if torch.jit.isinstance(env_edges_raw, torch.Tensor):
            env_edges = env_edges_raw
        else:
            raise RuntimeError("env_embed_mlp must return a Tensor.")
        
        if self.use_attention:
            assert self.node_attr_to_query is not None and self.latent_to_key is not None and self.rearrange_qk is not None
            Q = self.rearrange_qk(self.node_attr_to_query(node_attrs[edge_src]))
            K = self.rearrange_qk(self.latent_to_key(scalar_state))
            W = torch.einsum('emd,emd -> em', Q, K) * self.inv_sqrtd
            if self.attention_logit_clip > 0.0:
                W = torch.clamp(W, min=-self.attention_logit_clip, max=self.attention_logit_clip)
            attn_softmax = scatter_softmax(W, edge_src, dim=0)
            env_edges = torch.einsum('...d, ... -> ...d', env_edges, attn_softmax)
        
        env_nodes = scatter_sum(env_edges, edge_src, dim=0, dim_size=num_nodes)
        if self.use_mace_product:
            assert self.node_inv_to_product_mlp is not None
            assert self.reshape_in_module is not None
            env_nodes = self.reshape_in_module(self.product(node_feats=env_nodes, node_attrs=node_attrs, sc=None))
        env_nodes = self.node_env_norm(env_nodes)

        edge_local_env = env_nodes[edge_src]
        
        # === 3. Interact via Tensor Product ===
        tp_out = self.tp(equiv_state_tensor, edge_local_env)
        tp_out = self.tp_norm(tp_out)

        # === 4. Extract new features ===
        proj_out = self.projection(tp_out, edge_conditioning)
        new_equiv_state = torch.jit.annotate(Optional[torch.Tensor], None)
        if torch.jit.isinstance(proj_out, Tuple[torch.Tensor, Optional[torch.Tensor]]):
            new_scalar_state, new_equiv_state = proj_out
        else:
            raise RuntimeError("Projection must return scalar and equivariant tensors.")
        scalar_state = apply_residual_stream(scalar_state, new_scalar_state, residual_update_coeff)
        if (
            self.use_equivariant_residual
            and new_equiv_state is not None
            and new_equiv_state.shape == equiv_state_tensor.shape
        ):
            equiv_state_out = apply_residual_stream(
                equiv_state_tensor, new_equiv_state, residual_update_coeff
            )
        else:
            equiv_state_out = new_equiv_state

        return scalar_state, equiv_state_out
