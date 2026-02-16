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
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps

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


def _inverse_quadratic_alpha_cap(
    raw_residual: torch.Tensor,
    alpha: torch.Tensor,
    base_residual: torch.Tensor,
    it: int,
    alpha_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cap step size so applied residual is <= base_residual / (it+1)^2."""
    target = base_residual / float((it + 1) * (it + 1))
    proposed = raw_residual * alpha
    if proposed > target:
        alpha = torch.clamp(
            target / (raw_residual + 1e-12),
            min=0.0,
            max=alpha_max,
        )
    return alpha, target


def _clean_mlp_kwargs(kwargs: dict) -> dict:
    """Remove explicit IO dimensions so callers can override them safely."""
    out = dict(kwargs)
    out.pop("mlp_input_dimension", None)
    out.pop("mlp_output_dimension", None)
    return out


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
    combined_edge_inv_dim: int
    edge_conditioning_dim: int
    is_last_layer: bool
    irreps_in: dict
    attention_logit_clip: float
    residual_update_max: float
    use_equivariant_residual: bool


@compile_mode("script")
class InteractionModule(GraphModuleMixin, torch.nn.Module):
    conditioning_fields: List[str]
    __constants__ = ["use_fixed_point_recycling", "fp_use_static_context"]

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
        latent_module_kwargs={'mlp_latent_dimensions': [128, 128], 'mlp_nonlinearity': 'silu'},
        latent_dim: int = 256,
        eq_latent_multiplicity: int = 16,
        use_attention: bool = False,
        attention_head_dim: int = 32,
        attention_logit_clip: float = 0.0,
        use_mace_product: bool = False,
        product_correlation: int = 2,
        residual_update_max: float = 1.0,
        use_equivariant_residual: bool = True,
        use_fixed_point_recycling: bool = False,
        fp_max_iter: int = 16,
        fp_tol: float = 1e-3,
        fp_alpha: float = 0.5,
        fp_grad_steps: int = 16,
        fp_adaptive_damping: bool = False,
        fp_alpha_min: float = 0.05,
        fp_residual_growth_tol: float = 1.0,
        fp_first_layer_update_coeff: float = 0.0,
        fp_state_clip_value: float = 0.0,
        fp_enforce_inverse_quadratic: bool = False,
        fp_use_static_context: bool = True,
        fp_static_context_strength: float = 1.0,
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
        
        self.conditioning_fields = conditioning_fields if conditioning_fields is not None else []
        if residual_update_max <= 0.0:
            raise ValueError("`residual_update_max` must be > 0.")
        if attention_logit_clip < 0.0:
            raise ValueError("`attention_logit_clip` must be >= 0.")
        self.residual_update_max = float(residual_update_max)
        self.attention_logit_clip = float(attention_logit_clip)
        self.use_equivariant_residual = bool(use_equivariant_residual)
        self.use_fixed_point_recycling = bool(use_fixed_point_recycling)
        self.fp_max_iter = int(fp_max_iter)
        self.fp_tol = float(fp_tol)
        self.fp_alpha = float(fp_alpha)
        self.fp_grad_steps = int(fp_grad_steps)
        self.fp_adaptive_damping = bool(fp_adaptive_damping)
        self.fp_alpha_min = float(fp_alpha_min)
        self.fp_residual_growth_tol = float(fp_residual_growth_tol)
        self.fp_first_layer_update_coeff = float(fp_first_layer_update_coeff)
        self.fp_state_clip_value = float(fp_state_clip_value)
        self.fp_enforce_inverse_quadratic = bool(fp_enforce_inverse_quadratic)
        self.fp_use_static_context = bool(fp_use_static_context)
        self.fp_static_context_strength = float(fp_static_context_strength)
        if self.use_fixed_point_recycling:
            if self.fp_max_iter <= 0:
                raise ValueError("`fp_max_iter` must be > 0 when fixed-point recycling is enabled.")
            if self.fp_tol < 0.0:
                raise ValueError("`fp_tol` must be >= 0.")
            if not (0.0 < self.fp_alpha <= 1.0):
                raise ValueError("`fp_alpha` must be in (0, 1].")
            if self.fp_grad_steps < 0:
                raise ValueError("`fp_grad_steps` must be >= 0.")
            if not (0.0 < self.fp_alpha_min <= self.fp_alpha):
                raise ValueError("`fp_alpha_min` must be in (0, fp_alpha].")
            if self.fp_residual_growth_tol <= 0.0:
                raise ValueError("`fp_residual_growth_tol` must be > 0.")
            if self.fp_first_layer_update_coeff < 0.0:
                raise ValueError("`fp_first_layer_update_coeff` must be >= 0.")
            if self.fp_state_clip_value < 0.0:
                raise ValueError("`fp_state_clip_value` must be >= 0.")
            if not (0.0 < self.fp_static_context_strength <= 1.0):
                raise ValueError("`fp_static_context_strength` must be in (0, 1].")
        
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
        self.node_conditioning_dim = 0
        self.edge_conditioning_dim = 0
        for field in self.conditioning_fields:
            dim = self.irreps_in[field].dim
            if field in _NODE_FIELDS:
                self.node_conditioning_dim += dim
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
        projection_in_equiv_irreps = equiv_latent_irreps if self.use_fixed_point_recycling else last_layer_equiv_latent_irreps
        self._final_equiv_flattener = inverse_reshape_irreps(projection_in_equiv_irreps) if self.use_fixed_point_recycling else None

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
        static_fuser_kwargs = _clean_mlp_kwargs(latent_module_kwargs)
        self.fp_scalar_static_fuser = latent_module(
            mlp_input_dimension=2 * latent_dim,
            mlp_output_dimension=latent_dim,
            **static_fuser_kwargs,
        )
        self.fp_equiv_static_fuser = SO3_Linear(
            in_irreps=equiv_latent_irreps + equiv_latent_irreps,
            out_irreps=equiv_latent_irreps,
            internal_weights=True,
            shared_weights=True,
            pad_to_alignment=1,
        )
        self.fp_equiv_flatten = inverse_reshape_irreps(equiv_latent_irreps)
        self.fp_equiv_reshape = reshape_irreps(equiv_latent_irreps)

        self._latent_resnet_update_params = torch.nn.Parameter(torch.full((self.num_layers - 1,), -4.0))
        self.interaction_layers = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            is_last_layer = (i == self.num_layers - 1) and (not self.use_fixed_point_recycling)

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
                combined_edge_inv_dim = initial_scalar_latent_dim,
                attention_logit_clip = self.attention_logit_clip,
                residual_update_max = self.residual_update_max,
                use_equivariant_residual = self.use_equivariant_residual,
            )

            self.interaction_layers.append(InteractionLayer(layer_config))

        # Final projection to the desired output irreps
        self.final_projection = EquivariantScalarMLP(
            in_irreps=(latent_dim, projection_in_equiv_irreps), # Input from the recurrent latent state
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
        active_edges = torch.arange(data[self.edge_radial_emb_field].shape[0], device=data[AtomicDataDict.EDGE_INDEX_KEY].device)
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid() * self.residual_update_max

        # === 1. Prepare Initial Scalar and Equivariant Latents ===
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        # Build initial scalar latent
        scalar_latents_to_cat = [data[self.edge_radial_emb_field]]
        if self.has_edge_invariant_field_input:
            scalar_latents_to_cat.append(data[self.edge_invariant_field])
        if self.has_node_invariant_field_input:
            node_invariants = data[self.node_invariant_field]
            scalar_latents_to_cat.append(node_invariants[edge_center])
            scalar_latents_to_cat.append(node_invariants[edge_neigh])
        initial_scalar_latent = torch.cat(scalar_latents_to_cat, dim=-1)

        # Build initial equivariant latent
        eq_latents_to_cat = [data[self.edge_spharm_emb_field]]
        if self.has_edge_equivariant_field_input:
            eq_latents_to_cat.append(data[self.edge_equivariant_field])
        if self.has_node_equivariant_field_input:
            node_equivariants = data[self.node_equivariant_field]
            eq_latents_to_cat.append(node_equivariants[edge_center])
            eq_latents_to_cat.append(node_equivariants[edge_neigh])
        
        initial_equiv_latent = torch.cat(eq_latents_to_cat, dim=-1)
        if self.eq_concat_permutation is not None:
            initial_equiv_latent = initial_equiv_latent[:, self.eq_concat_permutation]

        # === 2. Generate Initial Latent State ===
        init_out = self.initial_latent_generator(
            (initial_scalar_latent, initial_equiv_latent),
        )
        if torch.jit.isinstance(init_out, Tuple[torch.Tensor, torch.Tensor]):
            scalar_latent, equiv_latent = init_out
        else:
            raise RuntimeError("initial_latent_generator must return scalar and equivariant tensors.")
        static_scalar_latent = scalar_latent
        static_equiv_latent = equiv_latent

        # Prepare conditioning tensors for interaction layers
        node_conditioning, edge_conditioning = prepare_conditioning_tensors(
            data=data,
            conditioning_fields=self.conditioning_fields,
        )

        if self.use_fixed_point_recycling:
            scalar_latent, equiv_latent, _, _ = self._fixed_point_refine(
                data=data,
                scalar_latent=scalar_latent,
                equiv_latent=equiv_latent,
                active_edges=active_edges,
                node_conditioning=node_conditioning,
                edge_conditioning=edge_conditioning,
                layer_update_coefficients=layer_update_coefficients,
                static_scalar_latent=static_scalar_latent,
                static_equiv_latent=static_equiv_latent,
            )
        else:
            scalar_latent, equiv_latent = self._run_interaction_stack(
                data=data,
                scalar_latent=scalar_latent,
                equiv_latent=equiv_latent,
                active_edges=active_edges,
                node_conditioning=node_conditioning,
                edge_conditioning=edge_conditioning,
                layer_update_coefficients=layer_update_coefficients,
            )
        equiv_for_final = equiv_latent
        if (
            self.use_fixed_point_recycling
            and equiv_for_final is not None
            and equiv_for_final.ndim == 3
        ):
            if self._final_equiv_flattener is None:
                raise RuntimeError("Fixed-point recycling expects an equivariant flattener for final projection.")
            equiv_for_final = self._final_equiv_flattener(equiv_for_final)

        final_out = self.final_projection((scalar_latent, equiv_for_final))
        if torch.jit.isinstance(final_out, torch.Tensor):
            data[self.out_field] = final_out
        else:
            raise RuntimeError("final_projection must return a Tensor.")
        return data

    def _run_interaction_stack(
        self,
        data: AtomicDataDict.Type,
        scalar_latent: torch.Tensor,
        equiv_latent: Optional[torch.Tensor],
        active_edges: torch.Tensor,
        node_conditioning: Optional[torch.Tensor],
        edge_conditioning: Optional[torch.Tensor],
        layer_update_coefficients: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        s = scalar_latent
        e = equiv_latent
        for i, layer in enumerate(self.interaction_layers):
            coeff = layer_update_coefficients[i - 1] if i > 0 else None
            if (
                self.use_fixed_point_recycling
                and i == 0
                and self.fp_first_layer_update_coeff > 0.0
            ):
                coeff = torch.as_tensor(
                    self.fp_first_layer_update_coeff,
                    dtype=s.dtype,
                    device=s.device,
                )
            s, e = layer(
                data=data,
                scalar_latent=s,
                equiv_latent=e,
                active_edges=active_edges,
                node_conditioning=node_conditioning,
                edge_conditioning=edge_conditioning,
                this_layer_update_coeff=coeff,
            )
        return s, e

    def _fuse_fixed_point_inputs(
        self,
        dynamic_scalar_latent: torch.Tensor,
        dynamic_equiv_latent: Optional[torch.Tensor],
        static_scalar_latent: torch.Tensor,
        static_equiv_latent: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.fp_use_static_context:
            return dynamic_scalar_latent, dynamic_equiv_latent

        blend = torch.as_tensor(
            self.fp_static_context_strength,
            dtype=dynamic_scalar_latent.dtype,
            device=dynamic_scalar_latent.device,
        )
        scalar_concat = torch.cat([dynamic_scalar_latent, static_scalar_latent], dim=-1)
        scalar_fused = self.fp_scalar_static_fuser(scalar_concat)
        scalar_input = (1.0 - blend) * dynamic_scalar_latent + blend * scalar_fused

        equiv_input = dynamic_equiv_latent
        if dynamic_equiv_latent is not None and static_equiv_latent is not None:
            dynamic_equiv_flat = self.fp_equiv_flatten(dynamic_equiv_latent)
            static_equiv_flat = self.fp_equiv_flatten(static_equiv_latent)
            equiv_concat = torch.cat([dynamic_equiv_flat, static_equiv_flat], dim=-1)
            equiv_fused_flat = self.fp_equiv_static_fuser(equiv_concat)
            equiv_fused = self.fp_equiv_reshape(equiv_fused_flat)
            equiv_input = (1.0 - blend) * dynamic_equiv_latent + blend * equiv_fused

        return scalar_input, equiv_input

    def _fixed_point_refine(
        self,
        data: AtomicDataDict.Type,
        scalar_latent: torch.Tensor,
        equiv_latent: Optional[torch.Tensor],
        active_edges: torch.Tensor,
        node_conditioning: Optional[torch.Tensor],
        edge_conditioning: Optional[torch.Tensor],
        layer_update_coefficients: torch.Tensor,
        static_scalar_latent: Optional[torch.Tensor] = None,
        static_equiv_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, torch.Tensor]:
        # 1) No-grad solve to approach a fixed point.
        with torch.no_grad():
            s_state = scalar_latent
            e_state = equiv_latent
            s_static = s_state if static_scalar_latent is None else static_scalar_latent
            e_static = e_state if static_equiv_latent is None else static_equiv_latent
            prev_residual = torch.as_tensor(-1.0, dtype=s_state.dtype, device=s_state.device)
            inv_quad_base = torch.as_tensor(-1.0, dtype=s_state.dtype, device=s_state.device)
            iters_used = 0
            last_residual = torch.as_tensor(float("inf"), dtype=s_state.dtype, device=s_state.device)
            for it in range(self.fp_max_iter):
                s_input, e_input = self._fuse_fixed_point_inputs(
                    dynamic_scalar_latent=s_state,
                    dynamic_equiv_latent=e_state,
                    static_scalar_latent=s_static,
                    static_equiv_latent=e_static,
                )
                s_prop, e_prop = self._run_interaction_stack(
                    data=data,
                    scalar_latent=s_input,
                    equiv_latent=e_input,
                    active_edges=active_edges,
                    node_conditioning=node_conditioning,
                    edge_conditioning=edge_conditioning,
                    layer_update_coefficients=layer_update_coefficients,
                )
                s_step = s_prop - s_state
                s_raw_rms = torch.sqrt(torch.mean(s_step.square()))
                if e_prop is not None and e_state is not None:
                    e_step = e_prop - e_state
                    e_raw_rms = torch.sqrt(torch.mean(e_step.square()))
                    raw_residual = torch.maximum(s_raw_rms, e_raw_rms)
                else:
                    e_step = None
                    raw_residual = s_raw_rms

                alpha = torch.as_tensor(self.fp_alpha, dtype=s_state.dtype, device=s_state.device)
                if self.fp_adaptive_damping and prev_residual >= 0.0:
                    target = prev_residual * self.fp_residual_growth_tol
                    proposed = raw_residual * alpha
                    if proposed > target:
                        alpha = torch.clamp(
                            target / (raw_residual + 1e-12),
                            min=self.fp_alpha_min,
                            max=self.fp_alpha,
                        )
                if self.fp_enforce_inverse_quadratic:
                    if inv_quad_base < 0.0:
                        inv_quad_base = raw_residual * alpha
                    alpha, _ = _inverse_quadratic_alpha_cap(
                        raw_residual=raw_residual,
                        alpha=alpha,
                        base_residual=inv_quad_base,
                        it=it,
                        alpha_max=self.fp_alpha,
                    )

                s_next = s_state + alpha * s_step
                if e_step is not None and e_state is not None:
                    e_next = e_state + alpha * e_step
                else:
                    e_next = e_prop

                if self.fp_state_clip_value > 0.0:
                    s_next = torch.clamp(s_next, min=-self.fp_state_clip_value, max=self.fp_state_clip_value)
                    if e_next is not None:
                        e_next = torch.clamp(e_next, min=-self.fp_state_clip_value, max=self.fp_state_clip_value)

                residual = raw_residual * alpha
                last_residual = residual

                s_state = s_next
                e_state = e_next
                prev_residual = residual
                iters_used = it + 1
                if self.fp_tol > 0.0 and residual < self.fp_tol:
                    break

        # 2) Short gradient-enabled refinement from detached fixed point.
        s_state = s_state.detach()
        if e_state is not None:
            e_state = e_state.detach()
        s_static = s_state if static_scalar_latent is None else static_scalar_latent
        e_static = e_state if static_equiv_latent is None else static_equiv_latent
        prev_residual = torch.as_tensor(-1.0, dtype=s_state.dtype, device=s_state.device)
        for _ in range(self.fp_grad_steps):
            s_input, e_input = self._fuse_fixed_point_inputs(
                dynamic_scalar_latent=s_state,
                dynamic_equiv_latent=e_state,
                static_scalar_latent=s_static,
                static_equiv_latent=e_static,
            )
            s_prop, e_prop = self._run_interaction_stack(
                data=data,
                scalar_latent=s_input,
                equiv_latent=e_input,
                active_edges=active_edges,
                node_conditioning=node_conditioning,
                edge_conditioning=edge_conditioning,
                layer_update_coefficients=layer_update_coefficients,
            )
            s_step = s_prop - s_state
            s_raw_rms = torch.sqrt(torch.mean(s_step.square()))
            if e_prop is not None and e_state is not None:
                e_step = e_prop - e_state
                e_raw_rms = torch.sqrt(torch.mean(e_step.square()))
                raw_residual = torch.maximum(s_raw_rms, e_raw_rms)
            else:
                e_step = None
                raw_residual = s_raw_rms

            alpha = torch.as_tensor(self.fp_alpha, dtype=s_state.dtype, device=s_state.device)
            if self.fp_adaptive_damping and prev_residual >= 0.0:
                target = prev_residual * self.fp_residual_growth_tol
                proposed = raw_residual * alpha
                if proposed > target:
                    alpha = torch.clamp(
                        target / (raw_residual + 1e-12),
                        min=self.fp_alpha_min,
                        max=self.fp_alpha,
                    )
            if self.fp_enforce_inverse_quadratic and last_residual > 0.0:
                proposed = raw_residual * alpha
                if proposed > last_residual:
                    alpha = torch.clamp(
                        last_residual / (raw_residual + 1e-12),
                        min=0.0,
                        max=self.fp_alpha,
                    )

            s_state = s_state + alpha * s_step
            if e_step is not None and e_state is not None:
                e_state = e_state + alpha * e_step
            else:
                e_state = e_prop

            if self.fp_state_clip_value > 0.0:
                s_state = torch.clamp(s_state, min=-self.fp_state_clip_value, max=self.fp_state_clip_value)
                if e_state is not None:
                    e_state = torch.clamp(e_state, min=-self.fp_state_clip_value, max=self.fp_state_clip_value)

            prev_residual = raw_residual * alpha
            last_residual = prev_residual
            if self.fp_tol > 0.0 and prev_residual < self.fp_tol:
                break

        return s_state, e_state, int(iters_used), last_residual


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
        scalar_latent: torch.Tensor,
        equiv_latent: Optional[torch.Tensor],
        active_edges: torch.Tensor,
        node_conditioning: Optional[torch.Tensor],
        edge_conditioning: Optional[torch.Tensor],
        this_layer_update_coeff: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # === Prepare inputs ===
        node_invariants = data[self.node_invariant_field]
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_nodes = node_invariants.shape[0]
        if equiv_latent is None:
            raise RuntimeError("InteractionLayer expected an equivariant latent tensor but received None.")
        equiv_latent_tensor = equiv_latent

        # === 1. Build local environment embedding ===
        env_edges_raw = self.env_embed_mlp((scalar_latent, equiv_latent_tensor))
        if torch.jit.isinstance(env_edges_raw, torch.Tensor):
            env_edges = env_edges_raw
        else:
            raise RuntimeError("env_embed_mlp must return a Tensor.")
        
        if self.use_attention:
            assert self.node_attr_to_query is not None and self.latent_to_key is not None and self.rearrange_qk is not None
            Q = self.rearrange_qk(self.node_attr_to_query(node_invariants[edge_center]))
            K = self.rearrange_qk(self.latent_to_key(scalar_latent))
            W = torch.einsum('emd,emd -> em', Q, K) * self.inv_sqrtd
            if self.attention_logit_clip > 0.0:
                W = torch.clamp(W, min=-self.attention_logit_clip, max=self.attention_logit_clip)
            attn_softmax = scatter_softmax(W, edge_center, dim=0)
            env_edges = torch.einsum('...d, ... -> ...d', env_edges, attn_softmax)
        
        env_nodes = scatter_sum(env_edges, edge_center, dim=0, dim_size=num_nodes)
        if self.use_mace_product:
            assert self.node_inv_to_product_mlp is not None
            assert self.reshape_in_module is not None
            env_nodes = self.reshape_in_module(self.product(node_feats=env_nodes, node_attrs=node_invariants, sc=None))
        env_nodes = self.node_env_norm(env_nodes)

        local_env_per_edge = env_nodes[edge_center]
        
        # === 3. Interact via Tensor Product ===
        tp_out = self.tp(equiv_latent_tensor, local_env_per_edge)
        tp_out = self.tp_norm(tp_out)

        # === 4. Extract new features ===
        proj_out = self.projection(tp_out, edge_conditioning)
        new_equiv_latent = torch.jit.annotate(Optional[torch.Tensor], None)
        if torch.jit.isinstance(proj_out, Tuple[torch.Tensor, Optional[torch.Tensor]]):
            new_scalar_latent, new_equiv_latent = proj_out
        else:
            raise RuntimeError("Projection must return scalar and equivariant tensors.")
        scalar_latent = apply_residual_stream(scalar_latent, new_scalar_latent, this_layer_update_coeff, active_edges)
        if (
            self.use_equivariant_residual
            and new_equiv_latent is not None
            and new_equiv_latent.shape == equiv_latent_tensor.shape
        ):
            equiv_latent_out = apply_residual_stream(
                equiv_latent_tensor, new_equiv_latent, this_layer_update_coeff, active_edges
            )
        else:
            equiv_latent_out = new_equiv_latent

        return scalar_latent, equiv_latent_out


def _scalar_slices_from_irreps(irreps: o3.Irreps) -> List[Tuple[int, int]]:
    """Return slices for parity-even scalars (0e) in a flattened irrep tensor."""
    out: List[Tuple[int, int]] = []
    off = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        if ir.l == 0 and ir.p == 1:
            out.append((off, off + dim))
        off += dim
    return out

def _extract_tp_scalars(tp_out: torch.Tensor, tp_scalar_slices_flat, mul: int) -> torch.Tensor:
    """
    Returns tp scalars as (E, mul * n_scalar_blocks) regardless of tp_out format.
    tp_scalar_slices_flat are slices in flat (E, irreps.dim) indexing.
    """
    if len(tp_scalar_slices_flat) == 0:
        return tp_out.new_zeros((tp_out.shape[0], 0))

    E = tp_out.shape[0]

    if tp_out.ndim == 2:
        # flat: (E, irreps.dim)
        return torch.cat([tp_out[:, s:e] for (s, e) in tp_scalar_slices_flat], dim=-1)

    if tp_out.ndim == 3:
        # channel: (E, mul, feat_per_mul)
        parts = []
        for (s, e) in tp_scalar_slices_flat:
            # convert flat slice [s:e] into per-mul feature slice
            if (s % mul) != 0 or (e % mul) != 0:
                raise ValueError(f"Scalar slice ({s},{e}) not divisible by mul={mul}; slices are inconsistent.")
            s_per = s // mul
            e_per = e // mul
            parts.append(tp_out[:, :, s_per:e_per])   # (E, mul, 1) for scalars
        scal_ch = torch.cat(parts, dim=-1)           # (E, mul, n_scalars)
        return scal_ch.reshape(E, -1)                # (E, mul * n_scalars)

    raise ValueError(f"Unexpected tp_out.ndim={tp_out.ndim}")


@compile_mode("script")
class InteractionLayerV0(torch.nn.Module):
    """
    Baseline interaction layer:
      - NO hypernetwork weights (all equivariant linears use internal weights)
      - Optional, *correctly scaled* attention (off by default)
      - Residual stream updates for BOTH scalar and equivariant latents
      - Still uses your TP backbone (Contracter) so it stays comparable
    """
    __constants__ = ["node_invariant_field", "use_attention", "is_last_layer"]

    def __init__(self, config):
        super().__init__()
        self.node_invariant_field = config.node_invariant_field
        self.is_last_layer = config.is_last_layer

        # ---- Attention (optional) ----
        self.use_attention = bool(config.use_attention)
        self.attn_head_dim = int(config.attention_head_dim)
        self.scale = 1.0 / math.sqrt(float(self.attn_head_dim))  # correct scaling

        # NOTE: to keep V0 clean, we only implement a very simple attention:
        # Q comes from node invariants at edge centers, K comes from scalar_latent.
        # If you don't want it, keep use_attention=False.
        self.node_attr_to_q = None
        self.edge_scalar_to_k = None
        if self.use_attention:
            # we assume node invariants are plain scalars; if they are irreps-packed, adapt before enabling attention
            # input dims:
            # - node invariants: inferred at runtime in forward (so we build small linears lazily not possible in script)
            # For simplicity in torchscript, you can disable attention or hardcode dims.
            raise RuntimeError(
                "InteractionLayerV0 attention is intentionally off-by-default. "
                "If you want it, wire fixed dims here (node_inv_dim, latent_dim) and remove this guard."
            )

        # ---- (1) Local env embedding: equiv_latent -> env_edges (internal weights) ----
        # This replaces env_embed_mlp((scalar, equiv)) that triggers hypernetwork weights.
        self.env_eq = SO3_Linear(
            config.equiv_latent_irreps,  # equiv latent irreps
            config.equiv_latent_irreps,  # same irreps for env messages
            internal_weights=True,
            shared_weights=True,
            pad_to_alignment=1,
        )

        # A tiny scalar gate so scalar_latent can still influence env aggregation
        # gate = 1 + 0.1 * tanh(MLP(scalar_latent, cond))
        gate_in_dim = int(config.latent_dim) + int(config.edge_conditioning_dim)
        self.env_gate = config.latent_module(
            mlp_input_dimension=gate_in_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=1,
            has_bias=True,
        )
        with torch.no_grad():
            # start at exactly gate=1 (since tanh(0)=0)
            if hasattr(self.env_gate, "sequential"):
                self.env_gate.sequential[-1].weight.zero_()
                if self.env_gate.sequential[-1].bias is not None:
                    self.env_gate.sequential[-1].bias.zero_()

        # Norms (use your existing layernorms if available in this file)
        self.node_env_norm = SO3_LayerNorm(config.equiv_latent_irreps)

        # ---- (2) Tensor product backbone ----
        instr = []
        out_irreps_list: List[Tuple[int, o3.Irrep]] = []
        i_out = 0
        for _, ir_out in config.tp_irreps_out:
            for i1, (_, ir1) in enumerate(config.tp_irreps_in):
                for i2, (mul2, ir2) in enumerate(config.equiv_latent_irreps):
                    if ir_out in ir1 * ir2:
                        instr.append((i1, i2, i_out))
                        out_irreps_list.append((mul2, ir_out))
                        i_out += 1
        tp_out_irreps = o3.Irreps(out_irreps_list)

        self.tp = Contracter(
            irreps_in1=config.tp_irreps_in,
            irreps_in2=config.equiv_latent_irreps,
            irreps_out=tp_out_irreps,
            instructions=instr,
            connection_mode="uuu",
            shared_weights=False,
            has_weight=False,
            normalization="component",
        )
        self.tp_norm = SO3_LayerNorm(tp_out_irreps)

        # ---- (3) Projection head without hypernetwork weights ----
        out_equiv_irreps = config.last_layer_equiv_latent_irreps if config.is_last_layer else config.equiv_latent_irreps

        # Scalar update reads only scalar blocks from TP output (+ conditioning)
        self.tp_scalar_slices = _scalar_slices_from_irreps(tp_out_irreps)
        tp_scalar_dim = 0
        for s, e in self.tp_scalar_slices:
            tp_scalar_dim += (e - s)

        scalar_in_dim = int(tp_scalar_dim) + int(config.edge_conditioning_dim)
        # Use your latent_module_kwargs if provided
        latent_kwargs = dict(config.latent_module_kwargs) if hasattr(config, "latent_module_kwargs") else {}
        if "mlp_input_dimension" in latent_kwargs:
            latent_kwargs.pop("mlp_input_dimension")
        if "mlp_output_dimension" in latent_kwargs:
            latent_kwargs.pop("mlp_output_dimension")

        self.scalar_update = config.latent_module(
            mlp_input_dimension=scalar_in_dim,
            mlp_output_dimension=int(config.latent_dim),
            **latent_kwargs,
        )

        # Equivariant update: plain internal-weight SO3 linear from TP output -> out equiv irreps
        self.equiv_update = SO3_Linear(
            tp_out_irreps,
            out_equiv_irreps,
            internal_weights=True,
            shared_weights=True,
            pad_to_alignment=1,
        )

        # Optional scale on equiv update, initialized to identity
        equiv_gate_in_dim = int(config.latent_dim) + int(config.edge_conditioning_dim)
        self.equiv_gate = config.latent_module(
            mlp_input_dimension=equiv_gate_in_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=1,
            has_bias=True,
        )
        with torch.no_grad():
            if hasattr(self.equiv_gate, "sequential"):
                self.equiv_gate.sequential[-1].weight.zero_()
                if self.equiv_gate.sequential[-1].bias is not None:
                    self.equiv_gate.sequential[-1].bias.zero_()

    def forward(
        self,
        data: AtomicDataDict.Type,
        scalar_latent: torch.Tensor,                 # [E, latent_dim]
        equiv_latent: Optional[torch.Tensor],         # [E, equiv_dim]
        active_edges: torch.Tensor,                   # indices into E
        node_conditioning: Optional[torch.Tensor],    # unused in V0 (kept for signature compatibility)
        edge_conditioning: Optional[torch.Tensor],    # [E, cond_dim] or None
        this_layer_update_coeff: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if equiv_latent is None:
            raise RuntimeError("InteractionLayerV0 expected equiv_latent but got None.")

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        num_nodes = int(data[self.node_invariant_field].shape[0])

        # ---- local env per edge ----
        env_edges = self.env_eq(equiv_latent)  # [E, env_equiv_dim]

        if edge_conditioning is None:
            gate_in = scalar_latent
        else:
            gate_in = torch.cat([scalar_latent, edge_conditioning], dim=-1)

        env_gate = 1.0 + 0.1 * torch.tanh(self.env_gate(gate_in))  # [E, 1]
        env_edges = env_edges * env_gate.unsqueeze(-1)

        env_nodes = scatter_sum(env_edges, edge_center, dim=0, dim_size=num_nodes)
        env_nodes = self.node_env_norm(env_nodes)
        local_env_per_edge = env_nodes[edge_center]

        # ---- TP ----
        tp_out = self.tp(equiv_latent, local_env_per_edge)
        tp_out = self.tp_norm(tp_out)

        # ---- scalar projection ----
        if len(self.tp_scalar_slices) > 0:
            mul = self.tp_norm.mul   # or self.node_env_norm.mul; any common-mul layernorm object
            tp_scalars = _extract_tp_scalars(tp_out, self.tp_scalar_slices, mul)
        else:
            tp_scalars = tp_out.new_zeros((tp_out.shape[0], 0))

        if edge_conditioning is None:
            scalar_in = tp_scalars
        else:
            scalar_in = torch.cat([tp_scalars, edge_conditioning], dim=-1)

        new_scalar = self.scalar_update(scalar_in)

        # ---- equiv projection ----
        new_equiv = self.equiv_update(tp_out)

        # mild gate (identity at init)
        if edge_conditioning is None:
            eg_in = scalar_latent
        else:
            eg_in = torch.cat([scalar_latent, edge_conditioning], dim=-1)
        equiv_gate = 1.0 + 0.1 * torch.tanh(self.equiv_gate(eg_in))
        new_equiv = new_equiv * equiv_gate.unsqueeze(-1)

        # ---- residual streams ----
        scalar_latent = apply_residual_stream(scalar_latent, new_scalar, this_layer_update_coeff, active_edges)

        if self.is_last_layer:
            # last layer may change irreps: don't residual-mix unless you guarantee same shape
            equiv_latent_out = new_equiv
        else:
            equiv_latent_out = apply_residual_stream(equiv_latent, new_equiv, this_layer_update_coeff, active_edges)

        return scalar_latent, equiv_latent_out
