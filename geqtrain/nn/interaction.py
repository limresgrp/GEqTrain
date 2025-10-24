import math
import torch

from typing import Any, Dict, Optional, List, Tuple, Union
from geqtrain.utils._model_utils import build_concatenation_permutation, process_out_irreps
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_softmax
from einops.layers.torch import Rearrange
from geqtrain.nn import SO3_Linear
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

# Try to import wandb for logging, but don't fail if it's not installed
try:
    import wandb
except ImportError:
    wandb = None
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
        edge_spharm_emb_field=AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
        edge_radial_emb_field=AtomicDataDict.EDGE_RADIAL_EMB_KEY,
        edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
        edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
        out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        latent_dim: int = 256,
        eq_latent_multiplicity: int = 16,
        use_attention: bool = False,
        head_dim: int = 16,
        use_mace_product: bool = False,
        product_correlation: int = 2,
        conditioning_fields: Optional[List[str]] = None,
        irreps_in = None,
        debug: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.node_invariant_field   = node_invariant_field
        self.node_equivariant_field = node_equivariant_field
        self.edge_radial_emb_field  = edge_radial_emb_field
        self.edge_invariant_field   = edge_invariant_field
        self.edge_spharm_emb_field  = edge_spharm_emb_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field              = out_field
        self.latent_dim = latent_dim
        
        self.conditioning_fields = conditioning_fields if conditioning_fields is not None else []
        self.debug = debug
        
        # --- Irreps Initialization and Validation ---
        required_irreps = [f for f in [self.node_invariant_field, self.edge_invariant_field, self.edge_equivariant_field] if f is not None]
        # Add radial and spharm fields as required inputs
        required_irreps.append(self.edge_radial_emb_field)
        required_irreps.append(self.edge_spharm_emb_field)
        required_irreps.extend(self.conditioning_fields)
        # Remove duplicates
        required_irreps = list(dict.fromkeys(required_irreps))

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

        permutation, initial_eq_latent_irreps = build_concatenation_permutation(eq_irreps_to_concat)
        self.register_buffer('eq_concat_permutation', permutation)

        # The local environment will have these irreps
        local_env_irreps = o3.Irreps([(eq_latent_multiplicity, ir) for _, ir in initial_eq_latent_irreps])

        # 2. Build list of dimensions for initial scalar latent
        initial_scalar_latent_dim = self.edge_radial_emb_field_irreps.dim
        if self.has_edge_invariant_field_input: initial_scalar_latent_dim += self.edge_invariant_field_irreps.dim
        if self.has_node_invariant_field_input: initial_scalar_latent_dim += 2 * self.node_invariant_field_irreps.dim

        default_irreps =o3.Irreps([
            (self.latent_dim, ir) if ir.l == 0 else (eq_latent_multiplicity, ir)
            for _, ir in self.edge_spharm_emb_field_irreps]
        )
        final_out_irreps = process_out_irreps(
            out_irreps=out_irreps,
            output_ls=output_ls,
            output_mul=None if output_mul=="hidden" else output_mul,
            default_irreps=default_irreps,
        )

        # The tensor product takes the l>0 eq_latent and the l>=0 env_nodes
        tp_recurrent_irreps = local_env_irreps
        tp_out_irreps = o3.Irreps([(eq_latent_multiplicity, ir) for _, ir in final_out_irreps])
        tps_irreps_in, tps_irreps_out = build_tps_irreps_list(num_layers, tp_recurrent_irreps, tp_out_irreps, tp_recurrent_irreps)

        # === Initial Latent Projection ===
        # This MLP will take the initial scalar and equivariant features and project them
        # into the initial latent space for the interaction layers.
        self.initial_latent_generator = EquivariantScalarMLP(
            in_irreps=(o3.Irreps(f"{initial_scalar_latent_dim}x0e"), initial_eq_latent_irreps),
            out_irreps=(o3.Irreps(f"{self.latent_dim}x0e"), local_env_irreps),
            latent_module=ScalarMLPFunction,
            latent_kwargs={'mlp_latent_dimensions': [128, 128], 'mlp_nonlinearity': 'silu'},
            equiv_linear_module=SO3_Linear,
            output_shape_spec="channel_wise",
        )

        self._latent_resnet_update_params = torch.nn.Parameter(torch.zeros(self.num_layers - 1))
        self.interaction_layers = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            is_last_layer = (i == self.num_layers - 1)
            linear_out_irreps = o3.Irreps(
                [(mul, ir) for mul, ir in local_env_irreps if ir.l != 0]
            ) if is_last_layer else local_env_irreps

            layer = InteractionLayer(
                latent_dim=self.latent_dim,
                eq_latent_multiplicity=eq_latent_multiplicity,
                use_attention=use_attention,
                head_dim=head_dim,
                use_mace_product=use_mace_product,
                product_correlation=product_correlation,
                tps_irreps_in=tps_irreps_in[i],
                tps_irreps_out=tps_irreps_out[i],
                local_env_irreps=local_env_irreps,
                linear_out_irreps=linear_out_irreps,
                irreps_in=self.irreps_in,
                node_invariant_field=self.node_invariant_field,
                edge_invariant_field=self.edge_invariant_field,
                edge_equivariant_field=self.edge_equivariant_field,
                is_last_layer=is_last_layer,
                node_conditioning_dim=self.node_conditioning_dim,
                edge_conditioning_dim=self.edge_conditioning_dim,
                combined_edge_inv_dim=initial_scalar_latent_dim,
                debug=self.debug,
            )
            self.interaction_layers.append(layer)

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
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

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
        scalar_latent, equiv_latent = self.initial_latent_generator(
            (initial_scalar_latent, initial_equiv_latent),
        )

        # Prepare conditioning tensors for interaction layers
        node_conditioning, edge_conditioning = prepare_conditioning_tensors(
            data=data,
            conditioning_fields=self.conditioning_fields,
        )

        # === 3. Interaction Loop ===
        for i, layer in enumerate(self.interaction_layers):
            scalar_latent, equiv_latent = layer(
                data=data,
                scalar_latent=scalar_latent,
                equiv_latent=equiv_latent,
                active_edges=active_edges,
                node_conditioning=node_conditioning,
                edge_conditioning=edge_conditioning,
                this_layer_update_coeff=layer_update_coefficients[i - 1] if i > 0 else None,
            )

            if self.debug and wandb.run is not None:
                log_feature_on_wandb(f"InteractionLayer/{i}.scalar_latent_out", scalar_latent, self.training)
                if equiv_latent is not None:
                    log_feature_on_wandb(f"InteractionLayer/{i}.eq_latent_out", equiv_latent, self.training)

        if scalar_latent is not None and equiv_latent is not None:
            final_output_features = torch.cat([scalar_latent, equiv_latent], dim=-1)
        elif scalar_latent is not None:
            final_output_features = scalar_latent
        else:
            final_output_features = equiv_latent
        data[self.out_field] = final_output_features

        if self.debug and wandb.run is not None:
            log_feature_on_wandb(f"{str(self)}.out_features.scalar", scalar_latent, self.training)
            if equiv_latent is not None: log_feature_on_wandb(f"{self.name}.out_features.vectorial", equiv_latent, self.training)
        
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
        eq_latent_multiplicity: int,
        head_dim: int,
        use_attention: bool,
        use_mace_product: bool,
        product_correlation: int,
        tps_irreps_in: o3.Irreps,
        tps_irreps_out: o3.Irreps,
        local_env_irreps: o3.Irreps,
        linear_out_irreps: o3.Irreps,
        irreps_in: dict,
        node_invariant_field: str,
        edge_invariant_field: str,
        edge_equivariant_field: str,
        combined_edge_inv_dim: int,
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
        self.node_invariant_field = node_invariant_field
        self.combined_edge_inv_dim = combined_edge_inv_dim

        # === Define dimensions and irreps ===
        node_inv_dim = irreps_in[node_invariant_field].dim
        edge_inv_dim = irreps_in[edge_invariant_field].dim
        node_inv_irreps = irreps_in[node_invariant_field]
        
        edge_mlp_kwargs = {"conditioning_dim": edge_conditioning_dim, 'latent_kwargs': {'mlp_latent_dimensions': [128, 128]}}
        node_mlp_kwargs = {"conditioning_dim": node_conditioning_dim, 'latent_kwargs': {'mlp_latent_dimensions': [128, 128]}}

        # === Environment embedding modules ===
        self.env_embed_mlp = EquivariantScalarMLP(
            in_irreps=(o3.Irreps(f"{self.latent_dim}x0e"), local_env_irreps),
            out_irreps=local_env_irreps,
            **edge_mlp_kwargs
        )
        self.node_env_norm = SO3_LayerNorm(local_env_irreps)

        # === Attention modules ===
        self.node_attr_to_query = None
        self.latent_to_key = None
        if self.use_attention:
            self.isqrtd = math.isqrt(head_dim)
            self.node_attr_to_query = ScalarMLPFunction(node_inv_dim, [], eq_latent_multiplicity * head_dim)
            self.latent_to_key = ScalarMLPFunction(self.latent_dim, [], eq_latent_multiplicity * head_dim)
            self.rearrange_qk = Rearrange('e (m d) -> e m d', m=eq_latent_multiplicity, d=head_dim)
        
        # === MACE product modules ===
        self.node_inv_to_product_mlp = None
        self.reshape_in_module = None
        if self.use_mace_product:
            # Project node invariants to match the multiplicity of the equivariant features
            self.node_inv_to_product_mlp = ScalarMLPFunction(
                mlp_input_dimension=node_inv_dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=eq_latent_multiplicity
            )

            self.product = EquivariantProductBasisBlock(
                node_feats_irreps=local_env_irreps,
                target_irreps=local_env_irreps,
                correlation=product_correlation,
                num_elements=node_inv_dim,
            )
            self.reshape_in_module = reshape_irreps(local_env_irreps)

        # === Tensor Product modules ===
        self.tp, self.tp_n_scalar_out, full_out_irreps = self._build_tp(tps_irreps_in, tps_irreps_out, local_env_irreps)
        self.tp_norm = SO3_LayerNorm(full_out_irreps)
        self.rearrange_scalars = Rearrange('e m s -> e (m s)')

        # === Final Scalar Projection === #
        self.scalar_linear_out = EquivariantScalarMLP(
            in_irreps=self.tp_n_scalar_out,
            out_irreps=self.latent_dim,
            **edge_mlp_kwargs
        )
        self.scalar_latent_norm = torch.nn.LayerNorm(self.latent_dim)

        # === Final Equivariant Projection === #
        # The input to this linear layer is the full output of the tensor product
        equiv_in_irreps_for_linear = full_out_irreps
        equiv_out_irreps_for_linear = linear_out_irreps
        self.equivariant_linear_out = EquivariantScalarMLP(
            in_irreps=equiv_in_irreps_for_linear,
            out_irreps=equiv_out_irreps_for_linear,
            output_shape_spec="flat" if is_last_layer else "channel_wise", # Only flatten if it's the very last layer output
            **edge_mlp_kwargs
        ) if len(equiv_in_irreps_for_linear) > 0 else None

    def _build_tp(self, tps_irreps_in, tps_irreps_out, local_env_irreps):
        tmp_i_out = 0
        tp_n_scalar_outs = 0
        instr = []
        full_out_irreps_list: List[Tuple[int, o3.Irrep]] = []
        for i_out, (_, ir_out) in enumerate(tps_irreps_out):
            for i_1, (_, ir_1) in enumerate(tps_irreps_in): # this is eq_latent (l>0)
                for i_2, (mul, ir_2) in enumerate(local_env_irreps): # this is env_nodes (l>=0)
                    if ir_out in ir_1 * ir_2:
                        if ir_out == SCALAR or ir_out == PSEUDO_SCALAR:
                            tp_n_scalar_outs += 1
                        instr.append((i_1, i_2, tmp_i_out))
                        full_out_irreps_list.append((mul, ir_out))
                        tmp_i_out += 1
        full_out_irreps = o3.Irreps(full_out_irreps_list)
        
        tp = Contracter(
            irreps_in1=tps_irreps_in, irreps_in2=local_env_irreps, irreps_out=full_out_irreps,
            instructions=instr, connection_mode="uuu", shared_weights=False, has_weight=False, normalization='component',
        )
        
        return tp, tp_n_scalar_outs, full_out_irreps
        
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

        # === 1. Build local environment embedding ===
        env_edges = self.env_embed_mlp((scalar_latent, equiv_latent))
        
        if self.use_attention:
            assert self.node_attr_to_query is not None and self.latent_to_key is not None and self.rearrange_qk is not None
            Q = self.rearrange_qk(self.node_attr_to_query(node_invariants[edge_center]))
            K = self.rearrange_qk(self.latent_to_key(scalar_latent))
            W = torch.einsum('emd,emd -> em', Q, K) * self.isqrtd
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
        assert equiv_latent is not None
        tp_out = self.tp(equiv_latent, local_env_per_edge)
        tp_out = self.tp_norm(tp_out)
        
        # === 4. Extract new features ===
        scalars = self.rearrange_scalars(tp_out[..., :self.tp_n_scalar_out])
        
        # New scalar latents for next layer are concatenation of old ones and new scalars from TP
        new_scalar_latent = self.scalar_linear_out(scalars, edge_conditioning)
        scalar_latent = apply_residual_stream(scalar_latent, new_scalar_latent, this_layer_update_coeff, active_edges)
        scalar_latent = self.scalar_latent_norm(scalar_latent)

        # New equivariant features for next layer
        if self.equivariant_linear_out is not None:
            equiv_latent = self.equivariant_linear_out(tp_out, edge_conditioning)
            return scalar_latent, equiv_latent
        return scalar_latent, None
