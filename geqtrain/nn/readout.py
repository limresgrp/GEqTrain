from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from e3nn.util.jit import compile_mode

# Assuming these are the correct import paths from your project structure
from geqtrain.data import (
    AtomicDataDict,
    _NODE_FIELDS,
    _GRAPH_FIELDS,
)
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.nn._heads import L0IndexedAttention
from geqtrain.nn._equivariant_scalar_mlp import EquivariantScalarMLP
from geqtrain.utils._model_utils import build_concatenation_permutation


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, nn.Module):
    conditioning_fields: List[str]
    """
    This module takes a feature tensor (`field`) and computes an output tensor
    (`out_field`). It can optionally condition its internal transformations on a
    second tensor (`conditioning_field`).
    """
    def __init__(
        self,
        irreps_in,
        # Input fields
        field: Optional[str] = None,
        invariant_field: Optional[str] = None,
        equivariant_field: Optional[str] = None,
        # Output fields
        out_field: Optional[str] = None, # The key where the output will be stored
        invariant_out_field: Optional[str] = None,
        scalar_out_field: Optional[str] = None, # New: secondary field for scalar output
        equivariant_out_field: Optional[str] = None,
        # Other params
        conditioning_fields: Optional[List[str]] = None, # List of keys to use for conditioning
        out_irreps: Union[o3.Irreps, str, None] = None,
        strict_irreps: bool = True,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        resnet: bool = False,
        ignore_amp: bool = False,
        bias: Optional[Union[float, List]] = None,
    ):
        super().__init__()

        # -- Input field validation --
        if field is not None:
            if invariant_field is not None or equivariant_field is not None:
                raise ValueError("Cannot specify both `field` and (`invariant_field` or `equivariant_field`).")
            self.field = field
            self.invariant_field = None
            self.equivariant_field = None
            self.input_mode = "single"
        else:
            if invariant_field is None and equivariant_field is None:
                raise ValueError("Must specify either `field` or at least one of `invariant_field`, `equivariant_field`.")
            self.field = None
            self.invariant_field = invariant_field
            self.equivariant_field = equivariant_field
            self.input_mode = "split"

        # -- Output field validation --
        if out_field is not None:
            if invariant_out_field is not None or equivariant_out_field is not None:
                raise ValueError("Cannot specify both `out_field` and (`invariant_out_field` or `equivariant_out_field`).")
            self.out_field = out_field
            self.invariant_out_field = None
            self.equivariant_out_field = None
            self.scalar_out_field = scalar_out_field
            self.output_mode = "single"
        else:
            if invariant_out_field is None and equivariant_out_field is None:
                # Default to writing to the input field(s) if no output is specified
                if self.input_mode == "single":
                    self.out_field = self.field
                    self.invariant_out_field = None
                    self.equivariant_out_field = None
                    self.output_mode = "single"
                    self.scalar_out_field = scalar_out_field
                else:
                    self.out_field = None
                    self.invariant_out_field = self.invariant_field
                    self.equivariant_out_field = self.equivariant_field
                    self.output_mode = "split"
                    self.scalar_out_field = None # Not supported in this default case
            else:
                self.out_field = None
                self.invariant_out_field = invariant_out_field
                self.equivariant_out_field = equivariant_out_field
                self.output_mode = "split"
                self.scalar_out_field = None # Not supported in this default case

        self.conditioning_fields = conditioning_fields if conditioning_fields is not None else []
        self.ignore_amp = ignore_amp
        self.resnet = resnet

        # --- Input/Output Irreps Determination ---
        required_irreps = []
        if self.field is not None: required_irreps.append(self.field)
        if self.invariant_field is not None: required_irreps.append(self.invariant_field)
        if self.equivariant_field is not None: required_irreps.append(self.equivariant_field)
        required_irreps.extend(self.conditioning_fields)

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=required_irreps,
        )

        # Determine the combined output irreps for the EquivariantScalarMLP processor
        processor_out_irreps_combined: o3.Irreps
        if out_irreps is None:
            if self.input_mode == 'single':
                # Case 1 & 2: input is single, output is single or split.
                # The processor's output irreps match the single input irreps.
                processor_out_irreps_combined = self.irreps_in[self.field]
            else: # self.input_mode == 'split'
                # Case 3 & 4: input is split, output is single or split.
                # The processor's output irreps are the combination of the split input irreps.
                processor_out_irreps_combined = o3.Irreps("")
                if self.invariant_field and self.invariant_field in self.irreps_in:
                    processor_out_irreps_combined += self.irreps_in[self.invariant_field]
                if self.equivariant_field and self.equivariant_field in self.irreps_in:
                    processor_out_irreps_combined += self.irreps_in[self.equivariant_field]
                if processor_out_irreps_combined.dim == 0:
                    raise ValueError("For split input, `out_irreps` is None, but neither `invariant_field` nor `equivariant_field` are found in `irreps_in`. Please provide `out_irreps` explicitly or ensure input fields exist.")
        else:
            processor_out_irreps_combined = out_irreps if isinstance(out_irreps, o3.Irreps) else o3.Irreps(out_irreps)

        # Determine the combined input irreps for the EquivariantScalarMLP processor
        processor_in_irreps: o3.Irreps
        self.split_index: int = 0
        self.n_scalars_in: int = 0
        if self.input_mode == "single":
            processor_in_irreps = self.irreps_in[self.field]
            self.n_scalars_in = sum(mul * ir.dim for mul, ir in processor_in_irreps if ir.l == 0)
            self.split_index = self.n_scalars_in
        else: # "split"
            combined_in_irreps_list = []
            if self.invariant_field and self.invariant_field in irreps_in: combined_in_irreps_list.append(o3.Irreps(irreps_in[self.invariant_field]))
            if self.equivariant_field and self.equivariant_field in irreps_in: combined_in_irreps_list.append(o3.Irreps(irreps_in[self.equivariant_field]))
            processor_in_irreps = (combined_in_irreps_list[0], combined_in_irreps_list[1])
            self.n_scalars_in = combined_in_irreps_list[0].dim
            self.split_index = self.n_scalars_in
            
        # Prepare irreps_out for GraphModuleMixin, reflecting the actual output fields
        gm_irreps_out = {}
        if self.output_mode == "single":
            processor_out_irreps = processor_out_irreps_combined
            gm_irreps_out[self.out_field] = processor_out_irreps
            if self.scalar_out_field is not None:
                scalar_part = o3.Irreps([(mul, ir) for mul, ir in processor_out_irreps if ir.l == 0])
                if scalar_part.dim > 0: gm_irreps_out[self.scalar_out_field] = scalar_part
        else:
            # Split the combined output irreps into scalar and equivariant parts for GraphModuleMixin
            scalar_out_irreps_for_gm = o3.Irreps([(mul, ir) for mul, ir in processor_out_irreps_combined if ir.l == 0])
            equivariant_out_irreps_for_gm = o3.Irreps([(mul, ir) for mul, ir in processor_out_irreps_combined if ir.l > 0])
            if self.invariant_out_field and len(scalar_out_irreps_for_gm) > 0:
                gm_irreps_out[self.invariant_out_field] = scalar_out_irreps_for_gm
            if self.equivariant_out_field and len(equivariant_out_irreps_for_gm) > 0:
                gm_irreps_out[self.equivariant_out_field] = equivariant_out_irreps_for_gm
            processor_out_irreps = (scalar_out_irreps_for_gm, equivariant_out_irreps_for_gm)

        self.irreps_out.update(gm_irreps_out)
        
        # --- Resnet ---
        self._resnet_update_coeff: Optional[nn.Parameter] = None
        if self.resnet:
            representative_out_field = self.out_field if self.output_mode == "single" else (self.invariant_out_field or self.equivariant_out_field)
            if representative_out_field not in self.irreps_in:
                 raise ValueError(f"For resnet=True, out_field='{representative_out_field}' must be in `irreps_in`")
            if self.irreps_in[representative_out_field] != processor_out_irreps_combined:
                 raise ValueError("For resnet=True, output irreps must match input irreps for the out_field.")
            # Start close to identity by using a strongly negative logit.
            self._resnet_update_coeff = nn.Parameter(torch.tensor([-5.0], dtype=torch.float32))
        
        # --- Conditioning Fields Validation and Dimension Calculation ---
        self.total_conditioning_dim = 0
        for cond_field in self.conditioning_fields:
            if cond_field not in self.irreps_in:
                raise ValueError(f"Conditioning field '{cond_field}' not found in irreps_in.")
            cond_irreps = self.irreps_in[cond_field]
            if not all(ir.l == 0 for _, ir in cond_irreps):
                raise ValueError(f"Conditioning field '{cond_field}' must have scalar (0e) irreps, but got {cond_irreps}.")
            self.total_conditioning_dim += cond_irreps.dim

        # --- Core Processing Module ---
        self.processor = EquivariantScalarMLP(
            in_irreps=processor_in_irreps,
            out_irreps=processor_out_irreps,
            conditioning_dim=self.total_conditioning_dim,
            latent_module=readout_latent,
            latent_kwargs=readout_latent_kwargs,
            strict_irreps=strict_irreps,
            output_shape_spec="flat",
        )
        # n_scalars_out is the total dimension of the scalar part of the output
        self.n_scalars_out = sum(mul * ir.dim for mul, ir in processor_out_irreps_combined if ir.l == 0)

        # --- Bias ---
        self.bias = None
        if self.n_scalars_out > 0:
            if bias is not None:
                if isinstance(bias, float):
                    bias_init = torch.full((self.n_scalars_out,), bias, dtype=torch.float32)
                elif isinstance(bias, list):
                    bias_init = torch.tensor(bias, dtype=torch.float32)
                    assert bias_init.shape[0] == self.n_scalars_out, "Length of bias list must match number of scalar outputs."
                else: 
                    raise ValueError("Bias must be a float or a list of floats.")
                self.bias = nn.Parameter(bias_init)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not torch.jit.is_scripting() and self.ignore_amp:
            with torch.amp.autocast('cuda', enabled=False):
                return self._forward_impl(data)
        return self._forward_impl(data)

    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # --- 1. Prepare inputs for EquivariantScalarMLP ---
        if self.input_mode == "single":
            features = data[self.field]
        else: # "split"
            if self.invariant_field is None or self.equivariant_field is None:
                raise ValueError("Split input requires both invariant_field and equivariant_field.")
            if self.invariant_field not in data or self.equivariant_field not in data:
                raise ValueError("No input features found for split input mode.")
            features = (
                data[self.invariant_field],
                data[self.equivariant_field],
            )

        conditioning_tensor: Optional[torch.Tensor] = None
        if len(self.conditioning_fields) > 0:
            conditioning_tensor_list = [data[f] for f in self.conditioning_fields]
            conditioning_tensor = torch.cat(conditioning_tensor_list, dim=-1)
            # Broadcast graph-level conditioning to node-level if needed
            if conditioning_tensor.shape[0] == 1 and self.input_mode == "single":
                conditioning_tensor = conditioning_tensor.expand(features.shape[0], -1)
            elif conditioning_tensor.shape[0] == 1 and self.input_mode == "split":
                conditioning_tensor = conditioning_tensor.expand(features[0].shape[0], -1)

        # --- 2. Run the core processor ---
        # The processor can return a single tensor or a tuple
        out_features_or_tuple = self.processor(features, conditioning_tensor)

        # --- 3. Handle ResNet connection ---
        if self.resnet:
            # Resnet is only supported for single output field mode for simplicity
            if self.output_mode != "single":
                raise NotImplementedError("ResNet is only supported for single `out_field` mode.")
            old_features = data[self.out_field]
            assert self._resnet_update_coeff is not None
            coeff = self._resnet_update_coeff.sigmoid()
            coefficient_old = torch.rsqrt(coeff.square() + 1)
            coefficient_new = coeff * coefficient_old
            out_features_or_tuple = coefficient_old * old_features + coefficient_new * out_features_or_tuple

        # --- 4. Handle bias and write outputs ---
        if self.output_mode == "single":
            self._apply_bias_and_write_single(data, out_features_or_tuple)
        else: # "split"
            self._apply_bias_and_write_split(data, out_features_or_tuple)

        return data

    def _apply_bias_and_write_single(self, data: AtomicDataDict.Type, out_features: torch.Tensor):
        if self.bias is not None:
            out_features[..., :self.n_scalars_out] = out_features[..., :self.n_scalars_out] + self.bias

        if self.scalar_out_field is not None and self.n_scalars_out > 0:
            data[self.scalar_out_field] = out_features[..., :self.n_scalars_out]

        data[self.out_field] = out_features

    def _apply_bias_and_write_split(self, data: AtomicDataDict.Type, out_tuple: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        out_scalars, out_equiv = out_tuple

        if self.invariant_out_field is not None:
            if out_scalars is None:
                raise ValueError(f"Module was configured to write to '{self.invariant_out_field}' but produced no scalar output.")
            if self.bias is not None:
                out_scalars = out_scalars + self.bias
            data[self.invariant_out_field] = out_scalars

        if self.equivariant_out_field is not None:
            # # # if out_equiv is None:
            # # #     raise ValueError(f"Module was configured to write to '{self.equivariant_out_field}' but produced no equivariant output.")
            data[self.equivariant_out_field] = out_equiv

@compile_mode("script")
class AttentionReadoutModule(ReadoutModule):
    """
    Extends ReadoutModule to include a non-local attention mechanism.

    Attention is applied to the scalar part of the input features *before*
    the standard readout process. This breaks strict locality, allowing
    information to be aggregated across nodes or graphs.
    """
    def __init__(
        self,
        num_heads: int = 8,
        dataset_mode: str = 'single', # single|ensemble
        **kwargs
    ):
        super().__init__(**kwargs)

        if self.input_mode != "single":
            raise NotImplementedError("AttentionReadoutModule currently only supports `field` (single tensor) input.")

        if self.n_scalars_in == 0:
            raise ValueError("AttentionReadoutModule requires scalar input features.")
        
        self.scalar_attnt_enabled = True
        idx_key = ""
        
        if self.field in _NODE_FIELDS:
            idx_key = AtomicDataDict.BATCH_KEY
        elif self.field in _GRAPH_FIELDS and dataset_mode == 'ensemble':
            idx_key = AtomicDataDict.ENSEMBLE_INDEX_KEY
        else:
            self.scalar_attnt_enabled = False
            
        if self.scalar_attnt_enabled:
            self.ensemble_attnt1 = L0IndexedAttention(
                irreps_in=self.irreps_in, field=self.field, out_field=self.field, 
                num_heads=num_heads, idx_key=idx_key, update_mlp=True
            )
            self.ensemble_attnt2 = L0IndexedAttention(
                irreps_in=self.irreps_in, field=self.field, out_field=self.field, 
                num_heads=num_heads, idx_key=idx_key
            )
        else:
            self.ensemble_attnt1 = None
            self.ensemble_attnt2 = None

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Applies attention to scalar features, then passes to the base ReadoutModule.
        """
        if self.scalar_attnt_enabled and self.ensemble_attnt1 is not None:
            # This is a pre-processing step for the main readout forward pass.
            # We modify the feature tensor in `data` before calling super().forward().
            features = data[self.field]
            scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)

            # The original L0IndexedAttention implementation seems to operate on a tensor of scalars.
            scalars = self.ensemble_attnt1(scalars, data)
            scalars = self.ensemble_attnt2(scalars, data)

            data[self.field] = torch.cat((scalars, equiv), dim=-1)

        return super().forward(data)
