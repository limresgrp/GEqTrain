"""Runtime feature corruption for reconstruction with ordinary heads and losses."""

import math
from typing import List, Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict, _NODE_FIELDS, _GRAPH_FIELDS
from geqtrain.data.AtomicData import register_fields
from ._graph_mixin import GraphModuleMixin


@compile_mode("script")
class MaskEdgeFeatures(GraphModuleMixin, torch.nn.Module):
    """Mask whole edge feature rows and expose detached reconstruction targets.

    Place after the producer of ``field`` and before any consumers. A normal
    head writes ``out_field``; the inference loop copies ``out_field_target``
    into the reference dictionary under ``out_field``. Use ``mask_field`` in
    the corresponding loss and metrics to supervise only corrupted edges.
    """

    _ref_data_keys: List[str]

    def __init__(
        self,
        field: str,
        out_field: str,
        mask_fraction: float = 0.15,
        mask_value: float = 0.0,
        enabled: bool = True,
        mask_in_eval: bool = False,
        mask_field: Optional[str] = None,
        irreps_in=None,
    ):
        super().__init__()
        if not math.isfinite(mask_fraction) or not 0.0 <= mask_fraction <= 1.0:
            raise ValueError("mask_fraction must be between 0 and 1.")
        if not math.isfinite(mask_value):
            raise ValueError("mask_value must be finite.")
        self.field = field
        self.out_field = out_field
        self.target_field = out_field + "_target"
        self.mask_field = mask_field if mask_field is not None else out_field + "_mask"
        if len({field, out_field, self.target_field, self.mask_field}) != 4:
            raise ValueError("Input, prediction, target and mask fields must be distinct.")
        self.mask_fraction = float(mask_fraction)
        self.mask_value = float(mask_value)
        self.enabled = bool(enabled)
        self.mask_in_eval = bool(mask_in_eval)
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=[field])
        irreps = self.irreps_in[field]
        if irreps is None:
            raise ValueError(f"MaskEdgeFeatures requires declared irreps for '{field}'.")
        if mask_value != 0.0 and any(ir != o3.Irrep("0e") for _, ir in irreps):
            raise ValueError("Non-scalar/pseudoscalar features require mask_value=0 to preserve equivariance.")
        generated_fields = [out_field, self.target_field, self.mask_field]
        for key in [field] + generated_fields:
            if key in _NODE_FIELDS or key in _GRAPH_FIELDS:
                raise ValueError(f"MaskEdgeFeatures requires edge fields, but '{key}' is registered otherwise.")
        for key in generated_fields:
            if key in self.irreps_in:
                raise ValueError(f"MaskEdgeFeatures would overwrite existing field '{key}'.")
        self.feature_dim = irreps.dim
        self.irreps_out.update({self.target_field: irreps, self.mask_field: None})
        register_fields(edge_fields=generated_fields)
        self._ref_data_keys = [self.target_field]

    @property
    def ref_data_keys(self) -> List[str]:
        return self._ref_data_keys

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        n_edges = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
        if features.dim() != 2 or features.shape[0] != n_edges or features.shape[1] != self.feature_dim:
            raise ValueError("Masked edge features must have shape [n_edges, irreps.dim].")
        # A separate, detached copy prevents both mutation and target-side gradients.
        data[self.target_field] = features.detach().clone()
        if self.enabled and (self.training or self.mask_in_eval) and self.mask_fraction > 0.0:
            mask = torch.rand(n_edges, device=features.device) < self.mask_fraction
        else:
            mask = torch.zeros(n_edges, dtype=torch.bool, device=features.device)
        data[self.mask_field] = mask
        data[self.field] = torch.where(mask.unsqueeze(-1), torch.full_like(features, self.mask_value), features)
        return data
