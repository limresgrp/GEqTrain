import torch
from torch.nn import Module, Parameter
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_add
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.data import AtomicDataDict
from typing import Optional, List


class EnsembleAggregator(GraphModuleMixin, Module):
    def __init__(self, irreps_in, field: str, out_field: Optional[str] = None, aggregation_method: str = "mean"):
        super().__init__()
        self.field = field
        self.out_field = out_field or field
        self.aggregation_method = aggregation_method
        in_irreps = irreps_in[field]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: in_irreps},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        ensemble_indices = data["ensemble_index"]

        # Use torch_scatter for aggregation
        if self.aggregation_method == "mean":
            aggregated_features = scatter_mean(features, ensemble_indices, dim=0)
        elif self.aggregation_method == "sum":
            aggregated_features = scatter_sum(features, ensemble_indices, dim=0)
        elif self.aggregation_method == "max":
            aggregated_features, _ = scatter_max(features, ensemble_indices, dim=0)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")

        # Update the data dictionary with the aggregated features
        data[self.out_field] = aggregated_features
        return data




class WeightedEnsembleAggregator(GraphModuleMixin, Module):
    def __init__(self, field: str, out_field: Optional[str], input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.field = field
        self.out_field = out_field or field
        self.weight_mlp = ScalarMLPFunction(input_dim, hidden_dims, output_dim=1)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        ensemble_indices = data["ensemble_index"]

        # Compute weights using ScalarMLPFunction
        weights = self.weight_mlp(features).squeeze(-1)

        # Normalize weights within each ensemble
        normalized_weights = scatter_add(weights, ensemble_indices, dim=0, dim_size=features.size(0))
        normalized_weights = weights / normalized_weights[ensemble_indices]

        # Apply weights to features
        weighted_features = features * normalized_weights.unsqueeze(-1)

        # Use torch_scatter for weighted sum aggregation
        aggregated_features = scatter_add(weighted_features, ensemble_indices, dim=0)

        # Update the data dictionary with the aggregated features
        data[self.out_field] = aggregated_features
        return data