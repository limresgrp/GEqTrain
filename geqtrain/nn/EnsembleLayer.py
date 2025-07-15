import torch
from torch.nn import Module, Parameter
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_add
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.data import AtomicDataDict
from typing import Optional, List
try:
    from entmax import sparsemax, entmax15, entmax_bisect, normmax_bisect, budget_bisect
except ImportError:
    sparsemax = entmax15 = entmax_bisect = normmax_bisect = budget_bisect = None

class EnsembleAggregator(GraphModuleMixin, Module):
    def __init__(self, irreps_in, field: str, out_field: Optional[str] = None, aggregation_method: str = "mean"):
        super().__init__()
        assert aggregation_method in ["mean", "sum", "max"], f"aggregation_method must be one of ['mean', 'sum', 'max'], got {aggregation_method}"
        self.field = field
        self.out_field = out_field or field
        self.aggregation_method = aggregation_method
        in_irreps = irreps_in[field]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: in_irreps},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if "ensemble_index" not in data: return data  # No ensemble index, nothing to aggregate
        assert self.field == AtomicDataDict.GRAPH_FEATURES_KEY
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
    def __init__(
        self,
        irreps_in,
        field: str,
        out_field: Optional[str],
        input_dim: int,
        softmax_func:str='softmax', # or 'entmax'
    ):
        assert softmax_func in ['softmax', 'entmax'], "softmax_func must be 'softmax' or 'entmax'"

        super().__init__()
        self.field = field
        self.out_field = out_field or field

        self.norm = torch.nn.LayerNorm(input_dim)
        self.hidden = 4*input_dim
        self.linear1 = torch.nn.Linear(input_dim, self.hidden)
        self.linear2 = torch.nn.Linear(input_dim, self.hidden)
        self.linear3 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear4 = torch.nn.Linear(self.hidden, 1)

        in_irreps = irreps_in[field]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: in_irreps},
        )
        self.split_index = [mul for mul,_ in self.irreps_in[self.field]][0]
        self.softmax_func = True if softmax_func == 'softmax' else False

    @torch.amp.autocast('cuda', enabled=False) # attention always kept to high precision, regardless of AMP
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if "ensemble_index" not in data: return data  # No ensemble index, nothing to aggregate
        assert self.field == AtomicDataDict.GRAPH_FEATURES_KEY
        features = data[self.field]
        ensemble_indices = data["ensemble_index"]

        features, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)

        # residual = features
        features = self.norm(features)
        a = self.linear1(features)
        b = self.linear2(features)
        features = self.linear3(a * torch.sigmoid(b))

        single_w_per_conf = self.linear4(features)
        # Get the unique ensemble indices and inverse mapping
        unique_idx, inverse = torch.unique(ensemble_indices, return_inverse=True)
        # Flatten single_w_per_conf to (N,)
        single_w_per_conf_flat = single_w_per_conf.squeeze(-1)

        if not self.softmax_func:
            # Compute entmax_bisect weights for each group (batch-wise)
            # entmax_bisect expects contiguous groups, so we process each group separately
            weights = torch.zeros_like(single_w_per_conf_flat)
            for idx in unique_idx:
                mask = (ensemble_indices == idx)
                group_scores = single_w_per_conf_flat[mask]
                group_weights = entmax_bisect(group_scores/0.2, alpha=1.9, dim=0)
                weights[mask] = group_weights

            features = features * weights.unsqueeze(-1)
            features = torch.cat((features, equiv), dim=-1)
            data[self.out_field] = features
            return data

        # Compute softmax weights for each group (batch-wise)
        # First, for numerical stability, subtract max per group
        max_per_group = scatter_max(single_w_per_conf_flat, inverse, dim=0)[0]
        max_per_group_expanded = max_per_group[inverse]
        exp_scores = torch.exp(single_w_per_conf_flat - max_per_group_expanded)
        sum_exp_per_group = scatter_sum(exp_scores, inverse, dim=0)
        sum_exp_per_group_expanded = sum_exp_per_group[inverse]
        softmax_weights = exp_scores / (sum_exp_per_group_expanded + 1e-8)
        features = features * softmax_weights.unsqueeze(-1)
        features = torch.cat((features, equiv), dim=-1)
        data[self.out_field] = features

        return data