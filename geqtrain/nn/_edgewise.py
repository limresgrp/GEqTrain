import torch
import torch.nn as nn
from typing import Optional
from torch_runstats.scatter import scatter

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise features.

    Includes optional per-species-pair edgewise scales.
    """

    out_field: str
    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = f"weighted_sum_{field}" if out_field is None else out_field

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps_in[field]},
            irreps_out={out_field: irreps_in[field]},
        )

        # self.embed_size = embed_size
        # self.heads = heads
        # self.head_dim = embed_size // heads

        # self.values =  nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.keys =    nn.Linear(self.head_dim, self.head_dim, bias=False)
        # self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        edge_feat = data[self.field]
        species = data[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)
        num_nodes = len(species)

        # edges_per_node = torch.bincount(edge_center, minlength=num_nodes)
        # edge_center_incremental = torch.cat([torch.arange(0, num_edges) for num_edges in edges_per_node])
        # index_tensor = torch.stack([edge_center, edge_center_incremental], dim=0)
        
        # self.irreps_in[self.field][0].mul

        # sparse_tensor = torch.sparse_coo_tensor(index_tensor, b, torch.Size([num_nodes, edges_per_node.max()]))
        # softmax_values = torch.sparse.softmax(sparse_tensor, dim=1).values()

        data[self.out_field] = scatter(edge_feat, edge_center, dim=0, dim_size=num_nodes)
        return data