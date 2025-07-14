import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn._embedding import BaseEmbedding


@compile_mode("script")
class AllegroEdgeEmbedding(BaseEmbedding):

    def __init__(self, latent_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.has_edge_attr  = False
        self.proj_to_latent = False
        
        # irreps
        node_irreps = self.irreps_in[self.node_field]
        edge_radial_irreps = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        
        # dims
        edge_dim = 2 * node_irreps.dim
        if self.edge_field in self.irreps_in:
            self.has_edge_attr = True
            edge_dim += self.irreps_in[self.edge_field].dim
        
        out_dim = edge_dim
        if latent_dim is not None:
            self.proj_to_latent = True
            out_dim = latent_dim
            self.register_parameter("linear_proj", torch.nn.Parameter(torch.randn(edge_dim, out_dim)))
            torch.nn.init.xavier_uniform_(self.linear_proj)

        self.register_parameter("radial_linear_proj", torch.nn.Parameter(torch.randn(edge_radial_irreps.dim, out_dim)))
        torch.nn.init.xavier_uniform_(self.radial_linear_proj)
        self.out_irreps = o3.Irreps(f"{out_dim}x0e")
    
    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_center, edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_attr               = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        node_attr               = data[self.node_field]

        edge_node_attrs_to_cat = [node_attr[edge_center], node_attr[edge_neigh]]
        if self.has_edge_attr:
            edge_node_attrs_to_cat.append(data.pop(self.edge_field))
        edge_node_attrs_emb = torch.cat(edge_node_attrs_to_cat, dim=-1)
        if self.proj_to_latent:
            edge_node_attrs_emb = edge_node_attrs_emb @ self.linear_proj
        
        edge_radial_emb_proj = edge_attr @ self.radial_linear_proj
        return edge_node_attrs_emb * edge_radial_emb_proj