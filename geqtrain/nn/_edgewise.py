import torch
import math
from typing import Optional
from einops.layers.torch import Rearrange
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_softmax
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """
    Sum edgewise features into nodes.

    Includes optional per-species-pair edgewise scales.
    """

    out_field: str
    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        use_attention: bool = False,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        attention_head_dim: int = 32,
        avg_num_neighbors: Optional[float] = 5.0,
        avg_num_neighbors_is_learnable: bool = False,
        irreps_in={},
    ):
        super().__init__()
        self.field = field
        self.use_attention = use_attention
        self.out_field = f"weighted_sum_{field}" if out_field is None else out_field

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps_in[field]},
            irreps_out={self.out_field: irreps_in[field]},
        )

        irreps = self.irreps_in[field]
        self.node_attr_to_query = None
        self.edge_feat_to_key = None

        if self.use_attention:
            self.edge_block_slices = []
            self.edge_scalar_slices = []
            self.node_scalar_slices = []
            self.attention_num_heads = 0
            self.n_scalars = 0
            self.n_node_scalars = 0

            max_mul = 0
            offset = 0
            for mul, ir in irreps:
                block_dim = mul * ir.dim
                self.edge_block_slices.append((offset, offset + block_dim, mul, ir.dim))
                if ir.l == 0 and ir.p == 1:
                    self.edge_scalar_slices.append((offset, offset + block_dim))
                    self.n_scalars += block_dim
                max_mul = max(max_mul, mul)
                offset += block_dim

            node_irreps = self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY]
            node_offset = 0
            for mul, ir in node_irreps:
                block_dim = mul * ir.dim
                if ir.l == 0 and ir.p == 1:
                    self.node_scalar_slices.append((node_offset, node_offset + block_dim))
                    self.n_node_scalars += block_dim
                node_offset += block_dim

            if self.n_scalars == 0:
                raise ValueError("EdgewiseReduce attention requires parity-even scalar edge features.")
            if self.n_node_scalars == 0:
                raise ValueError("EdgewiseReduce attention requires parity-even scalar node attributes.")

            self.attention_num_heads = max_mul

            if 'mlp_latent_dimensions' not in readout_latent_kwargs:
                readout_latent_kwargs['mlp_latent_dimensions'] = [64, 64]
            if 'zero_init_last_layer_weights' not in readout_latent_kwargs:
                readout_latent_kwargs['zero_init_last_layer_weights'] = True

            self.attention_head_dim = attention_head_dim
            self.isqrtd = 1 / math.sqrt(attention_head_dim)

            self.node_attr_to_query = readout_latent(
                mlp_input_dimension=self.n_node_scalars,
                mlp_output_dimension=self.attention_num_heads * self.attention_head_dim,
                **readout_latent_kwargs,
            )

            self.edge_feat_to_key = readout_latent(
                mlp_input_dimension=self.n_scalars,
                mlp_output_dimension=self.attention_num_heads * self.attention_head_dim,
                **readout_latent_kwargs,
            )

            self.rearrange_qk = Rearrange('e (h d) -> e h d', h=self.attention_num_heads, d=self.attention_head_dim)

        if not self.use_attention:
          if avg_num_neighbors_is_learnable:
            self.env_sum_normalization = torch.nn.Parameter(torch.as_tensor([avg_num_neighbors]).rsqrt())
          else:
            self.register_buffer("env_sum_normalization", torch.as_tensor([avg_num_neighbors]).rsqrt())
        else:
            self.env_sum_normalization = None

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_feat = data[self.field]

        num_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]

        if self.use_attention and self.node_attr_to_query is not None:
            node_attrs = data[AtomicDataDict.NODE_ATTRS_KEY]
            if len(self.node_scalar_slices) == 1:
                node_start, node_end = self.node_scalar_slices[0]
                node_scalars = node_attrs[:, node_start:node_end]
            else:
                node_scalars = torch.cat(
                    [node_attrs[:, start:end] for start, end in self.node_scalar_slices],
                    dim=-1,
                )
            Q = self.node_attr_to_query(node_scalars[edge_center])
            Q = self.rearrange_qk(Q)

            if len(self.edge_scalar_slices) == 1:
                edge_start, edge_end = self.edge_scalar_slices[0]
                edge_scalars = edge_feat[:, edge_start:edge_end]
            else:
                edge_scalars = torch.cat(
                    [edge_feat[:, start:end] for start, end in self.edge_scalar_slices],
                    dim=-1,
                )
            K = self.edge_feat_to_key(edge_scalars)
            K = self.rearrange_qk(K)

            W = torch.einsum('ehd,ehd -> eh', Q, K) * self.isqrtd
            attn_weights = scatter_softmax(W, edge_center, dim=0)
            weighted_blocks = []
            for start, end, mul, dim in self.edge_block_slices:
                block = edge_feat[:, start:end].reshape(edge_feat.shape[0], mul, dim)
                block = block * attn_weights[:, :mul].unsqueeze(-1)
                weighted_blocks.append(block.reshape(edge_feat.shape[0], mul * dim))
            edge_feat = torch.cat(weighted_blocks, dim=-1)

        # aggregation step
        data[self.out_field] = scatter_sum(edge_feat, edge_center, dim=0, dim_size=num_nodes)

        if not self.use_attention:
            assert self.env_sum_normalization is not None
            data[self.out_field] = data[self.out_field] * self.env_sum_normalization
        return data
