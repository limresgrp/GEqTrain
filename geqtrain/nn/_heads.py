import re
import e3nn
from e3nn import o3

import math
# !pip install geometric-vector-perceptron
import torch
from einops.layers.torch import Rearrange
from torch import nn
# from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from typing import List, Optional
from geqtrain.utils import add_tags_to_module
from torch.nn import GroupNorm
from torch.nn import functional as F
from geqtrain.nn.mace.irreps_tools import reshape_irreps
from geqtrain.data import AtomicDataDict


class L0IndexedAttention(GraphModuleMixin, nn.Module):
    # for now let's suppose that the input is a scalar field
    # and that the output is a scalar field
    # and that in/out shapes are the same
    def __init__(self,
        irreps_in,
        field: str,
        idx_key:str,
        out_field: Optional[str] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        update_mlp:bool=False,
        use_radial_bias:bool=True,
        sparse_attention:bool=False,
        attention_prenorm:bool=False,
        learn_query:bool=False,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        in_irreps = irreps_out = irreps_in[field]
        self.idx_key = idx_key # 'batch' or 'ensemble_index'
        self.n_heads = num_heads
        self.dropout = dropout
        self.update_mlp=update_mlp

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
        )

        irreps_as_dict = {i:mul for i, (mul, l) in enumerate(in_irreps)}
        # assert len(irreps_as_dict) == 1, f'Head to predict {field} has equivariant out: {str(self.irreps_in[self.out_field])}'
        self.n_inpt_scalars = irreps_as_dict[0]
        self.kqv_norm = nn.LayerNorm(self.n_inpt_scalars)
        self.out_proj = nn.Linear(self.n_inpt_scalars, self.n_inpt_scalars)

        self.head_dim = self.n_inpt_scalars//self.n_heads
        self.scale = math.sqrt(self.head_dim)

        self.use_radial_bias = use_radial_bias and (field == AtomicDataDict.NODE_FEATURES_KEY) or (field == AtomicDataDict.NODE_ATTRS_KEY)
        if self.use_radial_bias:
            rbf_emb_dim = irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY].dim
            self.bias_norm = nn.LayerNorm(rbf_emb_dim)
            self.bias_proj = torch.nn.Sequential(
                # nn.Linear(rbf_emb_dim, 4*rbf_emb_dim, bias=False),
                # nn.ReLU(), # nn.SiLU(), # relu to try to enforce a bit of sparsity
                # nn.Linear(4*rbf_emb_dim, num_heads)
                nn.Linear(rbf_emb_dim, num_heads)
            )

        # from sd3.5 https://arxiv.org/pdf/2403.03206, 5.3.2
        self.attention_prenorm = attention_prenorm
        self.pre_norm_k = None
        self.pre_norm_q = None
        self.pre_norm_v = None
        if self.attention_prenorm:
            self.pre_norm_k = nn.LayerNorm(self.head_dim)
            self.pre_norm_q = nn.LayerNorm(self.head_dim)
            self.pre_norm_v = nn.LayerNorm(self.head_dim)

        self.mlp = None
        if self.update_mlp:
            self.mlp = ScalarMLPFunction(
                mlp_input_dimension=self.n_inpt_scalars,
                mlp_latent_dimensions=[4*self.n_inpt_scalars],
                mlp_output_dimension=self.n_inpt_scalars,
                mlp_nonlinearity = "swiglu",
            )

        self.sparse_attention = sparse_attention

        self.learn_query = learn_query
        self.kqv_proj_output_size_multiplier = 3
        if self.learn_query and self.idx_key in ['ensemble_index', AtomicDataDict.GRAPH_FEATURES_KEY]:
            self.kqv_proj_output_size_multiplier = 2

        self.kqv_proj = nn.Linear(self.n_inpt_scalars, self.kqv_proj_output_size_multiplier*self.n_inpt_scalars, bias=False)
        self.query_embedding = None
        if self.kqv_proj_output_size_multiplier == 2:
            self.query_embedding = nn.Parameter(torch.randn(self.n_inpt_scalars))

        self.rearrange1 = Rearrange('batch source target heads -> batch heads source target')
        self.rearrange2 = Rearrange('d -> 1 d')
        # self.rearrange_qk = Rearrange('e (c d) -> e c d', c=self.n_scalars, d=self.head_dim)


    def _add_edge_based_bias(self, data:AtomicDataDict.Type):
        edge_radial_attrs = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY] # already modulated wrt r_max; shape: (E, rbf_emb_size)
        edge_index        = data[AtomicDataDict.EDGE_INDEX_KEY] # (2, E)
        batch_map         = data[AtomicDataDict.BATCH_KEY] # assings each node to a given mol
        unique_idx, counts = torch.unique(batch_map, return_counts=True) # num_of mols, num of atoms per mol
        max_count = counts.max() # max num of nodes in atoms in batch
        num_uniques = unique_idx.shape[0] # num of mols in batch

        num_total_edges, num_edge_features = edge_radial_attrs.shape
        device = edge_radial_attrs.device
        counts = counts.to(device)

        # --- Precompute cumulative node counts ---
        # This helps find the starting global index for each graph in edge_index
        # cum_counts will store the starting index offset for each graph, used to index into edge_index
        # Example: if counts is [10, 5, 8], cum_counts will be [0, 10, 15]
        zero_pad = torch.zeros(1, dtype=counts.dtype, device=device) # tensor([0]) with shape: torch.Size([1])
        cum_counts = torch.cat([zero_pad, counts.cumsum(dim=0)[:-1]], dim=0) # starting from [0] appends cum sum of counts

        # --- Determine batch index for each edge ---
        # Since edges are within graphs, the batch index of the source node
        # determines the edge's batch index.
        src_nodes_global = edge_index[0]
        edge_batch_indices = batch_map[src_nodes_global] # Shape: [E_total], for each edge, get graph idx

        # --- Calculate local node indices for each edge ---
        # Subtract the cumulative count (start offset) of the corresponding batch
        # from the global node indices.
        batch_start_offsets = cum_counts[edge_batch_indices] # Shape: [E_total]

        local_src_indices = edge_index[0] - batch_start_offsets # Shape: [E_total]
        local_tgt_indices = edge_index[1] - batch_start_offsets # Shape: [E_total]

        # --- Initialize the padded output tensor ---
        padding_value = 0.0
        padded_edge_attrs = torch.full(
            [int(num_uniques), int(max_count), int(max_count), int(self.n_heads)],
            padding_value,
            dtype=edge_radial_attrs.dtype,
            device=device
        )

        updated_edge_radial_attrs = self.bias_norm(edge_radial_attrs)
        updated_edge_radial_attrs = self.bias_proj(updated_edge_radial_attrs)
        # updated_edge_radial_attrs = torch.sigmoid(updated_edge_radial_attrs)

        # --- Use advanced indexing to assign attributes ---
        # We use the calculated batch indices and local node indices to directly
        # place the edge attributes into the correct locations in the padded tensor.
        padded_edge_attrs[edge_batch_indices, local_src_indices, local_tgt_indices] = updated_edge_radial_attrs

        padded_edge_attrs = self.rearrange1(padded_edge_attrs)
        return padded_edge_attrs


    # @torch.amp.autocast('cuda', enabled=False) # attention always kept to high precision, regardless of AMP
    def forward(self, features, data: AtomicDataDict.Type) -> torch.Tensor:
        '''forward logic: https://rbcborealis.com/wp-content/uploads/2021/08/T17_7.png'''
        if self.idx_key == AtomicDataDict.GRAPH_FEATURES_KEY:
            attention_idxs = torch.arange(features.shape[0], device=features.device)
        else:
            attention_idxs = data[self.idx_key] # either 'batch' or 'ensemble_index'
        assert attention_idxs.shape[0] == features.shape[0], f"attention_idxs ({attention_idxs.shape[0]}) and input ({features.shape[0]}) shapes do not match, cannot apply attention on {self.field}, only on node or ensemble idx"

        N, emb_dim = features.shape # N = num nodes or N ensemble confs
        _dtype = features.dtype
        _device = features.device
        residual = features

        # fetch and preprocess attnt idxs
        unique_idx, counts = torch.unique(attention_idxs, return_counts=True)
        max_count = counts.max() # max number of atoms or conformers in batch
        num_uniques = unique_idx.shape[0] # number of unique mols in batch, regardless of self.idx_key

        # mask to select how many atoms or conformers are there in/for that mol
        mask = torch.arange(max_count, device=_device)[None, :] < counts[:, None] # for each mol select what is NOT padding
        # assert torch.all(mask.sum(-1) == counts) == True

        # map features to kqv
        features = self.kqv_norm(features)
        kvq = self.kqv_proj(features)

        if self.kqv_proj_output_size_multiplier == 2:
            _k, _v = torch.chunk(kvq, 2, dim=-1)
            if self.query_embedding is not None:
                _q = self.rearrange2(self.query_embedding)
            else:
                # fallback: use _k as _q if no query_embedding is provided
                _q = _k
        else:
            _k, _q, _v = torch.chunk(kvq, 3, dim=-1)

        q = torch.zeros(num_uniques, max_count, emb_dim, device=_device, dtype=_dtype) # padded_q
        k = torch.zeros(num_uniques, max_count, emb_dim, device=_device, dtype=_dtype) # padded_k; in LLMs this is S,T,K
        v = torch.zeros(num_uniques, max_count, emb_dim, device=_device, dtype=_dtype) # padded_v

        k[mask] = _k # set data into padded tensors
        q[mask] = _q
        v[mask] = _v

        k = k.view(num_uniques, max_count, self.n_heads, self.head_dim) # split emb in nhead chunks for MHA
        q = q.view(num_uniques, max_count, self.n_heads, self.head_dim)
        v = v.view(num_uniques, max_count, self.n_heads, self.head_dim)

        k = k.transpose(1,2) # num_uniques, self.n_heads, max_count, self.head_dim
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        if self.attention_prenorm:
            if self.pre_norm_k is not None: k = self.pre_norm_k(k)
            if self.pre_norm_q is not None: q = self.pre_norm_q(q)
            if self.pre_norm_v is not None: v = self.pre_norm_v(v)

        qvt = q @ k.transpose(-1, -2) # square matrix of size (..., max_count, max_count)
        qvt = qvt / self.scale # scale by sqrt of head_dim

        bidirectional_mask = qvt == 0.0 # select all elements that come out the prod of rows/cols of zeros
        row_all_true_mask = bidirectional_mask.all(dim=-1, keepdim=True).expand_as(bidirectional_mask) # select rows that have only zeros (s.t. exclude them from softmax otherwise autograd breaks)

        if self.use_radial_bias:
            qvt += self._add_edge_based_bias(data)

        fill_value = -torch.inf # Or a large negative number like -1e9
        qvt.masked_fill_(bidirectional_mask, fill_value) # set all zeros to -inf
        qvt[row_all_true_mask] = 0.0 # replace rows of ALL -inf to 0s to avoid #!RuntimeError: Function 'SoftmaxBackward0' returned nan values in its 0th output; (but keep -inf if other vals in row are populated)

        # all the above has been done to set -inf maksing in in rows populated, and zeros in rows that are completely empty (due to padding)

        # if self.sparse_attention:
        #     # attnt_coeffs = entmax_bisect(qvt, alpha=1.3, dim=-1) # optional values for alpha 1.2. 1.25 1.3
        #     pass
        # else:
        attnt_coeffs = F.softmax(qvt, dim=-1)

        attnt_coeffs = attnt_coeffs.masked_fill(row_all_true_mask, 0.0) # zero-out all "padding only" rows

        # assert torch.allclose(
        #     attnt_coeffs.sum(-1).sum(-1)[:, 0],
        #     counts.float(),
        #     atol=1e-6
        # ), "Attention coefficients do not sum to the expected counts."

        temp_out_to_be_cat = attnt_coeffs @ v # so new we get back to shape: (bs, h, t, hdim)
        # goal shape: (bs, nh*t, c)
        temp_out_to_be_cat = temp_out_to_be_cat.transpose(1, 2) # bs t h hdim; such that we have the same input dimensions positions, safe reshaping
        tmp_out = temp_out_to_be_cat.reshape(num_uniques, max_count, self.n_heads*self.head_dim) # revert to input_shape
        tmp_out = tmp_out[mask]

        out = self.out_proj(tmp_out)
        out+=residual

        if self.update_mlp and self.mlp is not None:
            out = out+self.mlp(out)

        return out