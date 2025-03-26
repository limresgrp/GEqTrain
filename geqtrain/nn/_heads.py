import re
import e3nn
from e3nn import o3

import math
# !pip install geometric-vector-perceptron
import torch
from einops import rearrange
from torch import nn
from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm
from geqtrain.nn import GraphModuleMixin
from typing import List, Optional
from geqtrain.utils import add_tags_to_module
from torch.nn import GroupNorm
from torch.nn import functional as F


class FFBlock(torch.nn.Module):
    def __init__(self, inp_size, out_size:int|None=None, residual:bool=True, group_norm:bool=False):
        super().__init__()
        self.residual = residual
        out_size = out_size or inp_size
        self.block_list = [GroupNorm(num_groups=8, num_channels=inp_size)] if group_norm else [torch.nn.LayerNorm(inp_size)]
        self.block_list.extend([
            torch.nn.Linear(inp_size, 4*inp_size, bias=False), # bias = false by default since LN always present
            torch.nn.SiLU(),
            torch.nn.Linear(4*inp_size, out_size)
        ])
        self.block = torch.nn.Sequential(*self.block_list)

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out += x
        return out


class GVPGeqTrain(GraphModuleMixin, nn.Module):
    '''https://github.com/lucidrains/geometric-vector-perceptron'''
    def __init__(self, irreps_in, field: str, out_field: Optional[str] = None):
        super().__init__()
        self.field = field
        self.out_field = out_field
        in_irreps = irreps_in[field]
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: in_irreps}, # TODO FIX not real
        )

        self.dropout = GVPDropout(0.2)

        self.norm1 = GVPLayerNorm(64)
        self.layer1 = GVP(
            dim_vectors_in = 64, # env_embed_multiplicity
            dim_feats_in = 64, # l0 dim (i.e. latent_dim)

            dim_vectors_out = 128, # out_multiplicity
            dim_feats_out = 128, # new latent_dim

            vector_gating = True
        )

        self.norm2 = GVPLayerNorm(128)
        self.layer2 = GVP(
            dim_vectors_in = 128, # env_embed_multiplicity
            dim_feats_in = 128, # l0 dim (i.e. latent_dim)

            dim_vectors_out = 256, # out_multiplicity
            dim_feats_out = 256, # new latent_dim
            vector_gating = True
        )

        # for scalar property
        self.norm3 = GVPLayerNorm(256)
        self.final_layer = GVP(
            dim_vectors_in = 256, # env_embed_multiplicity # 64x1o
            dim_feats_in = 256, # l0 dim (i.e. latent_dim) # 512x0e

            dim_vectors_out = 64, # out_multiplicity
            dim_feats_out = 1, # new latent_dim

            vector_gating = True
        )
        self.l0_size = self.irreps_in[self.field][0].dim
        self.l1_size = self.irreps_in[self.field][1].dim

    def forward(self, data):
        features = data[self.field]

        feats, vectors = torch.split(features, [self.l0_size, self.l1_size], dim=-1)
        vectors = rearrange(vectors, "b (v c) -> b v c ", c=3)

        feats, vectors = self.norm1(feats, vectors)
        feats, vectors = self.layer1((feats, vectors))
        feats, vectors = self.dropout(feats, vectors)

        feats, vectors = self.norm2(feats, vectors)
        feats, vectors = self.layer2((feats, vectors))
        feats, vectors = self.dropout(feats, vectors)

        feats, vectors = self.norm3(feats, vectors)
        feats, vectors = self.final_layer((feats, vectors))

        data[self.out_field] = feats
        return data


class WeightedTP(GraphModuleMixin, nn.Module):
    def __init__(self, irreps_in, field: str, out_field: Optional[str] = None):
        '''
        #! with this i can also use higher lmaxs that get actually used in pred
        '''
        super().__init__()

        self.field = field
        self.out_field = out_field
        in_irreps = irreps_in[field]

        irreps_out = e3nn.o3.Irreps('1x0e')
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
        )

        self.l0_size = self.irreps_in[self.field][0].dim
        self.l1_size = self.irreps_in[self.field][1].dim

        scalars_out = 128
        irreps_out = f'{scalars_out}x0e'

        self.tp = e3nn.o3.FullyConnectedTensorProduct(
            irreps_in1=in_irreps,
            irreps_in2=in_irreps,
            irreps_out=irreps_out,
            internal_weights=False,
            irrep_normalization='component',
            path_normalization='element',
            shared_weights = False,
        )

        self.weights_embedder = FFBlock(self.l0_size, self.tp.weight_numel, residual=False)
        self.out_mlp = FFBlock(scalars_out, 1, residual=False)

    def __call__(self, data):
        x = data[self.field]
        # get scalars
        feats, _ = torch.split(x, [self.l0_size, self.l1_size], dim=-1)

        tp_weights = self.weights_embedder(feats)

        # x_tp_inpt = rearrange(x, "b (v c) -> b v c ", c=4)
        x = self.tp(x,x, tp_weights)

        data[self.out_field] = self.out_mlp(x)

        return data


class TransformerBlock(GraphModuleMixin, nn.Module):
    def __init__(self, irreps_in, field: str, out_field: Optional[str] = None):
        super().__init__()
        self.field = field
        self.out_field = out_field
        in_irreps = irreps_in[field]

        irreps_out = e3nn.o3.Irreps('1x0e')
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
        )

        self.l0_size = self.irreps_in[self.field][0].dim
        self.l1_size = self.irreps_in[self.field][1].dim

        # self.attention1 = nn.MultiheadAttention(embed_dim=self.l0_size, num_heads=8, dropout=0.1)
        # self.ff_block1 = FFBlock(self.l0_size, self.l0_size)

        # self.attention2 = nn.MultiheadAttention(embed_dim=self.l0_size, num_heads=8, dropout=0.1)
        # self.ff_block2 = FFBlock(self.l0_size, self.l0_size)

        # self.attention3 = nn.MultiheadAttention(embed_dim=self.l0_size, num_heads=8, dropout=0.1)
        # self.ff_block3 = FFBlock(self.l0_size, self.l0_size,residual=True)

        self.final_block = FFBlock(self.l0_size, 1, residual=False)

        # self.kqv_proj1 = FFBlock(self.l0_size, 3*self.l0_size, residual=False, group_norm=True)
        # self.kqv_proj2 = FFBlock(self.l0_size, 3*self.l0_size, residual=False)
        # self.kqv_proj3 = FFBlock(self.l0_size, 3*self.l0_size, residual=False)

        # self.dropout = nn.Dropout(.2)

        self.l1 = L0IndexedAttention(irreps_in,field,out_field)
        self.l2 = L0IndexedAttention(irreps_in,field,out_field)
        add_tags_to_module(self, '_wd')

    def forward(self, data):
        features = data[self.field]
        feats, _ = torch.split(features, [self.l0_size, self.l1_size], dim=-1)
        feats = self.l1(feats, data['ensemble_index'])
        feats = self.l2(feats, data['ensemble_index'])
        data[self.out_field] = self.final_block(feats)
        return data

    # def forward(self, data):
    #     features = data[self.field]
    #     feats, _ = torch.split(features, [self.l0_size, self.l1_size], dim=-1)
    #     ensemble_idxs = data['ensemble_index']

    #     # Get unique indices and counts
    #     unique_indices, inverse_indices, counts = torch.unique(ensemble_idxs, return_inverse=True, return_counts=True)
    #     max_count = counts.max()

    #     # Create padded tensor with shape (num_ensembles, max_count, feat_dim)
    #     padded_feats = torch.zeros(len(unique_indices), max_count, self.l0_size,
    #                              device=feats.device, dtype=feats.dtype)

    #     # Create mask for valid positions
    #     mask = torch.arange(max_count, device=feats.device)[None, :] < counts[:, None]

    #     # Fill the padded tensor
    #     padded_feats[mask] = feats

    #     # Process all ensembles in parallel
    #     k, q, v = torch.chunk(self.kqv_proj1(padded_feats), 3, dim=-1)
    #     attn_output, _ = self.attention1(k.transpose(0, 1), q.transpose(0, 1), v.transpose(0, 1))
    #     attn_output = self.dropout(attn_output.transpose(0, 1))
    #     feats = self.ff_block1(attn_output)
    #     feats = self.dropout(feats)

    #     k, q, v = torch.chunk(self.kqv_proj2(feats), 3, dim=-1)
    #     attn_output, _ = self.attention2(k.transpose(0, 1), q.transpose(0, 1), v.transpose(0, 1))
    #     attn_output = self.dropout(attn_output.transpose(0, 1))
    #     feats = self.ff_block2(attn_output)
    #     feats = self.dropout(feats)

    #     k, q, v = torch.chunk(self.kqv_proj3(feats), 3, dim=-1)
    #     attn_output, _ = self.attention3(k.transpose(0, 1), q.transpose(0, 1), v.transpose(0, 1))
    #     attn_output = self.dropout(attn_output.transpose(0, 1))
    #     feats = self.ff_block3(attn_output)
    #     feats = self.dropout(feats)

    #     ff_output = self.final_block(feats)

    #     # Gather results back to original shape
    #     data[self.out_field] = ff_output[mask][inverse_indices]
    #     return data


# def forward(self, data):
#     features = data[self.field]
#     feats, _ = torch.split(features, [self.l0_size, self.l1_size], dim=-1)

#     ensemble_idxs = data['ensemble_index']
#     unique_indices = torch.unique(ensemble_idxs)
#     split_tensors = [feats[ensemble_idxs == idx] for idx in unique_indices]

#     out = []
#     for f in split_tensors: # ugly but ok, could be vectorized via bmm (?)
#         k, q, v = torch.chunk(self.kqv_proj1(f), 3, dim=-1)
#         attn_output, _ = self.attention1(k, q, v)
#         attn_output = self.dropout(attn_output)
#         feats = self.ff_block1(attn_output)
#         feats = self.dropout(feats)

#         k, q, v = torch.chunk(self.kqv_proj2(feats), 3, dim=-1)
#         attn_output, _ = self.attention2(k, q, v)
#         attn_output = self.dropout(attn_output)
#         feats = self.ff_block2(attn_output)
#         feats = self.dropout(feats)

#         k, q, v = torch.chunk(self.kqv_proj3(feats), 3, dim=-1)
#         attn_output, _ = self.attention3(k, q, v)
#         attn_output = self.dropout(attn_output)
#         feats = self.ff_block3(attn_output)
#         feats = self.dropout(feats)

#         ff_output = self.final_block(feats)
#         out.append(ff_output)

#     # Reconstruct x
#     reconstructed_x = torch.zeros(data[self.field].shape[0], 1, dtype=out[0].dtype, device=out[0].device)
#     for i, idx in enumerate(unique_indices):
#         reconstructed_x[ensemble_idxs == idx] = out[i]

#     data[self.out_field] = reconstructed_x # expected out shape: torch.Size([bs, 1])
#     return data


# TODO: IDEA MAKE IT as a possible replacement for ScalarMLPFunction
'''mlp_input_dimension: Optional[int],
mlp_latent_dimensions: List[int],
mlp_output_dimension: Optional[int],
mlp_nonlinearity: Optional[str] = "silu",
use_layer_norm: bool = True,
use_weight_norm: bool = False,
dim_weight_norm: int = 0,
has_bias: bool = False,
bias: Optional[List] = None,
zero_init_last_layer_weights: bool = False,
dropout: Optional[float] = None,
dampen: bool = False,
wd: bool = False,
gain:Optional[float] = None'''

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
        dropout: float = 0.0
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        in_irreps = irreps_out = irreps_in[field]
        self.idx_key = idx_key # 'batch' or 'ensemble_index'
        self.n_heads = num_heads
        self.dropout = dropout

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
        )

        irreps_as_dict = {i:mul for i, (mul, l) in enumerate(in_irreps)}
        # assert len(irreps_as_dict) == 1, f'Head to predict {field} has equivariant out: {str(self.irreps_in[self.out_field])}'
        self.n_inpt_scalars = irreps_as_dict[0]
        self.kqv_norm = nn.LayerNorm(self.n_inpt_scalars)
        self.kqv_proj = nn.Linear(self.n_inpt_scalars, 3*self.n_inpt_scalars)
        self.out_proj = nn.Linear(self.n_inpt_scalars, self.n_inpt_scalars)

        self.head_dim =  self.n_inpt_scalars//self.n_heads
        self.scale = math.sqrt(self.head_dim)

        self.bias_norm = nn.LayerNorm(16)
        self.bias_proj = torch.nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads, bias=False)
        )# nn.Linear(16, num_heads, bias=False)


    def _add_edge_based_bias(self, data):
        edge_radial_attrs = data["edge_radial_attrs"]
        edge_index = data["edge_index"]
        batch_map = data['batch'] # Renamed from 'batch' to avoid conflict with loop var
        unique_idx, counts = torch.unique(batch_map, return_counts=True)
        max_count = counts.max()
        num_uniques = unique_idx.shape[0]

        num_total_edges, num_edge_features = edge_radial_attrs.shape
        device = edge_radial_attrs.device

        # --- Precompute cumulative node counts ---
        # This helps find the starting global index for each graph
        # Ensure counts is on the correct device
        counts = counts.to(device)
        # cum_counts will store the starting index offset for each graph
        # Example: if counts is [10, 5, 8], cum_counts will be [0, 10, 15]
        zero_pad = torch.zeros(1, dtype=counts.dtype, device=device)
        cum_counts = torch.cat([zero_pad, counts.cumsum(dim=0)[:-1]], dim=0)

        # --- Determine batch index for each edge ---
        # Since edges are within graphs, the batch index of the source node
        # determines the edge's batch index.
        src_nodes_global = edge_index[0]
        edge_batch_indices = batch_map[src_nodes_global] # Shape: [E_total]

        # --- Calculate local node indices for each edge ---
        # Subtract the cumulative count (start offset) of the corresponding batch
        # from the global node indices.
        batch_start_offsets = cum_counts[edge_batch_indices] # Shape: [E_total]

        local_src_indices = edge_index[0] - batch_start_offsets # Shape: [E_total]
        local_tgt_indices = edge_index[1] - batch_start_offsets # Shape: [E_total]

        # --- Initialize the padded output tensor ---
        padding_value = 0.0
        padded_edge_attrs = torch.full(
            (num_uniques, max_count, max_count, self.n_heads),
            padding_value,
            dtype=edge_radial_attrs.dtype,
            device=device
        )

        updated_edge_radial_attrs = self.bias_norm(edge_radial_attrs)
        updated_edge_radial_attrs = self.bias_proj(updated_edge_radial_attrs)
        updated_edge_radial_attrs = torch.nn.functional.sigmoid(updated_edge_radial_attrs)

        # --- Use advanced indexing to assign attributes ---
        # We use the calculated batch indices and local node indices to directly
        # place the edge attributes into the correct locations in the padded tensor.
        padded_edge_attrs[edge_batch_indices, local_src_indices, local_tgt_indices] = updated_edge_radial_attrs

        return padded_edge_attrs



    @torch.cuda.amp.autocast(enabled=False) # attention always kept to high precision, regardless of AMP
    def forward(self, features, data):
        N, emb_dim = features.shape # N = num nodes or num edge or num ensemble confs

        attention_idxs = data[self.idx_key]
        assert attention_idxs.shape[0] == features.shape[0], f"attention_idxs ({attention_idxs.shape[0]}) and input ({features.shape[0]}) shapes do not match, cannot apply attention on {self.field}, only on node or ensemble idx"

        _dtype = features.dtype
        _device = features.device

        residual = features

        features = self.kqv_norm(features)
        kvq = self.kqv_proj(features)
        _k, _q, _v = torch.chunk(kvq, 3, dim=-1)

        unique_idx, counts = torch.unique(attention_idxs, return_counts=True)
        max_count = counts.max()
        num_uniques = unique_idx.shape[0]

        k = torch.zeros(num_uniques, max_count, emb_dim, device=_device, dtype=_dtype) # padded_k
        q = torch.zeros(num_uniques, max_count, emb_dim, device=_device, dtype=_dtype) # padded_q
        v = torch.zeros(num_uniques, max_count, emb_dim, device=_device, dtype=_dtype) # padded_v

        mask = torch.arange(max_count, device=_device)[None, :] < counts[:, None]

        k[mask] = _k
        q[mask] = _q
        v[mask] = _v

        k = k.view(num_uniques, max_count, self.n_heads, self.head_dim)
        q = q.view(num_uniques, max_count, self.n_heads, self.head_dim)
        v = v.view(num_uniques, max_count, self.n_heads, self.head_dim)

        k = k.transpose(1,2) # num_uniques, self.n_heads, max_count, self.head_dim
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        bias_ = self._add_edge_based_bias(data)

        qvt = q @ k.transpose(-1, -2) # square matrix of size (..., max_count, max_count)
        qvt /= self.scale
        bias_ = rearrange(bias_, 'batch target source heads -> batch heads target source')
        qvt *= bias_
        attnt_coeffs = F.softmax(qvt, dim=-1)

        temp_out_to_be_cat = attnt_coeffs @ v # so new we get back to shape: (bs, h, t, hdim)
        # goal shape: (bs, nh*t, c)
        temp_out_to_be_cat = temp_out_to_be_cat.transpose(1, 2) # bs t h hdim; such that we have the same input dimensions positions, safe reshaping
        tmp_out = temp_out_to_be_cat.reshape(num_uniques, max_count, self.n_heads*self.head_dim) # revert to input_shape
        tmp_out = tmp_out[mask]

        out = self.out_proj(tmp_out)
        out+=residual

        return out


class L1Scalarizer(GraphModuleMixin, nn.Module): # todo allow 4 options: 1) norms only 2) cosine similarity 3) both 4) dot prod
    def __init__(self, irreps_in, field: str, out_field: Optional[str] = None, norm_order:int=2, output_l1:bool=True):
        super().__init__()
        self.field = field
        self.out_field = out_field or field
        self.norm_order = norm_order
        self.output_l1 = output_l1

        in_irreps = irreps_in[field]

        self.l0_size = in_irreps.ls.count(0)
        self.l1_size = 3*in_irreps.ls.count(1)
        out_irreps = o3.Irreps(str(irreps_in[self.field])+f'+{self.l1_size}x0e').regroup()


        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: out_irreps},
        )

    def forward(self, data):
        # todo check if this works also for if lmax=2
        features = data[self.field]

        feats, vectors = torch.split(features, [self.l0_size, self.l1_size], dim=-1)
        _vectors = rearrange(vectors, "b (v c) -> b v c ", c=3)

        sh = torch.norm(_vectors, p = self.norm_order, dim = -1)

        out = torch.cat((feats, sh), dim = 1) # scalars with norms
        if self.output_l1:
            out = torch.cat((out, vectors), dim = 1) # scalars with norms and l1s

        data[self.out_field] = out
        return data