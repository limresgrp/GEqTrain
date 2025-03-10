import re
import e3nn
from e3nn import o3


# !pip install geometric-vector-perceptron
import torch
from einops import rearrange
from torch import nn
from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm
from geqtrain.nn import GraphModuleMixin
from typing import List, Optional
from geqtrain.utils import add_tags_to_module
from torch.nn import GroupNorm

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

        self.l1 = EnsembleBasedAttention(irreps_in,field,out_field)
        self.l2 = EnsembleBasedAttention(irreps_in,field,out_field)
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


class EnsembleBasedAttention(GraphModuleMixin, nn.Module):
    # for now let's suppose that the input is a scalar field
    # and that the output is a scalar field
    # and that in/out shapes are the same
    def __init__(self,
        irreps_in,
        field: str,
        out_field: Optional[str] = None,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        in_irreps = irreps_out = irreps_in[field]

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_out},
        )

        irreps_as_dict = {i:mul for i, (mul, l) in enumerate(in_irreps)}
        # assert len(irreps_as_dict) == 1, f'Head to predict {field} has equivariant out: {str(self.irreps_in[self.out_field])}'
        self.n_inpt_scalars = irreps_as_dict[0]
        self.kqv_proj = FFBlock(self.n_inpt_scalars, 3*self.n_inpt_scalars, residual=False)
        self.attention = nn.MultiheadAttention(embed_dim=self.n_inpt_scalars, num_heads=num_heads, dropout=dropout)
        self.ff_block = FFBlock(self.n_inpt_scalars)

    @torch.cuda.amp.autocast(enabled=False) # attention always kept to high precision, regardless of AMP
    def forward(self, feats, ensemble_idxs):
        unique_indices = torch.unique(ensemble_idxs)
        split_tensors = [feats[ensemble_idxs == idx] for idx in unique_indices]
        bs, emb_dim = feats.shape

        out = []
        for f in split_tensors: # ugly but ok for now, can be vectorized via bmm (?) or padding / batching
            feats = self.kqv_proj(f)
            # feats = self.dropout(feats)
            k, q, v = torch.chunk(feats, 3, dim=-1)
            attn_output, _ = self.attention(k, q, v)
            attn_output += f
            feats = self.ff_block(attn_output)
            feats +=attn_output
            # feats = self.dropout(feats)
            out.append(feats)

        # Reconstruct x
        reconstructed_x = torch.zeros(bs, emb_dim, dtype=out[0].dtype, device=out[0].device)
        for i, idx in enumerate(unique_indices):
            reconstructed_x[ensemble_idxs == idx] = out[i]
        return reconstructed_x


    # def forward(self, data):
        # ensemble_idxs = data['ensemble_index']
        # feats = data[self.field]

        # # Get unique indices and counts
        # unique_indices, inverse_indices, counts = torch.unique(ensemble_idxs, return_inverse=True, return_counts=True)
        # max_count = counts.max()

        # # Create padded tensor with shape (num_ensembles, max_count, feat_dim)
        # padded_feats = torch.zeros(len(unique_indices), max_count, self.l0_size, device=feats.device, dtype=feats.dtype)

        # # Create mask for valid positions
        # mask = torch.arange(max_count, device=feats.device)[None, :] < counts[:, None]

        # # Fill the padded tensor
        # padded_feats[mask] = feats

        # # Process all ensembles in parallel
        # k, q, v = torch.chunk(self.kqv_proj(padded_feats), 3, dim=-1)

        # k+= padded_feats
        # q+= padded_feats
        # v+= padded_feats

        # attn_output, _ = self.attention(k.transpose(0, 1), q.transpose(0, 1), v.transpose(0, 1))
        # feats = self.ff_block(attn_output)

        # data[self.field] = feats[mask][inverse_indices]
        # return data

    # todo add here decorator for amp (at least on attnt)
    # def forward(self, feats, ensemble_idxs):
    #! NOT WORKING
    #     input = feats
    #     feats = self.kqv_proj(feats) + torch.cat([input, input, input], dim=-1)
    #     # get num of unique mols (i.e. how many ensembles in batch) and the num of conformers in each ensemble
    #     unique_indices, counts = torch.unique(ensemble_idxs, return_counts=True)
    #     max_count = counts.max()

    #     # Create tensor filled with padding with shape (num_ensembles, max_count, feat_dim)
    #     padded_feats = torch.zeros(len(unique_indices), max_count, 3*self.n_inpt_scalars, device=feats.device, dtype=feats.dtype)

    #     # Create mask for valid positions
    #     mask = torch.arange(max_count, device=feats.device)[None, :] < counts[:, None]

    #     # Fill the padded tensor with data
    #     padded_feats[mask] = feats

    #     # Process all ensembles in parallel
    #     k, q, v = torch.chunk(padded_feats, 3, dim=-1)

    #     attn_output, _ = self.attention(k.transpose(0, 1), q.transpose(0, 1), v.transpose(0, 1))
    #     # attn_output = attn_output.transpose(0, 1) + input.unsqueeze(0)
    #     feats = self.ff_block(attn_output)

    #     return feats[mask][ensemble_idxs]