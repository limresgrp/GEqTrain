import re
import e3nn

# !pip install geometric-vector-perceptron
import torch
from einops import rearrange
from torch import nn
from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm
from geqtrain.nn import GraphModuleMixin
from typing import List, Optional



class FFBlock(torch.nn.Module):
    def __init__(self, inp_size, out_size:int|None, residual:bool=True):
        super().__init__()
        self.residual = residual
        out_size = out_size or inp_size
        self.block = torch.nn.Sequential(
            torch.nn.LayerNorm(inp_size),
            torch.nn.Linear(inp_size, 4*inp_size),
            torch.nn.SiLU(),
            torch.nn.Linear(4*inp_size, out_size)
        )

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


class EnsembleMHA(nn.Module):
    def __init__(self, input_dims, out_dims):
        super().__init__()
        pass



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

        self.norm1 = nn.LayerNorm(self.l0_size)
        self.attention1 = nn.MultiheadAttention(embed_dim=self.l0_size, num_heads=8, dropout=0.1)
        self.ff_block1 = FFBlock(self.l0_size, self.l0_size)

        self.norm2 = nn.LayerNorm(self.l0_size)
        self.attention2 = nn.MultiheadAttention(embed_dim=self.l0_size, num_heads=8, dropout=0.1)
        self.ff_block2 = FFBlock(self.l0_size, self.l0_size)

        self.norm3 = nn.LayerNorm(self.l0_size)
        self.attention3 = nn.MultiheadAttention(embed_dim=self.l0_size, num_heads=8, dropout=0.1)
        self.final_block = FFBlock(self.l0_size,1,residual=False)

        self.kqv_proj1 = FFBlock(self.l0_size, 3*self.l0_size, residual=False)
        self.kqv_proj2 = FFBlock(self.l0_size, 3*self.l0_size, residual=False)
        self.kqv_proj3 = FFBlock(self.l0_size, 3*self.l0_size, residual=False)

    def forward(self, data):
        features = data[self.field]
        feats, _ = torch.split(features, [self.l0_size, self.l1_size], dim=-1)

        ensemble_idxs = data['ensemble_index']
        unique_indices = torch.unique(ensemble_idxs)
        split_tensors = [feats[ensemble_idxs == idx] for idx in unique_indices]

        out = []
        for f in split_tensors: # ugly but ok, could be vectorized via bmm (?)
            # feats = self.norm1(f)
            k, q, v = torch.chunk( self.kqv_proj1(f), 3, dim=-1)
            attn_output, _ = self.attention1(k, q, v)
            feats = self.ff_block1(attn_output)

            # feats = self.norm2(feats)
            k, q, v = torch.chunk( self.kqv_proj2(feats), 3, dim=-1)
            attn_output, _ = self.attention2(k, q, v)
            feats = self.ff_block2(attn_output)

            # feats = self.norm3(feats)
            k, q, v = torch.chunk( self.kqv_proj3(feats), 3, dim=-1)
            attn_output, _ = self.attention3(k, q, v)
            ff_output = self.final_block(attn_output)
            out.append(ff_output)

        # Reconstruct x
        reconstructed_x = torch.zeros(data[self.field].shape[0], 1, dtype=out[0].dtype, device=out[0].device)
        for i, idx in enumerate(unique_indices):
            reconstructed_x[ensemble_idxs == idx] = out[i]

        data[self.out_field] = reconstructed_x # expected out shape: torch.Size([bs, 1])
        return data