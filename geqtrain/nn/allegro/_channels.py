""" Adapted from https://github.com/mir-group/allegro
"""

from math import ceil

import torch

from e3nn.util.jit import compile_mode
from einops.layers.torch import Rearrange


@compile_mode("script")
class MakeWeightedChannels(torch.nn.Module):
    '''
    outputs a LC of input SH given some set of weights
    weights and input are provided from external code: this class does not contain learnable weights
    '''
    weight_numel: int
    multiplicity_out: int
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: int,
        pad_to_alignment: int = 1,
    ):
        super().__init__()
        assert all(mul == 1 for mul, ir in irreps_in)
        assert multiplicity_out >= 1
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        w_index = sum(([i] * ir.dim for i, (mul, ir) in enumerate(irreps_in)), [])
        # pad to padded length
        n_pad = (
            int(ceil(irreps_in.dim / pad_to_alignment)) * pad_to_alignment
            - irreps_in.dim
        )
        # use the last weight, what we use doesn't matter much
        w_index += [w_index[-1]] * n_pad
        self._num_irreps = len(irreps_in)
        self.register_buffer("_w_index", torch.as_tensor(w_index, dtype=torch.long))
        # there is
        self.multiplicity_out = multiplicity_out
        self.weight_numel = len(irreps_in) * multiplicity_out

        self.rearrange_weights = Rearrange(
            'e (m d) -> e m d',
            m=self.multiplicity_out,
            d=self._num_irreps
        )

    def forward(self, edge_attr, weights):
        # weights are [e, m, d], tensor of scalars
        # edge_attr are [e, d], geometric tensor
        # d runs over all irreps, which is why the weights need
        # to be indexed in order to go from [num_d] to [d]

        return torch.einsum(
            "ed,emd->emd",
            edge_attr,
            self.rearrange_weights(weights)[:, :, self._w_index],
        )
