import math
from typing import Optional
import torch

from e3nn import o3
from einops import rearrange


class SO3_Linear(torch.nn.Module):
    def __init__(
            self,
            in_irreps: o3.Irreps,
            out_irreps: o3.Irreps,
            bias: bool = True,
            normalization: Optional[str] = None,
            eps: float = 1e-5,
        ):
        '''
        Initializes the SO3_Linear layer.

        Args:
            in_irreps (o3.Irreps): Input irreducible representations.
            out_irreps (o3.Irreps): Output irreducible representations.
            bias (bool): Whether to include a bias term. Default is True.
            normalization (Optional[str]): Normalization method, either 'norm' or 'component'. Default is None.
            eps (float): A small value to avoid division by zero in normalization. Default is 1e-5.

        Attributes:
            in_irreps (o3.Irreps): Stored input irreducible representations.
            out_irreps (o3.Irreps): Stored output irreducible representations.
            mul (int): Multiplicity of the irreducible representations.
            bias (torch.nn.Parameter or None): Bias term if `bias` is True, else None.
            params (torch.nn.ParameterDict): Dictionary of weight parameters for different l values.
            l_dims_in (list): List of input dimensions for each l value.
            l_dims_out (list): List of output dimensions for each l value.
        
        Example Usage:
            in_irreps = o3.Irreps('8x0e+8x0e+8x1o')
            out_irreps = o3.Irreps('8x0e+8x1o+8x1o')
            
            so3_linear = SO3_Linear(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                bias=True,
                normalization='component'
            )
            
            x = in_irreps.randn(10, -1).reshape(10, 8, -1)
            output = so3_linear(x)
        '''

        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.mul = in_irreps[0].mul
        self.bias = None

        if normalization is not None:
            assert normalization in ['norm', 'component']
        self.normalization = normalization
        self.eps = eps
        
        params = {}
        l_dims_in = []
        l_dims_out = []

        for l in set(out_irreps.ls):
            l_in_irr = [irr for irr in in_irreps if irr.ir.l == l]
            assert all([irr.mul == self.mul for irr in l_in_irr])  # assert all have same multiplicity
            in_features = self.mul * len(l_in_irr)
            l_out_irr = [irr for irr in out_irreps if irr.ir.l == l]
            assert all([irr.mul == self.mul for irr in l_out_irr]) # assert all have same multiplicity
            out_features = self.mul * len(l_out_irr)
            
            l_weight = torch.nn.Parameter(torch.randn(out_features, in_features))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(l_weight, -bound, bound)
            params[f'l{l}_weight'] = l_weight

            l_dims_in.append([2 * l + 1] * len(l_in_irr))
            l_dims_out.append([2 * l + 1] * len(l_out_irr))

            if l == 0 and bias:
                self.bias = torch.nn.Parameter(torch.zeros(self.mul, len(l_out_irr)))
        
        self.params     = torch.nn.ParameterDict(params)
        self.l_dims_in  = l_dims_in
        self.l_dims_out = l_dims_out


    def forward(self, x: torch.Tensor):
        '''
        Forward pass for the SO3_Linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, mul, feature_dim).

        Returns:
            torch.Tensor: Output tensor after linear transformation and optional bias addition.
        '''

        out = []
        start = 0
        for _weight, _l_dims_in, _l_dims_out in zip(self.params.values(), self.l_dims_in, self.l_dims_out):
            length = sum(_l_dims_in)
            feature = x.narrow(dim=-1, start=start, length=length)
            feature = rearrange(feature, 'b m (i l) -> b l (m i)',     m=self.mul, l=_l_dims_in[0],  i=len(_l_dims_in))

            if self.normalization is not None:
                if self.normalization == 'norm':
                    feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
                elif self.normalization == 'component':
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]

                feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
                feature_norm = (feature_norm + self.eps).pow(-0.5)
                feature = feature * feature_norm

            feature = torch.einsum('bli, oi -> blo', feature, _weight)
            feature = rearrange(feature, 'b l (m o) -> b m (l o)', m=self.mul, l=_l_dims_out[0], o=len(_l_dims_out))

            out.append(feature)
            start += length
        
        out = torch.cat(out, dim=-1)
        if self.bias is not None:
            out[..., :sum(self.l_dims_out[0])] += self.bias.unsqueeze(0)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}("     + \
               f"in_irreps={self.in_irreps}, "   + \
               f"out_irreps={self.out_irreps}, " + \
               f"bias={self.bias is not None}, " + \
               f"norm={str(self.normalization)})"