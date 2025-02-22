import math
import torch

from e3nn import o3
from e3nn.util.jit import compile_mode
from einops.layers.torch import Rearrange


@compile_mode("script")
class SO3_Linear(torch.nn.Module):
    def __init__(
            self,
            in_irreps: o3.Irreps,
            out_irreps: o3.Irreps,
            bias: bool = True,
        ):
        '''
        FOR DOCUMENTATION purposes: UNDERSTAND EQUIVARIANT LINEAR LAYER

        Initializes the SO3_Linear layer.

        Args:
            in_irreps (o3.Irreps): Input irreducible representations.
            out_irreps (o3.Irreps): Output irreducible representations.
            bias (bool): Whether to include a bias term. Default is True.

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
            )

            x = in_irreps.randn(10, -1).reshape(10, 8, -1)
            output = so3_linear(x)
        '''

        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.mul_in = in_irreps[0].mul
        self.mul_out = out_irreps[0].mul
        self.bias = None

        scalars = 0
        params = {}
        lengths = []
        rearrange_in_list  = []
        rearrange_out_list = []

        for l in set(out_irreps.ls):
            l_in_irr = [irr for irr in in_irreps if irr.ir.l == l]
            assert all([irr.mul == self.mul_in for irr in l_in_irr])  # assert all have same multiplicity
            in_features = self.mul_in * len(l_in_irr)
            l_out_irr = [irr for irr in out_irreps if irr.ir.l == l]
            assert all([irr.mul == self.mul_out for irr in l_out_irr]) # assert all have same multiplicity
            out_features = self.mul_out * len(l_out_irr)

            l_weight = torch.nn.Parameter(torch.randn(out_features, in_features))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(l_weight, -bound, bound)
            params[f'l{l}_weight'] = l_weight

            _l_dims_in = [2 * l + 1] * len(l_in_irr)
            _l_dims_out = [2 * l + 1] * len(l_out_irr)
            lengths.append(sum(_l_dims_in))

            rearrange_in_list.append(Rearrange('b m (i l) -> b l (m i)', m=self.mul_in, l=_l_dims_in[0],  i=len(_l_dims_in)))
            rearrange_out_list.append(Rearrange('b l (m o) -> b m (o l)', m=self.mul_out, l=_l_dims_out[0], o=len(_l_dims_out)))

            if l == 0 and bias:
                scalars = len(l_out_irr)
                self.bias = torch.nn.Parameter(torch.zeros(self.mul_out, scalars))

        self.scalars            = scalars
        self.lengths            = lengths
        self.params             = torch.nn.ParameterDict(params)
        self.rearrange_in_list  = torch.nn.ModuleList(rearrange_in_list)
        self.rearrange_out_list = torch.nn.ModuleList(rearrange_out_list)

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
        for idx, (_weight, rearrange_in, rearrange_out) in enumerate(zip(self.params.values(), self.rearrange_in_list, self.rearrange_out_list)):
            _length = self.lengths[idx]
            feature = x.narrow(dim=-1, start=start, length=_length)
            feature = rearrange_in(feature)
            feature = torch.einsum('bli, oi -> blo', feature, _weight)
            feature = rearrange_out(feature)

            out.append(feature)
            start += _length

        out = torch.cat(out, dim=-1)
        if self.bias is not None:
            out[..., :self.scalars] += self.bias.unsqueeze(0)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}("     + \
               f"in_irreps={self.in_irreps}, "   + \
               f"out_irreps={self.out_irreps}, " + \
               f"bias={self.bias is not None})"


@compile_mode("script")
class SO3_LayerNorm(torch.nn.Module):
    def __init__(
            self,
            irreps: o3.Irreps,
            bias: bool = True,
            normalization: str = 'std',
            eps: float = 1e-5,
        ):
        '''
        Initializes the SO3_Linear layer.

        Args:
            irreps (o3.Irreps): Input irreducible representations.
            bias (bool): Whether to include a bias term. Default is True.
            normalization (Optional[str]): Normalization method, either 'norm', 'component' or 'std'. Default is std.
            eps (float): A small value to avoid division by zero in normalization. Default is 1e-5.

        Attributes:
            irreps (o3.Irreps): Stored input irreducible representations.
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
        self.irreps = irreps
        self.mul = irreps[0].mul
        self.bias = None

        assert normalization in ['norm', 'component', 'std']
        self.normalization = normalization
        self.eps = eps

        l_dims = []
        lengths = []
        l_dim0_list = []
        rearrange_in_list  = []
        rearrange_out_list = []

        if self.normalization == 'std':
            self.register_buffer('balance_degree_weight', torch.zeros(sum([(2*l+1) for l in set(irreps.ls)]), 1))
        else:
            self.balance_degree_weight = None

        start = 0
        for l in set(irreps.ls):
            l_irr = [irr for irr in irreps if irr.ir.l == l]
            assert all([irr.mul == self.mul for irr in l_irr])  # assert all have same multiplicity

            _l_dims = [2 * l + 1] * len(l_irr)
            l_dims.append(_l_dims)
            lengths.append(sum(_l_dims))
            _l_dim0 = _l_dims[0]
            l_dim0_list.append(_l_dim0)

            rearrange_in_list.append( Rearrange('b m (i l) -> b l (m i)', m=self.mul, l=_l_dim0, i=len(_l_dims)))
            rearrange_out_list.append(Rearrange('b l (m i) -> b m (i l)', m=self.mul, l=_l_dim0, i=len(_l_dims)))

            if self.normalization == 'std':
                self.balance_degree_weight[start : (start + _l_dim0), :] = (1.0 / (_l_dim0 * len(set(irreps.ls))))

            start += _l_dim0

            if l == 0 and bias:
                self.bias = torch.nn.Parameter(torch.zeros(self.mul, len(l_irr)))

        self.l_dims = l_dims
        self.lengths            = tuple(lengths)
        self.l_dim0_list        = tuple(l_dim0_list)
        self.rearrange_in_list  = torch.nn.ModuleList(rearrange_in_list)
        self.rearrange_out_list = torch.nn.ModuleList(rearrange_out_list)

        self.l_dim_norm = 1.
        if self.normalization == 'std':
            self.l_dim_norm = 1. / math.sqrt(len(self.l_dims))


    def forward(self, x: torch.Tensor):
        '''
        Forward pass for the SO3_Linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, mul, feature_dim).

        Returns:
            torch.Tensor: Output tensor after linear transformation and optional bias addition.
        '''

        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(-1)

        out = []
        start = 0
        l_start = 0
        for idx, (rearrange_in, rearrange_out) in enumerate(zip(self.rearrange_in_list, self.rearrange_out_list)):
            _length, _l_dim0 = self.lengths[idx], self.l_dim0_list[idx]

            feature = x.narrow(dim=-1, start=start, length=_length)
            feature = rearrange_in(feature)

            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            elif self.normalization == 'std':
                feature_norm = feature.pow(2)                               # [N, (2 * l) + 1, C]
                balance_degree_weight = self.balance_degree_weight.narrow(dim=0, start=l_start, length=_l_dim0)
                feature_norm = torch.einsum('blc, la -> bac', feature_norm, balance_degree_weight) # [N, 1, C]
            else:
                raise Exception(f'Invalid normalization: {self.normalization}. Use one among [norm, component, std]')

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5) * self.l_dim_norm
            feature = feature * feature_norm

            feature = rearrange_out(feature)

            out.append(feature)
            start += _length
            l_start += _l_dim0

        out = torch.cat(out, dim=-1)
        if self.bias is not None:
            out[..., :sum(self.l_dims[0])] += self.bias.unsqueeze(0)

        if out.shape != orig_shape:
            if len(orig_shape) == 2:
                a, b = orig_shape
                out = out.reshape(a, b)
            else:
                a, b, c = orig_shape
                out = out.reshape(a, b, c)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}("     + \
               f"irreps={self.irreps}, "         + \
               f"bias={self.bias is not None}, " + \
               f"norm={str(self.normalization)})"