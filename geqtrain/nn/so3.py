import math
from typing import Optional, NamedTuple, List
import torch

from e3nn import o3
from e3nn.util.jit import compile_mode
from einops.layers.torch import Rearrange


class _LinearInstruction(NamedTuple):
    in_start: int
    in_stop: int
    out_start: int
    out_stop: int
    path_idx: int
    mul_in: int
    mul_out: int
    dim: int
    weight_start: int
    weight_stop: int


@compile_mode("script")
class SO3_Linear(torch.nn.Module):
    __constants__ = ["in_mul", "out_dim", "internal_weights", "bias_start", "bias_stop"]
    instructions: List[_LinearInstruction]
    def __init__(
        self,
        in_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        bias: bool = True,
        internal_weights: bool = True,
        **kwargs,
    ):
        '''
        Initializes a fully-connected SO(3)-equivariant linear layer that
        correctly handles varied multiplicities.

        Args:
            in_irreps (o3.Irreps): Input irreducible representations.
            out_irreps (o3.Irreps): Output irreducible representations.
            bias (bool): Whether to include a bias term for the scalar (l=0) outputs.
            internal_weights (bool): If True, the module will have its own learnable weights.
                                     If False, it will expect weights to be passed in the forward call.
        '''
        super().__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.internal_weights = internal_weights

        self.paths = torch.nn.ModuleList()
        self.instructions = torch.jit.annotate(List[_LinearInstruction], [])
        self.weight_numel = 0
        self.in_mul = self.in_irreps[0].mul if len(self.in_irreps) > 0 else 0
        self.out_dim = self.out_irreps.dim
        self.bias_start = 0
        self.bias_stop = 0

        # Get convenient slicing information for our input and output tensors
        in_slices = {ir: s for ir, s in zip(self.in_irreps, self.in_irreps.slices())}
        out_slices = {ir: s for ir, s in zip(self.out_irreps, self.out_irreps.slices())}

        weight_start_idx = 0
        for ir_out, s_out in out_slices.items():
            for ir_in, s_in in in_slices.items():
                # A path is valid only if the irrep type is the same
                if ir_in.ir == ir_out.ir:
                    path_weight_numel = ir_in.mul * ir_out.mul
                    self.weight_numel += path_weight_numel

                    if self.internal_weights:
                        # Create a standard linear layer that will mix the multiplicities
                        path = torch.nn.Linear(ir_in.mul, ir_out.mul, bias=False)
                        self.init_path_weights(path) # Optional: Kaiming init
                        path_idx = len(self.paths)
                        self.paths.append(path)
                    else:
                        path_idx = -1 # Not used

                    # Store instructions for the forward pass
                    self.instructions.append(
                        _LinearInstruction(
                            in_start=s_in.start,
                            in_stop=s_in.stop,
                            out_start=s_out.start,
                            out_stop=s_out.stop,
                            path_idx=path_idx,
                            mul_in=ir_in.mul,
                            mul_out=ir_out.mul,
                            dim=ir_in.ir.dim,
                            weight_start=weight_start_idx,
                            weight_stop=weight_start_idx + path_weight_numel,
                        )
                    )
                    weight_start_idx += path_weight_numel
        
        self.bias = None
        self.bias_slice = None
        if bias:
            # The bias is a learnable parameter for each scalar output channel
            scalar_out_irreps = o3.Irreps([ir for ir in self.out_irreps if ir.ir.l == 0])
            if scalar_out_irreps.dim > 0:
                self.bias_slice = self.out_irreps.slices()[0]
                self.bias_start = self.bias_slice.start
                self.bias_stop = self.bias_slice.stop
                self.bias = torch.nn.Parameter(torch.zeros(scalar_out_irreps.dim, dtype=torch.float32))

    def init_path_weights(self, path: torch.nn.Linear):
        # A simple but effective initialization
        torch.nn.init.kaiming_uniform_(path.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        '''
        Forward pass for the SO3_Linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_irreps.dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_irreps.dim).
        '''
        if not self.internal_weights and weights is None:
            raise ValueError("`internal_weights` is False, but no `weights` tensor was provided to forward().")
        if not self.internal_weights and weights is not None:
            if weights.shape[-1] != self.weight_numel:
                raise ValueError(f"Provided weights tensor has wrong size. Expected {self.weight_numel}, got {weights.shape[-1]}")

        # Create an output tensor of the correct shape, filled with zeros.
        # We will add the results of each path to this tensor.
        output = torch.zeros(x.shape[0], self.out_dim, device=x.device, dtype=x.dtype)

        # Check if the input is flat (batch, features) or has a channel dim (batch, mul, features)
        is_flat_input = len(x.shape) == 2

        if self.internal_weights:
            for idx, path in enumerate(self.paths):
                ins = self.instructions[idx]
                # 1. Slice the input to get the features for one irrep type
                if is_flat_input:
                    input_chunk = x[:, ins.in_start:ins.in_stop]
                else:
                    # For channel-wise input, the features are distributed across the last dimension.
                    # We need to scale the slice indices by the multiplicity.
                    channel_slice = slice(ins.in_start // self.in_mul, ins.in_stop // self.in_mul)
                    input_chunk = x[..., channel_slice]

                # 2. Reshape to separate multiplicity and irrep dimensions
                # from (batch, mul_in * dim) -> (batch, mul_in, dim)
                if is_flat_input:
                    input_chunk = input_chunk.reshape(x.shape[0], ins.mul_in, ins.dim)
                
                # 3. Transpose for torch.nn.Linear, which expects features in the last dim
                # from (batch, mul_in, dim) -> (batch, dim, mul_in)
                input_chunk = input_chunk.transpose(1, 2)

                # 4. Apply the linear layer to the multiplicity channel
                processed_chunk = path(input_chunk) # output: (batch, dim, mul_out)

                # 5. Transpose back
                # from (batch, dim, mul_out) -> (batch, mul_out, dim)
                processed_chunk = processed_chunk.transpose(1, 2)

                # 6. Reshape back to a flat feature vector
                # from (batch, mul_out, dim) -> (batch, mul_out * dim)
                processed_chunk = processed_chunk.reshape(x.shape[0], -1)

                # 7. Add the result to the correct slice of the output tensor
                output[:, ins.out_start:ins.out_stop] += processed_chunk
        else: # use external weights
            assert weights is not None
            for ins in self.instructions:
                # 1. Slice input and weights
                weight_chunk = weights[..., ins.weight_start:ins.weight_stop]
                if is_flat_input:
                    input_chunk = x[:, ins.in_start:ins.in_stop]
                else:
                    # For channel-wise input, scale the slice indices by the multiplicity.
                    channel_slice = slice(ins.in_start // self.in_mul, ins.in_stop // self.in_mul)
                    input_chunk = x[..., channel_slice]

                # 2. Reshape for einsum
                # input: (batch, mul_in * dim) -> (batch, mul_in, dim)
                if is_flat_input:
                    input_chunk = input_chunk.reshape(x.shape[0], ins.mul_in, ins.dim)
                # weights: (..., mul_in * mul_out) -> (..., mul_out, mul_in)
                weight_chunk = weight_chunk.reshape(weights.shape[:-1] + (ins.mul_out, ins.mul_in))

                # 3. Apply weights with einsum
                processed_chunk = torch.einsum('...oi,...id->...od', weight_chunk, input_chunk)

                # 4. Reshape and add to output
                processed_chunk = processed_chunk.reshape(x.shape[0], -1)
                output[:, ins.out_start:ins.out_stop] += processed_chunk
        
        if self.bias is not None:
            output[:, self.bias_start:self.bias_stop] += self.bias

        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_irreps='{self.in_irreps}',\n"
            f"  out_irreps='{self.out_irreps}',\n"
            f"  bias={self.bias is not None},\n"
            f"  internal_weights={self.internal_weights},\n"
            f"  num_paths={len(self.paths)}\n)"
        )


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
            self.register_buffer('balance_degree_weight', torch.zeros(sum([(2*l+1) for l in set(irreps.ls)]), 1, dtype=torch.float32))
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
                self.bias = torch.nn.Parameter(torch.zeros(self.mul, len(l_irr), dtype=torch.float32))

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
