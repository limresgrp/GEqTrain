import math
from typing import Optional, NamedTuple, List, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode


class _LinearInstruction(NamedTuple):
    # Flat (Irreps) slice indices
    in_start: int
    in_stop: int
    out_start: int
    out_stop: int

    # Mixing dimensions for this irrep block
    mul_in: int
    mul_out: int
    dim: int  # = (2*l + 1)

    # Weight packing indices into the flattened `weights` tensor (external weights)
    weight_start: int
    weight_stop: int

    # Index into self.paths if using internal weights; -1 otherwise
    path_idx: int


class _BiasInstruction(NamedTuple):
    out_start: int
    out_stop: int
    bias_start: int
    bias_stop: int


class _NormGroup(NamedTuple):
    # Indices in "per-multiplicity feature space":
    # x_ch has shape (B, mul, feat_per_mul) and we slice last axis with [start:stop]
    start: int
    stop: int
    l_dim: int
    n_blocks: int
    is_scalar: bool  # l == 0


def _has_common_mul(irreps: o3.Irreps) -> Tuple[bool, int]:
    if len(irreps) == 0:
        return True, 0
    m0 = irreps[0].mul
    ok = all(mul == m0 for (mul, _) in irreps)
    return ok, int(m0)


@compile_mode("script")
class SO3_Linear(torch.nn.Module):
    """
    Fully-connected SO(3)/O(3)-equivariant linear layer that mixes *only* multiplicities
    within matching irrep types (same l and parity).

    Supported input formats:
      - flat:      (B, in_irreps.dim)
      - channel:   (B, mul, in_irreps.dim // mul)  only if in_irreps has a common multiplicity

    Output format:
      - If input is flat:      returns flat (B, out_irreps.dim)
      - If input is channel:  returns channel (B, mul_out, out_irreps.dim // mul_out) if out_irreps has common mul,
                              otherwise returns flat (B, out_irreps.dim)

    Notes:
      - `internal_weights=False` expects a `weights` tensor whose last dimension equals `self.weight_numel`.
        The weights are packed in the same order as `self.instructions`.
    """
    __constants__ = [
        "out_dim",
        "internal_weights",
        "in_has_common_mul",
        "in_mul_common",
        "out_has_common_mul",
        "out_mul_common",
    ]

    instructions: List[_LinearInstruction]
    bias_instructions: List[_BiasInstruction]

    def __init__(
        self,
        in_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
        bias: bool = True,
        internal_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)
        self.internal_weights = bool(internal_weights)

        self.out_dim = int(self.out_irreps.dim)

        # Channel format support flags
        self.in_has_common_mul, self.in_mul_common = _has_common_mul(self.in_irreps)
        self.out_has_common_mul, self.out_mul_common = _has_common_mul(self.out_irreps)

        # Build instructions in *list order* (never via dict)
        in_slices = list(self.in_irreps.slices())
        out_slices = list(self.out_irreps.slices())

        self.paths = torch.nn.ModuleList()
        self.instructions = torch.jit.annotate(List[_LinearInstruction], [])
        self.weight_numel = 0

        path_idx = 0
        w_cursor = 0
        for (mul_out, ir_out), s_out in zip(self.out_irreps, out_slices):
            for (mul_in, ir_in), s_in in zip(self.in_irreps, in_slices):
                if ir_in == ir_out:
                    dim = int(ir_in.dim)  # = 2*l+1

                    # Sanity: slice sizes match
                    if (s_in.stop - s_in.start) != int(mul_in * dim):
                        raise ValueError(
                            "SO3_Linear: input slice size mismatch vs irreps; check irreps/tensor layout."
                        )
                    if (s_out.stop - s_out.start) != int(mul_out * dim):
                        raise ValueError(
                            "SO3_Linear: output slice size mismatch vs irreps; check irreps/tensor layout."
                        )

                    w_len = int(mul_in * mul_out)
                    w_start = w_cursor
                    w_stop = w_cursor + w_len
                    w_cursor = w_stop
                    self.weight_numel += w_len

                    if self.internal_weights:
                        # Mix multiplicities only
                        lin = torch.nn.Linear(int(mul_in), int(mul_out), bias=False)
                        torch.nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
                        self.paths.append(lin)
                        use_path_idx = path_idx
                        path_idx += 1
                    else:
                        use_path_idx = -1

                    self.instructions.append(
                        _LinearInstruction(
                            in_start=int(s_in.start),
                            in_stop=int(s_in.stop),
                            out_start=int(s_out.start),
                            out_stop=int(s_out.stop),
                            mul_in=int(mul_in),
                            mul_out=int(mul_out),
                            dim=dim,
                            weight_start=w_start,
                            weight_stop=w_stop,
                            path_idx=use_path_idx,
                        )
                    )

        if w_cursor != self.weight_numel:
            raise RuntimeError("SO3_Linear: internal error building weight packing.")

        # Bias: only for scalar outputs (l=0). We add bias on those flat slices.
        self.bias = None
        self.bias_instructions = torch.jit.annotate(List[_BiasInstruction], [])
        if bias:
            b_cursor = 0
            for (mul_out, ir_out), s_out in zip(self.out_irreps, out_slices):
                if ir_out.l == 0:
                    seg_len = int(s_out.stop - s_out.start)  # = mul_out * 1
                    self.bias_instructions.append(
                        _BiasInstruction(
                            out_start=int(s_out.start),
                            out_stop=int(s_out.stop),
                            bias_start=b_cursor,
                            bias_stop=b_cursor + seg_len,
                        )
                    )
                    b_cursor += seg_len
            if b_cursor > 0:
                self.bias = torch.nn.Parameter(torch.zeros(b_cursor, dtype=torch.float32))

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (not self.internal_weights) and (weights is None):
            raise ValueError("SO3_Linear: internal_weights=False but no `weights` were provided.")

        # Input format detection
        x_is_flat = (x.ndim == 2)
        if not x_is_flat:
            if x.ndim != 3:
                raise ValueError(f"SO3_Linear expects x.ndim in {{2,3}}, got {x.ndim}.")
            if not self.in_has_common_mul:
                raise ValueError(
                    "SO3_Linear received channel-format input, but in_irreps does not have a common multiplicity."
                )
            if x.shape[1] != self.in_mul_common:
                raise ValueError(f"SO3_Linear channel input has mul={x.shape[1]} but expected {self.in_mul_common}.")

        B = int(x.shape[0])
        out = x.new_zeros((B, self.out_dim))

        if self.internal_weights:
            # One path per instruction
            for ins in self.instructions:
                # Slice input
                if x_is_flat:
                    xin = x[:, ins.in_start:ins.in_stop].reshape(B, ins.mul_in, ins.dim)  # (B, mul_in, dim)
                else:
                    # Convert flat indices to per-mul feature indices
                    start_per = ins.in_start // self.in_mul_common
                    stop_per = ins.in_stop // self.in_mul_common
                    xin = x[:, :, start_per:stop_per]  # (B, mul_in, dim)

                # Apply multiplicity mixing
                path = self.paths[ins.path_idx]
                y = path(xin.transpose(1, 2)).transpose(1, 2)  # (B, mul_out, dim)
                out[:, ins.out_start:ins.out_stop] += y.reshape(B, -1)

        else:
            # External packed weights
            w = weights
            if w is None:
                raise RuntimeError("SO3_Linear: unreachable weights is None in external branch.")

            if w.shape[-1] != self.weight_numel:
                raise ValueError(f"SO3_Linear: expected weights[..., {self.weight_numel}], got {w.shape}.")

            for ins in self.instructions:
                # Slice input
                if x_is_flat:
                    xin = x[:, ins.in_start:ins.in_stop].reshape(B, ins.mul_in, ins.dim)  # (B, mul_in, dim)
                else:
                    start_per = ins.in_start // self.in_mul_common
                    stop_per = ins.in_stop // self.in_mul_common
                    xin = x[:, :, start_per:stop_per]  # (B, mul_in, dim)

                # Slice and reshape weights to (..., mul_out, mul_in)
                w_chunk = w[..., ins.weight_start:ins.weight_stop].reshape(w.shape[:-1] + (ins.mul_out, ins.mul_in))

                # Multiply: (B, dim, mul_in) @ (..., mul_in, mul_out) -> (B, dim, mul_out)
                xin_t = xin.transpose(1, 2)  # (B, dim, mul_in)
                y_t = torch.matmul(xin_t, w_chunk.transpose(-1, -2))  # broadcast over leading dims
                y = y_t.transpose(1, 2)  # (B, mul_out, dim)

                out[:, ins.out_start:ins.out_stop] += y.reshape(B, -1)

        # Add scalar bias (flat)
        if self.bias is not None:
            for bi in self.bias_instructions:
                out[:, bi.out_start:bi.out_stop] += self.bias[bi.bias_start:bi.bias_stop]

        # Return in the most intuitive format: match input if possible.
        if (not x_is_flat) and self.out_has_common_mul and self.out_mul_common > 0:
            if self.out_dim % self.out_mul_common != 0:
                return out
            return out.reshape(B, self.out_mul_common, self.out_dim // self.out_mul_common)

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  in_irreps='{self.in_irreps}',\n"
            f"  out_irreps='{self.out_irreps}',\n"
            f"  bias={self.bias is not None},\n"
            f"  internal_weights={self.internal_weights},\n"
            f"  weight_numel={self.weight_numel},\n"
            f"  num_paths={len(self.paths)}\n)"
        )


@compile_mode("script")
class SO3_LayerNorm(torch.nn.Module):
    """
    SO(3)/O(3)-aware layer normalization for an Irreps tensor with a **common multiplicity**.

    Supported input formats:
      - flat:    (B, irreps.dim)
      - channel: (B, mul, irreps.dim // mul)

    Output format matches input format (flat-in -> flat-out, channel-in -> channel-out).

    Normalization options:
      - 'norm':      uses sum over m of squared components (||·||^2) per l
      - 'component': uses mean over m of squared components (component-wise)
      - 'std':       like 'component' but additionally scales by 1/sqrt(#unique l) to stabilize depth

    Bias:
      - If bias=True, a learnable bias is added only to scalar (l=0) channels, per multiplicity and per scalar block.
    """
    __constants__ = ["mul", "feat_per_mul", "normalization", "n_unique_l"]

    groups: List[_NormGroup]
    scalar_bias_instructions: List[_BiasInstruction]

    def __init__(
        self,
        irreps: o3.Irreps,
        bias: bool = True,
        normalization: str = "std",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        ok, mul = _has_common_mul(self.irreps)
        if not ok:
            raise ValueError(
                "SO3_LayerNorm requires irreps with a common multiplicity (e.g. '16x0e+16x1o+...')."
            )
        self.mul = int(mul)
        self.eps = float(eps)

        if normalization not in ("norm", "component", "std"):
            raise ValueError("normalization must be one of {'norm','component','std'}")
        self.normalization = normalization

        if self.mul == 0:
            self.feat_per_mul = 0
            self.groups = torch.jit.annotate(List[_NormGroup], [])
            self.scalar_bias_instructions = torch.jit.annotate(List[_BiasInstruction], [])
            self.bias = None
            self.n_unique_l = 0
            return

        self.feat_per_mul = int(self.irreps.dim // self.mul)
        if self.irreps.dim % self.mul != 0:
            raise ValueError("Irreps dim is not divisible by multiplicity; cannot use channel format.")

        # Build contiguous groups in the *given irreps order*.
        slices = list(self.irreps.slices())
        self.groups = torch.jit.annotate(List[_NormGroup], [])

        # Determine unique l values (for std stabilization)
        unique_ls = set([ir.ir.l for ir in self.irreps])
        self.n_unique_l = int(len(unique_ls))

        cur_l = -999
        cur_start = 0
        cur_stop = 0
        cur_l_dim = 0
        cur_n_blocks = 0

        for (mul_i, ir_i), sl in zip(self.irreps, slices):
            if int(mul_i) != self.mul:
                raise ValueError("SO3_LayerNorm: expected common multiplicity across irreps entries.")
            l = int(ir_i.l)
            l_dim = int(ir_i.dim)
            start_per = int(sl.start // self.mul)
            stop_per = int(sl.stop // self.mul)
            if (stop_per - start_per) != l_dim:
                raise ValueError("SO3_LayerNorm: unexpected slice length; check irreps vs tensor layout.")

            if cur_n_blocks == 0:
                cur_l = l
                cur_start = start_per
                cur_stop = stop_per
                cur_l_dim = l_dim
                cur_n_blocks = 1
            else:
                # merge only if consecutive in layout and same l
                if (l == cur_l) and (l_dim == cur_l_dim) and (start_per == cur_stop):
                    cur_stop = stop_per
                    cur_n_blocks += 1
                else:
                    self.groups.append(
                        _NormGroup(
                            start=cur_start,
                            stop=cur_stop,
                            l_dim=cur_l_dim,
                            n_blocks=cur_n_blocks,
                            is_scalar=(cur_l == 0),
                        )
                    )
                    cur_l = l
                    cur_start = start_per
                    cur_stop = stop_per
                    cur_l_dim = l_dim
                    cur_n_blocks = 1

        if cur_n_blocks > 0:
            self.groups.append(
                _NormGroup(
                    start=cur_start,
                    stop=cur_stop,
                    l_dim=cur_l_dim,
                    n_blocks=cur_n_blocks,
                    is_scalar=(cur_l == 0),
                )
            )

        # Bias for scalar blocks: pack scalar per-mul features in the order encountered
        self.bias = None
        self.scalar_bias_instructions = torch.jit.annotate(List[_BiasInstruction], [])
        if bias:
            b_cursor = 0
            for g in self.groups:
                if g.is_scalar:
                    seg_len = int(g.stop - g.start)  # scalar: l_dim=1 => length == n_blocks
                    self.scalar_bias_instructions.append(
                        _BiasInstruction(
                            out_start=g.start,
                            out_stop=g.stop,
                            bias_start=b_cursor,
                            bias_stop=b_cursor + seg_len,
                        )
                    )
                    b_cursor += seg_len
            if b_cursor > 0:
                # bias is stored in channel format: (mul, total_scalar_per_mul)
                self.bias = torch.nn.Parameter(torch.zeros(self.mul, b_cursor, dtype=torch.float32))

        # For 'std', stabilize by 1/sqrt(#unique l). For others, no extra scaling.
        if self.normalization == "std" and self.n_unique_l > 0:
            self.l_scale = 1.0 / math.sqrt(float(self.n_unique_l))
        else:
            self.l_scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mul == 0:
            return x

        x_is_flat = (x.ndim == 2)
        if x_is_flat:
            B, D = x.shape
            if int(D) != int(self.irreps.dim):
                raise ValueError(f"SO3_LayerNorm: expected last dim {self.irreps.dim}, got {D}.")
            x_ch = x.reshape(B, self.mul, self.feat_per_mul)
        else:
            if x.ndim != 3:
                raise ValueError(f"SO3_LayerNorm expects x.ndim in {{2,3}}, got {x.ndim}.")
            B = int(x.shape[0])
            if int(x.shape[1]) != self.mul:
                raise ValueError(f"SO3_LayerNorm: expected mul={self.mul}, got {x.shape[1]}.")
            if int(x.shape[2]) != self.feat_per_mul:
                raise ValueError(f"SO3_LayerNorm: expected feat_per_mul={self.feat_per_mul}, got {x.shape[2]}.")
            x_ch = x

        out = torch.empty_like(x_ch)

        # Normalize each contiguous l-group independently (one scale per batch per group)
        for g in self.groups:
            # slice group: (B, mul, n_blocks*l_dim)
            seg = x_ch[:, :, g.start:g.stop]

            # reshape -> (B, l_dim, C) with C = mul * n_blocks
            seg4 = seg.reshape(B, self.mul, g.n_blocks, g.l_dim)      # (B, mul, n_blocks, l_dim)
            seg4 = seg4.permute(0, 3, 1, 2)                           # (B, l_dim, mul, n_blocks)
            feat = seg4.reshape(B, g.l_dim, self.mul * g.n_blocks)    # (B, l_dim, C)

            if self.normalization == "norm":
                stat = feat.pow(2).sum(dim=1, keepdim=True)           # (B, 1, C)
            else:
                stat = feat.pow(2).mean(dim=1, keepdim=True)          # (B, 1, C)

            # one scalar per batch per group
            stat = stat.mean(dim=2, keepdim=True)                     # (B, 1, 1)
            inv = (stat + self.eps).rsqrt() * self.l_scale
            feat = feat * inv

            # back: (B, l_dim, C) -> (B, mul, n_blocks*l_dim)
            seg4 = feat.reshape(B, g.l_dim, self.mul, g.n_blocks).permute(0, 2, 3, 1)
            out[:, :, g.start:g.stop] = seg4.reshape(B, self.mul, g.n_blocks * g.l_dim)

        # Add scalar bias (channel format) on scalar groups only
        if self.bias is not None:
            for bi in self.scalar_bias_instructions:
                out[:, :, bi.out_start:bi.out_stop] += self.bias[:, bi.bias_start:bi.bias_stop].unsqueeze(0)

        if x_is_flat:
            return out.reshape(B, int(self.irreps.dim))
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(irreps={self.irreps}, norm={self.normalization}, bias={self.bias is not None})"
