# ----------------------------------------------------------------------------------
# File Name: kan.py
# 
# Original Authors: akaashdash [https://github.com/akaashdash], Blealtan [https://github.com/Blealtan]
# Source Repository: https://github.com/Blealtan/efficient-kan
# 
# Description:
# This file was originally created by the authors mentioned above. It implements
# the Kolmogorov-Arnold network in an efficient way. We would like to extend our
# sincere thanks to the original authors for their great work in developing this
# implementation.
# 
# License: MIT License
# ----------------------------------------------------------------------------------

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from geqtrain.nn.nonlinearities import select_nonlinearity_module


def _flatten_to_2d(x: torch.Tensor, mlp_input_dimension: int) -> Tuple[torch.Tensor, List[int]]:
    """Reshape (..., mlp_input_dimension) -> (N, mlp_input_dimension). Return flattened and original shape."""
    if x.size(-1) != mlp_input_dimension:
        raise ValueError(f"Expected last dim = {mlp_input_dimension}, got {x.size(-1)}.")
    orig = list(x.size())
    return x.reshape(-1, mlp_input_dimension), orig


class KANLinear(nn.Module):
    """
    KAN layer with:
      - spline path: sum_j sum_m w_{out,j,m} * B_m(x_j)
      - base path: Linear( base_act(x) )
    Optional linear mixing:
      - pre_mix: Linear(mlp_input_dimension -> mlp_input_dimension) before base+spline
      - post_mix: Linear(mlp_output_dimension -> mlp_output_dimension) after base+spline
    """

    def __init__(
        self,
        mlp_input_dimension: int,
        mlp_output_dimension: int,
        mlp_nonlinearity: Optional[str] = None,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-3.0, 3.0),
        grid_eps: float = 0.1,
        scale_noise: float = 0.05,
        use_base: bool = True,
        has_bias: bool = True,
        # Optional coordinate mixing
        pre_mix: bool = False,
        post_mix: bool = False,
        mix_bias: bool = False,
        # Spline scaling factor (per out,in)
        use_spline_scaler: bool = True,
        # Numerical stability
        eps: float = 1e-8,
    ):
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2.")
        if spline_order < 1:
            raise ValueError("spline_order must be >= 1.")
        if not (0.0 <= grid_eps <= 1.0):
            raise ValueError("grid_eps must be in [0, 1].")

        self.mlp_input_dimension = int(mlp_input_dimension)
        self.mlp_output_dimension = int(mlp_output_dimension)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.grid_eps = float(grid_eps)
        self.scale_noise = float(scale_noise)
        self.use_base = bool(use_base)
        self.eps = float(eps)

        self.base_activation = select_nonlinearity_module(mlp_nonlinearity)

        # Optional mixing layers
        self.pre_mix = nn.Linear(mlp_input_dimension, mlp_input_dimension, bias=mix_bias) if pre_mix else None
        self.post_mix = nn.Linear(mlp_output_dimension, mlp_output_dimension, bias=mix_bias) if post_mix else None

        # Base path
        self.base_linear = nn.Linear(mlp_input_dimension, mlp_output_dimension, bias=has_bias) if use_base else None

        # Spline parameters: (out, in, n_coeff)
        n_coeff = self.grid_size + self.spline_order
        self.spline_weight = nn.Parameter(torch.empty(mlp_output_dimension, mlp_input_dimension, n_coeff))
        self.use_spline_scaler = bool(use_spline_scaler)
        if self.use_spline_scaler:
            # scaler per (out,in)
            self.spline_scaler = nn.Parameter(torch.ones(mlp_output_dimension, mlp_input_dimension))
        else:
            self.register_parameter("spline_scaler", None)

        # Knot grid buffer: shape (in, grid_size + 2*spline_order + 1)
        self.register_buffer("grid", self._init_grid(), persistent=True)

        self.reset_parameters()

    def _init_grid(self) -> torch.Tensor:
        """Create uniform knots for each input channel, with boundary extension."""
        a, b = self.grid_range
        g = self.grid_size
        k = self.spline_order
        core = torch.linspace(a, b, steps=g + 1)  # (g+1,)
        step = (b - a) / g
        left = core[0] - step * torch.arange(k, 0, -1, dtype=core.dtype)    # (k,)
        right = core[-1] + step * torch.arange(1, k + 1, dtype=core.dtype)  # (k,)
        knots = torch.cat([left, core, right], dim=0)  # (g+2k+1,)
        return knots.unsqueeze(0).repeat(self.mlp_input_dimension, 1)  # (in, g+2k+1)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.use_spline_scaler:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def reset_parameters(self) -> None:
        # Base path init (stable)
        if self.base_linear is not None:
            nn.init.xavier_uniform_(self.base_linear.weight)
            if self.base_linear.bias is not None:
                nn.init.zeros_(self.base_linear.bias)

        # Mixing init
        if self.pre_mix is not None:
            nn.init.orthogonal_(self.pre_mix.weight)
            if self.pre_mix.bias is not None:
                nn.init.zeros_(self.pre_mix.bias)
        if self.post_mix is not None:
            nn.init.orthogonal_(self.post_mix.weight)
            if self.post_mix.bias is not None:
                nn.init.zeros_(self.post_mix.bias)

        # Spline init: near-zero function initially
        # Initialize spline coefficients by fitting small random values on interior knots.
        with torch.no_grad():
            # random "curve values" at (g+1) interior knot points
            # shape (g+1, in, out)
            noise = (torch.rand(self.grid_size + 1, self.mlp_input_dimension, self.mlp_output_dimension) - 0.5)
            noise = noise * (self.scale_noise / self.grid_size)

            # x points: interior knots only, shape (g+1, in)
            x_pts = self.grid[:, self.spline_order : -(self.spline_order)].T  # (g+1, in)
            # Fit scaled coefficients and then store as unscaled if using scaler.
            coeff_scaled = self.curve2coeff(x_pts, noise)  # (out, in, n_coeff)

            if self.use_spline_scaler:
                # keep scaler = 1 at init, so unscaled == scaled
                self.spline_scaler.data.fill_(1.0)
            self.spline_weight.data.copy_(coeff_scaled)

    def b_splines(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline bases for each input coordinate.
        Input:  (N, in)
        Output: (N, in, grid_size + spline_order)
        """
        if x2d.dim() != 2 or x2d.size(1) != self.mlp_input_dimension:
            raise ValueError("b_splines expects a 2D input with the configured last dimension.")

        x = x2d.unsqueeze(-1)  # (N, in, 1)
        grid = self.grid  # (in, M), M = grid_size + 2*k + 1
        k = self.spline_order

        # Degree 0 basis: indicator per interval
        # bases: (N, in, M-1)
        left = grid[:, :-1].unsqueeze(0)
        right = grid[:, 1:].unsqueeze(0)
        bases = ((x >= left) & (x < right)).to(x.dtype)

        # Cox-de Boor recursion
        for d in range(1, k + 1):
            # After each recursion, last dimension shrinks by 1.
            # denom shapes: (in, M-1-d)
            denom1 = (grid[:, d:-1] - grid[:, :-d-1]).clamp_min(self.eps).unsqueeze(0)
            denom2 = (grid[:, d+1:] - grid[:, 1:-d]).clamp_min(self.eps).unsqueeze(0)

            term1 = (x - grid[:, :-d-1].unsqueeze(0)) / denom1 * bases[:, :, :-1]
            term2 = (grid[:, d+1:].unsqueeze(0) - x) / denom2 * bases[:, :, 1:]
            bases = term1 + term2

        # Final count = M-1-k = grid_size + k
        expected = self.grid_size + self.spline_order
        if bases.size(-1) != expected:
            raise RuntimeError(f"Unexpected basis count: {bases.size(-1)} vs {expected}.")
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit spline coefficients (least squares) per input channel.

        x: (N, in)
        y: (N, in, out)  -- "unreduced" per-input contribution you want to preserve.
        Returns coeff: (out, in, n_coeff)
        """
        if x.dim() != 2 or x.size(1) != self.mlp_input_dimension:
            raise ValueError("x must be 2D with the configured input dimension.")
        if y.dim() != 3 or y.size(1) != self.mlp_input_dimension or y.size(2) != self.mlp_output_dimension:
            raise ValueError("y must be 3D with configured input/output dimensions.")

        # A: (in, N, n_coeff), B: (in, N, out)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        # lstsq per input channel
        sol = torch.linalg.lstsq(A, B).solution  # (in, n_coeff, out)
        coeff = sol.permute(2, 0, 1).contiguous()  # (out, in, n_coeff)
        return coeff

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01, max_samples: int = 16384) -> None:
        """
        Update knots using a blend of adaptive (quantile) and uniform grids.
        Preserves the current function by refitting coefficients onto the new grid.
        """
        x2d, _ = _flatten_to_2d(x, self.mlp_input_dimension)

        # Optional subsample for speed / stability
        if x2d.size(0) > max_samples:
            idx = torch.randperm(x2d.size(0), device=x2d.device)[:max_samples]
            x2d = x2d[idx]

        N = x2d.size(0)
        if N < self.grid_size + 1:
            # not enough samples to estimate quantiles: keep uniform grid
            return

        # Compute unreduced per-input contribution under CURRENT grid
        bases = self.b_splines(x2d)                 # (N, in, n_coeff)
        bases_t = bases.permute(1, 0, 2)            # (in, N, n_coeff)
        coeff_scaled = self.scaled_spline_weight    # (out, in, n_coeff)
        coeff_scaled_t = coeff_scaled.permute(1, 2, 0)  # (in, n_coeff, out)
        y_unreduced = torch.bmm(bases_t, coeff_scaled_t).permute(1, 0, 2)  # (N, in, out)

        # Build new grid per channel
        x_sorted = torch.sort(x2d, dim=0)[0]  # (N, in)
        # Quantile indices for G+1 points
        q_idx = torch.linspace(0, N - 1, self.grid_size + 1, device=x2d.device).long()
        grid_adapt = x_sorted[q_idx]  # (G+1, in)

        # Uniform grid spanning [min,max] with margin
        xmin = x_sorted[0] - margin
        xmax = x_sorted[-1] + margin
        step = (xmax - xmin).clamp_min(self.eps) / self.grid_size  # (in,)
        grid_uni = (torch.arange(self.grid_size + 1, device=x2d.device, dtype=x2d.dtype).unsqueeze(1)
                    * step.unsqueeze(0) + xmin.unsqueeze(0))  # (G+1, in)

        # Blend
        grid_core = self.grid_eps * grid_uni + (1.0 - self.grid_eps) * grid_adapt  # (G+1, in)
        grid_core = grid_core.transpose(0, 1).contiguous()  # (in, G+1)

        # Extend boundaries with constant step per channel
        k = self.spline_order
        left = grid_core[:, :1] - step.unsqueeze(1) * torch.arange(k, 0, -1, device=x2d.device, dtype=x2d.dtype).unsqueeze(0)
        right = grid_core[:, -1:] + step.unsqueeze(1) * torch.arange(1, k + 1, device=x2d.device, dtype=x2d.dtype).unsqueeze(0)
        new_grid = torch.cat([left, grid_core, right], dim=1)  # (in, G+2k+1)

        # Apply new grid and refit coefficients to preserve scaled function
        self.grid.copy_(new_grid)

        coeff_scaled_new = self.curve2coeff(x2d, y_unreduced)  # (out,in,n_coeff)
        if self.use_spline_scaler:
            denom = self.spline_scaler.unsqueeze(-1).clamp_min(self.eps)
            self.spline_weight.copy_(coeff_scaled_new / denom)
        else:
            self.spline_weight.copy_(coeff_scaled_new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2d, orig_shape = _flatten_to_2d(x, self.mlp_input_dimension)

        # Optional pre-mix
        if self.pre_mix is not None:
            x2d = self.pre_mix(x2d)

        # Base path
        out = x2d.new_zeros((x2d.shape[0], self.mlp_output_dimension))
        if self.base_linear is not None:
            if self.base_activation is not None:
                x2d = self.base_activation(x2d)
            out = out + self.base_linear(x2d)

        # Spline path
        bases = self.b_splines(x2d)  # (N,in,n_coeff)
        N = bases.size(0)
        spline_in = bases.reshape(N, -1)  # (N, in*n_coeff)
        w = self.scaled_spline_weight.reshape(self.mlp_output_dimension, -1)  # (out, in*n_coeff)
        out = out + F.linear(spline_in, w, bias=None)

        # Optional post-mix
        if self.post_mix is not None:
            out = self.post_mix(out)

        return out.reshape(orig_shape[:-1] + [self.mlp_output_dimension])

    def regularization_loss(self, reg_l1: float = 1.0, reg_entropy: float = 0.0) -> torch.Tensor:
        """
        Lightweight regularizer:
          - L1 on spline_weight magnitude (proxy for activation sparsity)
          - optional entropy term to discourage a single channel dominating
        """
        l1 = self.spline_weight.abs().mean(dim=-1)  # (out,in)
        l1_sum = l1.sum()
        loss = reg_l1 * l1_sum

        if reg_entropy != 0.0:
            p = l1 / (l1_sum + self.eps)
            ent = -(p * (p + self.eps).log()).sum()
            loss = loss + reg_entropy * ent
        return loss


class KAN(nn.Module):
    """
    Simple MLP-like stack of KANLinear blocks, with optional LayerNorm between blocks.
    """

    def __init__(
        self,
        mlp_input_dimension: int,
        mlp_latent_dimensions: Sequence[int],
        mlp_output_dimension: int,
        mlp_nonlinearity: Optional[str] = 'silu',
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-3.0, 3.0),
        grid_eps: float = 0.1,
        scale_noise: float = 0.05,
        use_base: bool = True,
        use_layer_norm: bool = True,
        has_bias: bool = True,
        # mixing policy
        pre_mix: Union[bool, Sequence[bool]] = False,
        post_mix: Union[bool, Sequence[bool]] = False,
        mix_bias: bool = False,
        use_spline_scaler: bool = True,
    ):
        super().__init__()
        dims = [mlp_input_dimension] + mlp_latent_dimensions + [mlp_output_dimension]
        if len(dims) < 2:
            raise ValueError("dims must have at least [in, out].")

        L = len(dims) - 1
        if isinstance(pre_mix, bool):
            pre_mix = [pre_mix] * L
        if isinstance(post_mix, bool):
            post_mix = [post_mix] * L
        if len(pre_mix) != L or len(post_mix) != L:
            raise ValueError("pre_mix/post_mix must be bool or have length len(dims)-1.")

        layers: List[nn.Module] = []
        for i in range(L):
            layers.append(
                KANLinear(
                    mlp_input_dimension=dims[i],
                    mlp_output_dimension=dims[i + 1],
                    mlp_nonlinearity=mlp_nonlinearity,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                    grid_eps=grid_eps,
                    scale_noise=scale_noise,
                    use_base=use_base,
                    has_bias=has_bias,
                    pre_mix=pre_mix[i],
                    post_mix=post_mix[i],
                    mix_bias=mix_bias,
                    use_spline_scaler=use_spline_scaler,
                )
            )
            if use_layer_norm and i < L - 1:
                layers.append(nn.LayerNorm(dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Update grids for all KANLinear layers, threading activations forward."""
        h = x
        for layer in self.layers:
            if isinstance(layer, KANLinear):
                layer.update_grid(h, margin=margin)
                h = layer(h)
            else:
                h = layer(h)

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        if update_grid and not torch.jit.is_scripting():
            self.update_grid(x)
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def regularization_loss(self, reg_l1: float = 1.0, reg_entropy: float = 0.0) -> torch.Tensor:
        loss = 0.0
        for layer in self.layers:
            if isinstance(layer, KANLinear):
                loss = loss + layer.regularization_loss(reg_l1=reg_l1, reg_entropy=reg_entropy)
        return loss
