from typing import Optional, Dict, Any

import torch
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin
from geqtrain.utils.savenload import load_callable


def _resolve_block_class(path: str):
    try:
        return load_callable(path)
    except Exception:
        return load_callable(path, prefix="geqtrain.nn")


@compile_mode("script")
class RecyclingModule(GraphModuleMixin, torch.nn.Module):
    """
    Generic fixed-point style wrapper around a GraphModule.

    The wrapped `block` is run repeatedly. A recycled state tensor is updated with:
      state <- state + alpha * (proposal - state)
    where `proposal` is read from `block_out_field` after each block call.

    This module is Hydra-friendly: users can either pass an instantiated `block`
    or provide `block_target` + `block_kwargs`.
    """

    __constants__ = ["state_field", "block_out_field", "out_field", "max_steps", "detach_between_steps"]

    def __init__(
        self,
        block: Optional[GraphModuleMixin] = None,
        block_target: Optional[str] = None,
        block_kwargs: Optional[Dict[str, Any]] = None,
        state_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        block_out_field: Optional[str] = None,
        out_field: Optional[str] = None,
        max_steps: int = 8,
        tol: float = 0.0,
        alpha: float = 0.7,
        adaptive_damping: bool = False,
        alpha_min: float = 0.05,
        residual_growth_tol: float = 1.0,
        state_clip_value: float = 0.0,
        detach_between_steps: bool = False,
        irreps_in=None,
    ):
        super().__init__()
        if max_steps <= 0:
            raise ValueError("`max_steps` must be > 0.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("`alpha` must be in (0, 1].")
        if not (0.0 < alpha_min <= alpha):
            raise ValueError("`alpha_min` must be in (0, alpha].")
        if residual_growth_tol <= 0.0:
            raise ValueError("`residual_growth_tol` must be > 0.")
        if state_clip_value < 0.0:
            raise ValueError("`state_clip_value` must be >= 0.")
        if tol < 0.0:
            raise ValueError("`tol` must be >= 0.")

        if block is None:
            if block_target is None:
                raise ValueError("Either `block` or `block_target` must be provided.")
            kwargs = dict(block_kwargs or {})
            if "irreps_in" not in kwargs:
                kwargs["irreps_in"] = irreps_in
            block_cls = _resolve_block_class(block_target)
            block = block_cls(**kwargs)

        if not isinstance(block, GraphModuleMixin):
            raise TypeError(f"`block` must be a GraphModuleMixin, got: {type(block).__name__}")

        self.block = block
        self.state_field = state_field
        self.block_out_field = block_out_field if block_out_field is not None else state_field
        self.out_field = out_field if out_field is not None else self.block_out_field

        if self.block_out_field not in self.block.irreps_out:
            raise ValueError(
                f"`block_out_field`='{self.block_out_field}' not found in wrapped block outputs: "
                f"{list(self.block.irreps_out.keys())}"
            )

        self.max_steps = int(max_steps)
        self.tol = float(tol)
        self.alpha = float(alpha)
        self.adaptive_damping = bool(adaptive_damping)
        self.alpha_min = float(alpha_min)
        self.residual_growth_tol = float(residual_growth_tol)
        self.state_clip_value = float(state_clip_value)
        self.detach_between_steps = bool(detach_between_steps)

        initial_irreps_in = irreps_in if irreps_in is not None else self.block.irreps_in
        irreps_out = dict(self.block.irreps_out)
        irreps_out[self.out_field] = self.block.irreps_out[self.block_out_field]
        self._init_irreps(irreps_in=initial_irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        state = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.state_field in data:
            state = data[self.state_field]
        prev_residual = torch.as_tensor(-1.0, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype, device=data[AtomicDataDict.POSITIONS_KEY].device)

        for it in range(self.max_steps):
            if state is not None:
                data[self.state_field] = state

            data = self.block(data)
            proposal = data[self.block_out_field]

            if state is None:
                state = proposal
                residual = torch.sqrt(torch.mean(state.square()))
            else:
                step = proposal - state
                raw_residual = torch.sqrt(torch.mean(step.square()))
                alpha = torch.as_tensor(self.alpha, dtype=proposal.dtype, device=proposal.device)

                if self.adaptive_damping and prev_residual >= 0.0:
                    target = prev_residual * self.residual_growth_tol
                    proposed = raw_residual * alpha
                    if proposed > target:
                        alpha = torch.clamp(
                            target / (raw_residual + 1e-12),
                            min=self.alpha_min,
                            max=self.alpha,
                        )

                state = state + alpha * step
                residual = raw_residual * alpha

            if self.state_clip_value > 0.0:
                state = torch.clamp(state, min=-self.state_clip_value, max=self.state_clip_value)

            prev_residual = residual
            if self.detach_between_steps and it < self.max_steps - 1:
                state = state.detach()

            if self.tol > 0.0 and residual < self.tol:
                break

        if state is None:
            raise RuntimeError("RecyclingModule produced no state.")

        data[self.block_out_field] = state
        data[self.out_field] = state
        return data
