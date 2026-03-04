from typing import Optional, Dict, Any, List, Mapping, Set

import torch
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from ._graph_mixin import GraphModuleMixin, SequentialGraphNetwork
from geqtrain.utils.savenload import load_callable


def _resolve_block_class(path: str):
    try:
        return load_callable(path)
    except Exception:
        return load_callable(path, prefix="geqtrain.nn")


def _broadcast_mask_as_bool(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = mask
    if out.dtype != torch.bool:
        out = out > 0
    if out.ndim == 0 and target.ndim > 0:
        out = out.view(1)
    while out.ndim < target.ndim:
        out = out.unsqueeze(-1)
    return out


def _instantiate_block_from_config(block_cfg: Mapping[str, Any], irreps_in: Optional[Dict[str, Any]]):
    """Instantiate a block config with explicit irreps propagation."""
    from hydra.utils import instantiate

    cfg = dict(block_cfg)
    target = cfg.get("_target_", None)
    is_seq_target = False
    if isinstance(target, str):
        is_seq_target = target.endswith("SequentialGraphNetwork")
    elif isinstance(target, type):
        is_seq_target = issubclass(target, SequentialGraphNetwork)
    elif target is not None:
        # Some callables are already resolved by Hydra helper utilities.
        is_seq_target = getattr(target, "__name__", "") == "SequentialGraphNetwork"

    # Special handling for nested SequentialGraphNetwork configs:
    # each child GraphModule must receive the running irreps.
    if is_seq_target:
        modules_cfg = cfg.get("modules", None)
        if not isinstance(modules_cfg, Mapping):
            raise ValueError(
                "When `_target_` is `geqtrain.nn.SequentialGraphNetwork`, "
                "`modules` must be a mapping of named module configs."
            )
        current_irreps = irreps_in if irreps_in is not None else {}
        built_modules = {}
        for name, module_cfg in modules_cfg.items():
            if not isinstance(module_cfg, Mapping):
                raise TypeError(f"Sequential block module '{name}' must be a mapping config.")
            module_cfg_dict = dict(module_cfg)
            if "irreps_in" in module_cfg_dict:
                module = instantiate(module_cfg_dict)
            else:
                module = instantiate(module_cfg_dict, irreps_in=current_irreps)
            if not isinstance(module, GraphModuleMixin):
                raise TypeError(
                    f"Sequential block module '{name}' did not instantiate to GraphModuleMixin; "
                    f"got {type(module).__name__}."
                )
            built_modules[str(name)] = module
            current_irreps = module.irreps_out
        return SequentialGraphNetwork(built_modules)

    # Generic target config
    if "irreps_in" in cfg:
        return instantiate(cfg)
    return instantiate(cfg, irreps_in=irreps_in)


@compile_mode("script")
class RecyclingModule(GraphModuleMixin, torch.nn.Module):
    """
    Generic fixed-point style wrapper around a GraphModule.

    The wrapped `block` is run repeatedly. A recycled state tensor is updated with:
      - absolute mode: state <- state + alpha * (proposal - state)
      - delta mode:    state <- state + alpha * proposal
    where `proposal` is read from `block_out_field` after each block call.

    This module is Hydra-friendly: users can either pass an instantiated `block`
    or provide `block_target` + `block_kwargs`.
    """

    feedback_from_fields: List[str]
    feedback_to_fields: List[str]
    feedback_slice_starts: List[int]
    feedback_slice_ends: List[int]
    scalar_sync_fields: List[str]
    scalar_sync_dims: List[int]

    __constants__ = [
        "state_field",
        "block_out_field",
        "out_field",
        "max_steps",
        "detach_between_steps",
        "feedback_detach",
        "feedback_apply_mask",
        "feedback_mask_suffix",
        "predict_delta",
        "sync_scalar_outputs",
    ]

    def __init__(
        self,
        block: Optional[Any] = None,
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
        feedback_from_fields: Optional[List[str]] = None,
        feedback_to_fields: Optional[List[str]] = None,
        feedback_slice_starts: Optional[List[int]] = None,
        feedback_slice_ends: Optional[List[int]] = None,
        feedback_detach: bool = True,
        feedback_apply_mask: bool = True,
        feedback_mask_suffix: str = "__mask__",
        predict_delta: bool = False,
        sync_scalar_outputs: bool = True,
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
        elif isinstance(block, Mapping):
            block = _instantiate_block_from_config(block, irreps_in=irreps_in)

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
        self.feedback_detach = bool(feedback_detach)
        self.feedback_apply_mask = bool(feedback_apply_mask)
        self.feedback_mask_suffix = str(feedback_mask_suffix)
        self.predict_delta = bool(predict_delta)
        self.sync_scalar_outputs = bool(sync_scalar_outputs)
        if len(self.feedback_mask_suffix) == 0:
            self.feedback_mask_suffix = "__mask__"

        self.feedback_from_fields = list(feedback_from_fields or [])
        self.feedback_to_fields = list(feedback_to_fields or [])
        if len(self.feedback_from_fields) != len(self.feedback_to_fields):
            raise ValueError(
                "`feedback_from_fields` and `feedback_to_fields` must have the same length."
            )
        n_feedback = len(self.feedback_from_fields)

        if feedback_slice_starts is None:
            self.feedback_slice_starts = [-1] * n_feedback
        else:
            self.feedback_slice_starts = [int(x) for x in feedback_slice_starts]
        if feedback_slice_ends is None:
            self.feedback_slice_ends = [-1] * n_feedback
        else:
            self.feedback_slice_ends = [int(x) for x in feedback_slice_ends]
        if len(self.feedback_slice_starts) != n_feedback or len(self.feedback_slice_ends) != n_feedback:
            raise ValueError(
                "`feedback_slice_starts` and `feedback_slice_ends` must match the number of feedback mappings."
            )

        self.scalar_sync_fields = []
        self.scalar_sync_dims = []
        if self.sync_scalar_outputs:
            seen: Set[str] = set()
            for mod in self.block.modules():
                if (
                    hasattr(mod, "_has_out_field")
                    and hasattr(mod, "_has_scalar_out_field")
                    and hasattr(mod, "_out_field")
                    and hasattr(mod, "_scalar_out_field")
                    and hasattr(mod, "n_scalars_out")
                ):
                    try:
                        has_out = bool(getattr(mod, "_has_out_field"))
                        has_scalar = bool(getattr(mod, "_has_scalar_out_field"))
                        out_field = str(getattr(mod, "_out_field"))
                        scalar_field = str(getattr(mod, "_scalar_out_field"))
                        n_scalars = int(getattr(mod, "n_scalars_out"))
                    except Exception:
                        continue
                    if has_out and has_scalar and out_field == self.block_out_field and n_scalars > 0:
                        if scalar_field not in seen:
                            seen.add(scalar_field)
                            self.scalar_sync_fields.append(scalar_field)
                            self.scalar_sync_dims.append(n_scalars)

        initial_irreps_in = irreps_in if irreps_in is not None else self.block.irreps_in

        for from_field in self.feedback_from_fields:
            if from_field not in self.block.irreps_out:
                raise ValueError(
                    f"`feedback_from_fields` entry '{from_field}' is not produced by wrapped block. "
                    f"Available outputs: {list(self.block.irreps_out.keys())}"
                )
        for to_field in self.feedback_to_fields:
            if to_field not in initial_irreps_in and to_field not in self.block.irreps_out:
                raise ValueError(
                    f"`feedback_to_fields` entry '{to_field}' is not present in irreps_in or block outputs. "
                    f"Available input keys: {list(initial_irreps_in.keys())}; "
                    f"block outputs: {list(self.block.irreps_out.keys())}"
                )

        irreps_out = dict(self.block.irreps_out)
        irreps_out[self.out_field] = self.block.irreps_out[self.block_out_field]
        for to_field in self.feedback_to_fields:
            if to_field in initial_irreps_in:
                irreps_out[to_field] = initial_irreps_in[to_field]
            elif to_field in self.block.irreps_out:
                irreps_out[to_field] = self.block.irreps_out[to_field]
        self._init_irreps(irreps_in=initial_irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        state = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.state_field in data:
            state = data[self.state_field]
        prev_residual = torch.as_tensor(-1.0, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype, device=data[AtomicDataDict.POSITIONS_KEY].device)
        pinned_values = torch.jit.annotate(Dict[str, torch.Tensor], {})
        pinned_masks = torch.jit.annotate(Dict[str, torch.Tensor], {})

        if self.feedback_apply_mask and len(self.feedback_to_fields) > 0:
            for i in range(len(self.feedback_to_fields)):
                to_field = self.feedback_to_fields[i]
                mask_key = to_field + self.feedback_mask_suffix
                if to_field in data and mask_key in data:
                    pinned_values[to_field] = data[to_field].clone()
                    pinned_masks[to_field] = _broadcast_mask_as_bool(data[mask_key], data[to_field])

        for it in range(self.max_steps):
            if len(pinned_masks) > 0:
                for i in range(len(self.feedback_to_fields)):
                    to_field = self.feedback_to_fields[i]
                    if to_field in pinned_masks and to_field in pinned_values:
                        pin_mask = pinned_masks[to_field]
                        pin_value = pinned_values[to_field]
                        if to_field in data:
                            current = data[to_field]
                            if current.shape == pin_value.shape:
                                data[to_field] = torch.where(pin_mask, pin_value, current)

                        # Keep state synchronized with pinned targets when feedback target is state_field.
                        if to_field == self.state_field and state is not None and state.shape == pin_value.shape:
                            state = torch.where(pin_mask, pin_value, state)

            if state is not None:
                data[self.state_field] = state

            data = self.block(data)
            proposal = data[self.block_out_field]

            if state is None:
                if self.predict_delta:
                    step = proposal
                    raw_residual = torch.sqrt(torch.mean(step.square()))
                    alpha = torch.as_tensor(self.alpha, dtype=proposal.dtype, device=proposal.device)
                    state = torch.zeros_like(proposal) + alpha * step
                    residual = raw_residual * alpha
                else:
                    state = proposal
                    residual = torch.sqrt(torch.mean(state.square()))
            else:
                step = proposal if self.predict_delta else (proposal - state)
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

            if len(self.feedback_from_fields) > 0:
                for i in range(len(self.feedback_from_fields)):
                    from_field = self.feedback_from_fields[i]
                    to_field = self.feedback_to_fields[i]
                    slice_start = self.feedback_slice_starts[i]
                    slice_end = self.feedback_slice_ends[i]

                    if from_field == self.block_out_field:
                        if state is None:
                            raise RuntimeError("Cannot apply feedback from block_out_field before state is initialized.")
                        feedback_value = state
                    else:
                        if from_field not in data:
                            raise RuntimeError(
                                f"Feedback source field '{from_field}' missing from block output data."
                            )
                        feedback_value = data[from_field]

                    if slice_start >= 0 or slice_end >= 0:
                        s = slice_start if slice_start >= 0 else 0
                        e = slice_end if slice_end >= 0 else feedback_value.shape[-1]
                        feedback_value = feedback_value[..., s:e]

                    if self.feedback_detach and it < self.max_steps - 1:
                        feedback_value = feedback_value.detach()

                    if self.feedback_apply_mask:
                        mask_key = to_field + self.feedback_mask_suffix
                        if to_field in pinned_masks and to_field in pinned_values:
                            pin_mask = pinned_masks[to_field]
                            keep_value = pinned_values[to_field]
                            if keep_value.shape != feedback_value.shape:
                                raise RuntimeError(
                                    f"Feedback mask merge shape mismatch for '{to_field}'."
                                )
                            feedback_value = torch.where(pin_mask, keep_value, feedback_value)
                        elif mask_key in data:
                            mask = _broadcast_mask_as_bool(data[mask_key], feedback_value)
                            if to_field in data:
                                keep_value = data[to_field]
                                if keep_value.shape != feedback_value.shape:
                                    raise RuntimeError(
                                        f"Feedback mask merge shape mismatch for '{to_field}'."
                                    )
                            else:
                                keep_value = torch.zeros_like(feedback_value)
                            feedback_value = torch.where(mask, keep_value, feedback_value)

                    data[to_field] = feedback_value
                    if to_field == self.state_field:
                        state = feedback_value

            prev_residual = residual
            if self.detach_between_steps and it < self.max_steps - 1:
                state = state.detach()

            if self.tol > 0.0 and residual < self.tol:
                break

        if state is None:
            raise RuntimeError("RecyclingModule produced no state.")

        data[self.block_out_field] = state
        data[self.out_field] = state
        if self.sync_scalar_outputs and len(self.scalar_sync_fields) > 0:
            for i in range(len(self.scalar_sync_fields)):
                scalar_field = self.scalar_sync_fields[i]
                scalar_dim = self.scalar_sync_dims[i]
                if scalar_dim <= state.shape[-1]:
                    data[scalar_field] = state[..., :scalar_dim]
        return data
