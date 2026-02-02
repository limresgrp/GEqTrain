import copy
import re
from typing import Any, Dict, List, Optional, Tuple

from hydra.utils import instantiate

from geqtrain.nn import GraphModuleMixin, SequentialGraphNetwork
from geqtrain.utils.config import Config
from geqtrain.utils.savenload import load_callable


def _extract_stack(config: Config) -> List[dict]:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, Config):
        model_cfg = model_cfg.as_dict()
    if not isinstance(model_cfg, dict):
        raise TypeError(f"Expected model config to be a dict, got {type(model_cfg)}")
    stack = model_cfg.get("stack")
    if stack is None:
        raise KeyError("Hydra model config is missing `model.stack`.")
    if not isinstance(stack, list):
        raise TypeError(f"Expected `model.stack` to be a list, got {type(stack)}")
    stack_prepend = model_cfg.get("stack_prepend")
    if stack_prepend is not None:
        if isinstance(stack_prepend, dict):
            stack_prepend = [stack_prepend]
        if not isinstance(stack_prepend, list):
            raise TypeError(
                f"Expected `model.stack_prepend` to be a list or dict, got {type(stack_prepend)}"
            )
        stack = list(stack_prepend) + list(stack)
    stack_append = model_cfg.get("stack_append")
    if stack_append is not None:
        if isinstance(stack_append, dict):
            stack_append = [stack_append]
        if not isinstance(stack_append, list):
            raise TypeError(
                f"Expected `model.stack_append` to be a list or dict, got {type(stack_append)}"
            )
        stack = list(stack) + list(stack_append)
    return stack


_PYTHON_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")
_DEFAULT_CALLABLE_PREFIXES = ("geqtrain.nn", "geqdiff.nn")


def _maybe_resolve_callable(value: Any, prefixes: Optional[Tuple[str, ...]] = None) -> Any:
    if not isinstance(value, str):
        return value
    if _PYTHON_PATH_RE.match(value):
        try:
            return load_callable(value)
        except Exception:
            return value
    if prefixes:
        for prefix in prefixes:
            try:
                return load_callable(value, prefix=prefix)
            except Exception:
                continue
    return value


def _resolve_callables(value: Any, callable_prefixes: Optional[Tuple[str, ...]] = None) -> Any:
    if isinstance(value, dict):
        resolved = {}
        for k, v in value.items():
            if k == "_target_":
                resolved[k] = v
                continue
            if isinstance(v, str) and f"{k}_kwargs" in value:
                resolved[k] = _maybe_resolve_callable(v, callable_prefixes)
            else:
                resolved[k] = _resolve_callables(v, callable_prefixes)
        return resolved
    if isinstance(value, list):
        return [_resolve_callables(v, callable_prefixes) for v in value]
    return _maybe_resolve_callable(value)


def _build_stack(
    stack_cfg: List[dict],
    initial_irreps_in: Optional[Dict] = None,
    dry_run: bool = False,
) -> SequentialGraphNetwork:
    irreps_in = copy.deepcopy(initial_irreps_in or {})
    if dry_run:
        irreps_in["_dry_run_mode"] = True

    built = []
    for idx, layer_cfg in enumerate(stack_cfg):
        if not isinstance(layer_cfg, dict):
            raise TypeError(f"Each stack entry must be a dict, got {type(layer_cfg)} at index {idx}")

        cfg = _resolve_callables(copy.deepcopy(layer_cfg), callable_prefixes=_DEFAULT_CALLABLE_PREFIXES)
        name = cfg.pop("name", None)

        if "irreps_in" not in cfg:
            layer = instantiate(cfg, irreps_in=irreps_in)
        else:
            layer = instantiate(cfg)

        if not isinstance(layer, GraphModuleMixin):
            raise TypeError(
                f"Stack layer {name or idx} did not return a GraphModuleMixin, got {type(layer).__name__}"
            )

        layer_name = name or getattr(layer, "name", f"layer{idx}")
        built.append((layer_name, layer))
        irreps_in = layer.irreps_out

    return SequentialGraphNetwork(dict(built))


def model_from_hydra_config(
    config: Config,
    initialize: bool = False,
    dataset=None,
    deploy: bool = False,
) -> Tuple[SequentialGraphNetwork, set]:
    stack_cfg = _extract_stack(config)

    if config.get("dry_run", False):
        dry_model = _build_stack(
            stack_cfg=stack_cfg,
            initial_irreps_in=config.get("irreps_in", {}),
            dry_run=True,
        )
        config["irreps_in"] = dry_model.irreps_out

    model = _build_stack(
        stack_cfg=stack_cfg,
        initial_irreps_in=config.get("irreps_in", {}),
        dry_run=False,
    )

    return model, set()
