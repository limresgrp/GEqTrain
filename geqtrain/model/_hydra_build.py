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
    return stack


_PYTHON_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def _resolve_callables(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_callables(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_callables(v) for v in value]
    if isinstance(value, str) and _PYTHON_PATH_RE.match(value):
        try:
            return load_callable(value)
        except Exception:
            return value
    return value


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

        cfg = _resolve_callables(copy.deepcopy(layer_cfg))
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
