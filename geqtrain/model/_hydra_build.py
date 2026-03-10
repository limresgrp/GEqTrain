import copy
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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

    def _flatten(items: Iterable[Any]) -> List[Any]:
        out: List[Any] = []
        for x in items:
            if x is None:
                continue
            if isinstance(x, list):
                out.extend(_flatten(x))
            else:
                out.append(x)
        return out

    def _is_layer_dict(d: dict) -> bool:
        return "_target_" in d

    def _normalize_stack_modifier(value: Any, *, key_name: str) -> List[dict]:
        """Normalize `model.stack_prepend` / `model.stack_append`.

        Supported forms:
        - dict (single layer dict with `_target_`)
        - list of layer dicts (can contain nested lists)
        - dict-of-groups: `{name: [layer_dicts...]}` (helps compose multiple templates via Hydra merges)
        """
        if value is None:
            return []
        if isinstance(value, list):
            items = _flatten(value)
            out: List[dict] = []
            for item in items:
                out.extend(_normalize_stack_modifier(item, key_name=key_name))
            return out
        if isinstance(value, dict):
            if _is_layer_dict(value):
                return [value]
            # dict-of-groups
            out: List[dict] = []
            for group_key, group_value in value.items():
                out.extend(_normalize_stack_modifier(group_value, key_name=f"{key_name}.{group_key}"))
            return out
        raise TypeError(f"Expected `{key_name}` to be a list or dict, got {type(value)}")

    stack = _flatten(stack)
    stack_prepend = _normalize_stack_modifier(model_cfg.get("stack_prepend"), key_name="model.stack_prepend")
    stack_append = _normalize_stack_modifier(model_cfg.get("stack_append"), key_name="model.stack_append")

    stack = list(stack_prepend) + list(stack) + list(stack_append)
    if not all(isinstance(x, dict) for x in stack):
        bad = [(i, type(x).__name__) for i, x in enumerate(stack) if not isinstance(x, dict)]
        raise TypeError(f"Hydra model stack entries must be dicts; got non-dicts at: {bad}")
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
                # Allow short `_target_` names like `ScalarMLPFunction` by resolving via prefixes.
                resolved[k] = _maybe_resolve_callable(v, callable_prefixes)
                continue
            if isinstance(v, str) and f"{k}_kwargs" in value:
                resolved[k] = _maybe_resolve_callable(v, callable_prefixes)
            else:
                resolved[k] = _resolve_callables(v, callable_prefixes)
        return resolved
    if isinstance(value, list):
        return [_resolve_callables(v, callable_prefixes) for v in value]
    return _maybe_resolve_callable(value)


def _get_stack_prefix(config: Config) -> Tuple[Optional[str], bool]:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, Config):
        model_cfg = model_cfg.as_dict()
    if not isinstance(model_cfg, dict):
        return None, True

    stack_prefix = model_cfg.get("prefix", None)
    if stack_prefix is None:
        stack_prefix = model_cfg.get("stack_prefix", None)
    if stack_prefix is not None and not isinstance(stack_prefix, str):
        raise TypeError(
            f"Expected `model.prefix` / `model.stack_prefix` to be a string, got {type(stack_prefix)}"
        )

    prefix_strict = model_cfg.get("prefix_strict", True)
    if not isinstance(prefix_strict, bool):
        raise TypeError(f"Expected `model.prefix_strict` to be a bool, got {type(prefix_strict)}")

    return stack_prefix, prefix_strict


def _apply_layer_overrides_from_prefix(
    layer_cfg: dict,
    config: Mapping[str, Any],
    *,
    stack_prefix: str,
    strict: bool,
) -> dict:
    """Apply overrides of the form `<stack_prefix>_<layer_name>_<param>` to a layer cfg dict.

    Resolution rules for `<param>` (in order):
    1) If `<param>` matches an existing key in `layer_cfg`, override it directly.
    2) If `<param>` targets a nested dict explicitly, override within it:
       - `<nested_dict_key>_<subkey>` (e.g. `basis_kwargs_r_max`)
       - `<nested_dict_key_without__kwargs>_<subkey>` (e.g. `basis_r_max` -> `basis_kwargs.r_max`)
    3) Otherwise, if `<param>` matches one or more keys inside nested dicts, override all matches.
    """
    if not stack_prefix:
        return layer_cfg
    if not isinstance(layer_cfg, dict):
        return layer_cfg

    layer_name = layer_cfg.get("name", None)
    if not isinstance(layer_name, str) or not layer_name:
        return layer_cfg

    prefix = f"{stack_prefix}_{layer_name}_"
    nested_dict_keys = [k for k, v in layer_cfg.items() if isinstance(k, str) and isinstance(v, dict)]

    for full_key, value in config.items():
        if not isinstance(full_key, str) or not full_key.startswith(prefix):
            continue
        tail = full_key[len(prefix) :]
        if not tail:
            continue

        # (2) Explicit nested dict targeting: `<nested_dict_key>_<subkey>` or `<nested_base>_<subkey>`
        explicit_updated = False
        if "_" in tail:
            for dkey in nested_dict_keys:
                candidates = [dkey]
                if dkey.endswith("_kwargs"):
                    candidates.append(dkey[: -len("_kwargs")])
                for cand in candidates:
                    if not cand:
                        continue
                    cand_prefix = f"{cand}_"
                    if not tail.startswith(cand_prefix):
                        continue
                    subkey = tail[len(cand_prefix) :]
                    if not subkey:
                        continue
                    if subkey not in layer_cfg[dkey] and strict:
                        raise KeyError(
                            f"Prefixed override `{full_key}` targets `{layer_name}.{dkey}.{subkey}`, "
                            f"but `{subkey}` is not present under `{dkey}`. Available: {sorted(layer_cfg[dkey].keys())}"
                        )
                    layer_cfg[dkey][subkey] = value
                    explicit_updated = True
                    break
                if explicit_updated:
                    break
        if explicit_updated:
            continue

        # (1) Direct key override
        if tail in layer_cfg:
            layer_cfg[tail] = value
            continue

        # (3) Fuzzy nested override: update all nested dicts that contain `tail`
        matches = [dkey for dkey in nested_dict_keys if tail in layer_cfg[dkey]]
        if matches:
            for dkey in matches:
                layer_cfg[dkey][tail] = value
            continue

        # Convenience alias for kwargs dicts:
        # allow `<prefix>_<layer>_latent_kwargs` to match a unique key like `readout_latent_kwargs`.
        if isinstance(tail, str) and tail.endswith("_kwargs"):
            candidates = [
                k
                for k in layer_cfg.keys()
                if isinstance(k, str) and k.endswith(f"_{tail}")
            ]
            if len(candidates) == 1:
                target_key = candidates[0]
                layer_cfg[target_key] = value
                continue

        if strict:
            available_direct = sorted(k for k in layer_cfg.keys() if isinstance(k, str))
            available_nested = {k: sorted(v.keys()) for k, v in layer_cfg.items() if isinstance(v, dict)}
            raise KeyError(
                f"Prefixed override `{full_key}` could not be applied.\n"
                f"- Layer: {layer_name}\n"
                f"- Tail: {tail}\n"
                f"- Direct keys: {available_direct}\n"
                f"- Nested dict keys: {sorted(available_nested.keys())}"
            )

    return layer_cfg


def _build_stack(
    stack_cfg: List[dict],
    initial_irreps_in: Optional[Dict] = None,
    config: Optional[Mapping[str, Any]] = None,
    stack_prefix: Optional[str] = None,
    prefix_strict: bool = True,
    dry_run: bool = False,
) -> SequentialGraphNetwork:
    irreps_in = copy.deepcopy(initial_irreps_in or {})
    if dry_run:
        irreps_in["_dry_run_mode"] = True

    built = []
    for idx, layer_cfg in enumerate(stack_cfg):
        if not isinstance(layer_cfg, dict):
            raise TypeError(f"Each stack entry must be a dict, got {type(layer_cfg)} at index {idx}")

        cfg = copy.deepcopy(layer_cfg)
        if stack_prefix and config is not None:
            cfg = _apply_layer_overrides_from_prefix(
                cfg,
                config,
                stack_prefix=stack_prefix,
                strict=prefix_strict,
            )
        cfg = _resolve_callables(cfg, callable_prefixes=_DEFAULT_CALLABLE_PREFIXES)
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
    stack_prefix, prefix_strict = _get_stack_prefix(config)
    config_dict = config.as_dict() if isinstance(config, Config) else dict(config)

    if config.get("dry_run", False):
        dry_model = _build_stack(
            stack_cfg=stack_cfg,
            initial_irreps_in=config.get("irreps_in", {}),
            config=config_dict,
            stack_prefix=stack_prefix,
            prefix_strict=prefix_strict,
            dry_run=True,
        )
        config["irreps_in"] = dry_model.irreps_out

    model = _build_stack(
        stack_cfg=stack_cfg,
        initial_irreps_in=config.get("irreps_in", {}),
        config=config_dict,
        stack_prefix=stack_prefix,
        prefix_strict=prefix_strict,
        dry_run=False,
    )

    return model, set()
