import copy
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from e3nn.o3 import Irreps

from geqtrain.data import AtomicDataDict


MEAN_KEY_PREFIX = "_mean_"
STD_KEY_PREFIX = "_std_"
TRANSFORM_KEY_PREFIX = "_transform_"
PER_TYPE_MODE = "per_type"
GLOBAL_MODE = "global"
SUPPORTED_MODES = {PER_TYPE_MODE, GLOBAL_MODE}
SUPPORTED_TRANSFORMS = {"none", "signed_log1p", "yeo_johnson"}


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "as_dict") and callable(config.as_dict):
        return config.as_dict()
    if hasattr(config, "__dict__"):
        return vars(config)
    return {}


def _parse_mode_string(mode_str: str) -> Tuple[Optional[str], Optional[str]]:
    raw = str(mode_str).strip()
    if raw == "":
        return None, None
    parts = raw.split(":", 1)
    mode = parts[0].strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Invalid normalization mode '{mode_str}'. Expected one of {sorted(SUPPORTED_MODES)}."
        )
    irreps = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return mode, irreps


def _normalize_transform_name(name: Optional[str]) -> str:
    if name is None:
        return "none"
    name = str(name).strip().lower()
    aliases = {
        "identity": "none",
        "null": "none",
        "log": "signed_log1p",
        "log1p": "signed_log1p",
        "signed_log": "signed_log1p",
        "signed_log1p": "signed_log1p",
        "yeojohnson": "yeo_johnson",
        "yeo-johnson": "yeo_johnson",
        "yeo_johnson": "yeo_johnson",
    }
    normalized = aliases.get(name, name)
    if normalized not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Invalid target transform '{name}'. Expected one of {sorted(SUPPORTED_TRANSFORMS)}."
        )
    return normalized


def _normalize_transform_spec(spec: Any) -> Dict[str, Any]:
    if spec is None:
        return {"name": "none"}
    if isinstance(spec, str):
        return {"name": _normalize_transform_name(spec)}
    if not isinstance(spec, Mapping):
        raise TypeError(f"Invalid transform config type: {type(spec)}")

    name = _normalize_transform_name(
        spec.get("name", spec.get("type", spec.get("kind")))
    )
    out = {"name": name}
    if name == "yeo_johnson":
        lam = spec.get("lambda", spec.get("lmbda", "auto"))
        if isinstance(lam, str):
            lam = lam.strip().lower()
            lam = "auto" if lam == "auto" else float(lam)
        elif lam is not None:
            lam = float(lam)
        out["lambda"] = "auto" if lam is None else lam
        out["grid_min"] = float(spec.get("grid_min", -2.0))
        out["grid_max"] = float(spec.get("grid_max", 2.0))
        out["grid_steps"] = int(spec.get("grid_steps", 121))
        out["max_samples"] = int(spec.get("max_samples", 200000))
    return out


def _parse_field_spec(
    field: str,
    value: Any,
    *,
    default_apply_on_dataset: bool,
) -> Dict[str, Any]:
    if isinstance(value, str):
        mode, irreps = _parse_mode_string(value)
        return {
            "mode": mode,
            "irreps": irreps,
            "transform": {"name": "none"},
            "apply_on_dataset": default_apply_on_dataset,
        }

    if not isinstance(value, Mapping):
        raise TypeError(
            f"Invalid normalization entry for field '{field}': expected str or dict, got {type(value)}."
        )

    mode_raw = value.get(
        "mode",
        value.get("scope", value.get("normalization_mode", value.get("standardization_mode"))),
    )
    mode = None
    irreps = value.get("irreps", value.get("irrep"))

    if isinstance(mode_raw, str):
        parsed_mode, parsed_irreps = _parse_mode_string(mode_raw)
        mode = parsed_mode
        if irreps is None:
            irreps = parsed_irreps
    elif mode_raw is not None:
        mode = str(mode_raw).strip().lower()
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Invalid normalization mode '{mode_raw}' for field '{field}'. "
                f"Expected one of {sorted(SUPPORTED_MODES)}."
            )

    transform_spec = _normalize_transform_spec(
        value.get("transform", value.get("target_transform"))
    )
    apply_on_dataset = value.get(
        "apply_on_dataset",
        value.get("dataset", value.get("for_dataset", default_apply_on_dataset)),
    )
    apply_on_dataset = bool(apply_on_dataset)
    return {
        "mode": mode,
        "irreps": irreps,
        "transform": transform_spec,
        "apply_on_dataset": apply_on_dataset,
    }


def _merge_specs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    if override.get("mode") is not None:
        merged["mode"] = override["mode"]
    if override.get("irreps") is not None:
        merged["irreps"] = override["irreps"]
    if "transform" in override:
        merged["transform"] = copy.deepcopy(override["transform"])
    if "apply_on_dataset" in override:
        merged["apply_on_dataset"] = bool(override["apply_on_dataset"])
    if "transform" not in merged or merged["transform"] is None:
        merged["transform"] = {"name": "none"}
    merged.setdefault("apply_on_dataset", True)
    return merged


def resolve_normalization_map(config: Any) -> Dict[str, Dict[str, Any]]:
    cfg = _as_mapping(config)
    out: Dict[str, Dict[str, Any]] = {}
    source = cfg.get("normalization", {})
    if not isinstance(source, Mapping):
        return out

    for field, value in source.items():
        parsed = _parse_field_spec(
            field,
            value,
            default_apply_on_dataset=True,
        )
        if field in out:
            out[field] = _merge_specs(out[field], parsed)
        else:
            out[field] = parsed

    for spec in out.values():
        spec.setdefault("transform", {"name": "none"})
        spec["transform"] = _normalize_transform_spec(spec["transform"])
        spec.setdefault("apply_on_dataset", True)
    return out


def get_per_type_stat_keys(field: str) -> Tuple[str, str]:
    return (
        f"{MEAN_KEY_PREFIX}.{PER_TYPE_MODE}.{field}",
        f"{STD_KEY_PREFIX}.{PER_TYPE_MODE}.{field}",
    )


def get_global_stat_keys(field: str) -> Tuple[str, str]:
    return (
        f"{MEAN_KEY_PREFIX}.{GLOBAL_MODE}.{field}",
        f"{STD_KEY_PREFIX}.{GLOBAL_MODE}.{field}",
    )


def get_transform_param_key(field: str, param: str) -> str:
    return f"{TRANSFORM_KEY_PREFIX}.{field}.{param}"


def _yeo_johnson_np(values: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    pos = values >= 0
    neg = ~pos

    if np.any(pos):
        if abs(lam) < 1e-12:
            out[pos] = np.log1p(values[pos])
        else:
            out[pos] = ((values[pos] + 1.0) ** lam - 1.0) / lam
    if np.any(neg):
        if abs(lam - 2.0) < 1e-12:
            out[neg] = -np.log1p(-values[neg])
        else:
            out[neg] = -((1.0 - values[neg]) ** (2.0 - lam) - 1.0) / (2.0 - lam)
    return out


def _estimate_yeo_johnson_lambda(values: torch.Tensor, transform_cfg: Dict[str, Any]) -> float:
    flat = values.detach().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return 1.0

    max_samples = int(transform_cfg.get("max_samples", 200000))
    if flat.numel() > max_samples:
        idx = torch.linspace(
            0,
            flat.numel() - 1,
            steps=max_samples,
            device=flat.device,
        ).long()
        flat = flat[idx]

    arr = flat.to(dtype=torch.float64).cpu().numpy()
    grid_min = float(transform_cfg.get("grid_min", -2.0))
    grid_max = float(transform_cfg.get("grid_max", 2.0))
    grid_steps = int(transform_cfg.get("grid_steps", 121))
    lambdas = np.linspace(grid_min, grid_max, grid_steps, dtype=np.float64)

    pos = arr >= 0
    neg = ~pos
    log_pos = np.log1p(arr[pos]) if np.any(pos) else None
    log_neg = np.log1p(-arr[neg]) if np.any(neg) else None

    best_lam = 1.0
    best_ll = -np.inf
    n = float(arr.size)
    eps = 1e-12

    for lam in lambdas:
        transformed = _yeo_johnson_np(arr, float(lam))
        variance = float(np.var(transformed))
        if variance <= eps:
            continue
        ll = -0.5 * n * np.log(variance)
        if log_pos is not None:
            ll += (lam - 1.0) * float(np.sum(log_pos))
        if log_neg is not None:
            ll += (1.0 - lam) * float(np.sum(log_neg))
        if ll > best_ll:
            best_ll = ll
            best_lam = float(lam)
    return best_lam


def _has_non_scalar_irreps(irreps: Optional[Irreps]) -> bool:
    return irreps is not None and any(ir.l > 0 for _, ir in irreps)


def _validate_irreps_compatible_shape(values: torch.Tensor, irreps: Irreps) -> None:
    if values.dim() == 0 or values.shape[-1] != irreps.dim:
        raise ValueError(
            f"Expected tensor with last dimension {irreps.dim} for irreps={irreps}, got shape {tuple(values.shape)}."
        )


def _collect_transform_fit_values(values: torch.Tensor, irreps: Irreps) -> torch.Tensor:
    _validate_irreps_compatible_shape(values, irreps)
    flat = values.reshape(-1, values.shape[-1])
    fit_values = []
    for (mul, ir), slc in zip(irreps, irreps.slices()):
        block = flat[:, slc]
        if ir.l == 0:
            fit_values.append(block.reshape(-1))
        else:
            block = block.reshape(flat.shape[0], mul, ir.dim)
            norms = torch.linalg.norm(block, dim=-1)
            fit_values.append(norms.reshape(-1))
    if len(fit_values) == 0:
        return flat.reshape(-1)
    return torch.cat(fit_values, dim=0)


def _apply_forward_scalar_transform(values: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    name = cfg["name"]
    if name == "none":
        return values
    if name == "signed_log1p":
        return torch.sign(values) * torch.log1p(torch.abs(values))
    if name == "yeo_johnson":
        lam = float(cfg.get("lambda", 1.0))
        out = torch.empty_like(values)
        pos = values >= 0
        neg = ~pos
        if pos.any():
            if abs(lam) < 1e-12:
                out[pos] = torch.log1p(values[pos])
            else:
                out[pos] = ((values[pos] + 1.0).pow(lam) - 1.0) / lam
        if neg.any():
            if abs(lam - 2.0) < 1e-12:
                out[neg] = -torch.log1p(-values[neg])
            else:
                out[neg] = -((1.0 - values[neg]).pow(2.0 - lam) - 1.0) / (2.0 - lam)
        return out
    raise ValueError(f"Unsupported transform '{name}'")


def _apply_inverse_scalar_transform(values: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    name = cfg["name"]
    if name == "none":
        return values
    if name == "signed_log1p":
        return torch.sign(values) * torch.expm1(torch.abs(values))
    if name == "yeo_johnson":
        lam = float(cfg.get("lambda", 1.0))
        out = torch.empty_like(values)
        pos = values >= 0
        neg = ~pos
        if pos.any():
            if abs(lam) < 1e-12:
                out[pos] = torch.expm1(values[pos])
            else:
                out[pos] = (lam * values[pos] + 1.0).clamp_min(1e-12).pow(1.0 / lam) - 1.0
        if neg.any():
            if abs(lam - 2.0) < 1e-12:
                out[neg] = 1.0 - torch.exp(-values[neg])
            else:
                base = (1.0 - (2.0 - lam) * values[neg]).clamp_min(1e-12)
                out[neg] = 1.0 - base.pow(1.0 / (2.0 - lam))
        return out
    raise ValueError(f"Unsupported transform '{name}'")


def fit_transform_parameters(
    values: torch.Tensor,
    transform_cfg: Dict[str, Any],
    irreps: Optional[Irreps] = None,
) -> Dict[str, Any]:
    cfg = _normalize_transform_spec(transform_cfg)
    if cfg["name"] == "yeo_johnson":
        lam = cfg.get("lambda", "auto")
        if lam == "auto":
            fit_values = values
            if _has_non_scalar_irreps(irreps):
                fit_values = _collect_transform_fit_values(values, irreps)
            cfg["lambda"] = _estimate_yeo_johnson_lambda(fit_values, cfg)
        else:
            cfg["lambda"] = float(lam)
    return cfg


def apply_forward_transform(
    values: torch.Tensor,
    transform_cfg: Dict[str, Any],
    irreps: Optional[Irreps] = None,
) -> torch.Tensor:
    cfg = _normalize_transform_spec(transform_cfg)
    if not _has_non_scalar_irreps(irreps):
        return _apply_forward_scalar_transform(values, cfg)

    assert irreps is not None
    _validate_irreps_compatible_shape(values, irreps)
    flat = values.reshape(-1, values.shape[-1])
    out = flat.clone()
    eps = torch.finfo(flat.dtype).eps

    for (mul, ir), slc in zip(irreps, irreps.slices()):
        if ir.l == 0:
            out[:, slc] = _apply_forward_scalar_transform(flat[:, slc], cfg)
            continue

        block = flat[:, slc].reshape(flat.shape[0], mul, ir.dim)
        norms = torch.linalg.norm(block, dim=-1, keepdim=True)
        transformed_norms = _apply_forward_scalar_transform(norms, cfg)
        scale = torch.where(
            norms > 0.0,
            transformed_norms / norms.clamp_min(eps),
            torch.zeros_like(norms),
        )
        out[:, slc] = (block * scale).reshape(flat.shape[0], mul * ir.dim)
    return out.reshape(values.shape)


def apply_inverse_transform(
    values: torch.Tensor,
    transform_cfg: Dict[str, Any],
    irreps: Optional[Irreps] = None,
) -> torch.Tensor:
    cfg = _normalize_transform_spec(transform_cfg)
    if not _has_non_scalar_irreps(irreps):
        return _apply_inverse_scalar_transform(values, cfg)

    assert irreps is not None
    _validate_irreps_compatible_shape(values, irreps)
    flat = values.reshape(-1, values.shape[-1])
    out = flat.clone()
    eps = torch.finfo(flat.dtype).eps

    for (mul, ir), slc in zip(irreps, irreps.slices()):
        if ir.l == 0:
            out[:, slc] = _apply_inverse_scalar_transform(flat[:, slc], cfg)
            continue

        block = flat[:, slc].reshape(flat.shape[0], mul, ir.dim)
        transformed_norms = torch.linalg.norm(block, dim=-1, keepdim=True)
        norms = _apply_inverse_scalar_transform(transformed_norms, cfg)
        scale = torch.where(
            transformed_norms > 0.0,
            norms / transformed_norms.clamp_min(eps),
            torch.zeros_like(transformed_norms),
        )
        out[:, slc] = (block * scale).reshape(flat.shape[0], mul * ir.dim)
    return out.reshape(values.shape)


def serialize_transform_params(field: str, transform_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    cfg = _normalize_transform_spec(transform_cfg)
    out: Dict[str, torch.Tensor] = {}
    if cfg["name"] == "yeo_johnson":
        out[get_transform_param_key(field, "lambda")] = torch.tensor(
            float(cfg.get("lambda", 1.0)),
            dtype=torch.float32,
        )
    return out


def _resolve_runtime_transform(
    field: str,
    ref_data: Mapping[str, Any],
    spec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = _normalize_transform_spec((spec or {}).get("transform"))
    if cfg["name"] == "yeo_johnson" and cfg.get("lambda", "auto") == "auto":
        key = get_transform_param_key(field, "lambda")
        lam_val = ref_data.get(key, None)
        if lam_val is None:
            cfg["lambda"] = 1.0
        elif torch.is_tensor(lam_val):
            cfg["lambda"] = float(lam_val.reshape(-1)[0].item())
        else:
            cfg["lambda"] = float(lam_val)
    return cfg


def _inverse_standardize_per_type(
    values: torch.Tensor,
    ref_data: Mapping[str, Any],
    field: str,
    irreps: Optional[Irreps],
) -> torch.Tensor:
    mean_key, std_key = get_per_type_stat_keys(field)
    if mean_key not in ref_data or std_key not in ref_data:
        return values
    if AtomicDataDict.NODE_TYPE_KEY not in ref_data:
        return values

    node_types = ref_data[AtomicDataDict.NODE_TYPE_KEY]
    if not torch.is_tensor(node_types):
        return values
    node_types = node_types.to(device=values.device, dtype=torch.long).squeeze(-1)
    if values.dim() == 0 or values.shape[0] != node_types.shape[0]:
        return values

    means = ref_data[mean_key]
    stds = ref_data[std_key]
    if not torch.is_tensor(means):
        means = torch.as_tensor(means, device=values.device, dtype=values.dtype)
    else:
        means = means.to(device=values.device, dtype=values.dtype)
    if not torch.is_tensor(stds):
        stds = torch.as_tensor(stds, device=values.device, dtype=values.dtype)
    else:
        stds = stds.to(device=values.device, dtype=values.dtype)

    means_expanded = means[node_types]
    stds_expanded = stds[node_types]
    out = values.clone()

    if irreps is not None:
        i = 0
        for (_, ir), slc in zip(irreps, irreps.slices()):
            mean_bc = means_expanded[:, i:i + 1]
            std_bc = stds_expanded[:, i:i + 1]
            if ir.l == 0:
                out[:, slc] = out[:, slc] * std_bc + mean_bc
            else:
                out[:, slc] = out[:, slc] * std_bc
            i += 1
        return out

    mean_bc = means_expanded
    std_bc = stds_expanded
    if mean_bc.dim() == 1:
        mean_bc = mean_bc.unsqueeze(-1)
    if std_bc.dim() == 1:
        std_bc = std_bc.unsqueeze(-1)
    while mean_bc.dim() < out.dim():
        mean_bc = mean_bc.unsqueeze(-1)
    while std_bc.dim() < out.dim():
        std_bc = std_bc.unsqueeze(-1)
    return out * std_bc + mean_bc


def _inverse_standardize_global(
    values: torch.Tensor,
    ref_data: Mapping[str, Any],
    field: str,
) -> torch.Tensor:
    mean_key, std_key = get_global_stat_keys(field)
    if mean_key not in ref_data or std_key not in ref_data:
        return values

    mean = ref_data[mean_key]
    std = ref_data[std_key]
    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
    else:
        mean = mean.to(device=values.device, dtype=values.dtype)
    if not torch.is_tensor(std):
        std = torch.as_tensor(std, device=values.device, dtype=values.dtype)
    else:
        std = std.to(device=values.device, dtype=values.dtype)

    if mean.numel() != 1:
        mean = mean.reshape(-1).mean()
    if std.numel() != 1:
        std = std.reshape(-1).mean()
    if abs(float(std.item())) <= 1e-8:
        return values
    return values * std + mean


def denormalize_tensor(
    values: torch.Tensor,
    ref_data: Mapping[str, Any],
    field: str,
    spec: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    if not torch.is_tensor(values):
        return values

    spec = spec or {}
    mode = spec.get("mode")
    irreps = None
    irreps_str = spec.get("irreps")
    if irreps_str:
        irreps = Irreps(irreps_str)

    out = values
    preferred_modes = [mode] if mode in SUPPORTED_MODES else []
    preferred_modes.extend([PER_TYPE_MODE, GLOBAL_MODE])
    seen = set()
    ordered_modes = [m for m in preferred_modes if not (m in seen or seen.add(m))]

    for current_mode in ordered_modes:
        if current_mode == PER_TYPE_MODE:
            updated = _inverse_standardize_per_type(out, ref_data, field, irreps)
        else:
            updated = _inverse_standardize_global(out, ref_data, field)
        if updated is not out:
            out = updated
            break

    transform_cfg = _resolve_runtime_transform(field, ref_data, spec)
    out = apply_inverse_transform(out, transform_cfg, irreps=irreps)
    return out


def denormalize_prediction_dict(
    pred: Mapping[str, Any],
    ref_data: Mapping[str, Any],
    normalization_fields: Mapping[str, Dict[str, Any]],
    *,
    in_place: bool = False,
) -> Dict[str, Any]:
    out = pred if in_place else dict(pred)
    for field, spec in normalization_fields.items():
        if field not in out:
            continue
        value = out[field]
        if not torch.is_tensor(value):
            continue
        out[field] = denormalize_tensor(value, ref_data, field, spec)
    return out
