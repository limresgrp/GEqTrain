import copy
from typing import Any, Dict, Mapping, Optional

import torch
import yaml

from geqtrain.data import AtomicDataDict
from geqtrain.utils.normalization import resolve_normalization_map


INFERENCE_METADATA_KEY = "inference_metadata_v1"
INFERENCE_METADATA_VERSION = 1


def _as_plain_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    return value


def _as_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "as_dict") and callable(config.as_dict):
        return config.as_dict()
    return {}


def build_inference_metadata_bundle(
    config: Any,
    normalization_stats_by_ensemble: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    cfg = _as_mapping(config)
    bundle: Dict[str, Any] = {
        "version": INFERENCE_METADATA_VERSION,
        "normalization": resolve_normalization_map(cfg),
        "denormalize_inference_outputs": bool(cfg.get("denormalize_inference_outputs", True)),
        "normalization_stats_by_ensemble": {},
        "default_ensemble": "0",
    }
    if normalization_stats_by_ensemble:
        stats_out: Dict[str, Dict[str, Any]] = {}
        for ensemble, stats in normalization_stats_by_ensemble.items():
            stats_out[str(ensemble)] = {k: _as_plain_value(v) for k, v in dict(stats).items()}
        bundle["normalization_stats_by_ensemble"] = stats_out
        if len(stats_out) > 0:
            bundle["default_ensemble"] = sorted(stats_out.keys())[0]
    return bundle


def dump_inference_metadata_bundle(bundle: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(bundle), sort_keys=False)


def load_inference_metadata_bundle(payload: Optional[str]) -> Dict[str, Any]:
    if payload is None or str(payload).strip() == "":
        return {}
    parsed = yaml.safe_load(payload)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _resolve_ensemble_key(ref_data: Mapping[str, Any], bundle: Mapping[str, Any]) -> Optional[str]:
    stats_by_ensemble = bundle.get("normalization_stats_by_ensemble", {})
    if not isinstance(stats_by_ensemble, Mapping) or len(stats_by_ensemble) == 0:
        return None

    ensemble_tensor = ref_data.get(AtomicDataDict.ENSEMBLE_INDEX_KEY, None)
    if torch.is_tensor(ensemble_tensor) and ensemble_tensor.numel() > 0:
        unique = torch.unique(ensemble_tensor.reshape(-1).to(dtype=torch.long)).tolist()
        if len(unique) == 1:
            candidate = str(int(unique[0]))
            if candidate in stats_by_ensemble:
                return candidate

    default_ensemble = str(bundle.get("default_ensemble", "0"))
    if default_ensemble in stats_by_ensemble:
        return default_ensemble
    return sorted(stats_by_ensemble.keys())[0]


def inject_inference_metadata_into_ref_data(
    ref_data: Dict[str, Any],
    batch_data: Optional[Mapping[str, Any]],
    inference_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(ref_data, dict):
        ref_data = dict(ref_data)
    if not isinstance(inference_metadata, Mapping) or len(inference_metadata) == 0:
        return ref_data

    # Prefer node types from the current batch when available (needed for per-type denormalization).
    if AtomicDataDict.NODE_TYPE_KEY not in ref_data and isinstance(batch_data, Mapping):
        node_types = batch_data.get(AtomicDataDict.NODE_TYPE_KEY, None)
        if node_types is not None:
            ref_data[AtomicDataDict.NODE_TYPE_KEY] = node_types

    ensemble_key = _resolve_ensemble_key(ref_data, inference_metadata)
    if ensemble_key is None:
        return ref_data

    stats_by_ensemble = inference_metadata.get("normalization_stats_by_ensemble", {})
    ensemble_stats = stats_by_ensemble.get(ensemble_key, {})
    if not isinstance(ensemble_stats, Mapping):
        return ref_data

    for key, value in ensemble_stats.items():
        if key not in ref_data:
            ref_data[key] = copy.deepcopy(value)
    return ref_data
