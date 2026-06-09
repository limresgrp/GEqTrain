# geqtrain/scripts/deploy.py

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from e3nn.o3 import Irreps
from geqtrain.utils._global_options import set_global_options
from geqtrain.utils.deploy import build_deployment, get_base_deploy_parser
from geqtrain.utils import Config
from geqtrain.utils.inference_metadata import (
    INFERENCE_METADATA_KEY,
    build_inference_metadata_bundle,
    dump_inference_metadata_bundle,
)
from geqtrain.utils.normalization import (
    fit_transform_parameters,
    GLOBAL_MODE,
    PER_TYPE_MODE,
    get_global_stat_keys,
    get_per_type_stat_keys,
    get_transform_param_key,
    resolve_normalization_map,
)
import numpy as np
import torch


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        raw = input(prompt + suffix).strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def _prompt_npz_path() -> Path:
    while True:
        raw = input("Enter path to .npz file (or leave blank to cancel): ").strip()
        if not raw:
            raise RuntimeError("NPZ input canceled by user.")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            print(f"Path not found: {path}")
            continue
        if path.suffix.lower() != ".npz":
            print("Please provide a .npz file.")
            continue
        return path


def _format_npz_entries(npz: np.lib.npyio.NpzFile) -> List[Tuple[str, str]]:
    entries = []
    for key in npz.files:
        value = npz[key]
        entries.append((key, f"shape={value.shape}, dtype={value.dtype}"))
    return entries


def _parse_npz_selection(keys: List[str], selection: str) -> List[str]:
    selection = selection.strip()
    if selection.lower() == "all":
        return keys
    if not selection:
        return []
    selected: List[str] = []
    tokens = [token.strip() for token in selection.split(",") if token.strip()]
    for token in tokens:
        if token.isdigit():
            idx = int(token)
            if idx < 1 or idx > len(keys):
                raise ValueError(f"Index out of range: {idx}")
            selected.append(keys[idx - 1])
        else:
            if token not in keys:
                raise ValueError(f"Unknown key: {token}")
            selected.append(token)
    # Preserve input order while de-duplicating
    seen = set()
    ordered = []
    for key in selected:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _serialize_npz_value(value: np.ndarray) -> str:
    if value.dtype.kind in ("O", "S", "U"):
        data = value.astype(str).tolist()
    else:
        data = value.tolist()
    payload = {
        "type": "ndarray",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data": data,
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _collect_custom_metadata() -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    print("\nAdd custom key/value metadata (press Enter on key to finish).")
    while True:
        key = input("Metadata key: ").strip()
        if not key:
            break
        value = input("Metadata value: ").strip()
        metadata[key] = value
    return metadata


def _collect_npz_metadata() -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    while _prompt_yes_no("Add metadata from a .npz file?", default=False):
        try:
            npz_path = _prompt_npz_path()
        except RuntimeError as exc:
            print(str(exc))
            break

        with np.load(npz_path, allow_pickle=True) as npz:
            entries = _format_npz_entries(npz)
            print(f"\nLoaded {npz_path} with fields:")
            for idx, (key, desc) in enumerate(entries, start=1):
                print(f"  {idx:>2}) {key} ({desc})")

            while True:
                selection = input("Select keys to include ('all', '1,2', or names): ").strip()
                try:
                    selected_keys = _parse_npz_selection(npz.files, selection)
                except ValueError as exc:
                    print(f"Selection error: {exc}")
                    continue
                if not selected_keys:
                    print("No keys selected; please choose at least one.")
                    continue
                break

            prefix = input("Optional prefix for metadata keys (press Enter for none): ").strip()

            for key in selected_keys:
                meta_key = f"{prefix}{key}" if prefix else key
                if meta_key in metadata:
                    print(f"Overwriting existing metadata key '{meta_key}'.")
                metadata[meta_key] = _serialize_npz_value(npz[key])

    return metadata


def _collect_interactive_metadata() -> Dict[str, str]:
    metadata = {}
    metadata.update(_collect_npz_metadata())
    if _prompt_yes_no("Add custom key/value metadata?", default=False):
        metadata.update(_collect_custom_metadata())
    return metadata


def _required_normalization_metadata_keys(config: Config) -> List[str]:
    keys = []
    for field, spec in resolve_normalization_map(config.as_dict()).items():
        mode = spec.get("mode")
        if mode == PER_TYPE_MODE:
            keys.extend(get_per_type_stat_keys(field))
        elif mode == GLOBAL_MODE:
            keys.extend(get_global_stat_keys(field))

        transform_cfg = spec.get("transform", {})
        if transform_cfg.get("name", "none") == "yeo_johnson":
            keys.append(get_transform_param_key(field, "lambda"))

    seen = set()
    return [key for key in keys if not (key in seen or seen.add(key))]


def _fit_missing_transform_param(
    config: Config,
    npz: np.lib.npyio.NpzFile,
    field: str,
    param: str,
):
    if param != "lambda" or field not in npz.files:
        return None
    normalization = resolve_normalization_map(config.as_dict()).get(field, {})
    transform_cfg = normalization.get("transform", {})
    if transform_cfg.get("name", "none") != "yeo_johnson":
        return None

    values = npz[field]
    mask_key = f"{field}__mask__"
    if mask_key in npz.files:
        mask = npz[mask_key].astype(bool)
        values = values[~mask]
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    fitted = fit_transform_parameters(
        values=torch.from_numpy(values),
        transform_cfg=transform_cfg,
        irreps=Irreps(normalization["irreps"]) if normalization.get("irreps") else None,
    )
    return float(fitted["lambda"])


def _load_normalization_stats_from_npz(config: Config, npz_path: Path) -> Dict[str, object]:
    required_keys = _required_normalization_metadata_keys(config)
    if not required_keys:
        return {}
    if not npz_path.is_file():
        raise FileNotFoundError(f"Normalization stats NPZ not found: {npz_path}")

    stats = {}
    with np.load(npz_path, allow_pickle=True) as npz:
        for key in required_keys:
            if key not in npz.files:
                if key.startswith("_transform_."):
                    parts = key.split(".")
                    if len(parts) == 3:
                        fitted_value = _fit_missing_transform_param(
                            config=config,
                            npz=npz,
                            field=parts[1],
                            param=parts[2],
                        )
                        if fitted_value is not None:
                            logging.info(
                                "Fitted missing transform metadata '%s' from %s.",
                                key,
                                npz_path,
                            )
                            stats[key] = fitted_value
                            continue
                logging.warning(
                    "Normalization stats NPZ is missing optional metadata key '%s'; "
                    "continuing without it.",
                    key,
                )
                continue
            value = npz[key]
            if value.ndim == 0:
                stats[key] = value.item()
            elif value.dtype.kind in ("i", "u", "f", "b"):
                stats[key] = torch.from_numpy(value)
            else:
                stats[key] = value.tolist()
    return stats


def _build_inference_metadata_from_npz(config: Config, npz_path: Path) -> Dict[str, str]:
    stats = _load_normalization_stats_from_npz(config, npz_path)
    bundle = build_inference_metadata_bundle(
        config,
        normalization_stats_by_ensemble={0: stats} if stats else {},
    )
    return {INFERENCE_METADATA_KEY: dump_inference_metadata_bundle(bundle)}

def main():
    parser = argparse.ArgumentParser(description="Deploy a GEqTrain model.")
    parser.add_argument("--verbose", default="INFO", type=str)
    # Get all the common arguments
    parser = get_base_deploy_parser(parser)
    parser.add_argument(
        "--normalization-stats-npz",
        type=Path,
        default=None,
        help=(
            "Processed training NPZ containing normalization metadata keys "
            "such as _mean_.per_type.cs_iso and _std_.per_type.cs_iso. "
            "When provided, these stats are embedded in inference_metadata_v1."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.verbose.upper()))

    model_path = args.model
    config_path = model_path.parent / "config.yaml"
    config = Config.from_file(str(config_path))
    set_global_options(config, warn_on_override=False)

    # Handle the generic --extra-metadata arg
    cli_metadata = {}
    for item in args.extra_metadata:
        if "=" not in item:
            raise ValueError(f"Invalid metadata format '{item}'. Expected key=value.")
        key, value = item.split("=", 1)
        cli_metadata[key] = value

    interactive_metadata = _collect_interactive_metadata() if args.interactive_metadata else {}
    extra_metadata = {}
    extra_metadata.update(interactive_metadata)
    extra_metadata.update(cli_metadata)
    if args.normalization_stats_npz is not None:
        extra_metadata.update(
            _build_inference_metadata_from_npz(
                config,
                args.normalization_stats_npz.expanduser().resolve(),
            )
        )

    # Call the core build function
    build_deployment(
        model_path=model_path,
        out_file=args.out_file,
        config=config,
        extra_metadata=extra_metadata,
    )

if __name__ == "__main__":
    main()
