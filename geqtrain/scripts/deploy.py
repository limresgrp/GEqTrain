# geqtrain/scripts/deploy.py

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from geqtrain.utils._global_options import set_global_options
from geqtrain.utils.deploy import build_deployment, get_base_deploy_parser
from geqtrain.utils import Config
import numpy as np


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

def main():
    parser = argparse.ArgumentParser(description="Deploy a GEqTrain model.")
    parser.add_argument("--verbose", default="INFO", type=str)
    # Get all the common arguments
    parser = get_base_deploy_parser(parser)
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

    # Call the core build function
    build_deployment(
        model_path=model_path,
        out_file=args.out_file,
        config=config,
        extra_metadata=extra_metadata,
    )

if __name__ == "__main__":
    main()
