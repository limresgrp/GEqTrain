from pathlib import Path
from typing import Iterable, Optional, Tuple

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .config import Config


def _is_legacy_config(path: Path) -> bool:
    return "old_config" in path.parts


def _split_config_path(path: Path) -> Tuple[Path, str]:
    parts = path.parts
    config_dir = path.parent
    config_name = path.stem
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "config":
            config_dir = Path(*parts[: idx + 1])
            rel_path = Path(*parts[idx + 1 :])
            config_name = rel_path.with_suffix("").as_posix()
            break
    return config_dir, config_name


def load_hydra_config(config_path: str, overrides: Optional[Iterable[str]] = None) -> Config:
    path = Path(config_path).expanduser().resolve()
    config_dir, config_name = _split_config_path(path)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["_hydra_config"] = True
    config = Config.from_dict(cfg_dict)
    config.filepath = str(path)
    return config


def load_config(
    config_path: str,
    overrides: Optional[Iterable[str]] = None,
    defaults: Optional[dict] = None,
) -> Config:
    path = Path(config_path).expanduser()
    if _is_legacy_config(path):
        config = Config.from_file(str(path), defaults=defaults or {})
        return config

    config = load_hydra_config(str(path), overrides=overrides)
    if defaults:
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
    return config
