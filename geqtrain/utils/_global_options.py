# geqtrain/utils/_global_options.py
""" Code adapted from https://github.com/mir-group/nequip
"""

import warnings
import torch
import e3nn

from geqtrain.data.AtomicData import register_fields

from .auto_init import instantiate


_latest_global_config = {}
_fields_registered = False


def set_global_options(config, warn_on_override: bool = False) -> None:
    """Set global options for torch, e3nn, etc. Does NOT register fields."""
    global _latest_global_config
    _latest_global_config.update(dict(config))
    # Set TF32 support
    # See https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if torch.cuda.is_available() and "allow_tf32" in config:
        if torch.torch.backends.cuda.matmul.allow_tf32 is not config["allow_tf32"]:
            # update the setting
            if warn_on_override:
                warnings.warn(
                    f"Setting the GLOBAL value for allow_tf32 to {config['allow_tf32']} which is different than the previous value of {torch.torch.backends.cuda.matmul.allow_tf32}"
                )
            torch.backends.cuda.matmul.allow_tf32 = config["allow_tf32"]
            torch.backends.cudnn.allow_tf32 = config["allow_tf32"]

    k = "_jit_fusion_strategy"
    if k in config:
        new_strat = config.get(k)
        old_strat = torch.jit.set_fusion_strategy(new_strat)
        if warn_on_override and old_strat != new_strat:
            warnings.warn(
                f"Setting the GLOBAL value for jit fusion strategy to `{new_strat}` which is different than the previous value of `{old_strat}`"
            )

    if "default_dtype" in config:
        torch.set_default_dtype({
            "float32": torch.float32,
            "float64": torch.float64
        }[config["default_dtype"]])
    if config.get("grad_anomaly_mode", False):
        torch.autograd.set_detect_anomaly(True)

    e3nn.set_optimization_defaults(**config.get("e3nn_optimization_defaults", {}))


def register_all_fields(config):
    """Register fields for AtomicData. Idempotent."""
    global _fields_registered
    if not _fields_registered:
        instantiate(register_fields, all_args=config)
        _fields_registered = True


def apply_global_config(config, warn_on_override: bool = False):
    """Set global options and register fields, only once per process."""
    set_global_options(config, warn_on_override=warn_on_override)
    register_all_fields(config)
