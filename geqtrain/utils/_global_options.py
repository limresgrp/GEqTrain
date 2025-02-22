""" Code adapted from https://github.com/mir-group/nequip
"""

import warnings

import torch

import e3nn
import e3nn.util.jit

from geqtrain.data import register_fields
from .auto_init import instantiate


# for multiprocessing, we need to keep track of our latest global options so
# that we can reload/reset them in worker processes. While we could be more careful here,
# to keep only relevant keys, configs should have only small values (no big objects)
# and those should have references elsewhere anyway, so keeping references here is fine.
_latest_global_config = {}


def _get_latest_global_options() -> dict:
    """Get the config used latest to ``_set_global_options``.

    This is useful for getting worker processes into the same state as the parent.
    """
    global _latest_global_config
    return _latest_global_config


def _set_global_options(config, warn_on_override: bool = False) -> None:
    """Configure global options of libraries like `torch` and `e3nn` based on `config`.

    Args:
        warn_on_override: if True, will try to warn if new options are inconsistant with previously set ones.
    """
    # update these options into the latest global config.
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

    # Register fields:
    instantiate(register_fields, all_args=config)
    return
