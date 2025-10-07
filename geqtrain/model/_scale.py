import logging
from typing import Optional
from geqtrain.data import AtomicDataDict
from geqtrain.data.dataset import compute_per_type_statistics
from geqtrain.utils import Config
from torch.utils.data import ConcatDataset

from geqtrain.nn import (
    SequentialGraphNetwork,
    PerNodeAttrsScaleModule,
    PerTypeUnscaleModule,
    PerTypeScaleModule,
)


def PerNodeAttrsScale(model, config: Config) -> SequentialGraphNetwork:
    logging.info("--- Building PerTypeScale Module ---")

    layers = {
        "wrapped_model": model,
        "scale": (PerNodeAttrsScaleModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def PerTypeScale(model, config:Config, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    logging.info("--- Building PerTypeScale Module ---")

    per_type_bias = config.get("per_type_bias", None)
    per_type_std = config.get("per_type_std", None)

    if per_type_bias is None and per_type_std is None:
        per_type_bias, per_type_std = compute_per_type_statistics(dataset, field=config.get("scale_field"), num_types=config.get("num_types"))

    layers = {
        "wrapped_model": model,
        "scale": (PerTypeScaleModule, dict(
            per_type_bias=per_type_bias,
            per_type_std=per_type_std,
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )


def PerTypeUnscale(model, config:Config, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    logging.info("--- Building PerTypeUnscale Module ---")

    per_type_bias = config.get("per_type_bias", None)
    per_type_std = config.get("per_type_std", None)

    if per_type_bias is None and per_type_std is None:
        per_type_bias, per_type_std = compute_per_type_statistics(dataset, field=config.get("scale_field"), num_types=config.get("num_types"))

    layers = {
        "scale": (PerTypeUnscaleModule, dict(
            per_type_bias=per_type_bias,
            per_type_std=per_type_std,
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )