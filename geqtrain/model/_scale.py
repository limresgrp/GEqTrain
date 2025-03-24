import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    PerNodeAttrsScaleModule,
    PerTypeScaleModule,
)


def PerNodeAttrsScale(model, config:Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> PerTypeScaleModule:
    logging.info("--- Building PerTypeScale Module ---")

    layers = {
        "wrapped_model": model,
        "scale": (PerNodeAttrsScaleModule, dict(
            field=AtomicDataDict.NODE_OUTPUT_KEY,
            out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def PerTypeScale(model, config:Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> PerTypeScaleModule:
    logging.info("--- Building PerTypeScale Module ---")

    layers = {
        "wrapped_model": model,
        "scale": (PerTypeScaleModule, dict(
            field=AtomicDataDict.NODE_OUTPUT_KEY,
            out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )