import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    SequentialGraphNetwork,
    PerTypeScaleModule,
)


def PerTypeScale(model, config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> PerTypeScaleModule:
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