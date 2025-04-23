import logging
from typing import Optional
from torch.utils.data import ConcatDataset
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    CombineModule,
)


def Combine(model, config: Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    logging.info("--- Building PerTypeScale Module ---")

    layers = {
        "wrapped_model": model,
        "combine": CombineModule,
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )