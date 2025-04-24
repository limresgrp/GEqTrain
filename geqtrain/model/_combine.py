import logging
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    CombineModule,
)


def Combine(model, config: Config) -> SequentialGraphNetwork:
    logging.info("--- Building PerTypeScale Module ---")

    layers = {
        "wrapped_model": model,
        "combine": CombineModule,
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )