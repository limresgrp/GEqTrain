import logging
from typing import Callable, Optional
from geqtrain.nn import SequentialGraphNetwork
from geqtrain.utils import Config


def Module(config: Config, model: Optional[SequentialGraphNetwork], cls: Callable, name: str) -> SequentialGraphNetwork:
    '''
    '''

    logging.info(f"--- Building Module ---")

    layers = {}
    if model is not None:
        layers.update({
            "wrapped_model": model,
        })
    layers.update({
        name: cls,
    })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )