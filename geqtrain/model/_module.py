import logging
from typing import Callable
from geqtrain.nn import SequentialGraphNetwork
from geqtrain.utils import Config


def Module(model, config: Config, cls: Callable, name: str) -> SequentialGraphNetwork:
    '''
    '''

    logging.info(f"--- Building Module ---")

    layers: dict = {
        "wrapped_model": model,
        name: cls,
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )