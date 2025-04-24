import logging
from typing import Optional, Dict

from geqtrain.data import AtomicDataDict
from geqtrain.nn import SequentialGraphNetwork

from geqtrain.utils import Config
from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset

from importlib import import_module


def Heads(model, config: Config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    '''
    instanciates a layer with multiple ReadoutModules
    '''

    logging.info("--- Building Heads Module ---")

    layers: dict = {
        "wrapped_model": model,
    }

    heads: Dict[str, Dict[str, str]] = config.get("heads")
    assert isinstance(heads, dict), f"'heads' must be a dict of dicts. Found type {type(heads)}"
    for head_name, head_kwargs in heads.items():
        if "field" not in head_kwargs:
            head_kwargs["field"] = AtomicDataDict.NODE_FEATURES_KEY

        if "model" not in head_kwargs:
            head_kwargs["model"] = "geqtrain.nn.ReadoutModule"
        
        head_kwargs.update(dict(ignore_amp=True,))
        
        model = head_kwargs.pop("model")
        try:
            module_name = ".".join(model.split(".")[:-1])
            class_name  = ".".join(model.split(".")[-1:])
            model = getattr(import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            logging.error(f"Failed to import or access the model '{model}': {e}")
            raise

        # ! ReadoutModule heads
        layers.update({head_name: (model, head_kwargs)})

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )