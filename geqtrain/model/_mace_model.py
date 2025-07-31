import logging
from typing import Optional
from geqtrain.model._embedding import buildEmbeddingLayers
from geqtrain.utils import Config
from geqtrain.nn import SequentialGraphNetwork, MACEModule



def MACEModel(config:Config, model: Optional[SequentialGraphNetwork]) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildEmbeddingLayers(config, model)
    layers.update(buildMACEModelLayers())

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildMACEModelLayers():
    logging.info("--- Building MACE Model ---")

    
    layers = {
        "interaction": MACEModule,
    }

    return layers