import logging
from typing import Optional
from geqtrain.data import AtomicDataDict
from geqtrain.model._embedding import buildEmbeddingLayers
from geqtrain.utils import Config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EmbeddingInputAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    GotenInteractionModule,
)



def GotenModel(config:Config, model: Optional[SequentialGraphNetwork]) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildEmbeddingLayers(config, model)
    layers.update(buildGotenModelLayers())

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGotenModelLayers():
    logging.info("--- Building Goten Model ---")

    layers = {
        "interaction": GotenInteractionModule,
    }

    return layers