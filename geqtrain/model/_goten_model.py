import logging

from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config

from geqtrain.model import update_config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EmbeddingAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    GotenInteractionModule,
)



def GotenModel(config:Config) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildGotenModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGotenModelLayers(config:Config):
    logging.info("--- Building Node Model ---")

    update_config(config)

    layers = {
        "node_attrs": (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            attributes=config.get('node_attributes'),
        )),
    }

    if 'edge_attributes' in config:
        edge_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            attributes=config.get('edge_attributes'),
        ))
        layers["edge_attrs"] = edge_embedder

    layers.update({
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
        "interaction": (GotenInteractionModule, dict(
            
        )),
    })

    return layers