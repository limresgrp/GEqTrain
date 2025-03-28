from typing import Optional
import logging

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.utils import Config

from geqtrain.model import update_config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    EmbeddingGraphAttrs,
)



def HeadlessNodeModel(
    config:Config, initialize: bool, dataset: Optional[ConcatDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildNodeModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildNodeModelLayers(config:Config):
    logging.info("--- Building Node Model ---")

    update_config(config)

    layers = {
        # -- Encode -- #
        "node_attrs":         EmbeddingNodeAttrs,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
        "graph_attrs":        EmbeddingGraphAttrs,
    }

    layers.update({
        "interaction": (InteractionModule, dict(
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            output_mul="hidden",
        )),
        "edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
    })

    return layers