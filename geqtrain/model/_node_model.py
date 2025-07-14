import logging
from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
)
from geqtrain.model._embedding import buildEmbeddingLayers


def HeadlessNodeModel(config:Config) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildEmbeddingLayers(config)
    layers.update(buildNodeModelLayers())

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildNodeModelLayers():
    logging.info("--- Building Node Model ---")

    layers = {
        "interaction": (InteractionModule, dict(
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            node_equivariant_field=AtomicDataDict.NODE_EQ_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            output_mul="hidden",
        )),
        "edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
    }

    return layers