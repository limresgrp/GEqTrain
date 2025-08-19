import logging
from typing import Optional
from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    NodewiseReduce,
    InteractionModule,
)
from geqtrain.model._embedding import buildEmbeddingLayers

def GraphModel(config:Config, model: Optional[SequentialGraphNetwork]) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildEmbeddingLayers(config, model)
    layers.update(buildGraphModelLayers())

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGraphModelLayers():
    logging.info("--- Building Graph Model ---")

    layers = {
        "interaction": (InteractionModule, dict(
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_RADIAL_EMB_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            output_mul="hidden",
        )),
        "edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
        "node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
        )),
    }

    return layers