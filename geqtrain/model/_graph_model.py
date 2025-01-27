from typing import Optional
import logging

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    NodewiseReduce,
    InteractionModule,
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    EmbeddingGraphAttrs,
    ReadoutModule,
)


def GraphModel(
    config:Config, initialize: bool, dataset: Optional[ConcatDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildHeadlessGraphModelLayers(config)

    layers.update({
        "head": (ReadoutModule, dict(
            field=AtomicDataDict.GRAPH_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_OUTPUT_KEY,
        )),
    })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )


def HeadlessGraphModel(
    config:Config, initialize: bool, dataset: Optional[ConcatDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildHeadlessGraphModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildHeadlessGraphModelLayers(config:Config):
    logging.info("--- Building Graph Model ---")

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
            out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        )),
        "node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_OUTPUT_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
        )),
    })

    return layers