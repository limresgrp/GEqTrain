from typing import Optional
import logging

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    NodewiseReduce,
    InteractionModule,
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    ReadoutModule,
)


def GraphModel(
    config, initialize: bool, dataset: Optional[ConcatDataset] = None
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    logging.debug("Building model")

    update_config(config)

    layers = {
        # -- Encode --
        "node_attrs":         EmbeddingNodeAttrs,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
    }

    layers.update(
        {
            "interaction": (
            InteractionModule,
                dict(
                    node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                    edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                    edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
                    out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                    output_mul="hidden",
                ),
            ),
            "edge_pooling": (
                EdgewiseReduce,
                dict(
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_OUTPUT_KEY,
                    reduce=config.get("edge_reduce", "sum"),
                ),
            ),
            "node_pooling": (
                NodewiseReduce,
                dict(
                    field=AtomicDataDict.NODE_OUTPUT_KEY,
                    out_field=AtomicDataDict.GRAPH_OUTPUT_KEY,
                ),
            ),
            "head": (
                ReadoutModule,
                dict(
                    field=AtomicDataDict.GRAPH_OUTPUT_KEY,
                    out_field=AtomicDataDict.GRAPH_OUTPUT_KEY,
                ),
            ),
        }
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )