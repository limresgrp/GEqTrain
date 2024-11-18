import logging
from typing import Optional

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
    EmbeddingGraphAttrs,
    ReadoutModule,
)


def GlobalGraphModel(
    config, initialize: bool, dataset: Optional[ConcatDataset] = None,
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
        # -- Optional -- "graph_attrs":        EmbeddingGraphAttrs,
    }

    layers.update(
        {
            "local_interaction": (
            InteractionModule,
                dict(
                    node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                    edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                    edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
                    out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_irreps=None,
                    output_ls=[0], # todo: only scalars supported in *first* interaction module atm
                ),
            ),
            "local_pooling": (
                EdgewiseReduce,
                dict(
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_FEATURES_KEY,
                    reduce=config.get("edge_reduce", "sum"),
                ),
            ),
            "update": (
                ReadoutModule,
                dict(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_ATTRS_KEY,
                    out_irreps=None,
                    resnet=True,
                ),
            ),
            "context_aware_interaction": (
            InteractionModule,
                dict(
                    node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                    edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                    edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
                    out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                    output_mul="hidden",
                ),
            ),
            "global_edge_pooling": (
                EdgewiseReduce,
                dict(
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_OUTPUT_KEY,
                    reduce=config.get("edge_reduce", "sum"),
                ),
            ),
            "global_node_pooling": (
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