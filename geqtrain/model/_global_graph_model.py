import logging

from e3nn import o3
from geqtrain.data import AtomicDataDict
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
    config,
    dataset,
    initialize: bool,
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    logging.debug("Building model")

    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config.get("parity", "o3_full")
        assert parity_setting in ("o3_full", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        # check consistency
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        config["irreps_edge_sh"] = irreps_edge_sh

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
                    output_hidden_irreps=True,
                    output_hidden_ls=[0],
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
                    output_hidden_irreps=True,
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