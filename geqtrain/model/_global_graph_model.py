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
    layers = buildGlobalGraphModelLayers(config)

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

def HeadlessGlobalGraphModel(
    config, initialize: bool, dataset: Optional[ConcatDataset] = None,
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildGlobalGraphModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGlobalGraphModelLayers(config):
    logging.info("--- Building Global Graph Model")

    update_config(config)

    layers = {
        # -- Encode -- #
        "node_attrs":         EmbeddingNodeAttrs,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
        "graph_attrs":        EmbeddingGraphAttrs,
    }

    layers.update({
        "local_interaction": (InteractionModule, dict(
            name = "local_interaction",
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_irreps=None,
            output_ls=[0],
        )),
        "local_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
            reduce=config.get("edge_reduce", "sum"),
        )),
        "update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
        )),
        #TODO: if one wants to play with updated scalars, you can create a module to be added here
        "context_aware_interaction": (InteractionModule, dict(
            name = "context_aware_interaction",
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            output_mul="hidden",
        )),
        "global_edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
        "global_node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
        )),
    })

    return layers