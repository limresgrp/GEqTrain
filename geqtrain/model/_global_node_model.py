import logging
from typing import Optional

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    EmbeddingGraphAttrs,
    ReadoutModule,
)


def GlobalNodeModel(config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildHeadlessGlobalNodeModelLayers(config)

    layers.update({
        "head": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        )),
    })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def HeadlessGlobalNodeModel(config, initialize: bool, dataset: Optional[ConcatDataset] = None) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildHeadlessGlobalNodeModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildHeadlessGlobalNodeModelLayers(config):
    logging.info("--- Building Global Node Model ---")

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
        )),
        "update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            out_irreps=None,
            resnet=True,
        )),
        "context_aware_interaction": (InteractionModule, dict(
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
    })

    return layers