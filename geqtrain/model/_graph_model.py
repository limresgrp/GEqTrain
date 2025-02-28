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
    EmbeddingAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    EmbeddingGraphAttrs,
    ReadoutModule,
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
    '''
    returns dict:
        - keys: layer names
        - values:
            either:
            - obj that inherits from (GraphModuleMixin, torch.nn.Module)
            - tuple of (obj that inherits from (GraphModuleMixin, torch.nn.Module), dict)
                where dict is kwargs for the associated obj constructor
    '''
    logging.info("--- Building Graph Model ---")

    update_config(config)

    layers = {
        # -- Encode -- #
        "node_attrs":         EmbeddingAttrs,
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