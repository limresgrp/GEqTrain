import logging
from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
    ReadoutModuleWithConditioning,
)
from geqtrain.model._embedding import buildEmbeddingLayers


def HeadlessGlobalNodeModel(config:Config) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildEmbeddingLayers(config)
    layers.update(buildGlobalNodeModelLayers())

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGlobalNodeModelLayers():
    logging.info("--- Building Global Node Model ---")

    layers = {
        "local_interaction": (InteractionModule, dict(
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            node_equivariant_field=AtomicDataDict.NODE_EQ_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_irreps=None,
            output_ls=[0],
        )),
        "local_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
        "update": (ReadoutModuleWithConditioning, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            conditioning_field=AtomicDataDict.NODE_ATTRS_KEY,
            out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
        )),
        "context_aware_interaction": (InteractionModule, dict(
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            node_equivariant_field=AtomicDataDict.NODE_EQ_ATTRS_KEY,
            edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
            edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            output_mul="hidden",
        )),
        "global_edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
    }

    return layers