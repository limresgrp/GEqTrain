import logging
from geqtrain.data import AtomicDataDict
from geqtrain.model import update_config
from geqtrain.utils import Config

from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
    EmbeddingAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    ReadoutModuleWithConditioning,
)



def HeadlessGlobalNodeModel(config:Config) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildHeadlessGlobalNodeModelLayers(config)

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildHeadlessGlobalNodeModelLayers(config:Config):
    logging.info("--- Building Global Node Model ---")

    update_config(config)

    if 'node_attributes' not in config:
        raise ValueError('Missing node_attributes in yaml')
    
    layers = {
        "node_attrs": (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            attributes=config.get('node_attributes'),
        )),
    }
    
    if 'edge_attributes' in config:
        edge_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            attributes=config.get('edge_attributes'),
        ))
        layers["edge_attrs"] = edge_embedder

    layers.update({
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
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
        "update": (ReadoutModuleWithConditioning, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            conditioning_field=AtomicDataDict.NODE_ATTRS_KEY,
            out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
        )),
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
    })

    return layers