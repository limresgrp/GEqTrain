import logging
from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config
from geqtrain.nn import (
    EmbeddingInputAttrs,
    EmbeddingAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
)


def buildEmbeddingLayers(config: Config, model = None):
    logging.info("--- Building Embeddings ---")

    from geqtrain.model.init_utils import update_config
    update_config(config)

    layers = {}
    if model is not None:
        layers.update({
            "wrapped_model": model,
        })
    
    layers.update({
        "node_input_attrs": (EmbeddingInputAttrs, dict(
            out_field     = AtomicDataDict.NODE_INPUT_ATTRS_KEY,
            eq_out_field  = AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY,
            attributes    = config.get('node_attributes'),
            eq_attributes = config.get('eq_node_attributes'),
        )),
    })

    if 'graph_attributes' in config or 'eq_graph_attributes' in config:
        layers["graph_input_attrs"] = (EmbeddingInputAttrs, dict(
            out_field     = AtomicDataDict.GRAPH_ATTRS_KEY,
            eq_out_field  = AtomicDataDict.GRAPH_EQ_ATTRS_KEY,
            attributes    = config.get('graph_attributes'),
            eq_attributes = config.get('eq_graph_attributes'),
        ))

    if 'edge_attributes' in config or 'eq_edge_attributes' in config:
        layers["edge_input_attrs"] = (EmbeddingInputAttrs, dict(
            out_field     = AtomicDataDict.EDGE_INPUT_ATTRS_KEY,
            eq_out_field  = AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY,
            attributes    = config.get('edge_attributes'),
            eq_attributes = config.get('eq_edge_attributes'),
        ))

    layers.update({
        "edge_radial_attrs" : BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
        "attrs"             : EmbeddingAttrs,
    })

    return layers
