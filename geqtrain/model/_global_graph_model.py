import logging
from typing import Optional

from geqtrain.data import AtomicDataDict
from torch.utils.data import ConcatDataset
from geqtrain.model import update_config
from geqtrain.nn.EnsembleLayer import EnsembleAggregator, WeightedEnsembleAggregator
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



def HeadlessGlobalGraphModel(
    config:Config, initialize: bool, dataset: Optional[ConcatDataset] = None,
) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildGlobalGraphModelLayers(config) # moreGNNLayers(config)

    # # todo extract
    # layers.update({
    #     "ensemble_aggregator": (EnsembleAggregator, dict(
    #         field=AtomicDataDict.GRAPH_FEATURES_KEY,
    #         out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
    #         aggregation_method= "max",
    #     )),
    # })

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGlobalGraphModelLayers(config:Config):
    logging.info("--- Building Global Graph Model")

    update_config(config)

    if 'node_attributes' in config:
        node_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            attributes=config.get('node_attributes'),
        ))
    else:
        raise ValueError('Missing node_attributes in yaml')

    layers = {
        # -- Encode -- #
        "node_attrs":         node_embedder,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
        "graph_attrs":        EmbeddingGraphAttrs,
    }

    if 'edge_attributes' in config:
        edge_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            attributes=config.get('edge_attributes'),
        ))
        layers["edge_attrs"] = edge_embedder

    layers.update({
        "update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_ATTRS_KEY,
            out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
        )),
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
        "update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
        )),
        "global_node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
            # residual_field=AtomicDataDict.NODE_ATTRS_KEY,
        )),
    })

    return layers


##################################################################################################

def appendNGNNLayers(config):

    N:int = config.get('gnn_layers', 2)
    modules = {}
    logging.info(f"--- Number of GNN layers {N}")

    # # attention on embeddings
    modules.update({
        "update_emb": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_ATTRS_KEY,
            out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
            num_heads=8, # this number must be a 0 reminder of the sum of catted nn.embedded features (node and edges)
        ))
    })

    for layer_idx in range(N-1):
        layer_name:str = 'local_interaction' if layer_idx == 0 else f"interaction_{layer_idx}"
        modules.update({
            layer_name : (InteractionModule, dict(
                name = layer_name,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_irreps=None,
                output_ls=[0],
            )),
            f"local_pooling_{layer_idx}": (EdgewiseReduce, dict(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                reduce=config.get("edge_reduce", "sum"),
            )),
            f"update_{layer_idx}": (ReadoutModule, dict(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
                out_irreps=None, # outs tensor of same o3.irreps of out_field
                resnet=True,
            )),
        })

    modules.update({
        "last_interaction_layer": (InteractionModule, dict(
            name = "last_interaction_layer",
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
        "update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
        )),
        "global_node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
            # residual_field=AtomicDataDict.NODE_ATTRS_KEY,
        )),
    })
    return modules

def moreGNNLayers(config:Config):
    logging.info("--- Building Global Graph Model")

    update_config(config)

    if 'node_attributes' in config:
        node_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.NODE_ATTRS_KEY,
            attributes=config.get('node_attributes'),
        ))
    else:
        raise ValueError('Missing node_attributes in yaml')

    if 'edge_attributes' in config:
        edge_embedder = (EmbeddingAttrs, dict(
            out_field=AtomicDataDict.EDGE_FEATURES_KEY,
            attributes=config.get('edge_attributes'),
        ))
    else:
        edge_embedder = None
        logging.info("--- Working without edge_attributes")

    if 'graph_attributes' in config:
        graph_embedder = EmbeddingGraphAttrs
    else:
        graph_embedder = None
        logging.info("--- Working without graph_attributes")

    layers = {
        # -- Encode -- #
        "node_attrs":         node_embedder,
        "edge_radial_attrs":  BasisEdgeRadialAttrs,
        "edge_angular_attrs": SphericalHarmonicEdgeAngularAttrs,
    }

    if edge_embedder != None:
        layers.update({"edge_attrs": edge_embedder})

    if graph_embedder != None:
        layers.update({"graph_attrs": graph_embedder})

    layers.update(appendNGNNLayers(config))

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )