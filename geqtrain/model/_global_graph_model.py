import logging
from typing import Optional
from geqtrain.data import AtomicDataDict
from geqtrain.utils import Config
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    NodewiseReduce,
    InteractionModule,
    ReadoutModule,
)
from geqtrain.model._embedding import buildEmbeddingLayers



def GlobalGraphModel(config:Config, model: Optional[SequentialGraphNetwork]) -> SequentialGraphNetwork:
    """Base model architecture.

    """
    layers = buildEmbeddingLayers(config, model)
    layers.update(appendNGNNLayers(config))

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )

def buildGlobalGraphModelLayers():
    logging.info("--- Building Global Graph Model")

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
        "local_edge_pooling": (EdgewiseReduce, dict(
            field=AtomicDataDict.EDGE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY,
        )),
        "local_update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
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
        "global_update": (ReadoutModule, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.NODE_FEATURES_KEY, # scalars only
            out_irreps=None, # outs tensor of same o3.irreps of out_field
            resnet=True,
        )),
        "global_node_pooling": (NodewiseReduce, dict(
            field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.GRAPH_FEATURES_KEY,
        )),
    }

    return layers


##################################################################################################

def appendNGNNLayers(config):

    N:int = config.get('gnn_layers', 2)
    modules = {}
    logging.info(f"--- Number of GNN layers {N}")

    # # # attention on embeddings
    # modules.update({
    #     "update_emb": (ReadoutModule, dict(
    #         field=AtomicDataDict.NODE_ATTRS_KEY,
    #         out_field=AtomicDataDict.NODE_ATTRS_KEY, # scalars only
    #         out_irreps=None, # outs tensor of same o3.irreps of out_field
    #         resnet=True,
    #         num_heads=8, # this number must be a 0 reminder of the sum of catted nn.embedded features (node and edges)
    #     ))
    # })

    for layer_idx in range(N-1):
        layer_name:str = 'local_interaction' if layer_idx == 0 else f"interaction_{layer_idx}"
        modules.update({
            layer_name : (InteractionModule, dict(
                name = layer_name,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                node_equivariant_field=AtomicDataDict.NODE_EQ_ATTRS_KEY,
                edge_invariant_field=AtomicDataDict.EDGE_ATTRS_KEY,
                edge_equivariant_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
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