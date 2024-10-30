from typing import Optional, List
import logging

from e3nn import o3
from geqtrain.data import AtomicDataDict, AtomicDataset
from geqtrain.nn import (
    SequentialGraphNetwork,
    EdgewiseReduce,
    InteractionModule,
)
from geqtrain.nn import (
    EmbeddingNodeAttrs,
    SphericalHarmonicEdgeAngularAttrs,
    BasisEdgeRadialAttrs,
    ReadoutModule,
    NodewiseReduce,
)


def ModelScalarGraphLvlOutput(
    config, initialize: bool, dataset = None
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
    }

    layers.update(
        {
            "local_interaction": (
            InteractionModule,
                dict(
                    name="local_interaction",
                    node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
                    edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
                    edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
                    out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_irreps=None, # if None -> hidden molteplicity del 2body and all ls till lmax; if not passed then takes out_irreps from yaml (which eg is 1x1o for scalar output),
                    output_ls=[0], # select/indexes in out_irreps which ls to out (instead of all out out_irreps)
                    # otherwise you can provide:
                    # output_mul="hidden", # output_mul x out_irreps from yaml since out_irreps not provided

                ),
            ),
            "local_pooling": (
                EdgewiseReduce,
                dict(
                    name="local_pooling",
                    field=AtomicDataDict.EDGE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_FEATURES_KEY,
                    reduce=config.get("edge_reduce", "sum"),
                ),
            ),
            "update": (
                # out_irreps options evaluated in the following order:
                # 1) o3.Irreps obj
                # 2) str castable to o3.Irreps obj (eg: 1x0e)
                # 3) an AtomicDataDict.__FIELD_NAME__ taken from GraphModuleMixin.irreps_in dict
                # 4) same irreps of out_field if out_field in GraphModuleMixin.irreps_in dict
                # 5) if none of the above outs irreps of same size of field
                # if out_irreps=None is passed, then option 4 is triggered is valid, else 5)
                # if out_irreps is not provided it takes out_irreps from yaml
                ReadoutModule,
                dict(
                    name="update",
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_irreps=AtomicDataDict.NODE_FEATURES_KEY,
                    resnet=True,
                ),
            ),
            "global_node_pooling": (
                NodewiseReduce,
                dict(
                    field=AtomicDataDict.NODE_FEATURES_KEY,
                    out_field=AtomicDataDict.GRAPH_OUTPUT_KEY,
                ),
            ),
            "head": (
                ReadoutModule,
                dict(
                    field=AtomicDataDict.GRAPH_OUTPUT_KEY,
                    out_field=AtomicDataDict.GRAPH_OUTPUT_KEY,
                    # input_ls=[0], # if output is only scalar, this is required
                ),
            ),
        }
    )

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )