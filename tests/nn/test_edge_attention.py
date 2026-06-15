import torch

from geqtrain.data import AtomicDataDict
from geqtrain.nn import EdgewiseReduce, InteractionModule


def test_edgewise_reduce_attention_does_not_use_node_attrs():
    torch.manual_seed(0)
    irreps_in = {"edge_features": "2x0e+2x1o"}
    module = EdgewiseReduce(
        field="edge_features",
        out_field="node_features",
        use_attention=True,
        attention_head_dim=4,
        irreps_in=irreps_in,
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.zeros(3, 3),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 0, 1, 1], [1, 2, 0, 2]],
            dtype=torch.long,
        ),
        "edge_features": torch.randn(4, 8),
        AtomicDataDict.NODE_ATTRS_KEY: torch.randn(3, 5),
    }

    out_1 = module(dict(data))["node_features"]
    data_changed = dict(data)
    data_changed[AtomicDataDict.NODE_ATTRS_KEY] = torch.randn(3, 5) * 100.0
    out_2 = module(data_changed)["node_features"]

    assert torch.allclose(out_1, out_2)


def test_interaction_edge_self_attention_uses_latent_query_projection():
    module = InteractionModule(
        num_layers=1,
        latent_dim=8,
        eq_latent_multiplicity=2,
        use_attention=True,
        attention_head_dim=4,
        irreps_in={
            AtomicDataDict.NODE_ATTRS_KEY: "4x0e",
            AtomicDataDict.EDGE_RADIAL_EMB_KEY: "3x0e",
            AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o",
        },
    )

    layer = module.interaction_layers[0]
    assert hasattr(layer, "latent_to_query")
    assert layer.latent_to_query is not None
    assert layer.node_state_to_query is None
    assert not hasattr(layer, "node_attr_to_query")


def test_interaction_node_feature_query_attention_uses_node_state_projection():
    module = InteractionModule(
        num_layers=1,
        latent_dim=8,
        eq_latent_multiplicity=2,
        use_attention=True,
        attention_mode="node_feature_query",
        attention_head_dim=4,
        irreps_in={
            AtomicDataDict.NODE_ATTRS_KEY: "4x0e",
            AtomicDataDict.EDGE_RADIAL_EMB_KEY: "3x0e",
            AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o",
        },
    )

    layer = module.interaction_layers[0]
    assert layer.latent_to_query is None
    assert layer.node_state_to_query is not None
    assert str(module.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]) == "8x0e"
