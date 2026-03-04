import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn.goten import GotenEdgeEmbedding
from tests.utils.deployability import assert_module_deployable


def _edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)


def _base_irreps_in(with_edge_input: bool = True):
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: o3.Irreps("3x0e"),
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: o3.Irreps("2x0e"),
        AtomicDataDict.EDGE_INDEX_KEY: None,
    }
    if with_edge_input:
        irreps_in[AtomicDataDict.EDGE_INPUT_ATTRS_KEY] = o3.Irreps("4x0e")
    return irreps_in


def test_goten_edge_embedding_radial_only_ignores_edge_attrs():
    module = GotenEdgeEmbedding(
        node_field=AtomicDataDict.NODE_ATTRS_KEY,
        edge_field=AtomicDataDict.EDGE_INPUT_ATTRS_KEY,
        include_edge_field=False,
        include_edge_radial=True,
        irreps_in=_base_irreps_in(with_edge_input=True),
    )

    edge_index = _edge_index()
    node_attr = torch.randn(3, 3)
    edge_radial = torch.randn(4, 2)
    edge_input_a = torch.zeros(4, 4)
    edge_input_b = torch.randn(4, 4)

    data_a = {
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.NODE_ATTRS_KEY: node_attr,
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: edge_radial,
        AtomicDataDict.EDGE_INPUT_ATTRS_KEY: edge_input_a,
    }
    data_b = dict(data_a)
    data_b[AtomicDataDict.EDGE_INPUT_ATTRS_KEY] = edge_input_b

    out_a = module(data_a)
    out_b = module(data_b)
    assert torch.allclose(out_a, out_b)


def test_goten_edge_embedding_can_use_edge_attrs():
    module = GotenEdgeEmbedding(
        node_field=AtomicDataDict.NODE_ATTRS_KEY,
        edge_field=AtomicDataDict.EDGE_INPUT_ATTRS_KEY,
        include_edge_field=True,
        include_edge_radial=True,
        irreps_in=_base_irreps_in(with_edge_input=True),
    )

    with torch.no_grad():
        module.W_erp.weight.zero_()
        module.W_erp.bias.zero_()
        radial_dim = o3.Irreps("2x0e").dim
        module.W_erp.weight[:, radial_dim:] = 1.0

    edge_index = _edge_index()
    node_attr = torch.ones(3, 3)
    edge_radial = torch.zeros(4, 2)
    edge_input_a = torch.zeros(4, 4)
    edge_input_b = torch.ones(4, 4)

    data_a = {
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.NODE_ATTRS_KEY: node_attr,
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: edge_radial,
        AtomicDataDict.EDGE_INPUT_ATTRS_KEY: edge_input_a,
    }
    data_b = dict(data_a)
    data_b[AtomicDataDict.EDGE_INPUT_ATTRS_KEY] = edge_input_b

    out_a = module(data_a)
    out_b = module(data_b)
    assert out_a.shape == out_b.shape == (4, 3)
    assert not torch.allclose(out_a, out_b)


def test_goten_edge_embedding_include_edge_field_is_optional(tmp_path):
    module = GotenEdgeEmbedding(
        node_field=AtomicDataDict.NODE_ATTRS_KEY,
        edge_field=AtomicDataDict.EDGE_INPUT_ATTRS_KEY,
        include_edge_field=True,
        include_edge_radial=True,
        irreps_in=_base_irreps_in(with_edge_input=False),
    )

    data = {
        AtomicDataDict.EDGE_INDEX_KEY: _edge_index(),
        AtomicDataDict.NODE_ATTRS_KEY: torch.randn(3, 3),
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: torch.randn(4, 2),
    }
    out = module(data)
    assert out.shape == (4, 3)
    assert_module_deployable(module, (data,), tmp_path=tmp_path)
