from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn._edge import BaseEdgeEmbedding, BaseEdgeEqEmbedding
from tests.utils.deployability import assert_module_deployable


def test_base_edge_embedding_deployable(tmp_path):
    edge_irreps = o3.Irreps("4x0e")
    module = BaseEdgeEmbedding(
        edge_field=AtomicDataDict.EDGE_ATTRS_KEY,
        irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: edge_irreps},
    )
    data = {AtomicDataDict.EDGE_ATTRS_KEY: edge_irreps.randn(6, -1)}

    out = module(data)
    assert out is not None
    assert out.shape == (6, edge_irreps.dim)
    assert_module_deployable(module, (data,), tmp_path=tmp_path)


def test_base_edge_eq_embedding_deployable(tmp_path):
    edge_eq_irreps = o3.Irreps("1x0e+1x1o")
    module = BaseEdgeEqEmbedding(
        edge_eq_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
        irreps_in={AtomicDataDict.EDGE_EQ_ATTRS_KEY: edge_eq_irreps},
    )
    data = {AtomicDataDict.EDGE_EQ_ATTRS_KEY: edge_eq_irreps.randn(5, -1)}

    out = module(data)
    assert out is not None
    assert out.shape == (5, edge_eq_irreps.dim)
    assert_module_deployable(module, (data,), tmp_path=tmp_path)
