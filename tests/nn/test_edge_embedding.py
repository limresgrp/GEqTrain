import torch
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


def test_base_edge_embedding_can_include_radial_and_project():
    edge_irreps = o3.Irreps("2x0e")
    radial_irreps = o3.Irreps("3x0e")
    out_irreps = o3.Irreps("4x0e")
    module = BaseEdgeEmbedding(
        edge_field=AtomicDataDict.EDGE_ATTRS_KEY,
        include_edge_radial=True,
        out_irreps=str(out_irreps),
        irreps_in={
            AtomicDataDict.EDGE_ATTRS_KEY: edge_irreps,
            AtomicDataDict.EDGE_RADIAL_EMB_KEY: radial_irreps,
        },
    )
    data = {
        AtomicDataDict.EDGE_ATTRS_KEY: edge_irreps.randn(7, -1),
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: radial_irreps.randn(7, -1),
    }
    out = module(data)
    assert out is not None
    assert out.shape == (7, out_irreps.dim)


def test_base_edge_eq_embedding_with_node_context_projection():
    node_eq_irreps = o3.Irreps("1x1o")
    edge_eq_irreps = o3.Irreps("1x1o")
    edge_spharm_irreps = o3.Irreps("1x0e+1x1o")
    out_irreps = o3.Irreps("2x1o")
    module = BaseEdgeEqEmbedding(
        node_eq_field=AtomicDataDict.NODE_EQ_ATTRS_KEY,
        edge_eq_field=AtomicDataDict.EDGE_EQ_ATTRS_KEY,
        include_edge_spharm=True,
        include_node_eq_center=True,
        include_node_eq_neighbor=True,
        out_irreps=str(out_irreps),
        irreps_in={
            AtomicDataDict.NODE_EQ_ATTRS_KEY: node_eq_irreps,
            AtomicDataDict.EDGE_EQ_ATTRS_KEY: edge_eq_irreps,
            AtomicDataDict.EDGE_SPHARMS_EMB_KEY: edge_spharm_irreps,
            AtomicDataDict.EDGE_INDEX_KEY: None,
        },
    )
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=torch.long)
    data = {
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.NODE_EQ_ATTRS_KEY: node_eq_irreps.randn(3, -1),
        AtomicDataDict.EDGE_EQ_ATTRS_KEY: edge_eq_irreps.randn(4, -1),
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: edge_spharm_irreps.randn(4, -1),
    }
    out = module(data)
    assert out is not None
    assert out.shape == (4, out_irreps.dim)
