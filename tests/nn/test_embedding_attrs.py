import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn import EmbeddingAttrs
from tests.utils.deployability import assert_module_deployable


def _make_edge_index(num_nodes: int, num_edges: int) -> torch.Tensor:
    src = torch.arange(num_edges) % num_nodes
    dst = (src + 1) % num_nodes
    return torch.stack([src, dst], dim=0)


def test_embedding_attrs_projects_node_and_edge_features():
    n_nodes, n_edges = 4, 6
    irreps_in = {
        AtomicDataDict.NODE_INPUT_ATTRS_KEY: "5x0e",
        AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY: "1x1o",
        AtomicDataDict.EDGE_INPUT_ATTRS_KEY: "3x0e",
        AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY: "1x1o",
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: "4x0e",
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o",
    }
    module = EmbeddingAttrs(
        irreps_in=irreps_in,
        node_out_irreps="8x0e",
        node_eq_out_irreps="2x1o",
        edge_out_irreps="6x0e",
        edge_eq_out_irreps="2x1o",
        edge_emb_kwargs={"include_edge_radial": True},
        edge_eq_emb_kwargs={
            "include_edge_spharm": True,
            "include_node_eq_center": True,
            "include_node_eq_neighbor": True,
        },
    )

    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(n_nodes, 3),
        AtomicDataDict.EDGE_INDEX_KEY: _make_edge_index(n_nodes, n_edges),
        AtomicDataDict.NODE_INPUT_ATTRS_KEY: o3.Irreps("5x0e").randn(n_nodes, -1),
        AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY: o3.Irreps("1x1o").randn(n_nodes, -1),
        AtomicDataDict.EDGE_INPUT_ATTRS_KEY: o3.Irreps("3x0e").randn(n_edges, -1),
        AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY: o3.Irreps("1x1o").randn(n_edges, -1),
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: o3.Irreps("4x0e").randn(n_edges, -1),
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: o3.Irreps("1x0e+1x1o").randn(n_edges, -1),
    }
    out = module(data)

    assert out[AtomicDataDict.NODE_ATTRS_KEY].shape == (n_nodes, o3.Irreps("8x0e").dim)
    assert out[AtomicDataDict.NODE_EQ_ATTRS_KEY].shape == (n_nodes, o3.Irreps("2x1o").dim)
    assert out[AtomicDataDict.EDGE_ATTRS_KEY].shape == (n_edges, o3.Irreps("6x0e").dim)
    assert out[AtomicDataDict.EDGE_EQ_ATTRS_KEY].shape == (n_edges, o3.Irreps("2x1o").dim)


def test_embedding_attrs_supports_radial_only_edge_embedding():
    n_nodes, n_edges = 3, 5
    irreps_in = {
        AtomicDataDict.NODE_INPUT_ATTRS_KEY: "4x0e",
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: "3x0e",
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o",
    }
    module = EmbeddingAttrs(
        irreps_in=irreps_in,
        edge_out_irreps="5x0e",
        edge_emb_kwargs={"include_edge_radial": True},
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(n_nodes, 3),
        AtomicDataDict.EDGE_INDEX_KEY: _make_edge_index(n_nodes, n_edges),
        AtomicDataDict.NODE_INPUT_ATTRS_KEY: o3.Irreps("4x0e").randn(n_nodes, -1),
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: o3.Irreps("3x0e").randn(n_edges, -1),
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: o3.Irreps("1x0e+1x1o").randn(n_edges, -1),
    }
    out = module(data)

    assert out[AtomicDataDict.NODE_ATTRS_KEY].shape == (n_nodes, o3.Irreps("4x0e").dim)
    assert out[AtomicDataDict.EDGE_ATTRS_KEY].shape == (n_edges, o3.Irreps("5x0e").dim)
    assert AtomicDataDict.EDGE_EQ_ATTRS_KEY not in out


def test_embedding_attrs_deployable(tmp_path):
    n_nodes, n_edges = 4, 6
    irreps_in = {
        AtomicDataDict.NODE_INPUT_ATTRS_KEY: "5x0e",
        AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY: "1x1o",
        AtomicDataDict.EDGE_INPUT_ATTRS_KEY: "3x0e",
        AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY: "1x1o",
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: "4x0e",
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o",
    }
    module = EmbeddingAttrs(
        irreps_in=irreps_in,
        node_out_irreps="8x0e",
        edge_out_irreps="6x0e",
        edge_emb_kwargs={"include_edge_radial": True},
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(n_nodes, 3),
        AtomicDataDict.EDGE_INDEX_KEY: _make_edge_index(n_nodes, n_edges),
        AtomicDataDict.NODE_INPUT_ATTRS_KEY: o3.Irreps("5x0e").randn(n_nodes, -1),
        AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY: o3.Irreps("1x1o").randn(n_nodes, -1),
        AtomicDataDict.EDGE_INPUT_ATTRS_KEY: o3.Irreps("3x0e").randn(n_edges, -1),
        AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY: o3.Irreps("1x1o").randn(n_edges, -1),
        AtomicDataDict.EDGE_RADIAL_EMB_KEY: o3.Irreps("4x0e").randn(n_edges, -1),
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: o3.Irreps("1x0e+1x1o").randn(n_edges, -1),
    }
    assert_module_deployable(module, (data,), tmp_path=tmp_path)
