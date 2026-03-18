from typing import Optional

import torch
from e3nn import o3

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.nn.goten import GotenInteractionModule
from tests.utils.deployability import assert_module_deployable
from tests.utils.equivariance import assert_AtomicData_equivariant


def _clone_data_dict(data_dict):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in data_dict.items()
    }


def _make_dummy_input(
    device: str = "cpu",
    *,
    graph_conditioning_dim: int = 0,
    node_eq_irreps: Optional[str] = None,
):
    num_nodes = 5
    pos = torch.randn(num_nodes, 3, device=device)
    data = AtomicData.from_points(pos=pos, r_max=5.0)
    data_dict = AtomicData.to_AtomicDataDict(data)
    num_edges = data_dict[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

    h_dim = 8
    sph_irreps = o3.Irreps("1x0e+1x1o+1x2e")
    data_dict[AtomicDataDict.NODE_ATTRS_KEY] = torch.randn(num_nodes, h_dim, device=device)
    data_dict[AtomicDataDict.EDGE_ATTRS_KEY] = torch.randn(num_edges, h_dim, device=device)
    data_dict[AtomicDataDict.EDGE_SPHARMS_EMB_KEY] = sph_irreps.randn(num_edges, -1, device=device)
    if node_eq_irreps is not None:
        data_dict[AtomicDataDict.NODE_EQ_ATTRS_KEY] = o3.Irreps(node_eq_irreps).randn(
            num_nodes, -1, device=device
        )
    if graph_conditioning_dim > 0:
        data_dict[AtomicDataDict.GRAPH_ATTRS_KEY] = torch.randn(1, graph_conditioning_dim, device=device)
    return data_dict


def _make_module(
    device: str = "cpu",
    *,
    scale_edge: bool = True,
    conditioning_fields: Optional[list] = None,
    graph_conditioning_dim: int = 0,
    node_eq_irreps: Optional[str] = None,
):
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: "8x0e",
        AtomicDataDict.EDGE_ATTRS_KEY: "8x0e",
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o+1x2e",
    }
    if node_eq_irreps is not None:
        irreps_in[AtomicDataDict.NODE_EQ_ATTRS_KEY] = str(o3.Irreps(node_eq_irreps))
    if graph_conditioning_dim > 0:
        irreps_in[AtomicDataDict.GRAPH_ATTRS_KEY] = f"{graph_conditioning_dim}x0e"
    if conditioning_fields is None:
        conditioning_fields = []
    return GotenInteractionModule(
        num_layers=2,
        r_max=5.0,
        num_heads=4,
        attn_dropout=0.0,
        layer_norm=True,
        steerable_norm=False,
        edge_updates=True,
        scale_edge=scale_edge,
        sep_htr=True,
        sep_dir=False,
        sep_tensor=False,
        out_field_node=AtomicDataDict.NODE_FEATURES_KEY,
        out_field_node_eq=None,
        out_field_edge=AtomicDataDict.EDGE_FEATURES_KEY,
        output_ls=[0],
        conditioning_fields=conditioning_fields,
        irreps_in=irreps_in,
    ).to(device)


def test_goten_interaction_module_deployable(tmp_path):
    module = _make_module(device="cpu", scale_edge=True)
    data_in = _make_dummy_input(device="cpu")
    assert_module_deployable(module, (data_in,), tmp_path=tmp_path, atol=1e-6)


def test_goten_interaction_module_uses_node_eq_attrs_for_initialization():
    node_eq_irreps = "8x1o+8x2e"
    module = _make_module(device="cpu", scale_edge=True, node_eq_irreps=node_eq_irreps).eval()
    data_in = _make_dummy_input(device="cpu", node_eq_irreps=node_eq_irreps)
    data_zero = _clone_data_dict(data_in)
    data_zero[AtomicDataDict.NODE_EQ_ATTRS_KEY] = torch.zeros_like(data_in[AtomicDataDict.NODE_EQ_ATTRS_KEY])

    with torch.no_grad():
        out_with_eq = module(_clone_data_dict(data_in))
        out_without_eq = module(data_zero)

    assert not torch.allclose(
        out_with_eq[AtomicDataDict.NODE_FEATURES_KEY],
        out_without_eq[AtomicDataDict.NODE_FEATURES_KEY],
    )


def test_goten_interaction_module_equivariant():
    module = _make_module(device="cpu", scale_edge=True)
    data_in = _make_dummy_input(device="cpu")
    assert_AtomicData_equivariant(func=module, data_in=data_in)


def test_goten_interaction_module_node_eq_init_deployable(tmp_path):
    node_eq_irreps = "4x1o"
    module = _make_module(device="cpu", scale_edge=True, node_eq_irreps=node_eq_irreps)
    data_in = _make_dummy_input(device="cpu", node_eq_irreps=node_eq_irreps)
    assert_module_deployable(module, (data_in,), tmp_path=tmp_path, atol=1e-6)


def test_goten_interaction_module_node_eq_init_equivariant():
    node_eq_irreps = "4x1o"
    module = _make_module(device="cpu", scale_edge=True, node_eq_irreps=node_eq_irreps)
    data_in = _make_dummy_input(device="cpu", node_eq_irreps=node_eq_irreps)
    assert_AtomicData_equivariant(func=module, data_in=data_in)


def test_goten_interaction_module_conditioning_deployable(tmp_path):
    conditioning_fields = [
        AtomicDataDict.GRAPH_ATTRS_KEY,
        AtomicDataDict.NODE_ATTRS_KEY,
        AtomicDataDict.EDGE_ATTRS_KEY,
    ]
    module = _make_module(
        device="cpu",
        scale_edge=True,
        conditioning_fields=conditioning_fields,
        graph_conditioning_dim=4,
    )
    data_in = _make_dummy_input(device="cpu", graph_conditioning_dim=4)
    assert_module_deployable(module, (data_in,), tmp_path=tmp_path, atol=1e-6)


def test_goten_interaction_module_conditioning_equivariant():
    conditioning_fields = [
        AtomicDataDict.GRAPH_ATTRS_KEY,
        AtomicDataDict.NODE_ATTRS_KEY,
        AtomicDataDict.EDGE_ATTRS_KEY,
    ]
    module = _make_module(
        device="cpu",
        scale_edge=True,
        conditioning_fields=conditioning_fields,
        graph_conditioning_dim=4,
    )
    data_in = _make_dummy_input(device="cpu", graph_conditioning_dim=4)
    assert_AtomicData_equivariant(func=module, data_in=data_in)
