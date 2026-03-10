from typing import Optional

import torch
from e3nn import o3

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.nn.goten import GotenInteractionModule
from tests.utils.deployability import assert_module_deployable
from tests.utils.equivariance import assert_AtomicData_equivariant


def _make_dummy_input(
    device: str = "cpu",
    *,
    graph_conditioning_dim: int = 0,
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
    if graph_conditioning_dim > 0:
        data_dict[AtomicDataDict.GRAPH_ATTRS_KEY] = torch.randn(1, graph_conditioning_dim, device=device)
    return data_dict


def _make_module(
    device: str = "cpu",
    *,
    scale_edge: bool = True,
    conditioning_fields: Optional[list] = None,
    graph_conditioning_dim: int = 0,
):
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: "8x0e",
        AtomicDataDict.EDGE_ATTRS_KEY: "8x0e",
        AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o+1x2e",
    }
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


def test_goten_interaction_module_equivariant():
    module = _make_module(device="cpu", scale_edge=True)
    data_in = _make_dummy_input(device="cpu")
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
