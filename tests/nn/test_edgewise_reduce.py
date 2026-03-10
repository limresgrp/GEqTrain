import pytest
import torch
from e3nn import o3

from geqtrain.data import AtomicData, AtomicDataDict, _NODE_FIELDS, _EDGE_FIELDS
from geqtrain.nn import EdgewiseReduce
from geqtrain.utils.pytorch_scatter import scatter_sum
from tests.utils.deployability import assert_module_deployable
from tests.utils.equivariance import assert_AtomicData_equivariant


def _create_dummy_data(irreps_in, num_nodes=6, device="cpu"):
    pos = torch.randn(num_nodes, 3, device=device)
    data = {
        AtomicDataDict.BATCH_KEY: torch.zeros(num_nodes, dtype=torch.long, device=device),
    }
    data = AtomicData.from_points(pos, r_max=5.0, **data)
    num_edges = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

    for key, irreps in irreps_in.items():
        if key in data:
            continue
        if key in _NODE_FIELDS:
            num_items = num_nodes
        elif key in _EDGE_FIELDS:
            num_items = num_edges
        else:
            num_items = 1
        if irreps is not None:
            data[key] = o3.Irreps(irreps).randn(num_items, -1, device=device)

    return AtomicData.to_AtomicDataDict(data.to(device)), num_edges


def _count_scalar_dims(irreps):
    return sum(mul * ir.dim for mul, ir in o3.Irreps(irreps) if ir.l == 0 and ir.p == 1)


def _scalar_slices(irreps):
    slices = []
    offset = 0
    for mul, ir in o3.Irreps(irreps):
        block_dim = mul * ir.dim
        if ir.l == 0 and ir.p == 1:
            slices.append((offset, offset + block_dim))
        offset += block_dim
    return slices


def test_edgewise_reduce_no_attention_matches_scatter_sum():
    torch.manual_seed(0)
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: "2x0e",
        AtomicDataDict.EDGE_FEATURES_KEY: "1x0e+2x1o",
    }
    data, _ = _create_dummy_data(irreps_in, num_nodes=5)
    module = EdgewiseReduce(
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.NODE_FEATURES_KEY,
        use_attention=False,
        avg_num_neighbors=4.0,
        irreps_in=irreps_in,
    )

    edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    edge_feat = data[AtomicDataDict.EDGE_FEATURES_KEY]
    num_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    expected = scatter_sum(edge_feat, edge_center, dim=0, dim_size=num_nodes)
    expected = expected * module.env_sum_normalization

    out = module(data)
    torch.testing.assert_close(out[module.out_field], expected)


def test_edgewise_reduce_attention_mixed_irreps_uses_scalar_attrs():
    torch.manual_seed(0)
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: "2x0e+1x0o",
        AtomicDataDict.EDGE_FEATURES_KEY: "1x0e+2x0o+3x1o+1x2e",
    }
    data, _ = _create_dummy_data(irreps_in, num_nodes=6)
    module = EdgewiseReduce(
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.NODE_FEATURES_KEY,
        use_attention=True,
        attention_head_dim=4,
        readout_latent_kwargs={
            "mlp_latent_dimensions": [],
            "mlp_nonlinearity": None,
            "use_layer_norm": False,
            "zero_init_last_layer_weights": False,
        },
        irreps_in=irreps_in,
    )

    edge_irreps = o3.Irreps(irreps_in[AtomicDataDict.EDGE_FEATURES_KEY])
    node_irreps = o3.Irreps(irreps_in[AtomicDataDict.NODE_ATTRS_KEY])
    assert module.n_scalars == _count_scalar_dims(edge_irreps)
    assert module.n_node_scalars == _count_scalar_dims(node_irreps)
    assert module.attention_num_heads == max(mul for mul, _ in edge_irreps)

    out = module(data)
    num_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    assert out[module.out_field].shape == (num_nodes, edge_irreps.dim)

    data_mod = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in data.items()
    }
    node_scalar_slices = _scalar_slices(node_irreps)
    assert len(node_scalar_slices) >= 1
    start, end = node_scalar_slices[0]
    data_mod[AtomicDataDict.NODE_ATTRS_KEY][:, start:end] += 5.0
    out_mod = module(data_mod)

    delta = (out_mod[module.out_field] - out[module.out_field]).abs().max().item()
    assert delta > 1e-4, "Attention output did not respond to scalar node attrs"


@pytest.mark.parametrize("use_attention", [False, True])
def test_edgewise_reduce_equivariant(use_attention):
    torch.manual_seed(0)
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: "2x0e+1x0o",
        AtomicDataDict.EDGE_FEATURES_KEY: "1x0e+2x0o+1x1o",
    }
    module = EdgewiseReduce(
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.NODE_FEATURES_KEY,
        use_attention=use_attention,
        attention_head_dim=4,
        irreps_in=irreps_in,
    )
    data, _ = _create_dummy_data(irreps_in, num_nodes=5)
    assert_AtomicData_equivariant(func=module, data_in=data)


@pytest.mark.parametrize("use_attention", [False, True])
def test_edgewise_reduce_deployable(use_attention, tmp_path):
    torch.manual_seed(0)
    irreps_in = {
        AtomicDataDict.NODE_ATTRS_KEY: "2x0e+1x0o",
        AtomicDataDict.EDGE_FEATURES_KEY: "1x0e+2x1o",
    }
    module = EdgewiseReduce(
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.NODE_FEATURES_KEY,
        use_attention=use_attention,
        attention_head_dim=4,
        irreps_in=irreps_in,
    ).to("cpu")
    data, _ = _create_dummy_data(irreps_in, num_nodes=4, device="cpu")
    assert_module_deployable(module, (data,), tmp_path=tmp_path)
