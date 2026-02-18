import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, RecyclingModule
from tests.utils.deployability import assert_module_deployable


class _ContractiveBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        state_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        out_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
        scale: float = 0.2,
        bias: float = 0.0,
        irreps_in=None,
    ):
        super().__init__()
        self.state_field = state_field
        self.out_field = out_field
        self.scale = float(scale)
        self.bias = float(bias)
        self.register_buffer("call_count", torch.zeros((), dtype=torch.long))
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[state_field],
            irreps_out={out_field: irreps_in[state_field]},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        self.call_count += 1
        data[self.out_field] = data[self.state_field] * self.scale + self.bias
        return data


def _make_data(n_nodes: int = 5, n_edges: int = 8, feat_dim: int = 4):
    return {
        AtomicDataDict.POSITIONS_KEY: torch.randn(n_nodes, 3),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.randn(n_edges, feat_dim),
    }


def test_recycling_module_converges_before_max_steps():
    irreps = {AtomicDataDict.EDGE_FEATURES_KEY: o3.Irreps("4x0e")}
    block = _ContractiveBlock(irreps_in=irreps, scale=0.2)
    module = RecyclingModule(
        block=block,
        state_field=AtomicDataDict.EDGE_FEATURES_KEY,
        block_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        max_steps=25,
        tol=1e-4,
        alpha=1.0,
        detach_between_steps=True,
        irreps_in=irreps,
    )
    data = _make_data()
    init_norm = torch.sqrt(torch.mean(data[AtomicDataDict.EDGE_FEATURES_KEY].square()))
    out = module(data)

    assert int(block.call_count.item()) < 25
    final_norm = torch.sqrt(torch.mean(out[AtomicDataDict.EDGE_FEATURES_KEY].square()))
    assert float(final_norm) < float(init_norm)


def test_recycling_module_wires_custom_output_field():
    irreps = {AtomicDataDict.EDGE_FEATURES_KEY: o3.Irreps("4x0e")}
    proposal_field = "proposal_edge_features"
    out_field = "recycled_edge_features"
    block = _ContractiveBlock(
        irreps_in=irreps,
        state_field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=proposal_field,
        scale=0.5,
    )
    module = RecyclingModule(
        block=block,
        state_field=AtomicDataDict.EDGE_FEATURES_KEY,
        block_out_field=proposal_field,
        out_field=out_field,
        max_steps=4,
        alpha=1.0,
        irreps_in=irreps,
    )
    out = module(_make_data())

    assert proposal_field in out
    assert out_field in out
    torch.testing.assert_close(out[proposal_field], out[out_field])


def test_recycling_module_deployable(tmp_path):
    irreps = {AtomicDataDict.EDGE_FEATURES_KEY: o3.Irreps("4x0e")}
    block = _ContractiveBlock(irreps_in=irreps, scale=0.5)
    module = RecyclingModule(
        block=block,
        state_field=AtomicDataDict.EDGE_FEATURES_KEY,
        block_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        max_steps=3,
        alpha=0.7,
        irreps_in=irreps,
    )
    data = _make_data()
    assert_module_deployable(module, (data,), tmp_path=tmp_path)
