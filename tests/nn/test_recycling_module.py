import torch
from e3nn import o3

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, RecyclingModule
from geqtrain.nn._graph_mixin import SequentialGraphNetwork
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


class _FeedbackBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        state_field: str = "state",
        cond_field: str = "cond",
        pred_field: str = "pred",
        irreps_in=None,
    ):
        super().__init__()
        self.state_field = state_field
        self.cond_field = cond_field
        self.pred_field = pred_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[state_field, cond_field],
            irreps_out={
                state_field: irreps_in[state_field],
                pred_field: irreps_in[cond_field],
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.state_field] = data[self.state_field] * 0.5
        data[self.pred_field] = data[self.cond_field] + 1.0
        return data


class _SlicedFeedbackBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        state_field: str = "state",
        pred_field: str = "pred",
        irreps_in=None,
    ):
        super().__init__()
        self.state_field = state_field
        self.pred_field = pred_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[state_field],
            irreps_out={
                state_field: irreps_in[state_field],
                pred_field: irreps_in[pred_field],
            },
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.state_field] = data[self.state_field] * 0.0
        n = data[self.state_field].shape[0]
        base = torch.arange(4, device=data[self.state_field].device, dtype=data[self.state_field].dtype)
        data[self.pred_field] = base.view(1, 4).repeat(n, 1)
        return data


class _ReadoutStyleBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        state_field: str = "state",
        scalar_field: str = "state_scalar",
        irreps_in=None,
    ):
        super().__init__()
        self.state_field = state_field
        self.scalar_field = scalar_field
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[state_field],
            irreps_out={
                state_field: irreps_in[state_field],
                scalar_field: o3.Irreps("1x0e"),
            },
        )
        # Mimic ReadoutModule metadata used by RecyclingModule auto-sync.
        self._has_out_field = True
        self._has_scalar_out_field = True
        self._out_field = state_field
        self._scalar_out_field = scalar_field
        self.n_scalars_out = 1

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        proposal = data[self.state_field] * 0.0 + 2.0
        data[self.state_field] = proposal
        # Intentionally set inconsistent scalar output; recycler should overwrite it.
        data[self.scalar_field] = proposal[..., :1] * 0.0
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


def test_recycling_module_feedback_respects_leak_mask():
    irreps = {
        "state": o3.Irreps("1x0e"),
        "cond": o3.Irreps("1x0e"),
    }
    block = _FeedbackBlock(irreps_in=irreps)
    module = RecyclingModule(
        block=block,
        state_field="state",
        block_out_field="state",
        max_steps=2,
        alpha=1.0,
        feedback_from_fields=["pred"],
        feedback_to_fields=["cond"],
        feedback_apply_mask=True,
        feedback_mask_suffix="__mask__",
        irreps_in=irreps,
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(2, 3),
        "state": torch.ones(2, 1),
        "cond": torch.tensor([[10.0], [0.0]]),
        "cond__mask__": torch.tensor([[True], [False]]),
    }
    out = module(data)
    expected = torch.tensor([[10.0], [2.0]])
    torch.testing.assert_close(out["cond"], expected)


def test_recycling_module_feedback_supports_component_slices():
    irreps = {
        "state": o3.Irreps("1x0e"),
        "pred": o3.Irreps("4x0e"),
        "cond_scalar": o3.Irreps("1x0e"),
        "cond_equiv": o3.Irreps("3x0e"),
    }
    block = _SlicedFeedbackBlock(irreps_in=irreps)
    module = RecyclingModule(
        block=block,
        state_field="state",
        block_out_field="state",
        max_steps=1,
        alpha=1.0,
        feedback_from_fields=["pred", "pred"],
        feedback_to_fields=["cond_scalar", "cond_equiv"],
        feedback_slice_starts=[0, 1],
        feedback_slice_ends=[1, 4],
        feedback_apply_mask=False,
        irreps_in=irreps,
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(3, 3),
        "state": torch.randn(3, 1),
        "cond_scalar": torch.zeros(3, 1),
        "cond_equiv": torch.zeros(3, 3),
    }
    out = module(data)
    torch.testing.assert_close(out["cond_scalar"], torch.zeros(3, 1))
    torch.testing.assert_close(
        out["cond_equiv"],
        torch.tensor([[1.0, 2.0, 3.0]]).repeat(3, 1),
    )


def test_recycling_module_instantiates_raw_sequential_block_config():
    irreps = {AtomicDataDict.EDGE_FEATURES_KEY: o3.Irreps("4x0e")}
    block_cfg = {
        "_target_": "geqtrain.nn.SequentialGraphNetwork",
        "modules": {
            "contractive": {
                "_target_": "tests.nn.test_recycling_module._ContractiveBlock",
                "state_field": AtomicDataDict.EDGE_FEATURES_KEY,
                "out_field": AtomicDataDict.EDGE_FEATURES_KEY,
                "scale": 0.5,
            }
        },
    }
    module = RecyclingModule(
        block=block_cfg,
        state_field=AtomicDataDict.EDGE_FEATURES_KEY,
        block_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        max_steps=2,
        alpha=1.0,
        irreps_in=irreps,
    )
    out = module(_make_data(feat_dim=4))
    assert AtomicDataDict.EDGE_FEATURES_KEY in out


def test_recycling_module_instantiates_resolved_sequential_block_target():
    irreps = {AtomicDataDict.EDGE_FEATURES_KEY: o3.Irreps("4x0e")}
    block_cfg = {
        "_target_": SequentialGraphNetwork,
        "modules": {
            "contractive": {
                "_target_": "tests.nn.test_recycling_module._ContractiveBlock",
                "state_field": AtomicDataDict.EDGE_FEATURES_KEY,
                "out_field": AtomicDataDict.EDGE_FEATURES_KEY,
                "scale": 0.5,
            }
        },
    }
    module = RecyclingModule(
        block=block_cfg,
        state_field=AtomicDataDict.EDGE_FEATURES_KEY,
        block_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        max_steps=2,
        alpha=1.0,
        irreps_in=irreps,
    )
    out = module(_make_data(feat_dim=4))
    assert AtomicDataDict.EDGE_FEATURES_KEY in out


def test_recycling_module_predict_delta_updates_state_incrementally():
    irreps = {"state": o3.Irreps("1x0e")}
    block = _ContractiveBlock(
        irreps_in=irreps,
        state_field="state",
        out_field="state",
        scale=0.0,
        bias=1.0,
    )
    module = RecyclingModule(
        block=block,
        state_field="state",
        block_out_field="state",
        max_steps=3,
        alpha=0.5,
        predict_delta=True,
        irreps_in=irreps,
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(2, 3),
        "state": torch.ones(2, 1),
    }
    out = module(data)
    expected = torch.full((2, 1), 2.5)
    torch.testing.assert_close(out["state"], expected)


def test_recycling_module_pins_masked_state_each_recycle_in_delta_mode():
    irreps = {"state": o3.Irreps("1x0e")}
    block = _ContractiveBlock(
        irreps_in=irreps,
        state_field="state",
        out_field="state",
        scale=0.0,
        bias=1.0,
    )
    module = RecyclingModule(
        block=block,
        state_field="state",
        block_out_field="state",
        max_steps=3,
        alpha=1.0,
        predict_delta=True,
        feedback_from_fields=["state"],
        feedback_to_fields=["state"],
        feedback_apply_mask=True,
        feedback_mask_suffix="__mask__",
        irreps_in=irreps,
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(2, 3),
        "state": torch.tensor([[10.0], [0.0]]),
        "state__mask__": torch.tensor([[True], [False]]),
    }
    out = module(data)
    # First entry is pinned to leaked value across all recycle steps; second accumulates deltas.
    expected = torch.tensor([[10.0], [3.0]])
    torch.testing.assert_close(out["state"], expected)


def test_recycling_module_syncs_scalar_companion_field_from_final_state():
    irreps = {"state": o3.Irreps("2x0e")}
    block = _ReadoutStyleBlock(irreps_in=irreps)
    module = RecyclingModule(
        block=block,
        state_field="state",
        block_out_field="state",
        max_steps=1,
        alpha=0.5,
        predict_delta=False,
        sync_scalar_outputs=True,
        irreps_in=irreps,
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(2, 3),
        "state": torch.ones(2, 2),
    }
    out = module(data)
    # state becomes 1.5 in absolute mode with proposal=2 and alpha=0.5
    torch.testing.assert_close(out["state"], torch.full((2, 2), 1.5))
    # scalar companion must reflect the final recycled state first scalar component.
    torch.testing.assert_close(out["state_scalar"], torch.full((2, 1), 1.5))
