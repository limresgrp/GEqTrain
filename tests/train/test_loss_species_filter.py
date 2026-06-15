import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.train._loss import LossWrapper
from geqtrain.train.loss import Loss
from geqtrain.train.components.setup import setup_loss


def _make_ref():
    register_fields(node_fields=["cs_iso"])
    return {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((3, 3), dtype=torch.float32),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        AtomicDataDict.NODE_TYPE_KEY: torch.tensor([[0], [1], [0]], dtype=torch.long),
        "cs_iso": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
    }


def test_loss_wrapper_node_type_indices_filters_only_selected_species():
    loss = LossWrapper("L1Loss", params={"node_type_indices": [1]})
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()

    out = loss(pred=pred, ref=ref, key="cs_iso", mean=False)

    assert out.shape == torch.Size([1, 1])
    assert torch.allclose(out, torch.tensor([[3.0]]))


def test_loss_wrapper_node_type_names_requires_type_names_list():
    loss = LossWrapper(
        "L1Loss",
        params={"node_type_names": ["H"], "type_names": ["X", "H", "C"]},
    )
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()

    out = loss(pred=pred, ref=ref, key="cs_iso", mean=False)

    assert out.shape == torch.Size([1, 1])
    assert torch.allclose(out, torch.tensor([[3.0]]))


def test_setup_loss_injects_global_type_names():
    loss = setup_loss(
        {
            "type_names": ["X", "H", "C"],
            "loss_coeffs": [
                {
                    "cs_iso": [
                        1.0,
                        "L1Loss",
                        {"node_type_names": ["H"]},
                    ]
                }
            ],
        }
    )

    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()
    out = loss.funcs["cs_iso_0"](pred=pred, ref=ref, key="cs_iso", mean=False)

    assert torch.allclose(out, torch.tensor([[3.0]]))


class _CustomMeanAbsLoss:
    def __init__(self, **kwargs):
        self.last_shape = None

    def __call__(self, pred, ref, key, mean=True, **kwargs):
        self.last_shape = pred[key].shape
        out = torch.abs(pred[key] - ref[key])
        return out.mean() if mean else out


def test_loss_framework_filters_before_custom_loss():
    loss = Loss(
        components=[
            {
                "cs_iso": [
                    1.0,
                    _CustomMeanAbsLoss,
                    {"node_type_names": ["H"], "type_names": ["X", "H", "C"]},
                ]
            }
        ]
    )
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()

    total, contrib = loss(pred=pred, ref=ref)

    assert torch.allclose(total, torch.tensor(3.0))
    assert torch.allclose(contrib["cs_iso_0"], torch.tensor(3.0))
    assert loss.funcs["cs_iso_0"].last_shape == torch.Size([1, 1])


def test_loss_wrapper_empty_species_filter_returns_graph_connected_zero():
    loss = LossWrapper("L1Loss", params={"node_type_indices": [2]})
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32, requires_grad=True)}
    ref = _make_ref()

    out = loss(pred=pred, ref=ref, key="cs_iso", mean=True)
    out.backward()

    assert out.requires_grad
    assert torch.allclose(out.detach(), torch.tensor(0.0))
    assert torch.allclose(pred["cs_iso"].grad, torch.zeros_like(pred["cs_iso"]))


def test_loss_framework_empty_species_filter_returns_graph_connected_zero_for_custom_loss():
    loss = Loss(
        components=[
            {
                "cs_iso": [
                    1.0,
                    _CustomMeanAbsLoss,
                    {"node_type_names": ["C"], "type_names": ["X", "H", "C"]},
                ]
            }
        ]
    )
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32, requires_grad=True)}
    ref = _make_ref()

    total, contrib = loss(pred=pred, ref=ref)
    total.backward()

    assert total.requires_grad
    assert torch.allclose(total.detach(), torch.tensor(0.0))
    assert torch.allclose(contrib["cs_iso_0"], torch.tensor(0.0))
    assert torch.allclose(pred["cs_iso"].grad, torch.zeros_like(pred["cs_iso"]))
    assert loss.funcs["cs_iso_0"].last_shape is None
