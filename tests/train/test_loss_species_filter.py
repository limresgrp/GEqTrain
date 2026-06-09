import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.train._loss import LossWrapper


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
