import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.train.metrics import Metrics, _Metric
from geqtrain.train._loss import StatefulMetric


def _make_ref():
    register_fields(node_fields=["cs_iso"])
    return {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((3, 3), dtype=torch.float32),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
        AtomicDataDict.NODE_TYPE_KEY: torch.tensor([[0], [1], [0]], dtype=torch.long),
        "cs_iso": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
    }


def test_metrics_stateless_node_type_names_filters_selected_species():
    metrics = Metrics(
        components=[
            {
                "cs_iso": [
                    1.0,
                    "L1Loss",
                    {"node_type_names": ["H"], "type_names": ["X", "H", "C"]},
                ]
            }
        ]
    )
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()

    metrics(pred=pred, ref=ref)
    result = metrics.metrics["cs_iso_0"].accumulator.current_result()

    assert torch.allclose(result, torch.tensor([3.0]))

    flat = metrics.flatten_metrics(
        {"cs_iso_0": metrics.metrics["cs_iso_0"].accumulator.current_result()},
        {"type_names": ["X", "H", "C"]},
    )
    assert flat == {"H_cs_iso_L1Loss_mean": 3.0}


class _RecordingStatefulMetric(StatefulMetric):
    def __init__(self):
        super().__init__()
        self.last_pred = None
        self.last_ref = None

    def update(self, pred: dict, ref: dict, key: str):
        self.last_pred = pred[key].detach().clone()
        self.last_ref = ref[key].detach().clone()

    def compute(self):
        return torch.abs(self.last_pred - self.last_ref).sum()

    def reset(self):
        self.last_pred = None
        self.last_ref = None


def test_metrics_stateful_node_type_names_filters_selected_species():
    metric = _Metric(
        _RecordingStatefulMetric(),
        {"node_type_names": ["H"], "type_names": ["X", "H", "C"]},
    )
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()

    metric.accumulate(pred=pred, ref=ref, key="cs_iso", normalization_fields={})

    assert torch.allclose(metric.accumulator.last_pred, torch.tensor([[5.0]]))
    assert torch.allclose(metric.accumulator.last_ref, torch.tensor([[2.0]]))


def test_metrics_per_species_node_type_names_uses_selected_species_even_if_not_centers():
    metrics = Metrics(
        components=[
            {
                "cs_iso": [
                    1.0,
                    "L1Loss",
                    {"PerSpecies": True},
                    {"node_type_names": ["H"], "type_names": ["X", "H", "C"]},
                ]
            }
        ]
    )
    pred = {"cs_iso": torch.tensor([[1.0], [5.0], [3.0]], dtype=torch.float32)}
    ref = _make_ref()

    batch_metrics = metrics(pred=pred, ref=ref)
    assert "cs_iso_0" in batch_metrics
    assert torch.allclose(batch_metrics["cs_iso_0"], torch.tensor([0.0, 3.0]))

    flat = metrics.flatten_metrics(
        {"cs_iso_0": metrics.metrics["cs_iso_0"].accumulator.current_result()},
        {"type_names": ["X", "H", "C"]},
    )
    assert flat == {"H_cs_iso_L1Loss_mean": 3.0}


def test_metrics_node_mask_center_length_aligns_species_filter():
    metrics = Metrics(
        components=[
            {
                "cs_iso": [
                    1.0,
                    "L1Loss",
                    {"node_type_names": ["H"], "type_names": ["X", "H", "C"]},
                    {"node_mask_field": "center_atoms_mask"},
                ]
            }
        ]
    )
    pred = {
        "cs_iso": torch.tensor([[1.0], [5.0]], dtype=torch.float32),
        AtomicDataDict.NODE_TYPE_KEY: torch.tensor([[0], [1], [2], [1]], dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[1, 3], [0, 2]], dtype=torch.long),
    }
    ref = {
        "cs_iso": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        AtomicDataDict.NODE_TYPE_KEY: torch.tensor([[0], [1], [2], [1]], dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[1, 3], [0, 2]], dtype=torch.long),
        "center_atoms_mask": torch.tensor([True, False], dtype=torch.bool),
    }

    batch_metrics = metrics(pred=pred, ref=ref)
    assert torch.allclose(batch_metrics["cs_iso_0"], torch.tensor([0.0]))
