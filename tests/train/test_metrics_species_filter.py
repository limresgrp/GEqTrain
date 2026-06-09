import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.train.metrics import Metrics, _Metric
from geqtrain.train._loss import StatefulMetric


def _make_ref():
    register_fields(node_fields=["cs_iso"])
    return {
        AtomicDataDict.POSITIONS_KEY: torch.zeros((3, 3), dtype=torch.float32),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
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
