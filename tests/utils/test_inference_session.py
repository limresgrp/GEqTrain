import torch

from geqtrain.inference import InferenceSession
from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.utils.config import Config
from geqtrain.utils.inference_metadata import INFERENCE_METADATA_KEY, dump_inference_metadata_bundle
from geqtrain.utils.torch_geometric import Batch, Data


class _EchoEnergyModel(torch.nn.Module):
    def forward(self, data):
        return {"energy": data["energy"]}


def _make_minibatch():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = Data(
        edge_index=edge_index,
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        node_types=torch.tensor([[0], [0]], dtype=torch.long),
        energy=torch.tensor([[1.5]], dtype=torch.float32),
    )
    g.ensemble_index = 0
    return Batch.from_data_list([g])


def test_inference_session_predict_is_transparent_across_model_sources(monkeypatch):
    config = Config.from_dict({"normalization": {"energy": "global"}})
    bundle = {
        "version": 1,
        "normalization": {"energy": {"mode": "global", "transform": {"name": "none"}, "irreps": None, "apply_on_dataset": True}},
        "denormalize_inference_outputs": True,
        "default_ensemble": "0",
        "normalization_stats_by_ensemble": {"0": {"_mean_.global.energy": 1.0, "_std_.global.energy": 2.0}},
    }
    metadata = {INFERENCE_METADATA_KEY: dump_inference_metadata_bundle(bundle)}

    def _fake_load_model(model_path_str, device="cpu"):
        return _EchoEnergyModel(), config, metadata

    monkeypatch.setattr(CheckpointHandler, "load_model", staticmethod(_fake_load_model))

    session = InferenceSession.from_model_path("dummy.pth", device="cpu")
    batch = _make_minibatch()
    out, _, _, _ = session.predict(batch)
    assert torch.allclose(out["energy"], torch.tensor([[4.0]], dtype=out["energy"].dtype))
    assert session.inference_metadata["normalization_stats_by_ensemble"]["0"]["_std_.global.energy"] == 2.0
