import numpy as np
import pytest
import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.data.dataset import NpzDataset
from geqtrain.train._loss import LossWrapper
from geqtrain.train.components.setup import setup_metrics
from geqtrain.utils.config import Config
from geqtrain.utils.torch_geometric import Batch


def _build_toy_dataset(tmp_path):
    register_fields(node_fields=["forces"], graph_fields=["energy"], fixed_fields=["node_types"])

    npz_path = tmp_path / "toy.npz"
    pos = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    forces = np.array(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    energy = np.array([1.0, 3.0], dtype=np.float32)
    node_types = np.array([0, 1], dtype=np.int64)
    np.savez(npz_path, coords=pos, forces=forces, energy=energy, atom_types=node_types)

    dataset = NpzDataset(
        root=str(tmp_path),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={
            "coords": "pos",
            "forces": "forces",
            "energy": "energy",
            "atom_types": "node_types",
        },
        extra_fixed_fields={AtomicDataDict.R_MAX_KEY: 3.0},
        node_attributes={
            AtomicDataDict.NODE_TYPE_KEY: {
                "embedding_mode": "one_hot",
                "num_types": 2,
                "fixed": True,
            }
        },
        standardize_fields={
            "energy": "global",
            "forces": "per_type:1x1e",
        },
    )
    return dataset, energy, forces


def test_standardize_and_destandardize_roundtrip(tmp_path):
    dataset, raw_energy, raw_forces = _build_toy_dataset(tmp_path)

    assert torch.isclose(dataset.data["energy"].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(dataset.data["energy"].std(), torch.tensor(1.0), atol=1e-5)

    batch = Batch.from_data_list([dataset.get(0), dataset.get(1)])
    node_types = batch[AtomicDataDict.NODE_TYPE_KEY].to(dtype=torch.long).squeeze(-1)
    stds = dataset.fixed_fields["_std_.per_type.forces"].to(batch["forces"].dtype)[node_types]
    expected_std_forces = torch.from_numpy(raw_forces.reshape(-1, 3)).to(batch["forces"].dtype) / stds
    assert torch.allclose(batch["forces"], expected_std_forces, atol=1e-5)

    loss = LossWrapper("L1Loss")
    destd_cfg = {"energy": "global", "forces": "per_type:1x1e"}

    pred_energy = {"energy": batch["energy"].clone()}
    ref_energy = batch.to_dict()
    energy_denorm, _ = loss._prepare_tensors(
        pred=pred_energy,
        ref=ref_energy,
        pred_key_name="energy",
        ref_key_name="energy",
        mean=False,
        destandardize_fields=destd_cfg,
    )
    expected_energy = torch.from_numpy(raw_energy).to(energy_denorm.dtype).reshape(-1, 1)
    assert torch.allclose(energy_denorm, expected_energy, atol=1e-5)

    pred_forces = {"forces": batch["forces"].clone()}
    ref_forces = batch.to_dict()
    forces_denorm, _ = loss._prepare_tensors(
        pred=pred_forces,
        ref=ref_forces,
        pred_key_name="forces",
        ref_key_name="forces",
        mean=False,
        destandardize_fields=destd_cfg,
    )
    expected_forces = torch.from_numpy(raw_forces.reshape(-1, 3)).to(forces_denorm.dtype)
    assert torch.allclose(forces_denorm, expected_forces, atol=1e-5)


def test_metrics_use_standardize_fields_when_destandardize_missing():
    config = Config.from_dict(
        {
            "metrics_components": [{"energy": ["L1Loss"]}],
            "standardize_fields": {"energy": "global"},
        }
    )
    metrics = setup_metrics(config)

    pred = {"energy": torch.tensor([[1.0]], dtype=torch.float32)}
    ref = {
        "energy": torch.tensor([[0.0]], dtype=torch.float32),
        "_mean_.global.energy": torch.tensor(2.0, dtype=torch.float32),
        "_std_.global.energy": torch.tensor(3.0, dtype=torch.float32),
    }
    out = metrics(pred=pred, ref=ref)
    value = next(iter(out.values())).item()

    # L1 difference in standardized space is 1, in original space is 3.
    assert value == pytest.approx(3.0)
