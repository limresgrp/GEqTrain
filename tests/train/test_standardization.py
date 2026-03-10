import numpy as np
import pytest
import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.data.dataset import NpzDataset
from geqtrain.train._loss import LogCoshLoss, LossWrapper
from geqtrain.train.components.inference import run_inference
from geqtrain.train.components.setup import setup_metrics
from geqtrain.utils.config import Config
from geqtrain.utils.normalization import get_transform_param_key, resolve_normalization_map
from geqtrain.utils.torch_geometric import Batch


def _build_toy_dataset(
    tmp_path,
    *,
    energy_values=None,
    normalization=None,
):
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
    if energy_values is None:
        energy = np.array([1.0, 3.0], dtype=np.float32)
    else:
        energy = np.asarray(energy_values, dtype=np.float32)
    node_types = np.array([0, 1], dtype=np.int64)
    np.savez(npz_path, coords=pos, forces=forces, energy=energy, atom_types=node_types)

    if normalization is None:
        normalization = {
            "energy": "global",
            "forces": "per_type:1x1e",
        }

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
        normalization=normalization,
    )
    return dataset, energy, forces


def _build_toy_tensor_dataset(tmp_path, *, transform):
    register_fields(node_fields=["cs_tensor"], fixed_fields=["node_types"])

    npz_path = tmp_path / "toy_tensor.npz"
    pos = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    cs_tensor = np.array(
        [
            [[-1.5, 0.2, -0.1, 0.3, -0.4, 0.5], [0.8, -0.6, 0.2, -0.3, 0.1, -0.2]],
            [[-0.9, 0.4, -0.2, 0.7, -0.5, 0.1], [1.3, -0.8, 0.5, -0.6, 0.2, -0.1]],
        ],
        dtype=np.float32,
    )
    node_types = np.array([0, 1], dtype=np.int64)
    np.savez(npz_path, coords=pos, cs_tensor=cs_tensor, atom_types=node_types)

    normalization = {
        "cs_tensor": {
            "mode": "per_type:1x0e+1x2e",
            "transform": transform,
        }
    }

    dataset = NpzDataset(
        root=str(tmp_path),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={
            "coords": "pos",
            "cs_tensor": "cs_tensor",
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
        normalization=normalization,
    )
    return dataset, cs_tensor, normalization


def test_normalize_and_denormalize_roundtrip(tmp_path):
    dataset, raw_energy, raw_forces = _build_toy_dataset(tmp_path)

    assert torch.isclose(dataset.data["energy"].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(dataset.data["energy"].std(), torch.tensor(1.0), atol=1e-5)

    batch = Batch.from_data_list([dataset.get(0), dataset.get(1)])
    node_types = batch[AtomicDataDict.NODE_TYPE_KEY].to(dtype=torch.long).squeeze(-1)
    stds = dataset.fixed_fields["_std_.per_type.forces"].to(batch["forces"].dtype)[node_types]
    expected_std_forces = torch.from_numpy(raw_forces.reshape(-1, 3)).to(batch["forces"].dtype) / stds
    assert torch.allclose(batch["forces"], expected_std_forces, atol=1e-5)

    loss = LossWrapper("L1Loss")
    normalization_fields = resolve_normalization_map({"normalization": dataset.normalization})

    pred_energy = {"energy": batch["energy"].clone()}
    ref_energy = batch.to_dict()
    energy_denorm, _ = loss._prepare_tensors(
        pred=pred_energy,
        ref=ref_energy,
        pred_key_name="energy",
        ref_key_name="energy",
        mean=False,
        normalization_fields=normalization_fields,
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
        normalization_fields=normalization_fields,
    )
    expected_forces = torch.from_numpy(raw_forces.reshape(-1, 3)).to(forces_denorm.dtype)
    assert torch.allclose(forces_denorm, expected_forces, atol=1e-5)


def test_metrics_use_normalization_map():
    config = Config.from_dict(
        {
            "metrics_components": [{"energy": ["L1Loss"]}],
            "normalization": {"energy": "global"},
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
    assert value == pytest.approx(3.0)


def test_legacy_normalization_keys_raise():
    with pytest.raises(ValueError, match="Legacy normalization keys"):
        Config.from_dict({"standardize_fields": {"energy": "global"}})


def test_signed_log1p_normalization_roundtrip(tmp_path):
    normalization = {
        "energy": {
            "mode": "global",
            "transform": "signed_log1p",
        }
    }
    dataset, raw_energy, _ = _build_toy_dataset(
        tmp_path,
        energy_values=[-2.0, 3.0],
        normalization=normalization,
    )
    batch = Batch.from_data_list([dataset.get(0), dataset.get(1)])

    loss = LossWrapper("L1Loss")
    normalization_fields = resolve_normalization_map(
        {"normalization": normalization},
    )
    pred_energy = {"energy": batch["energy"].clone()}
    ref_energy = batch.to_dict()
    energy_denorm, _ = loss._prepare_tensors(
        pred=pred_energy,
        ref=ref_energy,
        pred_key_name="energy",
        ref_key_name="energy",
        mean=False,
        normalization_fields=normalization_fields,
    )
    expected_energy = torch.from_numpy(raw_energy).to(energy_denorm.dtype).reshape(-1, 1)
    assert torch.allclose(energy_denorm, expected_energy, atol=1e-5)


def test_yeo_johnson_normalization_roundtrip(tmp_path):
    normalization = {
        "energy": {
            "mode": "global",
            "transform": {
                "name": "yeo_johnson",
                "lambda": "auto",
            },
        }
    }
    dataset, raw_energy, _ = _build_toy_dataset(
        tmp_path,
        energy_values=[-2.0, 3.0],
        normalization=normalization,
    )
    batch = Batch.from_data_list([dataset.get(0), dataset.get(1)])
    lambda_key = get_transform_param_key("energy", "lambda")
    assert lambda_key in batch

    loss = LossWrapper("L1Loss")
    normalization_fields = resolve_normalization_map(
        {"normalization": normalization},
    )
    pred_energy = {"energy": batch["energy"].clone()}
    ref_energy = batch.to_dict()
    energy_denorm, _ = loss._prepare_tensors(
        pred=pred_energy,
        ref=ref_energy,
        pred_key_name="energy",
        ref_key_name="energy",
        mean=False,
        normalization_fields=normalization_fields,
    )
    expected_energy = torch.from_numpy(raw_energy).to(energy_denorm.dtype).reshape(-1, 1)
    assert torch.allclose(energy_denorm, expected_energy, atol=1e-5)


@pytest.mark.parametrize(
    "transform_spec, expect_lambda",
    [
        ("signed_log1p", False),
        ({"name": "yeo_johnson", "lambda": "auto"}, True),
    ],
)
def test_equivariant_transform_roundtrip(tmp_path, transform_spec, expect_lambda):
    dataset, raw_tensor, normalization = _build_toy_tensor_dataset(
        tmp_path,
        transform=transform_spec,
    )
    batch = Batch.from_data_list([dataset.get(0), dataset.get(1)])
    if expect_lambda:
        lambda_key = get_transform_param_key("cs_tensor", "lambda")
        assert lambda_key in batch

    loss = LossWrapper("L1Loss")
    normalization_fields = resolve_normalization_map(
        {"normalization": normalization},
    )
    pred_tensor = {"cs_tensor": batch["cs_tensor"].clone()}
    ref_tensor = batch.to_dict()
    tensor_denorm, _ = loss._prepare_tensors(
        pred=pred_tensor,
        ref=ref_tensor,
        pred_key_name="cs_tensor",
        ref_key_name="cs_tensor",
        mean=False,
        normalization_fields=normalization_fields,
    )
    expected_tensor = torch.from_numpy(raw_tensor.reshape(-1, 6)).to(tensor_denorm.dtype)
    assert torch.allclose(tensor_denorm, expected_tensor, atol=1e-5)


class _InferenceEchoEnergy(torch.nn.Module):
    def forward(self, data):
        return {"energy": data["energy"]}


def test_run_inference_denormalizes_outputs_when_loss_fn_missing(tmp_path):
    dataset, raw_energy, _ = _build_toy_dataset(tmp_path)
    batch = Batch.from_data_list([dataset.get(0), dataset.get(1)])
    model = _InferenceEchoEnergy()
    out, _, _, _ = run_inference(
        model=model,
        data=batch,
        device=torch.device("cpu"),
        config={"normalization": {"energy": "global"}},
        loss_fn=None,
        is_train=False,
        current_epoch=0,
    )
    expected = torch.from_numpy(raw_energy).reshape(-1, 1).to(out["energy"].dtype)
    assert torch.allclose(out["energy"], expected, atol=1e-5)


def test_log_cosh_loss_is_robust_to_outliers():
    loss = LogCoshLoss(beta=1.0)

    pred = {"value": torch.tensor([[0.0], [10.0]], dtype=torch.float32)}
    ref = {"value": torch.tensor([[0.0], [0.0]], dtype=torch.float32)}
    out = loss(pred=pred, ref=ref, key="value", mean=True)

    mse = torch.nn.MSELoss()(pred["value"], ref["value"])
    assert out < mse
