from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from e3nn import o3

from geqtrain.data import AtomicData
from geqtrain.data._build import dataset_from_config
from geqtrain.data.dataloader import DataLoader
from geqtrain.model import model_from_config
from geqtrain.scripts.evaluate import main as evaluate
from geqtrain.train.components.inference import run_inference
from geqtrain.train.components.setup import setup_loss
from geqtrain.train.trainer import Trainer
from geqtrain.utils import load_hydra_config
from geqtrain.utils._global_options import apply_global_config


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "tutorial/xxMD-DFT/config"
EXPERIMENTS = ["01_baseline", "01_attention", "02_context_control", "02_masked_geometry"]


@pytest.mark.parametrize("num_basis", [8, 16])
def test_radial_reconstruction_head_tracks_basis_width(num_basis):
    config = load_hydra_config(
        str(CONFIG_ROOT / "experiment/02_masked_geometry.yaml"),
        overrides=[f"num_basis={num_basis}"],
    )
    model, _ = model_from_config(config, initialize=False)
    assert model.irreps_out["radial_reconstruction"] == model.irreps_out["radial_reconstruction_target"]
    # Exercise the nonempty-mask loss, not only clean validation's zero loss.
    masker = model._modules["mask_radial"]
    head = model._modules["radial_head"]
    masker.train()
    masker.mask_fraction = 1.0
    data = {
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
        "radial_emb": torch.randn(3, num_basis),
        "edge_features": head.irreps_in["edge_features"].randn(3, -1),
    }
    out = head(masker(data))
    ref = {"radial_reconstruction": out["radial_reconstruction_target"]}
    from geqtrain.train.loss import Loss
    loss = Loss([{"radial_reconstruction": ["MSELoss", {"mask_field": "radial_reconstruction_mask"}]}])
    value, _ = loss(out, ref)
    assert torch.isfinite(value)
    value.backward()
    assert any(p.grad is not None for p in head.parameters())


@pytest.fixture
def small_data(tmp_path):
    rng = np.random.default_rng(42)
    for split in ["train", "val", "test"]:
        np.savez(
            tmp_path / f"azo_{split}.npz",
            coords=rng.normal(size=(2, 4, 3)),
            atom_types=np.array([1, 6, 7, 1]),
            energy=np.array([1., 2.]),
            forces=rng.normal(size=(2, 4, 3)),
        )
    return tmp_path


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_tutorial_npz_loading_training_and_clean_eval(experiment, small_data):
    config = load_hydra_config(
        str(CONFIG_ROOT / "experiment" / f"{experiment}.yaml"),
        overrides=[f"data_root={small_data}", f"root={small_data}/results"],
    )
    apply_global_config(config)
    dataset = dataset_from_config(config, prefix="train")
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch.node_types.reshape(-1).tolist() == [1, 6, 7, 1] * 2
    assert batch.forces.shape == (8, 3)
    model, _ = model_from_config(config, initialize=False)
    loss = setup_loss(config)
    model.train()
    out, ref, _, _ = run_inference(model, batch, torch.device("cpu"), config.as_dict(), loss_fn=loss, is_train=True)
    value, _ = loss(out, ref)
    assert torch.isfinite(value)
    value.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    out, ref, _, _ = run_inference(model, batch, torch.device("cpu"), config.as_dict(), loss_fn=loss)
    assert out["energy"].shape == (2, 1)
    assert out["forces"].shape == (8, 3)
    for field in ["radial_reconstruction", "angular_reconstruction"]:
        if field in out:
            assert not out[field + "_mask"].any()
            torch.testing.assert_close(ref[field], out[field + "_target"])
    # Polar force vectors must rotate and reflect; energy must remain invariant.
    data = AtomicData.to_AtomicDataDict(batch)
    transform = -o3.rand_matrix(dtype=batch.pos.dtype)
    transformed = dict(data, pos=data["pos"] @ transform.T + 2.0)
    original_out = model(dict(data))
    transformed_out = model(transformed)
    torch.testing.assert_close(transformed_out["energy"], original_out["energy"], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(transformed_out["forces"], original_out["forces"] @ transform.T, atol=2e-3, rtol=2e-3)


def test_context_ablation_changes_only_enable_flag_and_run_identity():
    control = load_hydra_config(str(CONFIG_ROOT / "experiment/02_context_control.yaml")).as_dict()
    masked = load_hydra_config(str(CONFIG_ROOT / "experiment/02_masked_geometry.yaml")).as_dict()
    for config in [control, masked]:
        config.pop("mask_geometry")
        config.pop("run_name")
        config.pop("experiment_description")
        config.pop("filepath")
        for layer in config["model"]["stack"]:
            if layer["_target_"] == "geqtrain.nn.MaskEdgeFeatures":
                layer.pop("enabled")
        for name in ["mask_radial", "mask_angular"]:
            config["blocks"][name].pop("enabled")
    assert control == masked


@pytest.mark.parametrize("experiment", ["01_baseline", "02_masked_geometry"])
def test_tutorial_one_epoch_checkpoint(experiment, small_data):
    root = small_data / "runs"
    config = load_hydra_config(
        str(CONFIG_ROOT / "experiment" / f"{experiment}.yaml"),
        overrides=[f"data_root={small_data}", f"root={root}", "max_epochs=1",
                   "batch_size=2", "validation_batch_size=2", "+device=cpu"],
    )
    trainer = Trainer(config=config)
    trainer.train()
    assert (root / experiment / "best_model.pth").exists()
    assert (root / experiment / "metrics_epoch.csv").exists()
    test_config = yaml.safe_load((CONFIG_ROOT / "data/test.yaml").read_text())
    test_config["test_dataset_list"][0]["dataset_input"] = str(small_data / "azo_test.npz")
    test_path = small_data / "test_config.yaml"
    test_path.write_text(yaml.safe_dump(test_config))
    evaluate([str(root / experiment / "best_model.pth"), str(test_path), "-d", "cpu", "-b", "2"])
