import yaml
import torch

from geqtrain.utils.config import Config
from geqtrain.utils.deploy import build_deployment, load_deployed_model
from geqtrain.train.components.checkpointing import CheckpointHandler


def _extract_edge_input_attr_order(config_dict):
    for layer in config_dict["model"]["stack"]:
        if layer.get("name") == "edge_input_attrs":
            return list(layer["attributes"].keys())
    raise AssertionError("edge_input_attrs layer not found in deployed config")


def test_build_deployment_preserves_model_edge_attribute_order(monkeypatch, tmp_path):
    stale_file_config = Config(
        {
            "r_max": 12.0,
            "allow_tf32": False,
            "edge_attributes": {
                "bead_is_next": {"embedding_dimensionality": 8, "num_types": 2},
                "bead_is_prev": {"embedding_dimensionality": 8, "num_types": 2},
            },
            "model": {
                "stack": [
                    {
                        "name": "edge_input_attrs",
                        "attributes": {
                            "bead_is_next": {"embedding_dimensionality": 8, "num_types": 2},
                            "bead_is_prev": {"embedding_dimensionality": 8, "num_types": 2},
                        },
                    }
                ]
            },
        }
    )
    trainer_config = Config(
        {
            "r_max": 12.0,
            "allow_tf32": False,
            "edge_attributes": {
                "bead_is_prev": {"embedding_dimensionality": 8, "num_types": 2},
                "bead_is_next": {"embedding_dimensionality": 8, "num_types": 2},
            },
            "model": {
                "stack": [
                    {
                        "name": "edge_input_attrs",
                        "attributes": {
                            "bead_is_prev": {"embedding_dimensionality": 8, "num_types": 2},
                            "bead_is_next": {"embedding_dimensionality": 8, "num_types": 2},
                        },
                    }
                ]
            },
        }
    )

    def _fake_load_model_from_training_session(traindir, model_name="best_model.pth", device="cpu"):
        return torch.nn.Identity(), trainer_config

    monkeypatch.setattr(
        CheckpointHandler,
        "load_model_from_training_session",
        staticmethod(_fake_load_model_from_training_session),
    )
    monkeypatch.setattr("geqtrain.utils.deploy.apply_global_config", lambda config: None)
    monkeypatch.setattr("geqtrain.utils.deploy.verify_deployment", lambda *args, **kwargs: None)

    out_file = tmp_path / "deploy_test.pt"
    build_deployment(
        model_path=tmp_path / "best_model.pth",
        out_file=out_file,
        config=stale_file_config,
    )

    _, metadata = load_deployed_model(out_file, device="cpu", set_global_options=False)
    deployed_config = yaml.safe_load(metadata["config"])

    assert list(deployed_config["edge_attributes"].keys()) == ["bead_is_prev", "bead_is_next"]
    assert _extract_edge_input_attr_order(deployed_config) == ["bead_is_prev", "bead_is_next"]
