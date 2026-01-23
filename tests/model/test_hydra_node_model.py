from pathlib import Path

from geqtrain.model import model_from_config
from geqtrain.nn import SequentialGraphNetwork
from geqtrain.utils import load_hydra_config


def test_hydra_node_model_stack_builds():
    config_path = Path(__file__).resolve().parents[2] / "config" / "model" / "node_model.yaml"
    config = load_hydra_config(str(config_path))

    model, _ = model_from_config(config=config, initialize=False)

    assert isinstance(model, SequentialGraphNetwork)
    assert "interaction" in model._modules
    assert "edge_pooling" in model._modules
