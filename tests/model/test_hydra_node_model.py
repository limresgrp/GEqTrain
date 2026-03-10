from pathlib import Path

import torch

from geqtrain.model import model_from_config
from geqtrain.nn import SequentialGraphNetwork
from geqtrain.utils import load_hydra_config


def test_hydra_node_model_stack_builds():
    config_path = Path(__file__).resolve().parents[2] / "config" / "model" / "examples" / "node_model.yaml"
    config = load_hydra_config(str(config_path))

    model, _ = model_from_config(config=config, initialize=False)

    assert isinstance(model, SequentialGraphNetwork)
    assert "interaction" in model._modules
    assert "edge_pooling" in model._modules


def test_hydra_node_model_prefix_overrides_apply():
    config_path = Path(__file__).resolve().parents[2] / "config" / "model" / "examples" / "prefixed_node_model.yaml"
    config = load_hydra_config(str(config_path))

    model, _ = model_from_config(config=config, initialize=False)
    assert isinstance(model, SequentialGraphNetwork)

    interaction = model._modules["interaction"]
    out_irreps = interaction.irreps_out[interaction.out_field]
    assert any(ir.l == 0 and mul == 16 for mul, ir in out_irreps)

    weights_emb = interaction.initial_latent_generator.weights_emb
    linear_layers = [m for m in weights_emb.sequential if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].out_features == 12
    assert linear_layers[1].out_features == 13

    edge_radial = model._modules["edge_radial_attrs"]
    assert abs(edge_radial.basis.r_max - 7.0) < 1e-6
    assert abs(edge_radial.cutoff._factor - (1.0 / 7.0)) < 1e-6


def test_hydra_node_model_stack_append_groups_compose_in_order():
    config_path = Path(__file__).resolve().parents[2] / "config" / "model" / "examples" / "stacked_templates_example.yaml"
    config = load_hydra_config(str(config_path))

    model, _ = model_from_config(config=config, initialize=False)
    assert isinstance(model, SequentialGraphNetwork)

    names = list(model._modules.keys())
    assert "global_node_pooling" in names
    assert "head_energy" in names
    assert names.index("global_node_pooling") < names.index("head_energy")

    head = model._modules["head_energy"]
    linear_layers = [m for m in head.processor.scalar_processor.sequential if isinstance(m, torch.nn.Linear)]
    assert linear_layers[0].out_features == 12
    assert linear_layers[1].out_features == 13
