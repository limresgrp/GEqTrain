from pathlib import Path

import pytest
import torch
from e3nn import o3
from e3nn.util.jit import compile as compile_module

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.model import model_from_config
from geqtrain.nn import MaskEdgeFeatures
from geqtrain.train.loss import Loss
from geqtrain.train.metrics import Metrics
from geqtrain.train.components.inference import get_output_keys, run_inference
from geqtrain.train.components.callbacks import ValidationBatchPredictionLogger
from geqtrain.utils import load_hydra_config
from geqtrain.utils.torch_geometric import Batch


def make_module(irreps="2x0e", **kwargs):
    return MaskEdgeFeatures(
        field="edge_test_input", out_field="edge_reconstruction",
        irreps_in={"edge_test_input": irreps}, **kwargs,
    )


def make_data(dim=2, n=30, dtype=torch.float32):
    return {
        "edge_test_input": torch.randn(n, dim, dtype=dtype, requires_grad=True),
        AtomicDataDict.EDGE_INDEX_KEY: torch.stack([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mask_preserves_full_detached_target_and_unmasked_rows(dtype):
    module = make_module(mask_fraction=0.5)
    data = make_data(dtype=dtype)
    original = data["edge_test_input"]
    torch.manual_seed(4)
    out = module(dict(data))
    mask = out["edge_reconstruction_mask"]
    assert mask.any() and (~mask).any()
    torch.testing.assert_close(out["edge_reconstruction_target"], original)
    assert not out["edge_reconstruction_target"].requires_grad
    assert out["edge_reconstruction_target"].data_ptr() != original.data_ptr()
    torch.testing.assert_close(out["edge_test_input"][~mask], original[~mask])
    assert not out["edge_test_input"][mask].any()
    out["edge_test_input"].sum().backward()
    torch.testing.assert_close(original.grad, (~mask)[:, None].expand_as(original).to(dtype))
    assert not torch.equal(mask, module(dict(data))["edge_reconstruction_mask"])


@pytest.mark.parametrize("kwargs", [{"enabled": False}, {"mask_fraction": 0.0}])
def test_disabled_corruption_is_identity(kwargs):
    module = make_module(**kwargs)
    data = make_data()
    out = module(dict(data))
    torch.testing.assert_close(out["edge_test_input"], data["edge_test_input"])
    assert not out["edge_reconstruction_mask"].any()


def test_eval_is_clean_unless_explicitly_enabled():
    data = make_data()
    module = make_module(mask_fraction=1.0).eval()
    assert not module(dict(data))["edge_reconstruction_mask"].any()
    module.mask_in_eval = True
    assert module(dict(data))["edge_reconstruction_mask"].all()


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("nan")])
def test_invalid_probability_is_rejected(invalid):
    with pytest.raises(ValueError, match="mask_fraction"):
        make_module(mask_fraction=invalid)


def test_nonzero_equivariant_mask_and_field_collisions_are_rejected():
    with pytest.raises(ValueError, match="preserve equivariance"):
        make_module(irreps="1x1o", mask_value=1.0)
    with pytest.raises(ValueError, match="distinct"):
        make_module(mask_field="edge_test_input")


@pytest.mark.parametrize("reflection", [False, True])
def test_whole_vector_masking_is_equivariant_for_a_fixed_sample(reflection):
    irreps = o3.Irreps("1x0e+1x1o+1x2e")
    module = make_module(irreps, mask_fraction=0.4).double()
    data = make_data(dim=irreps.dim, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64) * (-1 if reflection else 1)
    d = irreps.D_from_matrix(rotation)
    rotated = dict(data, edge_test_input=data["edge_test_input"] @ d.T)
    torch.manual_seed(12)
    out = module(dict(data))
    torch.manual_seed(12)
    rot_out = module(rotated)
    for key in ["edge_test_input", "edge_reconstruction_target"]:
        torch.testing.assert_close(rot_out[key], out[key] @ d.T)


def test_module_scripts_and_supports_empty_edges():
    module = compile_module(make_module(mask_fraction=0.5))
    out = module(make_data(n=0))
    assert out["edge_reconstruction_target"].shape == (0, 2)


@pytest.mark.parametrize("loss_name", ["MSELoss", "geqtrain.train.LogCoshLoss"])
def test_shared_mask_filters_loss_and_metrics_before_the_loss(loss_name):
    make_module()
    target = torch.zeros(5, 2)
    prediction = torch.tensor([[1., 2.], [999., 999.], [3., 4.], [999., 999.], [float("nan"), 8.]], requires_grad=True)
    pred = {"edge_reconstruction": prediction, "selected_edges": torch.tensor([True, False, True, False, True])}
    ref = {"edge_reconstruction": target}
    params = {"mask_field": "selected_edges", "ignore_nan": True}
    loss = Loss([{"edge_reconstruction": [1.0, loss_name, params]}])
    value, _ = loss(pred, ref)
    expected = Loss([{"edge_reconstruction": [loss_name]}])(
        {"edge_reconstruction": prediction[:3:2]}, {"edge_reconstruction": target[:3:2]},
    )[0]
    torch.testing.assert_close(value, expected)
    value.backward()
    assert not prediction.grad[[1, 3, 4]].any()
    metrics = Metrics([{"edge_reconstruction": ["L1Loss", params]}])
    metrics(pred, ref)
    assert metrics.current_result()["edge_reconstruction_0"].item() == pytest.approx(2.5)


def test_empty_masks_give_differentiable_zero_and_no_metric_samples():
    make_module()
    pred = {"edge_reconstruction": torch.randn(4, 2, requires_grad=True), "selected_edges": torch.zeros(4, dtype=torch.bool)}
    ref = {"edge_reconstruction": torch.zeros(4, 2)}
    components = [{"edge_reconstruction": ["MSELoss", {"mask_field": "selected_edges"}]}]
    value, _ = Loss(components)(pred, ref)
    assert value.item() == 0.0
    value.backward()
    assert not pred["edge_reconstruction"].grad.any()
    metrics = Metrics(components)
    assert metrics(pred, ref) == {}


def test_required_edge_mask_does_not_silently_use_node_alignment():
    make_module()
    pred = {"edge_reconstruction": torch.zeros(5, 2)}
    ref = {"edge_reconstruction": torch.zeros(5, 2)}
    loss = Loss([{"edge_reconstruction": ["MSELoss", {"mask_field": "selected_edges"}]}])
    with pytest.raises(RuntimeError, match="missing"):
        loss(pred, ref)
    pred["selected_edges"] = torch.ones(2, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="5 rows"):
        loss(pred, ref)


def test_generic_mask_ands_with_node_filters_and_nan_filter():
    register_fields(node_fields=["masked_node_target"])
    target = torch.zeros(5, 1)
    target[3] = torch.nan
    pred = {
        "masked_node_target": torch.tensor([[2.], [20.], [30.], [40.], [50.]]),
        "node_types": torch.tensor([[0], [0], [1], [0], [0]]),
        "node_selection": torch.tensor([True, True, True, True, False]),
        "row_selection": torch.tensor([True, False, True, True, True]),
    }
    params = {"mask_field": "row_selection", "node_mask_field": "node_selection",
              "node_type_indices": [0], "ignore_nan": True}
    components = [{"masked_node_target": ["L1Loss", params]}]
    ref = {"masked_node_target": target}
    assert Loss(components)(pred, ref)[0].item() == 2.0
    metrics = Metrics(components)
    metrics(pred, ref)
    assert metrics.current_result()["masked_node_target_0"].item() == 2.0


def test_prediction_csv_uses_framework_mask_for_custom_loss():
    make_module()
    loss = Loss([{"edge_reconstruction": ["geqtrain.train.LogCoshLoss", {"mask_field": "selected_edges"}]}])
    logger = ValidationBatchPredictionLogger()
    logger._loss_func_by_key["edge_reconstruction"] = loss.funcs[loss.keys[0]]
    logger._target_filter_by_key["edge_reconstruction"] = loss.target_filters[loss.keys[0]]
    pred = {"edge_reconstruction": torch.tensor([[1., 2.], [90., 99.]]), "selected_edges": torch.tensor([True, False])}
    ref = {"edge_reconstruction": torch.zeros(2, 2)}
    predicted, target = logger._extract_pred_ref("edge_reconstruction", pred, ref)
    torch.testing.assert_close(predicted, pred["edge_reconstruction"][:1])
    torch.testing.assert_close(target, ref["edge_reconstruction"][:1])


@pytest.mark.parametrize("chunking", [False, True])
def test_hydra_geometry_interaction_heads_and_inference_targets(chunking):
    path = Path(__file__).resolve().parents[2] / "config/model/examples/masked_geometry.yaml"
    config = load_hydra_config(str(path), overrides=["geometry_mask_fraction=1.0"])
    model, _ = model_from_config(config=config, initialize=False)
    model.train()
    batch = AtomicData.from_points(
        pos=torch.stack([torch.arange(6.), torch.zeros(6), torch.zeros(6)], dim=-1),
        r_max=1.1, node_types=(torch.arange(6) % 4).unsqueeze(-1),
    )
    batch = Batch.from_data_list([batch])
    loss = Loss(config["loss_coeffs"])
    assert set(get_output_keys(loss)[0]) == {"radial_reconstruction", "angular_reconstruction"}
    out, ref, _, _ = run_inference(
        model, batch, torch.device("cpu"),
        {"chunking": chunking, "batch_max_atoms": 3}, loss_fn=loss, is_train=True,
    )
    for field, dim in [("radial_reconstruction", 8), ("angular_reconstruction", 9)]:
        assert out[field].shape == ref[field].shape
        assert out[field].shape[1] == dim
        assert not ref[field].requires_grad
        assert out[field + "_mask"].all()
    value, contrib = loss(out, ref)
    assert torch.isfinite(value)
    assert len(contrib) == 2
    if chunking:
        assert out["edge_index"].shape[1] < batch.edge_index.shape[1]
    value.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    out, ref, _, _ = run_inference(model, batch, torch.device("cpu"), {}, loss_fn=loss)
    assert loss(out, ref)[0].item() == 0.0
    assert not out["radial_reconstruction_mask"].any()
    assert not out["angular_reconstruction_mask"].any()
