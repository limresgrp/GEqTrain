import pytest
import torch
import torch.nn as nn
from tests.utils.deployability import assert_module_deployable
import copy
from e3nn import o3

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.data.dataset import _NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS, _FIXED_FIELDS
from geqtrain.nn import InteractionModule
from geqtrain.nn.interaction import _inverse_quadratic_alpha_cap
from tests.utils.equivariance import assert_AtomicData_equivariant

# Common parameters for tests
BATCH_SIZE = 1
NUM_ATOMS = 5


def _create_dummy_data(config, device):
    """Creates a dummy AtomicDataDict based on the test config."""
    irreps_in = config["irreps_in"] # type: ignore
    
    # Basic graph structure
    pos = torch.randn(NUM_ATOMS, 3)
    data = {
        AtomicDataDict.BATCH_KEY: torch.zeros(NUM_ATOMS),
    }
    data = AtomicData.from_points(pos, r_max = 5.0, **data)
    NUM_EDGES = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

    # Generate random data for each field defined in irreps_in
    for key, irreps in irreps_in.items():
        if key in data: continue
        
        if key in _NODE_FIELDS:
            num_items = NUM_ATOMS
        elif key in _EDGE_FIELDS:
            num_items = NUM_EDGES
        else: # Graph fields
            num_items = BATCH_SIZE

        if irreps is not None:
            data[key] = o3.Irreps(irreps).randn(num_items, -1, device=device)

    return AtomicData.to_AtomicDataDict(data.to(device)), NUM_EDGES


BASE_IRREPS_IN = {
    AtomicDataDict.NODE_ATTRS_KEY: "8x0e",
    AtomicDataDict.EDGE_RADIAL_EMB_KEY: "4x0e",
    AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o+1x2e",
}


SCALAR_ONLY_OUTPUT_CONFIG = {
    "name": "scalar_only_output",
    "params": {
        "num_layers": 2,
        "latent_dim": 16,
        "eq_latent_multiplicity": 4,
        "output_ls": [0],  # Only scalar output
        "output_mul": 8,
    },
    "irreps_in": BASE_IRREPS_IN,
    "expected_out_irreps": "8x0e",
}

TEST_CONFIGS = [
    {
        "name": "base_case_so3",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
        },
        "irreps_in": {
            AtomicDataDict.NODE_ATTRS_KEY: "8x0e",
            AtomicDataDict.EDGE_RADIAL_EMB_KEY: "4x0e",
            AtomicDataDict.EDGE_SPHARMS_EMB_KEY: "1x0e+1x1o+1x2e",
        },
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_node_eq_input",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "node_equivariant_field": AtomicDataDict.NODE_EQ_ATTRS_KEY,
        },
        "irreps_in": copy.deepcopy(BASE_IRREPS_IN) | {
            AtomicDataDict.NODE_EQ_ATTRS_KEY: "2x1o",
        },
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_edge_inv_eq_input",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "edge_invariant_field": AtomicDataDict.EDGE_ATTRS_KEY,
            "edge_equivariant_field": AtomicDataDict.EDGE_EQ_ATTRS_KEY,
        },
        "irreps_in": copy.deepcopy(BASE_IRREPS_IN) | {
            AtomicDataDict.EDGE_ATTRS_KEY: "3x0e",
            AtomicDataDict.EDGE_EQ_ATTRS_KEY: "2x1o",
        },
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_attention",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "use_attention": True,
            "attention_head_dim": 4,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_attention_stability_knobs",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "use_attention": True,
            "attention_head_dim": 4,
            "attention_logit_clip": 8.0,
            "residual_update_max": 0.25,
            "use_equivariant_residual": True,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_fixed_point_recycling",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "use_fixed_point_recycling": True,
            "fp_max_iter": 4,
            "fp_tol": 1e-4,
            "fp_alpha": 0.5,
            "fp_grad_steps": 1,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_fixed_point_recycling_stabilized",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "use_fixed_point_recycling": True,
            "fp_max_iter": 4,
            "fp_tol": 1e-4,
            "fp_alpha": 0.5,
            "fp_grad_steps": 1,
            "fp_adaptive_damping": True,
            "fp_alpha_min": 0.1,
            "fp_residual_growth_tol": 0.95,
            "fp_first_layer_update_coeff": 0.2,
            "fp_state_clip_value": 10.0,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_fixed_point_static_context",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "use_fixed_point_recycling": True,
            "fp_max_iter": 4,
            "fp_tol": 1e-4,
            "fp_alpha": 0.5,
            "fp_grad_steps": 1,
            "fp_use_static_context": True,
            "fp_static_context_strength": 0.5,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_mace_product",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "use_mace_product": True,
            "product_correlation": 2,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "with_conditioning",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "conditioning_fields": ["graph_features"],
        },
        "irreps_in": copy.deepcopy(BASE_IRREPS_IN) | {
            "graph_features": "12x0e",
        },
        "expected_out_irreps": "16x0e+4x1o+4x2e",
    },
    {
        "name": "custom_output_ls_mul",
        "params": {
            "num_layers": 2,
            "latent_dim": 16,
            "eq_latent_multiplicity": 4,
            "output_ls": [0, 1], # Only output l=0 and l=1
            "output_mul": 8,
        },
        "irreps_in": BASE_IRREPS_IN,
        "expected_out_irreps": "8x0e+8x1o",
    },
    SCALAR_ONLY_OUTPUT_CONFIG,
]


@pytest.mark.parametrize("config", TEST_CONFIGS, ids=[c["name"] for c in TEST_CONFIGS])
def test_interaction_module(config):
    """
    Tests the InteractionModule for shape correctness, irreps propagation, and equivariance.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = config["params"]
    irreps_in = config["irreps_in"] # type: ignore

    # 1. Test Instantiation
    try:
        model = InteractionModule(irreps_in=irreps_in, **params).to(device)
    except Exception as e:
        pytest.fail(f"Failed to instantiate InteractionModule for config '{config['name']}': {e}")

    # 2. Check output irreps
    out_field = params.get("out_field", AtomicDataDict.EDGE_FEATURES_KEY)
    expected_out_irreps = config.get("expected_out_irreps") # type: ignore
    if expected_out_irreps:
        assert str(model.irreps_out[out_field]) == expected_out_irreps, \
            f"Output irreps mismatch for '{config['name']}'. Expected {expected_out_irreps}, got {model.irreps_out[out_field]}"

    # 3. Create dummy data and test forward pass
    data_in, NUM_EDGES = _create_dummy_data(config, device)
    try:
        data_out = model(data_in)
    except Exception as e:
        pytest.fail(f"Forward pass failed for config '{config['name']}': {e}")

    # 4. Check output shapes
    assert out_field in data_out, f"Output field '{out_field}' not found in output data."
    output_tensor = data_out[out_field]
    expected_shape = (NUM_EDGES, model.irreps_out[out_field].dim)
    assert output_tensor.shape == expected_shape, \
        f"Output shape mismatch for '{config['name']}'. Expected {expected_shape}, got {output_tensor.shape}"

    # 5. Test Equivariance
    try:
        assert_AtomicData_equivariant(
            func=model,
            data_in=data_in,
        )
    except AssertionError as e:
        pytest.fail(f"Equivariance test failed for config '{config['name']}': {e}")


@pytest.mark.parametrize("config", [TEST_CONFIGS[0], SCALAR_ONLY_OUTPUT_CONFIG], ids=["base_case", "scalar_only_output"])
def test_interaction_module_deployable(config, tmp_path):
    """Smoke-test that InteractionModule can be scripted/frozen/saved and reloaded."""
    device = "cpu"
    params = config["params"]
    irreps_in = config["irreps_in"]  # type: ignore

    model = InteractionModule(irreps_in=irreps_in, **params).to(device)
    data_in, _ = _create_dummy_data(config, device)

    assert_module_deployable(model, (data_in,), tmp_path=tmp_path)


def test_fixed_point_recycling_stops_before_max_iters_with_contractive_map(monkeypatch):
    """The fixed-point solver should early-stop when the map is contractive."""
    cfg = {
        "num_layers": 2,
        "latent_dim": 16,
        "eq_latent_multiplicity": 4,
        "use_fixed_point_recycling": True,
        "fp_max_iter": 20,
        "fp_tol": 1e-3,
        "fp_alpha": 0.5,
        "fp_grad_steps": 0,
        "fp_enforce_inverse_quadratic": True,
    }
    model = InteractionModule(irreps_in=BASE_IRREPS_IN, **cfg).eval()

    def fake_stack(
        data,
        scalar_latent,
        equiv_latent,
        active_edges,
        node_conditioning,
        edge_conditioning,
        layer_update_coefficients,
    ):
        if equiv_latent is None:
            return scalar_latent * 0.5, None
        return scalar_latent * 0.5, equiv_latent * 0.5

    monkeypatch.setattr(model, "_run_interaction_stack", fake_stack)

    n_edges = 32
    scalar_latent = torch.randn(n_edges, cfg["latent_dim"])
    equiv_latent = torch.randn(n_edges, cfg["eq_latent_multiplicity"], 9)
    active_edges = torch.arange(n_edges, dtype=torch.long)
    layer_coeffs = torch.zeros(cfg["num_layers"] - 1)

    _, _, iters_used, last_residual = model._fixed_point_refine(
        data={},
        scalar_latent=scalar_latent,
        equiv_latent=equiv_latent,
        active_edges=active_edges,
        node_conditioning=None,
        edge_conditioning=None,
        layer_update_coefficients=layer_coeffs,
    )

    assert iters_used < cfg["fp_max_iter"]
    assert float(last_residual) < cfg["fp_tol"]


def test_fixed_point_static_context_fuses_dynamic_and_static():
    cfg = {
        "num_layers": 2,
        "latent_dim": 8,
        "eq_latent_multiplicity": 2,
        "use_fixed_point_recycling": True,
        "fp_use_static_context": True,
        "fp_static_context_strength": 1.0,
        "fp_grad_steps": 0,
    }
    model = InteractionModule(irreps_in=BASE_IRREPS_IN, **cfg).eval()

    class _TakeStaticScalar(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return x[:, self.dim:]

    class _TakeStaticEquiv(nn.Module):
        def forward(self, x):
            feat = x.shape[1] // 2
            return x[:, feat:]

    model.fp_scalar_static_fuser = _TakeStaticScalar(cfg["latent_dim"])
    model.fp_equiv_static_fuser = _TakeStaticEquiv()

    n_edges = 7
    dyn_s = torch.randn(n_edges, cfg["latent_dim"])
    stat_s = torch.randn(n_edges, cfg["latent_dim"])
    dyn_e = torch.randn(n_edges, cfg["eq_latent_multiplicity"], 9)
    stat_e = torch.randn(n_edges, cfg["eq_latent_multiplicity"], 9)

    fused_s, fused_e = model._fuse_fixed_point_inputs(
        dynamic_scalar_latent=dyn_s,
        dynamic_equiv_latent=dyn_e,
        static_scalar_latent=stat_s,
        static_equiv_latent=stat_e,
    )

    assert torch.allclose(fused_s, stat_s)
    assert fused_e is not None
    assert torch.allclose(fused_e, stat_e)


def test_inverse_quadratic_alpha_cap_enforces_bound():
    raw = torch.tensor(10.0)
    alpha_max = 0.5
    alpha = torch.tensor(alpha_max)
    base = raw * alpha  # residual at iter 0

    residuals = []
    for it in range(10):
        alpha_it, target = _inverse_quadratic_alpha_cap(
            raw_residual=raw,
            alpha=alpha,
            base_residual=base,
            it=it,
            alpha_max=alpha_max,
        )
        residual = raw * alpha_it
        residuals.append(float(residual))
        assert float(residual) <= float(target) + 1e-6

    # sequence should be non-increasing under the cap
    assert all(residuals[i + 1] <= residuals[i] + 1e-12 for i in range(len(residuals) - 1))
