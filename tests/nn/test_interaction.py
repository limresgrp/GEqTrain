import pytest
import torch
from tests.utils.deployability import assert_module_deployable
import copy
from e3nn import o3

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.data.dataset import _NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS, _FIXED_FIELDS
from geqtrain.nn import InteractionModule
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
            "head_dim": 4,
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
