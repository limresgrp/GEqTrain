import pytest
import torch
from e3nn import o3
# We import the lightweight test utility from e3nn directly
from e3nn.util.test import assert_equivariant, FLOAT_TOLERANCE
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.nn import EquivariantScalarMLP
from geqtrain.utils.deploy_test import assert_module_deployable

# Define a list of test configurations. Each dictionary represents one test case.
# This makes it easy to add new test cases in the future.
BASE_PARAMS = {"latent_kwargs": {"mlp_latent_dimensions": [32]}}
TEST_CONFIGS = [ # batch_size = 10
    # --- Basic Cases: Single In/Out, Flat I/O ---
    {
        "name": "Single In/Out, Scalars Only, Flat",
        "params": {"in_irreps": "4x0e", "out_irreps": "8x0e", "output_shape_spec": "flat"},
        "expected_output_shape": (10, 8),
        "output_mode": "single",
    },
    {
        "name": "Single In/Out, Equivariants Only, Flat",
        "params": {"in_irreps": "2x1o", "out_irreps": "3x1o", "output_shape_spec": "flat"},
        "expected_output_shape": (10, 3 * 3),
        "output_mode": "single",
    },
    {
        "name": "Single In/Out, Mixed, Flat",
        "params": {"in_irreps": "4x0e+2x1o", "out_irreps": "8x0e+3x1o", "output_shape_spec": "flat"},
        "expected_output_shape": (10, 8 + 3 * 3),
        "output_mode": "single",
    },
    # --- Split Output Cases ---
    {
        "name": "Single In, Split Out, Mixed, Flat",
        "params": {"in_irreps": "4x0e+2x1o", "out_irreps": ("8x0e", "3x1o"), "output_shape_spec": "flat"},
        "expected_output_shape": ((10, 8), (10, 3 * 3)),
        "output_mode": "split",
    },
    # --- Split Input Cases ---
    {
        "name": "Split In, Single Out, Mixed, Flat",
        "params": {"in_irreps": ("4x0e", "2x1o"), "out_irreps": "8x0e+3x1o", "output_shape_spec": "flat"},
        "expected_output_shape": (10, 8 + 3 * 3),
        "output_mode": "single",
        "input_mode": "split",
    },
    {
        "name": "Split In/Out, Mixed, Flat",
        "params": {"in_irreps": ("4x0e", "2x1o"), "out_irreps": ("8x0e", "3x1o"), "output_shape_spec": "flat"},
        "expected_output_shape": ((10, 8), (10, 3 * 3)),
        "output_mode": "split",
        "input_mode": "split",
    },
    # --- Channel-wise Output Cases ---
    {
        "name": "Single In/Out, Mixed, Channel-wise Output",
        "params": {"in_irreps": "4x0e+2x1o", "out_irreps": "8x0e+8x1o", "output_shape_spec": "channel_wise"},
        "expected_output_shape": (10, 8, 1 + 3),
        "output_mode": "single",
    },
    # --- Channel-wise Input Cases & output_shape_spec='input' ---
    {
        "name": "Single In/Out, Channel-wise Input, output_shape_spec='input'",
        "params": {"in_irreps": "4x0e+4x1o", "out_irreps": "8x0e+8x1o", "output_shape_spec": "input"},
        "expected_output_shape": (10, 8, 1 + 3),
        "output_mode": "single",
        "input_mode": "channel_wise",
    },
    {
        "name": "Flat Input, output_shape_spec='input'",
        "params": {"in_irreps": "4x0e+2x1o", "out_irreps": "8x0e+3x1o", "output_shape_spec": "input"},
        "expected_output_shape": (10, 8 + 3 * 3),
        "output_mode": "single",
    },
    # --- Conditioning Cases ---
    {
        "name": "Single In/Out, With Conditioning",
        "params": {"in_irreps": "4x0e+2x1o", "out_irreps": "8x0e+3x1o", "output_shape_spec": "flat", "conditioning_dim": 16},
        "expected_output_shape": (10, 8 + 3 * 3),
        "output_mode": "single",
        "conditioning_dim": 16,
    },
    {
        "name": "Equivariant Only In, With Conditioning (Weights from conditioning)",
        "params": {"in_irreps": "2x1o", "out_irreps": "3x1o", "output_shape_spec": "flat", "conditioning_dim": 16},
        "expected_output_shape": (10, 3 * 3),
        "output_mode": "single",
        "conditioning_dim": 16,
    },
]

@pytest.fixture(scope="session", autouse=True, params=["float32"])
def float_tolerance(request):
    """
    Run all tests with various PyTorch default dtypes.
    This is a session-wide, autouse fixture.
    """
    old_dtype = torch.get_default_dtype()
    dtype = {"float32": torch.float32, "float64": torch.float64}[request.param]
    torch.set_default_dtype(dtype)
    yield FLOAT_TOLERANCE[dtype]
    torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("config", TEST_CONFIGS, ids=[c["name"] for c in TEST_CONFIGS])
def test_equivariant_scalar_mlp_behavior(config, float_tolerance):
    """
    Tests the EquivariantScalarMLP for shape correctness and equivariance
    using the lightweight e3nn.utils.test.assert_equivariant.
    """
    params = config["params"]
    params.update(BASE_PARAMS)
    batch_size = 10 # Fixed batch size for tests
    input_mode = config.get("input_mode", "single")
    output_mode = config.get("output_mode", "single")

    # == 1. Test Instantiation ==
    try:
        model = EquivariantScalarMLP(**params)
    except Exception as e:
        pytest.fail(f"Failed to instantiate model for config '{config['name']}': {e}") # type: ignore

    # == 2. Define Wrapper, Irreps, and Input Tensors ==
    # We need to define a function for assert_equivariant
    # that takes *only* tensors as input, corresponding to irreps_in.
    # We create a lambda to wrap the model's forward pass.

    model_func_for_assert_equivariant = None # This will be the lambda that assert_equivariant calls
    irreps_in_list = [] # List of o3.Irreps for input arguments
    args_in = [] # List of example torch.Tensors for shape check
    
    has_conditioning = "conditioning_dim" in config
    conditioning_irreps = o3.Irreps(f"{config['conditioning_dim']}x0e") if has_conditioning else None
    
    # Define input reshaper if needed (for channel_wise input)
    input_reshaper = None
    if input_mode == "channel_wise":
        features_irreps = o3.Irreps(params["in_irreps"])
        input_reshaper = reshape_irreps(features_irreps)

    # Prepare irreps_in_list and args_in for the initial shape check
    if input_mode == "split":
        scalar_irreps = o3.Irreps(params["in_irreps"][0]) # type: ignore
        equiv_irreps = o3.Irreps(params["in_irreps"][1]) # type: ignore
        irreps_in_list = [scalar_irreps, equiv_irreps] # type: ignore
        args_in = [
            scalar_irreps.randn(batch_size, -1, dtype=torch.get_default_dtype()), # type: ignore
            equiv_irreps.randn(batch_size, -1, dtype=torch.get_default_dtype()) # type: ignore
        ]
    else: # input_mode == "single" or "channel_wise"
        features_irreps = o3.Irreps(params["in_irreps"]) # type: ignore
        irreps_in_list = [features_irreps] # type: ignore
        args_in = [features_irreps.randn(batch_size, -1, dtype=torch.get_default_dtype())] # type: ignore

    # Add conditioning to irreps_in_list and args_in if present
    if has_conditioning:
        irreps_in_list.append(conditioning_irreps) # type: ignore
        args_in.append(conditioning_irreps.randn(batch_size, -1, dtype=torch.get_default_dtype())) # type: ignore

    # Final model_func wrapper for assert_equivariant
    def model_func( *args_from_equiv):
        # Unpack args_from_equiv based on whether conditioning is present
        if has_conditioning:
            conditioning_tensor = args_from_equiv[-1]
            features_args = args_from_equiv[:-1]
        else:
            conditioning_tensor = None
            features_args = args_from_equiv

        # Prepare features for the model based on input_mode
        if input_mode == "split":
            # features_args is (scalar_tensor, equiv_tensor)
            features_for_model = (features_args[0], features_args[1])
        else: # single or channel_wise
            # features_args is (single_feature_tensor,)
            features_for_model = features_args[0]
            if input_mode == "channel_wise":
                # Reshape the flat tensor from assert_equivariant to channel-wise
                assert input_reshaper is not None
                features_for_model = input_reshaper(features_for_model)

        return model(features_for_model, conditioning_tensor)

    # Define output irreps
    # This list will be used for assert_equivariant. If the model outputs channel-wise,
    # we will flatten it for the equivariance test, so irreps_out_list should reflect the
    # flattened irreps.
    if output_mode == "split":
        irreps_out_for_equivariance = [o3.Irreps(params["out_irreps"][0]), o3.Irreps(params["out_irreps"][1])]
    else: # single # type: ignore
        irreps_out_for_equivariance = [o3.Irreps(params["out_irreps"])]

    # == 3. Test Forward Pass and Output Shape ==
    # We use our generated args_in to test the forward pass and shape
    try:
        # Call the model_func with args_in to get the actual output, before any flattening for equivariance test.
        # model_func already handles reshaping input from flat (for assert_equivariant) to channel-wise
        # if input_mode is "channel_wise".
        output = model_func(*args_in)
    except Exception as e:
        pytest.fail(f"Forward pass failed for config '{config['name']}': {e}")

    if output_mode == "split":
        assert isinstance(output, tuple), "Expected a tuple for split output mode"
        assert len(output) == 2, "Expected a tuple of length 2 for split output mode"
        assert output[0].shape == config["expected_output_shape"][0], f"Scalar output shape mismatch for {config['name']}"
        assert output[1].shape == config["expected_output_shape"][1], f"Equivariant output shape mismatch"
    else: # single output
        assert isinstance(output, torch.Tensor), f"Expected a single tensor for single output mode, but got {type(output)}"
        assert output.shape == config["expected_output_shape"], f"Output shape mismatch for {config['name']}. Expected {config['expected_output_shape']}, got {output.shape}"

    # == 4. Prepare model_func for Equivariance Test (flatten if necessary) ==
    # Determine if the model's output (before test-specific flattening) is channel-wise
    is_model_output_channel_wise = (
        params.get("output_shape_spec") == "channel_wise" or
        (params.get("output_shape_spec") == "input" and input_mode == "channel_wise")
    )

    model_func_for_equivariance = model_func
    if is_model_output_channel_wise:
        if output_mode == "single":
            # If the model's single output is channel-wise, flatten the entire tensor.
            out_irreps_for_flattening = o3.Irreps(params["out_irreps"]) # type: ignore
            reshaper = inverse_reshape_irreps(out_irreps_for_flattening)
            model_func_for_equivariance = lambda *args: reshaper(model_func(*args))
        elif output_mode == "split":
            # If the model's split output has a channel-wise equivariant part, flatten only that part. # type: ignore
            # params["out_irreps"] is a tuple (scalar_irreps_str, equiv_irreps_str)
            equiv_irreps_for_flattening = o3.Irreps(params["out_irreps"][1])
            reshaper = inverse_reshape_irreps(equiv_irreps_for_flattening)
            original_model_func_ref = model_func # Capture the original lambda
            model_func_for_equivariance = lambda *args: (lambda s, e: (s, reshaper(e)))(*original_model_func_ref(*args))

    # == 4. Test Equivariance ==
    # Now we use the e3nn utility. It will generate its own random data.
    try:
        # We pass batch_size to be used by the internal _rand_args
        # We also pass the tolerance from the fixture
        assert_equivariant( # type: ignore
            model_func_for_equivariance,
            irreps_in=irreps_in_list, 
            irreps_out=irreps_out_for_equivariance,
            tolerance=float_tolerance,
        )
    except AssertionError as e:
        pytest.fail(f"Equivariance test failed for config '{config['name']}': {e}")


def test_equivariant_scalar_mlp_deployable(tmp_path, float_tolerance):
    """Ensure a representative EquivariantScalarMLP can be scripted/frozen and reloaded."""
    model = EquivariantScalarMLP(
        in_irreps="2x1o",
        out_irreps="3x1o",
        conditioning_dim=16,
        output_shape_spec="flat",
        latent_kwargs=BASE_PARAMS["latent_kwargs"],
    )
    features_irreps = o3.Irreps("2x1o")
    features = features_irreps.randn(4, -1, dtype=torch.get_default_dtype())
    conditioning = torch.randn(4, 16, dtype=torch.get_default_dtype())

    assert_module_deployable(model, (features, conditioning), tmp_path=tmp_path)
