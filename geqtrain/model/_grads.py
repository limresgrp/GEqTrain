import logging
from geqtrain.nn import (
    SequentialGraphNetwork,
    SetRequireGradsOutput,
    GradientOutput,
)
from geqtrain.utils.config import Config


def GradOutput(model, config: Config) -> SequentialGraphNetwork:
    r"""
    Wraps a model that predicts energy to include gradient (force) computation.

    This function adds a gradient computation module to a given energy model.
    The gradient is computed with respect to a specified input field and added
    as an additional output field in the model.

    Args:
        model: The energy model to wrap. It must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.
        config: A configuration object containing the following keys:
            - "grad_output_of_field": The field to compute gradients of (e.g., energy).
            - "grad_output_wrt_field": The field with respect to which gradients are computed (e.g., positions).
            - "grad_output_out_field": The name of the output field for the computed gradients.
            - "grad_output_sign": The sign of the gradient (default is -1).

    Returns:
        A ``SequentialGraphNetwork`` wrapping the input model with gradient computation added.

    Raises:
        ValueError: If the specified output field for gradients already exists in the model.
    """

    logging.info("--- Building GradOutput Module ---")

    # Retrieve configuration parameters
    of_key    = config.get("grad_output_of_field")
    wrt_key   = config.get("grad_output_wrt_field")
    out_field = config.get("grad_output_out_field")
    sign = int(config.get("grad_output_sign", -1))

    # Check if the output field already exists in the model
    if out_field in model.irreps_out:
        raise ValueError(f"This model already has {out_field} outputs.")

    # Define the layers for the SequentialGraphNetwork
    layers = {
        "set_require_grads": SetRequireGradsOutput(
            of=of_key,
            wrt=wrt_key,
        ),
        "wrapped_model": model,
        "grads": (GradientOutput, dict(
            of=of_key,
            wrt=wrt_key,
            out_field=out_field,
            sign=sign,
        )),
    }

    # Create and return the SequentialGraphNetwork
    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )