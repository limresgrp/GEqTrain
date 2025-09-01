# geqtrain/nn/_gradients.py

import logging
from geqtrain.nn import (
    SequentialGraphNetwork,
    EnableGradients,
    ComputeGradient,
)
from geqtrain.utils.config import Config

def WithGradients(model, config: Config) -> SequentialGraphNetwork:
    """
    Wraps a model to add a gradient output field (e.g., forces from energy).
    """
    logging.info("--- Building a model wrapper with gradient outputs ---")

    # Update Config to inform Trainer that we need gradients also during Validation phase
    config.update({"model_requires_grads": True})
    
    # Retrieve configuration with new, clearer names
    gradient_of = config.get("gradient_of_field") # formerly "grad_output_of_field"
    gradient_wrt = config.get("gradient_wrt_field")  # formerly "grad_output_wrt_field"
    out_field = config.get("gradient_out_field")     # formerly "grad_output_out_field"
    sign = int(config.get("gradient_sign", -1))
    scales = config.get("gradient_scales", None)   # <-- New optional parameter

    if out_field in model.irreps_out:
        raise ValueError(f"This model already has an output field named '{out_field}'.")

    # Define the sequence of modules
    layers = {
        "enable_gradients": (EnableGradients, dict(
            gradient_of=gradient_of,
            gradient_wrt=gradient_wrt,
        )),
        "core_model": model,
        "compute_gradient": (ComputeGradient, dict(
            gradient_of=gradient_of,
            gradient_wrt=gradient_wrt,
            out_field=out_field,
            sign=sign,
            scales=scales, # Pass scales to the constructor
        )),
    }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )