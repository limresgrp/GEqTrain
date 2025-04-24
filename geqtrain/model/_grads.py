import logging
from geqtrain.nn import (
    SequentialGraphNetwork,
    GraphModuleMixin,
    SetRequireGradsOutput,
    GradientOutput,
    PartialForceOutput as PartialForceOutputModule,
    StressOutput as StressOutputModule,
)
from geqtrain.data import AtomicDataDict
from geqtrain.utils.config import Config


def GradOutput(model, config: Config) -> SequentialGraphNetwork:
    r"""Add forces to a model that predicts energy.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

    Returns:
        A ``GradientOutput`` wrapping ``model``.
    """

    logging.info("--- Building GradOutput Module ---")

    of_key = config.get("grad_output_of_field")
    wrt_key = config.get("grad_output_wrt_field")
    out_field = config.get("grad_output_out_field")
    sign = int(config.get("grad_output_sign", -1))
    if out_field in model.irreps_out:
        raise ValueError(f"This model already has {out_field} outputs.")

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

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )