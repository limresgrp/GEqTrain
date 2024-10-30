from geqtrain.nn import GraphModuleMixin, PerTypeScaleModule
from geqtrain.data import AtomicDataDict


def PerTypeScale(model: GraphModuleMixin, config) -> PerTypeScaleModule:
    r"""
    Args:
        model: the model to wrap. Must have ``AtomicDataDict.NODE_OUTPUT_KEY`` as an output.

    Returns:
        A ``PerTypeScaleModule`` wrapping ``model``.
    """

    return PerTypeScaleModule(
        func=model,
        field=AtomicDataDict.NODE_OUTPUT_KEY,
        out_field=AtomicDataDict.NODE_OUTPUT_KEY,
        num_types=config.get('num_types', 0),
        per_type_bias=config.get('per_type_bias', None),
        per_type_std=config.get('per_type_std', None),
    )
