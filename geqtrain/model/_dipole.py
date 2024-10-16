from geqtrain.nn import GraphModuleMixin, DipoleMomentModule
from geqtrain.data import AtomicDataDict


def DipoleMoment(model: GraphModuleMixin) -> DipoleMomentModule:
    r"""Compute dipole moment.

    Args:
        model: the energy model to wrap. Must have ``AtomicDataDict.NODE_OUTPUT_KEY`` as an output.

    Returns:
        A ``DipoleMomentModule`` wrapping ``model``.
    """

    return DipoleMomentModule(
        func=model,
        field=AtomicDataDict.NODE_OUTPUT_KEY,
        out_field=AtomicDataDict.GRAPH_OUTPUT_KEY,
    )