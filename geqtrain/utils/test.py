""" Adapted from https://github.com/mir-group/nequip
"""

from typing import Union, Optional, List

import torch
from e3nn import o3
from e3nn.util.test import equivariance_error, FLOAT_TOLERANCE

from geqtrain.nn import GraphModuleMixin
from geqtrain.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
)


PERMUTATION_FLOAT_TOLERANCE = {torch.float32: 1e-4, torch.float64: 1e-8}


def _inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def assert_permutation_equivariant(
    func: GraphModuleMixin,
    data_in: AtomicDataDict.Type,
    tolerance: Optional[float] = None,
    raise_error: bool = True,
) -> str:
    r"""Test the permutation equivariance of ``func``.

    Standard fields are assumed to be equivariant to node or edge permutations according to their standard interpretions; all other fields are assumed to be invariant to all permutations. Non-standard fields can be registered as node/edge permutation equivariant using ``register_fields``.

    Raises ``AssertionError`` if issues are found.

    Args:
        func: the module or model to test
        data_in: the example input data to test with
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    if tolerance is None:
        atol = PERMUTATION_FLOAT_TOLERANCE[torch.float32]
    else:
        atol = tolerance

    data_in = data_in.copy()
    device = data_in[AtomicDataDict.POSITIONS_KEY].device

    node_permute_fields = _NODE_FIELDS
    edge_permute_fields = _EDGE_FIELDS

    # Make permutations and make sure they are not identities
    n_node: int = len(data_in[AtomicDataDict.POSITIONS_KEY])
    while True:
        node_perm = torch.randperm(n_node, device=device)
        if n_node <= 1:
            break  # otherwise inf loop
        if not torch.all(node_perm == torch.arange(n_node, device=device)):
            break
    n_edge: int = data_in[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    while True:
        edge_perm = torch.randperm(n_edge, device=device)
        if n_edge <= 1:
            break  # otherwise inf loop
        if not torch.all(edge_perm == torch.arange(n_edge, device=device)):
            break
    # ^ note that these permutations are maps from the "to" index to the "from" index
    # because we index by them, the 0th element of the permuted array will be the ith
    # of the original array, where i = perm[0]. Thus i is "from" and 0 is to, so perm
    # interpreted as a map is a map from "to" to "from".

    perm_data_in = {}
    for k in data_in.keys():
        if k in node_permute_fields:
            perm_data_in[k] = data_in[k][node_perm]
        elif k in edge_permute_fields:
            perm_data_in[k] = data_in[k][edge_perm]
        else:
            perm_data_in[k] = data_in[k]

    perm_data_in[AtomicDataDict.EDGE_INDEX_KEY] = _inverse_permutation(node_perm)[
        data_in[AtomicDataDict.EDGE_INDEX_KEY]
    ][:, edge_perm]

    out_orig = func(data_in)
    out_perm = func(perm_data_in)

    assert set(out_orig.keys()) == set(
        out_perm.keys()
    ), "Permutation changed the set of fields returned by model"

    messages = []
    num_problems: int = 0
    for k in out_orig.keys():
        if k in node_permute_fields:
            if out_orig[k].dtype == torch.bool:
                err = (out_orig[k][node_perm] != out_perm[k]).max()
            else:
                err = (out_orig[k][node_perm] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k][node_perm], out_perm[k], atol=atol)
            if fail:
                num_problems += 1
            messages.append(
                f"   node permutation equivariance of field {k:20}       -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
        elif k in edge_permute_fields:
            err = (out_orig[k][edge_perm] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k][edge_perm], out_perm[k], atol=atol)
            if fail:
                num_problems += 1
            messages.append(
                f"   edge permutation equivariance of field {k:20}       -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
        elif k == AtomicDataDict.EDGE_INDEX_KEY:
            pass
        else:
            # Assume invariant
            if out_orig[k].dtype == torch.bool:
                err = (out_orig[k] != out_perm[k]).max()
            else:
                err = (torch.nan_to_num(out_orig[k]) - torch.nan_to_num(out_perm[k])).abs().max()
            fail = not torch.allclose(torch.nan_to_num(out_orig[k]), torch.nan_to_num(out_perm[k]), atol=atol)
            if fail:
                num_problems += 1
            messages.append(
                f"   edge & node permutation invariance for field {k:20} -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
    msg = "\n".join(messages)
    if num_problems == 0:
        return msg
    else:
        if raise_error:
            raise AssertionError(msg)
        else:
            return msg


def assert_AtomicData_equivariant(
    func: GraphModuleMixin,
    data_in: Union[
        AtomicData, AtomicDataDict.Type, List[Union[AtomicData, AtomicDataDict.Type]]
    ],
    cartesian_points_fields: List[str] = [],
    permutation_tolerance: Optional[float] = None,
    o3_tolerance: Optional[float] = None,
    **kwargs,
) -> str:
    r"""Test the rotation, translation, parity, and permutation equivariance of ``func``.

    For details on permutation testing, see ``assert_permutation_equivariant``.
    For details on geometric equivariance testing, see ``e3nn.util.test.assert_equivariant``.

    Raises ``AssertionError`` if issues are found.

    Args:
        func: the module or model to test
        data_in: the example input data(s) to test with. Only the first is used for permutation testing.
        **kwargs: passed to ``e3nn.util.test.assert_equivariant``

    Returns:
        A string description of the errors.
    """
    # Prevent pytest from showing this function in the traceback
    __tracebackhide__ = True

    device = next(func.parameters()).device
    if not isinstance(data_in, list):
        data_in = [data_in]
    data_in = [AtomicData.to_AtomicDataDict(d.to(device)) for d in data_in]

    # == Test permutation of graph nodes ==
    # since permutation is discrete and should not be data dependent, run only on one frame.
    permutation_message = assert_permutation_equivariant(
        func, data_in[0], tolerance=permutation_tolerance, raise_error=False
    )

    # == Test rotation, parity, and translation using e3nn ==
    irreps_in = {k: None for k in AtomicDataDict.ALLOWED_KEYS}
    irreps_in.update(func.irreps_in)
    irreps_in = {k: v for k, v in irreps_in.items() if k in data_in[0]}
    irreps_out = func.irreps_out.copy()
    # for certain things, we don't care what the given irreps are...
    # make sure that we test correctly for equivariance:
    cartesian_points_fields.extend([AtomicDataDict.POSITIONS_KEY])
    for irps in (irreps_in, irreps_out):
        for cartesian_points_field in cartesian_points_fields:
            if cartesian_points_field in irps:
                # it should always have been 1o vectors
                # since that's actually a valid Irreps
                assert o3.Irreps(irps[cartesian_points_field]) == o3.Irreps("1o")
                irps[cartesian_points_field] = "cartesian_points"

    def wrapper(*args):
        arg_dict = {k: v for k, v in zip(irreps_in, args)}
        output = func(arg_dict)
        return [output[k] for k in irreps_out]

    errs = [
        equivariance_error(
            wrapper,
            args_in=[d[k] for k in irreps_in],
            irreps_in=list(irreps_in.values()),
            irreps_out=list(irreps_out.values()),
            **kwargs,
        )
        for d in data_in
    ]

    # take max across errors
    errs = {k: torch.max(torch.vstack([e[k] for e in errs]), dim=0)[0] for k in errs[0]}

    if o3_tolerance is None:
        o3_tolerance = FLOAT_TOLERANCE[torch.float32]
    all_errs = []
    for case, err in errs.items():
        for key, this_err in zip(irreps_out.keys(), err):
            all_errs.append(case + (key, this_err))
    is_problem = [e[-1] > o3_tolerance for e in all_errs]

    message = (permutation_message + "\n") + "\n".join(
        "   (parity_k={:1d}, did_translate={:5}, field={:20})     -> max error={:.3e}".format(
            int(k[0]), str(bool(k[1])), str(k[2]), float(k[3])
        )
        for k, prob in zip(all_errs, is_problem)
    )

    if sum(is_problem) > 0 or "FAIL" in permutation_message:
        raise AssertionError(f"Equivariance test failed for cases:\n{message}")

    return message