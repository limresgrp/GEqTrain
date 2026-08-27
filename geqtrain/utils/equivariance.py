"""Equivariance test utilities used by the public ``geqtrain-test`` scripts.

Adapted from https://github.com/mir-group/nequip.
"""

from typing import Any, Dict, List, Optional, Union

import torch
from e3nn import o3
from e3nn.util.test import FLOAT_TOLERANCE, equivariance_error

from geqtrain.data import AtomicData, AtomicDataDict, _EDGE_FIELDS, _NODE_FIELDS
from geqtrain.nn import GraphModuleMixin


PERMUTATION_FLOAT_TOLERANCE = {torch.float32: 1e-4, torch.float64: 1e-8}


def _inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


def _classify_field_axis(
    key: str,
    value: Any,
    num_nodes: int,
    num_edges: int,
    node_fields: set,
    edge_fields: set,
) -> str:
    if key in node_fields:
        return "node"
    if key in edge_fields:
        return "edge"
    if key == AtomicDataDict.EDGE_INDEX_KEY:
        return "edge_index"
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return "other"
    if value.shape[0] == num_nodes and value.shape[0] != num_edges:
        return "node"
    if value.shape[0] == num_edges and value.shape[0] != num_nodes:
        return "edge"
    return "other"


def _infer_permutation_fields(
    func: GraphModuleMixin,
    data_dict: AtomicDataDict.Type,
    num_nodes: int,
    num_edges: int,
):
    node_fields = set(_NODE_FIELDS)
    edge_fields = set(_EDGE_FIELDS)

    declared_irreps: Dict[str, Any] = {}
    declared_irreps.update(getattr(func, "irreps_in", {}))
    declared_irreps.update(getattr(func, "irreps_out", {}))

    for key, irrep in declared_irreps.items():
        if key not in data_dict or irrep is None:
            continue
        axis = _classify_field_axis(
            key=key,
            value=data_dict[key],
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_fields=node_fields,
            edge_fields=edge_fields,
        )
        if axis == "node":
            node_fields.add(key)
        elif axis == "edge":
            edge_fields.add(key)

    return node_fields, edge_fields


def assert_permutation_equivariant(
    func: GraphModuleMixin,
    data_in: AtomicDataDict.Type,
    tolerance: Optional[float] = None,
    raise_error: bool = True,
) -> str:
    r"""Test permutation equivariance of ``func`` on one atomic graph."""
    print("TESTING PERMUTATION EQUIVARIANCE")
    __tracebackhide__ = True

    atol = PERMUTATION_FLOAT_TOLERANCE[torch.float32] if tolerance is None else tolerance
    data_in = data_in.copy()
    device = data_in[AtomicDataDict.POSITIONS_KEY].device

    num_nodes = len(data_in[AtomicDataDict.POSITIONS_KEY])
    num_edges = data_in[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
    node_permute_fields, edge_permute_fields = _infer_permutation_fields(
        func=func,
        data_dict=data_in,
        num_nodes=num_nodes,
        num_edges=num_edges,
    )

    while True:
        node_perm = torch.randperm(num_nodes, device=device)
        if num_nodes <= 1 or not torch.all(node_perm == torch.arange(num_nodes, device=device)):
            break
    while True:
        edge_perm = torch.randperm(num_edges, device=device)
        if num_edges <= 1 or not torch.all(edge_perm == torch.arange(num_edges, device=device)):
            break

    perm_data_in = {}
    for k in data_in.keys():
        axis = _classify_field_axis(
            key=k,
            value=data_in[k],
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_fields=node_permute_fields,
            edge_fields=edge_permute_fields,
        )
        if axis == "node":
            perm_data_in[k] = data_in[k][node_perm]
        elif axis == "edge":
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
    num_problems = 0
    for k in out_orig.keys():
        axis = _classify_field_axis(
            key=k,
            value=out_orig[k],
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_fields=node_permute_fields,
            edge_fields=edge_permute_fields,
        )
        if axis == "node":
            if out_orig[k].dtype == torch.bool:
                err = (out_orig[k][node_perm] != out_perm[k]).max()
            else:
                err = (out_orig[k][node_perm] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k][node_perm], out_perm[k], atol=atol)
            num_problems += int(fail)
            messages.append(
                f"   node permutation equivariance of field {k:20}       -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
        elif axis == "edge":
            err = (out_orig[k][edge_perm] - out_perm[k]).abs().max()
            fail = not torch.allclose(out_orig[k][edge_perm], out_perm[k], atol=atol)
            num_problems += int(fail)
            messages.append(
                f"   edge permutation equivariance of field {k:20}       -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )
        elif k != AtomicDataDict.EDGE_INDEX_KEY:
            if out_orig[k].dtype == torch.bool:
                err = (out_orig[k] != out_perm[k]).max()
            else:
                err = (torch.nan_to_num(out_orig[k]) - torch.nan_to_num(out_perm[k])).abs().max()
            fail = not torch.allclose(torch.nan_to_num(out_orig[k]), torch.nan_to_num(out_perm[k]), atol=atol)
            num_problems += int(fail)
            messages.append(
                f"   edge & node permutation invariance for field {k:20} -> max error={err:.3e}{'  FAIL' if fail else ''}"
            )

    msg = "\n".join(messages)
    if num_problems > 0 and raise_error:
        raise AssertionError(msg)
    return msg


def assert_AtomicData_equivariant(
    func: GraphModuleMixin,
    data_in: Union[
        AtomicData, AtomicDataDict.Type, List[Union[AtomicData, AtomicDataDict.Type]]
    ],
    cartesian_points_fields: List[str] = [],
    permutation_tolerance: Optional[float] = None,
    o3_tolerance: Optional[float] = None,
    input_irreps_overrides: Optional[Dict[str, Union[str, o3.Irreps]]] = None,
    output_irreps_overrides: Optional[Dict[str, Union[str, o3.Irreps]]] = None,
    **kwargs,
) -> str:
    r"""Test rotation, translation, parity, and permutation equivariance of ``func``."""
    __tracebackhide__ = True

    try:
        device = next(func.parameters()).device
    except StopIteration:
        device = None

    if not isinstance(data_in, list):
        data_in = [data_in]

    processed_data_in = []
    for d in data_in:
        if isinstance(d, dict):
            d = AtomicData.from_dict(d)
        target_device = d[AtomicDataDict.POSITIONS_KEY].device if device is None else device
        processed_data_in.append(AtomicData.to_AtomicDataDict(d.to(target_device)))
    data_in = processed_data_in

    permutation_message = assert_permutation_equivariant(
        func, data_in[0], tolerance=permutation_tolerance, raise_error=False
    )

    irreps_in = {k: None for k in AtomicDataDict.ALLOWED_KEYS}
    irreps_in.update(func.irreps_in)
    irreps_in.update({"atom_rows": None, "atom_cols": None})
    for k in data_in[0].keys():
        if k not in irreps_in:
            irreps_in[k] = None
    if input_irreps_overrides is not None:
        irreps_in.update(AtomicDataDict._fix_irreps_dict(input_irreps_overrides))
    irreps_in = {k: v for k, v in irreps_in.items() if k in data_in[0]}

    irreps_out = func.irreps_out.copy()
    if output_irreps_overrides is not None:
        irreps_out.update(AtomicDataDict._fix_irreps_dict(output_irreps_overrides))

    _cartesian_points_fields = list(cartesian_points_fields)
    if AtomicDataDict.POSITIONS_KEY not in _cartesian_points_fields:
        _cartesian_points_fields.append(AtomicDataDict.POSITIONS_KEY)
    for irps in (irreps_in, irreps_out):
        for cartesian_points_field in _cartesian_points_fields:
            if cartesian_points_field in irps:
                assert o3.Irreps(irps[cartesian_points_field]) == o3.Irreps("1o")
                irps[cartesian_points_field] = "cartesian_points"

    def wrapper(*args):
        arg_dict = {k: v for k, v in zip(irreps_in, args)}
        output = func(arg_dict)
        return [output[k] for k in irreps_out]

    print("TESTING ROTATION, PARITY, AND TRANSLATION")
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

    errs = {k: torch.max(torch.vstack([e[k] for e in errs]), dim=0)[0] for k in errs[0]}
    if o3_tolerance is None:
        o3_tolerance = FLOAT_TOLERANCE[torch.float32]

    all_errs = []
    for case, err in errs.items():
        for key, this_err in zip(irreps_out.keys(), err):
            all_errs.append(case + (key, this_err))
    is_problem = [e[-1] > o3_tolerance for e in all_errs]

    message = (permutation_message + "\n") + "\n".join(
        f"   (parity_k={int(k[0])}, did_translate={str(bool(k[1]))}, field={str(k[2]):20})     -> max error={float(k[3]):.3e}{'  FAIL' if prob else ''}"
        for k, prob in zip(all_errs, is_problem)
    )

    if sum(is_problem) > 0 or "FAIL" in permutation_message:
        raise AssertionError(f"Equivariance test failed for cases:\n{message}")

    return message
