import re
import logging
from typing import List

def parse_dict(coeffs: dict):
    for key, value in coeffs.items():
        logging.debug(f" parsing {key} {value}")
        coeff = 1.0
        func = "MSELoss"
        func_params = {}
        if isinstance(value, (float, int)):
            coeff = value
        elif isinstance(value, str) or callable(value):
            func = value
        elif isinstance(value, (list, tuple)):
            # list of [func], [func, param], [coeff, func], [coeff, func, params]
            if isinstance(value[0], (float, int)):
                coeff = value[0]
                if len(value) > 1:
                    func = value[1]
                if len(value) > 2:
                    assert isinstance(value[2], dict)
                    func_params = value[2]
            else:
                func = value[0]
                if len(value) > 1:
                    func_params = value[1]
        else:
            raise NotImplementedError(
                f"expected float, list or tuple, but get {type(value)}"
            )
        logging.debug(f" parsing {coeff} {func}")
        yield key, coeff, func, func_params

def find_matching_indices(ls: List[str], patterns: List[str]):
    matching_indices = []
    for i, string in enumerate(ls):
        for pattern in patterns:
            if re.search(pattern, string):
                matching_indices.append(i)
                break  # Stop checking other patterns if one matches
    return matching_indices