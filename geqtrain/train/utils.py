import re
import logging
from typing import List
import torch

def parse_dict(components: dict):
    for key, value in components.items():
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
            if '*' not in pattern and '?' not in pattern:
                pattern = f"^{pattern}$"
            if re.search(pattern, string):
                matching_indices.append(i)
                break  # Stop checking other patterns if one matches
    return matching_indices

def evaluate_end_chunking_condition(already_computed_nodes, batch_chunk_center_nodes, num_batch_center_nodes):
    '''evaluate ending condition
    if chunking is active -> if whole struct has been processed then batch is over
    already_computed_nodes is the stopping criteria to finish batch step'''
    if already_computed_nodes is None:
        if len(batch_chunk_center_nodes) < num_batch_center_nodes:
            already_computed_nodes = batch_chunk_center_nodes
    elif len(already_computed_nodes) + len(batch_chunk_center_nodes) == num_batch_center_nodes:
        already_computed_nodes = None
    else:
        assert len(already_computed_nodes) + len(batch_chunk_center_nodes) < num_batch_center_nodes
        already_computed_nodes = torch.cat([already_computed_nodes, batch_chunk_center_nodes], dim=0)
    return already_computed_nodes