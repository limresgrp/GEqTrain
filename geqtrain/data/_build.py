""" Adapted from https://github.com/mir-group/nequip
"""

import copy
import inspect
import numpy as np
from importlib import import_module
import logging
from typing import Dict, List, Optional, Union
from os import listdir
from os.path import isdir, isfile, join

import torch
from geqtrain.data.dataset import InMemoryConcatDataset, LazyLoadingConcatDataset
from geqtrain import data
from geqtrain.data import (
    AtomicDataDict,
    AtomicInMemoryDataset,
    _NODE_FIELDS,
    register_fields,
)
from geqtrain.utils import (
    instantiate,
    get_w_prefix,
    Config
)

from functools import partial
from multiprocessing import Pool, Lock, cpu_count, Manager


def get_class_name(config_dataset_type):
    # looks for dset type specified in yaml if present (dataset_list/dataset: {$class_name})
    if inspect.isclass(config_dataset_type):
        class_name = config_dataset_type
    else:
        try:
            module_name = ".".join(config_dataset_type.split(".")[:-1])
            class_name = ".".join(config_dataset_type.split(".")[-1:])
            class_name = getattr(import_module(module_name), class_name)
        except Exception:
            # default class defined in geqtrain.data or geqtrain.dataset
            dataset_name = config_dataset_type.lower()

            class_name = None
            for k, v in inspect.getmembers(data, inspect.isclass):
                if k.endswith("Dataset"):
                    if k.lower() == dataset_name:
                        class_name = v
                    if k[:-7].lower() == dataset_name:
                        class_name = v
                elif k.lower() == dataset_name:
                    class_name = v

    if class_name is None:
        raise NameError(f"dataset type {dataset_name} does not exists")
    return class_name

def update_config(config, _config_dataset, prefix):
    _config: dict = config.as_dict() if isinstance(config, Config) else config
    _config.update(_config_dataset)

    # if dataset r_max is not found, use the universal r_max
    eff_key = "extra_fixed_fields"
    prefixed_eff_key = f"{prefix}_{eff_key}"

    _config[prefixed_eff_key] = get_w_prefix(
        eff_key, {},
        prefix=prefix,
        arg_dicts=_config
    )

    _config[prefixed_eff_key][AtomicDataDict.R_MAX_KEY] = get_w_prefix(
        AtomicDataDict.R_MAX_KEY,
        prefix=prefix,
        arg_dicts=[_config[prefixed_eff_key], _config],
    )
    return _config

def get_dataset_file_names(prefix, _config_dataset):
    inpup_key = f"{prefix}_input"  # get input path
    assert inpup_key in _config_dataset, f"Missing {inpup_key} key in dataset config file."
    f_name = _config_dataset.get(inpup_key)

    if isdir(f_name):  # can be dir
        return [join(f_name, f) for f in listdir(f_name) if (isfile(join(f_name, f)) and not f.startswith('.'))]
    return [f_name]  # can be 1 file

def node_types_to_keep(config):
    # --- filter node target to train on based on node type or type name
    keep_type_names = config.get("keep_type_names", None)
    if keep_type_names is not None:
        from geqtrain.train.utils import find_matching_indices
        config["keep_node_types"] = torch.tensor(find_matching_indices(config["type_names"], keep_type_names))
    return config.get("keep_node_types", None) # keep_node_types

def node_types_to_exclude(config):
    # --- exclude edges from center node to specified node types
    exclude_type_names_from_edges = config.get("exclude_type_names_from_edges", None)
    if exclude_type_names_from_edges is not None:
        from geqtrain.train.utils import find_matching_indices
        config["exclude_node_types_from_edges"] = torch.tensor(find_matching_indices(config["type_names"], exclude_type_names_from_edges))
    return config.get("exclude_node_types_from_edges", None) # exclude_node_types_from_edges

def dataset_from_config(config, prefix: str = "dataset", loss=None) -> Union[InMemoryConcatDataset, LazyLoadingConcatDataset]:
    """
    TODO update docs here
    Called for each {prefix}_list in yaml, possible prefix: dataset, validation_dataset_list, test_dataset_list
    1) get dset type (eg <class 'geqtrain.data.dataset.NpzDataset'>)
    2) get data_path or data_file (folder containing files to be opened)
    3) registers fields to read/expose-in-code data at 2) in the correct form
    4) instanciate dataset(s)
        4.1) if many dsets
    5) return cat(dsets)

    initialize dataset based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Args:

    config (dict, geqtrain.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Returns:
        torch.utils.data.ConcatDataset: dataset
    """
    dataset_id_offset = 0
    config_dataset_list: List[Dict] = config.get(f"{prefix}_list", [config])
    for dataset_id, _config_dataset in enumerate(config_dataset_list):
        config_dataset_type = _config_dataset.get(prefix, None)
        if config_dataset_type is None:
            raise KeyError(
                f"Dataset with prefix `{prefix}` isn't present in this config!")

        class_name         = get_class_name(config_dataset_type)
        config             = update_config(config, _config_dataset, prefix)
        dataset_file_names = get_dataset_file_names(prefix, _config_dataset)

        # default behavior: in-memory loading
        inmemory = _config_dataset.get('inmemory', True)
        logging.info(f"Using {'' if inmemory else 'NOT-'}inmemory dataset.")

        # multiprocessing handling of npz reading
        m = Manager() #! https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processe
        mp_lock = m.Lock()
        mp_handle_single_dataset_file_name = partial(handle_single_dataset_file_name, config, dataset_id, prefix, class_name, inmemory, mp_lock, dataset_id_offset, loss)
        # TODO here compute batch_size wrt len data
        with Pool(processes=cpu_count()-5) as pool:
            l=pool.map(mp_handle_single_dataset_file_name, dataset_file_names, chunksize=1000)

        instances = list(l)
        if inmemory:
            return InMemoryConcatDataset(instances)
        return LazyLoadingConcatDataset(class_name, prefix, config, instances)


def remove_node_centers_for_NaN_targets_and_edges(
    dataset: AtomicInMemoryDataset,
    loss_func,
    keep_node_types: Optional[List[str]] = None,
    exclude_node_types_from_edges: Optional[List[str]] = None,
):
    data = dataset.data
    if AtomicDataDict.NODE_TYPE_KEY in data:
        node_types: torch.Tensor = data[AtomicDataDict.NODE_TYPE_KEY]
    else:
        node_types: torch.Tensor = dataset.fixed_fields[AtomicDataDict.NODE_TYPE_KEY]

    def get_node_types_mask(node_types, filter, data):
        return torch.isin(node_types.flatten(), filter.cpu()).repeat(data.__num_graphs__)

    def update_edge_index(data, edge_filter: torch.Tensor):
        data[AtomicDataDict.EDGE_INDEX_KEY] = data[AtomicDataDict.EDGE_INDEX_KEY][:, edge_filter]
        new_edge_index_slices = [0]
        for slice_to in data.__slices__[AtomicDataDict.EDGE_INDEX_KEY][1:]:
            new_edge_index_slices.append(edge_filter[:slice_to].sum())
        data.__slices__[AtomicDataDict.EDGE_INDEX_KEY] = torch.tensor(
            new_edge_index_slices, dtype=torch.long, device=edge_filter.device)
        if AtomicDataDict.EDGE_CELL_SHIFT_KEY in data:
            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][edge_filter]
            data.__slices__[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = data.__slices__[
                AtomicDataDict.EDGE_INDEX_KEY]

    # - Remove edges of atoms whose result is NaN - #
    if loss_func is not None:
        for key in loss_func.keys:
            if hasattr(loss_func.funcs[key], "ignore_nan") and loss_func.funcs[key].ignore_nan:
                key_clean = loss_func.remove_suffix(key)
                if key_clean not in _NODE_FIELDS:
                    continue
                if key_clean in data:
                    if keep_node_types is not None:
                        remove_node_types_mask = ~get_node_types_mask(node_types, keep_node_types, data)
                        data[key_clean][remove_node_types_mask] = torch.nan
                    val: torch.Tensor = data[key_clean]
                    if val.dim() == 1:
                        val = val.reshape(len(val), -1)

                    not_nan_edge_filter = torch.isin(data[AtomicDataDict.EDGE_INDEX_KEY][0], torch.argwhere(torch.any(~torch.isnan(val), dim=-1)).flatten())
                    update_edge_index(data, not_nan_edge_filter)

    # - Remove edges which connect center nodes with node types present in 'exclude_node_types_from_edges'
    if exclude_node_types_from_edges is not None:
        exclude_node_types_from_edges_mask = get_node_types_mask(node_types, exclude_node_types_from_edges, data)
        keep_edges_filter = ~torch.isin(data[AtomicDataDict.EDGE_INDEX_KEY][1], torch.nonzero(exclude_node_types_from_edges_mask).flatten())
        update_edge_index(data, keep_edges_filter)

    if data[AtomicDataDict.EDGE_INDEX_KEY].shape[-1] == 0:
        return None

    dataset.data = data
    return dataset

def handle_single_dataset_file_name(config, dataset_id, prefix, class_name, inmemory, mp_lock, dataset_id_offset, loss, dataset_file_name):
    _config = copy.deepcopy(config)
    _config[AtomicDataDict.DATASET_INDEX_KEY] = dataset_id + dataset_id_offset
    with mp_lock:
        dataset_id_offset += 1
    _config[f"{prefix}_file_name"] = dataset_file_name

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=_config)

    instance, _ = instantiate(
        class_name,  # dataset selected to be instanciated
        prefix=prefix,  # look for this prefix word in yaml to select get the params for the ctor
        positional_args={},
        optional_args=_config,
    )

    """
    !!! remove_node_centers_for_NaN_targets_and_edges is not supported for NOT-inmemory dataset !!!
    """
    if inmemory:
        # Filter out nan nodes and nodes with type_names that we don't want to keep
        instance = remove_node_centers_for_NaN_targets_and_edges(instance, loss, node_types_to_keep(config), node_types_to_exclude(config))
    if instance is not None:
        if inmemory:
            return instance
        else:
            out = {
                'dataset_file_name': dataset_file_name,
                'dataset_id': dataset_id + dataset_id_offset,
                'lazy_dataset': np.arange(instance.data.num_graphs),
            }
            del instance
            return out