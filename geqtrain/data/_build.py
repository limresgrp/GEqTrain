""" Adapted from https://github.com/mir-group/nequip
"""
import os
import copy
import inspect
import numpy as np
from importlib import import_module
import logging
from typing import Dict, List, Optional, Union
from os.path import isdir

import torch
from geqtrain.data.dataset import InMemoryConcatDataset, LazyLoadingConcatDataset
from geqtrain import data
from geqtrain.data import (
    AtomicDataDict,
    AtomicInMemoryDataset,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
    register_fields,
)
from geqtrain.utils import (
    instantiate,
    get_w_prefix,
    Config,
)

from functools import partial
from multiprocessing import Pool

def get_ignore_nan_loss_key_clean(config: Config, loss_key:str):
    from geqtrain.train.utils import parse_loss_metrics_dict
    loss_keys = set()
    for loss_dict in config.get(loss_key, []):
        key, _, _, func_params = list(parse_loss_metrics_dict(loss_dict))[0]
        if func_params.get('ignore_nan', False):
            loss_keys.add(key)
    return list(loss_keys)

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

def get_dataset_file_names_and_ensemble_indices(prefix, _config_dataset):
    inpup_key = f"{prefix}_input"  # get input path
    assert inpup_key in _config_dataset, f"Missing {inpup_key} key in dataset config file."
    f_name = _config_dataset.get(inpup_key)
    ensemble_idx = 0

    if isdir(f_name):  # can be dir
        out = []
        # Efficiently pre-filter entries using os.scandir
        with os.scandir(f_name) as entries:
            # Filter non-hidden files using entry.is_file() and entry.name
            for entry in entries:
                if entry.is_file() and not entry.name.startswith('.'):
                    out.append((ensemble_idx, os.path.join(f_name, entry.name)))
                    ensemble_idx += 1
            return out
    return [(ensemble_idx, f_name)]  # can be 1 file

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

def dataset_from_config(config,
                        prefix: str = "dataset",
                        loss_key: str='loss_coeffs') -> Union[InMemoryConcatDataset, LazyLoadingConcatDataset]:
    """
    loss_key in evaluate metrics_components
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

    config_dataset_list: List[Dict] = config.get(f"{prefix}_list", [config])
    assert len(config_dataset_list) == 1, f"Only one dataset can be specified in {prefix}_list"
    for _config_dataset in config_dataset_list:
        config_dataset_type = _config_dataset.get(prefix, None)
        if config_dataset_type is None:
            raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

        class_name                              = get_class_name(config_dataset_type)
        config                                  = update_config(config, _config_dataset, prefix)
        dataset_file_names_and_ensemble_indices = get_dataset_file_names_and_ensemble_indices(prefix, _config_dataset)

        # default behavior: in-memory loading
        inmemory = _config_dataset.get('inmemory', True)
        logging.info(f"Using {'' if inmemory else 'NOT-'}inmemory dataset.")

        # --- multiprocessing handling of npz reading
        key_clean_list = get_ignore_nan_loss_key_clean(config, loss_key)
        mp_handle_single_dataset_file_name = partial(handle_single_dataset_file_name, config, prefix, class_name, inmemory, key_clean_list)
        n_workers = int(min(len(dataset_file_names_and_ensemble_indices), config.get('dataset_num_workers', len(os.sched_getaffinity(0)))))  # pid=0 the calling process
        if n_workers>1:
            '''
            ! Known issue:
            if dataset is "in-memory" and n_workers>1 we have AND number of npz >=5000 (approximately):
                RuntimeError: unable to mmap ... bytes from file <filename not specified>: Cannot allocate memory (...)

            Solution:
            - option 1) faster preprocessing but slower training: keep n_workers>1 but use inmemory: false
            - option 2) slower preprocessing but faster train: keep inmemory: true (which is default behavior) but use n_workers = 1
            '''
            # if inmemory: an even split; elif NOT-inmemory: we can't afford loading the whole dset in different processes
            chunksize = int(max(len(dataset_file_names_and_ensemble_indices) // (n_workers if inmemory else n_workers * .25), 1))
            with Pool(processes=n_workers) as pool: # avoid ProcessPoolExecutor: https://stackoverflow.com/questions/18671528/processpoolexecutor-from-concurrent-futures-way-slower-than-multiprocessing-pool
                instances = pool.map(mp_handle_single_dataset_file_name, dataset_file_names_and_ensemble_indices, chunksize=chunksize)
        else:
            instances = [mp_handle_single_dataset_file_name(file_name) for file_name in dataset_file_names_and_ensemble_indices]

        instances = [el for el in instances if el is not None]
        if inmemory:
            return InMemoryConcatDataset(instances)
        return LazyLoadingConcatDataset(class_name, prefix, config, instances)


def handle_single_dataset_file_name(config,  prefix, class_name, inmemory, key_clean_list, dataset_file_names_and_ensemble_indices):
    _config = copy.deepcopy(config) # this might not be required but kept for saefty
    ensemble_index, dataset_file_name = dataset_file_names_and_ensemble_indices
    file_name_key = f"{prefix}_file_name"
    _config[file_name_key] = dataset_file_name
    ensemble_index_key = f"{prefix}_ensemble_index"
    _config[ensemble_index_key] = ensemble_index

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=_config)

    try:
        instance, _ = instantiate(
            class_name,     # dataset selected to be instanciated
            prefix=prefix,  # look for this prefix word in yaml to select get the params for the ctor
            positional_args={},
            optional_args=_config,
        )
    except FileNotFoundError:
        return None
    if instance.data is None:
        return None

    """
    !!! remove_node_centers_for_NaN_targets_and_edges is not supported for NOT-inmemory dataset !!!
    """
    if inmemory:
        if not config.get("equivariance_test", False):
            # Filter out nan nodes and nodes with type_names that we don't want to keep
            default_num_threads = torch.get_num_threads() # default is 64 (always?)
            torch.set_num_threads(1)  # torch.argwhere, torch.isin and torch.nonzero may parallelize on many cpus and clutter the machine
            instance = remove_node_centers_for_NaN_targets_and_edges(instance, key_clean_list, node_types_to_keep(config), node_types_to_exclude(config))
            torch.set_num_threads(default_num_threads)
        return instance

    # otherwise return the non-in-mem data struct that contains all info to reinstanciate instance at runtime
    out = {
        file_name_key: dataset_file_name,
        ensemble_index_key: ensemble_index,
        'lazy_dataset': np.arange(instance.data.num_graphs),
    }
    del instance
    return out

def remove_node_centers_for_NaN_targets_and_edges(
    dataset: AtomicInMemoryDataset,
    key_clean_list: List[str],
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

    def update_node_index(data, key_clean: str, node_filter: torch.Tensor):
        data[key_clean] = data[key_clean][node_filter]
        if len(node_filter) == 0:
            return
        node_filter_cumsum = node_filter.cumsum(0)
        new_node_index_slices = node_filter_cumsum[torch.as_tensor(data.__slices__[key_clean][1:]) - 1].tolist()
        new_node_index_slices.insert(0, 0)
        data.__slices__[key_clean] = torch.tensor(new_node_index_slices, dtype=torch.long, device=node_filter.device)
    
    def update_edge_index(data, edge_filter: torch.Tensor):
        data[AtomicDataDict.EDGE_INDEX_KEY] = data[AtomicDataDict.EDGE_INDEX_KEY][:, edge_filter]
        if len(edge_filter) == 0:
            return
        edge_filter_cumsum = edge_filter.cumsum(0)
        new_edge_index_slices = edge_filter_cumsum[torch.as_tensor(data.__slices__[AtomicDataDict.EDGE_INDEX_KEY][1:]) - 1].tolist()
        new_edge_index_slices.insert(0, 0)
        data.__slices__[AtomicDataDict.EDGE_INDEX_KEY] = torch.tensor(new_edge_index_slices, dtype=torch.long, device=edge_filter.device)
        if AtomicDataDict.EDGE_CELL_SHIFT_KEY in data:
            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][edge_filter]
            data.__slices__[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = data.__slices__[AtomicDataDict.EDGE_INDEX_KEY]

    # - Remove edges of atoms whose result is NaN - #
    node_center_edge_idcs = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    keep_edges_filter = torch.zeros(len(node_center_edge_idcs), dtype=torch.bool) # initialize edge filter tensor of dim (n_edges,)
    keep_nodes_filter = torch.zeros(len(node_types),            dtype=torch.bool) # initialize node filter tensor of dim (n_atoms,)
    for key_clean in key_clean_list:
        if key_clean not in data: continue
        if key_clean in _GRAPH_FIELDS:
            keep_edges_filter += 1
        if key_clean in _EDGE_FIELDS: # TODO remove edges for which we have nan targets
            keep_edges_filter += 1
        elif key_clean in _NODE_FIELDS:
            if keep_node_types is not None: # Set target values to nan for nodes not present in 'keep_node_types'
                remove_node_types_mask = ~get_node_types_mask(node_types, keep_node_types, data)
                data[key_clean][remove_node_types_mask] = torch.nan
            val: torch.Tensor = data[key_clean]
            if val.dim() == 1: val = val.reshape(len(val), -1)

            # Here we are performing the UNION of edge_idcs we want to keep, across different target keys
            keep_edges_filter += torch.isin(node_center_edge_idcs, torch.argwhere(torch.any(~torch.isnan(val), dim=-1)).flatten())
    
    # - Remove edges which connect center nodes with node types present in 'exclude_node_types_from_edges'
    if exclude_node_types_from_edges is not None:
        exclude_node_types_from_edges_mask = get_node_types_mask(node_types, exclude_node_types_from_edges, data)
        # Here we are performing the INTERSECTION between the edge_idcs we want to keep from previous filtering and a tensor that zeroes out edges we want to exclude
        keep_edges_filter *= ~torch.isin(data[AtomicDataDict.EDGE_INDEX_KEY][1], torch.nonzero(exclude_node_types_from_edges_mask).flatten())

    keep_nodes_filter[node_center_edge_idcs[keep_edges_filter].unique().flatten()] = True
    update_edge_index(data, keep_edges_filter)
    for key_clean in key_clean_list:
        if key_clean in _NODE_FIELDS and key_clean in data:
            update_node_index(data, key_clean, keep_nodes_filter)

    if data[AtomicDataDict.EDGE_INDEX_KEY].shape[-1] == 0:
        return None

    dataset.data = data
    return dataset