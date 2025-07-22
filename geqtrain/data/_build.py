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
from geqtrain import data
from geqtrain.data import (
    AtomicDataDict,
    AtomicInMemoryDataset,
    InMemoryConcatDataset,
    LazyLoadingConcatDataset,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
    _EXTRA_FIELDS,
    register_fields,
)
from geqtrain.utils import (
    instantiate,
    get_w_prefix,
    Config,
)
from geqtrain.utils.torch_geometric import Batch

from functools import partial
from multiprocessing import Pool

def get_key_clean(config: Config, loss_key:str):
    from geqtrain.train.utils import parse_loss_metrics_dict
    loss_keys = set()
    for loss_dict in config.get(loss_key, []):
        key, _, _, func_params = list(parse_loss_metrics_dict(loss_dict))[0]
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

def parse_dataset_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def get_dataset_file_names_and_ensemble_indices(prefix, _config_dataset):
    inpup_key = f"{prefix}_input"  # get input path
    assert inpup_key in _config_dataset, f"Missing {inpup_key} key in dataset config file."
    f_name = _config_dataset.get(inpup_key)
    ensemble_idx = 0

    if f_name.endswith(".txt") or isdir(f_name):
        out = []
        if f_name.endswith(".txt"):  # can be a .txt with a list of files
            files = parse_dataset_file(f_name)
        if isdir(f_name):  # can be dir
            # Efficiently pre-filter entries using os.scandir
            with os.scandir(f_name) as entries:
                files = [os.path.join(f_name, entry.name) for entry in entries if entry.is_file() and not entry.name.startswith('.')]
        for file in files:
            out.append((ensemble_idx, file))
            ensemble_idx += 1
        return out
    return [(ensemble_idx, f_name)]  # can be 1 file

def node_types_to_keep(config):
    # --- filter node target to train on based on node type or type name
    keep_type_names = config.get("keep_type_names", None)
    if keep_type_names is not None:
        from geqtrain.train.utils import find_matching_indices
        config["keep_node_types"] = torch.tensor(find_matching_indices(config["type_names"], keep_type_names))
    return config.get("keep_node_types", None)

def node_types_to_exclude_from_edges(config):
    # --- exclude edges from center node to specified node types
    exclude_type_names_from_edge_center = config.get("exclude_type_names_from_edge_center", None)
    if exclude_type_names_from_edge_center is not None:
        from geqtrain.train.utils import find_matching_indices
        config["exclude_node_types_from_edge_center"] = torch.tensor(find_matching_indices(config["type_names"], exclude_type_names_from_edge_center))
    
    exclude_type_names_from_edge_neigh = config.get("exclude_type_names_from_edge_neigh", None)
    if exclude_type_names_from_edge_neigh is not None:
        from geqtrain.train.utils import find_matching_indices
        config["exclude_node_types_from_edge_neigh"] = torch.tensor(find_matching_indices(config["type_names"], exclude_type_names_from_edge_neigh))
        
    return config.get("exclude_node_types_from_edge_center", None), config.get("exclude_node_types_from_edge_neigh", None)

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
    all_instances = []
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
        key_clean_list = get_key_clean(config, loss_key)
        mp_handle_single_dataset_file_name = partial(handle_single_dataset_file_name, config, prefix, class_name, inmemory, key_clean_list)
        n_workers = int(min(len(dataset_file_names_and_ensemble_indices), config.get('dataset_num_workers', len(os.sched_getaffinity(0)))))  # pid=0 the calling process
        if n_workers > 1:
            '''
            sysctl vm.max_map_count # Use this to check the limit of maps for shared memory
            sudo sysctl -w vm.max_map_count=NEW_VALUE # If necessary, change it with this command (valid until restart)
            '''
            # Ensure chunks are of size up to chunksize elements
            chunksize = 40000
            instances = []
            for i in range(0, len(dataset_file_names_and_ensemble_indices), chunksize):
                chunk = dataset_file_names_and_ensemble_indices[i:i + chunksize]
                with Pool(processes=n_workers) as pool:  # avoid ProcessPoolExecutor: https://stackoverflow.com/questions/18671528/processpoolexecutor-from-concurrent-futures-way-slower-than-multiprocessing-pool
                    results = pool.map(mp_handle_single_dataset_file_name, chunk)
                    instances.extend(copy.deepcopy(results))
                    del results
        else:
            instances = [mp_handle_single_dataset_file_name(file_name) for file_name in dataset_file_names_and_ensemble_indices]

        instances = [el for el in instances if el is not None]
        if not instances:
             raise ValueError("No valid data points found after filtering. Please check your dataset and filtering criteria.")
        all_instances.extend(instances)
    if inmemory:
        return InMemoryConcatDataset(all_instances)
    return LazyLoadingConcatDataset(class_name, prefix, config, all_instances)


def handle_single_dataset_file_name(config,  prefix, class_name, inmemory, key_clean_list, dataset_file_names_and_ensemble_indices):
    ensemble_index, dataset_file_name = dataset_file_names_and_ensemble_indices
    file_name_key = f"{prefix}_file_name"
    config[file_name_key] = dataset_file_name
    ensemble_index_key = f"{prefix}_ensemble_index"
    config[ensemble_index_key] = ensemble_index

    # Register fields:
    # This might reregister fields, but that's OK:
    instantiate(register_fields, all_args=config)

    try:
        instance, _ = instantiate(
            class_name,     # dataset selected to be instanciated
            prefix=prefix,  # look for this prefix word in yaml to select get the params for the ctor
            positional_args={},
            optional_args=config,
        )
    except RuntimeError as e:
        logging.warning(f"{e}. Nested exception: {e.__cause__}")
        return None
    
    if instance.data is None:
        return None

    # Apply filtering to the dataset instance.
    # This modifies the `instance.data` batch object in place.
    if not config.get("equivariance_test", False):
        default_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        
        instance = filter_dataset(
            instance,
            key_clean_list,
            node_types_to_keep(config),
            *node_types_to_exclude_from_edges(config)
        )
        torch.set_num_threads(default_num_threads)

    if instance is None:
        logging.warning(f"All data from {dataset_file_name} was filtered out.")
        return None

    if inmemory:
        return instance
    else:
        # For lazy loading, we save the filtered data back to the processed cache
        # and return the metadata needed to reconstruct it later.
        instance.save_processed()
        out = {
            file_name_key: dataset_file_name,
            ensemble_index_key: ensemble_index,
            'lazy_dataset': np.arange(instance.data.num_graphs),
        }
        del instance
        return out

def filter_dataset(
    dataset: AtomicInMemoryDataset,
    key_clean_list: List[str],
    keep_node_types: Optional[torch.Tensor] = None,
    exclude_node_types_from_edge_center: Optional[torch.Tensor] = None,
    exclude_node_types_from_edge_neigh: Optional[torch.Tensor] = None,
) -> Optional[AtomicInMemoryDataset]:
    """
    Filters a dataset by directly removing nodes and edges based on type,
    then filters remaining edges based on NaN targets and other criteria.
    This function creates a new filtered dataset object if any data remains.
    """
    data = dataset.data
    if data is None or data.num_graphs == 0:
        return None

    # Deconstruct the batch into a list of individual graphs to filter them one by one
    data_list = data.to_data_list()
    filtered_data_list = []
    original_graph_indices = []

    # Handle node types that can be fixed for the whole dataset
    has_per_graph_node_types = AtomicDataDict.NODE_TYPE_KEY in data
    fixed_node_types = None
    if not has_per_graph_node_types and AtomicDataDict.NODE_TYPE_KEY in dataset.fixed_fields:
        fixed_node_types = dataset.fixed_fields[AtomicDataDict.NODE_TYPE_KEY]

    for i, graph in enumerate(data_list):
        # 1. Primary node filtering based on `keep_node_types`
        # If keep_node_types is given, we immediately create a subgraph.
        if keep_node_types is not None:
            current_node_types = fixed_node_types if fixed_node_types is not None else graph[AtomicDataDict.NODE_TYPE_KEY]
            if current_node_types is not None:
                device = current_node_types.device
                nodes_to_keep_mask = torch.isin(current_node_types, keep_node_types.to(device))
                nodes_to_keep_indices = torch.where(nodes_to_keep_mask)[0]
                
                if nodes_to_keep_indices.shape[0] == 0:
                    continue  # Skip this graph entirely if no nodes of the desired type are present
                
                # Subgraph operation filters nodes, node attributes, and edges
                # to the induced subgraph of the kept nodes.
                graph = graph.subgraph(nodes_to_keep_indices)

                # Update fixed_node_types
                if fixed_node_types is not None:
                    fixed_node_types = fixed_node_types[nodes_to_keep_indices]

        # If graph is empty after primary filtering, skip it
        if graph.num_nodes == 0 or graph.num_edges == 0:
            continue

        # 2. Secondary edge filtering on the (potentially already filtered) graph
        edge_index = graph.edge_index
        edges_to_keep_mask = torch.ones(graph.num_edges, dtype=torch.bool, device=edge_index.device)

        # Filter based on NaN targets
        nan_edge_filters = []
        for key in key_clean_list:
            if key in _NODE_FIELDS and key in graph:
                target_vals = graph[key]
                if target_vals.dim() == 1:
                    target_vals = target_vals.unsqueeze(-1)
                valid_nodes = torch.any(~torch.isnan(target_vals), dim=-1)
                valid_node_indices = torch.where(valid_nodes)[0]
                nan_edge_filters.append(torch.isin(edge_index[0], valid_node_indices))

        if nan_edge_filters:
            edges_to_keep_mask &= torch.any(torch.stack(nan_edge_filters), dim=0)

        # Filter based on excluded center/neighbor types
        current_node_types = fixed_node_types if fixed_node_types is not None else graph[AtomicDataDict.NODE_TYPE_KEY]
        if current_node_types is not None:
            current_node_types = current_node_types.flatten()
            device = current_node_types.device
            if exclude_node_types_from_edge_center is not None:
                center_type_mask = torch.isin(current_node_types[edge_index[0]], exclude_node_types_from_edge_center.to(device))
                edges_to_keep_mask &= ~center_type_mask
            
            if exclude_node_types_from_edge_neigh is not None:
                neighbor_type_mask = torch.isin(current_node_types[edge_index[1]], exclude_node_types_from_edge_neigh.to(device))
                edges_to_keep_mask &= ~neighbor_type_mask
        
        # 3. Apply the edge filter and prune any resulting isolated nodes
        if not torch.all(edges_to_keep_mask):
            if not edges_to_keep_mask.any():
                continue # Skip if no edges are left

            # The most robust way to remove edges and then resulting isolated nodes
            # is to identify the nodes in the valid edges and create a final subgraph.
            nodes_in_kept_edges = edge_index[:, edges_to_keep_mask].unique()
            
            if nodes_in_kept_edges.shape[0] == 0:
                continue

            # Before the final subgraph, we MUST apply the edge mask to all edge attributes.
            # The subsequent subgraph call will then correctly handle the node attributes.
            graph.edge_index = edge_index[:, edges_to_keep_mask]
            for field in _EDGE_FIELDS:
                if field in graph and graph[field].shape[0] == edges_to_keep_mask.shape[0]:
                    graph[field] = graph[field][edges_to_keep_mask]
            
            # Now, create the final subgraph to remove isolated nodes.
            graph = graph.subgraph(nodes_in_kept_edges)

        if graph.num_nodes > 0 and graph.num_edges > 0:
            filtered_data_list.append(graph)
            original_graph_indices.append(i)

    if not filtered_data_list:
        return None

    # Re-batch the filtered graphs
    new_batch = Batch.from_data_list(filtered_data_list, exclude_keys=list(dataset.fixed_fields.keys()))

    # Copy over graph-level attributes from the original graphs that were kept
    for field in _GRAPH_FIELDS:
        if field in data:
            new_batch[field] = data[field][original_graph_indices]

    dataset.data = new_batch
    if fixed_node_types is not None:
        dataset.fixed_fields[AtomicDataDict.NODE_TYPE_KEY] = fixed_node_types
    return dataset