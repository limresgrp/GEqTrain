# geqtrain/data/_build.py
import os
import logging
import inspect
from typing import List, Optional, Union
from os.path import isdir
from functools import partial
from multiprocessing import Pool

import torch
import numpy as np

from geqtrain.data import dataset as geq_datasets
from geqtrain.data.dataset import AtomicInMemoryDataset, InMemoryConcatDataset, LazyLoadingConcatDataset, AtomicDataDict, _NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS
from geqtrain.utils import instantiate, Config
from geqtrain.utils.auto_init import get_w_prefix
from geqtrain.utils.torch_geometric.batch import Batch

# ==================================================================================================
#           Public Functions
# ==================================================================================================

def dataset_from_config(config: Config, prefix: str = "train", loss_key: str = 'loss_coeffs') -> Union[InMemoryConcatDataset, LazyLoadingConcatDataset]:
    """
    Initializes a dataset from a configuration object, supporting list-based
    and prefix-based definitions, lazy loading, and data filtering.
    """
    list_key = f"{prefix}_dataset_list"
    
    if list_key in config:
        logging.info(f"Building dataset from list '{list_key}'")
        configs_to_process = config[list_key]
    else:
        logging.info(f"Building dataset from prefixed keys starting with '{prefix}'")
        configs_to_process = [config.as_dict()]

    all_instances = []
    # Assume the inmemory flag is consistent for all items in a list
    is_inmemory = True

    for item_config in configs_to_process:
        # Layer the item-specific config on top of the global config
        instance_config_dict = _update_config(config, item_config, prefix)
        instance_config_dict.update(item_config)

        # Promote keys from the item_config to have the current prefix.
        # This gives them high priority for the `instantiate` function.
        for key, value in item_config.items():
            prefixed_key = f"{prefix}_{key}"
            if prefixed_key not in instance_config_dict:
                instance_config_dict[prefixed_key] = value
        
        instance_config = Config(instance_config_dict)

        dataset_type = instance_config.get(prefix, instance_config.get('dataset'))
        if dataset_type is None: raise KeyError(f"Dataset type not found. Looked for '{prefix}' and 'dataset'.")
        
        class_name = _get_class_name(dataset_type)
        dataset_input = instance_config.get(f"{prefix}_input", instance_config.get('dataset_input'))
        if dataset_input is None: raise KeyError(f"Dataset input path not found. Looked for '{prefix}_input' and 'dataset_input'.")

        files_to_process = _expand_dataset_input_path(dataset_input)
        is_inmemory = instance_config.get('inmemory', True)
        key_clean_list = _get_key_clean(instance_config, loss_key)

        worker_func = partial(
            _handle_single_file,
            config=instance_config, prefix=prefix, class_name=class_name,
            inmemory=is_inmemory, key_clean_list=key_clean_list
        )
        
        n_workers = int(min(len(files_to_process), instance_config.get('dataset_num_workers', 1)))
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                results = pool.map(worker_func, files_to_process)
        else:
            results = [worker_func(file_info) for file_info in files_to_process]

        all_instances.extend([res for res in results if res is not None])

    if not all_instances:
        raise ValueError("No valid data points found after processing. Please check your dataset and filtering criteria.")

    if is_inmemory:
        return InMemoryConcatDataset(all_instances)
    else:
        return LazyLoadingConcatDataset(class_name, prefix, all_instances)

# ==================================================================================================
#           Internal Helper Functions
# ==================================================================================================

def _get_key_clean(config: Config, loss_key: str):
    from geqtrain.train.utils import parse_loss_metrics_dict
    loss_keys = set()
    for loss_dict in config.get(loss_key, []):
        for key, _, _, _ in parse_loss_metrics_dict(loss_dict):
            loss_keys.add(key)
    return list(loss_keys)

def _get_class_name(config_dataset_type):
    if inspect.isclass(config_dataset_type): return config_dataset_type
    try:
        module_name, class_name_str = config_dataset_type.rsplit('.', 1)
        return getattr(__import__(module_name, fromlist=[class_name_str]), class_name_str)
    except (ImportError, AttributeError, ValueError):
        dataset_name = config_dataset_type.lower()
        for k, v in inspect.getmembers(geq_datasets, inspect.isclass):
            if k.lower() == dataset_name or k.replace("Dataset", "").lower() == dataset_name:
                return v
    raise NameError(f"Dataset type '{config_dataset_type}' does not exist.")

def _expand_dataset_input_path(path_str: str) -> List[tuple]:
    """Expands a directory path or .txt file into a list of (ensemble_idx, file_path) tuples."""
    if path_str.endswith(".txt"):
        files = [line.strip() for line in open(path_str, "r") if line.strip()]
    elif isdir(path_str):
        with os.scandir(path_str) as entries:
            files = [entry.path for entry in entries if entry.is_file() and not entry.name.startswith('.')]
    else:
        return [(0, path_str)] # It's just a single file
    return [(idx, file) for idx, file in enumerate(files)]

def _update_config(config, _config_dataset, prefix):
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

def _handle_single_file(file_info: tuple, config: Config, prefix: str, class_name: type, inmemory: bool, key_clean_list: list):
    """Worker function to process a single data file, including filtering."""
    ensemble_index, dataset_file_name = file_info
    config[f'{prefix}_file_name'] = dataset_file_name
    config[f'{prefix}_ensemble_index'] = ensemble_index

    try:
        instance, _ = instantiate(class_name, prefix=prefix, optional_args=config)
    except Exception as e:
        logging.warning(f"Failed to instantiate dataset for file {dataset_file_name}. Error: {e}")
        return None
    
    if instance.data is None or instance.data.num_graphs == 0:
        return None

    instance = _filter_dataset(instance, key_clean_list, _node_types_to_keep(config), *_node_types_to_exclude_from_edges(config))

    if instance is None:
        logging.warning(f"All data from {dataset_file_name} was filtered out.")
        return None

    if inmemory:
        return instance
    else:
        return {
            "config": config.as_dict(),
            f"{prefix}_file_name": dataset_file_name,
            f"{prefix}_ensemble_index": ensemble_index,
            'lazy_dataset': np.arange(instance.data.num_graphs),
        }

def _node_types_to_keep(config):
    from geqtrain.train.utils import find_matching_indices
    keep_type_names = config.get("keep_type_names")
    if keep_type_names:
        return torch.tensor(find_matching_indices(config["type_names"], keep_type_names))
    return config.get("keep_node_types")

def _node_types_to_exclude_from_edges(config):
    from geqtrain.train.utils import find_matching_indices
    exclude_center, exclude_neigh = None, None
    if config.get("exclude_type_names_from_edge_center"):
        exclude_center = torch.tensor(find_matching_indices(config["type_names"], config["exclude_type_names_from_edge_center"]))
    if config.get("exclude_type_names_from_edge_neigh"):
        exclude_neigh = torch.tensor(find_matching_indices(config["type_names"], config["exclude_type_names_from_edge_neigh"]))
    return exclude_center, exclude_neigh

def _filter_dataset(
    dataset: AtomicInMemoryDataset,
    key_clean_list: List[str],
    keep_node_types: Optional[torch.Tensor] = None,
    exclude_node_types_from_edge_center: Optional[torch.Tensor] = None,
    exclude_node_types_from_edge_neigh: Optional[torch.Tensor] = None,
) -> Optional[AtomicInMemoryDataset]:
    """
    Filters a dataset by operating on the entire Batch object at once using
    a robust subgraphing method.
    """
    data: Batch = dataset.data
    if data is None or data.num_graphs == 0:
        return None

    # --- 1. Compute the final node mask based on all conditions ---
    nodes_to_keep_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.pos.device)
    
    node_types_are_fixed_field = False
    node_types = data[AtomicDataDict.NODE_TYPE_KEY].flatten() if AtomicDataDict.NODE_TYPE_KEY in data else None
    if node_types is None:
        node_types_are_fixed_field = True
        # Repeat the fixed node types for each graph in the batch
        node_types = dataset.fixed_fields.get(AtomicDataDict.NODE_TYPE_KEY).flatten().repeat(data.num_graphs)

    if keep_node_types is not None and node_types is not None:
        nodes_to_keep_mask &= torch.isin(node_types, keep_node_types.to(node_types.device))

    # --- 2. Compute the final edge mask based on all conditions ---
    edge_index = data.edge_index
    edges_to_keep_mask = torch.ones(data.num_edges, dtype=torch.bool, device=edge_index.device)

    nan_filters = []
    for key in key_clean_list:
        if key in _NODE_FIELDS and key in data and torch.isnan(data[key]).any():
            valid_nodes = torch.all(~torch.isnan(data[key]), dim=-1)
            nan_filters.append(valid_nodes[edge_index[0]])
    if nan_filters:
        edges_to_keep_mask &= torch.all(torch.stack(nan_filters), dim=0)
    
    if node_types is not None:
        if exclude_node_types_from_edge_center is not None:
            edges_to_keep_mask &= ~torch.isin(node_types[edge_index[0]], exclude_node_types_from_edge_center.to(node_types.device))
        if exclude_node_types_from_edge_neigh is not None:
            edges_to_keep_mask &= ~torch.isin(node_types[edge_index[1]], exclude_node_types_from_edge_neigh.to(node_types.device))

    # --- 3. Prune nodes that are no longer part of any kept edges ---
    kept_edges = edge_index[:, edges_to_keep_mask]
    nodes_in_kept_edges = kept_edges.unique()
    final_node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.pos.device)
    final_node_mask[nodes_in_kept_edges] = True
    
    # Combine with the primary node filter
    final_node_mask &= nodes_to_keep_mask
    
    if not final_node_mask.any():
        return None

    # --- 4. Use our new robust subgraph method ---
    # This correctly handles renumbering and all node/edge/graph attributes.
    dataset.data = data.subgraph(subset=final_node_mask, edge_index=kept_edges)
    if node_types_are_fixed_field:
        fixed_node_types = dataset.fixed_fields.get(AtomicDataDict.NODE_TYPE_KEY)
        fixed_node_types = fixed_node_types[final_node_mask[:len(fixed_node_types)]]
        dataset.fixed_fields = fixed_node_types
    
    if dataset.data.num_graphs == 0:
        return None

    return dataset