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
#           Internal Helper Functions (Restored and Refactored)
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

    # Restore filtering logic
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
    Filters a dataset by directly removing nodes and edges based on type,
    then filters remaining edges based on NaN targets and other criteria.
    This function creates a new filtered dataset object if any data remains.
    """
    data = dataset.data
    if data is None or data.num_graphs == 0:
        return None
    
    has_to_filter = False
    if keep_node_types is not None:
        logging.info(f"Filtering nodes: keeping only node types {keep_node_types.tolist()}")
        has_to_filter = True
    if exclude_node_types_from_edge_center is not None or exclude_node_types_from_edge_neigh is not None:
        logging.info(
            f"Filtering edges: "
            f"excluding center node types {exclude_node_types_from_edge_center.tolist() if exclude_node_types_from_edge_center is not None else '[]'}, "
            f"excluding neighbor node types {exclude_node_types_from_edge_neigh.tolist() if exclude_node_types_from_edge_neigh is not None else '[]'}"
        )
        has_to_filter = True
    # Before skipping, be sure that there are no NaN node targets, otherwise it is necessary to filter edges from those central nodes
    for key in key_clean_list:
        if key in _NODE_FIELDS and key in data:
            if torch.isnan(data[key]).any():
                has_to_filter = True
                logging.info(f"NaN detected in node target '{key}' for a graph. Excluding center nodes with NaN targets from graph.")
    # If no filtering is necessary, skip to save time
    if not has_to_filter:
        return dataset

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