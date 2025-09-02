import logging
import contextlib
from copy import deepcopy
from typing import List, Optional
import numpy as np
import torch
from geqtrain.data import AtomicData, AtomicDataDict, _NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS
from geqtrain.train.loss import Loss

def get_output_keys(loss_func: Loss):
    output_keys, per_node_outputs_keys = [], []
    if loss_func is not None:
        for key in loss_func.keys:
            key_clean = loss_func.remove_suffix(key)
            if key_clean in _NODE_FIELDS.union(_GRAPH_FIELDS).union(_EDGE_FIELDS):
                output_keys.append(key_clean)
            if key_clean in _NODE_FIELDS:
                per_node_outputs_keys.append(key_clean)
    return output_keys, per_node_outputs_keys

def run_inference(
    model,
    data,
    device,
    loss_fn: Loss,
    config: dict,
    already_computed_nodes=None,
    is_train: bool=False,
):
    """
    Runs inference for a single batch, extracting options from the config object.
    """
    # Get options from the config dict, with sensible defaults
    mixed_precision = config.get('mixed_precision', False)
    chunking = config.get('chunking', False)
    batch_max_atoms = config.get('batch_max_atoms', 1000)
    ignore_chunk_keys = config.get('ignore_chunk_keys', [])
    dropout_edges = config.get('dropout_edges', 0.0)
    requires_grad = config.get('model_requires_grads', False)
    is_ddp = config.get('ddp', False)

    # Compute keys on-the-fly
    output_keys, per_node_outputs_keys = get_output_keys(loss_fn)
    has_graph_level_loss = any(key in _GRAPH_FIELDS for key in output_keys)
    effective_chunking = chunking and not has_graph_level_loss

    if chunking and has_graph_level_loss:
        graph_keys = [k for k in output_keys if k in _GRAPH_FIELDS]
        logging.warning(
            f"Chunking was enabled, but a graph-level loss was detected ({graph_keys}). "
            "Disabling chunking for this batch."
        )

    precision = torch.autocast(device_type='cuda' if torch.cuda.is_available() 
                               else 'cpu', dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
    
    batch = AtomicData.to_AtomicDataDict(data.to(device))
    batch_center_nodes = batch[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
    num_batch_center_nodes = len(batch_center_nodes)

    if not effective_chunking:
        input_data = {
            k: v.requires_grad_() if requires_grad and torch.is_floating_point(v) else v
            for k, v in batch.items()
            if k not in output_keys
        }
        ref_data = batch
    else:
        input_data, ref_data, batch_center_nodes = prepare_chunked_input_data(
            batch=batch,
            already_computed_nodes=already_computed_nodes,
            output_keys=output_keys,
            per_node_outputs_keys=per_node_outputs_keys,
            batch_max_atoms=batch_max_atoms,
            ignore_chunk_keys=ignore_chunk_keys,
            device=device,
        )

    if hasattr(data, "__slices__"):
        for slices_key, slices in data.__slices__.items():
            val = torch.tensor(slices, dtype=torch.long, device=device)
            input_data[f"{slices_key}_slices"] = val
            ref_data[f"{slices_key}_slices"] = val

    if dropout_edges > 0 and model.training:
        apply_dropout_edges(dropout_edges, input_data)

    use_no_grad = (not is_train) and (not config.get('model_requires_grads', False))
    cm = torch.no_grad() if use_no_grad else contextlib.nullcontext()
    with cm, precision:
        out = model(input_data)
        del input_data

    model_to_check = model.module if is_ddp else model
    if hasattr(model_to_check, 'ref_data_keys'):
        for k in model_to_check.ref_data_keys:
            if k in out:
                target = out[k]
                key_clean = k.replace("_target", "")
                ref_data[key_clean] = target

    return out, ref_data, batch_center_nodes, num_batch_center_nodes

def prepare_chunked_input_data(
    batch: AtomicDataDict.Type,
    already_computed_nodes: Optional[torch.Tensor],
    output_keys: List[str] = [],
    per_node_outputs_keys: List[str] = [],
    batch_max_atoms: int = 1000,
    ignore_chunk_keys: List[str] = [],
    device="cpu"
):
    # === Limit maximum batch size to avoid CUDA Out of Memory === #

    batch_chunk = deepcopy(batch)
    batch_chunk_edge_index = batch_chunk[AtomicDataDict.EDGE_INDEX_KEY]

    chunk = already_computed_nodes is not None
    if chunk:
        batch_chunk_edge_index = batch_chunk_edge_index[:, ~torch.isin(batch_chunk_edge_index[0], already_computed_nodes)]
    if len(batch_chunk_edge_index[0].unique()) == 0:
        return None, None, None

    # = Iteratively remove edges from batch_chunk = #
    # = ----------------------------------------- = #
    edge_fields_dict = {
        edge_field: batch[edge_field]
        for edge_field in _EDGE_FIELDS
        if edge_field in batch
    }
    offset = 0
    while len(batch_chunk_edge_index.unique()) > batch_max_atoms:

        def get_node_center_idcs(batch_chunk_edge_index: torch.Tensor, batch_max_atoms: int, offset: int):
            unique_set = set()

            for i, num in enumerate(batch_chunk_edge_index[1]):
                unique_set.add(num.item())

                if len(unique_set) >= batch_max_atoms:
                    node_center_idcs = batch_chunk_edge_index[0, :i+1].unique()
                    if len(node_center_idcs) == 1:
                        num_nodes = torch.isin(batch_chunk_edge_index[0], node_center_idcs).sum()
                        if num_nodes > batch_max_atoms:
                            raise ValueError(
                                f"At least one node in the graph has more neighbors ({num_nodes}) "
                                f"than the maximum allowed number of atoms in a batch ({batch_max_atoms}). "
                                "Please increase the value of 'batch_max_atoms' in the config file."
                            )
                        return node_center_idcs
                    return node_center_idcs[:-offset]
            return batch_chunk_edge_index[0].unique()

        def get_edge_filter(batch_chunk_edge_index: torch.Tensor, offset: int):
            node_center_idcs = get_node_center_idcs(batch_chunk_edge_index, batch_max_atoms, offset)
            edge_filter = torch.isin(batch_chunk_edge_index[0], node_center_idcs)
            return edge_filter

        chunk = True
        offset += 1
        fltr = get_edge_filter(batch_chunk_edge_index, offset)
        batch_chunk_edge_index = batch_chunk_edge_index[:, fltr]
        for k, v in edge_fields_dict.items():
            edge_fields_dict[k] = v[fltr]

    # = ----------------------------------------- = #

    if chunk:
        batch_chunk_node_indices = batch_chunk_edge_index.unique()
        batch_chunk[AtomicDataDict.EDGE_INDEX_KEY] = batch_chunk_edge_index
        batch_chunk[AtomicDataDict.BATCH_KEY]      = batch[AtomicDataDict.BATCH_KEY][batch_chunk_node_indices]
        for k, v in edge_fields_dict.items():
            batch_chunk[k] = v
        for per_node_output_key in per_node_outputs_keys:
            chunk_per_node_outputs_value = batch[per_node_output_key].clone()
            mask = torch.ones_like(chunk_per_node_outputs_value, dtype=torch.bool)
            mask[batch_chunk_edge_index[0].unique()] = False
            chunk_per_node_outputs_value[mask] = torch.nan
            batch_chunk[per_node_output_key] = chunk_per_node_outputs_value

    # === ---------------------------------------------------- === #
    # === ---------------------------------------------------- === #

    batch_chunk["ptr"] = torch.nn.functional.pad(torch.bincount(batch_chunk.get(
        AtomicDataDict.BATCH_KEY)).flip(dims=[0]), (0, 1), mode='constant').flip(dims=[0])

    edge_index = batch_chunk[AtomicDataDict.EDGE_INDEX_KEY]
    node_index = edge_index.unique(sorted=True)

    for key in batch_chunk.keys():
        if key in [
            AtomicDataDict.BATCH_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
        ] + ignore_chunk_keys:
            continue
        dim = np.argwhere(np.array(batch_chunk[key].size()) == len(batch_chunk[AtomicDataDict.BATCH_KEY])).flatten()
        if len(dim) == 1:
            if dim[0] == 0:
                batch_chunk[key] = batch_chunk[key][node_index]
            elif dim[0] == 1:
                batch_chunk[key] = batch_chunk[key][:, node_index]
            elif dim[0] == 2:
                batch_chunk[key] = batch_chunk[key][:, :, node_index]
            else:
                raise Exception('Dimension not implemented')

    last_idx = -1
    updated_edge_index = edge_index.clone()
    for idx in node_index:
        if idx > last_idx + 1:
            updated_edge_index[edge_index >= idx] -= idx - last_idx - 1
        last_idx = idx
    batch_chunk[AtomicDataDict.EDGE_INDEX_KEY] = updated_edge_index
    batch_chunk_center_nodes = edge_index[0].unique() # original center node indices

    del edge_index
    del node_index

    input_data = {
        k: v.to(device)
        for k, v in batch_chunk.items()
        if k not in output_keys
    }

    return input_data, batch_chunk, batch_chunk_center_nodes

def apply_dropout_edges(dropout_edges, input_data):
    edge_index = input_data[AtomicDataDict.EDGE_INDEX_KEY]
    num_edges = edge_index.size(1)
    num_dropout_edges = int(dropout_edges * num_edges)

        # Randomly select edges to drop
    drop_edges = torch.randperm(num_edges, device=edge_index.device)[:num_dropout_edges]
    keep_edges = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
    keep_edges[drop_edges] = False

        # Ensure at least some edges per node center
    node_centers = edge_index[0].unique()
    remaining_node_centers = edge_index[0, keep_edges].unique()
    combined = torch.cat((node_centers, remaining_node_centers))
    uniques, counts = combined.unique(return_counts=True)
    dropped_out_node_centers = uniques[counts == 1]
    for node in dropped_out_node_centers:
        node_edges = (edge_index[0] == node).nonzero(as_tuple=True)[0]
        keep_edges[node_edges[torch.randint(len(node_edges), (max(1, int((1-dropout_edges)*len(node_edges))),))]] = True

    input_data[AtomicDataDict.EDGE_INDEX_KEY] = edge_index[:, keep_edges]
    if AtomicDataDict.EDGE_CELL_SHIFT_KEY in input_data:
        input_data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = input_data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][keep_edges]