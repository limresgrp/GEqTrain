import torch
import logging
import contextlib
from typing import List, Optional
from geqtrain.data import AtomicData, AtomicDataDict, _NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS
from geqtrain.train.loss import Loss
from geqtrain.utils.torch_geometric import Batch


def get_output_keys(loss_fn: Loss):
    output_keys, per_node_outputs_keys = [], []
    if loss_fn is not None:
        for key in loss_fn.keys:
            key_clean = loss_fn.remove_suffix(key)
            if key_clean in _NODE_FIELDS.union(_GRAPH_FIELDS).union(_EDGE_FIELDS):
                output_keys.append(key_clean)
            if key_clean in _NODE_FIELDS:
                per_node_outputs_keys.append(key_clean)
    return output_keys, per_node_outputs_keys

def run_inference(
    model,
    data: Batch,
    device,
    config: dict,
    loss_fn: Optional[Loss] = None,
    already_computed_nodes=None,
    is_train: bool=False,
):
    """
    Runs inference for a single batch, extracting options from the config object.
    """
    mixed_precision = config.get('mixed_precision', False)
    chunking = config.get('chunking', False)
    batch_max_atoms = config.get('batch_max_atoms', 1000)
    chunk_ignore_keys = config.get('chunk_ignore_keys', [])
    dropout_edges = config.get('dropout_edges', 0.0)
    requires_grad = config.get('model_requires_grads', False)
    is_ddp = config.get('ddp', False)

    output_keys, per_node_outputs_keys = get_output_keys(loss_fn)
    has_graph_level_loss = any(key in _GRAPH_FIELDS for key in output_keys)
    effective_chunking = chunking and not has_graph_level_loss

    if chunking and has_graph_level_loss:
        graph_keys = [k for k in output_keys if k in _GRAPH_FIELDS]
        logging.warning(
            f"Chunking was enabled, but a graph-level loss was detected ({graph_keys}). "
            "Disabling chunking for this batch."
        )

    precision = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
    
    batch = data.to(device)
    batch_center_nodes = batch[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
    num_batch_center_nodes = len(batch_center_nodes)

    if effective_chunking:
        batch, batch_center_nodes = prepare_chunked_input_data(
            batch=batch,
            already_computed_nodes=already_computed_nodes,
            batch_max_atoms=batch_max_atoms,
            chunk_ignore_keys=chunk_ignore_keys,
        )
    
    input_data = {
        k: v.requires_grad_() if requires_grad and torch.is_floating_point(v) else v
        for k, v in batch
        if k not in output_keys and isinstance(v, torch.Tensor)
    }
    # Add batch.__slices__ with '_slices' suffix to input_data
    if hasattr(batch, "__slices__"):
        for k, v in batch.__slices__.items():
            input_data[f"{k}_slices"] = torch.tensor(v, device=device, dtype=torch.long)
    ref_data = batch.to_dict()

    # Remove target keys from input_data
    for key in output_keys:
        if key in input_data:
            del input_data[key]

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
    batch: Batch,
    already_computed_nodes: Optional[torch.Tensor],
    batch_max_atoms: int = 1000,
    chunk_ignore_keys: List[str] = [],
):
    """
    Prepares a chunk of data for processing by creating a subgraph.
    This function now operates directly on the Batch object.
    """
    chunk_edge_index = batch.edge_index.clone()

    if already_computed_nodes is not None:
        # Filter out edges where the source node has already been computed
        edge_mask = ~torch.isin(chunk_edge_index[0], already_computed_nodes)
        chunk_edge_index = chunk_edge_index[:, edge_mask]

    if len(chunk_edge_index[0].unique()) == 0:
        return None, None, None

    # Determine the center nodes for this chunk based on batch_max_atoms
    offset = 0
    while len(chunk_edge_index.unique()) > batch_max_atoms:

        def get_node_center_idcs(chunk_edge_index: torch.Tensor, batch_max_atoms: int, offset: int):
            unique_set = set()

            for i, num in enumerate(chunk_edge_index[1]):
                unique_set.add(num.item())

                if len(unique_set) >= batch_max_atoms:
                    node_center_idcs = chunk_edge_index[0, :i+1].unique()
                    if len(node_center_idcs) == 1:
                        num_nodes = torch.isin(chunk_edge_index[0], node_center_idcs).sum()
                        if num_nodes > batch_max_atoms:
                            raise ValueError(
                                f"At least one node in the graph has more neighbors ({num_nodes}) "
                                f"than the maximum allowed number of atoms in a batch ({batch_max_atoms}). "
                                "Please increase the value of 'batch_max_atoms' in the config file."
                            )
                        return node_center_idcs
                    return node_center_idcs[:-offset]
            return chunk_edge_index[0].unique()

        def get_edge_filter(chunk_edge_index: torch.Tensor, offset: int):
            node_center_idcs = get_node_center_idcs(chunk_edge_index, batch_max_atoms, offset)
            edge_filter = torch.isin(chunk_edge_index[0], node_center_idcs)
            return edge_filter

        offset += 1
        fltr = get_edge_filter(chunk_edge_index, offset)
        chunk_edge_index = chunk_edge_index[:, fltr]
    
    # Identify all nodes (centers and neighbors) that are part of this chunk
    chunk_nodes = chunk_edge_index.unique()
    chunk_center_nodes = chunk_edge_index[0].unique()

    # Create the subgraph using our robust subgraph method
    batch_chunk = batch.subgraph(chunk_nodes, chunk_edge_index, chunk_ignore_keys)

    if batch_chunk is None:
        return None, None
    
    return batch_chunk, chunk_center_nodes

def apply_dropout_edges(dropout_edges, input_data):
    edge_index = input_data[AtomicDataDict.EDGE_INDEX_KEY]
    num_edges = edge_index.size(1)
    num_dropout_edges = int(dropout_edges * num_edges)

    drop_edges = torch.randperm(num_edges, device=edge_index.device)[:num_dropout_edges]
    keep_edges = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
    keep_edges[drop_edges] = False

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

