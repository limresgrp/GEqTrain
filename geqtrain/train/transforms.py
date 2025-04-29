import torch
from geqtrain.data import AtomicDataDict, _EDGE_FIELDS
from copy import deepcopy

def edges_dropout(data, dropout_edges: float = 0.05):

    data = deepcopy(data)

    edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
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

    for k,v in data:
        try:
            if k in _EDGE_FIELDS and v is not None and k != AtomicDataDict.EDGE_INDEX_KEY:
                data[k] = v[keep_edges]
        except:
            pass

    data[AtomicDataDict.EDGE_INDEX_KEY] = edge_index[:,keep_edges]
    return data