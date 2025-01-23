"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""

import sys
from typing import List, Final

"""
     - Fixed field means that it is system-dependent and not batch-dependent.
     - Node- and edge-types are integers used to embed the input and create node/edge attributes.
     - Node- and edge-attributes remain fixed through the network. They are properties of nodes/edges.
     - Node- and edge-features change and are updated through the network. They represent the hidden state.
"""

### == Define allowed keys as constants == ###

R_MAX_KEY: Final[str] = "r_max"
# The positions of the nodes in the system
POSITIONS_KEY: Final[str] = "pos"
# [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX_KEY: Final[str] = "edge_index"
# [bool] | [3] wether to use pbc or not and on which axis
PBC_KEY: Final[str] = "pbc"
# [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = "edge_cell_shift"
# [n_batch, 3, 3] or [3, 3] tensor where rows are the cell vectors
CELL_KEY: Final[str] = "cell"
# [n_nodes] long tensor
ATOM_NUMBER_KEY: Final[str] = "atom_numbers"
# [n_nodes] long tensor
NODE_TYPE_KEY: Final[str] = "node_types"
# [n_edge] long tensor
EDGE_TYPE_KEY: Final[str] = "edge_types"

# [n_batch_nodes] index tensor of the node batch
BATCH_KEY: Final[str] = "batch"
# [n_batch_nodes] list of dataset raw file names
DATASET_RAW_FILE_NAME: Final[str] = "dataset_raw_file_name"

INPUT_STRUCTURE_KEYS: Final[List[str]] = [
    POSITIONS_KEY,
    EDGE_INDEX_KEY,
    NODE_TYPE_KEY,
    EDGE_TYPE_KEY,
]

# [n_nodes, dim] (possibly equivariant) node input attributes
NODE_ATTRS_KEY: Final[str] = "node_attrs" # usually one-hot encoded input category
# [n_nodes, dim] (possibly equivariant) features of each node
NODE_FEATURES_KEY: Final[str] = "node_features" # the processed version of NODE_ATTRS_KEY, used to do not overwrite NODE_ATTRS_KEY
# [n_nodes, dim] (possibly equivariant) output features of each node
NODE_OUTPUT_KEY: Final[str] = "node_output"

# [n_edges, dim] (possibly equivariant) edge input attributes
EDGE_ATTRS_KEY: Final[str] = "edge_attrs" # look at difference between NODE_ATTRS_KEY and NODE_FEATURES_KEY
# [n_edges, dim] (possibly equivariant) features of the edges
EDGE_FEATURES_KEY: Final[str] = "edge_features" # look at difference between NODE_FEATURES_KEY and NODE_ATTRS_KEY
# [n_edges, dim] (possibly equivariant) output features of the edges
EDGE_OUTPUT_KEY: Final[str] = "edge_output"
# [n_edges, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
# [n_edges] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH_KEY: Final[str] = "edge_lengths"
# [n_edges, dim] equivariant angular attributes of the edges
EDGE_ANGULAR_ATTRS_KEY: Final[str] = "edge_angular_attrs"
# [n_edges, dim] invariant radial attributes of the edges
EDGE_RADIAL_ATTRS_KEY: Final[str] = "edge_radial_attrs"

# [n_graphs, dim] invariant graph input attributes
GRAPH_ATTRS_KEY: Final[str] = "graph_attrs"
# [n_graphs, dim] (possibly equivariant) graph features of graph
GRAPH_FEATURES_KEY: Final[str] = "graph_features"
# [n_graphs, dim] (possibly equivariant) output features of graph
GRAPH_OUTPUT_KEY: Final[str] = "graph_output"

NOISE_KEY: Final[str] = "noise"

# Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]
