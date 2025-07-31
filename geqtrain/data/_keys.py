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

# (1)
R_MAX_KEY:             Final[str] = "r_max"           # cutoff radius
# [n_nodes, 3]
POSITIONS_KEY:         Final[str] = "pos"             # positions of the nodes in the system
# [2, n_edges]
EDGE_INDEX_KEY:        Final[str] = "edge_index"      # index tensor giving center -> neighbor relations
# (bool) | [3]
PBC_KEY:               Final[str] = "pbc"             # wether to use pbc or not and on which axis
# [n_edges, 3]
EDGE_CELL_SHIFT_KEY:   Final[str] = "edge_cell_shift" # tensor of how many periodic cells each edge crosses in each cell vector
# [n_batches, 3, 3] | [3, 3]
CELL_KEY:              Final[str] = "cell"            # tensor where rows are the cell vectors
# [n_nodes]
NODE_TYPE_KEY:         Final[str] = "node_types"
# [n_edges]
EDGE_TYPE_KEY:         Final[str] = "edge_types"
# [n_nodes]
BATCH_KEY:             Final[str] = "batch"           # index tensor of the node batch
# [n_batches]
DATASET_RAW_FILE_NAME: Final[str] = "dataset_raw_file_name" # dataset raw file names
# (1)
NOISE_KEY:             Final[str] = "noise"           # noise level to inject to coordinates

INPUT_STRUCTURE_KEYS:  Final[List[str]] = [
    POSITIONS_KEY,
    EDGE_INDEX_KEY,
    NODE_TYPE_KEY,
    EDGE_TYPE_KEY,
]

# [n_nodes, dim]
NODE_INPUT_ATTRS_KEY:    Final[str] = "node_input_attrs"    # node scalar input features
# [n_nodes, dim]
NODE_EQ_INPUT_ATTRS_KEY: Final[str] = "node_eq_input_attrs" # node equivariant input features
# [n_nodes, dim]
NODE_ATTRS_KEY:          Final[str] = "node_attrs"          # node scalar attributes (attributes do not change once computed)
# [n_nodes, dim]
NODE_EQ_ATTRS_KEY:       Final[str] = "node_eq_attrs"       # node equivariant attributes (attributes do not change once computed)
# [n_nodes, dim]
NODE_FEATURES_KEY:       Final[str] = "node_features"       # processed version of NODE_ATTRS_KEY and NODE_EQ_ATTRS_KEY

# [n_edges, dim]
EDGE_INPUT_ATTRS_KEY:    Final[str] = "edge_input_attrs"    # edge scalar input features
# [n_edges, dim]
EDGE_EQ_INPUT_ATTRS_KEY: Final[str] = "edge_eq_input_attrs" # edge equivariant input features
# [n_edges, dim]
EDGE_ATTRS_KEY:          Final[str] = "edge_attrs"          # edge scalar attributes (attributes do not change once computed)
# [n_edges, dim]
EDGE_EQ_ATTRS_KEY:       Final[str] = "edge_eq_attrs"       # edge equivariant attributes (attributes do not change once computed)
# [n_edges, dim]
EDGE_FEATURES_KEY:       Final[str] = "edge_features"       # processed version of EDGE_ATTRS_KEY and EDGE_EQ_ATTRS_KEY
# [n_edges, 3]
EDGE_VECTORS_KEY:        Final[str] = "edge_vectors"        # displacement vectors associated to edges
# [n_edges]
EDGE_LENGTH_KEY:         Final[str] = "edge_lengths"        # lengths of EDGE_VECTORS_KEY
# [n_edges, dim]
EDGE_SPHARMS_EMB_KEY:    Final[str] = "spharms_emb"         # spherical harmonics, embedding of EDGE_VECTORS_KEY (angular component)
# [n_edges, dim]
EDGE_RADIAL_EMB_KEY:     Final[str] = "radial_emb"          # radial basis functions, embedding of EDGE_LENGTH_KEY (radial component)

# [n_graphs, dim]
GRAPH_ATTRS_KEY:         Final[str] = "graph_attrs"         # graph scalar attributes (attributes do not change once computed)
# [n_graphs, dim]
GRAPH_FEATURES_KEY:      Final[str] = "graph_features"      # processed version of GRAPH_ATTRS_KEY

# Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]
