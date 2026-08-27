import torch

from geqtrain.data import AtomicDataDict
from geqtrain.utils._model_utils import prepare_conditioning_tensors


def test_prepare_conditioning_tensors_broadcasts_graph_attrs_to_nodes_and_edges():
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(5, 3),
        AtomicDataDict.BATCH_KEY: torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [
                [0, 0, 2, 3],
                [1, 2, 3, 4],
            ],
            dtype=torch.long,
        ),
        AtomicDataDict.GRAPH_ATTRS_KEY: torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    }

    node_cond, edge_cond = prepare_conditioning_tensors(
        data=data,
        conditioning_fields=[AtomicDataDict.GRAPH_ATTRS_KEY],
    )

    expected_node = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
            [3.0, 4.0],
        ]
    )
    expected_edge = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.0],
        ]
    )

    torch.testing.assert_close(node_cond, expected_node)
    torch.testing.assert_close(edge_cond, expected_edge)
