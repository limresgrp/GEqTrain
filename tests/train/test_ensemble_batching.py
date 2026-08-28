import numpy as np
import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.data.dataloader import Collater
from geqtrain.train._loss import prepare_target
from geqtrain.train.sampler import EnsembleBatchSampler
from geqtrain.utils.torch_geometric import Data


register_fields(node_fields=["cs_iso"])


class _EnsembleDataset:
    n_observations = np.asarray([3, 2])
    ensemble_indices = np.asarray([10, 20])


def _frame(ensemble_index: int, offset: float = 0.0) -> Data:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [1, 2, 0, 3, 0, 3, 1, 2],
        ],
        dtype=torch.long,
    )
    return Data(
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        + offset,
        edge_index=edge_index,
        node_types=torch.tensor([1, 6, 7, 8], dtype=torch.long),
        cs_iso=torch.tensor([10.0, 20.0, float("nan"), float("nan")]),
        ensemble_index=ensemble_index,
    )


def test_ensemble_batch_sampler_keeps_systems_together_and_caps_structures():
    sampler = EnsembleBatchSampler(
        _EnsembleDataset(),
        batch_size=1,
        shuffle=False,
        max_structures=2,
    )

    assert list(sampler) == [[0, 1], [3, 4]]
    assert len(sampler) == 2


def test_ensemble_atom_cap_selects_same_centers_and_keeps_neighbors():
    batch = Collater(
        ensemble_mode=True,
        ensemble_max_atoms=2,
        shuffle_ensemble_atoms=False,
    ).collate([_frame(7), _frame(7, offset=0.1)])

    assert batch.num_graphs == 2
    assert batch.num_nodes == 8
    assert torch.equal(
        batch[AtomicDataDict.ENSEMBLE_ATOM_INDEX_KEY],
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )

    edge_graphs = batch.batch[batch.edge_index[0]]
    local_sources = batch.edge_index[0] - batch.ptr[edge_graphs]
    assert set(local_sources.tolist()) == {0, 1}
    assert set(batch.edge_index[1].sub(batch.ptr[batch.batch[batch.edge_index[1]]]).tolist()) == {0, 1, 2, 3}


def test_prepare_target_averages_matching_atoms_across_conformers():
    batch = Collater(
        ensemble_mode=True,
        ensemble_max_atoms=2,
        shuffle_ensemble_atoms=False,
    ).collate([_frame(7), _frame(7, offset=0.1)])
    ref = batch.to_dict()
    pred = dict(ref)
    pred["cs_iso"] = torch.tensor(
        [[1.0], [2.0], [0.0], [0.0], [3.0], [4.0], [0.0], [0.0]],
        requires_grad=True,
    )

    prepared = prepare_target(
        pred=pred,
        ref=ref,
        key="cs_iso",
        pred_key_name="cs_iso",
        pred_key=pred["cs_iso"],
        ref_key=ref["cs_iso"],
        ignore_nan=True,
        aggregate_ensemble=True,
        ensemble_aggregation="mean",
    )

    assert torch.allclose(prepared.pred_key, torch.tensor([[2.0], [3.0]]))
    assert torch.allclose(prepared.ref_key, torch.tensor([[10.0], [20.0]]))
    assert torch.equal(prepared.node_types, torch.tensor([1, 6]))
