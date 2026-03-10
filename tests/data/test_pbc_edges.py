import numpy as np
import pytest
import torch

ase = pytest.importorskip("ase")

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.data.dataset import NpzDataset


def test_from_points_infers_pbc_from_cell_and_wraps_edges():
    pos = torch.tensor(
        [
            [0.05, 0.0, 0.0],
            [0.95, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cell = torch.eye(3, dtype=torch.float32)
    r_max = 0.2

    data = AtomicData.from_points(pos=pos, r_max=r_max, cell=cell)
    data_dict = AtomicData.to_AtomicDataDict(data)
    data_dict = AtomicDataDict.with_edge_vectors(data_dict, with_lengths=True)

    assert torch.equal(data_dict[AtomicDataDict.PBC_KEY], torch.tensor([True, True, True]))
    assert data_dict[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 2

    shifts = {tuple(x.tolist()) for x in data_dict[AtomicDataDict.EDGE_CELL_SHIFT_KEY].to(torch.int64)}
    assert shifts == {(-1, 0, 0), (1, 0, 0)}

    lengths = data_dict[AtomicDataDict.EDGE_LENGTH_KEY]
    assert torch.all(lengths <= r_max + 1e-6)
    assert torch.allclose(lengths, torch.tensor([0.1, 0.1]), atol=1e-6)


def test_from_points_without_cell_is_non_periodic():
    pos = torch.tensor(
        [
            [0.05, 0.0, 0.0],
            [0.95, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    r_max = 0.2

    data = AtomicData.from_points(pos=pos, r_max=r_max)

    assert torch.equal(data[AtomicDataDict.PBC_KEY], torch.tensor([False, False, False]))
    assert data[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 0


def test_from_points_explicit_non_pbc_overrides_cell():
    pos = torch.tensor(
        [
            [0.05, 0.0, 0.0],
            [0.95, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    cell = torch.eye(3, dtype=torch.float32)
    r_max = 0.2

    data = AtomicData.from_points(pos=pos, r_max=r_max, cell=cell, pbc=(False, False, False))

    assert torch.equal(data[AtomicDataDict.PBC_KEY], torch.tensor([False, False, False]))
    assert data[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 0


def test_npz_dataset_cell_mapping_enables_pbc_edges(tmp_path):
    pos = np.array(
        [
            [[0.05, 0.0, 0.0], [0.95, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    cell = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    npz_path = tmp_path / "pbc_toy.npz"
    np.savez(npz_path, pos=pos, Lattice=cell)

    r_max = 0.2
    dataset = NpzDataset(
        root=str(tmp_path),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={"pos": "pos", "Lattice": "cell"},
        extra_fixed_fields={AtomicDataDict.R_MAX_KEY: r_max},
    )

    data = dataset.data
    assert torch.equal(data[AtomicDataDict.PBC_KEY], torch.tensor([True, True, True]))
    assert data[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 2

    data_dict = data.to_dict()
    data_dict = AtomicDataDict.with_edge_vectors(data_dict, with_lengths=True)
    assert torch.all(data_dict[AtomicDataDict.EDGE_LENGTH_KEY] <= r_max + 1e-6)
