import numpy as np
import pytest
import torch

ase = pytest.importorskip("ase")

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.data._build import _filter_dataset
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


def test_nan_targets_remove_edge_centers_but_keep_neighbor_atoms(tmp_path):
    register_fields(node_fields=["cs_iso"])
    npz_path = tmp_path / "nan_targets.npz"
    np.savez(
        npz_path,
        pos=np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32),
        atom_types=np.array([[1, 6, 8]], dtype=np.int64),
        cs_iso=np.array([[[1.0], [np.nan], [np.nan]]], dtype=np.float32),
    )
    dataset = NpzDataset(
        root=str(tmp_path / "processed"),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={"pos": "pos", "atom_types": "node_types", "cs_iso": "cs_iso"},
        extra_fixed_fields={AtomicDataDict.R_MAX_KEY: 1.1},
    )

    filtered = _filter_dataset(dataset, ["cs_iso"])

    assert filtered is not None
    node_types = filtered.data[AtomicDataDict.NODE_TYPE_KEY].view(-1)
    center_types = node_types[filtered.data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()]
    assert torch.equal(torch.unique(center_types), torch.tensor([1]))
    assert set(node_types.tolist()) == {1, 6, 8}


def test_npz_frame_parallelism_matches_serial_build(tmp_path):
    frames = 4
    pos = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5 + 0.01 * i, 0.0, 0.0], [1.0, 0.0, 0.0]]
            for i in range(frames)
        ],
        dtype=np.float32,
    )
    npz_path = tmp_path / "parallel_frames.npz"
    np.savez(npz_path, pos=pos)
    kwargs = {
        "ensemble_index": 0,
        "file_name": str(npz_path),
        "key_mapping": {"pos": "pos"},
        "extra_fixed_fields": {AtomicDataDict.R_MAX_KEY: 1.1},
    }

    serial = NpzDataset(root=str(tmp_path / "serial"), frame_num_workers=1, **kwargs)
    parallel = NpzDataset(root=str(tmp_path / "parallel"), frame_num_workers=2, **kwargs)

    assert serial.data.num_graphs == parallel.data.num_graphs == frames
    assert torch.equal(serial.data[AtomicDataDict.EDGE_INDEX_KEY], parallel.data[AtomicDataDict.EDGE_INDEX_KEY])
