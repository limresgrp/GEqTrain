import numpy as np
import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.data._build import _filter_dataset, _node_types_to_exclude_from_edges, _node_types_to_keep
from geqtrain.data.dataset import NpzDataset
from geqtrain.train._loss import LossWrapper


TYPE_NAMES = ["X", "H", "He", "Li", "Be", "B", "C", "N"]
H, C, N = 1, 6, 7


def _write_masked_npz(path):
    """Write a small ShiftML3-style padded NPZ.

    ``__mask__`` values mark padded rows to discard before graph construction.
    NaNs in ``cs_iso`` mark atoms without supervision while keeping the atoms
    available as neighbors until dataset filtering chooses centers.
    """
    max_atoms = 4
    pos = np.zeros((3, max_atoms, 3), dtype=np.float32)
    atom_types = np.zeros((3, max_atoms), dtype=np.int64)
    cs_iso = np.full((3, max_atoms, 1), np.nan, dtype=np.float32)
    center_atoms_mask = np.zeros((3, max_atoms, 1), dtype=bool)
    row_mask = np.ones((3, max_atoms), dtype=bool)

    frames = [
        # H and C supervised; N remains a neighbor-only atom due NaN target.
        ([H, H, C, N], [1.0, 1.5, 6.0, np.nan], [True, False, False, True]),
        # No H in this frame, exercising empty filtered frames for H-only data.
        ([C, N], [7.0, np.nan], [True, True]),
        # H supervised; C remains neighbor-only due NaN target.
        ([H, H, N, C], [2.0, 2.5, 8.0, np.nan], [False, True, True, True]),
    ]

    for frame_idx, (types, targets, center_mask) in enumerate(frames):
        for atom_idx, (atom_type, target, keep_center) in enumerate(zip(types, targets, center_mask)):
            pos[frame_idx, atom_idx] = np.array([0.25 * atom_idx, 0.0, 0.0], dtype=np.float32)
            atom_types[frame_idx, atom_idx] = atom_type
            cs_iso[frame_idx, atom_idx, 0] = target
            center_atoms_mask[frame_idx, atom_idx, 0] = keep_center
            row_mask[frame_idx, atom_idx] = False

    np.savez(
        path,
        pos=pos,
        pos__mask__=np.repeat(row_mask[..., None], 3, axis=-1),
        atom_types=atom_types,
        atom_types__mask__=row_mask,
        cs_iso=cs_iso,
        cs_iso__mask__=np.repeat(row_mask[..., None], 1, axis=-1),
        center_atoms_mask=center_atoms_mask,
        center_atoms_mask__mask__=np.repeat(row_mask[..., None], 1, axis=-1),
    )


def _make_dataset(tmp_path, npz_path, name):
    register_fields(node_fields=["cs_iso", "center_atoms_mask"])
    return NpzDataset(
        root=str(tmp_path / name),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={
            "pos": AtomicDataDict.POSITIONS_KEY,
            "pos__mask__": f"{AtomicDataDict.POSITIONS_KEY}__mask__",
            "atom_types": AtomicDataDict.NODE_TYPE_KEY,
            "atom_types__mask__": f"{AtomicDataDict.NODE_TYPE_KEY}__mask__",
            "cs_iso": "cs_iso",
            "cs_iso__mask__": "cs_iso__mask__",
            "center_atoms_mask": "center_atoms_mask",
            "center_atoms_mask__mask__": "center_atoms_mask__mask__",
        },
        extra_fixed_fields={AtomicDataDict.R_MAX_KEY: 1.0},
    )


def _fixture_dataset(tmp_path, name="dataset"):
    npz_path = tmp_path / "toy_shiftml3_masked.npz"
    if not npz_path.exists():
        _write_masked_npz(npz_path)
    return _make_dataset(tmp_path, npz_path, name)


def _center_node_types(data):
    centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
    return data[AtomicDataDict.NODE_TYPE_KEY].view(-1)[centers]


def _neighbor_node_types(data):
    neigh = data[AtomicDataDict.EDGE_INDEX_KEY][1].unique()
    return data[AtomicDataDict.NODE_TYPE_KEY].view(-1)[neigh]


def test_nan_targets_remove_edge_centers_but_keep_neighbors_by_default(tmp_path):
    dataset = _fixture_dataset(tmp_path, "nan_centers")

    filtered = _filter_dataset(dataset, ["cs_iso"])

    assert filtered is not None
    data = filtered.data
    node_types = data[AtomicDataDict.NODE_TYPE_KEY].view(-1)
    center_types = _center_node_types(data)
    assert not torch.isnan(data["cs_iso"][data[AtomicDataDict.EDGE_INDEX_KEY][0]]).any()
    assert set(center_types.tolist()) == {H, C, N}
    assert (node_types == C).sum() > (center_types == C).sum()
    assert torch.isnan(data["cs_iso"][node_types == C]).any()


def test_keep_node_types_and_keep_type_names_prune_nodes_and_empty_frames(tmp_path):
    by_index = _filter_dataset(
        _fixture_dataset(tmp_path, "keep_node_types"),
        ["cs_iso"],
        keep_node_types=torch.tensor([H]),
    )
    by_name = _filter_dataset(
        _fixture_dataset(tmp_path, "keep_type_names"),
        ["cs_iso"],
        keep_node_types=_node_types_to_keep({"keep_type_names": ["H"], "type_names": TYPE_NAMES}),
    )

    assert by_index is not None
    assert by_name is not None
    for filtered in (by_index, by_name):
        data = filtered.data
        assert data.num_graphs == 2
        assert torch.equal(data[AtomicDataDict.NODE_TYPE_KEY].view(-1), torch.tensor([H, H, H, H]))
        assert data["center_atoms_mask"].shape[0] == data.num_nodes
        assert data["center_atoms_mask"].dtype == torch.float32


def test_exclude_type_names_from_edge_center_stacks_with_nan_center_filter(tmp_path):
    dataset = _fixture_dataset(tmp_path, "exclude_centers")
    exclude_center, exclude_neigh = _node_types_to_exclude_from_edges(
        {"exclude_type_names_from_edge_center": ["C"], "type_names": TYPE_NAMES}
    )

    filtered = _filter_dataset(dataset, ["cs_iso"], None, exclude_center, exclude_neigh)

    assert filtered is not None
    data = filtered.data
    center_types = _center_node_types(data)
    assert C not in center_types.tolist()
    assert not torch.isnan(data["cs_iso"][data[AtomicDataDict.EDGE_INDEX_KEY][0]]).any()
    assert data["center_atoms_mask"].shape[0] == data.num_nodes


def test_exclude_type_names_from_edge_neigh_removes_neighbors_only(tmp_path):
    dataset = _fixture_dataset(tmp_path, "exclude_neighbors")
    exclude_center, exclude_neigh = _node_types_to_exclude_from_edges(
        {"exclude_type_names_from_edge_neigh": ["H"], "type_names": TYPE_NAMES}
    )

    filtered = _filter_dataset(dataset, ["cs_iso"], None, exclude_center, exclude_neigh)

    assert filtered is not None
    data = filtered.data
    center_types = _center_node_types(data)
    neigh_types = _neighbor_node_types(data)
    assert H in center_types.tolist()
    assert H not in neigh_types.tolist()
    assert data["center_atoms_mask"].shape[0] == data.num_nodes


def test_prefiltered_node_mask_field_stays_aligned_for_loss_filtering(tmp_path):
    dataset = _filter_dataset(
        _fixture_dataset(tmp_path, "loss_mask_alignment"),
        ["cs_iso"],
        keep_node_types=torch.tensor([H, C]),
    )
    assert dataset is not None
    data = dataset.data
    assert data["center_atoms_mask"].shape[0] == data.num_nodes

    ref = data.to_dict()
    pred = dict(ref)
    pred["cs_iso"] = ref["cs_iso"].clone().requires_grad_(True) + 2.0
    loss = LossWrapper(
        "L1Loss",
        params={
            "node_type_names": ["H"],
            "type_names": TYPE_NAMES,
            "node_mask_field": "center_atoms_mask",
            "ignore_nan": True,
        },
    )

    out = loss(pred=pred, ref=ref, key="cs_iso", mean=False)

    centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
    selected = (
        (ref[AtomicDataDict.NODE_TYPE_KEY].view(-1) == H)
        & ref["center_atoms_mask"].view(-1).to(torch.bool)
    )
    selected &= torch.isin(torch.arange(data.num_nodes), centers)
    assert selected.sum().item() == 2
    assert out.shape == torch.Size([2, 1])
    assert torch.allclose(out, torch.full_like(out, 2.0))
