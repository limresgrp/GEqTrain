import numpy as np
import torch

from geqtrain.data import AtomicDataDict
from geqtrain.data.AtomicData import register_fields
from geqtrain.data.dataloader import Collater
from geqtrain.data._build import (
    _filter_dataset,
    _node_types_to_exclude_from_edges,
    _node_types_to_keep,
    _node_types_to_keep_for_edges,
)
from geqtrain.data.dataset import NpzDataset
from geqtrain.train._loss import LossWrapper
from geqtrain.utils.torch_geometric import Data


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


def test_multiple_target_availability_is_combined_with_logical_or(tmp_path):
    dataset = _fixture_dataset(tmp_path, "target_union")
    data = dataset.data
    data["aux_target"] = torch.full_like(data["cs_iso"], torch.nan)
    # Supervise one node that has no cs_iso target.
    missing_cs_node = torch.isnan(data["cs_iso"].view(-1)).nonzero()[0]
    data["aux_target"][missing_cs_node] = 1.0
    register_fields(node_fields=["aux_target"])

    filtered = _filter_dataset(dataset, ["cs_iso", "aux_target"])

    assert filtered is not None
    centers = filtered.data.edge_index[0].unique()
    assert torch.isfinite(filtered.data["aux_target"][centers]).any()
    assert torch.isnan(filtered.data["cs_iso"][centers]).any()


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


def test_keep_type_names_for_edge_center_keeps_other_types_as_neighbors(tmp_path):
    dataset = _fixture_dataset(tmp_path, "keep_centers")
    keep_center, keep_neigh = _node_types_to_keep_for_edges(
        {"keep_type_names_for_edge_center": ["H"], "type_names": TYPE_NAMES}
    )

    filtered = _filter_dataset(dataset, ["cs_iso"], None, None, None, keep_center, keep_neigh)

    assert filtered is not None
    data = filtered.data
    center_types = _center_node_types(data)
    node_types = data[AtomicDataDict.NODE_TYPE_KEY].view(-1)
    assert set(center_types.tolist()) == {H}
    assert C in node_types.tolist()
    assert N in node_types.tolist()
    assert not torch.isnan(data["cs_iso"][data[AtomicDataDict.EDGE_INDEX_KEY][0]]).any()


def test_keep_type_names_for_edge_neigh_removes_only_other_neighbors(tmp_path):
    dataset = _fixture_dataset(tmp_path, "keep_neighbors")
    keep_center, keep_neigh = _node_types_to_keep_for_edges(
        {"keep_type_names_for_edge_neigh": ["C", "N"], "type_names": TYPE_NAMES}
    )

    filtered = _filter_dataset(dataset, ["cs_iso"], None, None, None, keep_center, keep_neigh)

    assert filtered is not None
    data = filtered.data
    center_types = _center_node_types(data)
    neigh_types = _neighbor_node_types(data)
    assert H in center_types.tolist()
    assert H not in neigh_types.tolist()
    assert set(neigh_types.tolist()).issubset({C, N})


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


def test_keep_edge_center_filter_and_node_mask_field_are_combined(tmp_path):
    dataset = _fixture_dataset(tmp_path, "center_keep_mask_alignment")
    keep_center, keep_neigh = _node_types_to_keep_for_edges(
        {"keep_type_names_for_edge_center": ["C"], "type_names": TYPE_NAMES}
    )
    dataset = _filter_dataset(dataset, ["cs_iso"], None, None, None, keep_center, keep_neigh)

    assert dataset is not None
    data = dataset.data
    assert set(_center_node_types(data).tolist()) == {C}
    assert data["center_atoms_mask"].shape[0] == data.num_nodes

    ref = data.to_dict()
    pred = dict(ref)
    pred["cs_iso"] = ref["cs_iso"].clone().requires_grad_(True) + 2.0
    loss = LossWrapper(
        "L1Loss",
        params={
            "node_type_names": ["C"],
            "type_names": TYPE_NAMES,
            "node_mask_field": "center_atoms_mask",
            "ignore_nan": True,
        },
    )

    out = loss(pred=pred, ref=ref, key="cs_iso", mean=False)

    centers = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
    selected = (
        (ref[AtomicDataDict.NODE_TYPE_KEY].view(-1) == C)
        & ref["center_atoms_mask"].view(-1).to(torch.bool)
    )
    selected &= torch.isin(torch.arange(data.num_nodes), centers)
    assert selected.sum().item() == 1
    assert out.shape == torch.Size([1, 1])
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_collater_treats_missing_optional_node_mask_as_all_true():
    def with_mask():
        return Data(
            pos=torch.zeros(2, 3),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            center_atoms_mask=torch.tensor([[True], [False]]),
            ensemble_index=0,
        )

    def without_mask():
        return Data(
            pos=torch.zeros(3, 3),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            ensemble_index=0,
        )

    collater = Collater()
    for batch_list, expected in (
        ([with_mask(), without_mask()], torch.tensor([True, False, True, True, True])),
        ([without_mask(), with_mask()], torch.tensor([True, True, True, True, False])),
    ):
        batch = collater.collate(batch_list)
        assert "center_atoms_mask" in batch
        assert batch["center_atoms_mask"].shape == torch.Size([5, 1])
        assert torch.equal(batch["center_atoms_mask"].view(-1).to(torch.bool), expected)


def test_fixed_node_target_mask_inherits_fixed_status_and_preserves_alignment(tmp_path):
    register_fields(node_fields=["cs_iso"])
    npz_path = tmp_path / "fixed_target_mask.npz"
    pos = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.1, 0.0], [0.5, 0.1, 0.0], [1.0, 0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    node_types = np.array([H, C, N], dtype=np.int64)
    cs_iso = np.array([1.0, np.nan, 7.0], dtype=np.float32)
    cs_iso_valid = np.array([True, False, True], dtype=bool)
    np.savez(
        npz_path,
        coords=pos,
        atom_types=node_types,
        chemical_shifts=cs_iso,
        chemical_shift_mask=cs_iso_valid,
    )

    dataset = NpzDataset(
        root=str(tmp_path / "fixed_target_dataset"),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={
            "coords": AtomicDataDict.POSITIONS_KEY,
            "atom_types": AtomicDataDict.NODE_TYPE_KEY,
            "chemical_shifts": "cs_iso",
            "chemical_shift_mask": "cs_iso__mask__",
        },
        extra_fixed_fields={AtomicDataDict.R_MAX_KEY: 1.1},
        node_attributes={
            AtomicDataDict.NODE_TYPE_KEY: {"fixed": True, "num_types": 8},
            "cs_iso": {"fixed": True, "attribute_type": "numerical"},
        },
        normalization={"cs_iso": "per_type"},
    )

    assert len(dataset) == 2
    assert "cs_iso__mask__" not in dataset.fixed_fields
    assert "_mean_.per_type.cs_iso" in dataset.fixed_fields
    assert "_std_.per_type.cs_iso" in dataset.fixed_fields
    assert dataset.fixed_fields["cs_iso"].shape == torch.Size([3, 1])

    batch = Collater().collate([dataset[0], dataset[1]])
    assert batch[AtomicDataDict.POSITIONS_KEY].shape[0] == 6
    assert batch["cs_iso"].shape == torch.Size([6, 1])
    assert torch.equal(
        torch.isfinite(batch["cs_iso"]).view(-1),
        torch.tensor([True, False, True, True, False, True]),
    )
    assert torch.allclose(
        batch["cs_iso"][torch.isfinite(batch["cs_iso"])],
        torch.zeros(4),
    )


def test_graph_filter_promotes_and_aligns_fixed_node_fields(tmp_path):
    register_fields(node_fields=["cs_iso", "atom_role_ids"])
    npz_path = tmp_path / "fixed_fields_with_pruned_neighbors.npz"
    np.savez(
        npz_path,
        coords=np.array(
            [
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        atom_types=np.array([H, C, N], dtype=np.int64),
        atom_role_ids=np.array([10, 20, 30], dtype=np.int64),
        chemical_shifts=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    dataset = NpzDataset(
        root=str(tmp_path / "fixed_fields_with_pruned_neighbors_dataset"),
        ensemble_index=0,
        file_name=str(npz_path),
        key_mapping={
            "coords": AtomicDataDict.POSITIONS_KEY,
            "atom_types": AtomicDataDict.NODE_TYPE_KEY,
            "atom_role_ids": "atom_role_ids",
            "chemical_shifts": "cs_iso",
        },
        extra_fixed_fields={AtomicDataDict.R_MAX_KEY: 1.0},
        node_attributes={
            AtomicDataDict.NODE_TYPE_KEY: {"fixed": True, "num_types": 8},
            "atom_role_ids": {"fixed": True, "num_types": 31},
            "cs_iso": {"fixed": True, "attribute_type": "numerical"},
        },
    )

    keep_center, keep_neigh = _node_types_to_keep_for_edges(
        {"keep_type_names_for_edge_center": ["H"], "type_names": TYPE_NAMES}
    )
    dataset = _filter_dataset(
        dataset,
        ["cs_iso"],
        keep_node_types_for_edge_center=keep_center,
        keep_node_types_for_edge_neigh=keep_neigh,
    )

    assert dataset is not None
    assert AtomicDataDict.NODE_TYPE_KEY not in dataset.fixed_fields
    assert "atom_role_ids" not in dataset.fixed_fields
    assert "cs_iso" not in dataset.fixed_fields
    assert dataset.data.ptr.diff().tolist() == [2, 2]

    first, second = dataset[0], dataset[1]
    assert first[AtomicDataDict.NODE_TYPE_KEY].view(-1).tolist() == [H, C]
    assert second[AtomicDataDict.NODE_TYPE_KEY].view(-1).tolist() == [H, N]
    assert first["atom_role_ids"].view(-1).tolist() == [10, 20]
    assert second["atom_role_ids"].view(-1).tolist() == [10, 30]
    assert first["cs_iso"].view(-1).tolist() == [1.0, 2.0]
    assert second["cs_iso"].view(-1).tolist() == [1.0, 3.0]
