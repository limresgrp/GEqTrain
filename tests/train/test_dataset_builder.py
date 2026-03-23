import numpy as np
import pytest
import torch

from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.utils import Config


class _MetadataDataset:
    def __init__(self, counts):
        self.n_observations = torch.tensor(counts, dtype=torch.long)


def _make_builder(monkeypatch, counts, config_dict=None):
    config = Config.from_dict(config_dict or {})

    def _fake_dataset_from_config(_config, prefix, metadata_only=False):
        assert metadata_only
        if prefix == "train":
            return _MetadataDataset(counts)
        if prefix == "validation":
            raise KeyError(prefix)
        raise AssertionError(f"Unexpected dataset prefix: {prefix}")

    monkeypatch.setattr(
        "geqtrain.train.components.dataset_builder.dataset_from_config",
        _fake_dataset_from_config,
    )
    return DatasetBuilder(config, np.random.default_rng(0))


def _assert_partition(counts, train_idcs, val_idcs):
    for n_obs, train_split, val_split in zip(counts, train_idcs, val_idcs):
        assert set(train_split).isdisjoint(val_split)
        assert sorted(train_split + val_split) == list(range(n_obs))


def test_dataset_builder_uses_default_80_20_split(monkeypatch):
    counts = [5, 5]
    builder = _make_builder(monkeypatch, counts)

    train_idcs, val_idcs = builder.resolve_split_indices()

    assert [len(split) for split in train_idcs] == [4, 4]
    assert [len(split) for split in val_idcs] == [1, 1]
    _assert_partition(counts, train_idcs, val_idcs)


def test_dataset_builder_uses_configured_train_split_fraction(monkeypatch):
    counts = [4, 6]
    builder = _make_builder(monkeypatch, counts, {"train_split_fraction": 0.9})

    train_idcs, val_idcs = builder.resolve_split_indices()

    assert [len(split) for split in train_idcs] == [4, 5]
    assert [len(split) for split in val_idcs] == [0, 1]
    _assert_partition(counts, train_idcs, val_idcs)


def test_dataset_builder_rejects_invalid_train_split_fraction(monkeypatch):
    builder = _make_builder(monkeypatch, [5, 5], {"train_split_fraction": 1.0})

    with pytest.raises(ValueError, match="train_split_fraction"):
        builder.resolve_split_indices()
