import numpy as np
import torch

from geqtrain.data import AtomicDataDict, InMemoryConcatDataset
from geqtrain.train.components.dataset_builder import DatasetBuilder
from geqtrain.utils import Config
from geqtrain.utils.normalization import get_per_type_stat_keys


class _MiniDataset(torch.utils.data.Dataset):
    def __init__(self, ensemble_index: int, values: torch.Tensor, mean: float, std: float):
        super().__init__()
        self.ensemble_index = ensemble_index
        self.data = {
            "target": values.clone(),
            AtomicDataDict.NODE_TYPE_KEY: torch.zeros((values.shape[0], 1), dtype=torch.long),
        }
        mean_key, std_key = get_per_type_stat_keys("target")
        self.fixed_fields = {
            mean_key: torch.tensor([[mean]], dtype=values.dtype),
            std_key: torch.tensor([[std]], dtype=values.dtype),
        }
        self._normalization_specs = {
            "target": {
                "mode": "per_type",
                "irreps": "1x0e",
                "transform": {"name": "none"},
                "apply_on_dataset": True,
            }
        }

    def index_select(self, _idx):
        return self

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return {}


def test_validation_uses_train_normalization_when_splits_are_separate(monkeypatch):
    # Raw values are [1, 3].
    # Validation dataset is initially normalized with val stats mean=0,std=1 -> [1,3].
    # After alignment to train stats mean=10,std=2 expected normalized values are [-4.5,-3.5].
    train_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[0.0], [0.0]]), mean=10.0, std=2.0)
    val_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[1.0], [3.0]]), mean=0.0, std=1.0)
    train_concat = InMemoryConcatDataset([train_sub])
    val_concat = InMemoryConcatDataset([val_sub])

    def _fake_dataset_from_config(_config, prefix, metadata_only=False):
        assert not metadata_only
        if prefix == "train":
            return train_concat
        if prefix == "validation":
            return val_concat
        raise KeyError(prefix)

    monkeypatch.setattr(
        "geqtrain.train.components.dataset_builder.dataset_from_config",
        _fake_dataset_from_config,
    )

    cfg = Config.from_dict(
        {
            "normalization": {
                "target": {
                    "mode": "per_type:1x0e",
                    "transform": "none",
                }
            }
        }
    )
    builder = DatasetBuilder(cfg, np.random.default_rng(0))
    _, final_val = builder.build_datasets_from_indices(train_idcs=[[0]], val_idcs=[[0]])

    aligned_vals = final_val.datasets[0].data["target"]
    assert torch.allclose(aligned_vals, torch.tensor([[-4.5], [-3.5]]), atol=1e-6)

    mean_key, std_key = get_per_type_stat_keys("target")
    assert torch.allclose(final_val.datasets[0].fixed_fields[mean_key], torch.tensor([[10.0]]))
    assert torch.allclose(final_val.datasets[0].fixed_fields[std_key], torch.tensor([[2.0]]))


def test_validation_normalization_alignment_can_be_disabled(monkeypatch):
    train_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[0.0], [0.0]]), mean=10.0, std=2.0)
    val_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[1.0], [3.0]]), mean=0.0, std=1.0)
    train_concat = InMemoryConcatDataset([train_sub])
    val_concat = InMemoryConcatDataset([val_sub])

    def _fake_dataset_from_config(_config, prefix, metadata_only=False):
        assert not metadata_only
        if prefix == "train":
            return train_concat
        if prefix == "validation":
            return val_concat
        raise KeyError(prefix)

    monkeypatch.setattr(
        "geqtrain.train.components.dataset_builder.dataset_from_config",
        _fake_dataset_from_config,
    )

    cfg = Config.from_dict(
        {
            "share_train_normalization_across_splits": False,
            "normalization": {
                "target": {
                    "mode": "per_type:1x0e",
                    "transform": "none",
                }
            },
        }
    )
    builder = DatasetBuilder(cfg, np.random.default_rng(0))
    _, final_val = builder.build_datasets_from_indices(train_idcs=[[0]], val_idcs=[[0]])

    aligned_vals = final_val.datasets[0].data["target"]
    assert torch.allclose(aligned_vals, torch.tensor([[1.0], [3.0]]), atol=1e-6)


def test_test_dataset_uses_train_normalization_when_available(monkeypatch):
    train_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[0.0], [0.0]]), mean=10.0, std=2.0)
    test_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[1.0], [3.0]]), mean=0.0, std=1.0)
    train_concat = InMemoryConcatDataset([train_sub])
    test_concat = InMemoryConcatDataset([test_sub])

    def _fake_dataset_from_config(_config, prefix, metadata_only=False):
        assert not metadata_only
        if prefix == "train":
            return train_concat
        if prefix == "test":
            return test_concat
        raise KeyError(prefix)

    monkeypatch.setattr(
        "geqtrain.train.components.dataset_builder.dataset_from_config",
        _fake_dataset_from_config,
    )

    cfg = Config.from_dict(
        {
            "normalization": {
                "target": {
                    "mode": "per_type:1x0e",
                    "transform": "none",
                }
            }
        }
    )
    builder = DatasetBuilder(cfg, np.random.default_rng(0))
    final_test = builder.build_test()

    aligned_vals = final_test.datasets[0].data["target"]
    assert torch.allclose(aligned_vals, torch.tensor([[-4.5], [-3.5]]), atol=1e-6)


def test_test_dataset_keeps_own_normalization_if_train_missing(monkeypatch):
    test_sub = _MiniDataset(ensemble_index=0, values=torch.tensor([[1.0], [3.0]]), mean=0.0, std=1.0)
    test_concat = InMemoryConcatDataset([test_sub])

    def _fake_dataset_from_config(_config, prefix, metadata_only=False):
        assert not metadata_only
        if prefix == "test":
            return test_concat
        if prefix == "train":
            raise KeyError(prefix)
        raise KeyError(prefix)

    monkeypatch.setattr(
        "geqtrain.train.components.dataset_builder.dataset_from_config",
        _fake_dataset_from_config,
    )

    cfg = Config.from_dict(
        {
            "normalization": {
                "target": {
                    "mode": "per_type:1x0e",
                    "transform": "none",
                }
            }
        }
    )
    builder = DatasetBuilder(cfg, np.random.default_rng(0))
    final_test = builder.build_test()

    aligned_vals = final_test.datasets[0].data["target"]
    assert torch.allclose(aligned_vals, torch.tensor([[1.0], [3.0]]), atol=1e-6)
