import numpy as np
import torch

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.nn.so3 import SO3_Linear


def test_atomic_data_float_fields_follow_default_dtype():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        data = AtomicData.from_points(
            pos=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
            r_max=1.0,
            node_types=np.array([0, 0], dtype=np.int64),
            target=np.array([[1.0], [2.0]], dtype=np.float32),
        )
    finally:
        torch.set_default_dtype(old_dtype)

    assert data[AtomicDataDict.POSITIONS_KEY].dtype == torch.float64
    assert data["target"].dtype == torch.float64
    assert data[AtomicDataDict.EDGE_INDEX_KEY].dtype == torch.long


def test_atomic_data_tensor_float_fields_follow_default_dtype():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        data = AtomicData.from_points(
            pos=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32),
            r_max=1.0,
            target=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        )
    finally:
        torch.set_default_dtype(old_dtype)

    assert data[AtomicDataDict.POSITIONS_KEY].dtype == torch.float64
    assert data["target"].dtype == torch.float64


def test_model_parameters_follow_default_dtype():
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        layer = SO3_Linear("1x0e", "1x0e", bias=True)
    finally:
        torch.set_default_dtype(old_dtype)

    assert next(layer.parameters()).dtype == torch.float64
