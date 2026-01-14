import numpy as np
import torch
import pytest
from e3nn.o3 import Irreps

from geqtrain.data import AtomicDataDict
from geqtrain.data.dataset import parse_attrs
from geqtrain.nn._embedding_attrs import EmbeddingInputAttrs


def _make_irreps_in(extra=None):
    irreps_in = {
        AtomicDataDict.POSITIONS_KEY: Irreps("1o"),
        AtomicDataDict.EDGE_INDEX_KEY: None,
    }
    if extra:
        irreps_in.update(extra)
    return irreps_in


def test_parse_attrs_numerical_binning():
    values = np.array([-5.0, 0.0, 4.9, 5.0, 10.0, 19.9, 20.0, np.nan], dtype=np.float32)
    fields = {"binned": values}
    fixed_fields = {}
    attrs = {
        "binned": {
            "attribute_type": "numerical",
            "embedding_mode": "one_hot",
            "num_types": 4,
            "actual_num_types": 5,
            "min_value": 0.0,
            "max_value": 20.0,
            "can_be_undefined": True,
        }
    }

    node_fields, _ = parse_attrs(attrs, fields, fixed_fields)
    out = node_fields["binned"].cpu().numpy().tolist()

    assert out == [0, 0, 0, 1, 2, 3, 3, 4]


def test_embedding_input_attrs_node_mixed_equivariant():
    num_atoms = 5
    attributes = {
        "cat_embed": {
            "attribute_type": "categorical",
            "embedding_mode": "embedding",
            "embedding_dimensionality": 3,
            "num_types": 4,
            "actual_num_types": 4,
        },
        "cat_hot": {
            "attribute_type": "categorical",
            "embedding_mode": "one_hot",
            "num_types": 2,
            "actual_num_types": 2,
        },
        "num_attr": {
            "attribute_type": "numerical",
            "embedding_dimensionality": 2,
        },
    }
    eq_attributes = {
        "eq_attr": {
            "attribute_type": "numerical",
            "irreps": "1x1o",
            "embedding_dimensionality": 3,
        }
    }

    module = EmbeddingInputAttrs(
        out_field=AtomicDataDict.NODE_INPUT_ATTRS_KEY,
        eq_out_field=AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY,
        attributes=attributes,
        eq_attributes=eq_attributes,
        irreps_in=_make_irreps_in(
            {
                "cat_embed": None,
                "cat_hot": None,
                "num_attr": Irreps("2x0e"),
                "eq_attr": Irreps("1x1o"),
            }
        ),
    )

    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3),
        "cat_embed": torch.tensor([[0], [1], [2], [3], [0]], dtype=torch.long),
        "cat_hot": torch.tensor([[1], [0], [1], [1], [0]], dtype=torch.long),
        "num_attr": torch.randn(num_atoms, 2),
        "eq_attr": torch.randn(num_atoms, 3),
    }

    out = module(data)
    assert AtomicDataDict.NODE_INPUT_ATTRS_KEY in out
    assert AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY in out

    node_attrs = out[AtomicDataDict.NODE_INPUT_ATTRS_KEY]
    node_eq = out[AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY]
    assert node_attrs.shape == (num_atoms, 7)
    assert node_eq.shape == (num_atoms, 3)
    assert node_attrs.dtype == data[AtomicDataDict.POSITIONS_KEY].dtype
    assert node_eq.dtype == data[AtomicDataDict.POSITIONS_KEY].dtype


def test_embedding_input_attrs_binned_numerical_one_hot():
    num_atoms = 4
    raw_vals = np.array([-1.0, 0.0, 3.9, np.nan], dtype=np.float32)
    fields = {"binned": raw_vals}
    attrs = {
        "binned": {
            "attribute_type": "numerical",
            "embedding_mode": "one_hot",
            "num_types": 4,
            "actual_num_types": 5,
            "min_value": 0.0,
            "max_value": 4.0,
            "can_be_undefined": True,
        }
    }
    node_fields, _ = parse_attrs(attrs, fields, {})

    module = EmbeddingInputAttrs(
        out_field=AtomicDataDict.NODE_INPUT_ATTRS_KEY,
        attributes=attrs,
        irreps_in=_make_irreps_in({"binned": None}),
    )
    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(num_atoms, 3),
        "binned": node_fields["binned"],
    }

    out = module(data)
    node_attrs = out[AtomicDataDict.NODE_INPUT_ATTRS_KEY]
    assert node_attrs.shape == (num_atoms, 5)
    assert torch.allclose(node_attrs.sum(dim=-1), torch.ones(num_atoms))
    assert node_attrs[-1, -1] == pytest.approx(1.0)


def test_embedding_input_attrs_edge_mix_equivariant():
    num_edges = 6
    attributes = {
        "edge_cat": {
            "attribute_type": "categorical",
            "embedding_mode": "one_hot",
            "num_types": 3,
            "actual_num_types": 3,
        },
        "edge_binned": {
            "attribute_type": "numerical",
            "embedding_mode": "embedding",
            "embedding_dimensionality": 4,
            "num_types": 5,
            "actual_num_types": 6,
            "min_value": 0.0,
            "max_value": 10.0,
            "can_be_undefined": True,
        },
    }
    eq_attributes = {
        "edge_eq": {
            "attribute_type": "numerical",
            "irreps": "1x1o",
            "embedding_dimensionality": 3,
        }
    }

    raw_vals = np.array([0.0, 1.0, 5.0, 9.9, 10.0, np.nan], dtype=np.float32)
    edge_fields, _ = parse_attrs(attributes, {"edge_binned": raw_vals}, {})

    module = EmbeddingInputAttrs(
        out_field=AtomicDataDict.EDGE_INPUT_ATTRS_KEY,
        eq_out_field=AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY,
        attributes=attributes,
        eq_attributes=eq_attributes,
        irreps_in=_make_irreps_in(
            {
                "edge_cat": None,
                "edge_binned": None,
                "edge_eq": Irreps("1x1o"),
            }
        ),
    )

    data = {
        AtomicDataDict.POSITIONS_KEY: torch.randn(3, 3),
        "edge_cat": torch.tensor([[0], [1], [2], [1], [0], [2]], dtype=torch.long),
        "edge_binned": edge_fields["edge_binned"],
        "edge_eq": torch.randn(num_edges, 3),
    }

    out = module(data)
    edge_attrs = out[AtomicDataDict.EDGE_INPUT_ATTRS_KEY]
    edge_eq = out[AtomicDataDict.EDGE_EQ_INPUT_ATTRS_KEY]
    assert edge_attrs.shape == (num_edges, 7)
    assert edge_eq.shape == (num_edges, 3)
