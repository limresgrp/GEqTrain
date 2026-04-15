import torch
from e3nn import o3

from geqtrain.data import AtomicData, AtomicDataDict
from geqtrain.nn import (
    BasisEdgeRadialAttrs,
    EdgewiseReduce,
    EmbeddingAttrs,
    EmbeddingInputAttrs,
    InteractionModule,
    ReadoutModule,
    SequentialGraphNetwork,
    SphericalHarmonicEdgeAngularAttrs,
)
from tests.utils.equivariance import assert_AtomicData_equivariant


def _make_graph(num_nodes: int = 6, r_max: float = 5.0):
    data = AtomicData.from_points(
        pos=torch.randn(num_nodes, 3),
        r_max=r_max,
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )
    return AtomicData.to_AtomicDataDict(data)


def _fill_irrep_field(data, key: str, irreps: str, size: int):
    data[key] = o3.Irreps(irreps).randn(size, -1)


def test_readout_equivariant_with_custom_node_equivariant_input():
    num_nodes = 7
    data = _make_graph(num_nodes=num_nodes)

    irreps_in = {"latent_custom": "6x0e+2x1o"}
    _fill_irrep_field(data, "latent_custom", irreps_in["latent_custom"], num_nodes)

    module = ReadoutModule(
        irreps_in=irreps_in,
        field="latent_custom",
        out_field="velocity_custom",
        out_irreps="1x1o",
        readout_latent_kwargs={"mlp_latent_dimensions": [16]},
    )

    assert_AtomicData_equivariant(func=module, data_in=data, ntrials=1)


def test_interaction_equivariant_with_custom_equivariant_inputs():
    num_nodes = 6
    data = _make_graph(num_nodes=num_nodes)
    num_edges = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

    irreps_in = {
        "node_scalar_custom": "8x0e",
        "node_eq_custom": "2x1o",
        "edge_radial_custom": "4x0e",
        "edge_spharm_custom": "1x0e+1x1o+1x2e",
        "edge_scalar_custom": "3x0e",
        "edge_eq_custom": "1x1o",
    }
    _fill_irrep_field(data, "node_scalar_custom", irreps_in["node_scalar_custom"], num_nodes)
    _fill_irrep_field(data, "node_eq_custom", irreps_in["node_eq_custom"], num_nodes)
    _fill_irrep_field(data, "edge_radial_custom", irreps_in["edge_radial_custom"], num_edges)
    _fill_irrep_field(data, "edge_spharm_custom", irreps_in["edge_spharm_custom"], num_edges)
    _fill_irrep_field(data, "edge_scalar_custom", irreps_in["edge_scalar_custom"], num_edges)
    _fill_irrep_field(data, "edge_eq_custom", irreps_in["edge_eq_custom"], num_edges)

    module = InteractionModule(
        num_layers=2,
        latent_dim=16,
        eq_latent_multiplicity=4,
        node_invariant_field="node_scalar_custom",
        node_equivariant_field="node_eq_custom",
        edge_invariant_field="edge_scalar_custom",
        edge_equivariant_field="edge_eq_custom",
        edge_spharm_emb_field="edge_spharm_custom",
        edge_radial_emb_field="edge_radial_custom",
        out_field="edge_features_custom",
        use_attention=False,
        use_mace_product=False,
        irreps_in=irreps_in,
    )

    assert_AtomicData_equivariant(func=module, data_in=data, ntrials=1)


def test_template_like_stack_equivariant_with_eq_input_attributes():
    num_nodes = 6
    data = _make_graph(num_nodes=num_nodes, r_max=7.0)

    data["shape_scalar_features"] = torch.randn(num_nodes, 4)
    data["dipole_strength"] = torch.randn(num_nodes, 1)
    data["ligand_mask"] = torch.randn(num_nodes, 1)
    data["pocket_mask"] = torch.randn(num_nodes, 1)
    data["conditioning"] = torch.randn(num_nodes, 16)
    _fill_irrep_field(data, "shape_equiv_features", "1x1o+1x2e+1x3o", num_nodes)
    _fill_irrep_field(data, "dipole_direction", "1x1o", num_nodes)

    node_input_attrs = EmbeddingInputAttrs(
        attributes={
            "shape_scalar_features": {"attribute_type": "numerical", "embedding_dimensionality": 4},
            "dipole_strength": {"attribute_type": "numerical", "embedding_dimensionality": 1},
            "ligand_mask": {"attribute_type": "numerical", "embedding_dimensionality": 1},
            "pocket_mask": {"attribute_type": "numerical", "embedding_dimensionality": 1},
            "conditioning": {"attribute_type": "numerical", "embedding_dimensionality": 16},
        },
        eq_attributes={
            "shape_equiv_features": {"attribute_type": "numerical", "irreps": "1x1o+1x2e+1x3o", "embedding_dimensionality": 15},
            "dipole_direction": {"attribute_type": "numerical", "irreps": "1x1o", "embedding_dimensionality": 3},
        },
        out_field=AtomicDataDict.NODE_INPUT_ATTRS_KEY,
        eq_out_field=AtomicDataDict.NODE_EQ_INPUT_ATTRS_KEY,
        irreps_in={},
    )
    edge_radial_attrs = BasisEdgeRadialAttrs(
        basis_kwargs={"r_max": 7.0, "num_basis": 16},
        cutoff_kwargs={"r_max": 7.0},
        irreps_in=node_input_attrs.irreps_out,
    )
    edge_angular_attrs = SphericalHarmonicEdgeAngularAttrs(
        irreps_edge_sh=None,
        l_max=3,
        parity="o3_full",
        irreps_in=edge_radial_attrs.irreps_out,
    )
    attrs = EmbeddingAttrs(
        irreps_in=edge_angular_attrs.irreps_out,
        node_out_irreps="32x0e",
        edge_out_irreps="32x0e",
    )
    interaction = InteractionModule(
        num_layers=2,
        latent_dim=64,
        eq_latent_multiplicity=8,
        use_attention=False,
        use_mace_product=False,
        irreps_in=attrs.irreps_out,
    )
    edge_pooling = EdgewiseReduce(
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.NODE_FEATURES_KEY,
        use_attention=False,
        irreps_in=interaction.irreps_out,
    )
    position_head = ReadoutModule(
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field="velocity",
        out_irreps="1x1o",
        normalize_equivariant_output=False,
        readout_latent_kwargs={"mlp_latent_dimensions": [64], "mlp_nonlinearity": "silu"},
        irreps_in=edge_pooling.irreps_out,
    )
    shape_equiv_head = ReadoutModule(
        field=AtomicDataDict.NODE_FEATURES_KEY,
        out_field="shape_equiv_velocity",
        out_irreps="1x1o+1x2e+1x3o",
        normalize_equivariant_output=False,
        readout_latent_kwargs={"mlp_latent_dimensions": [64], "mlp_nonlinearity": "silu"},
        irreps_in=position_head.irreps_out,
    )

    model = SequentialGraphNetwork([
        node_input_attrs,
        edge_radial_attrs,
        edge_angular_attrs,
        attrs,
        interaction,
        edge_pooling,
        position_head,
        shape_equiv_head,
    ])

    assert_AtomicData_equivariant(
        func=model,
        data_in=data,
        input_irreps_overrides={
            "shape_equiv_features": "1x1o+1x2e+1x3o",
            "dipole_direction": "1x1o",
        },
        ntrials=1,
    )
