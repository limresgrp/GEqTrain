# tests/test_readout_module.py
import torch
from e3nn import o3
from geqtrain.data import AtomicDataDict
from geqtrain.nn.readout import ReadoutModule

def _dummy_atomic_dict(num_atoms, irreps_in):
    pos = torch.randn(num_atoms, 3)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    data = {
        AtomicDataDict.POSITIONS_KEY: pos,
        AtomicDataDict.BATCH_KEY: batch,
    }
    for key, irreps in irreps_in.items():
        if key in data: 
            continue
        ir = o3.Irreps(irreps)
        data[key] = ir.randn(num_atoms, -1)
    return data

def test_readout_single_field_scalar_out_and_bias():
    irreps_in = { "node_features": "4x0e+2x1o" }
    readout = ReadoutModule(
        irreps_in=irreps_in,
        field="node_features",
        out_field="node_features_out",
        scalar_out_field="node_scalars",
        out_irreps="8x0e+3x1o",
        readout_latent_kwargs={"mlp_latent_dimensions": [16]},
        bias=0.5,
    )

    data = _dummy_atomic_dict(num_atoms=7, irreps_in=irreps_in)
    out = readout(data)

    assert "node_features_out" in out
    assert "node_scalars" in out

    out_full = out["node_features_out"]
    out_scalars = out["node_scalars"]

    assert out_full.shape[0] == 7
    assert out_scalars.shape[0] == 7

    # Check that scalar_out_field matches first n_scalars_out entries
    n_scalars = readout.n_scalars_out
    torch.testing.assert_close(out_scalars, out_full[..., :n_scalars])

    # Check bias actually changed scalar mean
    mean_scalars = out_scalars.mean().item()
    assert abs(mean_scalars) > 1e-3, "Bias seems to have no effect on scalar outputs"

def test_readout_split_in_split_out_with_conditioning():
    irreps_in = {
        "node_scalars": "4x0e",
        "node_equiv": "2x1o",
        "graph_cond": "3x0e",
    }
    readout = ReadoutModule(
        irreps_in=irreps_in,
        invariant_field="node_scalars",
        equivariant_field="node_equiv",
        invariant_out_field="node_scalars_out",
        equivariant_out_field="node_equiv_out",
        conditioning_fields=["graph_cond"],
        readout_latent_kwargs={"mlp_latent_dimensions": [16]},
    )

    num_atoms = 10
    data = _dummy_atomic_dict(num_atoms=num_atoms, irreps_in=irreps_in)
    # graph_cond is graph-level; overwrite shape
    data["graph_cond"] = o3.Irreps("3x0e").randn(1, -1)  # single graph

    out = readout(data)

    assert "node_scalars_out" in out
    assert "node_equiv_out" in out

    assert out["node_scalars_out"].shape[0] == num_atoms
    assert out["node_equiv_out"].shape[0] == num_atoms

    # Check that total_conditioning_dim matches what we expect
    assert readout.total_conditioning_dim == o3.Irreps("3x0e").dim

def test_readout_resnet_behaves_near_identity():
    irreps_in = { "feat": "4x0e+1x1o" }
    readout = ReadoutModule(
        irreps_in=irreps_in,
        field="feat",
        out_field="feat",
        resnet=True,
        readout_latent_kwargs={"mlp_latent_dimensions": [8]},
    )

    data = _dummy_atomic_dict(num_atoms=6, irreps_in=irreps_in)
    x_old = data["feat"].clone()

    out = readout(data)
    x_new = out["feat"]

    # For coeff=0, we expect almost identity (modulo numerical noise)
    rel_err = (x_new - x_old).pow(2).mean().sqrt() / (x_old.pow(2).mean().sqrt() + 1e-12)
    assert rel_err < 0.3, f"ResNet init too far from identity: {rel_err.item()}"

from geqtrain.nn.readout import AttentionReadoutModule

def test_attention_readout_smoke_and_gradients():
    irreps_in = { "node_features": "4x0e+2x1o" }
    module = AttentionReadoutModule(
        irreps_in=irreps_in,
        field="node_features",
        out_field="node_features_out",
        out_irreps="4x0e+2x1o",
        readout_latent_kwargs={"mlp_latent_dimensions": [16]},
        num_heads=2,
    )

    data = _dummy_atomic_dict(num_atoms=9, irreps_in=irreps_in)
    # make batch index so _NODE_FIELDS path triggers
    data[AtomicDataDict.BATCH_KEY] = torch.zeros(9, dtype=torch.long)

    for v in data.values():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            v.requires_grad_(True)

    out = module(data)
    loss = out["node_features_out"].pow(2).mean()
    loss.backward()

    # check some grads exist
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "No gradient in AttentionReadoutModule"
