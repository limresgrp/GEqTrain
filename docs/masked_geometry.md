# Masked geometry reconstruction

`geqtrain.nn.MaskEdgeFeatures` performs optional **runtime** preprocessing inside
the normal model stack. No auxiliary-head API, separate trainer, or dataset
targets are needed. Use ordinary `ReadoutModule` heads and `loss_coeffs`.
Existing models do not change unless a masking module is added.

The complete example is
[`config/model/examples/masked_geometry.yaml`](../config/model/examples/masked_geometry.yaml).
It builds radial and spherical-harmonic embeddings, masks them independently,
passes them through `EmbeddingAttrs` and `InteractionModule`, and reconstructs
them with two edge readouts. Add the usual task heads and supervised loss entries
when incorporating it into an experiment.

## Placement and configuration

Insert a masker **after the feature producer but before its first consumer**:

```yaml
model:
  stack:
    - ${stack_blocks.node_input_attrs}
    - ${stack_blocks.edge_radial_attrs}
    - ${stack_blocks.edge_angular_attrs}
    - _target_: geqtrain.nn.MaskEdgeFeatures
      name: mask_radial
      field: radial_emb
      out_field: radial_reconstruction
      enabled: true
      mask_fraction: 0.15
      mask_value: 0.0
      mask_in_eval: false
    - _target_: geqtrain.nn.MaskEdgeFeatures
      name: mask_angular
      field: spharms_emb
      out_field: angular_reconstruction
      mask_fraction: 0.15
    - ${stack_blocks.attrs}
    # ... interaction modules, followed by the heads below ...
```

`radial_emb` and `spharms_emb` are GEqTrain's actual geometry embedding keys.
`field` can also name another declared edge feature with known irreps and shape
`[number_of_edges, irreps.dim]`; the implementation does not depend on a radial
basis class. The model must actually consume the masked field. If a model
recomputes geometry from positions or reads an unmasked copy, this does not hide
that alternative input. For example, masking `radial_emb` alone is insufficient
if a different architecture only consumes edge lengths.

Place masking before `EmbeddingAttrs` when it incorporates the geometry into
`edge_attrs` or `edge_eq_attrs`. Masking the original embeddings afterwards would
leave an unmasked copy available to the interaction.

## Targets and ordinary heads

For `out_field: angular_reconstruction`, the module creates:

| Field | Contents |
| --- | --- |
| `angular_reconstruction_target` | Detached copy of the complete original feature vector for **every** edge |
| `angular_reconstruction_mask` | Boolean `[number_of_edges]` mask; true means corrupted |
| The original input field | Original values on unselected rows; mask value on selected rows |

`mask_field` on the module can override the generated mask name. Input,
prediction, target and mask names must be distinct. Generated fields are
registered as edge fields automatically, not as NPZ/dataset attributes.

The ordinary inference loop collects the `_target` field into the reference
dictionary under `angular_reconstruction`. A normal head writes the prediction
under that same name in the prediction dictionary:

```yaml
    # Place after the interaction; both heads consume edge_features, not targets.
    - _target_: geqtrain.nn.ReadoutModule
      name: radial_head
      field: edge_features
      out_field: radial_reconstruction
      out_irreps: 8x0e  # Match num_basis=8 in this example.
      strict_irreps: false
      readout_latent_kwargs:
        mlp_latent_dimensions: [64]
    - _target_: geqtrain.nn.ReadoutModule
      name: angular_head
      field: edge_features
      out_field: angular_reconstruction
      out_irreps: 1x0e+1x1o+1x2e  # Full SH vector, l_max=2, parity=o3_full.
      normalize_equivariant_output: false
      readout_latent_kwargs:
        mlp_latent_dimensions: [64]
```

Match the exact input irreps, including multiplicities, ordering and parity.
Angular reconstruction includes the `l=0` component when the embedding contains
it. It is not a scalar regression on the norm or an angle. Targets retain the
producer's basis scaling and SH normalization; there is no extra normalization
or inverse normalization. If calling the model directly rather than through
`run_inference`, collect its `ref_data_keys` into your reference dictionary.

## Losses and metrics

```yaml
loss_coeffs:
  # ... usual supervised losses ...
  - radial_reconstruction:
    - 0.1
    - MSELoss
    - mask_field: radial_reconstruction_mask
  - angular_reconstruction:
    - 0.1
    - MSELoss
    - mask_field: angular_reconstruction_mask

metrics_components:
  - angular_reconstruction:
    - L1Loss
    - mask_field: angular_reconstruction_mask
```

`mask_field` is a general **row-aligned** loss/metric filter. The framework ANDs
it with other requested filters, including `ignore_nan`, before calling the
loss. It applies equally to built-in PyTorch losses and custom GEqTrain losses.
Do not use node-type filters or `PerSpecies` on edge reconstruction outputs.

Unlike the optional `node_mask_field`, an explicitly configured `mask_field`
must exist and have the same row count as the prediction. Missing or incorrectly
shaped masks raise an error rather than silently supervising all edges. Empty
masks yield a differentiable zero loss; metrics do not accumulate any samples.
Loss contributions and their configured coefficients use the usual logging.
For MSELoss the reduction averages squared component errors over selected edges
and all vector components. L1 is available but, unlike vector MSE, componentwise
L1 of an angular irrep is not rotation-invariant.

## Sampling and evaluation

- Each module independently samples a Bernoulli mask per edge on every training
  forward pass. `mask_fraction` is a probability, not an exact count. The two
  directions of a directed edge are sampled independently.
- The complete row is replaced, never individual angular components. Zero is
  required for nontrivial irreps, including pseudoscalars, to preserve
  equivariance. Invariant `0e` features permit a finite nonzero `mask_value`.
  Zero is the default sentinel, not a claim that radial features are centered.
- Targets are detached, while unmasked inputs retain their gradient path.
  Atom positions are not perturbed. This is feature reconstruction, not
  coordinate denoising.
- `enabled: false` or `mask_fraction: 0` leaves inputs unchanged and produces an
  empty supervision mask. This does not remove the configured prediction heads.
- By default, `model.eval()` leaves geometry intact and reconstruction losses
  contribute zero. Masked reconstruction metrics have no samples in validation.
  Normal supervised validation and model selection remain clean. Set
  `mask_in_eval: true` explicitly to evaluate reconstruction on corrupted inputs;
  this also corrupts supervised validation and deployment inputs and is
  stochastic. Use a separate diagnostic evaluation if clean validation matters.
- Masking occurs after batching, chunk selection and edge dropout, so generated
  targets/masks follow the actual edges forwarded. Chunked forwards sample new
  masks independently and need not match an unchunked pass exactly.
- No geometry corruption is stored in processed dataset caches. Adding runtime
  edge reconstruction targets does not provide new node labels or retain nodes
  excluded by existing dataset filters.

Equivariance holds for a fixed mask (transform and permute the mask with its
edges); two independently sampled masks need not give identical predictions.
Use clean eval mode for the usual deterministic equivariance test.

## Architectural limitation

Masking is model-independent; successful reconstruction is not. An equivariant
head cannot recover a direction from invariant scalars alone. If masking removes
all directional information available to an edge, a purely edge-local model may
be forced to output zero for its non-scalar reconstruction. The backbone needs
an alternative **contextual** equivariant path, for example neighboring
unmasked directions propagated into that edge's representation. Simply exposing
the original target direction to the head would instead create a shortcut.
Treat this as an optional regularizer and compare with an unmasked baseline;
the implementation does not guarantee improved force-field accuracy.
