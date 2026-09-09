# Lesson 2: masked geometry reconstruction

Prerequisite: [lesson 1](01_force_prediction.md). The supervised task, data splits,
cutoff, batch size and physical output heads remain unchanged.

## 1. Why add another task?

We ask the representation to reconstruct a fraction of the geometry embeddings
that were hidden from it. This uses information already available in the graph,
without adding labels to the NPZ or changing coordinates. It is an optional
regularizer, not a guarantee of improved force accuracy.

The training corruption affects the supervised energy/force path too. Physical
targets still refer to the original geometry. That makes regression deliberately
harder and potentially ambiguous; keep masking and auxiliary coefficients small
and judge their utility on clean validation.

## 2. First add contextual equivariant information

A masked edge cannot recover its direction from scalar features alone. The
context model therefore pools first-stage equivariant edge features to nodes,
then passes those node features to a second interaction stage through
`node_equivariant_field: node_features`. This provides directions from other
unmasked edges at the endpoints, not a direct copy of the target edge direction.

Read [`model/context.yaml`](config/model/context.yaml):

```text
geometry -> optional masks -> attributes -> first interaction
                                        -> pooled node context
                                        -> context interaction
                                        -> reconstruction edge heads
                                        -> final node pooling -> energy/force heads
```

This is a larger model than lesson 1. To avoid attributing the benefit of extra
capacity to the regularizer, train its unmasked control first:

```bash
geqtrain-train tutorial/xxMD-DFT/config/experiment/02_context_control.yaml -d cuda:0
```

The control retains exactly the same modules and reconstruction heads but sets
`mask_geometry: false`. Auxiliary masks are empty and their losses contribute
zero. Thus the additional heads are present but not supervised in this control.

## 3. Turn on runtime masks

The masked experiment changes `mask_geometry` to true:

```bash
geqtrain-train tutorial/xxMD-DFT/config/experiment/02_masked_geometry.yaml -d cuda:0
```

The same generic module is instantiated once for radial embeddings and once for
spherical harmonics. Both are placed before `EmbeddingAttrs`, so it cannot retain
an unmasked copy in `edge_attrs` or `edge_eq_attrs`.

```yaml
_target_: geqtrain.nn.MaskEdgeFeatures
field: spharms_emb
out_field: angular_reconstruction
enabled: ${mask_geometry}
mask_fraction: ${mask_fraction}
mask_in_eval: false
```

Each training forward independently samples edges with probability 0.15 for each
feature family. The whole selected feature vector is replaced by zero. The
opposite directions of a directed edge are not tied. Masks are resampled during
training, not stored once in the processed dataset cache.

The unmodified complete vector for every edge is saved in
`angular_reconstruction_target`, detached from autograd. The Boolean
`angular_reconstruction_mask` marks the corrupted edges. The usual inference
loop places the target into its reference dictionary as `angular_reconstruction`.
Do not feed the `_target` field into a prediction head.

## 4. Ordinary heads and ordinary loss entries

There is no `auxiliary_head` API. Both heads are normal `ReadoutModule` entries
reading `edge_features`. Their output irreps exactly match their targets:

| Prediction | Target shape per edge | Irreps |
| --- | --- | --- |
| `radial_reconstruction` | `num_basis` basis values | `${num_basis}x0e` |
| `angular_reconstruction` | 9 SH components, including l=0 | `1x0e+1x1o+1x2e` |

The radial basis has `trainable: false` in all tutorial variants, so these
reconstruction targets have a stable definition. The radial head's irreps follow
`num_basis` automatically through YAML interpolation. If changing `l_max` or parity, update the angular head
to the exact SH representation, not just a matching flattened dimension.

[`train/reconstruction.yaml`](config/train/reconstruction.yaml) extends the
supervised configuration and repeats the loss list, because Hydra replaces lists
rather than appending their entries:

```yaml
loss_coeffs:
  - energy: [1.0, MSELoss]
  - forces: [1.0, MSELoss]
  - radial_reconstruction:
    - 0.05
    - MSELoss
    - mask_field: radial_reconstruction_mask
  - angular_reconstruction:
    - 0.05
    - MSELoss
    - mask_field: angular_reconstruction_mask
```

The framework filters rows **before** calling each loss. Only masked edges
contribute to reconstruction MSE, averaged over selected edges and all components.
Full angular-vector MSE is rotation-invariant in the orthonormal irrep basis;
componentwise L1 would not share that property. Zero masking preserves the
equivariant structure, whereas filling non-scalar components with a nonzero
constant would not.

An empty mask gives zero auxiliary loss, not an error. Do not add `ignore_nan`
as a substitute for the mask or alter the dataset targets to implement this task.
The same `mask_field` option works for normal metric entries if you run a separate
masked diagnostic pass.

## 5. Read the results correctly

Training logs include four loss contributions. The auxiliary terms are in basis/
SH units, not force or energy units. Their coefficients balance objectives; raw
loss magnitudes across these tasks do not share a physical interpretation.

Validation runs in `model.eval()` with intact inputs. Reconstruction masks are
empty, auxiliary losses contribute zero, and `metrics_epoch.csv` tracks the
ordinary energy/force validation errors. Best-model selection therefore uses the
same clean physical objective as the control. There is no informative masked
reconstruction validation metric in this default setup.

Compare `02_context_control` against `02_masked_geometry`, ideally over several
seeds. Do not use a comparison against the smaller lesson 1 model alone to claim
that auxiliary supervision helped. To vary the corruption level in a new run:

```bash
geqtrain-train tutorial/xxMD-DFT/config/experiment/02_masked_geometry.yaml -d cuda:0 \
  -o mask_fraction=0.05 -o run_name=02_masked_geometry_low_mask
```

For a stricter ablation, also try corruption with reconstruction coefficients
set to zero: that separates corruption as augmentation from reconstruction as
supervision. Give that configuration its own run name. Avoid tuning on test.

`mask_in_eval: true` would corrupt supervised validation and deployed inference
as well, so do not enable it for the clean comparison or molecular dynamics.
Always evaluate models in eval mode. A deterministic equivariance check should
use clean inputs, or explicitly transform the same mask along with the graph.

The test command from lesson 1 also works with this model's checkpoint; the plain
test YAML intentionally replaces the loss/metric lists with physical targets only.

## Further reading

See the framework's [masked geometry documentation](../../docs/masked_geometry.md)
for field contracts, mask/NaN interactions, chunking and architectural limitations.
These examples do not guarantee recovery when all available directional context
is masked, nor do auxiliary objectives make the direct force head conservative.
