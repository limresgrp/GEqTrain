# Lesson 1: energy and force prediction

## 1. Read the experiment, then the three configuration groups

Open [`config/experiment/01_baseline.yaml`](config/experiment/01_baseline.yaml).
Its defaults load `base`, `data/azo`, `model/baseline` and `train/supervised`.
`_self_` lets values in the current file override inherited values.

[`data/azo.yaml`](config/data/azo.yaml) maps `coords` to `pos` and `atom_types`
to `node_types`. `forces` is a node field, whereas `energy` is a graph field.
The fixed `[24]` atom-type array is reused for each conformer. No padding masks,
type filters, ensemble averaging or automatic 80/20 split are used here.
Validation is explicitly supplied in its own file.

`num_types: 8` accommodates the largest raw category index, 7. The complete
`type_names` list has placeholders for unused atomic numbers. Do not replace it
with `[H, C, N]` unless you also remap the NPZ indices.

`normalization: {}` deliberately keeps targets in native units: the supplied
training energies already span approximately 0 to 8.31, rather than a huge
absolute electronic-energy offset. This also makes the first loss/metric
comparison straightforward. The larger local experiment's independent energy
and force normalizations and 1000:1 loss weights are not copied blindly.

## 2. Follow the forward computation

Read [`model/baseline.yaml`](config/model/baseline.yaml), expanding its references
using [`model/blocks.yaml`](config/model/blocks.yaml):

```text
atomic numbers -> one-hot input -> node attributes
positions/neighbor edges -> radial basis + spherical harmonics
attributes + geometry -> InteractionModule -> edge_features
edge_features -> EdgewiseReduce -> node_features
node_features -> scalar atomic energies -> sum -> molecular energy
node_features -> equivariant vector head -> forces
```

The initial model uses a radius cutoff of 5 in the coordinate units, eight fixed
radial basis functions, two interaction layers, 32 scalar latent channels and
four copies of each higher-order irrep. `l_max: 2` and `parity: o3_full` produce
`1x0e+1x1o+1x2e` angular features.

`energy` has irreps `1x0e`: it is invariant under rotation and inversion.
`forces` has irreps `1x1o`: it is a polar vector. An axial `1x1e` head would be
the wrong transformation law for forces. Intermediate `l=2` features can enrich
the representation even though neither output is a rank-two tensor.

The energy head uses `strict_irreps: false` because it reads only the scalar
portion of mixed node features. The force head keeps output normalization off:
forces need a learned magnitude, not a unit-vector constraint.

## 3. Understand loss and metrics

[`train/supervised.yaml`](config/train/supervised.yaml) declares:

```yaml
loss_coeffs:
  - energy: [1.0, MSELoss]
  - forces: [1.0, MSELoss]
metrics_components:
  - energy: [L1Loss]
  - forces: [L1Loss]
```

Energy MSE is averaged over graphs; force MSE is averaged over atoms and Cartesian
components. The total is the weighted sum of those two terms. Force MAE is a
componentwise absolute error, not the mean Euclidean vector error. Losses are in
squared native units; MAEs are in native units. Consequently their magnitudes
are not directly comparable. Equal coefficients are an initial teaching choice,
not an assertion that the two physical objectives are optimally balanced.

All atoms have finite force targets here, so no additional NaN or species filter
is required. `metrics_key: validation_loss` selects the best checkpoint using
the combined supervised validation objective, not either individual MAE.

## 4. Train the baseline

From the repository root:

```bash
geqtrain-train tutorial/xxMD-DFT/config/experiment/01_baseline.yaml -d cuda:0
```

For a first short run with a separate output directory:

```bash
geqtrain-train tutorial/xxMD-DFT/config/experiment/01_baseline.yaml -d cpu \
  -o max_epochs=2 -o run_name=01_baseline_short
```

Full dataset preprocessing can take longer than model startup and is cached by
GEqTrain. The one-step smoke check in the main README avoids that initial cost.
Use `-o batch_size=2 -o validation_batch_size=2` if memory is tight. Keep workers
at their supplied single-process settings initially. Do not enable graph chunking
for this example: the molecular energy target requires the complete graph.

Results go under `results/tutorial_xxmd/<run_name>/`. Inspect `log`,
`metrics_epoch.csv` and `best_model.pth`. Batch CSVs are off by default; enable
them with `-o log_batch_csv=true` for debugging. `metrics_epoch.csv` contains
validation summaries, not training-batch averages.

Existing run directories can trigger restart behavior. Give every ablation a
new `run_name`; changing the model inside an existing run is not a new experiment.
Record the seed, cutoff, architecture, coefficients and selected epoch when
comparing runs. A short run is a functionality check, not an accuracy result.

## 5. Add attention without changing the task

Inspect [`model/attention.yaml`](config/model/attention.yaml). It inherits the
baseline and overrides only the shared interaction block:

```yaml
blocks:
  interaction:
    use_attention: true
    attention_mode: node_feature_query
    attention_head_dim: 8
    node_state_use_residual: true
    node_state_pooling: mean
```

Run:

```bash
geqtrain-train tutorial/xxMD-DFT/config/experiment/01_attention.yaml -d cuda:0
```

The interaction pools updated edges into node states between layers; those states
supply contextual queries. Attention weights depend on scalar components, not
individual Cartesian coordinates. Residual node-state updates preserve part of
the preceding state. This avoids all-edge-pairs attention within each neighborhood.

The explicit final `EdgewiseReduce` is retained in both variants intentionally:
both heads read a fresh final edge reduction rather than switching the attention
variant to its accumulated residual node state. That reduction overwrites
`node_features` at the end; intermediate states still serve the attention queries.
Compare validation MAEs, runtime and memory to the baseline. Attention adds
parameters and is not guaranteed to improve either error.

## 6. Evaluate the held-out test split

Only after choosing settings on validation:

```bash
geqtrain-evaluate results/tutorial_xxmd/01_baseline/best_model.pth \
  tutorial/xxMD-DFT/config/data/test.yaml \
  -d cuda:0 -b 4 -l results/tutorial_xxmd/test_baseline
```

Replace the checkpoint and log directory for other runs. This test YAML is plain,
self-contained YAML because the evaluator merges it into the saved training
configuration rather than composing a fresh Hydra experiment. It requests only
energy and force metrics, including when evaluating lesson 2 models.

## 7. What would make this a conservative force field?

A direct vector head is equivariant, but equivariance does not imply
`F = -grad(E)`, zero net force, or energy conservation along an MD trajectory.
Do not use these beginner regressors as validated MD potentials.

An advanced extension would replace the force head with `EnableGradients` before
the geometry calculation and `ComputeGradient` after the molecular energy sum,
with `model_requires_grads: true` so evaluation also permits derivatives.
That requires second derivatives during force-loss training, consistent energy/
length units and consistent normalization. Also inspect the current
`ComputeGradient`: its scale factors are trainable parameters, not fixed unit
conversions. A strictly energy-gradient force implementation must constrain the
scale appropriately. This lesson deliberately avoids silently presenting that
path as a guaranteed conservative model.

Continue to [lesson 2](02_masked_geometry.md).
