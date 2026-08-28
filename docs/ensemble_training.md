# Ensemble training

`dataset_mode: ensemble` is intended for datasets where several graph frames
represent the same physical system and share atom ordering. A common example is
an NMR structure ensemble with coordinates shaped `[M, N, 3]` and one fixed
target per atom shaped `[N, ...]`.

## Batching and aggregation

In ensemble mode, `batch_size` and `validation_batch_size` count systems, not
individual conformers. GEqTrain batches all selected conformers of a system
together and records each node's stable atom index within that system.

For node targets, filtering is applied first. Predictions that survive the
filter are then grouped by `(ensemble_index, ensemble_atom_index)` and reduced
according to `ensemble_aggregation`. Losses and metrics therefore operate on
one prediction per physical atom, not one prediction per conformer.

```yaml
dataset_mode: ensemble
ensemble_loss_on_aggregate: true
ensemble_aggregation: mean

# Systems per optimizer step.
batch_size: 1
validation_batch_size: 1
```

All conformers in one system must have the same atom count and atom ordering.

## Memory controls

Two independent limits are available:

```yaml
# Conformers selected per system.
ensemble_max_structures: 4

# Prediction center atoms selected per system.
ensemble_max_atoms: 256
```

`ensemble_max_structures` samples conformers during training. Validation uses a
deterministic prefix. `validation_ensemble_max_structures` can override the
training value; if omitted, it inherits `ensemble_max_structures`.

`ensemble_max_atoms` selects local atom IDs from the centers shared by every
selected conformer. Exactly the same IDs are used in each conformer. The other
atoms are retained as neighbors, so selected centers keep their complete local
environment. This reduces edge and activation memory without turning the
selected atoms' neighborhoods into induced subgraphs. Validation selection is
deterministic and can be overridden with `validation_ensemble_max_atoms`.

Set any limit to `null` to disable it. For full validation while limiting
training, set the corresponding `validation_...` option explicitly to `null`.

Memory is dominated approximately by:

```text
selected conformers * selected centers * neighbors per center * latent width
```

All node tensors are still retained because non-center atoms remain available
as neighbors. The edge/interaction tensors are the main intended saving.

## Target filtering

Dataset preprocessing retains an edge center when at least one configured node
target is available there. This is a logical OR across targets. Each loss and
metric then applies its own filters with logical AND:

- center-node membership;
- `node_type_names` or `node_type_indices`;
- `node_mask_field`, when present;
- finite prediction and reference rows when `ignore_nan: true`.

A configured `node_mask_field` that is absent from a dataset remains optional
and imposes no additional restriction. If a mask is scientifically required,
it must be present in the data rather than relying on this fallback.

Ensemble-aware DDP batching is not currently implemented. GEqTrain raises an
explicit error for `dataset_mode: ensemble` with multi-process DDP instead of
splitting a system across ranks.
