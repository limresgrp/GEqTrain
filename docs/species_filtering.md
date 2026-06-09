# Species Filtering

GEqTrain has two different ways to focus training on a subset of atom types.

## 1) `keep_type_names`

This is a dataset filter, defined in the data config.

Example:

```yaml
keep_type_names: [H]
```

What it does:

- removes all non-H atoms before graph construction and training
- changes the graph itself
- changes neighbors, edges, and any downstream statistics
- reduces memory and compute

Use this when you want to train on a reduced problem and do not need non-H atoms as context.

## 2) Loss-side species masking

This is a supervision filter, defined on the loss entry.

Example:

```yaml
loss_coeffs:
  - cs_iso:
    - 1.0
    - geqtrain.train.LogCoshLoss
    - node_type_names: [H]
      type_names: [X, H, C]
```

What it does:

- keeps the full graph as input
- computes the loss only on the selected atom types
- preserves non-H atoms as context for message passing
- does not change the dataset or edge construction
- in the current implementation, an explicit species mask takes precedence over the legacy edge-center node filter for that loss term

Use this when the model should see the full molecular environment, but supervision should be restricted to a subset of species.

The same flags are also accepted in `metrics_components`, so you can report metrics on the same atom subset without changing the graph.

## Practical difference

- `keep_type_names` answers: "which atoms exist in the graph?"
- loss-side masking answers: "which atoms contribute to the loss?"

## Notes

- The loss-side filter currently works for node-level targets.
- The metric-side filter uses the same node-level convention.
- Use `node_type_indices` if you already know the integer atom type ids.
- Use `node_type_names` together with `type_names` if you want to write species labels instead of indices.
