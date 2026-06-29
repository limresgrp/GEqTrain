# Precision Configuration

GEqTrain has two separate precision controls.

## `default_dtype`

`default_dtype` controls the floating dtype used when building datasets and initializing model parameters/buffers.

```yaml
default_dtype: float64
mixed_precision: false
allow_tf32: false
```

Use this for full double-precision training. Changing `default_dtype` changes the processed-dataset cache key, so float32 and float64 runs use separate cached datasets.

Supported values are:

- `float32`
- `float64`

Integer/index fields such as `edge_index`, `node_types`, and batch indices remain integer tensors.

## `mixed_precision`

`mixed_precision` controls autocast during the model forward pass. It does not change stored dataset dtype or model parameter dtype.

```yaml
mixed_precision: true
mixed_precision_dtype: bfloat16  # bfloat16 | float16
```

The default mixed precision dtype is `bfloat16`, matching the previous behavior. Disable mixed precision when using `default_dtype: float64`; autocast is explicitly meant to lower selected operations to reduced precision.

## Safe Float32 Downcasts

Some code paths still cast to float32 intentionally for scalar logging, diagnostics, curriculum sampler priorities, and external metrics such as AUROC. These values do not feed back into model forward computation or stored training targets.
