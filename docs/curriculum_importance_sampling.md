# Curriculum Importance Sampling

GEqTrain can train with batch-level curriculum importance sampling. Anchor epochs
iterate once over every training example in ordinary fixed batches. Priority
epochs sample those same batches with replacement according to a smoothed
loss-derived distribution.

```yaml
loss_coeffs:
  - cs_iso:
    - 1.0
    - geqtrain.train.LogCoshLoss
  - cs_tensor:
    - 100.0
    - geqtrain.train.LogCoshLoss
    - name: tensor_priority

curriculum_importance_sampling:
  enabled: true
  loss: tensor_priority
  anchor_interval: 5
  alpha: 0.5
  beta_warmup_epochs: 10
  gamma: 0.2
  gamma_final: 1.0
  gamma_warmup_epochs: 100
  error_ema: 0.8
  histogram_bins: 10
```

`loss` selects the configured loss component used to score each batch. If a loss
has `name`, use that name without the automatic numeric suffix. If omitted or set
to `loss` / `total`, the total weighted loss is used.

Sampling rule:

- Epoch `0` and every `anchor_interval` epochs are anchor epochs.
- Anchor epochs use each batch once and refresh batch errors.
- Priority epochs draw `len(anchor_batches)` batches with replacement.
- Batch priority is proportional to `error ** alpha`.
- The final probability is mixed with uniform sampling by `beta`, which ramps
from `0` to `1` over `beta_warmup_epochs`.
- During priority epochs the training loss is multiplied by
  `(1 / (num_batches * probability)) ** gamma`.
- `gamma` can ramp to `gamma_final` over `gamma_warmup_epochs`.

Validation is always evaluated with its normal uniform dataloader, independent
of training sampling. Best-model checkpointing therefore still runs on every
validation epoch. Trainer checkpoints include the sampler state, so restarts
preserve the learned batch-error distribution.

Current limitations:

- Curriculum sampling is batch-level, not atom-level or graph-level within a
  batch.
- It is currently supported only for non-DDP training.
- It is not compatible with `dataset_mode: ensemble`.
