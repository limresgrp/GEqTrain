# Inference Pipeline (Training Checkpoint and Deployed Model)

GEqTrain now provides a unified inference path that works for both:

- training-session checkpoints (`best_model.pth`, `last_model.pth` with `config.yaml` / `trainer.pth`)
- deployed TorchScript models (`geqtrain-deploy` output)

## 1) Unified API

Use `InferenceSession`:

```python
from geqtrain.inference import InferenceSession

session = InferenceSession.from_model_path("/path/to/model.pth", device="cuda:0")
out, ref_data, center_nodes, n_center = session.predict(batch)
```

This loader is model-type agnostic:

- If the path is a deployed model, metadata is read from TorchScript extra files.
- If deployed loading fails, it falls back to training-session loading.

## 2) Metadata schema for inference

A common metadata bundle is used under key:

- `inference_metadata_v1`

Structure:

- `version`
- `normalization`
- `denormalize_inference_outputs`
- `normalization_stats_by_ensemble`
- `default_ensemble`

This bundle is:

- saved in deployed models (`geqtrain.utils.deploy.build_deployment`)
- saved in `trainer.pth` state during training checkpoints
- synthesized from config if no explicit bundle is available

## 3) Automatic denormalization behavior

`run_inference(...)` supports automatic denormalization of predictions when:

- `loss_fn is None`
- `is_train is False`
- `denormalize_inference_outputs` is true (default)

Denormalization now uses two sources transparently:

1. Batch-attached stats (existing behavior).
2. Inference metadata stats fallback (new behavior), injected into `ref_data` when missing.

This is what enables deployed-model inference to denormalize outputs even when the incoming dataset does not carry `_mean_/_std_/_transform_` fields.

## 4) Deployment behavior

During deployment, GEqTrain tries to collect train-fitted normalization statistics per ensemble and saves them in deployment metadata.

- If collection succeeds: deployed model carries full denormalization stats.
- If collection fails: deployment still succeeds, and a warning is emitted; denormalization may then require batch-level stats.

## 5) Evaluation script

`geqtrain/scripts/evaluate.py` now loads models through `CheckpointHandler.load_model(...)`, so evaluation follows the same transparent path for deployed and non-deployed models, including inference metadata handling.
