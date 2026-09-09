# Learning GEqTrain incrementally

Start with a small task, understand its data/model/loss contract, and change one
mechanism at a time. Run commands from the GEqTrain repository root after
following the [environment setup](../README.md#quick-setup).

| Lesson | What you learn |
| --- | --- |
| [1. Energy and force prediction](xxMD-DFT/01_force_prediction.md) | NPZ fields, Hydra composition, embeddings, interactions, scalar/vector heads, losses, evaluation, then attention |
| [2. Masked geometry reconstruction](xxMD-DFT/02_masked_geometry.md) | Runtime masking, detached targets, ordinary auxiliary readouts/losses, equivariant context, controlled comparisons |
| [Chemical-shift prediction](chemical_shift_prediction/README.md) | A separate application with scalar/tensor targets and dataset conversion |

The first two lessons share the bundled [xxMD-DFT example](xxMD-DFT/README.md).
They do not require files from `local_config` or a personal scratch directory.
