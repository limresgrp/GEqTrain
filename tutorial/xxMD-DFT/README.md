# xxMD-DFT: from supervised prediction to geometry reconstruction

This folder contains two incremental GEqTrain tutorials using the supplied
azobenzene NPZ files. Read [lesson 1](01_force_prediction.md), then
[lesson 2](02_masked_geometry.md). These are small educational configurations,
not reproductions of a published benchmark or production molecular-dynamics
potentials.

## Setup and first check

From the repository root, activate the environment installed using
[`venv_setup.sh`](../../venv_setup.sh):

```bash
source .venv-geqtrain/bin/activate
python tutorial/xxMD-DFT/smoke_test.py
```

The check reads two training frames and runs one optimization step and a clean
evaluation for each configuration. It does not process the full datasets or
save checkpoints. Optional arguments are `--device cuda:0` and
`--config tutorial/xxMD-DFT/config/experiment/01_baseline.yaml`.
The printed errors are from an essentially untrained model, not benchmark
results. The small CPU check uses one Torch thread to avoid oversubscription.

## Files and experiments

| Experiment | Purpose |
| --- | --- |
| `config/experiment/01_baseline.yaml` | Small invariant-energy/equivariant-force regression model |
| `config/experiment/01_attention.yaml` | Same task with contextual scalar-query attention |
| `config/experiment/02_context_control.yaml` | Extra equivariant context stage, no corruption or auxiliary contribution |
| `config/experiment/02_masked_geometry.yaml` | Same context model with masking and reconstruction losses |

The local `config/` is its own Hydra configuration root. `data/` describes raw
data, `model/` describes the forward computation, `train/` describes optimization
and supervision, and `experiment/` composes them. `model/blocks.yaml` contains
shared layer definitions. This directory does not import personal configs or
require the repository's global `config/model/stack_blocks`.

Paths to data and results are relative to the **working directory**, not the
YAML file. All documented commands assume the GEqTrain repository root.
Training data location can be overridden with `-o data_root=/absolute/path`;
the directory must contain the three filenames below. Evaluation's plain YAML
uses an explicit path: edit `config/data/test.yaml` if relocating the data.

## Supplied dataset

| File | Frames | Atoms per frame |
| --- | ---: | ---: |
| `data/azo_train.npz` | 4,195 | 24 |
| `data/azo_val.npz` | 2,110 | 24 |
| `data/azo_test.npz` | 2,065 | 24 |

Each contains `coords [frames,24,3]`, `forces [frames,24,3]`, `energy [frames]`,
and `atom_types [24]`. Atom types are atomic numbers 1, 6 and 7 (H, C and N), not
three contiguous category indices. Coordinates, energies and forces were all
finite when checked. Only `node_types` is fixed across frames; positions and
targets vary. All atoms can be edge centers and neighbors within the cutoff.

The NPZ files have no unit or provenance metadata. The tutorials preserve their
numeric units and existing energy reference without claiming eV, Hartree, or a
specific unit conversion. Confirm the upstream conversion/provenance before
interpreting errors in physical units or comparing to published xxMD numbers.
Keep the original xxMD dataset attribution, citation and redistribution license
with any public release of these supplied files; this tutorial does not establish
their redistribution rights.

Validation and test are separate supplied splits. Do not use test frames for
early stopping or hyperparameter selection. Molecular-dynamics frames can be
correlated; the mere presence of separate files does not establish an independent
or leakage-free benchmark split.

## Scope

The introductory models have **independent energy and force heads**. Forces
transform as polar vectors, but are not constrained to equal `-grad(E)`.
They are supervised regression baselines, not guaranteed energy-conserving MD
potentials. This intentionally simplifies the larger gradient-based local xxMD
experiment. See lesson 1 before extending them to energy-gradient forces.
