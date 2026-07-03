# GEqTrain

GEqTrain is a configuration-driven framework for building, training, evaluating,
and deploying E(3)-equivariant graph neural networks with PyTorch and e3nn.

The core design is to keep dataset semantics, geometric features, model stacks,
losses, metrics, normalization, and deployment behavior in configuration files.
This makes it possible to retarget the same equivariant infrastructure across
node-, edge-, and graph-level scientific learning tasks without rewriting model
code.

## Quick Setup

The recommended setup path is the repository script:

```bash
git clone https://github.com/limresgrp/GEqTrain.git
cd GEqTrain
./venv_setup.sh --torch-backend auto
source .venv-geqtrain/bin/activate
```

For a specific Torch backend, pass one of uv's supported backends:

```bash
./venv_setup.sh --torch-backend cu121
./venv_setup.sh --torch-backend cpu
```

Useful setup options:

```bash
./venv_setup.sh --help
./venv_setup.sh --recreate
./venv_setup.sh --python 3.11 --torch-backend cu124
./venv_setup.sh --no-torch
./venv_setup.sh --dev
```

The script creates a local virtual environment, installs PyTorch, installs
GEqTrain in editable mode, and adds runtime/tutorial dependencies. If `uv` is
not installed, the script installs it first.

Manual fallback:

```bash
python -m venv .venv-geqtrain
source .venv-geqtrain/bin/activate
pip install torch
pip install -e .
pip install scipy matplotlib pandas plotly
```

## Basic Usage

Single-GPU training:

```bash
geqtrain-train config/experiment/shiftml3.yaml -d cuda:0
```

CPU or explicit device:

```bash
geqtrain-train config/experiment/shiftml3.yaml -d cpu
```

Distributed training on visible GPUs:

```bash
torchrun --nproc_per_node=2 geqtrain/scripts/train.py \
  config/experiment/shiftml3.yaml \
  --ddp
```

Other entry points:

```bash
geqtrain-evaluate --help
geqtrain-test-equivariance --help
geqtrain-deploy --help
```

## Configuration Layout

GEqTrain experiments are composed with Hydra:

```yaml
defaults:
  - /base
  - /data: shiftml3
  - /model: shiftml3_interaction
  - /train: shiftml3
  - _self_
```

The main groups are:

- `config/data`: raw data sources, key mappings, typed fields, normalization, and filtering.
- `config/model`: model stacks and readout heads.
- `config/train`: losses, metrics, optimization, batching, and logging.
- `config/experiment`: complete experiment compositions.

## ShiftML3 NMR Tutorial

The repository includes a self-contained chemical-shift tutorial under
`tutorial/chemical_shift_prediction`.

It demonstrates scalar and tensorial NMR shielding prediction on data derived
from the ShiftML3 molecular-solid dataset. The original dataset and benchmark
are from the ShiftML3 work; see the ShiftML3 repository and paper for dataset
provenance, licensing, and benchmark definitions:

```text
https://github.com/lab-cosmo/shiftml
```

Tutorial files:

- `tutorial/chemical_shift_prediction/train.xyz`
- `tutorial/chemical_shift_prediction/valid.xyz`
- `tutorial/chemical_shift_prediction/test.xyz`
- `tutorial/chemical_shift_prediction/build_dataset.py`
- `tutorial/chemical_shift_prediction/cartesian_to_spherical.py`

The XYZ files contain the tutorial split. They are large enough that Git LFS is
recommended if they are stored on GitHub.

Build packed NPZ files:

```bash
python tutorial/chemical_shift_prediction/build_dataset.py \
  --inputs tutorial/chemical_shift_prediction/train.xyz \
  --outputs tutorial/chemical_shift_prediction/train.npz

python tutorial/chemical_shift_prediction/build_dataset.py \
  --inputs tutorial/chemical_shift_prediction/valid.xyz \
  --outputs tutorial/chemical_shift_prediction/valid.npz

python tutorial/chemical_shift_prediction/build_dataset.py \
  --inputs tutorial/chemical_shift_prediction/test.xyz \
  --outputs tutorial/chemical_shift_prediction/test.npz
```

For faster conversion, add `--workers N`.

The dataset builder reads extended XYZ files, extracts atomic numbers and
periodic-table row/column attributes, stores isotropic shielding targets, and
converts Cartesian shielding tensors to irreducible spherical components using
e3nn's canonical rank-2 tensor basis.

Run the scalar-plus-tensor tutorial experiment:

```bash
geqtrain-train config/experiment/shiftml3.yaml -d cuda:0
```

Relevant configs:

- `config/experiment/shiftml3.yaml`
- `config/data/shiftml3.yaml`
- `config/model/shiftml3_interaction.yaml`
- `config/train/shiftml3.yaml`

The tutorial config predicts:

- `cs_iso` as `1x0e`
- `cs_tensor` as `1x1o + 1x2e`

Metrics are reported in physical units by applying the inverse of the
training-fitted normalization parameters. Tensor metrics are also broken down by
irreducible order when the target irreps are known.

## HPC Notes

On clusters, the same setup script can be used inside an interactive allocation
or containerized software environment:

```bash
./venv_setup.sh --no-torch
source .venv-geqtrain/bin/activate
```

Use `--no-torch` when the cluster environment already provides a compatible
PyTorch/CUDA stack.

Minimal SLURM launch pattern:

```bash
#!/bin/bash
#SBATCH --job-name=geqtrain
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00

cd /path/to/GEqTrain
source .venv-geqtrain/bin/activate

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun geqtrain-train config/experiment/shiftml3.yaml \
  --ddp \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT"
```
