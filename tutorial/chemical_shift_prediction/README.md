# Chemical-Shift Prediction Tutorial

This tutorial prepares a ShiftML3-derived molecular-solid NMR dataset for
GEqTrain and runs the scalar-plus-tensor shielding experiment.

The original dataset and benchmark come from ShiftML3. See the ShiftML3
repository and paper for provenance and licensing:

```text
https://github.com/lab-cosmo/shiftml
```

## Files

- `train.xyz`, `valid.xyz`, `test.xyz`: tutorial split in extended XYZ format.
- `build_dataset.py`: converts extended XYZ files to GEqTrain masked NPZ files.
- `cartesian_to_spherical.py`: converts Cartesian shielding tensors to e3nn
  irreducible tensor components.

The XYZ files are large; use Git LFS when storing them on GitHub.

## Build NPZ Datasets

```bash
python build_dataset.py --inputs train.xyz --outputs train.npz --workers 4
python build_dataset.py --inputs valid.xyz --outputs valid.npz --workers 4
python build_dataset.py --inputs test.xyz --outputs test.npz --workers 4
```

The generated NPZ files are ignored by Git.

## Run

From the repository root:

```bash
geqtrain-train config/experiment/shiftml3.yaml -d cuda:0
```

Relevant configs:

- `config/experiment/shiftml3.yaml`
- `config/data/shiftml3.yaml`
- `config/model/shiftml3_interaction.yaml`
- `config/model/stack_blocks/common.yaml`
- `config/train/shiftml3.yaml`

The model predicts `cs_iso` as `1x0e` and `cs_tensor` as
`1x1o + 1x2e`.
