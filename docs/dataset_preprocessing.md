# Dataset Preprocessing Pipeline

This document describes how GEqTrain builds datasets, applies filtering, and normalizes targets across train/validation/test.

## 1) Dataset construction flow

`dataset_from_config(...)` (`geqtrain/data/_build.py`) is the entrypoint.

- It reads either:
  - `<prefix>_dataset_list` (`train_dataset_list`, `validation_dataset_list`, `test_dataset_list`), or
  - prefixed single-dataset keys (`train_*`, `validation_*`, `test_*`).
- For each dataset item, it instantiates the configured dataset class (for example `npz` -> `NpzDataset`).
- It expands `dataset_input` from:
  - a single file,
  - a directory (all files inside),
  - or a `.txt` file listing paths.
- It returns:
  - `InMemoryConcatDataset` when `inmemory: true`,
  - `LazyLoadingConcatDataset` when `inmemory: false`.

## 2) Per-file preprocessing inside `AtomicInMemoryDataset.process()`

For each source dataset:

- Load raw arrays via `get_data()`.
- Apply `key_mapping` (for `NpzDataset`) and split data into:
  - node fields,
  - edge fields,
  - graph fields,
  - extra fields,
  - fixed fields.
- Merge `extra_fixed_fields` into fixed fields.
- Parse configured attributes (`node_attributes`, `edge_attributes`, `graph_attributes`, `extra_attributes`):
  - `attribute_type: numerical` keeps float tensors.
  - `attribute_type: categorical` maps to integer tokens.
  - Binned numerical attributes are also converted to integer tokens.
  - `embedding_mode` can be `embedding`, `one_hot`, or `positional`.
- Build `AtomicData` objects:
  - if `edge_index` exists: use provided graph,
  - otherwise: build neighbors from `pos` + `r_max`.
- Cache processed output under `processed_datasets/processed_dataset_<hash>/`.

## 3) Filtering options

Filtering is applied in `_filter_dataset(...)` (`geqtrain/data/_build.py`) after dataset instantiation:

- Node filtering:
  - `keep_node_types` (index-based), or
  - `keep_type_names` (name-based, mapped to indices).
- Edge filtering:
  - `exclude_type_names_from_edge_center` / `exclude_type_names_from_edge_neigh`,
  - and corresponding index-based variants.
- NaN-aware filtering:
  - edges connected to nodes with NaNs in loss-relevant node targets are removed.
  - this filtering runs even when no explicit `keep_type_names` or edge-type exclusion is configured.
  - affected atoms may remain as neighbors; they are removed only as edge centers unless they become isolated.
- After edge filtering, isolated nodes are pruned.

Dataset processing parallelism:

- With multiple input files, `dataset_num_workers` parallelizes across files.
- With one NPZ file, `dataset_num_workers` parallelizes frame construction inside that file.
- Single-NPZ workers use spawned processes and temporary chunk files, avoiding forked Torch state and large `AtomicData` payloads through multiprocessing pipes.

## 4) Normalization and transforms

Normalization config is under `normalization:`.  
Parsing and runtime utilities are in `geqtrain/utils/normalization.py`.

Supported options:

- `mode`:
  - `per_type[:irreps]`
  - `global[:irreps]`
- `transform`:
  - `none`
  - `signed_log1p`
  - `yeo_johnson` (lambda auto-fit by default)
- `apply_on_dataset`:
  - `true` (default): preprocess dataset values,
  - `false`: keep raw dataset values, but still use normalization metadata for output denormalization.

Important equivariant behavior:

- For per-type normalization with irreps, `l>0` channels are scaled by std only (mean subtraction only for scalar `l=0` blocks).
- Fitted stats/transform parameters are stored in dataset `fixed_fields` (keys like `_mean_.*`, `_std_.*`, `_transform_.*`).

## 5) Train/validation/test consistency

`DatasetBuilder` (`geqtrain/train/components/dataset_builder.py`) controls split behavior.

- If only train data is provided and no explicit indices/counts are set:
  - default split is `train_split_fraction: 0.8` (80/20 train/validation),
  - configurable via `train_split_fraction`.
- When train and validation are separate datasets:
  - by default (`share_train_normalization_across_splits: true`), validation is re-standardized to train-fitted stats.
- Test behavior:
  - by default, test is also aligned to train-fitted stats when the train dataset can be loaded.
  - if train data cannot be loaded in that run, test keeps its own fitted stats and a warning is logged.
  - set `share_train_normalization_across_splits: false` to disable alignment.

## 6) Evaluation and denormalization

`geqtrain/scripts/evaluate.py`:

- Loads the model checkpoint and training config.
- Merges the user-provided test config on top.
- Builds test dataset through `DatasetBuilder.build_test()`.
- Denormalizes predictions/references at logging time via `denormalize_tensor(...)` using normalization metadata attached to batch data.

## 7) Practical recommendations

- For comparable train/validation/test metrics, keep `share_train_normalization_across_splits: true`.
- Use `per_type` for node-level targets whose scale depends on species/type.
- For equivariant targets, provide irreps in normalization mode (`per_type:<irreps>`).
- If you run evaluation on a new external test dataset, keep the original train dataset path available so train-fitted normalization can be reused automatically.
