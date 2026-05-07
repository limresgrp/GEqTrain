# geqtrain/train/components/dataset_builder.py
from typing import List, Tuple, Dict, Any
import torch
import logging
import numpy as np
from geqtrain.data import InMemoryConcatDataset, LazyLoadingConcatDataset
from e3nn.o3 import Irreps
from geqtrain.data._build import dataset_from_config
from geqtrain.data import AtomicDataDict
from geqtrain.utils.normalization import (
    GLOBAL_MODE,
    PER_TYPE_MODE,
    apply_forward_transform,
    denormalize_tensor,
    get_global_stat_keys,
    get_per_type_stat_keys,
    get_transform_param_key,
    resolve_normalization_map,
)

def save_txt_file(filename, arrays):
    with open(filename, "w") as f:
        for arr in arrays:
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            np.savetxt(f, [np.asarray(arr)], fmt="%d")  # write as one row

def parse_txt_file(filename):
    arrays = []
    has_multi_col = False
    with open(filename, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                arr = np.fromstring(line, sep=" ").astype(int).tolist()
                if len(arr) > 1:
                    has_multi_col = True
                arrays.append(arr)
    if not arrays:
        return arrays
    if has_multi_col:
        return arrays
    # If every non-empty line has a single index, treat the file as one list.
    flat = [value for row in arrays for value in row]
    return [flat]

class DatasetBuilder:
    """Handles instantiation, splitting, and indexing of datasets."""
    def __init__(self, config, dataset_rng, logger=None, output=None):
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.output = output
        self.dataset_rng = dataset_rng

    def _get_default_train_split_fraction(self) -> float:
        train_split_fraction = self.config.get("train_split_fraction", 0.8)
        try:
            train_split_fraction = float(train_split_fraction)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Config key 'train_split_fraction' must be a float strictly between 0 and 1."
            ) from exc

        if not (0.0 < train_split_fraction < 1.0):
            raise ValueError(
                f"Config key 'train_split_fraction' must be strictly between 0 and 1, got {train_split_fraction}."
            )
        return train_split_fraction

    # --- For generating indices efficiently ---
    def resolve_split_indices(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Resolves train/val indices without loading the full dataset.
        Returns:
            (list of tensors, list of tensors): train_idcs, val_idcs
        """
        self.logger.info("Resolving dataset indices...")

        # The rest of this logic is moved from the old `build_train_val`
        train_idcs, val_idcs = self._resolve_train_val_indices()

        # Save the indices (only on master process, guarded by self.output)
        if self.output is not None:
            save_txt_file(self.output.generate_file(f"train_idcs.txt"), train_idcs)
            save_txt_file(self.output.generate_file(f"val_idcs.txt"), val_idcs)
        
        return train_idcs, val_idcs

    # --- For building final datasets from indices ---
    def build_datasets_from_indices(self, train_idcs, val_idcs):
        """
        Instantiates the full datasets and applies the provided indices.
        """
        self.logger.info("Building final datasets from indices...")

        # Now, instantiate the full, potentially in-memory datasets
        train_dset = dataset_from_config(self.config, prefix="train")
        try:
            val_dset = dataset_from_config(self.config, prefix="validation")
        except KeyError:
            val_dset = None

        if val_dset is not None:
            self._align_target_normalization_to_train(train_dset=train_dset, target_dset=val_dset, target_name="validation")
        
        final_train_dset = self._index_dataset(train_dset, train_idcs)
        final_val_dset = self._index_dataset(val_dset if val_dset else train_dset, val_idcs)
        return final_train_dset, final_val_dset

    def build_test(self):
        """Builds and returns the final test dataset."""
        # 1. Instantiate the raw datasets
        self.logger.info("Building test dataset...")
        test_dset = dataset_from_config(self.config, prefix="test")
        train_dset = None
        try:
            train_dset = dataset_from_config(self.config, prefix="train")
        except Exception as exc:
            self.logger.warning(
                "Could not load train dataset for test normalization alignment; "
                "test data will keep its own fitted normalization. Error: %s",
                exc,
            )
        if train_dset is not None:
            self._align_target_normalization_to_train(
                train_dset=train_dset,
                target_dset=test_dset,
                target_name="test",
            )

        test_idcs = self.config.get("test_idcs")
        n_test = self.config.get("n_test")
        
        # 2. Get the indices
        # --- Step 2.1: Resolve indices from all explicit sources ---
        if isinstance(test_idcs, str):
            self.logger.info(f"Loading test indices from file: {test_idcs}")
            test_idcs = parse_txt_file(test_idcs)
        # --- Step 2.2: Update counts from any resolved indices ---
        if test_idcs is not None:
            if n_test is not None:
                self.logger.info("test_idcs were provided; the value of test_idcs in the config will be ignored and updated to match the actual indices.")
            n_test = [len(t) for t in test_idcs]
        # --- Step 2.3: Decide the final strategy based on what has been resolved ---
        if test_idcs is None:
            self.logger.info("Generating new test indices from counts.")
            if n_test is None:
                n_test = [len(d) for d in test_dset.datasets]
            if not isinstance(n_test, list): n_test = [n_test]
            test_idcs = self._generate_indices_from_pool(test_dset, n_test, pool_name="test")
        
        # 3. Create the final indexed datasets
        final_test_dset = self._index_dataset(test_dset, test_idcs)
        return final_test_dset

    def _resolve_train_val_indices(self):
        """
        Robustly determines training and validation indices and updates n_train/n_val counts.
        """

        train_idcs = self.config.get("train_idcs", None)
        val_idcs   = self.config.get("val_idcs", None)
        n_train    = self.config.get("n_train", None)
        n_val      = self.config.get("n_val", None)

        # --- Step 1: Resolve indices from all explicit sources ---
        
        # Check for file paths for any indices not already loaded
        if isinstance(train_idcs, str):
            self.logger.info(f"Loading training indices from file: {train_idcs}")
            train_idcs = parse_txt_file(train_idcs)
        
        if isinstance(val_idcs, str):
            self.logger.info(f"Loading validation indices from file: {val_idcs}")
            val_idcs = parse_txt_file(val_idcs)

        # --- Step 2: Update counts from any resolved indices ---
        if train_idcs is not None:
            if n_train is not None:
                self.logger.info("train_idcs were provided; the value of n_train in the config will be ignored and updated to match the actual indices.")
            n_train = [len(t) for t in train_idcs]
        
        if val_idcs is not None:
            if n_val is not None:
                self.logger.info("n_val were provided; the value of n_val in the config will be ignored and updated to match the actual indices.")
            n_val = [len(t) for t in val_idcs]

        # --- Step 3: Decide the final strategy based on what has been resolved ---
        if train_idcs is not None and val_idcs is not None:
            self.logger.info("Using fully provided training and validation indices.")
            return train_idcs, val_idcs

        # Load lightweight, metadata-only versions of the datasets
        # This assumes dataset_from_config can be modified to support this
        train_dset_meta = dataset_from_config(self.config, prefix="train", metadata_only=True)
        try:
            val_dset_meta = dataset_from_config(self.config, prefix="validation", metadata_only=True)
        except KeyError:
            val_dset_meta = None
        
        if train_idcs is not None: # But val_idcs is None
            self.logger.info("Generating validation set from data points not used for training.")
            val_idcs = self._generate_indices_from_pool(train_dset_meta, n_val, exclusion_pool=train_idcs, pool_name="validation")
            return train_idcs, val_idcs
        elif val_idcs is not None: # But train_idcs is None
            self.logger.warning("Validation indices were provided, but training indices were not. Selecting training data from remaining samples.")
            train_idcs = self._generate_indices_from_pool(train_dset_meta, n_train, exclusion_pool=val_idcs, pool_name="training")
            return train_idcs, val_idcs
        else: # Neither is resolved yet, so split from counts or default
            return self._split_from_counts(train_dset_meta, val_dset_meta, n_train, n_val)

    def _generate_indices_from_pool(self, dset, n_to_generate, exclusion_pool=None, pool_name="data"):
        """
        Generates a list of lists of indices from a dataset.

        Args:
            dset: The dataset to sample from.
            n_to_generate (list or None): A list with the number of samples to generate for each sub-dataset.
                                          If None, all available samples are taken.
            exclusion_pool (list of lists, optional): Indices to exclude from the sampling pool.
            pool_name (str, optional): Name of the set being generated for logging.
        """
        generated_idcs = []
        if n_to_generate is not None and not isinstance(n_to_generate, list):
            n_to_generate = [n_to_generate]

        n_observations_list = dset.n_observations.tolist()
        exclusion_pool = exclusion_pool if exclusion_pool is not None else [[] for _ in n_observations_list]

        for i, (n_obs, exclude_idcs) in enumerate(zip(n_observations_list, exclusion_pool)):
            if n_obs == 0:
                generated_idcs.append([])
                continue

            available_idcs = np.arange(n_obs)
            if len(exclude_idcs) > 0:
                is_excluded_mask = np.zeros(n_obs, dtype=bool)
                is_excluded_mask[exclude_idcs] = True
                available_idcs = available_idcs[~is_excluded_mask]
            
            n_available = len(available_idcs)
            if n_available == 0:
                generated_idcs.append([])
                continue
            
            n_to_take = n_available
            if n_to_generate is not None:
                n_req = n_to_generate[i]
                if n_req > n_available:
                    self.logger.warning(f"Requested {n_req} {pool_name} samples for dataset {i}, but only {n_available} are available. Using all available.")
                else:
                    n_to_take = n_req
            else:
                 self.logger.info(f"Using all {n_available} available samples for {pool_name} in dataset {i}.")
            
            permutation = self.dataset_rng.permutation(n_available)
            selected_idcs = available_idcs[permutation[:n_to_take]]
            generated_idcs.append(selected_idcs.tolist())
            
        return generated_idcs

    def _split_from_counts(self, train_dset, val_dset, n_train, n_val):
        """Generates train/validation indices based on n_train/n_val counts."""
        self.logger.info("Generating new train/validation split from counts or default.")
        if n_train is not None and not isinstance(n_train, list): n_train = [n_train]
        if n_val is not None and not isinstance(n_val, list): n_val = [n_val]

        val_dset_provided = val_dset is not None

        def get_n_train_list():
            # This logic remains the same: if user provides explicit counts, use them.
            if n_train: return n_train
            if val_dset_provided: return train_dset.n_observations.tolist()
            if n_val: return [n - v for n, v in zip(train_dset.n_observations, n_val)]

            train_split_fraction = self._get_default_train_split_fraction()
            validation_split_fraction = 1.0 - train_split_fraction
            self.logger.warning(
                "No 'n_train' or 'n_val' provided; using the default "
                f"{train_split_fraction:.0%}/{validation_split_fraction:.0%} "
                "train/validation split on the total dataset size."
            )
            
            n_observations_list = train_dset.n_observations.tolist()
            num_datasets = len(n_observations_list)
            total_observations = sum(n_observations_list)

            if total_observations == 0:
                return [0] * num_datasets # Handle empty dataset case

            # 1. Calculate the total number of training samples required from the configured fraction.
            target_n_train = int(total_observations * train_split_fraction)

            # 2. Distribute this total count across the sub-datasets
            # We "deal" one training sample to each dataset in a round-robin fashion
            # until we've allocated the total number of required training samples.
            n_train_list = [0] * num_datasets
            allocated_count = 0
            current_dset_idx = 0
            
            while allocated_count < target_n_train:
                # If the current sub-dataset still has unallocated samples, assign one to train
                if n_train_list[current_dset_idx] < n_observations_list[current_dset_idx]:
                    n_train_list[current_dset_idx] += 1
                    allocated_count += 1
                
                # Move to the next dataset for the next allocation
                current_dset_idx = (current_dset_idx + 1) % num_datasets
            
            return n_train_list

        def get_n_valid_list(n_train_list):
            if n_val: return n_val
            source_dset = val_dset if val_dset_provided else train_dset
            if val_dset_provided: return source_dset.n_observations.tolist()
            return [n - t for n, t in zip(source_dset.n_observations, n_train_list)]

        n_train_list = get_n_train_list()
        n_valid_list = get_n_valid_list(n_train_list)

        train_idcs, val_idcs = [], []

        train_idcs = self._generate_indices_from_pool(train_dset, n_train_list, pool_name="training")
        
        if val_dset_provided:
            val_idcs = self._generate_indices_from_pool(val_dset, n_valid_list, pool_name="validation")
        else:
            val_idcs = self._generate_indices_from_pool(train_dset, n_valid_list, exclusion_pool=train_idcs, pool_name="validation")

        return train_idcs, val_idcs
    
    def _index_dataset(self, dataset, indices):
        """Selects a subset of a ConcatDataset using a list of lists of indices."""
        indexed_subdatasets = []
        if not isinstance(indices, list):
             raise TypeError(f"indices must be a list of lists, but got {type(indices)}")
        
        for d, idcs_list in zip(dataset.datasets, indices):
            if len(idcs_list) > 0:
                idcs_tensor = torch.tensor(idcs_list, dtype=torch.long)

                if isinstance(dataset, InMemoryConcatDataset):
                    indexed_subdatasets.append(d.index_select(idcs_tensor))
                elif isinstance(dataset, LazyLoadingConcatDataset):
                    indexed_subdatasets.append(d[idcs_tensor].reshape(-1))
        
        if isinstance(dataset, InMemoryConcatDataset):
            if not indexed_subdatasets: return None
            return InMemoryConcatDataset(indexed_subdatasets)
        if isinstance(dataset, LazyLoadingConcatDataset):
            if not any(len(i) > 0 for i in indices): return None
            return dataset.from_indexed_dataset(indices)
        raise TypeError(f"Unsupported dataset type for indexing: {type(dataset)}")

    def _align_target_normalization_to_train(self, train_dset, target_dset, target_name: str):
        if not bool(self.config.get("share_train_normalization_across_splits", True)):
            return
        normalization_specs = resolve_normalization_map(self.config)
        if len(normalization_specs) == 0:
            return
        if not isinstance(train_dset, InMemoryConcatDataset) or not isinstance(target_dset, InMemoryConcatDataset):
            self.logger.warning(
                "Skipping normalization alignment for %s dataset because only InMemoryConcatDataset is currently supported.",
                target_name,
            )
            return
        if len(train_dset.datasets) == 0 or len(target_dset.datasets) == 0:
            return

        train_by_ensemble = {getattr(ds, "ensemble_index", idx): ds for idx, ds in enumerate(train_dset.datasets)}
        fallback_train_ds = train_dset.datasets[0]

        for idx, target_ds in enumerate(target_dset.datasets):
            ensemble_id = getattr(target_ds, "ensemble_index", idx)
            train_ref_ds = train_by_ensemble.get(ensemble_id, fallback_train_ds)
            if ensemble_id not in train_by_ensemble:
                self.logger.warning(
                    "No train normalization reference found for %s ensemble %s; falling back to first train dataset.",
                    target_name,
                    ensemble_id,
                )
            self._restandardize_dataset_to_reference(
                target_ds=target_ds,
                reference_fixed_fields=train_ref_ds.fixed_fields,
                normalization_specs=normalization_specs,
                target_name=target_name,
            )

    def _resolve_reference_transform_cfg(
        self,
        field: str,
        spec: Dict[str, Any],
        reference_fixed_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfg = dict(spec.get("transform", {"name": "none"}))
        if cfg.get("name", "none") == "yeo_johnson":
            lam_key = get_transform_param_key(field, "lambda")
            lam_val = reference_fixed_fields.get(lam_key, None)
            if lam_val is None:
                cfg["lambda"] = 1.0
            elif torch.is_tensor(lam_val):
                cfg["lambda"] = float(lam_val.reshape(-1)[0].item())
            else:
                cfg["lambda"] = float(lam_val)
        return cfg

    def _apply_standardization_with_reference(
        self,
        values: torch.Tensor,
        field: str,
        spec: Dict[str, Any],
        reference_fixed_fields: Dict[str, Any],
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        mode = spec.get("mode")
        irreps_str = spec.get("irreps")
        irreps = Irreps(irreps_str) if irreps_str else None

        if mode == PER_TYPE_MODE:
            mean_key, std_key = get_per_type_stat_keys(field)
            if mean_key not in reference_fixed_fields or std_key not in reference_fixed_fields:
                raise KeyError(
                    f"Reference normalization for field '{field}' is missing per-type stats keys "
                    f"'{mean_key}' and/or '{std_key}'."
                )
            means = reference_fixed_fields[mean_key]
            stds = reference_fixed_fields[std_key]
            if not torch.is_tensor(means):
                means = torch.as_tensor(means, device=values.device, dtype=values.dtype)
            else:
                means = means.to(device=values.device, dtype=values.dtype)
            if not torch.is_tensor(stds):
                stds = torch.as_tensor(stds, device=values.device, dtype=values.dtype)
            else:
                stds = stds.to(device=values.device, dtype=values.dtype)

            means_expanded = means[node_types]
            stds_expanded = stds[node_types]
            out = values.clone()

            if irreps is not None:
                i = 0
                for (_, ir), slc in zip(irreps, irreps.slices()):
                    if ir.l == 0:
                        out[:, slc] -= means_expanded[:, i:i + 1]
                        out[:, slc] /= stds_expanded[:, i:i + 1]
                    else:
                        out[:, slc] /= stds_expanded[:, i:i + 1]
                    i += 1
                return out

            mean_bc = means_expanded
            std_bc = stds_expanded
            if mean_bc.dim() == 1:
                mean_bc = mean_bc.unsqueeze(-1)
            if std_bc.dim() == 1:
                std_bc = std_bc.unsqueeze(-1)
            while mean_bc.dim() < out.dim():
                mean_bc = mean_bc.unsqueeze(-1)
            while std_bc.dim() < out.dim():
                std_bc = std_bc.unsqueeze(-1)
            return (out - mean_bc) / std_bc

        if mode == GLOBAL_MODE:
            mean_key, std_key = get_global_stat_keys(field)
            if mean_key not in reference_fixed_fields or std_key not in reference_fixed_fields:
                raise KeyError(
                    f"Reference normalization for field '{field}' is missing global stats keys "
                    f"'{mean_key}' and/or '{std_key}'."
                )
            mean = reference_fixed_fields[mean_key]
            std = reference_fixed_fields[std_key]
            if not torch.is_tensor(mean):
                mean = torch.as_tensor(mean, device=values.device, dtype=values.dtype)
            else:
                mean = mean.to(device=values.device, dtype=values.dtype)
            if not torch.is_tensor(std):
                std = torch.as_tensor(std, device=values.device, dtype=values.dtype)
            else:
                std = std.to(device=values.device, dtype=values.dtype)

            if mean.numel() != 1:
                mean = mean.reshape(-1).mean()
            if std.numel() != 1:
                std = std.reshape(-1).mean()
            if abs(float(std.item())) <= 1e-8:
                return values
            return (values - mean) / std

        raise ValueError(
            f"Invalid normalization mode '{mode}' for field '{field}'. "
            f"Expected '{PER_TYPE_MODE}' or '{GLOBAL_MODE}'."
        )

    def _copy_reference_normalization_keys(
        self,
        field: str,
        spec: Dict[str, Any],
        reference_fixed_fields: Dict[str, Any],
        target_fixed_fields: Dict[str, Any],
    ):
        mode = spec.get("mode")
        if mode == PER_TYPE_MODE:
            mean_key, std_key = get_per_type_stat_keys(field)
        elif mode == GLOBAL_MODE:
            mean_key, std_key = get_global_stat_keys(field)
        else:
            return
        if mean_key in reference_fixed_fields:
            target_fixed_fields[mean_key] = reference_fixed_fields[mean_key]
        if std_key in reference_fixed_fields:
            target_fixed_fields[std_key] = reference_fixed_fields[std_key]

        if spec.get("transform", {}).get("name", "none") == "yeo_johnson":
            lam_key = get_transform_param_key(field, "lambda")
            if lam_key in reference_fixed_fields:
                target_fixed_fields[lam_key] = reference_fixed_fields[lam_key]

    def _restandardize_dataset_to_reference(
        self,
        target_ds,
        reference_fixed_fields: Dict[str, Any],
        normalization_specs: Dict[str, Dict[str, Any]],
        target_name: str,
    ):
        if getattr(target_ds, "data", None) is None or getattr(target_ds, "fixed_fields", None) is None:
            return
        if AtomicDataDict.NODE_TYPE_KEY not in target_ds.data:
            return

        # Build a ref mapping that denormalize_tensor can consume (stats + node types).
        current_ref = dict(target_ds.fixed_fields)
        current_ref[AtomicDataDict.NODE_TYPE_KEY] = target_ds.data[AtomicDataDict.NODE_TYPE_KEY]
        node_types = target_ds.data[AtomicDataDict.NODE_TYPE_KEY].to(dtype=torch.long).squeeze(-1)

        for field, spec in normalization_specs.items():
            if field not in target_ds.data:
                continue

            irreps_str = spec.get("irreps")
            irreps = Irreps(irreps_str) if irreps_str else None
            # 1) Recover raw-space values from current (possibly split-specific) normalization.
            raw_values = denormalize_tensor(
                target_ds.data[field].clone(),
                current_ref,
                field,
                spec,
            )
            # 2) Apply reference transform (fitted on train).
            reference_transform_cfg = self._resolve_reference_transform_cfg(
                field=field,
                spec=spec,
                reference_fixed_fields=reference_fixed_fields,
            )
            transformed = apply_forward_transform(
                raw_values,
                reference_transform_cfg,
                irreps=irreps,
            ).to(target_ds.data[field].dtype)
            # 3) Apply reference standardization (fitted on train).
            target_ds.data[field] = self._apply_standardization_with_reference(
                values=transformed,
                field=field,
                spec=spec,
                reference_fixed_fields=reference_fixed_fields,
                node_types=node_types,
            ).to(target_ds.data[field].dtype)

            # 4) Replace fixed fields for inverse transform/denormalization at eval time.
            self._copy_reference_normalization_keys(
                field=field,
                spec=spec,
                reference_fixed_fields=reference_fixed_fields,
                target_fixed_fields=target_ds.fixed_fields,
            )

        self.logger.info(
            "Aligned %s dataset normalization to train-fitted statistics for ensemble %s.",
            target_name,
            getattr(target_ds, "ensemble_index", "unknown"),
        )
