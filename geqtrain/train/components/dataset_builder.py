# geqtrain/train/components/dataset_builder.py
import torch
import logging
import numpy as np
from geqtrain.data import InMemoryConcatDataset, LazyLoadingConcatDataset
from geqtrain.data._build import dataset_from_config

def save_txt_file(filename, arrays):
    with open(filename, "w") as f:
        for arr in arrays:
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            np.savetxt(f, [arr], fmt="%d")  # write as one row

def parse_txt_file(filename):
    arrays = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                arr = np.fromstring(line, sep=" ")
                arrays.append(arr)
    return arrays

class DatasetBuilder:
    """Handles instantiation, splitting, and indexing of datasets."""
    def __init__(self, config, dataset_rng, logger=None, output=None):
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.output = output
        self.dataset_rng = dataset_rng

    # --- For generating indices efficiently ---
    def resolve_split_indices(self):
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
        
        final_train_dset = self._index_dataset(train_dset, train_idcs)
        final_val_dset = self._index_dataset(val_dset if val_dset else train_dset, val_idcs)
        return final_train_dset, final_val_dset

    def build_test(self):
        """Builds and returns the final test dataset."""
        # 1. Instantiate the raw datasets
        test_dset = dataset_from_config(self.config, prefix="test")
        
        # 2. Get the indices
        # --- Step 2.1: Resolve indices from all explicit sources ---
        if isinstance(self.test_idcs, str):
            self.logger.info(f"Loading test indices from file: {self.test_idcs}")
            self.test_idcs = [torch.tensor(arr, dtype=torch.long) for arr in parse_txt_file(self.test_idcs)]
        # --- Step 2.2: Update counts from any resolved indices ---
        if self.test_idcs is not None:
            if self.n_test is not None:
                self.logger.info("test_idcs were provided; the value of test_idcs in the config will be ignored and updated to match the actual indices.")
            self.n_test = [len(t) for t in self.test_idcs]
        # --- Step 2.3: Decide the final strategy based on what has been resolved ---
        if self.test_idcs is not None:
            self.logger.info("Using fully provided test indices.")
        else:
            if self.n_test is not None and not isinstance(self.n_test, list): self.n_test = [self.n_test]
            test_idcs = []
            for i, (n_obs, n_t) in enumerate(zip(test_dset.n_observations, self.n_test)):
                if n_t > n_obs: raise ValueError(f"n_test[{i}]={n_t} is > n_observations[{i}]={n_obs}")
                permutation = torch.randperm(n_obs, generator=self.dataset_rng)
                test_idcs.append(permutation[:n_t])
            self.test_idcs = test_idcs
        
        # 3. Create the final indexed datasets
        final_test_dset = self._index_dataset(test_dset, self.test_idcs)
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
            train_idcs = [torch.tensor(arr, dtype=torch.long) for arr in parse_txt_file(train_idcs)]
        
        if isinstance(val_idcs, str):
            self.logger.info(f"Loading validation indices from file: {val_idcs}")
            val_idcs = [torch.tensor(arr, dtype=torch.long) for arr in parse_txt_file(val_idcs)]

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
            val_idcs = self._generate_val_idcs_from_remaining(train_dset_meta, train_idcs, n_val)
            return train_idcs, val_idcs
            
        elif val_idcs is not None: # But train_idcs is None
            train_idcs = self._generate_train_idcs_from_remaining(train_dset_meta, val_idcs, n_train)
            return train_idcs, val_idcs
        else: # Neither is resolved yet, so split from counts or default
            return self._split_from_counts(train_dset_meta, val_dset_meta, n_train, n_val)

    def _generate_val_idcs_from_remaining(self, dset, train_idcs, n_valid_req):
        """Generates val_idcs by sampling from data not in train_idcs."""
        self.logger.info("Generating validation set from data points not used for training.")
        val_idcs = []
        
        if n_valid_req is not None and not isinstance(n_valid_req, list): n_valid_req = [n_valid_req]

        for i, (n_obs, train_idcs_for_i) in enumerate(zip(dset.n_observations, train_idcs)):
            all_obs = torch.arange(n_obs)
            is_train_mask = torch.zeros(n_obs, dtype=torch.bool)
            is_train_mask[train_idcs_for_i.long()] = True
            available_val_idcs = all_obs[~is_train_mask]
            n_available = len(available_val_idcs)

            if n_available == 0: val_idcs.append(torch.tensor([], dtype=torch.long)); continue

            if n_valid_req is None:
                n_to_take = n_available
                self.logger.info(f"Using all {n_available} remaining samples for validation in dataset {i}.")
            else:
                n_req = n_valid_req[i]
                if n_req > n_available:
                    self.logger.warning(f"Requested {n_req} validation samples, but only {n_available} are available. Using all available.")
                    n_to_take = n_available
                else: n_to_take = n_req
            
            permutation = torch.randperm(n_available, generator=self.dataset_rng)
            selected_idcs = available_val_idcs[permutation[:n_to_take]]
            val_idcs.append(selected_idcs)
        return val_idcs

    def _generate_train_idcs_from_remaining(self, dset, val_idcs, n_train_req):
        """Generates train_idcs by sampling from data not in val_idcs."""
        self.logger.warning("Validation indices were provided, but training indices were not. Proceeding to select training data from remaining samples.")
        train_idcs = []
        
        if n_train_req is not None and not isinstance(n_train_req, list): n_train_req = [n_train_req]

        for i, (n_obs, val_idcs_for_i) in enumerate(zip(dset.n_observations, val_idcs)):
            all_obs = torch.arange(n_obs)
            is_val_mask = torch.zeros(n_obs, dtype=torch.bool)
            is_val_mask[val_idcs_for_i.long()] = True
            available_train_idcs = all_obs[~is_val_mask]
            n_available = len(available_train_idcs)

            if n_available == 0: train_idcs.append(torch.tensor([], dtype=torch.long)); continue

            if n_train_req is None:
                n_to_take = n_available
                self.logger.info(f"Using all {n_available} remaining samples for training in dataset {i}.")
            else:
                n_req = n_train_req[i]
                if n_req > n_available:
                    self.logger.warning(f"Requested {n_req} training samples, but only {n_available} are available. Using all available.")
                    n_to_take = n_available
                else: n_to_take = n_req
            
            permutation = torch.randperm(n_available, generator=self.dataset_rng)
            selected_idcs = available_train_idcs[permutation[:n_to_take]]
            train_idcs.append(selected_idcs)
        return train_idcs
        
    def _split_from_counts(self, train_dset, val_dset, n_train, n_val):
        """Generates train/validation indices based on n_train/n_val counts."""
        self.logger.info("Generating new train/validation split from counts or default.")
        if n_train is not None and not isinstance(n_train, list): n_train = [n_train]
        if n_val is not None and not isinstance(n_val, list): n_val = [n_val]

        val_dset_provided = val_dset is not None

        def get_n_train_list():
            if n_train: return n_train
            if val_dset_provided: return train_dset.n_observations.tolist()
            if n_val: return [n - v for n, v in zip(train_dset.n_observations, n_val)]
            self.logger.warning("No 'n_train' or 'n_val' provided; using default 80/20 split.")
            return (train_dset.n_observations * 0.8).astype(int).tolist()

        def get_n_valid_list(n_train_list):
            if n_val: return n_val
            source_dset = val_dset if val_dset_provided else train_dset
            if val_dset_provided: return source_dset.n_observations.tolist()
            return [n - t for n, t in zip(source_dset.n_observations, n_train_list)]

        n_train_list = get_n_train_list()
        n_valid_list = get_n_valid_list(n_train_list)

        train_idcs, val_idcs = [], []

        for i, (n_obs, n_t) in enumerate(zip(train_dset.n_observations, n_train_list)):
            if n_t > n_obs: raise ValueError(f"n_train[{i}]={n_t} is > n_observations[{i}]={n_obs}")
            permutation = torch.randperm(n_obs, generator=self.dataset_rng)
            train_idcs.append(permutation[:n_t])
            if not val_dset_provided:
                n_v = n_valid_list[i]
                if n_t + n_v > n_obs: raise ValueError("n_train + n_val > n_observations")
                val_idcs.append(permutation[n_t : n_t + n_v])
        
        if val_dset_provided:
            for i, (n_obs, n_v) in enumerate(zip(val_dset.n_observations, n_valid_list)):
                 if n_v > n_obs: raise ValueError(f"n_val[{i}]={n_v} is > n_observations[{i}]={n_obs}")
                 permutation = torch.randperm(n_obs, generator=self.dataset_rng)
                 val_idcs.append(permutation[:n_v])

        return train_idcs, val_idcs
    
    def _index_dataset(self, dataset, indices):
        """Selects a subset of a ConcatDataset using a list of lists of indices."""
        indexed_subdatasets = []
        if not isinstance(indices, list) or not all(isinstance(i, (list, torch.Tensor)) for i in indices):
            raise TypeError(f"indices must be a list of lists/tensors, but got {type(indices)}")
        
        for d, idcs in zip(dataset.datasets, indices):
            if len(idcs) > 0:
                if isinstance(dataset, InMemoryConcatDataset):
                    indexed_subdatasets.append(d.index_select(idcs))
                elif isinstance(dataset, LazyLoadingConcatDataset):
                    indexed_subdatasets.append(d[idcs].reshape(-1))
        
        if isinstance(dataset, InMemoryConcatDataset):
            return InMemoryConcatDataset(indexed_subdatasets)
        if isinstance(dataset, LazyLoadingConcatDataset):
            return dataset.from_indexed_dataset(indices)
        raise TypeError(f"Unsupported dataset type for indexing: {type(dataset)}")