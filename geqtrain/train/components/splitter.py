# components/splitter.py
import torch
from geqtrain.train.components.setup import parse_idcs_file

class DatasetSplitter:
    """
    Handles the logic of splitting datasets into training and validation sets
    based on a hierarchical set of rules.
    """
    def __init__(self, trainer):
        # The splitter needs access to certain trainer states and configs
        self.config = trainer.config
        self.logger = trainer.logger
        self.dataset_rng = trainer.dataset_rng
        
        # Initial state from the trainer
        self.train_idcs = trainer.train_idcs
        self.val_idcs = trainer.val_idcs
        self.n_train = trainer.n_train
        self.n_valid = trainer.n_valid

    def split(self, train_dset, val_dset=None):
        """
        Robustly determines training and validation indices and updates n_train/n_valid counts.
        """
        # --- Step 1: Resolve indices from all explicit sources ---
        # `self.train_idcs` is already populated if `n_train: load` was used
        
        # Check for file paths for any indices not already loaded
        if self.train_idcs is None and isinstance(self.n_train, str):
            self.logger.info(f"Loading training indices from file: {self.n_train}")
            self.train_idcs = [parse_idcs_file(self.n_train)]
        
        if self.val_idcs is None and isinstance(self.n_valid, str):
            self.logger.info(f"Loading validation indices from file: {self.n_valid}")
            self.val_idcs = [parse_idcs_file(self.n_valid)]

        # --- Step 2: Update counts from any resolved indices ---
        # This is the key fix to prevent errors with the 'load' string.
        if self.train_idcs is not None:
            self.n_train = [len(t) for t in self.train_idcs]
        
        if self.val_idcs is not None:
            self.n_valid = [len(t) for t in self.val_idcs]

        # --- Step 3: Decide the final strategy based on what has been resolved ---
        if self.train_idcs is not None and self.val_idcs is not None:
            self.logger.info("Using fully provided training and validation indices.")
            return self.train_idcs, self.val_idcs

        elif self.train_idcs is not None: # But val_idcs is None
            self.val_idcs = self._generate_val_idcs_from_remaining(train_dset, self.train_idcs)
            return self.train_idcs, self.val_idcs
            
        elif self.val_idcs is not None: # But train_idcs is None
            self.train_idcs = self._generate_train_idcs_from_remaining(train_dset, self.val_idcs)
            return self.train_idcs, self.val_idcs
        else: # Neither is resolved yet, so split from counts or default
            return self._split_from_counts(train_dset, val_dset)

    def _generate_val_idcs_from_remaining(self, dset, train_idcs):
        """Generates val_idcs by sampling from data not in train_idcs."""
        self.logger.info("Generating validation set from data points not used for training.")
        val_idcs = []
        n_valid_req = self.n_valid
        
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

    def _generate_train_idcs_from_remaining(self, dset, val_idcs):
        """Generates train_idcs by sampling from data not in val_idcs."""
        self.logger.warning("Validation indices were provided, but training indices were not. Proceeding to select training data from remaining samples.")
        train_idcs = []
        n_train_req = self.n_train
        
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
        
    def _split_from_counts(self, train_dset, val_dset=None):
        """Generates train/validation indices based on n_train/n_valid counts."""
        self.logger.info("Generating new train/validation split from counts or default.")
        n_train, n_valid = self.n_train, self.n_valid
        if n_train is not None and not isinstance(n_train, list): n_train = [n_train]
        if n_valid is not None and not isinstance(n_valid, list): n_valid = [n_valid]

        val_dset_provided = val_dset is not None

        def get_n_train_list():
            if n_train: return n_train
            if val_dset_provided: return train_dset.n_observations.tolist()
            if n_valid: return [n - v for n, v in zip(train_dset.n_observations, n_valid)]
            self.logger.warning("No 'n_train' or 'n_valid' provided; using default 80/20 split.")
            return (train_dset.n_observations * 0.8).astype(int).tolist()

        def get_n_valid_list(n_train_list):
            if n_valid: return n_valid
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
                if n_t + n_v > n_obs: raise ValueError("n_train + n_valid > n_observations")
                val_idcs.append(permutation[n_t : n_t + n_v])
        
        if val_dset_provided:
            for i, (n_obs, n_v) in enumerate(zip(val_dset.n_observations, n_valid_list)):
                 if n_v > n_obs: raise ValueError(f"n_valid[{i}]={n_v} is > n_observations[{i}]={n_obs}")
                 permutation = torch.randperm(n_obs, generator=self.dataset_rng)
                 val_idcs.append(permutation[:n_v])

        return train_idcs, val_idcs