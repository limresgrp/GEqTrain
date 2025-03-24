from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import math


def _group_by_ensemble(n_observations, ensemble_indices):
    """
    Groups dataset indices by their ensemble index.
    """
    ensemble_dict = {}
    offset = 0
    for n_observations, ensemble_index in zip(n_observations, ensemble_indices):
        if ensemble_index not in ensemble_dict:
            ensemble_dict[ensemble_index] = []
        ensemble_dict[ensemble_index].extend(list(range(offset, offset + n_observations)))  # Store all conformations
        offset += n_observations
    
    return list(ensemble_dict.values())

class EnsembleSampler(Sampler):
    """
    Ensures that all conformations of a molecule (from the same npz file) 
    appear in the same batch.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group indices by ensemble (each molecule)
        self.ensemble_indices = _group_by_ensemble(self.dataset.n_observations, self.dataset.ensemble_indices)
        self.n_obs = self.dataset.n_observations.sum()

    def __iter__(self):
        """
        Returns batches, ensuring all conformations of a molecule appear together.
        """
        np.random.shuffle(self.ensemble_indices)  # Shuffle molecules
        batch = []
        
        for ensemble in self.ensemble_indices:
            batch.extend(ensemble)
            while len(batch) >= self.batch_size:
                yield batch[:self.batch_size]  # Yield a full batch
                batch = batch[self.batch_size:]  # Keep remaining elements for next batch
        
        if batch:  # Yield any remaining elements
            yield batch

    def __len__(self):
        # Return the number of batches
        return (self.n_obs + self.batch_size - 1) // self.batch_size



class EnsembleDistributedSampler(DistributedSampler):
    """
    Distributed sampler that ensures all conformations of a molecule (ensemble) 
    are always assigned to the same worker.
    !!! TODO STILL NOT WORKING PROPERLY !!!
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)

        # Step 1: Group dataset indices by ensemble
        self.all_ensemble_indices = _group_by_ensemble(self.dataset.n_observations, self.dataset.ensemble_indices)
        self.n_obs = self.dataset.n_observations.sum()

        # Step 2: Adjust total size to be divisible across workers
        self.num_samples = math.ceil((self.n_obs - self.num_replicas) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        # Step 3: Heuristically assign ensembles to distributed workers
        self.ensemble_indices = self._assign_ensembles_to_workers()

    def _assign_ensembles_to_workers(self):
            """
            Heuristically assigns ensembles to workers to minimize the number of dropped observations.
            """
            # Shuffle ensembles at the beginning of each epoch
            if self.shuffle:
                np.random.shuffle(self.all_ensemble_indices)

            # Initialize worker assignments
            worker_assignments = [[] for _ in range(self.num_replicas)]
            worker_sizes = [0] * self.num_replicas

            # Assign ensembles to workers
            for ensemble in self.all_ensemble_indices:
                # Find the worker with the least number of samples
                min_worker = np.argmin(worker_sizes)
                worker_assignments[min_worker].append(ensemble)
                worker_sizes[min_worker] += len(ensemble)

            # Ensure each worker has exactly self.num_samples samples
            for i in range(self.num_replicas):
                while worker_sizes[i] > self.num_samples:
                    diff = worker_sizes[i] - self.num_samples
                    worker_assignment = worker_assignments[i]
                    for ensemble_id in range(min(len(worker_assignment), diff)):
                        worker_assignment[ensemble_id] = worker_assignment[ensemble_id][:-1]
                        worker_sizes[i] -= 1

            return worker_assignments[self.rank]

    def __iter__(self):
        """
        Returns batches, ensuring all conformations of a molecule appear together.
        """
        np.random.shuffle(self.ensemble_indices)  # Shuffle molecules
        batch = []
        
        for ensemble in self.ensemble_indices:
            batch.extend(ensemble)
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]  # Yield a full batch
                batch = batch[self.batch_size:]  # Keep remaining elements for next batch
        
        if batch:  # Yield any remaining elements
            yield batch
        yield self.indices  # Yield distributed indices for current GPU
