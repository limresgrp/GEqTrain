from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np


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
    The ensembled batch is created following the order of batch['dataset_raw_file_name']
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Group indices by ensemble (each molecule)
        self.ensemble_indices = _group_by_ensemble(self.dataset.n_observations, self.dataset.ensemble_indices)
        self.n_obs = self.dataset.n_observations.sum()

    def __iter__(self):
        """
        np.random.shuffle is called once per epoch (once for train and once for val)
        yields batches, called once per batch step while iteraing dloader
        Returns batches, ensuring all conformations of a molecule appear together.
        """
        np.random.shuffle(self.ensemble_indices)  # Shuffle molecules
        batch = []

        for ensemble in self.ensemble_indices:
            batch.extend(ensemble)
            while len(batch) > self.batch_size:
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
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)

        # Step 1: Group dataset indices by ensemble
        self.ensemble_indices = _group_by_ensemble(self.dataset.n_observations, self.dataset.ensemble_indices)

        # Step 2: Adjust total size to be divisible across workers
        self.total_size = len(self.ensemble_indices) - (len(self.ensemble_indices) % self.num_replicas)

        # Step 3: Split ensembles across distributed workers
        self.indices = self._get_distributed_indices()

    def _get_distributed_indices(self):
        """
        Splits ensemble indices across GPUs, ensuring each GPU gets whole molecules.
        """
        # Shuffle ensembles at the beginning of each epoch
        if self.shuffle:
            np.random.shuffle(self.ensemble_indices)

        # Flatten into a single list of indices
        flat_indices = [idx for ensemble in self.ensemble_indices for idx in ensemble]

        # Ensure even distribution across workers (truncate if needed)
        flat_indices = flat_indices[:self.total_size]

        # Partition into `num_replicas` parts (one for each GPU)
        indices = flat_indices[self.rank::self.num_replicas]
        return indices

    def __iter__(self):
        yield self.indices  # Yield distributed indices for current GPU

    def __len__(self):
        return len(self.indices)
