from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import math
import torch


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

class EnsembleBatchSampler(Sampler):
    """Batch complete ensembles without splitting their conformers.

    ``batch_size`` counts ensembles (systems), not individual conformers.
    ``max_structures`` optionally samples a fixed number of conformers from
    each system on every training epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        max_structures: int = None,
    ):
        if int(batch_size) < 1:
            raise ValueError("Ensemble batch_size must be at least 1.")
        if max_structures is not None and int(max_structures) < 1:
            raise ValueError("ensemble_max_structures must be null or at least 1.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_structures = None if max_structures is None else int(max_structures)
        self.epoch = 0
        self.ensemble_indices = _group_by_ensemble(
            self.dataset.n_observations,
            self.dataset.ensemble_indices,
        )

    def set_epoch(self, epoch: int):
        self.epoch = max(0, int(epoch))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        ensemble_order = np.arange(len(self.ensemble_indices))
        if self.shuffle:
            rng.shuffle(ensemble_order)

        batch = []
        for order_idx, ensemble_idx in enumerate(ensemble_order):
            structures = np.asarray(self.ensemble_indices[int(ensemble_idx)], dtype=np.int64)
            if self.max_structures is not None and len(structures) > self.max_structures:
                if self.shuffle:
                    structures = rng.choice(structures, size=self.max_structures, replace=False)
                else:
                    structures = structures[:self.max_structures]
            batch.extend(structures.tolist())

            if (order_idx + 1) % self.batch_size == 0:
                yield batch
                batch = []

        if batch:
            yield batch

    def __len__(self):
        return (len(self.ensemble_indices) + self.batch_size - 1) // self.batch_size


# Backwards-compatible name for external imports.
EnsembleSampler = EnsembleBatchSampler



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


class CurriculumBatchSampler(Sampler):
    """Batch-level curriculum importance sampler.

    Anchor epochs iterate over every fixed batch exactly once. Priority epochs
    sample the same number of batches with replacement according to smoothed
    loss-derived probabilities.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        anchor_interval: int = 5,
        alpha: float = 0.5,
        beta_warmup_epochs: int = 10,
        gamma: float = 0.2,
        gamma_final: float = None,
        gamma_warmup_epochs: int = None,
        error_ema: float = 0.8,
        eps: float = 1.0e-12,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("CurriculumBatchSampler requires batch_size > 0.")

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.anchor_interval = max(1, int(anchor_interval))
        self.alpha = float(alpha)
        self.beta_warmup_epochs = max(1, int(beta_warmup_epochs))
        self.gamma = float(gamma)
        self.gamma_final = self.gamma if gamma_final is None else float(gamma_final)
        self.gamma_warmup_epochs = max(1, int(gamma_warmup_epochs or self.beta_warmup_epochs))
        self.error_ema = float(error_ema)
        if not 0.0 <= self.error_ema < 1.0:
            raise ValueError("curriculum_importance_sampling.error_ema must be in [0, 1).")
        self.eps = float(eps)

        self.batches = self._make_batches()
        self.num_batches = len(self.batches)
        if self.num_batches == 0:
            raise ValueError("CurriculumBatchSampler cannot sample an empty dataset.")

        self.errors = np.ones(self.num_batches, dtype=np.float64)
        self.observed_errors = np.full(self.num_batches, np.nan, dtype=np.float64)
        self.probabilities = np.full(self.num_batches, 1.0 / self.num_batches, dtype=np.float64)
        self.epoch = 0
        self.current_batch_id = None
        self.current_probability = 1.0 / self.num_batches
        self.current_importance_weight = 1.0
        self.last_epoch_was_anchor = True
        self.loss_key = None

    def _make_batches(self):
        return [
            list(range(start, min(start + self.batch_size, len(self.dataset))))
            for start in range(0, len(self.dataset), self.batch_size)
        ]

    def set_epoch(self, epoch: int):
        self.epoch = max(0, int(epoch))
        self.last_epoch_was_anchor = self.is_anchor_epoch(self.epoch)
        self.observed_errors[:] = np.nan

    def is_anchor_epoch(self, epoch: int = None) -> bool:
        epoch = self.epoch if epoch is None else max(0, int(epoch))
        return epoch == 0 or (epoch % self.anchor_interval == 0)

    def _beta(self, epoch: int = None):
        epoch = self.epoch if epoch is None else max(0, int(epoch))
        return min(1.0, epoch / self.beta_warmup_epochs)

    def _gamma(self, epoch: int = None):
        epoch = self.epoch if epoch is None else max(0, int(epoch))
        progress = min(1.0, epoch / self.gamma_warmup_epochs)
        return self.gamma + progress * (self.gamma_final - self.gamma)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.is_anchor_epoch():
            order = np.arange(self.num_batches)
            if self.shuffle:
                rng.shuffle(order)
        else:
            order = rng.choice(self.num_batches, size=self.num_batches, replace=True, p=self.probabilities)

        gamma = self._gamma()
        for batch_id in order.tolist():
            prob = float(self.probabilities[batch_id])
            self.current_batch_id = int(batch_id)
            self.current_probability = prob
            if self.is_anchor_epoch():
                self.current_importance_weight = 1.0
            else:
                self.current_importance_weight = float((1.0 / (self.num_batches * max(prob, self.eps))) ** gamma)
            yield self.batches[batch_id]

        self.current_batch_id = None
        self.current_probability = 1.0 / self.num_batches
        self.current_importance_weight = 1.0

    def update_batch_loss(self, loss_value):
        if self.current_batch_id is None:
            return
        if torch.is_tensor(loss_value):
            loss_value = loss_value.detach().float().item()
        loss_value = float(loss_value)
        if not math.isfinite(loss_value):
            return
        batch_id = int(self.current_batch_id)
        previous = self.observed_errors[batch_id]
        if math.isnan(previous):
            self.observed_errors[batch_id] = loss_value
        else:
            self.observed_errors[batch_id] = 0.5 * previous + 0.5 * loss_value

    def on_epoch_end(self):
        observed = np.isfinite(self.observed_errors)
        if np.any(observed):
            self.errors[observed] = (
                self.error_ema * self.errors[observed]
                + (1.0 - self.error_ema) * self.observed_errors[observed]
            )
        priorities = np.power(np.clip(self.errors, self.eps, None), self.alpha)
        priority_probs = priorities / priorities.sum()
        uniform = np.full(self.num_batches, 1.0 / self.num_batches, dtype=np.float64)
        beta = self._beta(self.epoch + 1)
        self.probabilities = (1.0 - beta) * uniform + beta * priority_probs
        self.probabilities = self.probabilities / self.probabilities.sum()

    def state_dict(self):
        return {
            "errors": self.errors.copy(),
            "probabilities": self.probabilities.copy(),
            "epoch": self.epoch,
            "loss_key": self.loss_key,
        }

    def load_state_dict(self, state):
        if not state:
            return
        for key in ("errors", "probabilities"):
            if key in state:
                value = np.asarray(state[key], dtype=np.float64)
                if value.shape == (self.num_batches,):
                    setattr(self, key, value.copy())
        self.epoch = int(state.get("epoch", self.epoch))
        self.loss_key = state.get("loss_key", self.loss_key)

    def ascii_histogram(self, bins: int = 10, width: int = 28) -> str:
        bins = max(1, int(bins))
        values = self.probabilities
        counts, edges = np.histogram(values, bins=bins)
        max_count = max(int(counts.max()), 1)
        lines = []
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            bar = "#" * int(round(width * int(count) / max_count))
            lines.append(f"{left:.3e}-{right:.3e} | {bar} {int(count)}")
        return "\n".join(lines)
