# geqtrain/train/_loss.py

import torch
from geqtrain.utils import instantiate_from_cls_name
from geqtrain.data import AtomicDataDict, _NODE_FIELDS
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_mean, scatter_max

def ensemble_predictions_and_targets(predictions, targets, ensemble_indices, aggregation_fn=scatter_sum):
    ''' checks whether field has already been ensembled, if not, ensembles it using ensemble_indices and the specified aggregation_fn'''
    unique_ensembles = torch.unique(ensemble_indices)

    # ensemble predictions
    if predictions.shape == torch.Size([]) and unique_ensembles.shape[0] == 1:
        is_input_already_ensembled = True
    else:
        is_input_already_ensembled = unique_ensembles.shape[0] == predictions.shape[0]

    if not is_input_already_ensembled:
        predictions = aggregation_fn(predictions, ensemble_indices)

    # ensemble targets
    if targets.shape == torch.Size([]) and unique_ensembles.shape[0] == 1:
        is_output_already_ensembled = True
    else:
        is_output_already_ensembled = unique_ensembles.shape[0] == targets.shape[0]

    if not is_output_already_ensembled:
        targets = aggregation_fn(targets, ensemble_indices) # acts just as selection and ordering wrt unique_ensembles

    return predictions, targets


class LossWrapper:
    """
    A wrapper for standard torch.nn loss functions that adds capabilities like
    ignoring NaNs and filtering to include only center nodes of edges.
    """
    def __init__(self, func_name: str, params: dict = {}):
        self.func_name = func_name
        self.params = params

        self.ignore_nan = self.params.pop("ignore_nan", False)
        self.node_level_filter = self.params.pop("node_level_filter", "auto")  # node filtering mode: 'auto', True, or False

        # New: Handle deep supervision parameters
        self.supervision_weights = self.params.pop("supervision_weights", None)

        if self.supervision_weights is not None:
            if not isinstance(self.supervision_weights, list) or not all(isinstance(w, (int, float)) for w in self.supervision_weights):
                raise ValueError(
                    f"Invalid 'supervision_weights': {self.supervision_weights}. "
                    "Must be a list of numbers (floats or ints)."
                )
            # Will be initialized as a tensor on the correct device in __call__
            self._supervision_weights_tensor = None
            self.supervision_output_dim = len(self.supervision_weights)
        else:
            self._supervision_weights_tensor = None
            self.supervision_output_dim = None

        torch_params = self.params  # Remaining params are for the torch loss function

        # Instantiate the underlying torch loss function
        self.func, _ = instantiate_from_cls_name(
            torch.nn, class_name=func_name, prefix="",
            positional_args=dict(reduction="none"), optional_args=torch_params, all_args={},
        )

    def __call__(self, pred: dict, ref: dict, key: str, mean: bool = True, **kwargs):
        # If using deep supervision, the prediction is stored under a different key
        if self.supervision_weights is not None:
            pred_key_name = key + AtomicDataDict.DEEP_SUPERVISION_SUFFIX
        else:
            pred_key_name = key

        pred_key, ref_key = self._prepare_tensors(pred, ref, pred_key_name, key)
        
        # Initialize supervision weights tensor on the correct device if needed
        if self.supervision_output_dim is not None and self._supervision_weights_tensor is None:
            self._supervision_weights_tensor = torch.tensor(
                self.supervision_weights,
                device=pred_key.device,
                dtype=pred_key.dtype
            )

        # 1. Handle supervision dimension and weights
        if self.supervision_output_dim is not None:
            # Check if pred_key has the expected supervision dimension
            if pred_key.dim() < 1 or pred_key.shape[-1] != self.supervision_output_dim:
                raise ValueError(
                    f"Prediction for key '{pred_key_name}' has shape {pred_key.shape}, "
                    f"but the number of supervision weights is {self.supervision_output_dim}. "
                    "The last dimension of the prediction must match the number of weights."
                )

            # Reshape ref_key to match pred_key's supervision dimension
            # Example: pred_key [N, F, K], ref_key [N, F] -> ref_key [N, F, 1] -> broadcast to [N, F, K]
            if ref_key.dim() == pred_key.dim() - 1: # If ref_key is missing the last supervision dim
                ref_key = ref_key.unsqueeze(-1).expand_as(pred_key)
            elif ref_key.dim() == pred_key.dim(): # If ref_key already has the same number of dims
                if ref_key.shape[-1] == 1: # If ref_key is [..., 1], expand it
                    ref_key = ref_key.expand_as(pred_key)
                elif ref_key.shape[-1] != self.supervision_output_dim: # If ref_key has a different last dim, it's an error
                    raise ValueError(
                        f"Reference for key '{key}' has shape {ref_key.shape}, "
                        f"which is incompatible with the number of supervision weights ({self.supervision_output_dim}) "
                        f"and prediction shape {pred_key.shape}."
                    )
            else: # If ref_key has a completely different number of dimensions
                 raise ValueError(
                    f"Reference for key '{key}' has shape {ref_key.shape}, "
                    f"which is incompatible with the number of supervision weights ({self.supervision_output_dim}) "
                    f"and prediction shape {pred_key.shape}."
                )
        else: # No supervision, ensure shapes match for non-supervision case
            if ref_key.shape != pred_key.shape:
                ref_key = ref_key.reshape(pred_key.shape)

        # 1. Determine if node-level filtering should be applied
        apply_filter = False
        if self.node_level_filter is True:
            apply_filter = True
        elif self.node_level_filter == 'auto' and key in _NODE_FIELDS:
            apply_filter = True

        # 2. Apply the filter if required
        if apply_filter:
            num_atoms = pred.get(AtomicDataDict.POSITIONS_KEY).shape[0]
            # Get the unique indices of nodes that are the center of an edge
            center_nodes_idx = pred[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
            # Safety check: only filter tensors that are per-atom
            if pred_key.shape[0] == num_atoms:
                pred_key = pred_key[center_nodes_idx]
            if ref_key.shape[0] == num_atoms:
                ref_key = ref_key[center_nodes_idx]

        # 3. Handle NaNs (on the potentially filtered tensors)
        if self.ignore_nan:
            not_nan_mask = torch.isfinite(pred_key) & torch.isfinite(ref_key)
            pred_key = torch.nan_to_num(pred_key, nan=0.0)
            ref_key = torch.nan_to_num(ref_key, nan=0.0)

            loss = self.func(pred_key, ref_key)
            loss = loss * not_nan_mask
            
            if self.supervision_output_dim is not None:
                loss = loss * self._supervision_weights_tensor # Apply supervision weights
                loss = loss.sum(dim=-1) # Sum over supervision dimension
                not_nan_mask_sum = not_nan_mask.sum(dim=-1) # Sum mask over supervision dimension
                if mean: return loss.sum() / not_nan_mask_sum.clamp(min=1).sum()
                loss[not_nan_mask_sum == 0] = torch.nan # Mark samples with no valid predictions as nan
                return loss

            if mean: return loss.sum() / not_nan_mask.sum().clamp(min=1)
            loss[~not_nan_mask] = torch.nan # Mark samples with no valid predictions as nan
            return loss
        else:
            loss = self.func(pred_key, ref_key)
            if self.supervision_output_dim is not None:
                loss = loss * self._supervision_weights_tensor # Apply recycling weights
                loss = loss.sum(dim=-1) # Sum over recycling dimension
            return loss.mean() if mean else loss

    def _prepare_tensors(self, pred: dict, ref: dict, pred_key_name: str, ref_key_name: str):
        pred_key = pred.get(pred_key_name)
        assert isinstance(pred_key, torch.Tensor), f"Prediction for '{pred_key_name}' not a tensor."
        ref_key = ref.get(ref_key_name)
        assert isinstance(ref_key, torch.Tensor), f"Reference for '{ref_key_name}' not a tensor."
        return pred_key, ref_key

    def __str__(self):
        return self.func_name


class RMSDMetric:
    """
    Computes the Root Mean Square Deviation for each sample in a batch.
    This metric correctly handles NaNs and is designed to work with the
    RunningStats accumulator using 'rms' reduction.
    """
    def __init__(self, ignore_nan: bool = False, **kwargs):
        self.mse = torch.nn.MSELoss(reduction="none")
        self.ignore_nan = ignore_nan
        # Signal to the Metrics class that this should be accumulated with RMS
        self.extra_params = {"reduction": "rms"}

    def __call__(self, pred: dict, ref: dict, key: str, mean: bool = True, **kwargs):
        if mean:
            raise Exception("RMSDMetric is intended for evaluation and cannot be used as a training loss.")

        pred_key, ref_key = pred[key], ref[key]
        if ref_key.shape != pred_key.shape:
            ref_key = ref_key.reshape(pred_key.shape)

        # Calculate the element-wise squared error
        squared_error = self.mse(pred_key, ref_key)

        # Calculate the mean squared error for each sample (across the feature dimension)
        per_sample_mse = torch.mean(squared_error, dim=-1)

        # 1. Compute the square root to get the per-sample RMSD.
        # RunningStats will square this value again during its RMS accumulation.
        rmsd = torch.sqrt(per_sample_mse)

        # 2. NAN HANDLING: Invalidate the entire sample if any of its
        #    feature values were originally NaN.
        if self.ignore_nan:
            # A sample is valid only if all its features are finite
            is_valid_sample = torch.all(torch.isfinite(pred_key) & torch.isfinite(ref_key), dim=-1)
            # Set the RMSD to NaN for invalid samples so they are ignored by RunningStats
            rmsd[~is_valid_sample] = torch.nan

        return rmsd

    def __str__(self):
        return "RMSD"


class FocalLossBinaryAccuracy:
    """
    Implementation of Focal Loss for binary classification tasks.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, pred: dict, ref: dict, key: str, mean: bool = True, **kwargs):
        logits = pred[key]
        target = ref[key].float()

        bce_loss = self.bce(logits, target)
        p_t = torch.exp(-bce_loss) # This is p if target=1, and 1-p if target=0

        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce_loss

        return focal_loss.mean() if mean else focal_loss

    def __str__(self):
        return "FocalLoss"


class StatefulMetric:
    """Base class for metrics that need state across batches (e.g., update, compute)."""
    def __init__(self, **params):
        # This base class can be expanded if common functionality is needed
        pass

    def update(self, pred: dict, ref: dict, key: str):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class BinaryAUROCMetric(StatefulMetric):
    """
    Stateful wrapper for torcheval's BinaryAUROC that safely handles NaNs.
    """
    def __init__(self, **params):
        super().__init__()
        try:
            from torcheval.metrics import BinaryAUROC
        except ImportError:
            raise ImportError("Please `pip install torcheval` to use BinaryAUROCMetric.")
        self.metric = BinaryAUROC(**params)
        self.device = 'cpu'

    def update(self, pred: dict, ref: dict, key: str):
        logits = pred[key].detach().squeeze()
        target = ref[key].detach().squeeze()

        if AtomicDataDict.ENSEMBLE_INDEX_KEY in pred:
            assert AtomicDataDict.ENSEMBLE_INDEX_KEY in ref
            logits, target = ensemble_predictions_and_targets(logits, target, pred[AtomicDataDict.ENSEMBLE_INDEX_KEY])
            n_ens = pred[AtomicDataDict.ENSEMBLE_INDEX_KEY].shape[0]/torch.unique(pred[AtomicDataDict.ENSEMBLE_INDEX_KEY]).shape[0]
            target /= n_ens
            not_nan_filter = (scatter_sum(ref[key].squeeze(), pred[AtomicDataDict.ENSEMBLE_INDEX_KEY])+1)
            not_nan_filter = torch.nan_to_num(not_nan_filter, nan=0.0)
            not_nan_filter = torch.where((not_nan_filter != 0) & (not_nan_filter != 1), torch.ones_like(not_nan_filter), not_nan_filter)

        if target.dim() == 0: # if batch_size = 1
            target = target.unsqueeze(0)
            logits = logits.unsqueeze(0)

        # Create a mask to filter out rows with NaNs in either logits or target
        valid_mask = torch.isfinite(logits) & torch.isfinite(target)

        if not torch.all(valid_mask):
            logits = logits[valid_mask]
            target = target[valid_mask]

        # Ensure metric is on the correct device
        if self.device != logits.device:
            self.device = logits.device
            self.metric.to(self.device)

        # Update with cleaned data, ensuring target is int
        if logits.numel() > 0 and target.numel() > 0:  # Only update if there is valid data
            self.metric.update(logits, target.int())

    def compute(self):
        return self.metric.compute().clone()

    def reset(self):
        self.metric.reset()

    def __str__(self):
        return "BinaryAUROC"