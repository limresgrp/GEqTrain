# geqtrain/train/_loss.py

from typing import Tuple
import logging
from e3nn.o3 import Irreps
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
        self.params = {} if params is None else dict(params)

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
    
    def _get_pred_key_name(self, base_key: str) -> str:
        """Determines the correct prediction key based on whether deep supervision is used."""
        if self.supervision_weights is not None:
            return base_key + AtomicDataDict.DEEP_SUPERVISION_SUFFIX
        return base_key

    def _initialize_supervision_weights(self, device, dtype):
        """Initializes the supervision weights tensor on the correct device, if not already done."""
        if self.supervision_output_dim is not None and self._supervision_weights_tensor is None:
            self._supervision_weights_tensor = torch.tensor(
                self.supervision_weights,
                device=device,
                dtype=dtype
            )

    def _handle_supervision_shapes(self, pred_key: torch.Tensor, ref_key: torch.Tensor, pred_key_name: str, ref_key_name: str) -> torch.Tensor:
        """Ensures reference tensor shape is compatible with the prediction tensor, especially for deep supervision."""
        if self.supervision_output_dim is not None:
            if pred_key.dim() < 1 or pred_key.shape[-1] != self.supervision_output_dim:
                raise ValueError(
                    f"Prediction for key '{pred_key_name}' has shape {pred_key.shape}, "
                    f"but the number of supervision weights is {self.supervision_output_dim}. "
                    "The last dimension of the prediction must match the number of weights."
                )
            if ref_key.dim() == pred_key.dim() - 1:
                ref_key = ref_key.unsqueeze(-1).expand_as(pred_key)
            elif ref_key.dim() == pred_key.dim():
                if ref_key.shape[-1] == 1:
                    ref_key = ref_key.expand_as(pred_key)
                elif ref_key.shape[-1] != self.supervision_output_dim:
                    raise ValueError(
                        f"Reference for key '{ref_key_name}' has shape {ref_key.shape}, "
                        f"which is incompatible with the number of supervision weights ({self.supervision_output_dim}) "
                        f"and prediction shape {pred_key.shape}."
                    )
            else:
                raise ValueError(
                    f"Reference for key '{ref_key_name}' has shape {ref_key.shape}, "
                    f"which is incompatible with the number of supervision weights ({self.supervision_output_dim}) "
                    f"and prediction shape {pred_key.shape}."
                )
        else:
            if ref_key.shape != pred_key.shape:
                try:
                    ref_key = ref_key.reshape(pred_key.shape)
                except: pass
        return ref_key

    def _apply_node_filter(self, pred_key: torch.Tensor, ref_key: torch.Tensor, data: dict, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filters tensors to include only center nodes if the key is a node-level property."""
        apply_filter = False
        if self.node_level_filter is True:
            apply_filter = True
        elif self.node_level_filter == 'auto' and key in _NODE_FIELDS:
            apply_filter = True

        if apply_filter:
            num_atoms = data.get(AtomicDataDict.POSITIONS_KEY).shape[0]
            center_nodes_idx = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
            if pred_key.shape[0] == num_atoms:
                pred_key = pred_key[center_nodes_idx]
            if ref_key.shape[0] == num_atoms:
                ref_key = ref_key[center_nodes_idx]
        return pred_key, ref_key

    def _calculate_loss(self, pred_key: torch.Tensor, ref_key: torch.Tensor, mean: bool) -> torch.Tensor:
        """Computes the loss, handling NaNs and applying supervision weights."""
        if self.ignore_nan:
            not_nan_mask = torch.isfinite(pred_key) & torch.isfinite(ref_key)
            pred_key = torch.nan_to_num(pred_key, nan=0.0)
            ref_key = torch.nan_to_num(ref_key, nan=0.0)
            loss = self.func(pred_key, ref_key)
            loss = loss * not_nan_mask

            if self.supervision_output_dim is not None:
                loss = loss * self._supervision_weights_tensor
                loss = loss.sum(dim=-1)
                not_nan_mask_sum = not_nan_mask.sum(dim=-1)
                if mean:
                    return loss.sum() / not_nan_mask_sum.clamp(min=1).sum()
                loss[not_nan_mask_sum == 0] = torch.nan
                return loss

            if mean:
                return loss.sum() / not_nan_mask.sum().clamp(min=1)
            loss[~not_nan_mask] = torch.nan
            return loss
        else:
            loss = self.func(pred_key, ref_key)
            if self.supervision_output_dim is not None:
                loss = loss * self._supervision_weights_tensor
                loss = loss.sum(dim=-1)
            return loss.mean() if mean else loss

    def __call__(self, pred: dict, ref: dict, key: str, mean: bool = True, destandardize_fields: dict = {}, **kwargs):
        # 1. Determine prediction key and prepare tensors
        pred_key_name = self._get_pred_key_name(key)
        pred_key, ref_key = self._prepare_tensors(pred, ref, pred_key_name, key, mean, destandardize_fields)

        # 2. Initialize supervision weights if needed
        self._initialize_supervision_weights(pred_key.device, pred_key.dtype)

        # 3. Ensure reference and prediction shapes are compatible
        ref_key = self._handle_supervision_shapes(pred_key, ref_key, pred_key_name, key)

        # 4. Apply node-level filtering if necessary
        pred_key, ref_key = self._apply_node_filter(pred_key, ref_key, pred, key)

        # 5. Calculate and return the loss
        return self._calculate_loss(pred_key, ref_key, mean)

    def _prepare_tensors(self, pred: dict, ref: dict, pred_key_name: str, ref_key_name: str, mean: bool, destandardize_fields: dict = {}):
        pred_key = pred.get(pred_key_name)
        assert isinstance(pred_key, torch.Tensor), f"Prediction for '{pred_key_name}' not a tensor."
        ref_key = ref.get(ref_key_name)
        assert isinstance(ref_key, torch.Tensor), f"Reference for '{ref_key_name}' not a tensor."

        # De-standardization for metrics (when mean=False)
        if not mean:
            # Define prefixes for standardization keys
            MEAN_KEY_PREFIX = "_mean_"
            STD_KEY_PREFIX = "_std_"
            PER_TYPE_PREFIX = "per_type"
            GLOBAL_PREFIX = "global"

            # Check for per-type standardization stats
            per_type_mean_key = f"{MEAN_KEY_PREFIX}.{PER_TYPE_PREFIX}.{ref_key_name}"
            per_type_std_key = f"{STD_KEY_PREFIX}.{PER_TYPE_PREFIX}.{ref_key_name}"
            
            # Check for global standardization stats
            global_mean_key = f"{MEAN_KEY_PREFIX}.{GLOBAL_PREFIX}.{ref_key_name}"
            global_std_key = f"{STD_KEY_PREFIX}.{GLOBAL_PREFIX}.{ref_key_name}"

            irreps = None
            if ref_key_name in destandardize_fields:
                mode_str = destandardize_fields[ref_key_name]
                parts = mode_str.split(':', 1)
                irreps_str = parts[1] if len(parts) > 1 else None
                if irreps_str:
                    irreps = Irreps(irreps_str)

            if per_type_mean_key in ref and per_type_std_key in ref:
                means = ref[per_type_mean_key].to(pred_key.device)
                stds = ref[per_type_std_key].to(pred_key.device)
                node_types = ref[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1)

                means_expanded = means[node_types]
                stds_expanded = stds[node_types]

                if irreps:
                    i = 0
                    for (mul, ir), slice in zip(irreps, irreps.slices()):
                        mean_bc = means_expanded[:, i:i+1]
                        std_bc = stds_expanded[:, i:i+1]
                        if ir.l == 0:
                            pred_key[:, slice] = pred_key[:, slice] * std_bc + mean_bc
                            ref_key[:, slice] = ref_key[:, slice] * std_bc + mean_bc
                        else: # l > 0, de-standardize norm
                            # De-standardize pred
                            norm_pred = torch.linalg.norm(pred_key[:, slice], dim=-1, keepdim=True)
                            # The standardized norm is (norm - mean) / std.
                            # To get the original norm back: norm = standardized_norm * std + mean
                            # Here, the standardized value is the norm of the vector in pred_key.
                            new_norm_pred = norm_pred * std_bc + mean_bc
                            scale_pred = new_norm_pred / norm_pred.clamp(min=1e-8)
                            pred_key[:, slice] = pred_key[:, slice] * scale_pred

                            # De-standardize ref
                            norm_ref = torch.linalg.norm(ref_key[:, slice], dim=-1, keepdim=True)
                            new_norm_ref = norm_ref * std_bc + mean_bc
                            scale_ref = new_norm_ref / norm_ref.clamp(min=1e-8)
                            ref_key[:, slice] = ref_key[:, slice] * scale_ref
                        i += 1
                else:
                    # Fallback for scalar fields without irreps info
                    mean_bc = means_expanded.view(-1, *([1] * (pred_key.dim() - 1)))
                    std_bc = stds_expanded.view(-1, *([1] * (pred_key.dim() - 1)))
                    pred_key = pred_key * std_bc + mean_bc

                    mean_bc_ref = means_expanded.view(-1, *([1] * (ref_key.dim() - 1)))
                    std_bc_ref = stds_expanded.view(-1, *([1] * (ref_key.dim() - 1)))
                    ref_key = ref_key * std_bc_ref + mean_bc_ref
                
            elif global_mean_key in ref and global_std_key in ref:
                if irreps:
                    # This case is ambiguous for global stats with multiple irreps.
                    # The per-type logic is preferred for equivariant fields.
                    logging.warning(
                        f"Global de-standardization for equivariant field '{ref_key_name}' is not fully supported and may be incorrect. "
                        "Consider using per-type standardization."
                    )
                global_mean = ref[global_mean_key]
                global_std = ref[global_std_key]

                if not torch.is_tensor(global_mean):
                    global_mean = torch.as_tensor(global_mean, device=pred_key.device, dtype=pred_key.dtype)
                else:
                    global_mean = global_mean.to(device=pred_key.device, dtype=pred_key.dtype)

                if not torch.is_tensor(global_std):
                    global_std = torch.as_tensor(global_std, device=pred_key.device, dtype=pred_key.dtype)
                else:
                    global_std = global_std.to(device=pred_key.device, dtype=pred_key.dtype)

                if global_mean.numel() != 1:
                    global_mean = global_mean.reshape(-1).mean()
                if global_std.numel() != 1:
                    global_std = global_std.reshape(-1).mean()

                if global_std.item() > 1e-8:
                    pred_key = pred_key * global_std + global_mean
                    ref_key = ref_key * global_std + global_mean

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
        self.ensemble_mode = params.pop("ensemble_mode", "auto")
        if self.ensemble_mode not in ("auto", "always", "never"):
            raise ValueError("ensemble_mode must be one of: auto, always, never")
        self.ensemble_reduce = params.pop("ensemble_reduce", "mean")
        if self.ensemble_reduce not in ("mean", "sum"):
            raise ValueError("ensemble_reduce must be one of: mean, sum")
        try:
            from torcheval.metrics import BinaryAUROC
        except ImportError:
            raise ImportError("Please `pip install torcheval` to use BinaryAUROCMetric.")
        self.metric = BinaryAUROC(**params)
        self.device = 'cpu'
        self._warned_about_target = False
        self._warned_about_ensemble = False

    def update(self, pred: dict, ref: dict, key: str):
        logits = pred[key].detach().squeeze()
        target = ref[key].detach().squeeze()
        ensemble_indices = None

        if AtomicDataDict.ENSEMBLE_INDEX_KEY in pred:
            assert AtomicDataDict.ENSEMBLE_INDEX_KEY in ref
            ensemble_indices = pred[AtomicDataDict.ENSEMBLE_INDEX_KEY].detach().squeeze()

        if target.dim() == 0: # if batch_size = 1
            target = target.unsqueeze(0)
            logits = logits.unsqueeze(0)
            if ensemble_indices is not None and ensemble_indices.dim() == 0:
                ensemble_indices = ensemble_indices.unsqueeze(0)

        # Create a mask to filter out rows with NaNs in either logits or target
        valid_mask = torch.isfinite(logits) & torch.isfinite(target)

        if not torch.all(valid_mask):
            logits = logits[valid_mask]
            target = target[valid_mask]
            if ensemble_indices is not None and valid_mask.dim() == 1:
                ensemble_indices = ensemble_indices[valid_mask]

        if logits.numel() == 0 or target.numel() == 0:
            return

        if ensemble_indices is not None and self.ensemble_mode != "never":
            ensemble_indices = ensemble_indices.to(dtype=torch.long, device=logits.device)
            num_unique = torch.unique(ensemble_indices).numel()
            has_duplicates = num_unique < ensemble_indices.numel()
            should_ensemble = (
                self.ensemble_mode == "always"
                or (self.ensemble_mode == "auto" and has_duplicates and num_unique > 1)
            )
            if should_ensemble:
                if logits.dim() > 1:
                    if logits.shape[0] == ensemble_indices.shape[0]:
                        sample_dim = 0
                    elif logits.shape[-1] == ensemble_indices.shape[0]:
                        sample_dim = logits.dim() - 1
                    else:
                        raise ValueError(
                            "Ensemble indices shape does not align with logits; "
                            f"logits shape={logits.shape}, ensemble_indices shape={ensemble_indices.shape}."
                        )
                else:
                    sample_dim = 0
                reduce_fn = scatter_mean if self.ensemble_reduce == "mean" else scatter_sum
                logits = reduce_fn(logits, ensemble_indices, dim=sample_dim)
                target = reduce_fn(target, ensemble_indices, dim=sample_dim)
            elif self.ensemble_mode == "auto" and num_unique == 1 and not self._warned_about_ensemble:
                logging.warning(
                    "BinaryAUROCMetric: ensemble_index has a single unique value; "
                    "skipping ensemble aggregation. Set ensemble_mode='always' to force it."
                )
                self._warned_about_ensemble = True

        target = target.float()
        is_binary = torch.all((target == 0) | (target == 1)).item()
        if not is_binary:
            threshold = 0.0 if target.min().item() < 0.0 else 0.5
            target = (target > threshold).int()
            if not self._warned_about_target:
                logging.warning(
                    "BinaryAUROCMetric expects targets in {0,1}. "
                    "Auto-binarizing using threshold %.3f for key '%s'.",
                    threshold,
                    key,
                )
                self._warned_about_target = True
        else:
            target = target.int()

        # Ensure metric is on the correct device
        if self.device != logits.device:
            self.device = logits.device
            self.metric.to(self.device)

        # Update with cleaned data, ensuring target is int
        if logits.numel() > 0 and target.numel() > 0:  # Only update if there is valid data
            self.metric.update(logits, target)

    def compute(self):
        return self.metric.compute().clone()

    def reset(self):
        self.metric.reset()

    def __str__(self):
        return "BinaryAUROC"
