# geqtrain/train/_loss.py

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging
import math
import torch
import torch.nn.functional as F
from geqtrain.utils import instantiate_from_cls_name
from geqtrain.data import AtomicDataDict, _NODE_FIELDS
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_mean, scatter_max
from geqtrain.utils.normalization import denormalize_tensor

@dataclass
class PreparedTarget:
    pred: Dict[str, torch.Tensor]
    ref: Dict[str, torch.Tensor]
    pred_key: torch.Tensor
    ref_key: torch.Tensor
    node_types: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


def graph_zero_like(tensor: torch.Tensor) -> torch.Tensor:
    """Return a scalar zero that remains connected to `tensor`'s autograd graph."""
    return tensor.sum() * 0.0


def _row_count(tensor: torch.Tensor) -> Optional[int]:
    if not torch.is_tensor(tensor) or tensor.ndim == 0:
        return None
    return int(tensor.shape[0])


def _center_nodes(data: dict) -> Optional[torch.Tensor]:
    if AtomicDataDict.EDGE_INDEX_KEY not in data:
        return None
    return data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()


def _can_index_by_center(tensor: torch.Tensor, center_nodes: Optional[torch.Tensor], row_count: int) -> bool:
    if center_nodes is None or not torch.is_tensor(tensor) or tensor.ndim == 0:
        return False
    if int(center_nodes.numel()) != row_count or center_nodes.numel() == 0:
        return False
    return int(tensor.shape[0]) > int(center_nodes.max().item())


def _align_rows(tensor: Optional[torch.Tensor], row_count: int, center_nodes: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or not torch.is_tensor(tensor) or tensor.ndim == 0:
        return tensor
    if int(tensor.shape[0]) == row_count:
        return tensor
    if _can_index_by_center(tensor, center_nodes, row_count):
        return tensor[center_nodes]
    return None


def _align_pred_ref_rows(
    pred_key: torch.Tensor,
    ref_key: torch.Tensor,
    key: str,
    center_nodes: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_rows = _row_count(pred_key)
    ref_rows = _row_count(ref_key)
    if pred_rows is None or ref_rows is None or pred_rows == ref_rows or key not in _NODE_FIELDS:
        return pred_key, ref_key

    if _can_index_by_center(ref_key, center_nodes, pred_rows):
        ref_key = ref_key[center_nodes]
    elif _can_index_by_center(pred_key, center_nodes, ref_rows):
        pred_key = pred_key[center_nodes]
    return pred_key, ref_key


def _finite_row_mask(pred_key: torch.Tensor, ref_key: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(pred_key) & torch.isfinite(ref_key)
    if finite.ndim == 0:
        return finite.reshape(1)
    if finite.ndim > 1:
        finite = finite.reshape(finite.shape[0], -1).all(dim=1)
    return finite


def resolve_node_type_indices(params: dict, owner: str) -> Optional[torch.Tensor]:
    node_type_indices = params.get("node_type_indices", None)
    node_type_names = params.get("node_type_names", None)
    type_names = params.get("type_names", None)

    if node_type_indices is not None and node_type_names is not None:
        raise ValueError("Specify only one of `node_type_indices` or `node_type_names`.")

    if node_type_indices is not None:
        if isinstance(node_type_indices, (int, str)):
            node_type_indices = [node_type_indices]
        return torch.tensor([int(v) for v in node_type_indices], dtype=torch.long)

    if node_type_names is None:
        return None

    if type_names is None:
        raise ValueError(
            f"`node_type_names` was provided for a {owner}, but no `type_names` list was found in the {owner} parameters. "
            "Add `type_names` next to the entry or use `node_type_indices` instead."
        )
    if isinstance(node_type_names, str):
        node_type_names = [node_type_names]
    if isinstance(type_names, str):
        type_names = [type_names]
    type_name_to_idx = {str(name): idx for idx, name in enumerate(type_names)}
    missing = [name for name in node_type_names if str(name) not in type_name_to_idx]
    if missing:
        raise ValueError(
            f"Unknown node type names in {owner} filter: {missing}. Available type names: {list(type_name_to_idx.keys())}"
        )
    return torch.tensor([type_name_to_idx[str(name)] for name in node_type_names], dtype=torch.long)


def prepare_target(
    *,
    pred: dict,
    ref: dict,
    key: str,
    pred_key_name: str,
    pred_key: torch.Tensor,
    ref_key: torch.Tensor,
    node_type_indices: Optional[torch.Tensor] = None,
    node_mask_field: Optional[str] = None,
    ignore_nan: bool = False,
    denormalize: bool = False,
    normalization_fields: Optional[Dict[str, Dict]] = None,
) -> PreparedTarget:
    """Build and apply one row mask for a loss/metric target."""
    ref = {} if ref is None else ref
    node_data = pred if AtomicDataDict.EDGE_INDEX_KEY in pred else ref
    center_nodes = _center_nodes(node_data)

    pred_key, ref_key = _align_pred_ref_rows(pred_key, ref_key, key, center_nodes)
    if ref_key.shape != pred_key.shape:
        try:
            ref_key = ref_key.reshape(pred_key.shape)
        except Exception:
            pass

    row_count = _row_count(pred_key)
    if row_count is None:
        prepared_pred = dict(pred)
        prepared_ref = dict(ref)
        prepared_pred[pred_key_name] = pred_key
        prepared_ref[key] = ref_key
        return PreparedTarget(prepared_pred, prepared_ref, pred_key, ref_key)

    masks = []
    node_types = None
    if key in _NODE_FIELDS and (AtomicDataDict.NODE_TYPE_KEY in pred or AtomicDataDict.NODE_TYPE_KEY in ref):
        node_type_source = pred if AtomicDataDict.NODE_TYPE_KEY in pred else ref
        node_types = _align_rows(
            node_type_source[AtomicDataDict.NODE_TYPE_KEY].squeeze(-1),
            row_count,
            center_nodes,
        )

    if node_type_indices is not None:
        if node_types is None:
            raise ValueError(
                f"Cannot apply node type filter for '{key}': node_types could not be aligned to {row_count} rows."
            )
        masks.append(torch.isin(node_types, node_type_indices.to(node_types.device)))

    if node_mask_field is not None:
        mask_source = pred if node_mask_field in pred else ref
        if node_mask_field in mask_source:
            node_mask = mask_source[node_mask_field]
            if not torch.is_tensor(node_mask):
                node_mask = torch.as_tensor(node_mask, device=pred_key.device)
            node_mask = _align_rows(node_mask.to(device=pred_key.device, dtype=torch.bool).squeeze(-1), row_count, center_nodes)
            if node_mask is None:
                raise ValueError(
                    f"Cannot apply node mask field '{node_mask_field}' for '{key}': mask could not be aligned to {row_count} rows."
                )
            masks.append(node_mask)

    if ignore_nan:
        masks.append(_finite_row_mask(pred_key, ref_key))

    final_mask = None
    if masks:
        final_mask = masks[0].to(device=pred_key.device, dtype=torch.bool)
        for mask in masks[1:]:
            final_mask = final_mask & mask.to(device=pred_key.device, dtype=torch.bool)
        pred_key = pred_key[final_mask]
        ref_key = ref_key[final_mask]
        if node_types is not None:
            node_types = node_types.to(device=final_mask.device)[final_mask]

    prepared_pred = dict(pred)
    prepared_ref = dict(ref)
    prepared_pred[pred_key_name] = pred_key
    prepared_ref[key] = ref_key
    if node_types is not None:
        node_types = node_types.to(device=pred_key.device)
        prepared_pred[AtomicDataDict.NODE_TYPE_KEY] = node_types.reshape(-1, 1)
        prepared_ref[AtomicDataDict.NODE_TYPE_KEY] = node_types.reshape(-1, 1)

    if denormalize:
        normalization_fields = normalization_fields or {}
        spec = normalization_fields.get(key, {})
        pred_key = denormalize_tensor(pred_key.clone(), prepared_ref, key, spec)
        ref_key = denormalize_tensor(ref_key.clone(), prepared_ref, key, spec)
        prepared_pred[pred_key_name] = pred_key
        prepared_ref[key] = ref_key

    return PreparedTarget(prepared_pred, prepared_ref, pred_key, ref_key, node_types, final_mask)


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
        self.node_mask_field = self.params.pop("node_mask_field", self.params.pop("node_mask_key", None))
        self.node_type_indices = self._resolve_node_type_indices()

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

    def _resolve_node_type_indices(self) -> Optional[torch.Tensor]:
        out = resolve_node_type_indices(self.params, "loss")
        self.params.pop("node_type_indices", None)
        self.params.pop("node_type_names", None)
        self.params.pop("type_names", None)
        return out
    
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

    def _resolve_node_mask(self, pred: dict, ref: dict):
        if self.node_mask_field is None:
            return None

        mask_source = pred if self.node_mask_field in pred else ref
        if self.node_mask_field not in mask_source:
            return None

        node_mask = mask_source[self.node_mask_field]
        if not torch.is_tensor(node_mask):
            node_mask = torch.as_tensor(node_mask)
        node_mask = node_mask.to(dtype=torch.bool)
        if node_mask.ndim > 1:
            node_mask = node_mask.squeeze(-1)
        return node_mask

    def _apply_node_filter(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
        pred: dict,
        ref: dict = None,
        key: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compatibility wrapper around the shared target preparation helper."""
        if key is None:
            if isinstance(ref, str):
                key = ref
                ref = pred
                pred = {}
            else:
                raise TypeError(
                    "_apply_node_filter expected either (pred_key, ref_key, pred, ref, key) "
                    "or legacy (pred_key, ref_key, ref, key) arguments."
                )

        prepared = prepare_target(
            pred=pred,
            ref=ref,
            key=key,
            pred_key_name=key,
            pred_key=pred_key,
            ref_key=ref_key,
            node_type_indices=self.node_type_indices,
            node_mask_field=self.node_mask_field,
            ignore_nan=False,
            denormalize=False,
        )
        return prepared.pred_key, prepared.ref_key

    def _calculate_loss(self, pred_key: torch.Tensor, ref_key: torch.Tensor, mean: bool) -> torch.Tensor:
        """Computes the loss, handling NaNs and applying supervision weights."""
        if pred_key.numel() == 0:
            return graph_zero_like(pred_key) if mean else pred_key
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

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Dict[str, Dict] = None,
        **kwargs,
    ):
        pred_key_name = self._get_pred_key_name(key)
        pred_key = pred.get(pred_key_name)
        assert isinstance(pred_key, torch.Tensor), f"Prediction for '{pred_key_name}' not a tensor."
        ref_key = ref.get(key)
        assert isinstance(ref_key, torch.Tensor), f"Reference for '{key}' not a tensor."

        self._initialize_supervision_weights(pred_key.device, pred_key.dtype)
        ref_key = self._handle_supervision_shapes(pred_key, ref_key, pred_key_name, key)

        if not kwargs.get("skip_target_filter", False):
            prepared = prepare_target(
                pred=pred,
                ref=ref,
                key=key,
                pred_key_name=pred_key_name,
                pred_key=pred_key,
                ref_key=ref_key,
                node_type_indices=self.node_type_indices,
                node_mask_field=self.node_mask_field,
                ignore_nan=self.ignore_nan,
                denormalize=not mean,
                normalization_fields=normalization_fields,
            )
            pred_key, ref_key = prepared.pred_key, prepared.ref_key
        elif not mean:
            normalization_fields = normalization_fields or {}
            spec = normalization_fields.get(key, {})
            pred_key = denormalize_tensor(pred_key.clone(), ref, key, spec)
            ref_key = denormalize_tensor(ref_key.clone(), ref, key, spec)

        return self._calculate_loss(pred_key, ref_key, mean)

    def _prepare_tensors(
        self,
        pred: dict,
        ref: dict,
        pred_key_name: str,
        ref_key_name: str,
        mean: bool,
        normalization_fields: Dict[str, Dict] = None,
    ):
        pred_key = pred.get(pred_key_name)
        assert isinstance(pred_key, torch.Tensor), f"Prediction for '{pred_key_name}' not a tensor."
        ref_key = ref.get(ref_key_name)
        assert isinstance(ref_key, torch.Tensor), f"Reference for '{ref_key_name}' not a tensor."

        # De-standardization for metrics (when mean=False)
        if not mean:
            normalization_fields = normalization_fields or {}

            spec = normalization_fields.get(ref_key_name, {})
            pred_key = denormalize_tensor(pred_key.clone(), ref, ref_key_name, spec)
            ref_key = denormalize_tensor(ref_key.clone(), ref, ref_key_name, spec)

        return pred_key, ref_key

    def __str__(self):
        return self.func_name


class LogCoshLoss(LossWrapper):
    """
    Numerically stable Log-Cosh regression loss.
    Behaves like MSE for small errors and like L1 for large errors.
    """
    def __init__(self, beta: float = 1.0, **params):
        if beta <= 0:
            raise ValueError("`beta` must be > 0 for LogCoshLoss.")
        self.beta = float(beta)
        super().__init__(func_name="L1Loss", params=params)
        self.func_name = "LogCoshLoss"

    def _calculate_loss(self, pred_key: torch.Tensor, ref_key: torch.Tensor, mean: bool) -> torch.Tensor:
        if pred_key.numel() == 0:
            return graph_zero_like(pred_key) if mean else pred_key
        diff = (pred_key - ref_key) / self.beta
        abs_diff = torch.abs(diff)
        log_cosh = self.beta * self.beta * (
            abs_diff + F.softplus(-2.0 * abs_diff) - math.log(2.0)
        )

        if self.ignore_nan:
            not_nan_mask = torch.isfinite(pred_key) & torch.isfinite(ref_key)
            loss = torch.where(not_nan_mask, log_cosh, torch.zeros_like(log_cosh))

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
            loss = loss.clone()
            loss[~not_nan_mask] = torch.nan
            return loss

        if self.supervision_output_dim is not None:
            log_cosh = log_cosh * self._supervision_weights_tensor
            log_cosh = log_cosh.sum(dim=-1)
        return log_cosh.mean() if mean else log_cosh

    def __str__(self):
        return "LogCoshLoss"


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
