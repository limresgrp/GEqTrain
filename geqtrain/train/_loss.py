# geqtrain/train/_loss.py

import torch
from geqtrain.utils import instantiate_from_cls_name
from geqtrain.data import AtomicDataDict, _NODE_FIELDS

class LossWrapper:
    """
    A wrapper for standard torch.nn loss functions that adds capabilities like
    ignoring NaNs and filtering to include only center nodes of edges.
    """
    def __init__(self, func_name: str, params: dict = {}):
        self.func_name = func_name
        self.params = params
        self.ignore_nan = params.get("ignore_nan", False)
        self.node_level_filter = params.get("node_level_filter", "auto") # node filtering mode: 'auto', True, or False

        torch_params = {
            k: v for k, v in params.items() if k not in ["ignore_nan", "node_level_filter"]
        }

        self.func, _ = instantiate_from_cls_name(
            torch.nn, class_name=func_name, prefix="",
            positional_args=dict(reduction="none"), optional_args=torch_params, all_args={},
        )

    def __call__(self, pred: dict, ref: dict, key: str, mean: bool = True, **kwargs):
        pred_key, ref_key = self._prepare_tensors(pred, ref, key)

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

            if mean:
                return loss.sum() / not_nan_mask.sum().clamp(min=1)
            
            loss[~not_nan_mask] = torch.nan
            return loss
        else:
            loss = self.func(pred_key, ref_key)
            return loss.mean() if mean else loss

    def _prepare_tensors(self, pred: dict, ref: dict, key: str):
        # ... (this method remains the same)
        pred_key = pred.get(key)
        assert isinstance(pred_key, torch.Tensor), f"Prediction for '{key}' not a tensor."
        ref_key = ref.get(key)
        assert isinstance(ref_key, torch.Tensor), f"Reference for '{key}' not a tensor."
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
        logits = pred[key].detach().squeeze(-1)
        target = ref[key].detach().squeeze(-1)

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
        if logits.numel() > 0: # Only update if there is valid data
            self.metric.update(logits, target.int())
    
    def compute(self):
        return self.metric.compute().clone()
        
    def reset(self):
        self.metric.reset()

    def __str__(self):
        return "BinaryAUROC"