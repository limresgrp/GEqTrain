# components/setup.py
from typing import Dict, List
import torch
import numpy as np
import logging
import pytorch_warmup as warmup

from geqtrain.utils import instantiate, instantiate_from_cls_name
from geqtrain.data import _NODE_FIELDS, _GRAPH_FIELDS, _EDGE_FIELDS
from geqtrain.train._key import VALIDATION, TRAIN
from geqtrain.train.loss import Loss, LossStat
from geqtrain.train.metrics import Metrics
from geqtrain.train.components.early_stopping import EarlyStopping
from torch_ema import ExponentialMovingAverage

def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

def parse_idcs_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return np.array([int(line.strip()) for line in lines if line.strip()], dtype=int)

def get_output_keys(loss_func: Loss):
    output_keys, per_node_outputs_keys = [], []
    if loss_func is not None:
        for key in loss_func.keys:
            key_clean = loss_func.remove_suffix(key)
            if key_clean in _NODE_FIELDS.union(_GRAPH_FIELDS).union(_EDGE_FIELDS):
                output_keys.append(key_clean)
            if key_clean in _NODE_FIELDS:
                per_node_outputs_keys.append(key_clean)
    return output_keys, per_node_outputs_keys

def setup_loss(config):
    loss, _ = instantiate(
        builder=Loss,
        prefix="loss",
        positional_args=dict(components=config.get('loss_coeffs')),
        all_args=config,
    )
    return loss

def setup_metrics(config):
    metrics, _ = instantiate(
        builder=Metrics,
        prefix="metrics",
        positional_args=dict(components=config.get('metrics_components')),
        all_args=config,
    )
    return metrics

def setup_optimizer(model, config):
    """
    Sets up the optimizer, creating parameter groups with custom hyperparameters
    based on tags assigned to model parameters.
    """
    # Get all params that require grad
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

    # This dict maps tags to custom optimizer keyword arguments
    param_groups_dict = {
        'dampen': {'lr': config.get('learning_rate', 1e-3) * 1.e-2},
        'nowd':   {'weight_decay': 0.0},
        '_wd':    {'weight_decay': config.get('head_wds', 0.0)},
    }
    if 'fine_tune_lr' in config:
        param_groups_dict.update({'tune': {'lr': config.get('fine_tune_lr')}})

    # Helper to create a new group template for a parameter
    def merge_groups(param, tags: list):
        merged_kwargs = {}
        for tag in tags:
            if tag in param_groups_dict:
                merged_kwargs.update(param_groups_dict[tag])
        return {'params': [param], **merged_kwargs}

    # Helper to combine parameters into the minimum number of groups to optimize performance
    def merge_or_create_group(optim_groups: List[Dict], group_to_add: Dict):
        # Check if a group with the exact same hyperparameters already exists
        for existing_group in optim_groups:
            existing_keys = set(existing_group.keys()) - {'params'}
            new_keys = set(group_to_add.keys()) - {'params'}
            if existing_keys == new_keys:
                if all(existing_group[k] == group_to_add[k] for k in existing_keys):
                    # If so, append the parameter to the existing group
                    existing_group['params'].extend(group_to_add['params'])
                    return
        # Otherwise, create a new group
        optim_groups.append(group_to_add)

    # Main logic to build the final list of parameter groups
    optim_groups = []
    for param in param_dict.values():
        tags = []
        # Add any user-defined tags from the model parameter
        if hasattr(param, 'tags'):
            tags.extend(param.tags)
        # Automatically add 'nowd' (no weight decay) for bias and 1D parameters
        if param.dim() < 2:
            tags.append('nowd')
        
        # Create a group for this parameter
        group = merge_groups(param, list(set(tags)))
        # Merge it into the final list
        merge_or_create_group(optim_groups, group)

    # Instantiate the optimizer with the generated parameter groups
    optim, _ = instantiate_from_cls_name(
        module=torch.optim,
        class_name=config.get('optimizer_name', 'Adam'),
        prefix="optimizer",
        positional_args=dict(params=optim_groups, lr=config.get('learning_rate', 1e-3)),
        all_args=config,
    )
    
    # Log the parameter groups for debugging and verification
    logging.info(f"Optimizer {type(optim).__name__} initialized with {len(optim.param_groups)} parameter groups:")
    for i, group in enumerate(optim.param_groups):
        group_info = {k: v for k, v in group.items() if k != 'params'}
        num_params = sum(p.numel() for p in group['params'])
        logging.info(f"  Group {i}: {group_info}, num_params={num_params}")

    return optim

def setup_scheduler(optimizer, config, steps_per_epoch):
    scheduler_name = config.get('lr_scheduler_name', 'none')
    if scheduler_name == 'none':
        return None, None

    # Handle Warmup
    warmup_scheduler = None
    warmup_steps = 0
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * steps_per_epoch
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_steps)

    # Add specific logic for schedulers that need dynamic parameters
    if scheduler_name == "CosineAnnealingLR":
        total_number_of_steps = steps_per_epoch * config.get('max_epochs', 1000)
        # Account for warmup steps
        if warmup_epochs > 0:
            total_number_of_steps -= warmup_steps
        
        # Set T_max in the config so it's picked up by the instantiation function
        config['T_max'] = total_number_of_steps
        
        # Set a default for eta_min if not provided
        if 'eta_min' not in config:
            config['eta_min'] = config.get('learning_rate', 1e-3) * 1e-2
        
        logging.info(f"CosineAnnealingLR scheduler configured with T_max = {config['T_max']}")

    # Main Scheduler Instantiation
    lr_scheduler, _ = instantiate_from_cls_name(
        module=torch.optim.lr_scheduler,
        class_name=scheduler_name,
        prefix="lr_scheduler",
        positional_args=dict(optimizer=optimizer),
        all_args=config,
    )
    logging.info(f"Using scheduler: {scheduler_name}")
    return lr_scheduler, warmup_scheduler

def setup_early_stopping(config):
    """
    Initialize early stopping conditions, gathering all `early_stopping_*` arguments
    from the config, including the new `metric_criteria`.
    """
    _, kwargs = instantiate(
        EarlyStopping,
        prefix="early_stopping",
        optional_args=config.get('early_stopping_kwargs', {}),
        all_args=config,
        return_args_only=True,
    )

    # The `instantiate` helper will find `early_stopping_patiences`, 
    # `early_stopping_criteria`, etc., and put them in kwargs.
    
    # Rename 'metric_criteria' to 'criteria' for the constructor
    if 'metric_criteria' in kwargs:
        kwargs['criteria'] = kwargs.pop('metric_criteria')

    n_args = 0
    # Prepend "validation_" to metric keys if they don't have a prefix
    for arg_name, arg_dict in kwargs.items():
        if isinstance(arg_dict, dict):
            new_dict = {}
            for k, v in arg_dict.items():
                k_lower = k.lower()
                if (
                    k_lower.startswith(VALIDATION)
                    or k_lower.startswith(TRAIN)
                    or k_lower in ["lr", "wall", "cumulative_wall"]
                ):
                    new_dict[k] = v
                else:
                    new_dict[f"{VALIDATION}_{k}"] = v
            kwargs[arg_name] = new_dict
            n_args += len(new_dict)
            
    return EarlyStopping(**kwargs) if n_args > 0 else None

def setup_ema(model, config):
    if not config.get('use_ema', False):
        return None
    logging.info("Using Exponential Moving Average for model parameters")
    return ExponentialMovingAverage(model.parameters(), decay=0.999)