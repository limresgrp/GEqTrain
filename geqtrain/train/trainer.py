""" Adapted from https://github.com/mir-group/nequip
"""
from collections import defaultdict
from e3nn import o3

import inspect
import logging
import wandb
import contextlib
from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Callable, Optional, Union, Tuple, List, Dict
from pathlib import Path
import torch.distributed as dist
import numpy as np
import torch
import re
from torch.utils.data import DistributedSampler, Sampler
import math
from geqtrain.data import (
    DataLoader,
    AtomicData,
    AtomicDataDict,
    InMemoryConcatDataset,
    LazyLoadingConcatDataset,
    _NODE_FIELDS,
    _GRAPH_FIELDS,
    _EDGE_FIELDS,
)
from geqtrain.utils import (
    Output,
    Config,
    instantiate_from_cls_name,
    instantiate,
    save_file,
    load_file,
    load_callable,
    atomic_write,
    finish_all_writes,
    atomic_write_group,
    gradfilter_ema,
)

from geqtrain.model import model_from_config
from geqtrain.train.utils import evaluate_end_chunking_condition
from geqtrain.train.sampler import EnsembleSampler, EnsembleDistributedSampler
from geqtrain.train import sync_tensor_across_GPUs, sync_dict_of_tensors_across_GPUs

from .loss import Loss, LossStat
from .metrics import Metrics
from ._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION
from .early_stopping import EarlyStopping

from geqtrain.train.grad_clipping_utils import Queue, gradient_clipping
from geqtrain.nn.mace.irreps_tools import reshape_irreps


def get_latest_lr(optimizer, model, param_name: str) -> float:
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param_name == [name for name, p in model.named_parameters() if p is param][0]:
                return param_group['lr']
    raise ValueError(f"Parameter {param_name} not found in optimizer.")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_output_keys(loss_func: Loss):
    '''
    returns fields/keys that have to be predicted (i.e. for which we need to compute a loss)
    can take as input a metrics object (for an example see evaluate.py)
    '''
    output_keys, per_node_outputs_keys = [], []
    if loss_func is not None:
        for key in loss_func.keys:
            key_clean = loss_func.remove_suffix(key)
            if key_clean in _NODE_FIELDS.union(_GRAPH_FIELDS).union(_EDGE_FIELDS):
                output_keys.append(key_clean)
            if key_clean in _NODE_FIELDS:
                per_node_outputs_keys.append(key_clean)
    return output_keys, per_node_outputs_keys

def run_inference(
    model,
    data,
    device,
    already_computed_nodes=None,
    output_keys: List[str] = [],
    per_node_outputs_keys: List[str] = [],
    cm=contextlib.nullcontext(),
    mixed_precision: bool = False,
    skip_chunking: bool = False,
    batch_max_atoms: int = 1000,
    ignore_chunk_keys: List[str] = [],
    dropout_edges: float = 0.,
    requires_grad: bool = False,
    is_ddp: bool = False,
    **kwargs,
):
    #! IMPO keep torch.bfloat16 for AMP: https://discuss.pytorch.org/t/why-bf16-do-not-need-loss-scaling/176596
    precision = torch.autocast(device_type='cuda' if torch.cuda.is_available(
    ) else 'cpu', dtype=torch.bfloat16) if mixed_precision else contextlib.nullcontext()
    # AtomicDataDict is the dstruct that is taken as input from each forward
    batch = AtomicData.to_AtomicDataDict(data.to(device))

    batch_index = batch[AtomicDataDict.EDGE_INDEX_KEY]
    num_batch_center_nodes = len(batch_index[0].unique())

    if skip_chunking:
        input_data = {
            k: v.requires_grad_() if requires_grad and torch.is_floating_point(v) else v
            for k, v in batch.items()
            # if k not in output_keys
        }
        ref_data = batch
        batch_center_nodes = batch_index[0].unique()
    else:
        input_data, ref_data, batch_center_nodes = prepare_chunked_input_data(
            already_computed_nodes=already_computed_nodes,
            batch=batch,
            data=data,
            output_keys=output_keys,
            per_node_outputs_keys=per_node_outputs_keys,
            batch_max_atoms=batch_max_atoms,
            ignore_chunk_keys=ignore_chunk_keys,
            device=device,
        )

    if hasattr(data, "__slices__"):
        for slices_key, slices in data.__slices__.items():
            val = torch.tensor(slices, dtype=torch.long, device=device)
            input_data[f"{slices_key}_slices"] = val
            ref_data[f"{slices_key}_slices"] = val

    if dropout_edges > 0:
        apply_dropout_edges(dropout_edges, input_data)

    with cm, precision:
        out = model(input_data)
        del input_data

    # if a module of model has ref_data_keys as attr
    # then take the string associated to that field and
    # write it into ref_data as {str}+_target
    try: # TODO check this and make it compatible with all models
        for (name, module) in (model.module if is_ddp else model):
            if hasattr(module, 'ref_data_keys'):
                for k in module.ref_data_keys:
                    target = out[k]
                    key_clean = k.replace("_target", "")
                    ref_data[key_clean] = target
    except:
        pass

    return out, ref_data, batch_center_nodes, num_batch_center_nodes

def apply_dropout_edges(dropout_edges, input_data):
    edge_index = input_data[AtomicDataDict.EDGE_INDEX_KEY]
    num_edges = edge_index.size(1)
    num_dropout_edges = int(dropout_edges * num_edges)

        # Randomly select edges to drop
    drop_edges = torch.randperm(num_edges, device=edge_index.device)[:num_dropout_edges]
    keep_edges = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
    keep_edges[drop_edges] = False

        # Ensure at least some edges per node center
    node_centers = edge_index[0].unique()
    remaining_node_centers = edge_index[0, keep_edges].unique()
    combined = torch.cat((node_centers, remaining_node_centers))
    uniques, counts = combined.unique(return_counts=True)
    dropped_out_node_centers = uniques[counts == 1]
    for node in dropped_out_node_centers:
        node_edges = (edge_index[0] == node).nonzero(as_tuple=True)[0]
        keep_edges[node_edges[torch.randint(len(node_edges), (max(1, int((1-dropout_edges)*len(node_edges))),))]] = True

    input_data[AtomicDataDict.EDGE_INDEX_KEY] = edge_index[:, keep_edges]
    if AtomicDataDict.EDGE_CELL_SHIFT_KEY in input_data:
        input_data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = input_data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][keep_edges]

def prepare_chunked_input_data(
    already_computed_nodes: Optional[torch.Tensor],
    batch: AtomicDataDict.Type,
    data: AtomicDataDict.Type,
    output_keys: List[str] = [],
    per_node_outputs_keys: List[str] = [],
    batch_max_atoms: int = 1000,
    ignore_chunk_keys: List[str] = [],
    device="cpu"
):
    # === Limit maximum batch size to avoid CUDA Out of Memory === #

    chunk = already_computed_nodes is not None
    batch_chunk = deepcopy(batch)
    batch_chunk_index = batch_chunk[AtomicDataDict.EDGE_INDEX_KEY]
    edge_fields_dict = {
        edge_field: batch[edge_field]
        for edge_field in _EDGE_FIELDS
        if edge_field in batch
    }

    if chunk:
        batch_chunk_index = batch_chunk_index[:, ~torch.isin(batch_chunk_index[0], already_computed_nodes)]
    batch_chunk_center_node_idcs = batch_chunk_index[0].unique()
    if len(batch_chunk_center_node_idcs) == 0:
        return None, None, None

    # = Iteratively remove edges from batch_chunk = #
    # = ----------------------------------------- = #

    offset = 0
    while len(batch_chunk_index.unique()) > batch_max_atoms:

        def get_node_center_idcs(batch_chunk_index: torch.Tensor, batch_max_atoms: int, offset: int):
            unique_set = set()

            for i, num in enumerate(batch_chunk_index[1]):
                unique_set.add(num.item())

                if len(unique_set) >= batch_max_atoms:
                    node_center_idcs = batch_chunk_index[0, :i+1].unique()
                    if len(node_center_idcs) == 1:
                        num_nodes = torch.isin(batch_chunk_index[0], node_center_idcs).sum()
                        if num_nodes > batch_max_atoms:
                            raise ValueError(
                                f"At least one node in the graph has more neighbors ({num_nodes}) "
                                f"than the maximum allowed number of atoms in a batch ({batch_max_atoms}). "
                                "Please increase the value of 'batch_max_atoms' in the config file."
                            )
                        return node_center_idcs
                    return node_center_idcs[:-offset]
            return batch_chunk_index[0].unique()

        def get_edge_filter(batch_chunk_index: torch.Tensor, offset: int):
            node_center_idcs = get_node_center_idcs(batch_chunk_index, batch_max_atoms, offset)
            edge_filter = torch.isin(batch_chunk_index[0], node_center_idcs)
            return edge_filter

        chunk = True
        offset += 1
        fltr = get_edge_filter(batch_chunk_index, offset)
        batch_chunk_index = batch_chunk_index[:, fltr]
        for k, v in edge_fields_dict.items():
            edge_fields_dict[k] = v[fltr]

    # = ----------------------------------------- = #

    if chunk:
        batch_chunk[AtomicDataDict.EDGE_INDEX_KEY] = batch_chunk_index
        batch_chunk[AtomicDataDict.BATCH_KEY] = data[AtomicDataDict.BATCH_KEY][batch_chunk_index.unique()]
        for k, v in edge_fields_dict.items():
            batch[k] = v
        if AtomicDataDict.EDGE_CELL_SHIFT_KEY in batch:
            batch[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = batch[AtomicDataDict.EDGE_CELL_SHIFT_KEY][batch_chunk_index.unique()]
        for per_node_output_key in per_node_outputs_keys:
            chunk_per_node_outputs_value = batch[per_node_output_key].clone()
            mask = torch.ones_like(chunk_per_node_outputs_value, dtype=torch.bool)
            mask[batch_chunk_index[0].unique()] = False
            chunk_per_node_outputs_value[mask] = torch.nan
            batch_chunk[per_node_output_key] = chunk_per_node_outputs_value

    # === ---------------------------------------------------- === #
    # === ---------------------------------------------------- === #

    batch_chunk["ptr"] = torch.nn.functional.pad(torch.bincount(batch_chunk.get(
        AtomicDataDict.BATCH_KEY)).flip(dims=[0]), (0, 1), mode='constant').flip(dims=[0])

    edge_index = batch_chunk[AtomicDataDict.EDGE_INDEX_KEY]
    node_index = edge_index.unique(sorted=True)

    for key in batch_chunk.keys():
        if key in [
            AtomicDataDict.BATCH_KEY,
            AtomicDataDict.EDGE_INDEX_KEY,
        ] + ignore_chunk_keys:
            continue
        dim = np.argwhere(np.array(batch_chunk[key].size()) == len(
            data[AtomicDataDict.BATCH_KEY])).flatten()
        if len(dim) == 1:
            if dim[0] == 0:
                batch_chunk[key] = batch_chunk[key][node_index]
            elif dim[0] == 1:
                batch_chunk[key] = batch_chunk[key][:, node_index]
            elif dim[0] == 2:
                batch_chunk[key] = batch_chunk[key][:, :, node_index]
            else:
                raise Exception('Dimension not implemented')

    last_idx = -1
    updated_edge_index = edge_index.clone()
    for idx in node_index:
        if idx > last_idx + 1:
            updated_edge_index[edge_index >= idx] -= idx - last_idx - 1
        last_idx = idx
    batch_chunk[AtomicDataDict.EDGE_INDEX_KEY] = updated_edge_index
    batch_chunk_center_nodes = edge_index[0].unique()

    del edge_index
    del node_index

    input_data = {
        k: v.to(device)
        for k, v in batch_chunk.items()
        if k not in output_keys
    }

    return input_data, batch_chunk, batch_chunk_center_nodes

def _init(loss_func, dataset, model):
    init_loss = getattr(loss_func, "init_loss", None)
    if callable(init_loss):
        num_data = 0
        for ds in dataset:
            num_data += len(ds[AtomicDataDict.POSITIONS_KEY])
        init_loss(model, num_data)


class Trainer:
    """Customizable class used to train a model to minimise a set of loss functions.

    Args:
        model: PyTorch model

        seed (int): random seed number
        dataset_seed (int): random seed for dataset operations

        loss_coeffs (dict): dictionary to store coefficient and loss functions

        max_epochs (int): maximum number of epochs

        learning_rate (float): initial learning rate
        lr_scheduler_name (str): scheduler name
        lr_scheduler_kwargs (dict): parameters to initialize the scheduler

        optimizer_name (str): name for optimizer
        optimizer_kwargs (dict): parameters to initialize the optimizer

        batch_size (int): size of each batch
        validation_batch_size (int): batch size for evaluating the model for validation
        shuffle (bool): parameters for dataloader
        n_train (int): # of frames for training
        n_val (int): # of frames for validation
        exclude_keys (list):  fields from dataset to ignore.
        dataloader_num_workers (int): `num_workers` for the `DataLoader`s
        train_idcs (optional, list):  list of frames to use for training
        val_idcs (list):  list of frames to use for validation
        train_val_split (str):  "random" or "sequential"

        init_callbacks (list): list of callback function at the begining of the training
        end_of_epoch_callbacks (list): list of callback functions at the end of each epoch
        end_of_batch_callbacks (list): list of callback functions at the end of each batch
        end_of_train_callbacks (list): list of callback functions between traing/validation
        final_callbacks (list): list of callback functions at the end of the training

        log_batch_freq (int): frequency to log at the end of a batch
        log_epoch_freq (int): frequency to save at the end of an epoch
        save_checkpoint_freq (int): frequency to save the intermediate checkpoint. no saving when the value is not positive.

        verbose (str): verbosity level, i.e. "INFO", "WARNING", "DEBUG". case insensitive

    Additional Attributes:

        init_keys (list): list of parameters needed to reconstruct this instance
        dl_train (DataLoader): training data
        dl_val (DataLoader): test data
        iepoch (int): # of epoches ran
        stop_arg (str): reason why the training stops
        batch_mae (float): the mae of the latest batch
        mae_dict (dict): all loss, mae of the latest validation
        best_metrics (float): current best validation mae
        best_epoch (float): current best epoch
        best_model_path (str): path to save the best model
        last_model_path (str): path to save the latest model
        trainer_save_path (str): path to save the trainer.
             Default is trainer.(date).pth at the current folder


    The pseudocode of the workflow and location of the callback functions

    ```
    init():
        initialize optimizer, schduler and loss function

    train():
       init model
       init_callbacks
       while (not stop):
            training batches
                end_of_batch_callbacks
            end_of_train_callbacks
            validation_batches
            end_of_epoch_callbacks
       final_callbacks
    ```
    """

    stop_keys = ["max_epochs", "early_stopping", "early_stopping_kwargs"]
    object_keys = ["lr_sched", "optim", "early_stopping_conds", "warmup_scheduler"]
    lr_scheduler_module = torch.optim.lr_scheduler
    optim_module = torch.optim

    def __init__(
        self,
        model_requires_grads: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_master: bool = True,
        seed: Optional[int] = None,
        dataset_seed: Optional[int] = None,
        loss_coeffs: Union[dict, str] = None,
        train_on_keys: Optional[List[str]] = None,
        type_names: Optional[List[str]] = None,
        metrics_components: Optional[Union[dict, str]] = None,
        metrics_key: str = f"{VALIDATION}_" + ABBREV.get(LOSS_KEY, LOSS_KEY),
        early_stopping: Optional[Callable] = None,
        early_stopping_kwargs: Optional[dict] = None,
        max_epochs: int = 10000,
        learning_rate: float = 1e-2,
        lr_scheduler_name: str = "none",
        lr_scheduler_kwargs: Optional[dict] = None,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        max_gradient_norm: float = float("inf"),
        exclude_keys: list = [],
        batch_size: int = 5,
        validation_batch_size: int = 5,
        shuffle: bool = True,
        n_train: Optional[Union[List[int], int]] = None,
        n_val: Optional[Union[List[int], int]] = None,
        dataloader_num_workers: int = 0,
        train_idcs: Optional[Union[List, List[List]]] = None,
        val_idcs: Optional[Union[List, List[List]]] = None,
        train_val_split: str = "random",  # ['random', 'sequential']
        skip_chunking: bool = False,
        batch_max_atoms: int = 1000,
        ignore_chunk_keys: List[str] = [],
        # Callbacks
        initial_callbacks: List = [],
        end_of_batch_callbacks: List = [],
        end_of_train_callbacks: List = [],
        end_of_valid_callbacks: List = [],
        end_of_epoch_callbacks: List = [],
        final_callbacks: List = [],

        log_batch_freq: int = 1,
        log_epoch_freq: int = 1,
        save_checkpoint_freq: int = -1,
        report_init_validation: bool = True,
        verbose="INFO",
        sanitize_gradients: bool = False,
        target_names: Optional[List] = None,
        mixed_precision: bool = False,
        hooks: Dict = {},
        use_grokfast: bool = False,
        use_ema: bool = False,
        debug: bool = False,
        dropout_edges: float = 0.,
        warmup_epochs: int | str = -1,
        head_wds: float = 0.0,
        accumulation_steps: int = 1, # default: 1 -> standard behavior of updating weights at each batch step
        metric_criteria:str='decreasing', # or 'increasing'
        **kwargs,
    ):
        # Initialize parameters and store them in self.kwargs
        # Deep copy the provided keyword arguments to avoid unintended modifications
        self.kwargs = deepcopy(kwargs)

        # Iterate over all initialization keys (parameters of the constructor)
        # and set them as attributes of the instance
        self._local_kwargs = {}
        for key in self.init_keys:
            setattr(self, key, locals()[key]) # Assign the value of the parameter to the instance
            self._local_kwargs[key] = locals()[key]  # Store the parameter in the kwargs dictionary
        self._initialize_parameters(kwargs)

        # Ordered list of initialization methods
        self.init_order = [
            self._init_base_attrs,
            self._init_randomness,
            self._init_output_paths,
            self._init_loss,
            self._init_grad_norm,
            self._init_metrics,
            self._init_callbacks,
            self._init_early_stopping,
        ]

        # Call initialization methods in order
        for init_method in self.init_order:
            init_method()

        self.logger.info(f"Torch device: {self.device}")

    def _initialize_parameters(self, kwargs):
        pass

    def _init_base_attrs(self):
        self.model = None
        self.cumulative_wall = 0

        # --- initialize training states
        self.best_metrics = float("inf") if self.metric_criteria == 'decreasing' else float('-inf')
        self.best_epoch = 0
        self.iepoch = -1 if self.report_init_validation else 0
        self.grads = None # as requested in https://github.com/ironjr/grokfast

        '''Gradient Accumulation: simulate larger BS when hardware memory is insufficient to process large batches.
        Instead of updating the model parameters after every batch, gradients are accumulated over multiple batches, and the model parameters are updated only after a specified number of steps.
        accumulation_steps: number of steps over which to accumulate gradients.
        Accumulation: In the batch_step method, gradients are accumulated over multiple batches. The optimizer's zero_grad method is called only at the start of the accumulation cycle.
        Optimization Step: After the specified number of accumulation steps, the gradients are used to update the model parameters, and the accumulation counter is reset.'''
        self.accumulation_counter = 0  # Counter for gradient accumulation

        if hasattr(self, 'rank'):
            assert self.device == self.rank

        self.torch_device = torch.device(self.device)

    def _init_randomness(self):
        """Initialize randomness for reproducibility."""
        if hasattr(self, "seed") and self.seed:
            set_seed(self.seed)
        if hasattr(self, "dataset_seed") and self.dataset_seed:
            self.dataset_rng = torch.Generator()
            self.dataset_rng.manual_seed(self.dataset_seed)

    def _init_output_paths(self):
        # --- get I/O handler
        self.output = Output.get_output(dict(**self._local_kwargs, **self.kwargs))

        """Initialize output paths for saving logs and models."""
        if not hasattr(self, "output"):
            raise ValueError("Output handler is required for initialization.")
        self.logfile           = self.output.open_logfile("log", propagate=True)
        self.epoch_log         = self.output.open_logfile("metrics_epoch.csv", propagate=False)
        self.init_epoch_log    = self.output.open_logfile("metrics_initialization.csv", propagate=False)

        # --- add filenames if not defined
        self.config_path       = self.output.generate_file("config.yaml")
        self.best_model_path   = self.output.generate_file("best_model.pth")
        self.last_model_path   = self.output.generate_file("last_model.pth")
        self.trainer_save_path = self.output.generate_file("trainer.pth")

        # logs for weights update and gradient
        if self.debug:
            self.log_updates   = self.output.open_logfile("log_updates", propagate=False)
            self.log_ratio     = self.output.open_logfile("log_ratio", propagate=False)

        self.batch_log = {
            TRAIN:      self.output.open_logfile(f"metrics_batch_{ABBREV[TRAIN]}.csv", propagate=False),
            VALIDATION: self.output.open_logfile(f"metrics_batch_{ABBREV[VALIDATION]}.csv", propagate=False),
        }

    def _init_loss(self):
        """Initialize loss functions."""
        self.loss, _ = instantiate(
            builder=Loss,
            prefix="loss",
            positional_args=dict(components=self.loss_coeffs),
            all_args=self.kwargs,
        )
        self.loss_stat = LossStat(self.loss)
        self.output_keys, self.per_node_outputs_keys = get_output_keys(self.loss)

        train_on_keys = self.loss.keys
        if (self.train_on_keys is not None) and (set(self.train_on_keys) != set(train_on_keys)):
            logging.info("Different training keys found.")
        self.train_on_keys = train_on_keys

    def _init_grad_norm(self):
        self.gradnorms = []
        self.gradnorms_clip = []
        self.gradnorms_queue = Queue()

    def _init_metrics(self):
        """Initialize metrics."""
        if not hasattr(self, "metrics_components"):
            self.metrics_components = []

        self.metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=self.metrics_components),
            all_args=self.kwargs,
        )

        self.metrics_metadata = {
            'type_names':   self.type_names,
            'target_names': self.target_names or ['target'],
        }

        if not (self.metrics_key.lower().startswith(VALIDATION) or self.metrics_key.lower().startswith(TRAIN)):
            raise RuntimeError(f"metrics_key should start with either {VALIDATION} or {TRAIN}")
        if self.report_init_validation and self.metrics_key.lower().startswith(TRAIN):
            raise RuntimeError(f"metrics_key may not start with {TRAIN} when report_init_validation=True")

    def _init_callbacks(self):
        """Initialize callbacks."""
        self._initial_callbacks      = [load_callable(callback) for callback in self.initial_callbacks]
        self._end_of_batch_callbacks = [load_callable(callback) for callback in self.end_of_batch_callbacks]
        self._end_of_train_callbacks = [load_callable(callback) for callback in self.end_of_train_callbacks]
        self._end_of_valid_callbacks = [load_callable(callback) for callback in self.end_of_valid_callbacks]
        self._end_of_epoch_callbacks = [load_callable(callback) for callback in self.end_of_epoch_callbacks]
        self._final_callbacks        = [load_callable(callback) for callback in self.final_callbacks]

    def _init_early_stopping(self):
        """Initialize early stopping conditions."""
        key_mapping, kwargs = instantiate(
            EarlyStopping,
            prefix="early_stopping",
            optional_args=self.early_stopping_kwargs,
            all_args=self.kwargs,
            return_args_only=True,
        )
        n_args = 0
        for key, item in kwargs.items():
            # prepend VALIDATION string if k is not with
            if isinstance(item, dict):
                new_dict = {}
                for k, v in item.items():
                    if (
                        k.lower().startswith(VALIDATION)
                        or k.lower().startswith(TRAIN)
                        or k.lower() in ["lr", "wall", "cumulative_wall"]
                    ):
                        new_dict[k] = item[k]
                    else:
                        new_dict[f"{VALIDATION}_{k}"] = item[k]
                kwargs[key] = new_dict
                n_args += len(new_dict)
        self.early_stopping_conds = EarlyStopping(**kwargs) if n_args > 0 else None

    def _init_num_of_optim_steps_per_epoch(self) -> None:
        '''
        Returns the number of optimizer steps performed per epoch by the current process.

        This accounts for the number of batches processed by this specific rank
        and gradient accumulation steps.
        '''
        if not hasattr(self, "dl_train") or self.dl_train is None:
            raise ValueError("Attribute self.dl_train is missing or None. Cannot infer number of steps per epoch.")

        if not isinstance(self.dl_train, DataLoader):
             raise TypeError("self.dl_train must be a PyTorch DataLoader instance.")

        # len(dataloader) correctly gives the number of batches yielded
        # by this DataLoader instance, whether using DistributedSampler or not.
        num_batches_per_process = len(self.dl_train)

        if num_batches_per_process == 0:
             self.logger.warning(f"DataLoader for the current process is empty (length is 0). "
                                 f"This might happen with DDP, small datasets, and large world sizes. "
                                 f"Returning 0 optimizer steps for this epoch.")
             return 0

        # Calculate the number of optimizer steps for this process
        self.num_optim_steps = math.ceil(num_batches_per_process / self.accumulation_steps)

        # --- Optional: Enhanced Logging ---
        is_distributed = False
        sampler_type = "N/A"
        world_size = 1
        rank = 0
        if hasattr(self.dl_train, 'sampler') and self.dl_train.sampler is not None:
            sampler_type = type(self.dl_train.sampler).__name__
            # Check for common distributed sampler types
            is_distributed = isinstance(self.dl_train.sampler, (DistributedSampler,)) # Add EnsembleDistributedSampler if needed
            if is_distributed:
                try:
                    # Attempt to get world size and rank if possible (requires torch.distributed)
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        world_size = dist.get_world_size()
                        rank = dist.get_rank()
                    else: # Fallback if not initialized or available
                        # Sometimes sampler stores these
                        if hasattr(self.dl_train.sampler, 'num_replicas'):
                            world_size = self.dl_train.sampler.num_replicas
                        if hasattr(self.dl_train.sampler, 'rank'):
                            rank = self.dl_train.sampler.rank
                except ImportError:
                    self.logger.warning("torch.distributed not available for detailed DDP logging.")
                except Exception as e:
                    self.logger.warning(f"Could not get DDP info (world_size/rank): {e}")


        self.logger.info(
            f"Calculating optim steps per epoch: "
            f"DataLoader length (batches for this process rank={rank}/{world_size}): {num_batches_per_process}, "
            f"Accumulation steps: {self.accumulation_steps}, "
            f"Sampler type: {sampler_type}, "
            f"Distributed: {is_distributed}"
        )
        self.logger.info(f"Number of optim steps per epoch (for this process): {self.num_optim_steps}")
        # --- End Optional Logging ---

    def _init_model_dependent_objects(self):
        '''
        Initializes the following components required for training:
        - Optimizer: Sets up the optimizer for training the model.
        - Scheduler: Configures the learning rate scheduler for dynamic adjustment of the learning rate.
        - Losses: Initialize losses (if necessary), passing information on dataset and model.
        - Exponential Moving Average (EMA): Configures the EMA for model parameters if enabled.
        - Model Irreps Validation: Ensures that the model's irreducible representations (irreps) are compatible with the loss function.

        Additionally, this method sets up Exponential Moving Average (EMA) for model parameters if enabled.
        Note: All methods here require the `self.model` attribute to be initialized.
        '''

        self._init_optimizer()
        self._init_scheduler()
        self._init_losses()
        self._init_ema()

        self._validate_model_irreps()

    def _init_optimizer(self):
        assert self.model is not None, "Model must not be None. Please ensure the model is properly initialized."

        # get all params that require grad
        param_dict = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        # if you assign one or more tags to a parameter (e.g. param.tags = ['dampen']),
        # the correspondent kwargs in 'param_groups_dict' will overwrite the default kwargs of the optimizer
        param_groups_dict = {
            'dampen':       {'lr': self.learning_rate * 1.e-2},
            'nowd':         {'weight_decay': 0.0},
            "_wd":          {'weight_decay': self.head_wds},
        }
        if 'fine_tune_lr' in self.kwargs:
            param_groups_dict.update({'tune':{'lr': self.kwargs['fine_tune_lr']}})

        def merge_groups(param, param_groups):
            # overrides default dict for optim
            merged_kwargs = {}
            for param_group in param_groups:
                merged_kwargs.update(param_groups_dict[param_group])
            return {'params': [param], **merged_kwargs}

        # Function to merge a parameter with an existing group or create a new one
        def merge_or_create_group(optim_groups: List[Dict], group: Dict):

            def merge_group(group, optim_group):
                if optim_group.keys() == group.keys():
                    if all([optim_group[key] == group[key] for key in optim_group.keys() if key != 'params']):
                        optim_group['params'].extend(
                            group['params'])  # Append params if found
                        return True
                return False

            # Try to find an existing group with the same keys
            for optim_group in optim_groups:
                if merge_group(group, optim_group):
                    return

            # If no group with the same keys is found, add the new group
            optim_groups.append(group)

        # gathering/parsing params to build optim groups
        optim_groups = []
        for p in param_dict.values():
            param_groups = []
            if getattr(p, 'tags', None) is not None:
                for tag in getattr(p, 'tags'):
                    param_groups.append(tag)
            if p.dim() < 2:
                param_groups.append('nowd')
            # here if tag=freeze then req grad to F

            group = merge_groups(p, param_groups)
            merge_or_create_group(optim_groups, group)

        # tag setted at model_from_config execution
        # parse all params with tag freeze and set thier req grad to F
        # check that head has req grad to T
        # in  - HeadlessGlobalNodeModel: load # {lr, freeze} <- apply tags here, needed to be set
        # another tag for trunk 1e-6

        self.optim, self.optimizer_kwargs = instantiate_from_cls_name(
            module=torch.optim,
            class_name=self.optimizer_name,
            prefix="optimizer",
            positional_args=dict(params=optim_groups, lr=self.learning_rate),
            all_args=self.kwargs,
            optional_args=self.optimizer_kwargs,
        )

    def _init_scheduler(self):
        assert (
            self.lr_scheduler_name
            in ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "CosineAnnealingLR", "none"]
        ) or (
            (len(self.end_of_epoch_callbacks) +
             len(self.end_of_batch_callbacks)) > 0
        ), f"{self.lr_scheduler_name} cannot be used unless callback functions are defined"

        self.lr_sched = None
        self.lr_scheduler_kwargs = {}
        if self.lr_scheduler_name != "none":
            # note: lr_scheduler_T_max is used for schedulers that require max num of steps
            # e.g. T_max in https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#cosineannealinglr
            self._init_warmup()

            if self.lr_scheduler_name == "CosineAnnealingLR":
                total_number_of_steps = self.num_optim_steps * self.max_epochs
                if self.warmup_epochs > 0:
                    total_number_of_steps -= self.warmup_steps
                self.kwargs['lr_scheduler_T_max'] = total_number_of_steps
                self.kwargs['eta_min'] = self.lr_scheduler_min_lr if hasattr(self, "lr_scheduler_min_lr") else self.learning_rate*.1

            self.lr_sched, self.lr_scheduler_kwargs = instantiate_from_cls_name(
                module=torch.optim.lr_scheduler,
                class_name=self.lr_scheduler_name,
                prefix="lr_scheduler",
                positional_args=dict(optimizer=self.optim),
                optional_args=self.lr_scheduler_kwargs,
                all_args=self.kwargs,
            )

    def _init_warmup(self):
        """Parse warmup period from configuration."""
        if not isinstance(self.warmup_epochs, int):
            match = re.match(r'^(\d+(?:\.\d+)?)%$', self.warmup_epochs.strip())
            if match:
                self.warmup_epochs = int((self.max_epochs / 100) * float(match.group(1)))
            else:
                raise ValueError(f"Invalid {match.string} format provided, it must be eg: '7.1%' in yaml, with ''")

        if self.warmup_epochs > 0:
            import pytorch_warmup as warmup
            self.warmup_steps = self.num_optim_steps * self.warmup_epochs
            self.warmup_scheduler = warmup.LinearWarmup(self.optim, self.warmup_steps) #! lrs updated inplace at this call: lr*=1/warmup_period

    def _init_losses(self):
        for loss_func in self.loss.funcs.values():
            _init(loss_func, self.dataset_train, self.model)
        for loss_func in self.metrics.funcs.values():
            _init(loss_func, self.dataset_train, self.model)

    def _init_ema(self):
        """Set up Exponential Moving Average (EMA) for model parameters."""
        self.ema = None
        if self.use_ema:
            self.logger.info("Using EMA for model parameters")
            from torch_ema import ExponentialMovingAverage  # External library for EMA: https://pypi.org/project/torch-ema/
            self.ema = ExponentialMovingAverage(
            self.model.parameters(),
            decay=0.999,  # Decay rate for EMA (adjustable)
            use_num_updates=True,  # Dynamically adjust decay based on updates
            )

    def _validate_model_irreps(self):
        """Validate the model's irreducible representations (irreps)"""
        if hasattr(self.model, "irreps_out"):
            for key in self.train_on_keys:
                if self.loss.remove_suffix(key) not in self.model.irreps_out:
                    raise RuntimeError(f"Loss function include fields {self.loss.remove_suffix(key)} that are not predicted by the model {self.model.irreps_out}")

    def log_data_points(self, dataset, prefix: str):
        loss_clean_keys = [self.loss.remove_suffix(key) for key in self.loss.keys]
        counts = {}
        for data in dataset:
            for loss_clean_key in loss_clean_keys:
                counts[loss_clean_key] = counts.get(loss_clean_key, 0) + torch.sum(~torch.isnan(data[loss_clean_key])).item()
        for k, v in counts.items():
            self.logger.info(f"{prefix} data points for field {k}: {v}")

    @property
    def is_ddp(self):
        '''returns T/F whether this trainer is being used in a ddp setting'''
        return hasattr(self, 'rank')

    @property
    def init_keys(self):
        '''
        return init_keys (list): list of parameters needed to reconstruct this instance
        '''
        return [
            key
            for key in list(inspect.signature(Trainer.__init__).parameters.keys())
            if key not in (["self", "kwargs", "model"] + Trainer.object_keys)
        ]

    @property
    def params(self):
        '''
        returns self.as_dict
        '''
        return self.as_dict(state_dict=False, training_progress=False, kwargs=False)

    @property
    def dataset_params(self):
        if isinstance(self.dataset_train, LazyLoadingConcatDataset):
            return self.dataset_train.config
        elif isinstance(self.dataset_train, InMemoryConcatDataset):
            return self.dataset_train.datasets[0].config
        raise ValueError(f'Dataset currently used is of type ({type(self.dataset_train)}), which is not supported')

    @property
    def logger(self):
        return logging.getLogger(self.logfile)

    @property
    def epoch_logger(self):
        return logging.getLogger(self.epoch_log)

    @property
    def init_epoch_logger(self):
        return logging.getLogger(self.init_epoch_log)

    @property
    def stop_cond(self):
        """kill the training early"""

        if not self.is_master:
            return

        if self.early_stopping_conds is not None and hasattr(self, "mae_dict") and self._is_warmup_period_over():
            early_stop, early_stop_args, debug_args = self.early_stopping_conds(self.mae_dict)
            if debug_args is not None:
                self.logger.debug(debug_args)
            if early_stop:
                self.stop_arg = early_stop_args
                return True

        if self.iepoch >= self.max_epochs:
            self.stop_arg = "max epochs"
            return True

        return False

    def update_kwargs(self, config):
        self.kwargs.update({key: value for key, value in config.items() if key not in self.init_keys})

    def as_dict(
        self,
        state_dict: bool = False,
        training_progress: bool = False,
        kwargs: bool = True,
    ):
        """convert instance to a dictionary
        Args:

        state_dict (bool): if True, the state_dicts of the optimizer and lr scheduler will be included
        """

        dictionary = {}

        for key in self.init_keys:
            dictionary[key] = getattr(self, key, None)

        if kwargs:
            dictionary.update(getattr(self, "kwargs", {}))

        if state_dict:
            dictionary["state_dict"] = {}
            for key in self.object_keys:
                item = getattr(self, key, None)
                if item is not None:
                    dictionary["state_dict"][key] = item.state_dict()
            dictionary["state_dict"]["rng_state"] = torch.get_rng_state()
            dictionary["state_dict"]["dataset_rng_state"] = self.dataset_rng.get_state()
            if torch.cuda.is_available():
                dictionary["state_dict"]["cuda_rng_state"] = torch.cuda.get_rng_state(device=self.torch_device)
            dictionary["state_dict"]["cumulative_wall"] = self.cumulative_wall

        if training_progress:
            dictionary["progress"] = {}
            for key in ["iepoch", "best_epoch"]:
                dictionary["progress"][key] = self.__dict__.get(key, -1)
            dictionary["progress"]["best_metrics"] = self.__dict__.get("best_metrics", float("inf"))
            dictionary["progress"]["stop_arg"] = self.__dict__.get("stop_arg", None)

            # TODO: these might not both be available, str defined, but no weights
            dictionary["progress"]["best_model_path"] = self.best_model_path
            dictionary["progress"]["last_model_path"] = self.last_model_path
            dictionary["progress"]["trainer_save_path"] = self.trainer_save_path
            if hasattr(self, "config_save_path"):
                dictionary["progress"]["config_save_path"] = self.config_save_path

        return dictionary

    def save_ema_model(self, path, blocking: bool = True):
        if self.use_ema:
            # If using EMA, store the EMA validation model
            # that gave us the good val metrics that made the model "best"
            # in the first place
            cm = self.ema.average_parameters()
        else:
            # otherwise, do nothing
            cm = contextlib.nullcontext()

        with cm:
            self.save_model(path, blocking=blocking)

    def save(self, filename: Optional[str] = None, format=None, blocking: bool = True):
        """save the file as filename

        Args:

        filename (str): name of the file
        format (str): format of the file. yaml and json format will not save the weights.
        """

        if not self.is_master:
            return

        if filename is None:
            filename = self.trainer_save_path

        logger = self.logger

        state_dict = (
            True
            if format == "torch"
            or filename.endswith(".pt")
            or filename.endswith(".pth")
            else False
        )

        filename = save_file(
            item=self.as_dict(state_dict=state_dict, training_progress=True),
            supported_formats=dict(torch=["pt", "pth"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
            blocking=blocking,
        )
        logger.debug(f"Saved trainer to {filename}")

        self.save_ema_model(self.last_model_path, blocking=blocking)
        logger.debug(f"Saved last model to to {self.last_model_path}")

        return filename

    @classmethod
    def from_file(
        cls, filename: str, format: Optional[str] = None, append: Optional[bool] = None
    ):
        """load a model from file

        Args:

        filename (str): name of the file
        append (bool): if True, append the old model files and append the same logfile
        """

        dictionary = load_file(
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
        )
        return cls.from_dict(dictionary, append)

    def load_state_dicts_for_restart(self, config: Config):
        dictionary: dict = config.as_dict()

        # ! progress implies that we are resuming training from last_model_path
        assert "progress" in dictionary, "key: 'progress' not present in dictionary, are you running a restart?"
        assert "state_dict" in dictionary, "key: 'state_dict' not present in dictionary, are you running a restart?"
        assert self.optim, "trying to reload state for restart; optimizer must be already instanciated"

        if 'lr_scheduler_name' in dictionary:
            assert self.lr_sched, "trying to reload state for restart; optimizer must be already instanciated"
        if 'early_stopping_lower_bounds' in dictionary or 'early_stopping_patiences' in dictionary:
            assert self.lr_sched, "trying to reload state for restart; optimizer must be already instanciated"
        if 'warmup_epochs' in dictionary:
            assert self.lr_sched, "trying to reload state for restart; optimizer must be already instanciated"

        dictionary = deepcopy(dictionary)
        progress = dictionary["progress"]

        state_dict = dictionary.pop("state_dict") # encapsulates all the states of the optimizer, scheduler, etc. - i.e. the training state
        dictionary.pop("progress")

        logging.info("Reload optimizer and scheduler states")
        for key in self.__class__.object_keys:
            item = getattr(self, key, None)
            if item is not None:
                try:
                    item.load_state_dict(state_dict[key])
                    logging.info(f"{key} state reloaded!")
                except Exception as e:
                    logging.warning(f"Failed to reload state for {key}: {e}")
            else:
                logging.info(f"{key} state NOT found cannot reload it")

        self.cumulative_wall = state_dict["cumulative_wall"]

        torch.set_rng_state(state_dict["rng_state"])
        self.dataset_rng.set_state(state_dict["dataset_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state_dict["cuda_rng_state"])

        self.iepoch = progress["iepoch"]
        self.best_metrics = progress["best_metrics"]
        self.best_epoch = progress["best_epoch"]
        stop_arg = progress.pop("stop_arg", None)

        if self.stop_cond:
            raise RuntimeError(f"The previous run has properly stopped with {stop_arg}. Please either increase the max_epoch or change early stop criteria")

    @classmethod
    def from_config(cls, config: Config, append: Optional[bool] = None):
        return cls.from_dict(config.as_dict(), append)

    @classmethod
    def from_dict(cls, dictionary: dict, append: Optional[bool] = None):
        """load model from dictionary

        Args:

        dictionary (dict):
        append (bool): if True, append the old model files and append the same logfile
        """
        dictionary = deepcopy(dictionary)

        # update the append option
        if append is not None:
            dictionary["append"] = append

        model = None
        iepoch = -1
        if "model" in dictionary:
            model = dictionary.pop("model")
        elif "fine_tune" in dictionary:
            model_pth_path = Path(dictionary["fine_tune"])
            assert isfile(model_pth_path), f"model weights & bias are not saved, {model_pth_path} provided is not a file"
        elif "progress" in dictionary:
            model_pth_path = Path(dictionary["progress"]["last_model_path"])
            assert isfile(model_pth_path), f"model weights & bias are not saved, {model_pth_path} provided is not a file"

        if "progress" in dictionary or "fine_tune" in dictionary:
            model, _ = cls.load_model_from_training_session(
                traindir=model_pth_path.parent,
                model_name=model_pth_path.name,
                config_dictionary=dictionary,
            )
            logging.debug(f"Reload the model from {model_pth_path}")

        # set up the trainer with the default "fresh_start" configuration
        logging.info("Loading Trainer...")
        trainer = cls(**dictionary)
        logging.info("Trainer successfully loaded!")

        trainer.best_epoch = 0
        trainer.iepoch = iepoch
        stop_arg = None

        if trainer.stop_cond:
            raise RuntimeError(f"The previous run has properly stopped with {stop_arg}. Please either increase the max_epoch or change early stop criteria")

        return trainer, model # care, here model CAN be None if no fine-tuning nor progress

    @staticmethod
    def load_model_from_training_session(
        traindir,
        model_name="best_model.pth",
        device="cpu",
        config_dictionary: Optional[dict] = None,
        for_inference: bool = False,
    ) -> Tuple[torch.nn.Module, Config]:
        traindir = str(traindir)
        model_name = str(model_name)

        if config_dictionary is None:
            config = Config.from_file(traindir + "/config.yaml")
        else:
            config = Config.from_dict(config_dictionary)

        model, weights_to_train_from_scratch = model_from_config(
            config=config,
            initialize=False,
            deploy=for_inference,
        ) # raises if returned model is None

        model.to(device=torch.device(device), dtype=torch.float32)
        model_state_dict = torch.load(traindir + "/" + model_name, map_location=device, weights_only=False)
        # drop weights that must be initialized from scratch (if any)
        model_state_dict = {k: v for k, v in model_state_dict.items() if k not in weights_to_train_from_scratch}
        strict = config.get('load_model_state_strict', True) and not ('fine_tune' in config)
        strict=False
        out = model.load_state_dict(model_state_dict, strict=strict)
        print(f"Model loading message: {out}")
        return model, config

    def init_dataset(self, config, train_dset, val_dset):
        self.load_dataset_idcs(train_dset, val_dset)
        self.init_dataloader(config)
        self._init_num_of_optim_steps_per_epoch()

    def init_model(self, **kwargs):
        assert "model" in kwargs
        model = kwargs.get("model")
        self._set_model(model=model)
        self._init_model_dependent_objects()

    def _set_model(self, model):
        self.model = model
        self.model.to(self.torch_device)

        # register hook to clamp gradients
        for p in self.model.parameters():

            if self.sanitize_gradients:

                def sanitize_fn(grad):
                    # Replace NaN values in the gradient with zero
                    grad[torch.isnan(grad)] = 0
                    return grad

                p.register_hook(sanitize_fn)

        self.num_weights = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Number of weights: {self.num_weights}")
        self.logger.info(f"Number of trainable weights: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def train(self):
        """Training"""
        if getattr(self, "dl_train", None) is None:
            raise RuntimeError("You must call `set_dataset()` before calling `train()`")

        for callback in self._initial_callbacks:
            callback(self)

        self.init_log()
        self.wall = perf_counter()
        self.previous_cumulative_wall = self.cumulative_wall

        with atomic_write_group():
            if self.iepoch == -1:
                self.save()

        # hooks_handler = ForwardHookHandler(self, self.hooks)

        # actual train loop
        while not self.stop_cond:
            self.accumulation_counter = 0  # Reset accumulation counter at the start of each epoch
            self.epoch_step()
            if hasattr(self, 'world_size'):
                dist.barrier()
            self.end_of_epoch_save()

        for callback in self._final_callbacks:
            callback(self)

        self.final_log()

        self.save()
        # hooks_handler.deregister_hooks()
        finish_all_writes()

    def _log_updates(self):
        '''
        logs:
        - update to params due to optim step
        - grad_to_weight_ratio: param.grad.std()/param.std()
        '''
        update_log = logging.getLogger(self.log_updates)
        grad_to_weight_ratio_log = logging.getLogger(self.log_ratio)

        # build titles for logging file(s)
        # done only once
        if not hasattr(self, 'update_logging_titles'):
            self.update_logging_titles = [
                param_name
                for param_name, param in self.model.named_parameters()
                if (
                    param.grad is not None and
                    param.dim() > 1 and
                    "bias" not in param_name and
                    "norm" not in param_name
                )
            ]
            _titles = ""
            for t in self.update_logging_titles:
                _titles += f"{t}, "
            _titles = _titles.strip().rstrip(',')
            update_log.info(_titles)
            grad_to_weight_ratio_log.info(_titles)

        # log the values
        update_speed, grad_ratio = "", ""
        with torch.no_grad():
            for param_name, param in self.model.named_parameters():
                if (
                    param.grad is not None and
                    param.dim() > 1 and
                    "bias" not in param_name and
                    "norm" not in param_name
                ):
                    lr = get_latest_lr(self.optim, self.model, param_name)
                    update = ((lr*param.grad).std() /
                              param.std()).log10()  # .item()
                    grad_to_weight_ratio = param.grad.std()/param.std()
                    update_speed += f"{update:.4}, "
                    grad_ratio += f"{grad_to_weight_ratio:.4}, "

        update_log.info(update_speed.strip().rstrip(','))
        grad_to_weight_ratio_log.info(grad_ratio.strip().rstrip(','))

    def _batch_lvl_lrscheduler_step(self):
        # this call must be done from all processes in case of distributed training SINCE IT IS NOT ACTING WRT LOSS/METRICS
        # idea: 2 bool comparison are always going to be more performant then str comparison if len(str)>2
        if hasattr(self, "using_batch_lvl_lrscheduler"):
            if not self.using_batch_lvl_lrscheduler:
                return

        # todo: instead of str comparison could use a dict with k:lr_sched_name, v: 0/1 whether that scheduler is being used + assert check!
        # idea: for loop on num_of_possible_lr_scheduler is surely faster then str cmpr thru the whole lr scheduler name
        if self.lr_scheduler_name == "CosineAnnealingLR":
            self.lr_sched.step()
            if hasattr(self, "using_batch_lvl_lrscheduler"):
                return
            setattr(self, "using_batch_lvl_lrscheduler", True)

        elif self.lr_scheduler_name == "CosineAnnealingWarmRestarts":
            self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)
            if hasattr(self, "using_batch_lvl_lrscheduler"):
                return
            setattr(self, "using_batch_lvl_lrscheduler", True)

    def _epoch_lvl_lrscheduler_step(self):

        if not self.is_master:
            return

        if hasattr(self, "using_batch_lvl_lrscheduler"):
            if self.using_batch_lvl_lrscheduler:
                return

        if self.iepoch > 0 and self.lr_scheduler_name == "ReduceLROnPlateau":
            self.lr_sched.step(metrics=self.mae_dict[self.metrics_key])
            if hasattr(self, "using_batch_lvl_lrscheduler"):
                return
            setattr(self, "using_batch_lvl_lrscheduler", False)

    def _is_warmup_period_over(self):
        # this call must be done from all processes in case of distributed training
        # when this returns true -> start normal lr_scheduler.step() call
        if self.warmup_epochs == -1:
            return True
        n_warmup_steps_already_done = self.warmup_scheduler.last_step
        return n_warmup_steps_already_done + 1 >= self.warmup_steps

    @torch.no_grad()
    def _update_metrics(self, out:Dict[str, torch.Tensor], ref_data:Dict[str, torch.Tensor]) -> None:
        self.batch_metrics = self.metrics(pred=out, ref=ref_data)

    @torch.no_grad()
    def _accumulate_losses(self, loss:torch.Tensor, loss_contrib:Dict[str, torch.Tensor]) -> None:
        '''
        during trainining it must be called after backward+optim
        '''
        self.batch_losses = self.loss_stat(loss, loss_contrib)

    def lr_sched_step(self, batch_lvl:bool) -> None:
        if batch_lvl:
            if not self._is_warmup_period_over():
                with self.warmup_scheduler.dampening():  # @ entering of this cm lrs are dampened iff warmup steps are not over
                    pass
            else:
                self._batch_lvl_lrscheduler_step()
        else: # epoch lvl
            if self.warmup_epochs == -1:
                self._epoch_lvl_lrscheduler_step()
            elif self._is_warmup_period_over():  # warmup present, just need to check if _is_warmup_period_over
                self._epoch_lvl_lrscheduler_step()

    def batch_step(self, data, ctx_mngr, validation:bool=False) -> bool:
        already_computed_nodes = None
        while True:
            out, ref_data, batch_chunk_center_nodes, num_batch_center_nodes = run_inference(
                model=self.model,
                data=data,
                device=self.torch_device,
                already_computed_nodes=already_computed_nodes,
                output_keys=self.output_keys,
                per_node_outputs_keys=self.per_node_outputs_keys,
                cm=ctx_mngr,
                mixed_precision=self.mixed_precision,
                skip_chunking=self.skip_chunking,
                batch_max_atoms=self.batch_max_atoms,
                ignore_chunk_keys=self.ignore_chunk_keys,
                dropout_edges=self.dropout_edges if not validation else 0.,
                requires_grad=self.model_requires_grads,
                is_ddp=self.is_ddp,
            )

            loss, loss_contrib = self.loss(pred=out, ref=ref_data, epoch=self.iepoch)

            if (self.ibatch + 1) % self.log_batch_freq == 0 or (self.ibatch + 1) == self.n_batches and not validation:
                with torch.no_grad():
                    # only master logs its features
                    model = self.model.module if self.is_ddp else self.model
                    irreps_as_dict = {ir.l:mul for (mul, ir) in model.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]}
                    node_reprs = out[AtomicDataDict.NODE_FEATURES_KEY]
                    split_sizes = [mul*(2*l+1) for l,mul in irreps_as_dict.items()]
                    reshapers = [reshape_irreps(o3.Irreps(str(ir))) for ir in model.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]]
                    reprs = torch.split(node_reprs, split_sizes, dim=1)
                    for l, repr in enumerate(reprs):
                        norm = torch.mean(torch.norm(reshapers[l](repr).squeeze(), p=1, dim=(1 if l == 0 else (1, 2)))) / irreps_as_dict[l]
                        self.node_features_norms[f'node_repr_l_{l}_mean'].append(norm)

            # normalized wrt self.accumulation_steps: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3?permalink_comment_id=2907818#gistcomment-2907818
            # also https://discuss.pytorch.org/t/accumulate-gradient/129309/4 [look at referecend post aswell]
            # thus division is required since self.loss.__call__ has mean=T by default
            loss = loss / self.accumulation_steps # average of averages

            self._update_metrics(out, ref_data)
            del ref_data

            if not validation:
                # ref for how ddp works: https://pytorch.org/docs/stable/notes/ddp.html#internal-design;
                # TLDR: every process must call gradfilter_ema (if grokfast), gradient_clipping, optim.step() ema.update(), lr_sched_step and accumulation_counter++
                loss.backward()
                self.accumulation_counter += 1

                if self.use_grokfast:
                    self.grads = gradfilter_ema(self.model, grads=self.grads)

                # if self.debug:
                #     self._log_updates()

                if self.accumulation_counter == self.accumulation_steps:
                    # grad clipping: avoid "shocks" to the model (params) during optimization;
                    # returns norms; their expected trend is from high to low and stabilize
                    grad_norm, max_grad_norm = gradient_clipping(self.model, self.gradnorms_queue, self.max_gradient_norm, self.is_master)
                    if self.is_master:
                        self.gradnorms.append(grad_norm.item())
                        self.gradnorms_clip.append(float(max_grad_norm))

                    self.optim.step()
                    if self.use_ema:
                        self.ema.update()

                    self.optim.zero_grad(set_to_none=True)
                    self.lr_sched_step(batch_lvl=True)
                    self.accumulation_counter = 0

            self._accumulate_losses(loss, loss_contrib)

            # evaluate ending condition
            if self.skip_chunking:
                return True

            already_computed_nodes = evaluate_end_chunking_condition(already_computed_nodes, batch_chunk_center_nodes, num_batch_center_nodes)
            if already_computed_nodes is None:
                return True

    def reset_metrics(self):
        self.loss_stat.reset()
        self.loss_stat.to(self.torch_device)
        self.metrics.reset()
        self.metrics.to(self.torch_device)

    def epoch_step(self):
        dataloaders = {TRAIN: self.dl_train, VALIDATION: self.dl_val}
        categories = [TRAIN, VALIDATION] if self.iepoch >= 0 else [VALIDATION]
        dataloaders = [dataloaders[c] for c in categories]
        self.metrics_dict = {}
        self.loss_dict = {}
        self.gradnorms = []
        self.gradnorms_clip = []
        self.node_features_norms = defaultdict(list)

        for category, dloader in zip(categories, dataloaders):
            self.reset_metrics()
            self.n_batches = len(dloader)

            emacm = False
            if category == VALIDATION:
                self.model.eval()
                if self.use_ema:
                    emacm = True
            else:
                self.model.train()
                self.optim.zero_grad(set_to_none=True)

            if category == TRAIN and isinstance(self.dl_train.sampler, (DistributedSampler, EnsembleDistributedSampler)):
                # https://cerfacs.fr/coop/pytorch-multi-gpu#distributed-data-parallelism-ddp:~:text=train_sampler.set_epoch(epoch)
                self.dl_train.sampler.set_epoch(self.iepoch)

            cm = contextlib.nullcontext() if (self.model_requires_grads or not category == VALIDATION) else torch.no_grad()
            for self.ibatch, batch in enumerate(dloader):
                if emacm:
                    with self.ema.average_parameters(): #! warning: not possible to add this to cm
                        success = self.batch_step(data=batch, ctx_mngr=cm, validation=(category == VALIDATION))
                else:
                    success = self.batch_step(data=batch, ctx_mngr=cm, validation=(category == VALIDATION))

                if success:
                    self.end_of_batch_log(batch_type=category)
                    for callback in self._end_of_batch_callbacks:
                        callback(self)

            self.metrics_dict[category] = self.metrics.current_result()
            self.loss_dict[category] = self.loss_stat.current_result()

            _end_of_category_callbacks = {
                VALIDATION: self._end_of_valid_callbacks,
                TRAIN:      self._end_of_train_callbacks,
            }
            for callback in _end_of_category_callbacks.get(category, []):
                    callback(self)

        self.iepoch += 1
        self.end_of_epoch_log()

        # if the iepoch for the past epoch was -1, it will now be 0
        # for -1 (report_init_validation: True) we aren't training, so it's wrong
        # to step the LR scheduler even if it will have no effect with this particular
        # scheduler at the beginning of training.
        self.lr_sched_step(batch_lvl=False)

        for callback in self._end_of_epoch_callbacks:
            callback(self)

    def end_of_batch_log(self, batch_type: str):
        """
        store all the loss/mae of each batch
        """
        if not self.is_master:
            return

        mat_str = f"{self.iepoch+1:5d}, {self.ibatch+1:5d}"
        log_str = f"  {self.iepoch+1:5d} {self.ibatch+1:5d}"

        header = "epoch, batch"
        log_header = "# Epoch batch"

        # print and store loss value in batch_logger
        for name, value in self.batch_losses.items():
            mat_str += f", {value:16.5g}"
            header += f", {name}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12.12}"

        # append details from metrics
        metrics = self.metrics.flatten_metrics(
            metrics=self.batch_metrics,
            metrics_metadata=self.metrics_metadata,
        )

        for key, value in metrics.items():  # log metrics
            mat_str += f", {value:16.5g}"
            header += f", {key}"
            log_str += f" {value:12.3g}"
            log_header += f" {key:>12.12}"

        batch_logger = logging.getLogger(self.batch_log[batch_type])

        if self.ibatch == 0:
            self.logger.info("")
            self.logger.info(f"{batch_type}")
            self.logger.info(log_header)
            init_step = -1 if self.report_init_validation else 0
            if (self.iepoch == init_step and batch_type == VALIDATION) or (
                self.iepoch == 0 and batch_type == TRAIN
            ):
                batch_logger.info(header)

        batch_logger.info(mat_str)
        if (self.ibatch + 1) % self.log_batch_freq == 0 or (self.ibatch + 1) == self.n_batches:
            self.logger.info(log_str)

    def end_of_epoch_save(self):
        """
        save model and trainer details
        """
        if not self.is_master:
            return

        with atomic_write_group():
            # allow current_metrics to be None at first epoch in case tracked metric is a training metric; mae_dict.keys = list of metrics listed in yaml under metrics_components
            current_metrics = self.mae_dict.get(self.metrics_key, None)
            if not current_metrics:
                return

            # evaluation criteria of what is 'best'
            is_improved = current_metrics < self.best_metrics if self.metric_criteria == 'decreasing' else current_metrics > self.best_metrics
            if is_improved:
                self.best_metrics = current_metrics
                self.best_epoch = self.iepoch

                self.save_ema_model(self.best_model_path, blocking=False)

                self.logger.info(f"! Best model saved {self.best_epoch:8d} {self.best_metrics:8.3f}")

            if (self.iepoch + 1) % self.log_epoch_freq == 0:
                self.save(blocking=False)

            if (
                self.save_checkpoint_freq > 0
                and (self.iepoch + 1) % self.save_checkpoint_freq == 0
            ):
                ckpt_path = self.output.generate_file(f"ckpt{self.iepoch+1}.pth")
                self.save_ema_model(ckpt_path, blocking=False)

    def save_model(self, path, blocking: bool = True):
        with atomic_write(path, blocking=blocking, binary=True) as write_to:
            torch.save(self.model.state_dict(), write_to)

    def init_log(self):
        if not self.is_master:
            return

        if self.iepoch > 0:
            self.logger.info("! Restarting training ...")
        else:
            self.logger.info("! Starting training ...")

    def final_log(self):
        if not self.is_master:
            return

        self.logger.info(f"! Stop training: {self.stop_arg}")
        wall = perf_counter() - self.wall
        self.cumulative_wall = wall + self.previous_cumulative_wall
        self.logger.info(f"Wall time: {wall}")
        self.logger.info(f"Cumulative wall time: {self.cumulative_wall}")

    def end_of_epoch_log(self):
        """
        log validation details at the end of each epoch
        """

        if not self.is_master:
            return

        lr = self.optim.param_groups[0]["lr"]
        wall = perf_counter() - self.wall
        self.cumulative_wall = wall + self.previous_cumulative_wall
        self.mae_dict = dict(
            LR=lr,
            epoch=self.iepoch,
            wall=wall,
            cumulative_wall=self.cumulative_wall,
        )

        header = "epoch, wall, LR"

        categories = [TRAIN, VALIDATION] if self.iepoch > 0 else [VALIDATION]
        log_header = {}
        log_str = {}

        strings = ["Epoch", "wal", "LR"]
        mat_str = f"{self.iepoch:10d}, {wall:8.3f}, {lr:8.3g}"
        for cat in categories:
            log_header[cat] = "# "
            log_header[cat] += " ".join([f"{s:>8s}" for s in strings])
            log_str[cat] = f"{self.iepoch:10d} {wall:8.3f} {lr:8.3g}"

        for category in categories:

            met = self.metrics.flatten_metrics(
                metrics=self.metrics_dict[category],
                metrics_metadata=self.metrics_metadata,
            )

            # append details from loss
            for key, value in self.loss_dict[category].items():
                mat_str += f", {value:16.5g}"
                header += f",{category}_{key}"
                log_str[category] += f" {value:12.3g}"
                log_header[category] += f" {key:>12.12}"
                self.mae_dict[f"{category}_{key}"] = value

            # append details from metrics
            for key, value in met.items():
                mat_str += f", {value:12.3g}"
                header += f",{category}_{key}"
                log_str[category] += f" {value:12.3g}"
                log_header[category] += f" {key:>12.12}"
                self.mae_dict[f"{category}_{key}"] = value

        self.norm_dict = dict(Grad_norm=self.gradnorms, Grad_clip_val=self.gradnorms_clip, **self.node_features_norms)

        if self.iepoch == 0:
            self.init_epoch_logger.info(header)
            self.init_epoch_logger.info(mat_str)
        elif self.iepoch == 1:
            self.epoch_logger.info(header)

        if self.iepoch > 0:
            self.epoch_logger.info(mat_str)

        if self.iepoch > 0:
            self.logger.info("\n\n  Train      " + log_header[TRAIN])
            self.logger.info("! Train      " + log_str[TRAIN])
            self.logger.info("! Validation " + log_str[VALIDATION])
        else:
            self.logger.info("\n\n  Initialization     " +
                             log_header[VALIDATION])
            self.logger.info("! Initial Validation " + log_str[VALIDATION])

        wall = perf_counter() - self.wall
        self.logger.info(f"Wall time: {wall}")

    def __del__(self):

        if not hasattr(self, 'logger'):
            return

        logger = self.logger
        for hdl in logger.handlers:
            hdl.flush()
            hdl.close()
        logger.handlers = []

        for _ in range(len(logger.handlers)):
            logger.handlers.pop()

    def load_dataset_idcs(self,
        dataset: Union[InMemoryConcatDataset, LazyLoadingConcatDataset],
        validation_dataset: Union[InMemoryConcatDataset, LazyLoadingConcatDataset]
    ) -> None: # TODO rename method

        if self.train_idcs is None or self.val_idcs is None:
            # split_dataset to be done before eventual validation_dataset = dataset executed below
            self.train_idcs, self.val_idcs = self.split_dataset(dataset, validation_dataset)

        # default behavior: if no val_dset then val_dset is train_dset
        if validation_dataset is None:
            validation_dataset = dataset

        # --- initialize n_train and n_val
        assert isinstance(self.n_train, (list, int, type(None))), "n_train must be of type list, int, or None"
        self.n_train = self.n_train if isinstance(self.n_train, list) or self.n_train is None else [self.n_train]
        assert isinstance(self.n_val, (list, int, type(None))), "n_val must be of type list, int, or None"
        self.n_val = self.n_val if isinstance(self.n_val, list) or self.n_val is None else [self.n_val]

        assert len(self.n_train) == len(dataset.n_observations)
        assert len(self.n_val) == len(validation_dataset.n_observations)

        def index_dataset(dataset, indices):
            '''
            indexed_dataset: list of 1d-np.arrays containing idxs of each selected element inside each and every .npz
            '''
            indexed_dataset = []
            for data, idcs in zip(dataset.datasets, indices):
                if len(idcs) > 0:
                    if isinstance(dataset, InMemoryConcatDataset):
                        data = data.index_select(idcs)
                        indexed_dataset.append(data)
                    elif isinstance(dataset, LazyLoadingConcatDataset):
                        indexed_dataset.append(data[idcs].reshape(-1))

            if isinstance(dataset, InMemoryConcatDataset):
                return InMemoryConcatDataset(indexed_dataset)
            elif isinstance(dataset, LazyLoadingConcatDataset):
                return dataset.from_indexed_dataset(indexed_dataset)

        self.dataset_train = index_dataset(dataset, self.train_idcs)
        self.dataset_val   = index_dataset(validation_dataset, self.val_idcs)

        self.logger.info(f"Training data structures: {len(self.dataset_train)} | Validation data structures: {len(self.dataset_val)}")
        if self.debug:
            if isinstance(self.dataset_train, InMemoryConcatDataset):
                self.log_data_points(self.dataset_train, prefix='Training')
            if isinstance(self.dataset_val, InMemoryConcatDataset):
                self.log_data_points(self.dataset_val, prefix='Validation')

    def split_dataset(
        self,
        train_dset: Union[InMemoryConcatDataset, LazyLoadingConcatDataset],
        val_dset: Union[InMemoryConcatDataset, LazyLoadingConcatDataset]
    ):
        '''
        This function ALWAYS creates train_dset and a val_dset stored inside trainer
        if val_dset not provided: 80/20 split of train set is performed

        dset.n_observations: list of ints i.e. list of num_of_obs present in npz
        self.n_val (and self.n_train): list of ints i.e. list of num_of_obs that have to be put in val_dset out of given npz
        '''

        val_dset_provided_in_yaml:bool = True if val_dset is not None else False

        def n_train_obs_for_each_npz():
            # returns: list of ints i.e. list of num_of_obs that have to be put in train_dset out of given npz

            def split_80_20(dataset):
                logging.warning("No 'n_train' nor 'n_valid' parameters were provided. Using default 80-20%")
                n_observations = np.array(dataset.n_observations)
                ones_mask = n_observations == 1
                n_observations[~ones_mask] = (0.8 * n_observations[~ones_mask]).astype(int)
                num_ones = np.sum(ones_mask)
                ones = np.copy(n_observations[ones_mask])
                ones[np.random.choice(num_ones, int(0.2*num_ones), replace=False)] = 0
                n_observations[ones_mask] = ones
                return n_observations.tolist()

            if self.n_train: return self.n_train # if already defined, return
            if val_dset_provided_in_yaml: return train_dset.n_observations # train can be itself since val is an indipendent dset (i.e. return all idxs of train)
            # build n_train as "complmement" of n_val: from each_train_npz[i] drop a self.n_val[i]:int observations out of it
            if self.n_val: return [n - val_i for n, val_i in zip(train_dset.n_observations, self.n_val)]
            return split_80_20(train_dset)

        def n_val_obs_for_each_npz():
            if self.n_val: return self.n_val
            if val_dset_provided_in_yaml: return val_dset.n_observations # val can be itself since it is an indipendent dset
            return [n - train_i for n, train_i in zip(train_dset.n_observations, self.n_train)]

        self.n_train = list(n_train_obs_for_each_npz())
        self.n_val   = list(n_val_obs_for_each_npz())

        def get_idxs_permuation(n_obs):
            '''
            example behaviour:
            torch.arange(12): tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
            torch.randperm(12): tensor([ 5,  2,  1, 11,  7,  6, 10,  8,  4,  9,  3,  0])
            '''
            if self.train_val_split == "random": return torch.randperm(n_obs, generator=self.dataset_rng)
            elif self.train_val_split == "sequential": return torch.arange(n_obs)
            else: raise NotImplementedError(f"splitting mode {self.train_val_split} not implemented")

        val_idcs = []
        if val_dset_provided_in_yaml:
            # sampling observations from val_dset: sample from each npz the amount of obs requested
            for n_obs, n_val in zip(val_dset.n_observations, self.n_val):
                if n_val > n_obs: raise ValueError(f"Too little data for validation. Please reduce n_val. n_val: {n_val}, total: {n_obs}")
                idcs = get_idxs_permuation(n_obs)
                val_idcs.append(idcs[:n_val])

        train_idcs = []
        for _index, (n_obs, n_train) in enumerate(zip(train_dset.n_observations, self.n_train)):
            # sampling observations from train_dset: sample from each npz the amount of obs requested
            if n_train > n_obs: raise ValueError(f"Too little data for training. Please reduce n_train. n_train: {n_train}, total: {n_obs}")
            idcs = get_idxs_permuation(n_obs)
            train_idcs.append(idcs[: n_train])

            if not val_dset_provided_in_yaml:
                # sampling from train_dset also for val_dset
                assert len(self.n_train) == len(self.n_val)
                n_val = self.n_val[_index]
                if (n_train + n_val) > n_obs: raise ValueError(f"too little data for training and validation. please reduce n_train and n_val. n_train: {n_train} n_val: {n_val} total: {n_obs}")
                val_idcs.append(idcs[n_train: n_train + n_val])

        return train_idcs, val_idcs

    def init_dataloader(
        self,
        config,
        sampler                 : Sampler | None=None,
        validation_sampler      : Sampler | None=None,
        batch_sampler           : Sampler | None=None,
        batch_validation_sampler: Sampler | None=None,
        ):
        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation

        train_dloader_n_workers = config.get('train_dloader_n_workers', self.dataloader_num_workers)
        val_dloader_n_workers = config.get('val_dloader_n_workers', self.dataloader_num_workers)
        using_multiple_workers = self.dataloader_num_workers or train_dloader_n_workers or val_dloader_n_workers

        dl_kwargs = dict(
            exclude_keys=self.exclude_keys,
            # keep stuff around in memory
            persistent_workers=using_multiple_workers and self.max_epochs > 1,
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=self.torch_device != torch.device("cpu"),
            timeout=config.get('dloader_timeout', 30) if using_multiple_workers > 0 else 0,
            generator=self.dataset_rng,
        )
        dataset_mode = config.get("dataset_mode", "single")
        assert dataset_mode in ["single", "ensemble"], f"Expected 'single' or 'ensemble', got {dataset_mode}"
        use_ensemble = dataset_mode == 'ensemble'
        if use_ensemble:
            if batch_sampler is None:
                batch_sampler = EnsembleSampler(self.dataset_train, self.batch_size)
        else:
            dl_kwargs.update(dict(
                batch_size=self.batch_size,
                shuffle=(sampler is None) and self.shuffle,
                # timeout=0,
            ))

        if using_multiple_workers:
            dl_kwargs['prefetch_factor'] = config.get('dloader_prefetch_factor', 2)

        dl_kwargs['graph_fields'] = _GRAPH_FIELDS # needed for ddp

        self.dl_train = DataLoader(
            num_workers=train_dloader_n_workers,
            dataset=self.dataset_train,
            sampler=sampler,
            batch_sampler=batch_sampler,
            **dl_kwargs,
        )

        # validation, on the other hand, shouldn't shuffle
        # we still pass the generator just to be safe
        if use_ensemble:
            if batch_validation_sampler is None:
                batch_validation_sampler = EnsembleSampler(self.dataset_val, self.validation_batch_size)
        else:
            dl_kwargs.update(dict(
                batch_size=self.validation_batch_size,
                shuffle=False,
            ))

        self.dl_val = DataLoader(
            num_workers=val_dloader_n_workers,
            dataset=self.dataset_val,
            sampler=validation_sampler,
            batch_sampler=batch_validation_sampler,
            **dl_kwargs,
        )


class TrainerWandB(Trainer):
    """Trainer class that adds WandB features"""

    def init(self, **kwargs):
        super(TrainerWandB, self).init(**kwargs)

        if not self.is_master:
            return

        # upload some new fields to wandb
        wandb.config.update({"num_weights": self.num_weights}, allow_val_change=True)

        if self.kwargs.get("wandb_watch", False):
            wandb_watch_kwargs = self.kwargs.get("wandb_watch_kwargs", {})
            if "log" not in wandb_watch_kwargs:
                wandb_watch_kwargs["log"] = None  # do not log sys info
            wandb.watch(self.model, self.loss, **wandb_watch_kwargs)

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        if 'validation_loss' in self.mae_dict:
            val_loss = self.mae_dict['validation_loss']
            if val_loss > 0:
                self.mae_dict.update({'validation_log_loss': math.log(val_loss)})
        if 'training_loss' in self.mae_dict:
            train_loss = self.mae_dict['training_loss']
            if train_loss > 0:
                self.mae_dict.update({'training_log_loss': math.log(train_loss)})
        wandb.log(self.mae_dict)
        for k, v in self.norm_dict.items():
            for norm in v:
                wandb.log({k: norm})


class DistributedTrainer(Trainer):

    def __init__(self, rank: int, world_size: int, *args, **kwargs):
        kwargs["device"] = rank
        self.rank = rank
        self.world_size = world_size
        if 'is_master' in kwargs: # to avoid passing it twice to super().__init__
            kwargs.pop('is_master')
        super().__init__(is_master=rank == 0, *args, **kwargs)

    def init(self, **kwargs):
        # Set the device for this process
        torch.cuda.set_device(self.rank)
        super(DistributedTrainer, self).init(**kwargs)

    def init_dataloader(
        self,
        config,
        sampler                 : Sampler | None=None,
        validation_sampler      : Sampler | None=None,
        batch_sampler           : Sampler | None=None,
        batch_validation_sampler: Sampler | None=None,
    ):
        dataset_mode = config.get("dataset_mode", "single")
        assert dataset_mode in ["single", "ensemble"], f"Expected 'single' or 'ensemble', got {dataset_mode}"
        use_ensemble = dataset_mode == 'ensemble'

        if use_ensemble:
            sampler_class = EnsembleDistributedSampler
            kwargs_keys = ['batch_sampler', 'batch_validation_sampler']
        else:
            sampler_class = DistributedSampler
            kwargs_keys = ['sampler', 'validation_sampler']

        _sampler = sampler_class(
            self.dataset_train,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True, # default already T in pyt impl; care: all resources say that *DataLoader* must have shuffle=F since DistributedSampler is used
        )

        _validation_sampler = sampler_class(
            self.dataset_val,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True, # default already T in pyt impl; care: all resources say that *DataLoader* must have shuffle=F since DistributedSampler is used
        )

        kwargs = {k: v for k, v in zip(kwargs_keys, [_sampler, _validation_sampler])}
        super().init_dataloader(config=config, **kwargs)

    def _set_model(self, model):
        super()._set_model(model)
        from geqtrain.nn import DDP
        self.model = DDP(self.model, device_ids=[self.rank]) #, find_unused_parameters=True) # is for debug purposes only, heavy hit on performance

    def save_model(self, path, blocking: bool = True):
        with atomic_write(path, blocking=blocking, binary=True) as write_to:
            torch.save(self.model.module.state_dict(), write_to)

    def _update_metrics(self, out:Dict[str, torch.Tensor], ref_data:Dict[str, torch.Tensor]) -> None:
        # collect targets/predictions from different processes for current batch/key, preds/targets can be of different shapes across processes (e.g. if atom-wise, whereas in mol-wise they SHOULD be the same as batch size)
        _keys = set(self.metrics.clean_keys) # all keys in out/ref_data for which we must compute metric
        sync_dict_of_tensors_across_GPUs(out, self.world_size, _keys)
        sync_dict_of_tensors_across_GPUs(ref_data, self.world_size, _keys)

        if not self.is_master:
            return

        super()._update_metrics(out, ref_data)

    @torch.no_grad()
    def _accumulate_losses(self, loss:torch.Tensor, loss_contrib:Dict[str, torch.Tensor]) -> None:
        sync_dict_of_tensors_across_GPUs(loss_contrib, self.world_size)
        syncd_loss = sync_tensor_across_GPUs(loss, self.world_size)
        if self.is_master:
            self.batch_losses = self.loss_stat(syncd_loss, loss_contrib)


class DistributedTrainerWandB(TrainerWandB, DistributedTrainer):

    def init(self, **kwargs):
        super(DistributedTrainerWandB, self).init(**kwargs)

    def end_of_epoch_log(self):
        if not self.is_master:
            return
        super().end_of_epoch_log()
