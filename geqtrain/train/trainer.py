""" Adapted from https://github.com/mir-group/nequip
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import inspect
import logging
import wandb
import contextlib
from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Callable, Optional, Union, Tuple, List
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from geqtrain.data import DataLoader, AtomicData, AtomicDataDict
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
)
from geqtrain.model import model_from_config

from .loss import Loss, LossStat
from .metrics import Metrics
from ._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION
from .early_stopping import EarlyStopping


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
        optim_kwargs (dict): parameters to initialize the optimizer

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
    object_keys = ["lr_sched", "optim", "early_stopping_conds"]
    lr_scheduler_module = torch.optim.lr_scheduler
    optim_module = torch.optim

    def __init__(
        self,
        model,
        model_builders: Optional[list] = [],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: Optional[int] = None,
        dataset_seed: Optional[int] = None,
        noise: Optional[float] = None,
        loss_coeffs: Union[dict, str] = None,
        train_on_keys: Optional[List[str]] = None,
        type_names: Optional[List[str]] = None,
        keep_type_names: Optional[List[str]] = None,
        metrics_components: Optional[Union[dict, str]] = None,
        metrics_key: str = f"{VALIDATION}_" + ABBREV.get(LOSS_KEY, LOSS_KEY),
        early_stopping_conds: Optional[EarlyStopping] = None,
        early_stopping: Optional[Callable] = None,
        early_stopping_kwargs: Optional[dict] = None,
        max_epochs: int = 10000,
        learning_rate: float = 1e-2,
        lr_scheduler_name: str = "none",
        lr_scheduler_kwargs: Optional[dict] = None,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        max_gradient_norm: float = float("inf"),
        prod_for = 1,
        prod_every = 0,
        exclude_keys: list = [],
        batch_size: int = 5,
        validation_batch_size: int = 5,
        shuffle: bool = True,
        n_train: Optional[Union[List[int], int]] = None,
        n_val: Optional[Union[List[int], int]] = None,
        dataloader_num_workers: int = 0,
        train_idcs: Optional[Union[list, list[list]]] = None,
        val_idcs: Optional[Union[list, list[list]]] = None,
        train_val_split: str = "random",
        batch_max_atoms: int = 3000,
        ignore_chunk_keys: List[str] = [],
        init_callbacks: list = [],
        end_of_epoch_callbacks: list = [],
        end_of_batch_callbacks: list = [],
        end_of_train_callbacks: list = [],
        final_callbacks: list = [],
        log_batch_freq: int = 1,
        log_epoch_freq: int = 1,
        save_checkpoint_freq: int = -1,
        report_init_validation: bool = True,
        verbose="INFO",
        **kwargs,
    ):
        self._initialized = False
        self.cumulative_wall = 0
        logging.debug("* Initialize Trainer")

        # store all init arguments
        self.model = model

        _local_kwargs = {}
        for key in self.init_keys:
            setattr(self, key, locals()[key])
            _local_kwargs[key] = locals()[key]

        output = Output.get_output(dict(**_local_kwargs, **kwargs))
        self.output = output

        self.logfile = output.open_logfile("log", propagate=True)
        self.epoch_log = output.open_logfile("metrics_epoch.csv", propagate=False)
        self.init_epoch_log = output.open_logfile(
            "metrics_initialization.csv", propagate=False
        )
        self.batch_log = {
            TRAIN: output.open_logfile(
                f"metrics_batch_{ABBREV[TRAIN]}.csv", propagate=False
            ),
            VALIDATION: output.open_logfile(
                f"metrics_batch_{ABBREV[VALIDATION]}.csv", propagate=False
            ),
        }

        # add filenames if not defined
        self.best_model_path = output.generate_file("best_model.pth")
        self.last_model_path = output.generate_file("last_model.pth")
        self.trainer_save_path = output.generate_file("trainer.pth")
        self.config_path = self.output.generate_file("config.yaml")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.dataset_rng = torch.Generator()
        if dataset_seed is not None:
            self.dataset_rng.manual_seed(dataset_seed)

        self.logger.info(f"Torch device: {self.device}")
        self.torch_device = torch.device(self.device)

        # sort out all the other parameters
        # for samplers, optimizer and scheduler
        self.kwargs = deepcopy(kwargs)
        self.optimizer_kwargs = deepcopy(optimizer_kwargs)
        self.lr_scheduler_kwargs = deepcopy(lr_scheduler_kwargs)
        self.early_stopping_kwargs = deepcopy(early_stopping_kwargs)
        self.early_stopping_conds = None

        # initialize training states
        self.best_metrics = float("inf")
        self.best_epoch = 0
        self.iepoch = -1 if self.report_init_validation else 0

        self.loss, _ = instantiate(
            builder=Loss,
            prefix="loss",
            positional_args=dict(coeffs=self.loss_coeffs),
            all_args=self.kwargs,
        )
        self.loss_stat = LossStat(self.loss)

        # what do we train on?
        self.train_on_keys = self.loss.keys
        if train_on_keys is not None:
<<<<<<< HEAD
            if set(train_on_keys) != set(self.train_on_keys):
                logging.info("Different training keys found.")

=======
            assert set(train_on_keys) == set(self.train_on_keys)

>>>>>>> a91e61e (WIP single target training, not working)
        # initialize n_train and n_val
        self.n_train = n_train if isinstance(n_train, list) or n_train is None else [n_train]
        self.n_val = n_val if isinstance(n_val, list) or n_val is None else [n_val]

        # load all callbacks
        self._init_callbacks = [load_callable(callback) for callback in init_callbacks]
        self._end_of_epoch_callbacks = [
            load_callable(callback) for callback in end_of_epoch_callbacks
        ]
        self._end_of_batch_callbacks = [
            load_callable(callback) for callback in end_of_batch_callbacks
        ]
        self._end_of_train_callbacks = [
            load_callable(callback) for callback in end_of_train_callbacks
        ]
        self._final_callbacks = [
            load_callable(callback) for callback in final_callbacks
        ]

        self.init()

    def init_objects(self):
        # initialize optimizer
        self.optim, self.optimizer_kwargs = instantiate_from_cls_name(
            module=torch.optim,
            class_name=self.optimizer_name,
            prefix="optimizer",
            positional_args=dict(params=self.model.parameters(), lr=self.learning_rate),
            all_args=self.kwargs,
            optional_args=self.optimizer_kwargs,
        )

        self.max_gradient_norm = (
            float(self.max_gradient_norm)
            if self.max_gradient_norm is not None
            else float("inf")
        )

        # initialize scheduler
        assert (
            self.lr_scheduler_name
            in ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "none"]
        ) or (
            (len(self.end_of_epoch_callbacks) + len(self.end_of_batch_callbacks)) > 0
        ), f"{self.lr_scheduler_name} cannot be used unless callback functions are defined"
        self.lr_sched = None
        self.lr_scheduler_kwargs = {}
        if self.lr_scheduler_name != "none":
            self.lr_sched, self.lr_scheduler_kwargs = instantiate_from_cls_name(
                module=torch.optim.lr_scheduler,
                class_name=self.lr_scheduler_name,
                prefix="lr_scheduler",
                positional_args=dict(optimizer=self.optim),
                optional_args=self.lr_scheduler_kwargs,
                all_args=self.kwargs,
            )

        # initialize early stopping conditions
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

        if hasattr(self.model, "irreps_out"):
            for key in self.train_on_keys:
                if self.loss.remove_suffix(key) not in self.model.irreps_out:
                    raise RuntimeError(
                        f"Loss function include fields {self.loss.remove_suffix(key)} that are not predicted by the model {self.model.irreps_out}"
                    )

    @property
    def init_keys(self):
        return [
            key
            for key in list(inspect.signature(Trainer.__init__).parameters.keys())
            if key not in (["self", "kwargs", "model"] + Trainer.object_keys)
        ]

    @property
    def params(self):
        return self.as_dict(state_dict=False, training_progress=False, kwargs=False)

    def update_kwargs(self, config):
        self.kwargs.update(
            {key: value for key, value in config.items() if key not in self.init_keys}
        )

    @property
    def logger(self):
        return logging.getLogger(self.logfile)

    @property
    def epoch_logger(self):
        return logging.getLogger(self.epoch_log)

    @property
    def init_epoch_logger(self):
        return logging.getLogger(self.init_epoch_log)

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
                dictionary["state_dict"]["cuda_rng_state"] = torch.cuda.get_rng_state(
                    device=self.torch_device
                )
            dictionary["state_dict"]["cumulative_wall"] = self.cumulative_wall

        if training_progress:
            dictionary["progress"] = {}
            for key in ["iepoch", "best_epoch"]:
                dictionary["progress"][key] = self.__dict__.get(key, -1)
            dictionary["progress"]["best_metrics"] = self.__dict__.get(
                "best_metrics", float("inf")
            )
            dictionary["progress"]["stop_arg"] = self.__dict__.get("stop_arg", None)

            # TODO: these might not both be available, str defined, but no weights
            dictionary["progress"]["best_model_path"] = self.best_model_path
            dictionary["progress"]["last_model_path"] = self.last_model_path
            dictionary["progress"]["trainer_save_path"] = self.trainer_save_path
            if hasattr(self, "config_save_path"):
                dictionary["progress"]["config_save_path"] = self.config_save_path

        return dictionary

    def save_config(self, blocking: bool = True) -> None:
        save_file(
            item=self.as_dict(state_dict=False, training_progress=False),
            supported_formats=dict(yaml=["yaml"]),
            filename=self.config_path,
            enforced_format=None,
            blocking=blocking,
        )

    def save(self, filename: Optional[str] = None, format=None, blocking: bool = True):
        """save the file as filename

        Args:

        filename (str): name of the file
        format (str): format of the file. yaml and json format will not save the weights.
        """

        if filename is None:
            filename = self.trainer_save_path

        logger = self.logger

        state_dict = (
            True
            if format == "torch"
            or filename.endswith(".pth")
            or filename.endswith(".pt")
            else False
        )

        filename = save_file(
            item=self.as_dict(state_dict=state_dict, training_progress=True),
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
            blocking=blocking,
        )
        logger.debug(f"Saved trainer to {filename}")

        self.save_model(self.last_model_path, blocking=blocking)
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

    @classmethod
    def from_dict(cls, dictionary, append: Optional[bool] = None):
        """load model from dictionary

        Args:

        dictionary (dict):
        append (bool): if True, append the old model files and append the same logfile
        """

        dictionary = deepcopy(dictionary)

        # update the restart and append option
        if append is not None:
            dictionary["append"] = append

        model = None
        iepoch = -1
        if "model" in dictionary:
            model = dictionary.pop("model")
        elif "progress" in dictionary:
            progress = dictionary["progress"]

            # load the model from file
            if dictionary.get("fine_tune"):
                if isfile(progress["best_model_path"]):
                    load_path = Path(progress["best_model_path"])
                else:
                    raise AttributeError("model weights & bias are not saved")
            else:
                iepoch = progress["iepoch"]
                if isfile(progress["last_model_path"]):
                    load_path = Path(progress["last_model_path"])
                else:
                    raise AttributeError("model weights & bias are not saved")

            model, _ = cls.load_model_from_training_session(
                traindir=load_path.parent,
                model_name=load_path.name,
                config_dictionary=dictionary,
            )
            logging.debug(f"Reload the model from {load_path}")

            dictionary.pop("progress")

        state_dict = dictionary.pop("state_dict", None)

        trainer = cls(model=model, **dictionary)

        if state_dict is not None and trainer.model is not None and not dictionary.get("fine_tune"):
            logging.debug("Reload optimizer and scheduler states")
            for key in cls.object_keys:
                item = getattr(trainer, key, None)
                if item is not None:
                    item.load_state_dict(state_dict[key])
            trainer._initialized = True
            trainer.cumulative_wall = state_dict["cumulative_wall"]

            torch.set_rng_state(state_dict["rng_state"])
            trainer.dataset_rng.set_state(state_dict["dataset_rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])

        if "progress" in dictionary:
            trainer.best_metrics = progress["best_metrics"]
            trainer.best_epoch = progress["best_epoch"]
            stop_arg = progress.pop("stop_arg", None)
        else:
            trainer.best_metrics = float("inf")
            trainer.best_epoch = 0
            stop_arg = None
        trainer.iepoch = iepoch

        # final sanity check
        if trainer.stop_cond:
            raise RuntimeError(
                f"The previous run has properly stopped with {stop_arg}."
                "Please either increase the max_epoch or change early stop criteria"
            )

        return trainer

    @staticmethod
    def load_model_from_training_session(
        traindir,
        model_name="best_model.pth",
        device="cpu",
        config_dictionary: Optional[dict] = None,
    ) -> Tuple[torch.nn.Module, Config]:
        traindir = str(traindir)
        model_name = str(model_name)

        if config_dictionary is not None:
            config = Config.from_dict(config_dictionary)
        else:
            config = Config.from_file(traindir + "/config.yaml")

        model: torch.nn.Module = model_from_config(
            config=config,
            initialize=False,
        )
        if model is not None:  # TODO: why would it be?
            # TODO: this is not exactly equivalent to building with
            # this set as default dtype... does it matter?
            model.to(
                device=torch.device(device),
                dtype=torch.get_default_dtype(),
            )
            model_state_dict = torch.load(
                traindir + "/" + model_name, map_location=device
            )
            model.load_state_dict(model_state_dict)

        return model, config

    def init(self):
        """initialize optimizer"""
        if self.model is None:
            return

        self.model.to(self.torch_device)

        self.num_weights = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Number of weights: {self.num_weights}")
        self.logger.info(
            f"Number of trainable weights: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        self.init_objects()

        self._initialized = True
        self.cumulative_wall = 0

    def init_metrics(self):
        if self.metrics_components is None:
            self.metrics_components = []
            for key, func in self.loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.lower().startswith("perspecies"),
                }
                self.metrics_components.append((key, "mae", params))
                self.metrics_components.append((key, "rmse", params))

        self.metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=self.metrics_components),
            all_args=self.kwargs,
        )

        if not (
            self.metrics_key.lower().startswith(VALIDATION)
            or self.metrics_key.lower().startswith(TRAIN)
        ):
            raise RuntimeError(
                f"metrics_key should start with either {VALIDATION} or {TRAIN}"
            )
        if self.report_init_validation and self.metrics_key.lower().startswith(TRAIN):
            raise RuntimeError(
                f"metrics_key may not start with {TRAIN} when report_init_validation=True"
            )

    def train(self):

        """Training"""
        if getattr(self, "dl_train", None) is None:
            raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        if not self._initialized:
            self.init()

        for callback in self._init_callbacks:
            callback(self)

        self.init_log()
        self.wall = perf_counter()
        self.previous_cumulative_wall = self.cumulative_wall

        with atomic_write_group():
            if self.iepoch == -1:
                self.save()
            if self.iepoch in [-1, 0]:
                self.save_config()

        self.init_metrics()

        while not self.stop_cond:

            self.epoch_step()
            self.end_of_epoch_save()

        for callback in self._final_callbacks:
            callback(self)

        self.final_log()

        self.save()
        finish_all_writes()

    @classmethod
    def prepare_chunked_input_data(
        cls,
        already_computed_nodes: Optional[torch.Tensor],
        batch: AtomicDataDict.Type,
        data: AtomicDataDict.Type,
        per_node_outputs_keys: List[str] = [],
        per_node_outputs_values: List[torch.Tensor] = [],
        batch_max_atoms: int = 1000,
        ignore_chunk_keys: List[str] = [],
        device = "cpu"
    ):
        # === Limit maximum batch size to avoid CUDA Out of Memory === #
        # === ---------------------------------------------------- === #
        # === ---------------------------------------------------- === #

        chunk = already_computed_nodes is not None
        batch_chunk = deepcopy(batch)
        batch_chunk_index = batch_chunk[AtomicDataDict.EDGE_INDEX_KEY]

        if chunk:
            batch_chunk_index = batch_chunk_index[:, ~torch.isin(batch_chunk_index[0], already_computed_nodes)]
        batch_chunk_center_node_idcs = batch_chunk_index[0].unique()
        if len(batch_chunk_center_node_idcs) == 0:
            return

        # = Iteratively remove edges from batch_chunk = #
        # = ----------------------------------------- = #

        offset = 0
        while len(batch_chunk_index.unique()) > batch_max_atoms:

            def get_node_center_idcs(batch_chunk_index: torch.Tensor, batch_max_atoms: int, offset: int):
                unique_set = set()

                for i, num in enumerate(batch_chunk_index[1]):
                    unique_set.add(num.item())

                    if len(unique_set) >= batch_max_atoms:
                        return batch_chunk_index[0, :i+1].unique()[:-offset]
                return batch_chunk_index[0].unique()[:-offset]

            def get_edge_filter(batch_chunk_index: torch.Tensor, offset: int):
                node_center_idcs = get_node_center_idcs(batch_chunk_index, batch_max_atoms, offset)
                edge_filter = torch.isin(batch_chunk_index[0], node_center_idcs)
                return edge_filter

            chunk = True
            offset += 1
            batch_chunk_index = batch_chunk_index[:, get_edge_filter(batch_chunk_index, offset)]

        # = ----------------------------------------- = #

        if chunk:
            batch_chunk[AtomicDataDict.EDGE_INDEX_KEY] = batch_chunk_index
            batch_chunk[AtomicDataDict.BATCH_KEY] = data[AtomicDataDict.BATCH_KEY][batch_chunk_index.unique()]
            for per_node_output_key, per_node_outputs_value in zip(per_node_outputs_keys, per_node_outputs_values):
                chunk_per_node_outputs_value = per_node_outputs_value.clone()
                mask = torch.ones_like(chunk_per_node_outputs_value, dtype=torch.bool)
                mask[batch_chunk_index[0].unique()] = False
                chunk_per_node_outputs_value[mask] = torch.nan
                batch_chunk[per_node_output_key] = chunk_per_node_outputs_value

        # === ---------------------------------------------------- === #
        # === ---------------------------------------------------- === #

        if hasattr(data, "__slices__"):
            for slices_key, slices in data.__slices__.items():
                batch_chunk[f"{slices_key}_slices"] = torch.tensor(slices, dtype=int)
        batch_chunk["ptr"] = torch.nn.functional.pad(torch.bincount(batch_chunk.get(AtomicDataDict.BATCH_KEY)).flip(dims=[0]), (0, 1), mode='constant').flip(dims=[0])

        edge_index = batch_chunk[AtomicDataDict.EDGE_INDEX_KEY]
        node_index = edge_index.unique(sorted=True)

        for key in batch_chunk.keys():
            if key in [
                AtomicDataDict.BATCH_KEY,
                AtomicDataDict.EDGE_INDEX_KEY,
            ] + ignore_chunk_keys:
                continue
            dim = np.argwhere(np.array(batch_chunk[key].size()) == len(data[AtomicDataDict.BATCH_KEY])).flatten()
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
        }

        return input_data, batch_chunk, batch_chunk_center_nodes

    def batch_step(self, data, validation=False):
        # no need to have gradients from old steps taking up memory
        self.optim.zero_grad(set_to_none=True)

        if validation:
            self.model.eval()
            # cm = torch.no_grad()
            cm = contextlib.nullcontext()
        else:
            self.model.train()
            cm = contextlib.nullcontext()

        batch = AtomicData.to_AtomicDataDict(data.to(self.torch_device)) # AtomicDataDict is the dstruct that is taken as input from each forward

        with cm:
            out = self.model(batch) # forward of the model
        del input_data

        ref = batch['mu']
        if not validation:
            loss, loss_contrib = self.loss(pred=out, ref=ref) # compute loss

            self.optim.zero_grad(set_to_none=True) # 0 grad

            loss.backward() # compue grads

            if self.max_gradient_norm < float("inf"): # grad clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_gradient_norm
                )

            for n, param in self.model.named_parameters(): # replaces possible nan gradients to 0 #! bad?
                if param.grad is not None and torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

            self.optim.step()
            # self.model.normalize_weights() #? scales parms by their norm (Not weight_norm)

            if self.lr_scheduler_name == "CosineAnnealingWarmRestarts": # lr scheduler step
                self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)

        with torch.no_grad(): # val step if required and comp metrics for log
            if validation:
                loss, loss_contrib = self.loss(pred=out, ref=ref)

            self.batch_losses = self.loss_stat(loss, loss_contrib)
            self.batch_metrics = self.metrics(pred=out, ref=ref)


    # def batch_step(self, data, validation=False):
    #     # no need to have gradients from old steps taking up memory
    #     self.optim.zero_grad(set_to_none=True)

    #     if validation:
    #         self.model.eval()
    #         cm = torch.no_grad()
    #     else:
    #         self.model.train()
    #         cm = contextlib.nullcontext()

    #     batch = AtomicData.to_AtomicDataDict(data.to(self.torch_device)) # AtomicDataDict is the dstruct that is taken as input from each forward
    #     # # # keep_bead_types = self.keep_bead_types

    #     # # Remove edges of atoms whose result is NaN
    #     per_node_outputs_keys = []
    #     # for key in self.loss.coeffs:
    #     #     if hasattr(self.loss.funcs[key], "ignore_nan") and self.loss.funcs[key].ignore_nan:
    #     #         key_clean = self.loss.remove_suffix(key)
    #     #         batch[key_clean] = batch[key_clean].squeeze()
    #     #         if key_clean in batch and len(batch[key_clean]) == len(batch[AtomicDataDict.BATCH_KEY]):
    #     #             val = batch[key_clean].reshape(len(batch[key_clean]), -1)
    #     #             # # # if keep_bead_types is not None:
    #     #             # # #     # Remove edges of atoms that do not appear in keep_bead_types
    #     #             # # #     keep_bead_types = torch.tensor(keep_bead_types, device=self.torch_device)
    #     #             # # #     batch[key_clean][~torch.isin(batch[AtomicDataDict.NODE_TYPE_KEY].flatten(), keep_bead_types)] = torch.nan

    #     #             not_nan_edge_filter = torch.isin(batch[AtomicDataDict.EDGE_INDEX_KEY][0], torch.argwhere(torch.any(~torch.isnan(val), dim=1)).flatten())
    #     #             batch[AtomicDataDict.EDGE_INDEX_KEY] = batch[AtomicDataDict.EDGE_INDEX_KEY][:, not_nan_edge_filter]
    #     #             batch[AtomicDataDict.BATCH_KEY] = batch[AtomicDataDict.BATCH_KEY][batch[AtomicDataDict.EDGE_INDEX_KEY].unique()]
    #     #             per_node_outputs_keys.append(key_clean)

    #     per_node_outputs_values = []
    #     for per_node_output_key in per_node_outputs_keys:
    #         if per_node_output_key in batch:
    #             per_node_outputs_values.append(batch.get(per_node_output_key))

    #     batch_index = batch[AtomicDataDict.EDGE_INDEX_KEY]
    #     num_batch_center_nodes = len(batch_index[0].unique())
    #     already_computed_nodes = None

    #     while True:

    #         input_data, batch_chunk, batch_chunk_center_nodes = self.prepare_chunked_input_data(
    #             already_computed_nodes=already_computed_nodes,
    #             batch=batch,
    #             data=data,
    #             per_node_outputs_keys=per_node_outputs_keys,
    #             per_node_outputs_values=per_node_outputs_values,
    #             batch_max_atoms=self.batch_max_atoms,
    #             ignore_chunk_keys=self.ignore_chunk_keys,
    #             device=self.torch_device,
    #         )

    #         if self.noise is not None: #!?
    #             input_data[AtomicDataDict.NOISE] = self.noise * torch.randn_like(input_data[AtomicDataDict.POSITIONS_KEY])

    #         with cm:
    #             out = self.model(input_data) # forward of the model
    #         del input_data

    #         if already_computed_nodes is None: # already_computed_nodes is the stopping criteria to finish batch step
    #             if len(batch_chunk_center_nodes) < num_batch_center_nodes:
    #                 already_computed_nodes = batch_chunk_center_nodes
    #         elif len(already_computed_nodes) + len(batch_chunk_center_nodes) == num_batch_center_nodes:
    #             already_computed_nodes = None
    #         else:
    #             assert len(already_computed_nodes) + len(batch_chunk_center_nodes) < num_batch_center_nodes
    #             already_computed_nodes = torch.cat([already_computed_nodes, batch_chunk_center_nodes], dim=0)

    #         if not validation:
    #             loss, loss_contrib = self.loss(pred=out, ref=batch_chunk) # compute loss

    #             self.optim.zero_grad(set_to_none=True) # 0 grad

    #             loss.backward() # compue grads

    #             if self.max_gradient_norm < float("inf"): # grad clipping
    #                 torch.nn.utils.clip_grad_norm_(
    #                     self.model.parameters(), self.max_gradient_norm
    #                 )

    #             for n, param in self.model.named_parameters(): # replaces possible nan gradients to 0 #! bad?
    #                 if param.grad is not None and torch.isnan(param.grad).any():
    #                     param.grad[torch.isnan(param.grad)] = 0

    #             self.optim.step()
    #             self.model.normalize_weights() # scales parms by their norm (Not weight_norm)

    #             if self.lr_scheduler_name == "CosineAnnealingWarmRestarts": # lr scheduler step
    #                 self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)

    #         with torch.no_grad(): # val step if required and comp metrics for log
    #             if validation:
    #                 loss, loss_contrib = self.loss(pred=out, ref=batch_chunk)

    #             self.batch_losses = self.loss_stat(loss, loss_contrib)
    #             self.batch_metrics = self.metrics(pred=out, ref=batch_chunk)

    #         del batch_chunk

    #         if already_computed_nodes is None:
    #             return

    @property
    def stop_cond(self):
        """kill the training early"""

        if self.early_stopping_conds is not None and hasattr(self, "mae_dict"):
            early_stop, early_stop_args, debug_args = self.early_stopping_conds(
                self.mae_dict
            )
            if debug_args is not None:
                self.logger.debug(debug_args)
            if early_stop:
                self.stop_arg = early_stop_args
                return True

        if self.iepoch >= self.max_epochs:
            self.stop_arg = "max epochs"
            return True

        return False

    def reset_metrics(self):
        self.loss_stat.reset()
        self.loss_stat.to(self.torch_device)
        self.metrics.reset()
        self.metrics.to(self.torch_device)

    def epoch_step(self):
        self.model.prod(max(0, self.iepoch) % (self.prod_for + self.prod_every) < self.prod_every)

        dataloaders = {TRAIN: self.dl_train, VALIDATION: self.dl_val}
        categories = [TRAIN, VALIDATION] if self.iepoch >= 0 else [VALIDATION]
        dataloaders = [
            dataloaders[c] for c in categories
        ]  # get the right dataloaders for the catagories we actually run
        self.metrics_dict = {}
        self.loss_dict = {}

        for category, dataset in zip(categories, dataloaders):
            self.reset_metrics()
            self.n_batches = len(dataset)
            for self.ibatch, batch in enumerate(dataset):
                self.batch_step(
                    data=batch,
                    validation=(category == VALIDATION),
                )
                self.end_of_batch_log(batch_type=category)
                for callback in self._end_of_batch_callbacks:
                    callback(self)
            self.metrics_dict[category] = self.metrics.current_result()
            self.loss_dict[category] = self.loss_stat.current_result()

            if category == TRAIN:
                for callback in self._end_of_train_callbacks:
                    callback(self)

        self.iepoch += 1

        self.end_of_epoch_log()

        # if the iepoch for the past epoch was -1, it will now be 0
        # for -1 (report_init_validation: True) we aren't training, so it's wrong
        # to step the LR scheduler even if it will have no effect with this particular
        # scheduler at the beginning of training.
        if self.iepoch > 0 and self.lr_scheduler_name == "ReduceLROnPlateau":
            self.lr_sched.step(metrics=self.mae_dict[self.metrics_key])

        for callback in self._end_of_epoch_callbacks:
            callback(self)

    def end_of_batch_log(self, batch_type: str):
        """
        store all the loss/mae of each batch
        """

        mat_str = f"{self.iepoch+1:5d}, {self.ibatch+1:5d}"
        log_str = f"  {self.iepoch+1:5d} {self.ibatch+1:5d}"

        header = "epoch, batch"
        log_header = "# Epoch batch"

        # print and store loss value
        for name, value in self.batch_losses.items():
            mat_str += f", {value:16.5g}"
            header += f", {name}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12.12}"

        # append details from metrics
        metrics, skip_keys = self.metrics.flatten_metrics(
            metrics=self.batch_metrics,
            type_names=self.type_names,
        )
        for key, value in metrics.items():

            mat_str += f", {value:16.5g}"
            header += f", {key}"
            if key not in skip_keys:
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
        if (self.ibatch + 1) % self.log_batch_freq == 0 or (
            self.ibatch + 1
        ) == self.n_batches:
            self.logger.info(log_str)

    def end_of_epoch_save(self):
        """
        save model and trainer details
        """
        with atomic_write_group():
            current_metrics = self.mae_dict[self.metrics_key]
            if current_metrics < self.best_metrics:
                self.best_metrics = current_metrics
                self.best_epoch = self.iepoch

                self.save_model(self.best_model_path, blocking=False)

                self.logger.info(
                    f"! Best model {self.best_epoch:8d} {self.best_metrics:8.3f}"
                )

            if (self.iepoch + 1) % self.log_epoch_freq == 0:
                self.save(blocking=False)

            if (
                self.save_checkpoint_freq > 0
                and (self.iepoch + 1) % self.save_checkpoint_freq == 0
            ):
                ckpt_path = self.output.generate_file(f"ckpt{self.iepoch+1}.pth")
                self.save_model(ckpt_path, blocking=False)

    def save_model(self, path, blocking: bool = True):
        with atomic_write(path, blocking=blocking, binary=True) as write_to:
            torch.save(self.model.state_dict(), write_to)

    def init_log(self):
        if self.iepoch > 0:
            self.logger.info("! Restarting training ...")
        else:
            self.logger.info("! Starting training ...")

    def final_log(self):

        self.logger.info(f"! Stop training: {self.stop_arg}")
        wall = perf_counter() - self.wall
        self.cumulative_wall = wall + self.previous_cumulative_wall
        self.logger.info(f"Wall time: {wall}")
        self.logger.info(f"Cumulative wall time: {self.cumulative_wall}")

    def end_of_epoch_log(self):
        """
        log validation details at the end of each epoch
        """

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

            met, skip_keys = self.metrics.flatten_metrics(
                metrics=self.metrics_dict[category],
                type_names=self.type_names,
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
                if key not in skip_keys:
                    log_str[category] += f" {value:12.3g}"
                    log_header[category] += f" {key:>12.12}"
                self.mae_dict[f"{category}_{key}"] = value

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
            self.logger.info("\n\n  Initialization     " + log_header[VALIDATION])
            self.logger.info("! Initial Validation " + log_str[VALIDATION])

        wall = perf_counter() - self.wall
        self.logger.info(f"Wall time: {wall}")

    def __del__(self):

        if not self._initialized:
            return

        logger = self.logger
        for hdl in logger.handlers:
            hdl.flush()
            hdl.close()
        logger.handlers = []

        for i in range(len(logger.handlers)):
            logger.handlers.pop()

    def set_dataset(
        self,
        dataset: ConcatDataset,
        validation_dataset: Optional[ConcatDataset] = None,
    ) -> None:
        """Set the dataset(s) used by this trainer.

        Training and validation datasets will be sampled from
        them in accordance with the trainer's parameters.

        If only one dataset is provided, the train and validation
        datasets will both be sampled from it. Otherwise, if
        `validation_dataset` is provided, it will be used.
        """

        if validation_dataset is None:
            logging.warn("No validation dataset was provided. Using a subset of the train dataset as validation dataset.")
        if self.train_idcs is None or self.val_idcs is None:
            if self.n_train is None:
                if validation_dataset is None:
                    if self.n_val is not None:
                        self.n_train = [len(ds) - n_valid for ds, n_valid in zip(dataset.datasets, self.n_val)]
                    else:
                        logging.warn("No 'n_train' nor 'n_valid' parameters were provided. Using default 80-20%")
                        self.n_train = [int(0.8*len(ds)) for ds in dataset.datasets]
                else:
                    self.n_train = [len(ds) for ds in dataset.datasets]
            if self.n_val is None:
                if validation_dataset is not None:
                    self.n_val = [len(ds) for ds in validation_dataset.datasets]
                else:
                    self.n_val = [len(ds) - n_train for ds, n_train in zip(dataset.datasets, self.n_train)]
            self.train_idcs, self.val_idcs = [], []
            # Sample both from `dataset`:
            if validation_dataset is not None:
                for _validation_dataset, n_val in zip(validation_dataset.datasets, self.n_val):
                    total_n = len(_validation_dataset)
                    if n_val > total_n:
                        raise ValueError(
                            "too little data for validation. please reduce n_val"
                        )
                    if self.train_val_split == "random":
                        idcs = torch.randperm(total_n, generator=self.dataset_rng)
                    elif self.train_val_split == "sequential":
                        idcs = torch.arange(total_n)
                    else:
                        raise NotImplementedError(
                            f"splitting mode {self.train_val_split} not implemented"
                        )
                    self.val_idcs.append(idcs[:n_val])

            # If validation_dataset is None, Sample both from `dataset`
            for _index, (_dataset, n_train) in enumerate(zip(dataset.datasets, self.n_train)):
                total_n = len(_dataset)

                if n_train > total_n:
                    raise ValueError(
                        f"too little data for training. please reduce n_train. n_train: {n_train} total: {total_n}"
                    )

                if self.train_val_split == "random":
                    idcs = torch.randperm(total_n, generator=self.dataset_rng)
                elif self.train_val_split == "sequential":
                    idcs = torch.arange(total_n)
                else:
                    raise NotImplementedError(
                        f"splitting mode {self.train_val_split} not implemented"
                    )

                self.train_idcs.append(idcs[: n_train])
                if validation_dataset is None:
                    assert len(self.n_train) == len(self.n_val)
                    n_val = self.n_val[_index]
                    if (n_train + n_val) > total_n:
                        raise ValueError(
                            f"too little data for training and validation. please reduce n_train and n_val. n_train: {n_train} n_val: {n_val} total: {total_n}"
                        )
                    self.val_idcs.append(idcs[n_train : n_train + n_val])
        if validation_dataset is None:
            validation_dataset = dataset

        # assert len(self.n_train) == len(dataset.datasets)
        assert len(self.n_val) == len(validation_dataset.datasets)

        # torch_geometric datasets inherantly support subsets using `index_select`
        indexed_datasets_train = []
        for _dataset, train_idcs in zip(dataset.datasets, self.train_idcs):
            indexed_datasets_train.append(_dataset.index_select(train_idcs))
        self.dataset_train = ConcatDataset(indexed_datasets_train)

        indexed_datasets_val = []
        for _dataset, val_idcs in zip(validation_dataset.datasets, self.val_idcs):
            indexed_datasets_val.append(_dataset.index_select(val_idcs))
        self.dataset_val = ConcatDataset(indexed_datasets_val)

        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        dl_kwargs = dict(
            exclude_keys=self.exclude_keys,
            num_workers=self.dataloader_num_workers,
            # keep stuff around in memory
            persistent_workers=(
                self.dataloader_num_workers > 0 and self.max_epochs > 1
            ),
            # PyTorch recommends this for GPU since it makes copies much faster
            pin_memory=(self.torch_device != torch.device("cpu")),
            # avoid getting stuck
            timeout=(10 if self.dataloader_num_workers > 0 else 0),
            # use the right randomness
            generator=self.dataset_rng,
        )
        self.dl_train = DataLoader(
            dataset=self.dataset_train,
            shuffle=self.shuffle,  # training should shuffle
            batch_size=self.batch_size,
            **dl_kwargs,
        )
        # validation, on the other hand, shouldn't shuffle
        # we still pass the generator just to be safe
        self.dl_val = DataLoader(
            dataset=self.dataset_val,
            batch_size=self.validation_batch_size,
            **dl_kwargs,
        )


class TrainerWandB(Trainer):
    """Trainer class that adds WandB features"""

    def end_of_epoch_log(self):
        Trainer.end_of_epoch_log(self)
        wandb.log(self.mae_dict)

    def init(self):
        super().init()

        if not self._initialized:
            return

        # upload some new fields to wandb
        wandb.config.update({"num_weights": self.num_weights})

        if self.kwargs.get("wandb_watch", False):
            wandb_watch_kwargs = self.kwargs.get("wandb_watch_kwargs", {})
            wandb.watch(self.model, **wandb_watch_kwargs)