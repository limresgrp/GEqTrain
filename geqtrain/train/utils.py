import re
import logging
from typing import List
import torch
from typing import Tuple, Union
from geqtrain.data import dataset_from_config
from geqtrain.data.dataset import InMemoryConcatDataset, LazyLoadingConcatDataset
from geqtrain.utils import Config
from geqtrain.data import register_fields
from geqtrain.utils.auto_init import instantiate

def parse_loss_metrics_dict(components: dict):
    # parses loss and metric yaml blocks
    # key:str, coeff:flat, func:eg:MSELoss, func_params:dict for init of func
    for key, value in components.items():
        logging.debug(f" parsing {key} {value}")
        coeff = 1.0
        func = "MSELoss"
        func_params = {}
        if isinstance(value, (float, int)):
            coeff = value
        elif isinstance(value, str) or callable(value):
            func = value
        elif isinstance(value, (list, tuple)):
            # list of [func], [func, param], [coeff, func], [coeff, func, params]
            if isinstance(value[0], (float, int)):
                coeff = value[0]
                if len(value) > 1:
                    func = value[1]
                if len(value) > 2:
                    assert isinstance(value[2], dict)
                    func_params = value[2]
            else:
                func = value[0]
                if len(value) > 1:
                    func_params = value[1]
        else:
            raise NotImplementedError(
                f"expected float, list or tuple, but get {type(value)}"
            )
        logging.debug(f" parsing {coeff} {func}")
        yield key, coeff, func, func_params

def find_matching_indices(ls: List[str], patterns: List[str]):
    matching_indices = []
    for i, string in enumerate(ls):
        for pattern in patterns:
            if '*' not in pattern and '?' not in pattern:
                pattern = f"^{pattern}$"
            if re.search(pattern, string):
                matching_indices.append(i)
                break  # Stop checking other patterns if one matches
    return matching_indices

def evaluate_end_chunking_condition(already_computed_nodes, batch_chunk_center_nodes, num_batch_center_nodes):
    '''evaluate ending condition
    if chunking is active -> if whole struct has been processed then batch is over
    already_computed_nodes is the stopping criteria to finish batch step'''
    if already_computed_nodes is None:
        if len(batch_chunk_center_nodes) < num_batch_center_nodes:
            already_computed_nodes = batch_chunk_center_nodes
    elif len(already_computed_nodes) + len(batch_chunk_center_nodes) == num_batch_center_nodes:
        already_computed_nodes = None
    else:
        assert len(already_computed_nodes) + len(batch_chunk_center_nodes) < num_batch_center_nodes
        already_computed_nodes = torch.cat([already_computed_nodes, batch_chunk_center_nodes], dim=0)
    return already_computed_nodes

def instanciate_train_val_dsets(config: Config) -> Tuple[Union[InMemoryConcatDataset, LazyLoadingConcatDataset], Union[InMemoryConcatDataset, LazyLoadingConcatDataset]]:
    train_dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {train_dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(f"Successfully loaded the validation data set of type {validation_dataset}...")
    except KeyError:
        logging.warning("No validation dataset was provided. Using a subset of the train dataset as validation dataset.")
        validation_dataset = None
    return train_dataset, validation_dataset

def load_trainer_and_model(rank: int, world_size: int, config: Config, is_restart=False):
    if config.use_dt:
        config.update({
            "rank": rank,
            "world_size": world_size,
        })

    if config.wandb:
        if rank == 0:
            if is_restart:
                from geqtrain.utils.wandb import resume_wandb_run
                resume_wandb_run(config)
            else:
                from geqtrain.utils.wandb import init_n_update_wandb
                init_n_update_wandb(config)

        if config.use_dt:
            from geqtrain.train import DistributedTrainerWandB
            trainer, model = DistributedTrainerWandB.from_config(config)
        else:
            from geqtrain.train import TrainerWandB
            trainer, model = TrainerWandB.from_config(config)
    else:
        if config.use_dt:
            from geqtrain.train import DistributedTrainer
            trainer, model = DistributedTrainer.from_config(config)
        else:
            from geqtrain.train import Trainer
            trainer, model = Trainer.from_config(config)

    # Register fields:
    # DO NOT REMOVE since needed for ddp
    instantiate(register_fields, all_args=config)

    return trainer, model