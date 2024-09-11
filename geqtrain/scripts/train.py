""" Adapted from https://github.com/mir-group/nequip
"""

""" Train a network."""
import logging
import argparse
import os
import shutil

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401
import torch.distributed as dist

from os.path import isdir
from pathlib import Path

from geqtrain.model import model_from_config
from geqtrain.utils import Config, load_file
from geqtrain.data import dataset_from_config
from geqtrain.utils._global_options import _set_global_options
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.utils.test import assert_AtomicData_equivariant

import warnings
warnings.filterwarnings("ignore")


def setup_process(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(args=None, running_as_script: bool = True):
    args, config = parse_command_line(args)

    if running_as_script:
        set_up_script_logger(config.get("log", None), config.verbose)

    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not (config.append or config.fine_tune):
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    if not found_restart_file:
        func = fresh_start
    elif config.fine_tune:
        if config.use_dt:
            raise NotImplementedError("Could not fine tune in Distributed Training yet.")
        func = fine_tune
    else:
        if config.use_dt:
            raise NotImplementedError("Could not restart training in Distributed Training yet.")
        func = restart
    
    if config.use_dt:
        # Manually set the environment variables for multi-GPU setup
        world_size = args.world_size
        os.environ['WORLD_SIZE'] = str(world_size)  # Number of GPUs/processes to use
        if args.master_addr:
            os.environ['MASTER_ADDR'] = str(args.master_addr)
        if args.master_port:
            os.environ['MASTER_PORT'] = str(args.master_port)
        
        # Spawn one process per GPU
        import torch.multiprocessing as mp
        mp.spawn(func, args=(world_size, config.as_dict(),), nprocs=world_size, join=True)
    else:
        func(None, None, config.as_dict())

    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Train (or restart training of) a model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device on which to run the training. could be either 'cpu' or 'cuda[:n]'",
        default=None,
    )
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training on first frame of the validation dataset",
        action="store_true",
    )
    parser.add_argument(
        "--fine-tune",
        help="enable the fine-tuning mode. The configuration file should contain the dataset on which to perform fine-tuning",
        action="store_true",
    )
    parser.add_argument(
        "--log",
        help="log file to store all the screen logging",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "-ws",
        "--world-size",
        help="Number of available GPUs for Distributed Training",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-ma",
        "--master-addr",
        help="set MASTER_ADDR environment variable for Distributed Training",
        default=None,
    )
    parser.add_argument(
        "-mp",
        "--master-port",
        help="set MASTER_PORT environment variable for Distributed Training",
        default=None,
    )
    args = parser.parse_args(args=args)

    # Check consistency
    if args.world_size is not None:
        if args.device is not None:
            raise argparse.ArgumentError("Cannot specify device when using Distributed Training")
        if args.equivariance_test:
            raise argparse.ArgumentError("You can run Equivariance Test on single CPU/GPU only")

    config = Config.from_file(args.config)

    flags = ("device", "equivariance_test", "fine_tune", "grad_anomaly_mode")
    config.update({flag: getattr(args, flag) for flag in flags if getattr(args, flag) is not None})
    config.update({"use_dt": args.world_size is not None})

    return args, config


def fresh_start(rank, world_size, config):
    try:
        config = Config.from_dict(config)
        # we use add_to_config cause it's a fresh start and need to record it
        _set_global_options(config)

        if config.use_dt:
            # Setup the process for distributed training
            setup_process(rank, world_size)

        # = Make the trainer =
        if config.wandb:
            import wandb  # noqa: F401
            # download parameters from wandb in case of sweeping
            from geqtrain.utils.wandb import init_n_update
            if rank == 0:
                config = init_n_update(config)
            if config.use_dt:
                from geqtrain.train import DistributedTrainerWandB
                trainer = DistributedTrainerWandB(rank=rank, world_size=world_size, **dict(config))
            else:
                from geqtrain.train import TrainerWandB
                trainer = TrainerWandB(**dict(config))
        else:
            if config.use_dt:
                from geqtrain.train import DistributedTrainer
                trainer = DistributedTrainer(rank=rank, world_size=world_size, **dict(config))
            else:
                from geqtrain.train import Trainer
                trainer = Trainer(**dict(config))
        
        shutil.copyfile(Path(config.filepath).resolve(), trainer.config_path)

        config.update(trainer.params)

        # = Load the dataset =
        dataset = dataset_from_config(config, prefix="dataset") # ConcatDataset of many NpzDatasets
        logging.info(f"Successfully loaded the data set of type {dataset}...")
        try:
            validation_dataset = dataset_from_config(config, prefix="validation_dataset")
            logging.info(f"Successfully loaded the validation data set of type {validation_dataset}...")
        except KeyError:
            # It couldn't be found
            validation_dataset = None

        # = Train/validation split =
        trainer.set_dataset(dataset, validation_dataset)
        trainer.set_dataloader()
        
        # = Update config with dataset-related params = #
        config.update(trainer.dataset_params)

        # = Build model =
        model = model_from_config(config=config, initialize=True, dataset=trainer.dataset_train)
        logging.info("Successfully built the network...")

        # Equivar test
        if config.equivariance_test:
            model.eval()
            errstr = assert_AtomicData_equivariant(
                model, trainer.dataset_train[0]
            )
            model.train()
            logging.info(
                "Equivariance test passed; equivariance errors:\n"
                f"{errstr}"
            )
            del errstr

        # Set the trainer
        trainer.init(model=model)

        # Store any updated config information in the trainer
        trainer.update_kwargs(config)

        # Train
        trainer.save()
        trainer.train()
    finally:
        if config.use_dt:
            cleanup()

    return


def fine_tune(rank, world_size, config):

    # load the dictionary
    restart_file = f"{config.root}/{config.run_name}/trainer.pth"
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )

    # compare dictionary to config and update stop condition related arguments
    for k in config.keys():
        if k == "fine_tuning_run_name":
            dictionary["run_name"] = config[k]
            logging.info(f'Update "run_name" to {dictionary["run_name"]}')
            dictionary["n_train"] = None
            dictionary["n_val"] = None
        elif config[k] != dictionary.get(k, ""):
            if k in [
                "fine_tune", "dataset_list", "validation_dataset_list", "seed", "max_epochs",
                "wandb", "wandb_project", "log_batch_freq", "verbose", "append", "keep_type_names",
                "n_train", "n_val", "batch_size", "validation_batch_size",
                "max_epochs", "learning_rate", "loss_coeffs", "device",
                "optimizer_name", "optimizer_params", "metrics_components",
                "lr_scheduler_name", "lr_scheduler_patience", "lr_scheduler_factor", "noise",
            ]:
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif k.startswith("early_stop"):
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif isinstance(config[k], type(dictionary.get(k, ""))):
                raise ValueError(
                    f'Key "{k}" is different in config and the result trainer.pth file. Please double check'
                )

    # Remove keys from dictionary that must be recomputed
    for k in ["train_idcs", "val_idcs"]:
        dictionary.pop(k)

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    _set_global_options(config)

    # = Make the trainer =
    if config.wandb:
        import wandb  # noqa: F401
        # download parameters from wandb in case of sweeping
        from geqtrain.utils.wandb import init_n_update
        from geqtrain.train import TrainerWandB
        config = init_n_update(config)
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from geqtrain.train import Trainer
        trainer = Trainer.from_dict(dictionary)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    trainer.set_dataset(dataset, validation_dataset)
    trainer.set_dataloader()

    # reset scheduler
    trainer.lr_sched._reset()

    # Train
    trainer.save()
    trainer.train()

    return


def restart(rank, world_size, config):
    # load the dictionary
    restart_file = f"{config.root}/{config.run_name}/trainer.pth"
    dictionary = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )

    # compare dictionary to config and update stop condition related arguments
    for k in config.keys():
        if config[k] != dictionary.get(k, ""):
            if k in ["max_epochs", "loss_coeffs", "learning_rate", "device",
                     "metrics_components", "noise"]:
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif k.startswith("early_stop"):
                dictionary[k] = config[k]
                logging.info(f'Update "{k}" to {dictionary[k]}')
            elif isinstance(config[k], type(dictionary.get(k, ""))):
                raise ValueError(
                    f'Key "{k}" is different in config and the result trainer.pth file. Please double check'
                )

    # recursive loop, if same type but different value
    # raise error

    config = Config(dictionary, exclude_keys=["state_dict", "progress"])

    # dtype, etc.
    _set_global_options(config)

    # note, the from_dict method will check whether the code version
    # in trainer.pth is consistent and issue warnings
    if config.wandb:
        from geqtrain.utils.wandb import resume

        resume(config)
        from geqtrain.train import TrainerWandB
        trainer = TrainerWandB.from_dict(dictionary)
    else:
        from geqtrain.train import Trainer
        trainer = Trainer.from_dict(dictionary)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}..."
        )
    except KeyError:
        # It couldn't be found
        validation_dataset = None

    trainer.set_dataset(dataset, validation_dataset)
    trainer.set_dataloader()

    # Train
    trainer.save()
    trainer.train()

    return


if __name__ == "__main__":
    main(running_as_script=True)
