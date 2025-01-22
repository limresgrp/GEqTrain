""" Adapted from https://github.com/mir-group/nequip
"""

""" Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import warnings
from geqtrain.utils.test import assert_AtomicData_equivariant
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.utils._global_options import _set_global_options
from geqtrain.utils import Config, load_file
from geqtrain.model import model_from_config
from pathlib import Path
from os.path import isdir
import torch.distributed as dist
import logging
import argparse
import os
import shutil
from typing import Dict, Optional, Tuple, Union
import numpy as np  # noqa: F401
from geqtrain.data import dataset_from_config
from geqtrain.data.dataset import InMemoryConcatDataset, LazyLoadingConcatDataset
import torch

warnings.filterwarnings("ignore")


def setup_distributed_training(rank:int, world_size:int):
    """Initialize the process group for distributed training
    Args:
        rank (int): rank of the current process (i.e. device id assigned to the process)
        world_size (int): number of processes (i.e. number of GPUs that are going to be used for training)
    """
    for device_id in range(world_size):
        # Before init the process group, call torch.cuda.set_device(args.rank) to assign different GPUs to different processes.
        # https://github.com/pytorch/pytorch/issues/18689
        torch.cuda.set_device(device_id)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup_distributed_training(rank):
    logging.info(f"Rank: {rank} | Destroying process group")
    dist.barrier()
    dist.destroy_process_group()


def configure_dist_training(args):

    def get_free_port():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))
        addr = s.getsockname()
        s.close()
        return addr[1]

    # Manually set the environment variables for multi-GPU setup
    world_size = args.world_size
    # Number of GPUs/processes to use
    os.environ['WORLD_SIZE'] = str(world_size)
    if args.master_addr:
        os.environ['MASTER_ADDR'] = str(args.master_addr)
    if args.master_port:
        port = get_free_port() if args.master_port == 'rand' else args.master_port
        os.environ['MASTER_PORT'] = str(port)

    # Spawn one process per GPU
    # mp.set_start_method('spawn', force=True)

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "INFO" # "DETAIL"  # set to DETAIL for runtime logging.
    return world_size


# geqtrain-train ./config/halicin.yaml -ws 2
def main(args=None, running_as_script: bool = True):
    args, config = parse_command_line(args)

    if running_as_script:
        set_up_script_logger(config.get("log", None), config.verbose)

    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not (config.append):
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    if not found_restart_file:
        func = fresh_start
    else:
        func = restart

    config = Config.from_dict(config)
    _set_global_options(config)
    train_dataset, validation_dataset = instanciate_train_val_dsets(config)

    if config.use_dt:
        import torch.multiprocessing as mp
        world_size = configure_dist_training(args)
        # autonomous handling of rank, each process runs func
        mp.spawn(func, args=(world_size, config.as_dict(), train_dataset, validation_dataset,), nprocs=world_size, join=True)
    else:
        func(rank=0, world_size=1, config=config.as_dict(), train_dataset=train_dataset, validation_dataset=validation_dataset)
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
        "-ws", # if wd is present then use_dt
        "--world-size",
        help="Number of available GPUs for Distributed Training",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-ma",
        "--master-addr",
        help="set MASTER_ADDR environment variable for Distributed Training",
        default='localhost',
    )
    parser.add_argument(
        "-mp",
        "--master-port",
        help="set MASTER_PORT environment variable for Distributed Training",
        default='rand',
    )
    args = parser.parse_args(args=args)

    # Check consistency
    if args.world_size is not None:
        if args.device is not None:
            raise argparse.ArgumentError("Cannot specify device when using Distributed Training")
        if args.equivariance_test:
            raise argparse.ArgumentError("You can run Equivariance Test on single CPU/GPU only")

    config = Config.from_file(args.config)

    flags = ("device", "equivariance_test", "grad_anomaly_mode")
    config.update({flag: getattr(args, flag)
                  for flag in flags if getattr(args, flag) is not None})
    config.update({"use_dt": args.world_size is not None})

    return args, config


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


def fresh_start(rank, world_size, config, train_dataset, validation_dataset):
    try:
        # recast config to be a Config obj
        config = Config.from_dict(config)
        assert isinstance(config, Config), "config must be of type Config"

        if config.use_dt:
            setup_distributed_training(rank, world_size)

        trainer, model = load_trainer_and_model(rank, world_size, config)
        # Copy conf file in results folder
        shutil.copyfile(Path(config.filepath).resolve(), trainer.config_path)
        config.update(trainer.params)

        trainer.init_dataset(config, train_dataset, validation_dataset)

        # = Update config with dataset-related params = #
        config.update(trainer.dataset_params)

        # = Build model =
        if model is None:
            logging.info("Building the network...")
            model = model_from_config(
                config=config, initialize=True, dataset=trainer.dataset_train)
            logging.info("Successfully built the network!")

        # Equivar test
        if config.equivariance_test:
            logging.info("Running equivariance test...")
            model.eval()
            errstr = assert_AtomicData_equivariant(
                model, trainer.dataset_train[0])
            model.train()
            logging.info(
                f"Equivariance test passed; equivariance errors:\n{errstr}")
            del errstr

        trainer.init(model=model)
        trainer.update_kwargs(config)

        # Run training
        trainer.save()
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Process manually stopped!")
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        try:
            if config.use_dt:
                cleanup_distributed_training(rank)
        except:
            pass


def restart(rank, world_size, config, train_dataset, validation_dataset):

    def check_for_param_updates():
        # compare old_config to config and update stop condition related arguments

        modifiable_params = ["max_epochs", "loss_coeffs", "learning_rate", "device", "metrics_components",
                         "noise", "use_dt", "wandb", "batch_size", "validation_batch_size"]

        for k,v in config.items():
            if v != old_config.get(k, ""):
                if k in modifiable_params:
                    old_config[k] = v
                    logging.info(f'Update "{k}" to {old_config[k]}')
                elif k.startswith("early_stop"):
                    old_config[k] = v
                    logging.info(f'Update "{k}" to {old_config[k]}')
                elif k == 'filepath':
                    assert Path(config[k]).resolve() == Path(old_config[k]).resolve()
                    old_config[k] = v
                elif isinstance(v, type(old_config.get(k, ""))):
                    raise ValueError(f'Key "{k}" is different in config and the result trainer.pth file. Please double check')

    try:
        # trainer to dict: parsed dict is the used to instanciate Config
        restart_file = f"{config['root']}/{config['run_name']}/trainer.pth"
        old_config = load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=restart_file,
            enforced_format="torch",
        )

        check_for_param_updates()

        config = Config(old_config, exclude_keys=["state_dict", "progress"])
        _set_global_options(config)

        if config.use_dt:
            setup_distributed_training(rank, world_size)

        trainer, model = load_trainer_and_model(rank, world_size, config, old_config=old_config, is_restart=True)
        trainer.init_dataset(config, train_dataset, validation_dataset)
        trainer.init(model=model)
        trainer.update_kwargs(config)

        # Run training
        trainer.save()
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Process manually stopped!")
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        try:
            if old_config.get("use_dt", False):
                cleanup_distributed_training(rank)
        except:
            pass
    return


def load_trainer_and_model(rank: int, world_size: int, config: Config, old_config: Optional[Dict] = None, is_restart=False):
    if old_config is None:
        old_config = dict(config)
    if config.use_dt:
        old_config.update({
            "rank": rank,
            "world_size": world_size,
        })
    if config.wandb:
        if rank == 0:
            if is_restart:
                from geqtrain.utils.wandb import resume
                resume(config)
            else:
                from geqtrain.utils.wandb import init_n_update
                init_n_update(config)
        if config.use_dt:
            from geqtrain.train import DistributedTrainerWandB
            trainer, model = DistributedTrainerWandB.from_dict(old_config)
        else:
            from geqtrain.train import TrainerWandB
            trainer, model = TrainerWandB.from_dict(old_config)
    else:
        if config.use_dt:
            from geqtrain.train import DistributedTrainer
            trainer, model = DistributedTrainer.from_dict(old_config)
        else:
            from geqtrain.train import Trainer
            trainer, model = Trainer.from_dict(old_config)
    return trainer, model


if __name__ == "__main__":
    main(running_as_script=True)
