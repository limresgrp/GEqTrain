""" Adapted from https://github.com/mir-group/nequip
"""

""" Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import warnings
from geqtrain.utils.test import assert_AtomicData_equivariant
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.utils._global_options import _set_global_options
from geqtrain.data import dataset_from_config
from geqtrain.utils import Config, load_file
from geqtrain.model import model_from_config
from pathlib import Path
from os.path import isdir
import torch.distributed as dist
import logging
import argparse
import os
import shutil
from typing import Dict, Optional
import numpy as np  # noqa: F401


warnings.filterwarnings("ignore")



def setup_process(rank, world_size):
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup(rank):
    logging.info(f"Rank: {rank} | Destroying process group")
    dist.destroy_process_group()


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
        if config.use_dt:
            raise NotImplementedError(
                "Could not restart training in Distributed Training yet.")
        func = restart

    if config.use_dt:
        # Manually set the environment variables for multi-GPU setup
        world_size = args.world_size
        # Number of GPUs/processes to use
        os.environ['WORLD_SIZE'] = str(world_size)
        if args.master_addr:
            os.environ['MASTER_ADDR'] = str(args.master_addr)
        if args.master_port:
            os.environ['MASTER_PORT'] = str(args.master_port)

        # Spawn one process per GPU
        import torch.multiprocessing as mp
        mp.spawn(func, args=(world_size, config.as_dict(),),
                 nprocs=world_size, join=True)
    else:
        func(0, 1, config.as_dict())

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
            raise argparse.ArgumentError(
                "Cannot specify device when using Distributed Training")
        if args.equivariance_test:
            raise argparse.ArgumentError(
                "You can run Equivariance Test on single CPU/GPU only")

    config = Config.from_file(args.config)

    flags = ("device", "equivariance_test", "grad_anomaly_mode")
    config.update({flag: getattr(args, flag)
                  for flag in flags if getattr(args, flag) is not None})
    config.update({"use_dt": args.world_size is not None})

    return args, config


def fresh_start(rank, world_size, config):
    try:
        config = Config.from_dict(config)
        _set_global_options(config)

        if config.use_dt:
            # Setup the process for distributed training
            setup_process(rank, world_size)

        trainer, model = load_trainer_and_model(rank, world_size, config)
        # Copy conf file in results folder
        shutil.copyfile(Path(config.filepath).resolve(), trainer.config_path)

        config.update(trainer.params)

        # sets both train and val dsets
        trainer.set_dataset(*load_dataset(config))
        trainer.set_dataloader(config)

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
    except KeyboardInterrupt:
        logging.info("Process manually stopped!")
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        if config.use_dt:
            cleanup(rank)

    return


def restart(rank, world_size, config):
    try:
        # load the dictionary
        restart_file = f"{config['root']}/{config['run_name']}/trainer.pth"
        dictionary = load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=restart_file,
            enforced_format="torch",
        )

        # compare dictionary to config and update stop condition related arguments
        for k in config.keys():
            if config[k] != dictionary.get(k, ""):
                # modifiable things if restart
                if k in ["max_epochs", "loss_coeffs", "learning_rate", "device",
                         "metrics_components", "noise", "use_dt", "wandb", "batch_size", "validation_batch_size"]:
                    dictionary[k] = config[k]
                    logging.info(f'Update "{k}" to {dictionary[k]}')
                elif k.startswith("early_stop"):
                    dictionary[k] = config[k]
                    logging.info(f'Update "{k}" to {dictionary[k]}')
                elif isinstance(config[k], type(dictionary.get(k, ""))):
                    raise ValueError(
                        f'Key "{k}" is different in config and the result trainer.pth file. Please double check'
                    )

        config = Config(dictionary, exclude_keys=["state_dict", "progress"])
        _set_global_options(config)

        if config.use_dt:
            # Setup the process for distributed training
            setup_process(rank, world_size)

        trainer, model = load_trainer_and_model(
            rank, world_size, config, dictionary=dictionary, is_restart=True)
        trainer.set_dataset(*load_dataset(config))
        trainer.set_dataloader(config)

        trainer.init(model=model)

        # Store any updated config information in the trainer
        trainer.update_kwargs(config)

        # Train
        trainer.save()
        trainer.train()
    except KeyboardInterrupt:
        logging.info("Process manually stopped!")
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        try:
            if dictionary.get("use_dt", False):
                cleanup(rank)
        except:
            pass
    return


def load_dataset(config):
    dataset = dataset_from_config(config, prefix="dataset")
    logging.info(f"Successfully loaded the data set of type {dataset}...")
    try:
        validation_dataset = dataset_from_config(
            config, prefix="validation_dataset")
        logging.info(
            f"Successfully loaded the validation data set of type {validation_dataset}...")
    except KeyError:  # It couldn't be found in yaml
        validation_dataset = None
    return dataset, validation_dataset


def load_trainer_and_model(rank: int, world_size: int, config: Config, dictionary: Optional[Dict] = None, is_restart=False):
    if dictionary is None:
        dictionary = dict(config)
    if config.use_dt:
        dictionary.update({
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
            trainer, model = DistributedTrainerWandB.from_dict(dictionary)
        else:
            from geqtrain.train import TrainerWandB
            trainer, model = TrainerWandB.from_dict(dictionary)
    else:
        if config.use_dt:
            from geqtrain.train import DistributedTrainer
            trainer, model = DistributedTrainer.from_dict(dictionary)
        else:
            from geqtrain.train import Trainer
            trainer, model = Trainer.from_dict(dictionary)
    return trainer, model


if __name__ == "__main__":
    main(running_as_script=True)
