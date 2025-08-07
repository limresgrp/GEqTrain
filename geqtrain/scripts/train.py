""" Adapted from https://github.com/mir-group/nequip
"""

""" Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import logging
import os
from typing import Tuple
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from geqtrain.train.distributed_training_utils import get_distributed_env
from geqtrain.utils.test import assert_AtomicData_equivariant
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.utils._global_options import _set_global_options
from geqtrain.utils import Config, load_file
from geqtrain.model import model_from_config
from pathlib import Path
from os.path import isdir
import logging
import argparse
import shutil
from geqtrain.train import (
    setup_distributed_training,
    cleanup_distributed_training,
    configure_dist_training,
    instanciate_train_val_dsets,
    load_trainer_and_model,
)

def parse_command_line(args=None) -> Tuple[argparse.Namespace, Config]:
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
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use torch DistributedDataParallel via torchrun."
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

    config = Config.from_file(args.config)

    flags = ("device", "equivariance_test", "grad_anomaly_mode")
    config.update({flag: getattr(args, flag)
                  for flag in flags if getattr(args, flag) is not None})
    config.update({"use_dt": args.ddp})

    return args, config

def main(args=None):
    
    # --- WORKAROUND FOR NCCL P2P ISSUES ON NON-SLURM SYSTEMS ---
    # Check if we are running in a SLURM environment by looking for a SLURM-specific variable.
    # if 'SLURM_JOB_ID' not in os.environ:
        # If not in SLURM, we are likely on a local workstation/server that might have
        # driver or IOMMU issues interfering with NCCL's P2P communication.
        # We disable P2P to force a more robust communication path via system memory.
        # This is NOT set for SLURM jobs, allowing them to use the optimal path if the
        # cluster is configured correctly.
        # logging.info("Not in a SLURM environment, setting NCCL_P2P_DISABLE=1 as a workaround for potential hangs.")
        # os.environ['NCCL_P2P_DISABLE'] = '1'
    # --- END WORKAROUND ---

    args, config = parse_command_line(args)
    set_up_script_logger(config.verbose)

    # --- REFACTORED DDP-SAFE RESTART LOGIC ---

    # 1. Initialize the process group first. Communication is essential for a clean restart.
    rank = 0
    world_size = 1
    if config.use_dt:
        logging.info(f"[Rank {rank}] Setting up distributed training...")
        configure_dist_training(args)
        setup_distributed_training()
        logging.info(f"[Rank {rank}] Distributed training setup complete.")
        rank, world_size, _ = get_distributed_env()

    # 2. Only the master process (Rank 0) checks the file system for a restart checkpoint.
    checkpoint_info = {} # Use a dictionary to hold restart data
    if rank == 0:
        found_restart_file = isdir(f"{config.root}/{config.run_name}")
        if found_restart_file:
            logging.info(f"[Rank 0] Found restart file at {config.root}/{config.run_name}. Preparing to resume.")
            # The master loads the updated config and progress from the checkpoint file.
            updated_config, progress_config = check_for_config_updates(config)
            checkpoint_info = {
                "is_restart": True,
                "config_dict": updated_config.as_dict(),
                "progress_config_dict": progress_config.as_dict()
            }
        else:
            checkpoint_info = {"is_restart": False}

    # 3. Broadcast the decision and data from Rank 0 to all other processes.
    if config.use_dt:
        import torch.distributed as dist
        # Use a list for broadcast_object_list, which is robust for complex objects
        broadcast_list = [checkpoint_info]
        dist.broadcast_object_list(broadcast_list, src=0)
        checkpoint_info = broadcast_list[0]

    # 4. All processes now act on the synchronized information.
    if checkpoint_info.get("is_restart", False):
        # All ranks now have the correct, updated config from the master.
        logging.info(f"[Rank {rank}] Resuming training from checkpoint.")
        config = Config(checkpoint_info["config_dict"])
        progress_config = Config(checkpoint_info["progress_config_dict"])
        func = partial(restart, progress_config=progress_config.as_dict())
    else:
        logging.info(f"[Rank {rank}] Starting a fresh training run.")
        func = fresh_start

    _set_global_options(config)

    # 5. Execute the determined function (fresh_start or restart) on all ranks.
    # We pass the config dictionary, which is now consistent across all processes.
    func(rank=rank, world_size=world_size, config=config.as_dict())

    return


def fresh_start(rank: int, world_size: int, config: dict):
    try:
        import torch
        from geqtrain.train.distributed_training_utils import get_distributed_env
        _rank, _ws, local_rank = get_distributed_env()
        logging.info(f"--> [Rank {rank}] START. Global Rank: {rank}, Local Rank: {local_rank}, "
                     f"Device: {torch.cuda.current_device()}, "
                     f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        logging.info(f"[Rank {rank}] Starting fresh_start function.")
        
        assert isinstance(config, dict), f"config must be of type Dict. It is of type {type(config)}"
        config = Config.from_dict(config)
        _set_global_options(config)

        # Initialize the process group FIRST. Communication is needed for the fix.
        if config.use_dt:
            logging.info(f"[Rank {rank}] Setting up distributed training...")
            setup_distributed_training()
            logging.info(f"[Rank {rank}] Distributed training setup complete.")

        # This will be a fast read operation for everyone.
        logging.info(f"[Rank {rank}] Loading dataset from cache...")
        train_dataset, validation_dataset = instanciate_train_val_dsets(config)
        logging.info(f"[Rank {rank}] Dataset loaded successfully.")

        logging.info(f"[Rank {rank}] Loading trainer and model...")
        trainer, model = load_trainer_and_model(rank, world_size, config)
        logging.info(f"[Rank {rank}] Trainer and model loaded.")
        
        # Copy conf file in results folder
        # Only the master process should perform file system operations.
        if trainer.is_master:
            shutil.copyfile(Path(config.filepath).resolve(), trainer.config_path)

        config.update(trainer.params)
        logging.info(f"[Rank {rank}] Initializing dataset...")
        trainer.init_dataset(config, train_dataset, validation_dataset)
        logging.info(f"[Rank {rank}] Dataset initialized.")

        # = Update config with dataset-related params = #
        config.update(trainer.dataset_params)

        # = Build model =
        if model is None:
            logging.info(f"[Rank {rank}] Building the network...")
            model, _ = model_from_config(config=config, initialize=True, dataset=trainer.dataset_train)
            logging.info(f"[Rank {rank}] Successfully built the network!")

        logging.info(f"[Rank {rank}] Initializing model in trainer...")
        trainer.init_model(model=model)
        logging.info(f"[Rank {rank}] Model initialized in trainer.")
        
        trainer.update_kwargs(config)

        # ... (equivariance test logic) ...

        # Run training
        logging.info(f"[Rank {rank}] Saving initial state...")
        trainer.save()
        logging.info(f"[Rank {rank}] Starting training loop...")
        trainer.train()
        logging.info(f"[Rank {rank}] Training loop finished.")

    except KeyboardInterrupt:
        logging.info(f"[Rank {rank}] Process manually stopped!")
    except Exception as e:
        import traceback
        logging.error(f"[Rank {rank}] An exception occurred: {e}")
        logging.error(f"[Rank {rank}] Full traceback:\n{traceback.format_exc()}")
        raise e
    finally:
        try:
            if config.get("use_dt", False):
                cleanup_distributed_training()
        except:
            pass

def restart(rank, world_size, config: dict, progress_config: dict):
    try:
        assert isinstance(config, dict), f"config must be of type Dict. It is of type {type(config)}"
        config = Config.from_dict(config)
        assert isinstance(progress_config, dict), f"progress_config must be of type Dict. It is of type {type(progress_config)}"
        progress_config = Config.from_dict(progress_config)

        train_dataset, validation_dataset = instanciate_train_val_dsets(config)

        if config.use_dt:
            setup_distributed_training()

        trainer, model = load_trainer_and_model(rank, world_size, progress_config, is_restart=True)
        trainer.init_dataset(config, train_dataset, validation_dataset)
        trainer.init_model(model=model)
        trainer.load_state_dicts_for_restart(progress_config)
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
            if config.get("use_dt", False):
                cleanup_distributed_training()
        except:
            pass
    return

def check_for_config_updates(config):
    # compare old_config to config and update stop condition related arguments
    restart_file = f"{config['root']}/{config['run_name']}/trainer.pth"
    old_config = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )
    if old_config.get("fine_tune", False):
        raise ValueError("Cannot restart training of a fine-tuning run")

    modifiable_params = ["max_epochs", "loss_coeffs", "learning_rate", "device", "metrics_components", "log_batch_freq", "use_ema",
                        "noise", "use_dt", "wandb", "batch_size", "validation_batch_size", "train_dloader_n_workers", "heads", "avg_num_neighbors",
                        "val_dloader_n_workers", "dloader_prefetch_factor", "dataset_num_workers", "inmemory", "transforms", "use_grokfast",
                        "report_init_validation", "metrics_key", "max_gradient_norm", "dropout", "dropout_edges", "optimizer_params", "head_wds"
                    ] # todo: "num_types" should be added here after moving binning functionality away from dataset creation

    for k,v in config.items():
        if v != old_config.get(k, ""):
            if k in modifiable_params:
                logging.info(f'Update "{k}" from {old_config[k]} to {v}')
                old_config[k] = v
            elif k.startswith("early_stop"):
                logging.info(f'Update "{k}" from {old_config[k]} to {v}')
                old_config[k] = v
            elif k == 'filepath':
                assert Path(config[k]).resolve() == Path(old_config[k]).resolve()
                old_config[k] = v
            elif k in ['dataset_list', 'validation_dataset_list']:
                assert isinstance(v, list), "dataset_list/validation_dataset_list must be of type list"
                assert isinstance(old_config[k], list), "dataset_list/validation_dataset_list must be of type list"
                assert len(v) == 1, "for now only 1 dataset under dataset_list/validation_dataset_list is allowed"
                assert len(old_config[k]), "for now only 1 dataset under dataset_list/validation_dataset_list is allowed"
                new_dset_and_kwargs = v[0]
                old_dset_and_kwargs = old_config[k][0]
                for dlist_k in new_dset_and_kwargs.keys():
                    if dlist_k in modifiable_params:
                        continue
                    if new_dset_and_kwargs[dlist_k] != old_dset_and_kwargs[dlist_k]:
                        raise ValueError(f'Key "{k}" is different in config and the result trainer.pth file. Please double check')
            elif isinstance(v, type(old_config.get(k, ""))):
                raise ValueError(f'Key "{k}" is different in config and the result trainer.pth file. Please double check')

    config          = Config(old_config, exclude_keys=["state_dict", "progress"])
    progress_config = Config(old_config)
    return config, progress_config


if __name__ == "__main__":
    main()
