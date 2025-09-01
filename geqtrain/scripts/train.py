""" Train a network. """
import logging
import argparse
import shutil
import os
import torch.distributed as dist
from pathlib import Path
from os.path import isdir
from geqtrain.utils import Config, load_file
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.train.trainer import Trainer
from geqtrain.utils._global_options import _set_global_options


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Train (or restart training of) a model.")
    parser.add_argument("config", help="YAML file configuring the model, dataset, and other options")
    parser.add_argument("-d", "--device", help="Device to run on (e.g. 'cpu', 'cuda:0'). Overrides automatic DDP device assignment.")
    parser.add_argument("--ddp", action="store_true", help="Use torch DistributedDataParallel. Assumes a torchrun launch.")
    parser.add_argument("-ma", "--master-addr", help="MASTER_ADDR for DDP. Defaults to 'localhost'.")
    parser.add_argument("-mp", "--master-port", help="MASTER_PORT for DDP. Defaults to a random free port.")
    parser.add_argument("-u", "--find-unused-parameters", action="store_true", help="Enable DDP's find_unused_parameters flag. Useful for models with conditional logic.")

    args = parser.parse_args(args=args)
    config = Config.from_file(args.config)

    # Update config with command-line arguments
    config['ddp'] = args.ddp
    if args.device:
        config['device'] = args.device
    if args.master_addr:
        config['master_addr'] = args.master_addr
    if args.master_port:
        config['master_port'] = args.master_port
    if args.find_unused_parameters:
        config['find_unused_parameters'] = args.find_unused_parameters
        
    return config

def check_for_config_updates(new_config):
    """
    Load a saved config, merge new parameters, enforce current runtime settings,
    and return a clean, final Config object.
    """
    restart_file = f"{new_config['root']}/{new_config['run_name']}/trainer.pth"
    saved_state = load_file(
        filename=restart_file,
        supported_formats=dict(torch=["pt", "pth"]),
        enforced_format="torch",
        map_location="cpu",
        weights_only=False,
    )

    # Start with the dictionary from the saved run
    final_config_dict = saved_state['config'].as_dict()
    new_config_dict = new_config.as_dict()
    
    # 1. Update user-modifiable parameters
    modifiable_params = [
        "max_epochs", "learning_rate", "loss_coeffs", "metrics_components", "log_batch_freq",
        "use_ema", "wandb", "dataset_list", "validation_dataset_list", "test_dataset_list",
        "batch_size", "validation_batch_size", "dataloader_num_workers",
    ]
    logging.info("Checking for updated user-modifiable parameters...")
    for key in new_config_dict:
        if key in final_config_dict and new_config_dict[key] != final_config_dict[key]:
            if key in modifiable_params:
                logging.info(f'Updating parameter "{key}" from `{final_config_dict[key]}` to `{new_config_dict[key]}`')
                final_config_dict[key] = new_config_dict[key]
            else:
                raise ValueError(f'Parameter "{key}" is not user-modifiable and cannot be changed during restart.')

    # 2. Enforce critical runtime parameters from the new command
    # This ensures DDP status, device, etc., are always from the new command.
    runtime_params = ['ddp', 'device', 'find_unused_parameters']
    logging.info("Enforcing current runtime parameters...")
    for key in runtime_params:
        if key in new_config_dict and new_config_dict[key] is not None:
            if final_config_dict.get(key) != new_config_dict[key]:
                 logging.info(f"Overwriting runtime parameter '{key}' with new value '{new_config_dict[key]}'")
            final_config_dict[key] = new_config_dict[key]
        else:
            # If not specified in the new command, remove the old key
            if key in final_config_dict:
                logging.info(f"Removing stale runtime parameter '{key}' from loaded config.")
                final_config_dict.pop(key)
    
    # 3. Create a brand new, clean Config object from the final dictionary
    final_config = Config.from_dict(final_config_dict)
    final_config.filepath = new_config.filepath
            
    return final_config

def main(args=None):
    """The main entry point for training."""
    # 1. Parse config from current command line
    config = parse_command_line(args)
    set_up_script_logger(config.verbose)

    # 2. Determine if it's a restart and create the final config
    is_restart = isdir(f"{config.root}/{config.run_name}")
    if is_restart:
        logging.info("Found existing run directory, attempting to restart training.")
        final_config = check_for_config_updates(config)
        final_config['restart'] = True
    else:
        logging.info("Starting a fresh training run.")
        final_config = config
        final_config['restart'] = False
    
    # 3. Set up environment and global options
    if final_config.get('wandb') and final_config.get('ddp'):
        os.environ["WANDB_START_METHOD"] = "thread"
    _set_global_options(final_config)

    trainer = None
    try:
        # 4. Initialize and run the Trainer
        trainer = Trainer(config=final_config)
        
        if trainer.dist.is_master and not is_restart:
            config_path = trainer.output.generate_file("config.yaml")
            shutil.copyfile(Path(config.filepath).resolve(), config_path)
            logging.info(f"Copied config file to {config_path}")

        trainer.train()

    except KeyboardInterrupt:
        logging.warning("Training manually interrupted. Initiating synchronized shutdown.")
        raise
    
    except Exception as e:
        logging.error("An uncaught error occurred during training:")
        logging.exception(e)

        if "find_unused_parameters" in str(e):
            logging.error(
                "\n HINT: This error often occurs in DDP when your model has parameters that "
                "are not used in the forward pass. If this is intentional, "
                "you can resolve this by adding `find_unused_parameters: true` to your YAML config file "
                "or calling the training script with option '--find-unused-parameters' ('-u')."
            )
        raise e

    finally:
        # 5. Unified shutdown sequence for all exit scenarios
        is_ddp = trainer is not None and trainer.dist.is_distributed and dist.is_initialized()
        if is_ddp: dist.barrier()

        if trainer is not None and trainer.dist.is_master:
            logging.info("Master rank performing final cleanup...")
            if trainer.config.get('wandb'):
                import wandb
                if wandb.run is not None:
                    wandb.finish()

        if is_ddp:
            dist.barrier()
            logging.info(f"Rank {trainer.dist.rank} cleaning up distributed process group.")
            dist.destroy_process_group()
        
        logging.info("Script shutdown complete.")

if __name__ == "__main__":
    main()