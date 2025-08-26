""" Train a network. """
import logging
import argparse
import shutil
import os
import torch.distributed as dist
from pathlib import Path
from os.path import isdir
from geqtrain.utils import Config, load_file, _set_global_options
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.train import Trainer


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Train (or restart training of) a model.")
    parser.add_argument("config", help="YAML file configuring the model, dataset, and other options")
    parser.add_argument("-d", "--device", help="Device to run on (e.g. 'cpu', 'cuda:0'). Overrides automatic DDP device assignment.")
    parser.add_argument("--ddp", action="store_true", help="Use torch DistributedDataParallel. Assumes a torchrun launch.")
    parser.add_argument("-ma", "--master-addr", help="MASTER_ADDR for DDP. Defaults to 'localhost'.")
    parser.add_argument("-mp", "--master-port", help="MASTER_PORT for DDP. Defaults to a random free port.")

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
        
    return config

def check_for_config_updates(new_config):
    """
    Load the config from a saved training session and merge in any modifiable
    parameters from the new config file.
    """
    restart_file = f"{new_config['root']}/{new_config['run_name']}/trainer.pth"
    saved_state = load_file(supported_formats=dict(torch=["pt", "pth"]), filename=restart_file, enforced_format="torch")
    
    final_config = saved_state['config']
    
    modifiable_params = [
        "max_epochs", "learning_rate", "loss_coeffs", "metrics_components",
        "log_batch_freq", "use_ema", "wandb"
    ]
    
    logging.info("Checking for updated parameters...")
    for key in modifiable_params:
        if key in new_config and new_config[key] != final_config.get(key):
            logging.info(f'Updating parameter "{key}" from `{final_config.get(key)}` to `{new_config[key]}`')
            final_config[key] = new_config[key]
            
    return final_config

def main(args=None):
    """The main entry point for training."""
    config = parse_command_line(args)
    
    if config.get('wandb') and config.get('ddp'):
        os.environ["WANDB_START_METHOD"] = "thread"

    set_up_script_logger(config.verbose)
    
    trainer = None
    try:
        if config.get('ddp', False):
            if 'MASTER_ADDR' not in os.environ: os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
            if 'MASTER_PORT' not in os.environ:
                port = config.get('master_port')
                if port is None:
                    import socket
                    from contextlib import closing
                    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                        s.bind(('', 0)); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        port = s.getsockname()[1]
                os.environ['MASTER_PORT'] = str(port)
                logging.info(f"Using MASTER_ADDR={os.environ['MASTER_ADDR']} and MASTER_PORT={os.environ['MASTER_PORT']}")

        is_restart = isdir(f"{config.root}/{config.run_name}")
        
        if is_restart:
            logging.info("Found existing run directory, attempting to restart training.")
            final_config = check_for_config_updates(config)
            final_config['restart'] = True
        else:
            logging.info("Starting a fresh training run.")
            final_config = config
            final_config['restart'] = False
            if 'fine_tune' in final_config: logging.info(f"Fine-tuning from: {final_config['fine_tune']}")

        _set_global_options(final_config)
        trainer = Trainer(config=final_config)
        
        if trainer.dist.is_master and not is_restart:
            config_path = trainer.output.generate_file("config.yaml")
            shutil.copyfile(Path(config.filepath).resolve(), config_path)
            logging.info(f"Copied config file to {config_path}")

        trainer.train()

    except KeyboardInterrupt:
        logging.warning("Training manually interrupted. Initiating synchronized shutdown.")
    
    except Exception as e:
        logging.error("An uncaught error occurred during training:")
        logging.exception(e)
        # Re-raise the exception to ensure the script exits with a non-zero code,
        # but only after the `finally` block has run.
        raise e

    finally:
        # This unified shutdown sequence runs for normal exit, interrupts, and errors.
        is_ddp = trainer is not None and trainer.dist.is_distributed and dist.is_initialized()

        if is_ddp:
            # Step 1: All processes wait here. Ensures none rush ahead.
            dist.barrier()

        # Step 2: Master-rank performs its final exclusive tasks.
        if trainer is not None and trainer.dist.is_master:
            logging.info("Master rank performing final cleanup...")
            # Cleanly shut down wandb
            if trainer.config.get('wandb'):
                import wandb
                if wandb.run is not None:
                    wandb.finish()

        # Step 3: All processes are now free. They will all shut down together.
        if is_ddp:
            # The barrier isn't strictly necessary here but can prevent race conditions
            # with the final log messages.
            dist.barrier()
            logging.info(f"Rank {trainer.dist.rank} cleaning up distributed process group.")
            dist.destroy_process_group()
        
        logging.info("Script shutdown complete.")

if __name__ == "__main__":
    main()