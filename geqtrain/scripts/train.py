""" Train a network. """
import logging
import argparse
import os
import torch.distributed as dist
from geqtrain.utils import load_config
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.train.trainer import Trainer


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(description="Train (or restart training of) a model.")
    parser.add_argument("config", help="YAML file configuring the model, dataset, and other options")
    parser.add_argument("-d", "--device", help="Device to run on (e.g. 'cpu', 'cuda:0'). Overrides automatic DDP device assignment.")
    parser.add_argument("--ddp", action="store_true", help="Use torch DistributedDataParallel. Assumes a torchrun launch.")
    parser.add_argument("-ma", "--master-addr", help="MASTER_ADDR for DDP. Defaults to 'localhost'.")
    parser.add_argument("-mp", "--master-port", help="MASTER_PORT for DDP. Defaults to a random free port.")
    parser.add_argument("-u", "--find-unused-parameters", action="store_true", help="Enable DDP's find_unused_parameters flag. Useful for models with conditional logic.")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="Hydra override string (e.g. 'model.num_layers=3'). Can be provided multiple times.",
    )

    args = parser.parse_args(args=args)
    config = load_config(args.config, overrides=args.override)

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

def main(args=None):
    """The main entry point for training."""
    # 1. Parse config from current command line
    config = parse_command_line(args)
    set_up_script_logger(config.verbose)

    # 2. Initialize Trainer
    trainer = None
    try:
        # 3. Initialize and run Trainer
        # The Trainer's __init__ will handle:
        #   a. Initializing DDP.
        #   b. Rank 0 checking the directory.
        #   c. Rank 0 broadcasting the result (`is_restart`).
        #   d. All ranks loading or creating the final_config based on the broadcasted status.
        trainer = Trainer(config=config)
        final_config = trainer.config # Get the final, resolved config from the trainer
        
        # 4. Set up environment
        if final_config.get('wandb') and final_config.get('ddp'):
            os.environ["WANDB_START_METHOD"] = "thread"

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
