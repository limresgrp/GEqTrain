""" Adapted from https://github.com/mir-group/nequip
"""

""" Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import logging
import warnings
warnings.filterwarnings("ignore")
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


def main(args=None):
    args, config = parse_command_line(args)
    set_up_script_logger(config.verbose)
    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not (config.append):
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    fine_tune = config.get("fine_tune", False)
    if not found_restart_file or fine_tune:
        if fine_tune:
            logging.info("--- Fine-tuning model ---")
        elif not found_restart_file:
            logging.info("--- Starting fresh training ---")
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
            model, _ = model_from_config(config=config, initialize=True, dataset=trainer.dataset_train)
            logging.info("Successfully built the network!")

        # Equivar test
        if config.equivariance_test:
            logging.info("Running equivariance test...")
            model.eval()
            errstr = assert_AtomicData_equivariant(model, trainer.dataset_train[0])
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

    def check_for_config_updates():
        # compare old_config to config and update stop condition related arguments

        modifiable_params = ["max_epochs", "loss_coeffs", "learning_rate", "device", "metrics_components",
                         "noise", "use_dt", "wandb", "batch_size", "validation_batch_size", "train_dloader_n_workers",
                         "val_dloader_n_workers", "dloader_prefetch_factor", "dataset_num_workers", "inmemory", "transforms",
                        ]

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

    try:
        # trainer to dict: parsed dict is the used to instanciate Config
        restart_file = f"{config['root']}/{config['run_name']}/trainer.pth"
        old_config = load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=restart_file,
            enforced_format="torch",
        )
        if old_config.get("fine_tune", False):
            raise ValueError("Cannot restart training of a fine-tuning run")

        check_for_config_updates()

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


if __name__ == "__main__":
    main()
