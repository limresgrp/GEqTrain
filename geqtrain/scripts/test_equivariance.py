""" Train a network."""
import logging
import argparse
import warnings
import torch

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from os.path import isdir

from geqtrain.model import model_from_config
from geqtrain.utils import Config
from geqtrain.data import dataset_from_config
from geqtrain.utils import load_file
from geqtrain.utils.test import assert_AtomicData_equivariant
from geqtrain.utils._global_options import _set_global_options
from geqtrain.scripts._logger import set_up_script_logger

default_config = dict(
    wandb=False,
    dataset_statistics_stride=1,
    default_dtype="float32",
    train_on_delta=False,
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    equivariance_test=False,
    grad_anomaly_mode=False,
    fine_tune=False,
    append=False,
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)


def main(args=None, running_as_script: bool = True):
    config = parse_command_line(args)

    if running_as_script:
        set_up_script_logger(config.get("log", None), config.verbose)
    logger = logging.getLogger("geqtrain-test-equivariance")
    logger.setLevel(logging.INFO)

    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not (config.append or config.fine_tune):
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    test_equivariance(config, logger)

    return


def parse_command_line(args=None):
    parser = argparse.ArgumentParser(
        description="Check equivariance of the model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        '--cartesian-fields',
        nargs='+',
        default=[],
        help='List of fields resembling Cartesian coordinates, necessitating equivariant responses to translations and rotations.')
    args = parser.parse_args(args=args)

    config = Config.from_file(args.config, defaults=default_config)
    config["cartesian_fields"] = args.cartesian_fields
    config["grad_anomaly_mode"] = True

    return config


def test_equivariance(config, logger):
    _set_global_options(config)

    # = Load the dataset =
    dataset = dataset_from_config(config, prefix="dataset")
    logger.info(f"Successfully loaded the data set of type {dataset}...")

    # = Build model =
    final_model = model_from_config(
        config=config, initialize=True, dataset=dataset
    )
    logger.info("Successfully built the network...")

    # by doing this here we check also any keys custom builders may have added
    _check_old_keys(config)

    # Equivar test
    
    n_train: int = len(dataset)
    final_model.eval()
    final_model.to(config.get('device', 'cpu'))
    indexes = torch.randperm(n_train)[:1]
    errstr = assert_AtomicData_equivariant(
        final_model,
        [dataset[i] for i in indexes],
        config.get("cartesian_fields", []),
    )

    logger.info(
        "Equivariance test passed; equivariance errors:\n"
        "   Errors are in real units, where relevant.\n"
        "   Please note that the large scale of the typical\n"
        "   shifts to the (atomic) energy can cause\n"
        "   catastrophic cancellation and give incorrectly\n"
        "   the equivariance error as zero for those fields.\n"
        f"{errstr}"
    )
    del errstr, indexes, n_train

    return

def _check_old_keys(config) -> None:
    """check ``config`` for old/depricated keys and emit corresponding errors/warnings"""
    # compile_model
    k = "compile_model"
    if k in config:
        if config[k]:
            raise ValueError("the `compile_model` option has been removed")
        else:
            warnings.warn("the `compile_model` option has been removed")


if __name__ == "__main__":
    main(running_as_script=True)