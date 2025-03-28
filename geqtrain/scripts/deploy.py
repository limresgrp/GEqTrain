import os
from os.path import dirname
import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final
from typing import Tuple, Dict, Union
from pathlib import Path

import argparse
import pathlib
import logging
import yaml
import torch

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import numpy as np  # noqa: F401

from e3nn.util.jit import script

from geqtrain.train import Trainer
from geqtrain.utils import Config
from geqtrain.utils._global_options import _set_global_options

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph() # needed to compile einops. See https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops

CONFIG_KEY: Final[str] = "config"
TORCH_VERSION_KEY: Final[str] = "torch_version"
E3NN_VERSION_KEY: Final[str] = "e3nn_version"
CODE_COMMITS_KEY: Final[str] = "code_commits"
R_MAX_KEY: Final[str] = "r_max"
N_SPECIES_KEY: Final[str] = "n_species"
TYPE_NAMES_KEY: Final[str] = "type_names"
JIT_FUSION_STRATEGY: Final[str] = "_jit_fusion_strategy"
TF32_KEY: Final[str] = "allow_tf32"

_ALL_METADATA_KEYS = [
    CONFIG_KEY,
    TORCH_VERSION_KEY,
    E3NN_VERSION_KEY,
    R_MAX_KEY,
    N_SPECIES_KEY,
    TYPE_NAMES_KEY,
    JIT_FUSION_STRATEGY,
    TF32_KEY,
]


def _sanity_checks(config):
    if config.get("use_weight_norm", False):
        raise Exception("""Trying to compile with 'use_weight_norm' set to True.
                        This is not supported by PyTorch, as if you use PyTorch < 2, the scripting fails for a missing __name__ attribute in WeightNorm class.
                        If you use PyTorch version >= 2, the new implementation uses torch.nn.utils.parametrize.register_parametrization() which is not supported by scripting.
                        This is a lose-lose situation.""")


def _compile_for_deploy(model):
    model.eval()

    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)

    return model


def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    set_global_options: Union[str, bool] = "warn",
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]:
    r"""Load a deployed model.

    Args:
        model_path: the path to the deployed model's ``.pth`` file.

    Returns:
        model, metadata dictionary
    """
    metadata = {k: "" for k in _ALL_METADATA_KEYS}
    try:
        model = torch.jit.load(model_path, map_location=device, _extra_files=metadata)
    except RuntimeError as e:
        raise ValueError(
            f"{model_path} does not seem to be a deployed model file. Did you forget to deploy it using `nequip-deploy`? \n\n(Underlying error: {e})"
        )
    # Confirm its TorchScript
    assert isinstance(model, torch.jit.ScriptModule)
    # Make sure we're in eval mode
    model.eval()
    # Freeze on load:
    if freeze and hasattr(model, "training"):
        # hasattr is how torch checks whether model is unfrozen
        # only freeze if already unfrozen
        model = torch.jit.freeze(model)
    # Everything we store right now is ASCII, so decode for printing
    metadata = {k: v.decode("ascii") for k, v in metadata.items()}
    # Set up global settings:
    assert set_global_options in (True, False, "warn")
    if set_global_options:
        global_config_dict = {}
        global_config_dict["allow_tf32"] = bool(int(metadata[TF32_KEY]))
        # JIT strategy
        strategy = metadata.get(JIT_FUSION_STRATEGY, "")
        if strategy != "":
            strategy = [e.split(",") for e in strategy.split(";")]
            strategy = [(e[0], int(e[1])) for e in strategy]
        else:
            strategy = [("STATIC", 2)]
        global_config_dict[JIT_FUSION_STRATEGY] = strategy
        
        # call to actually set the global options
        _set_global_options(
            global_config_dict,
            warn_on_override=set_global_options == "warn",
        )
    return model, metadata


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Deploy and view information about previously deployed GEqTrain models."
    )
    # backward compat for 3.6
    if sys.version_info[1] > 6:
        required = {"required": True}
    else:
        required = {}
    parser.add_argument("--verbose", help="log level", default="INFO", type=str)
    subparsers = parser.add_subparsers(dest="command", title="commands", **required)

    build_parser = subparsers.add_parser("build", help="Build a deployment model")
    build_parser.add_argument(
        "-td",
        "--train-dir",
        help="Path to a working directory from a training session to deploy.",
        type=Path,
        default=None,
    )
    build_parser.add_argument(
        "-m",
        "--model",
        help="A deployed or pickled GEqTrain model to load. If omitted, defaults to `best_model.pth` in `train_dir`.",
        type=Path,
        default=None,
    )
    build_parser.add_argument(
        "-o",
        "--out-file",
        help="Output file for deployed model.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-e",
        "--extra-metadata",
        help="Additional key-value pairs to add to the metadata dictionary. Format: key=value. Use quotation marks for values with spaces, e.g., key=\"value with spaces\".",
        nargs='*',
        default=[]
    )

    args = parser.parse_args(args=args)

    logging.basicConfig(level=getattr(logging, args.verbose.upper()))

    # Do the defaults:
    if args.train_dir:
        if args.model is None:
            args.model = args.train_dir / "best_model.pth"
    if isinstance(args.model, str):
        model_path = Path(args.model)
    else:
        model_path = args.model
    
    logging.info(f"Loading {model_path} from training session...")
    config = Config.from_file(str(model_path.parent / "config.yaml"))

    _set_global_options(config)

    # -- load model --
    model, model_config = Trainer.load_model_from_training_session(
        model_path.parent, model_name=model_path.name, device="cpu", for_inference=True
    )

    _sanity_checks(model_config)

    # -- compile --
    model = _compile_for_deploy(model)
    logging.info("Compiled & optimized model.")

    # Deploy
    metadata: dict = {}

    metadata[R_MAX_KEY] = str(float(config["r_max"]))
    
    if int(torch.__version__.split(".")[1]) >= 11 and JIT_FUSION_STRATEGY in config:
        metadata[JIT_FUSION_STRATEGY] = ";".join("%s,%i" % e for e in config[JIT_FUSION_STRATEGY])
    metadata[TF32_KEY] = str(int(config["allow_tf32"]))
    metadata[CONFIG_KEY] = yaml.dump(dict(config))

    # Add extra metadata from command line arguments
    for item in args.extra_metadata:
        key, value = item.split('=')
        metadata[key] = value

    metadata = {k: v.encode("ascii") for k, v in metadata.items()}
    os.makedirs(dirname(args.out_file), exist_ok=True)
    torch.jit.save(model, args.out_file, _extra_files=metadata)

    return


if __name__ == "__main__":
    main()
