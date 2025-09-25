# geqtrain/deploy/core.py

import os
import pathlib
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Final, Tuple, Union

import torch
from geqtrain.train.components.checkpointing import CheckpointHandler
from geqtrain.utils._global_options import set_global_options as set_global_options_func, apply_global_config
from e3nn.util.jit import script

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

def check_submodules_scripting(sequential_module_to_test):
    print(f"Found {len(sequential_module_to_test)} modules in model.")
    print("Attempting to script and freeze each submodule individually...")
    print("-----------------------------------------------------------")

    for i, _submodule in enumerate(sequential_module_to_test):
        submodule_name, submodule = _submodule
        print(f"\n[Submodule {i+1}/{len(sequential_module_to_test)}] Testing: {submodule_name} ({type(submodule).__name__})")

        try:
            # Ensure the submodule is in eval mode.
            # This is crucial as freezing is for inference.
            submodule.eval()

            # Step 1: Try to script the submodule
            print(f"  Attempting torch.jit.script(submodule)...")
            scripted_submodule = torch.jit.script(submodule)
            print(f"  torch.jit.script SUCCEEDED.")

            # Step 2: Try to freeze the scripted submodule
            print(f"  Attempting torch.jit.freeze(scripted_submodule)...")
            frozen_submodule = torch.jit.freeze(scripted_submodule)
            print(f"  torch.jit.freeze SUCCEEDED for {submodule_name}.")

        except RuntimeError as e:
            print(f"  RuntimeError for submodule {i+1} ({submodule_name}):")
            print(f"  Error: {e}")
        except Exception as e:
            print(f"  An UNEXPECTED error occurred for submodule {i+1} ({submodule_name}):")
            print(f"  Error: {e}")
        print("-----------------------------------------------------------")

def _sanity_checks(model, config):
    if config.get("use_weight_norm", False):
        raise Exception("""Trying to compile with 'use_weight_norm' set to True.
                        This is not supported by PyTorch, as if you use PyTorch < 2, the scripting fails for a missing __name__ attribute in WeightNorm class.
                        If you use PyTorch version >= 2, the new implementation uses torch.nn.utils.parametrize.register_parametrization() which is not supported by scripting.
                        This is a lose-lose situation.""")
    check_submodules_scripting(model)

def _compile_for_deploy(model):
    model.eval()
    if not isinstance(model, torch.jit.ScriptModule):
        model = script(model)
    return torch.jit.freeze(model)

def get_base_deploy_parser(parser=None):
    """Adds the common deployment arguments to a parser."""
    if parser is None:
        import argparse
        parser = argparse.ArgumentParser()
    
    # Arguments common to both GEqTrain and HEroBM
    parser.add_argument("-m", "--model", type=Path, default=None)
    parser.add_argument("-o", "--out-file", default="deployed.pth", type=Path)
    parser.add_argument(
        "-e", "--extra-metadata", nargs='*', default=[],
        help="Add key-value pairs to metadata. Format: key=value."
    )
    return parser

def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    set_global_options: Union[str, bool] = "warn",
    extra_metadata: Optional[Dict[str, str]] = None,
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]:
    r"""Load a deployed model.

    Args:
        model_path: the path to the deployed model's ``.pth`` file.

    Returns:
        model, metadata dictionary
    """
    metadata = {k: "" for k in _ALL_METADATA_KEYS}
    if extra_metadata is not None: metadata.update(extra_metadata)
    try:
        model = torch.jit.load(model_path, map_location=device, _extra_files=metadata)
    except RuntimeError as e:
        raise ValueError(f"{model_path} does not seem to be a deployed model file. Did you forget to deploy it using `deploy`? \n\n(Underlying error: {e})")

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
        if metadata.get(TF32_KEY):
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
        set_global_options_func(
            global_config_dict,
            warn_on_override=set_global_options == "warn",
        )
    return model, metadata

def verify_deployment(out_file, extra_metadata):
    try:
        _, verification_meta = load_deployed_model(out_file, device="cpu", extra_metadata=extra_metadata)
        for k in extra_metadata.keys():
            if k not in verification_meta.keys():
                logging.error(f"VERIFICATION FAILED: Loaded metadata is missing key {k}.")
                return
        logging.info("VERIFICATION SUCCEEDED: Metadata was loaded correctly.")
    except Exception as e:
        logging.error(f"VERIFICATION FAILED: An error occurred while trying to reload the model: {e}")

def build_deployment(
    model_path: Path,
    out_file: Path,
    config: dict,
    extra_metadata: Optional[Dict[str, str]] = None
):
    """Core logic to build a deployed model."""
    logging.info(f"Loading {model_path} from training session...")
    apply_global_config(config)

    # -- load model --
    model, model_config = CheckpointHandler.load_model_from_training_session(
        traindir=model_path.parent, 
        model_name=model_path.name, 
        device="cpu",
    )
    _sanity_checks(model, model_config)

    # -- compile --
    model = _compile_for_deploy(model)
    logging.info("Compiled & optimized model.")

    # -- build metadata --
    metadata = {}
    metadata[R_MAX_KEY] = str(float(config["r_max"]))
    metadata[TF32_KEY] = str(int(config["allow_tf32"]))
    metadata[CONFIG_KEY] = yaml.safe_dump(dict(config))
    # ... other generic metadata ...
    
    # Add any extra metadata passed from the command line
    if extra_metadata:
        metadata.update(extra_metadata)

    # -- save and verify --
    encoded_metadata = {k: v.encode("ascii") for k, v in metadata.items()}
    os.makedirs(out_file.parent, exist_ok=True)
    torch.jit.save(model, out_file, _extra_files=encoded_metadata)
    logging.info(f"Successfully deployed model to {out_file}")

    # Your verification step here...
    logging.info("Verifying saved model...")
    verify_deployment(out_file, extra_metadata)
