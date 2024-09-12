""" Adapted from https://github.com/mir-group/nequip
"""

import inspect
from importlib import import_module
from typing import Dict, List
from os import listdir
from os.path import isdir, isfile, join

from torch.utils.data import ConcatDataset
from geqtrain import data
from geqtrain.data import (
    AtomicDataDict,
    register_fields,
)

from geqtrain.utils import (
    instantiate,
    get_w_prefix,
    Config
    )


def dataset_from_config(config, prefix: str = "dataset") -> ConcatDataset:
    """
    1) get dset type
    2) get data_path or data_file
    3) registers fields to read/expose-in-code data at 2) in the correct form
    4) instanciate dataset(s)
        4.1) if many dsets -> cat them
    5) return

    initialize dataset based on a config instance

    It needs dataset type name (case insensitive),
    and all the parameters needed in the constructor.

    Args:

    config (dict, geqtrain.utils.Config): dict/object that store all the parameters
    prefix (str): Optional. The prefix of all dataset parameters

    Returns:
        torch.utils.data.ConcatDataset: dataset
    """

    instances = []
    dataset_id_offset = 0
    config_dataset_list: List[Dict] = config.get(f"{prefix}_list", [config])
    for dataset_id, _config_dataset in enumerate(config_dataset_list):
        config_dataset_type = _config_dataset.get(prefix, None)
        if config_dataset_type is None:
            raise KeyError(f"Dataset with prefix `{prefix}` isn't present in this config!")

        # looks for dset type specified in yaml if present (dataset_list/dataset: {$class_name} )
        if inspect.isclass(config_dataset_type):
            class_name = config_dataset_type
        else:
            try:
                module_name = ".".join(config_dataset_type.split(".")[:-1])
                class_name = ".".join(config_dataset_type.split(".")[-1:])
                class_name = getattr(import_module(module_name), class_name)
            except Exception:
                # default class defined in geqtrain.data or geqtrain.dataset
                dataset_name = config_dataset_type.lower()

                class_name = None
                for k, v in inspect.getmembers(data, inspect.isclass):
                    if k.endswith("Dataset"):
                        if k.lower() == dataset_name:
                            class_name = v
                        if k[:-7].lower() == dataset_name:
                            class_name = v
                    elif k.lower() == dataset_name:
                        class_name = v

        if class_name is None:
            raise NameError(f"dataset type {dataset_name} does not exists")

        inpup_key = f"{prefix}_input" # get input path
        assert inpup_key in _config_dataset, f"Missing {inpup_key} key in dataset config file."
        f_name = _config_dataset.get(inpup_key)
        if isdir(f_name): # can be dir
            dataset_file_names = [join(f_name, f) for f in listdir(f_name) if isfile(join(f_name, f))]
        else:
            dataset_file_names = [f_name] # can be 1 file

        _config: dict = config.as_dict() if isinstance(config, Config) else config
        _config.update(_config_dataset)

        # if dataset r_max is not found, use the universal r_max
        eff_key = "extra_fixed_fields"
        prefixed_eff_key = f"{prefix}_{eff_key}"

        _config[prefixed_eff_key] = get_w_prefix(
            eff_key, {},
            prefix=prefix,
            arg_dicts=_config
        )

        _config[prefixed_eff_key][AtomicDataDict.R_MAX_KEY] = get_w_prefix(
            AtomicDataDict.R_MAX_KEY,
            prefix=prefix,
            arg_dicts=[_config[prefixed_eff_key], _config],
        )

        for dataset_file_name in dataset_file_names:
            _config[AtomicDataDict.DATASET_INDEX_KEY] = dataset_id + dataset_id_offset # dataset id
            dataset_id_offset += 1
            _config[f"{prefix}_file_name"] = dataset_file_name

            # Register fields:
            # This might reregister fields, but that's OK:
            instantiate(register_fields, all_args=_config)

            instance, _ = instantiate(
                class_name, # dataset selected to be instanciated
                prefix=prefix, # look for this prefix word in yaml to select get the params for the ctor
                positional_args={},
                optional_args=_config,
            )

            instances.append(instance)

    return ConcatDataset(instances)
