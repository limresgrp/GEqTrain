""" Adapted from https://github.com/mir-group/nequip
"""

"""
Class to holde a bunch of hyperparameters associate with either training or a model.

The interface is inteneded to be as close to the wandb.config class as possible. But it does not have any locked
entries as in wandb.config

Examples:

    Initialization
    ```
    config = Config()
    config = Config(dict(a=1, b=2))
    ```

    add a new parameter

    ```
    config['key'] = default_value
    config.key = default_value
    ```

    set up typehint for a parameter
    ```
    config['_key_type'] = int
    config._key_type = int
    config.set_type(key, int)
    ```

    update with a dictionary
    ```
    config.update(dictionary={'a':3, 'b':4})
    ```

    If a parameter is updated, the updated value will be formatted back to the same type.

"""
import inspect

from copy import deepcopy
from typing import Optional

from geqtrain.utils.savenload import save_file, load_file


ATOMIC_NUMBER_MAP = {k.upper(): v for k, v in {
    "H" : 1,  "He": 2,  "Li": 3,  "Be": 4,  "B" : 5,  "C" : 6,  "N" : 7,
    "O" : 8,  "F" : 9,  "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P" : 15, "S" : 16, "Cl": 17, "Ar": 18, "K" : 19, "Ca": 20, "Sc": 21,
    "Ti": 22, "V" : 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
    "Kr": 36, "Rb": 37, "Sr": 38, "Y" : 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
    "Sn": 50, "Sb": 51, "Te": 52, "I" : 53, "Xe": 54, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W" : 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
    "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91,
    "U" : 92
}.items()}

INVERSE_ATOMIC_NUMBER_MAP = {v: k for k, v in ATOMIC_NUMBER_MAP.items()}


DEFAULT_CONFIG = dict(
    wandb=False,
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_fusion_strategy=[("DYNAMIC", 3)],
)


class Config(object):
    def __init__(
        self,
        config: Optional[dict] = None,
        allow_list: Optional[list] = None,
        exclude_keys: Optional[list] = None,
    ):

        object.__setattr__(self, "_items", dict())
        object.__setattr__(self, "_item_types", dict())
        object.__setattr__(self, "_allow_list", list())
        object.__setattr__(self, "_allow_all", True)

        if allow_list is not None:
            self.add_allow_list(allow_list, default_values={})

        if config is not None and exclude_keys is not None:
            config = {key: value for key, value in config.items() if key not in exclude_keys}
        if config is not None:
            self.update(config)

    def __repr__(self):
        return str(dict(self))

    __str__ = __repr__

    def keys(self):
        return self._items.keys()

    def _as_dict(self):
        return self._items

    def as_dict(self):
        return dict(self)

    def __getitem__(self, key):
        return self._items[key]

    def get_type(self, key):
        """Get Typehint from item_types dict or previous defined value
        Args:

            key: name of the variable
        """

        return self._item_types.get(key, None)

    def set_type(self, key, typehint):
        """set typehint for a variable

        Args:

            key: name of the variable
            typehint: type of the variable
        """

        self._item_types[key] = typehint

    def add_allow_list(self, keys, default_values={}):
        """add key to allow_list"""

        object.__setattr__(self, "_allow_all", False)
        object.__setattr__(
            self, "_allow_list", list(set(self._allow_list).union(set(keys)))
        )
        self.update(default_values)

    def allow_list(self):
        return self._allow_list

    def __setitem__(self, key, val):

        # typehint
        if key.endswith("_type") and key.startswith("_"):

            k = key[1:-5]
            if (not self._allow_all) and key not in self._allow_list:
                return None

            self._item_types[k] = val

        # normal value
        else:

            if (not self._allow_all) and key not in self._allow_list:
                return None

            typehint = self.get_type(key)

            # try to format the variable
            try:
                val = typehint(val) if typehint is not None else val
            except Exception:
                raise TypeError(
                    f"Wrong Type: Parameter {key} should be {typehint} type."
                    f"But {type(val)} is given"
                )

            self._items[key] = deepcopy(val)
            return key

    def items(self):
        return self._items.items()

    __setattr__ = __setitem__

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __contains__(self, key):
        return key in self._items

    def pop(self, *args):
        return self._items.pop(*args)

    def update_w_prefix(
        self,
        dictionary: dict,
        prefix: str,
        allow_val_change=None,
    ):
        """Mock of wandb.config function

        Add a dictionary of parameters to the
        The key of the parameter cannot be started as "_"

        Args:

            dictionary (dict): dictionary of parameters and their typehint to update
            allow_val_change (None): mock for wandb.config, not used.

        Returns:

        """

        # override with prefix
        l_prefix = len(prefix) + 1
        prefix_dict = {
            k[l_prefix:]: v for k, v in dictionary.items() if k.startswith(prefix + "_")
        }
        keys = self.update(prefix_dict, allow_val_change=allow_val_change)
        keys = {k: f"{prefix}_{k}" for k in keys}

        for suffix in ["params", "kwargs"]:
            if f"{prefix}_{suffix}" in dictionary:
                key3 = self.update(
                    dictionary[f"{prefix}_{suffix}"],
                    allow_val_change=allow_val_change,
                )
                keys.update({k: f"{prefix}_{suffix}.{k}" for k in key3})
        return keys

    def update(self, dictionary: dict, allow_val_change=None):
        """Mock of wandb.config function

        Add a dictionary of parameters to the config
        The key of the parameter cannot be started as "_"

        Args:

            dictionary (dict): dictionary of parameters and their typehint to update
            allow_val_change (None): mock for wandb.config, not used.

        Returns:
            keys (set): set of keys being udpated

        """

        keys = []

        if 'include' in dictionary:
            include_files = dictionary.pop('include')
            if not isinstance(include_files, list):
                include_files = [include_files]
            for include_file in include_files:
                included_dict = load_file(supported_formats={"yaml": ("yml", "yaml"), "json": "json"}, filename=include_file)
                self.update(included_dict)

        # first log in all typehints or hidden variables
        for k, value in dictionary.items():
            if k.startswith("_"):
                keys += [self.__setitem__(k, value)]

        # then log in the values
        for k, value in dictionary.items():
            if not k.startswith("_"):
                keys += [self.__setitem__(k, value)]

        return set(keys) - set([None])

    def get(self, *args):
        return self._items.get(*args)

    def persist(self):
        """mock wandb.config function"""
        pass

    def setdefaults(self, d):
        """mock wandb.config function"""
        pass

    def update_locked(self, d, user=None):
        """mock wandb.config function"""
        pass

    def save(self, filename: str, format: Optional[str] = None):
        """Print config to file."""

        supported_formats = {"yaml": ("yml", "yaml"), "json": "json"}
        return save_file(
            item=dict(self),
            supported_formats=supported_formats,
            filename=filename,
            enforced_format=format,
        )

    @staticmethod
    def from_file(filename: str, format: Optional[str] = None, defaults: dict = DEFAULT_CONFIG):
        """Load arguments from file"""

        supported_formats = {"yaml": ("yml", "yaml"), "json": "json"}
        dictionary = load_file(
            supported_formats=supported_formats,
            filename=filename,
            enforced_format=format,
        )

        config = Config.from_dict(dictionary, defaults)
        config.filepath = filename

        return config

    @staticmethod
    def from_dict(dictionary: dict, defaults: dict = {}):
        c = Config(defaults)
        c.update(dictionary)
        c.parse_node_types()
        c.parse_attributes()
        # c.parse_targets_metadata()
        return c

    @staticmethod
    def from_class(class_type, remove_kwargs: bool = False):
        """return Config class instance based on init function of the input class
        the instance will only allow to store init function related variables
        the type hints are all set to None, so no automatic format conversion is applied

        class_type: torch.module children class type
        remove_kwargs (optional, bool): the same as Config.from_function

        Returns:

        config (Config):
        """

        if inspect.isclass(class_type):
            return Config.from_function(
                class_type.__init__, remove_kwargs=remove_kwargs
            )
        elif callable(class_type):
            return Config.from_function(class_type, remove_kwargs=remove_kwargs)
        else:
            raise ValueError(
                f"from_class only takes class type or callable, but got {class_type}"
            )

    @staticmethod
    def from_function(function, remove_kwargs=False):
        """return Config class instance based on the function of the input class
        the instance will only allow to store init function related variables
        the type hints are all set to None, so no automatic format conversion is applied

        Args:

        function: function name
        remove_kwargs (optional, bool): if True, kwargs are removed from the keys
             and the returned instance will only takes the init params of the class_type.
             if False and kwargs exists, the config only initialized with the default param values,
             but it can take any other keys

        Returns:

        config (Config):
        """

        sig = inspect.signature(function)

        default_params = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        param_keys = list(sig.parameters.keys())
        if param_keys[0] == "self":
            param_keys = param_keys[1:]

        for key in param_keys:
            default_params[f"_{key}_type"] = None

        # do not restrict variables when kwargs exists
        if "kwargs" in param_keys and not remove_kwargs:
            return Config(config=default_params)
        elif "kwargs" in param_keys:
            param_keys.remove("kwargs")
            return Config(config=default_params, allow_list=param_keys)
        else:
            return Config(config=default_params, allow_list=param_keys)

    def parse_node_types(self):
        if "type_names" in self:
            type_names = [str(type_name) for type_name in self["type_names"]]
            self["type_names"] = type_names
            num_types = len(type_names)
            # check consistency
            assert self.get("num_types", num_types) == num_types
            self["num_types"] = num_types
        elif "num_types" in self:
            num_types = self["num_types"]
            self["type_names"] = [f"type_{str(i)}" for i in range(num_types)]

    def parse_attributes(self):
        if "node_attributes" in self and "node_types" in self["node_attributes"]:
            if "num_types" in self["node_attributes"]["node_types"]:
                assert self["node_attributes"]["node_types"]["num_types"] == self["num_types"]
            else:
                self["node_attributes"]["node_types"]["num_types"] = self["num_types"]
        for attr in ["node_attributes", "edge_attributes", "graph_attributes", "extra_attributes"]:
            if attr not in self:
                continue
            for key, field in self[attr].items():
                num_types = int(field.get("num_types", 0))
                can_be_undefined = field.get("can_be_undefined", False)
                self[attr][key].update({
                    "actual_num_types": num_types + int(can_be_undefined)
                })

    def parse_targets_metadata(self):
        '''
        parses information relatet to target_* from yaml
        '''

        if "target_names" in self:
            target_names = [str(target_names) for target_names in self["target_names"]]
            self["target_names"] = target_names
            num_targets = len(target_names)
             # check consistency
            assert len(self.get("target_names", [])) == num_targets
        if "target_means" in self:
            target_means = [float(x) for x in self['target_means']]
            self.target_means = target_means
            n = len(target_means)
            assert len(target_means) == n
        if "target_stds" in self:
            target_stds = [float(x) for x in self['target_stds']]
            self.target_stds = target_stds
            n = len(target_stds)
            assert len(target_stds) == n

