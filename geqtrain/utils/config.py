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


DEFAULT_CONFIG = dict(
    wandb=False,
    dataset_statistics_stride=1,
    default_dtype="float32",
    allow_tf32=False,  # TODO: until we understand equivar issues
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    fine_tune=False,
    append=False,
    _jit_bailout_depth=2,  # avoid 20 iters of pain, see https://github.com/pytorch/pytorch/issues/52286
    # Quote from eelison in PyTorch slack:
    # https://pytorch.slack.com/archives/CDZD1FANA/p1644259272007529?thread_ts=1644064449.039479&cid=CDZD1FANA
    # > Right now the default behavior is to specialize twice on static shapes and then on dynamic shapes.
    # > To reduce warmup time you can do something like setFusionStrartegy({{FusionBehavior::DYNAMIC, 3}})
    # > ... Although we would wouldn't really expect to recompile a dynamic shape fusion in a model,
    # > provided broadcasting patterns remain fixed
    # We default to DYNAMIC alone because the number of edges is always dynamic,
    # even if the number of atoms is fixed:
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
            config = {
                key: value for key, value in config.items() if key not in exclude_keys
            }
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
        return Config.from_dict(dictionary, defaults)

    @staticmethod
    def from_dict(dictionary: dict, defaults: dict = {}):
        c = Config(defaults)
        c.update(dictionary)
        c.parse_node_types()
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

    load = from_file

    def parse_node_types(self):
        if "type_names" in self:
            type_names = [str(type_name) for type_name in self["type_names"]]
            self["type_names"] = type_names
            num_types = len(type_names)
            # check consistency
            assert self.get("num_types", num_types) == num_types
        elif "num_types" in self:
            num_types = self["num_types"]
            self["type_names"] = [f"type_{str(i)}" for i in range(num_types)]
        else:
            num_types = 1
            self["type_names"] = ["type_0"]
        self["num_types"] = num_types