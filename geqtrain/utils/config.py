""" Adapted from https://github.com/mir-group/nequip
"""

"""
Class to hold a bunch of hyperparameters associated with either training or a model.

The interface is intended to be as close to the wandb.config class as possible.
"""
import inspect

from copy import deepcopy
from typing import Optional, List

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
    allow_tf32=False,
    verbose="INFO",
    model_debug_mode=False,
    equivariance_test=False,
    grad_anomaly_mode=False,
    append=False,
    _jit_fusion_strategy=[("DYNAMIC", 10)],
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

    def as_dict(self):
        return dict(self)

    def __getitem__(self, key):
        return self._items[key]

    def get_type(self, key):
        """Get Typehint from item_types dict or previously defined value."""
        return self._item_types.get(key, None)

    def set_type(self, key, typehint):
        """Set typehint for a variable."""
        self._item_types[key] = typehint

    def add_allow_list(self, keys, default_values={}):
        """Add keys to the allow_list, restricting which keys can be set."""
        object.__setattr__(self, "_allow_all", False)
        object.__setattr__(
            self, "_allow_list", list(set(self._allow_list).union(set(keys)))
        )
        self.update(default_values)

    def allow_list(self):
        return self._allow_list

    def __setitem__(self, key, val):
        # Handle typehint declarations like `_key_type`
        if key.endswith("_type") and key.startswith("_"):
            k = key[1:-5]
            if (not self._allow_all) and k not in self._allow_list:
                return None
            self._item_types[k] = val
        else:
            if (not self._allow_all) and key not in self._allow_list:
                return None
            
            typehint = self.get_type(key)
            # try to cast the variable to its typehint if it exists
            if typehint is not None:
                try:
                    val = typehint(val)
                except Exception:
                    raise TypeError(
                        f"Wrong Type: Parameter '{key}' should be of type {typehint}, "
                        f"but received type {type(val)}."
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
        
    def __getstate__(self):
        """Called by pickle to get the object's state for serialization."""
        return {
            "_items": self._items,
            "_item_types": self._item_types,
            "_allow_list": self._allow_list,
            "_allow_all": self._allow_all,
            "filepath": getattr(self, "filepath", None),
        }

    def __setstate__(self, state):
        """Called by pickle to restore the object's state from deserialization."""
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def pop(self, *args):
        return self._items.pop(*args)

    def update_w_prefix(self, dictionary: dict, prefix: str):
        """
        Update config with keys from a dictionary that match a given prefix.
        For example, if prefix='model', it will look for 'model_key' in the dictionary.
        """
        keys = {}
        # Override with prefix_key
        prefix_with_underscore = prefix + "_"
        l_prefix = len(prefix_with_underscore)
        prefix_dict = {
            k[l_prefix:]: v for k, v in dictionary.items() if k.startswith(prefix_with_underscore)
        }
        updated_keys = self.update(prefix_dict)
        keys.update({k: f"{prefix_with_underscore}{k}" for k in updated_keys})

        # Also check for nested dictionaries like `model_kwargs`
        for suffix in ["params", "kwargs"]:
            nested_key = f"{prefix}_{suffix}"
            if nested_key in dictionary and isinstance(dictionary[nested_key], dict):
                nested_updated_keys = self.update(dictionary[nested_key])
                keys.update({k: f"{nested_key}.{k}" for k in nested_updated_keys})
        return keys

    def update(self, dictionary: dict):
        """
        Update the config with a dictionary of parameters.
        Keys starting with '_' are treated as private or typehints and set first.
        """
        keys = []
        if 'include' in dictionary:
            include_files = dictionary.pop('include')
            if not isinstance(include_files, list):
                include_files = [include_files]
            for include_file in include_files:
                included_dict = load_file(supported_formats={"yaml": ("yml", "yaml"), "json": "json"}, filename=include_file)
                self.update(included_dict)
        
        # first set all typehints or hidden variables
        for k, value in dictionary.items():
            if k.startswith("_"):
                keys.append(self.__setitem__(k, value))
        # then set the actual values
        for k, value in dictionary.items():
            if not k.startswith("_"):
                keys.append(self.__setitem__(k, value))

        return set(k for k in keys if k is not None)

    def get(self, *args):
        return self._items.get(*args)

    def save(self, filename: str, format: Optional[str] = None):
        """Save the current config to a file (YAML or JSON)."""
        supported_formats = {"yaml": ("yml", "yaml"), "json": "json"}
        return save_file(
            item=dict(self),
            supported_formats=supported_formats,
            filename=filename,
            enforced_format=format,
        )

    @staticmethod
    def from_file(filename: str, format: Optional[str] = None, defaults: dict = DEFAULT_CONFIG):
        """Load a config from a YAML or JSON file."""
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
        """Create a config instance from a dictionary."""
        c = Config(defaults)
        c.update(dictionary)
        c.parse_node_types()
        c.parse_attributes()
        return c

    # =====================================================================
    # UPDATED/NEW METHODS FOR MRO INSPECTION
    # =====================================================================

    @classmethod
    def from_callable(cls, fun, remove_kwargs: bool = True):
        """
        Creates a Config object from a single callable (function or method).
        This method does NOT inspect inheritance.
        """
        if inspect.isclass(fun):
            raise TypeError("from_callable is for functions/methods, not classes. Use from_class for classes.")
        
        sig = inspect.signature(fun)
        param_keys = list(sig.parameters.keys())
        
        # If it's a method, 'self' is the first arg, so we skip it
        if len(param_keys) > 0 and param_keys[0] == "self":
            param_keys = param_keys[1:]

        default_params = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        if "kwargs" in param_keys and not remove_kwargs:
            # If kwargs are present and not removed, the config can accept any key
            return Config(config=default_params)
        else:
            if "kwargs" in param_keys:
                param_keys.remove("kwargs")
            # Otherwise, restrict the config to only the specified keys
            return Config(config=default_params, allow_list=param_keys)
            
    @classmethod
    def from_class(cls, class_type, remove_kwargs: bool = True):
        """
        Creates a Config instance based on the __init__ method of the input class.

        If the class's __init__ method contains a `**kwargs` parameter, this
        method will inspect the entire inheritance hierarchy (MRO) to build a
        complete list of valid arguments from all parent classes.

        Otherwise, it will only consider the arguments explicitly defined in the
        __init__ method of the class itself, ignoring parents. This is useful
        for classes that intentionally override parent arguments.
        """
        if not inspect.isclass(class_type):
            raise ValueError(f"from_class only takes a class, but got {type(class_type)}")
        
        has_kwargs = False
        try:
            init_sig = inspect.signature(class_type.__init__)
            params = init_sig.parameters.values()
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        except (ValueError, TypeError):
            # Handles built-in types like `object` that don't have a useful __init__ signature
            pass

        if has_kwargs:
            # **kwargs is present, so we inspect the full inheritance tree
            # to collect arguments from all parent classes.
            mro = list(inspect.getmro(class_type))
            return cls.from_classes(mro, remove_kwargs=remove_kwargs)
        else:
            # No **kwargs, so we only inspect the class's __init__ itself.
            # This prevents pulling in arguments from parents that the class
            # doesn't intend to accept (like `transform` in NpzDataset).
            return cls.from_callable(class_type.__init__, remove_kwargs=remove_kwargs)

    @classmethod
    def from_classes(cls, class_mro: List[type], remove_kwargs: bool = True):
        """
        Builds a config from a list of classes by inspecting their __init__ methods.
        This is useful for handling inheritance, where a child class's __init__ might
        not explicitly list all arguments from its parents.
        
        Args:
            class_mro (list): A list of classes, typically from `inspect.getmro()`.
            remove_kwargs (bool): If True, `**kwargs` arguments are ignored.
        """
        all_params = {}
        # Iterate through the Method Resolution Order
        for c in class_mro:
            if hasattr(c, '__init__') and callable(c.__init__):
                try:
                    sig = inspect.signature(c.__init__)
                    for name, param in sig.parameters.items():
                        # Exclude 'self' and, optionally, 'kwargs'
                        if name == 'self' or (remove_kwargs and param.kind == inspect.Parameter.VAR_KEYWORD):
                            continue
                        # Add parameter only if not already seen from a more specific subclass
                        if name not in all_params:
                            all_params[name] = param.default if param.default is not inspect.Parameter.empty else None
                except (ValueError, TypeError):
                    # Some built-ins like 'object' don't have a useful signature
                    continue
        return cls(config=all_params, allow_list=list(all_params.keys()))

    def parse_node_types(self):
        """Parses and validates type_names and num_types fields."""
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
        """Parses and validates attribute fields."""
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
