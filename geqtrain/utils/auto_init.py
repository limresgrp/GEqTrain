""" Adapted from https://github.com/mir-group/nequip
"""

from typing import Optional, Union, List
import inspect
import logging

from geqtrain.utils.savenload import load_callable

from .config import Config


def instantiate_from_cls_name(
    module,
    class_name: str,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: Optional[dict] = None,
    all_args: Optional[dict] = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
):
    """Initialize a class based on a string class name. (Delegates to `instantiate`)"""

    if class_name is None:
        raise NameError("class_name type is not defined ")

    # first obtain a list of all classes in this module
    the_class = getattr(module, class_name, None)

    if not inspect.isclass(the_class):
        raise NameError(f"{class_name} type is not found or not a class in {module.__name__} module")

    return instantiate(
        builder=the_class,
        prefix=prefix,
        positional_args=positional_args,
        optional_args=optional_args,
        all_args=all_args,
        remove_kwargs=remove_kwargs,
        return_args_only=return_args_only,
    )


def instantiate(
    builder,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: dict = None,
    all_args: dict = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
    parent_builders: list = [],
):
    """
    Automatic initializing class instance by matching keys in the parameter 
    dictionary to the constructor functions.
    It intelligently handles inheritance based on `**kwargs`.
    """

    prefix_list = [builder.__name__] if inspect.isclass(builder) else []
    if isinstance(prefix, str):
        prefix_list += [prefix]
    elif isinstance(prefix, list):
        prefix_list += prefix
    else:
        raise ValueError(f"prefix has the wrong type {type(prefix)}")

    if inspect.isclass(builder):
        # from_class() will decide whether to do a shallow or deep (MRO) inspection
        # based on the presence of `**kwargs` in the __init__ signature.
        config = Config.from_class(builder, remove_kwargs=remove_kwargs)
    else:
        # Fallback for non-class builders (functions)
        config = Config.from_callable(builder, remove_kwargs=remove_kwargs)
    # =====================================================================

    # be strict about _kwargs keys:
    allow = config.allow_list()
    for key in allow:
        bname = key[:-7]
        if key.endswith("_kwargs") and bname not in allow:
            raise KeyError(
                f"Instantiating {builder.__name__}: found kwargs argument `{key}`, but no parameter `{bname}` for the corresponding builder. (Did you rename `{bname}` but forget to change `{bname}_kwargs`?) Either add a parameter for `{bname}` if you are trying to allow construction of a submodule, or, if `{bname}_kwargs` is just supposed to be a dictionary, rename it without `_kwargs`."
            )
    del allow

    key_mapping = {}
    if all_args is not None:
        _keys = config.update(all_args)
        key_mapping["all"] = {k: k for k in _keys}
        for prefix_str in prefix_list:
            _keys = config.update_w_prefix(all_args, prefix=prefix_str)
            key_mapping["all"].update(_keys)

    if optional_args is not None:
        _keys = config.update(optional_args)
        key_mapping["optional"] = {k: k for k in _keys}
        for prefix_str in prefix_list:
            _keys = config.update_w_prefix(optional_args, prefix=prefix_str)
            key_mapping["optional"].update(_keys)

    if "all" in key_mapping and "optional" in key_mapping:
        key_mapping["all"] = {
            k: v for k, v in key_mapping["all"].items() if k not in key_mapping["optional"]
        }

    final_optional_args = dict(config)

    if len(parent_builders) > 0:
        _positional_args = {
            k: v for k, v in positional_args.items() if k in config.allow_list()
        }
        positional_args = _positional_args

    init_args = final_optional_args.copy()
    init_args.update(positional_args)

    search_keys = [key for key in init_args if key + "_kwargs" in config.allow_list()]
    for key in search_keys:
        sub_builder = init_args[key]
        if sub_builder is None:
            continue

        if isinstance(sub_builder, str):
            sub_builder = load_callable(sub_builder, prefix=prefix)
            final_optional_args[key] = sub_builder

        if (
            callable(sub_builder) and sub_builder not in parent_builders
            and key + "_kwargs" not in positional_args
        ):
            sub_prefix_list = [sub_builder.__name__, key] + [p + "_" + key for p in prefix_list]

            nested_km, nested_kwargs = instantiate(
                sub_builder,
                prefix=sub_prefix_list,
                positional_args={},
                optional_args=optional_args,
                all_args=all_args,
                remove_kwargs=remove_kwargs,
                return_args_only=True,
                parent_builders=[builder] + parent_builders,
            )
            found_keys = set()
            if 'all' in nested_km:
                found_keys.update(nested_km['all'].keys())
            if 'optional' in nested_km:
                found_keys.update(nested_km['optional'].keys())

            filtered_nested_kwargs = {k: v for k, v in nested_kwargs.items() if k in found_keys}

            # Now combine the filtered kwargs with any explicitly provided _kwargs dict
            filtered_nested_kwargs.update(final_optional_args.get(key + "_kwargs", {}))
            final_optional_args[key + "_kwargs"] = filtered_nested_kwargs

            for t in key_mapping:
                key_mapping[t].update({key + "_kwargs." + k: v for k, v in nested_km[t].items()})
        elif sub_builder in parent_builders:
            raise RuntimeError(f"cyclic recursion in builder {parent_builders} {sub_builder}")

    for key in positional_args:
        final_optional_args.pop(key, None)
        for t in key_mapping:
            key_mapping[t].pop(key, None)

    if return_args_only:
        return key_mapping, final_optional_args

    logging.debug(f"instantiate {builder.__name__}")

    # Final sanitation: remove any arguments that are not actually in the final builder's
    # signature. This is a safeguard against overly broad MRO scans (e.g. `*args` from `object`)
    # that `Config` might pick up.
    allowed_keys = [x for x in config.allow_list() if x != 'args']
    final_optional_args = {k: v for k, v in final_optional_args.items() if k in allowed_keys}

    for t in key_mapping:
        for k, v in key_mapping[t].items():
            string = f" {t:>10s}_args :  {k:>50s}"
            if k != v:
                string += f" <- {v:>50s}"
            logging.debug(string)
    logging.debug(f"...{builder.__name__}_param = dict(")
    logging.debug(f"...   optional_args = {final_optional_args},")
    logging.debug(f"...   positional_args = {positional_args})")

    try:
        instance = builder(**positional_args, **final_optional_args)
    except Exception as e:
        raise RuntimeError(f"Failed to build object with prefix `{prefix}` using builder `{builder.__name__}` with args: positional={positional_args}, optional={final_optional_args}") from e

    return instance, final_optional_args

def get_w_prefix(
    key: List[str],
    *kwargs,
    arg_dicts: List[dict] = [],
    prefix: Optional[Union[str, List[str]]] = [],
):
    """
    act as the get function and try to search for the value key from arg_dicts
    """

    # detect the input parameters needed from params
    config = Config(config={}, allow_list=[key])

    # sort out all possible prefixes
    if isinstance(prefix, str):
        prefix_list = [prefix]
    elif isinstance(prefix, list):
        prefix_list = prefix
    else:
        raise ValueError(f"prefix is with a wrong type {type(prefix)}")

    if not isinstance(arg_dicts, list):
        arg_dicts = [arg_dicts]

    # extract all the parameters that has the pattern prefix_variable
    # debug container to record all the variable name transformation
    key_mapping = {}
    for idx, arg_dict in enumerate(arg_dicts[::-1]):
        # fetch paratemeters that directly match the name
        _keys = config.update(arg_dict)
        key_mapping[idx] = {k: k for k in _keys}
        # fetch paratemeters that match prefix + "_" + name
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                arg_dict,
                prefix=prefix_str,
            )
            key_mapping[idx].update(_keys)

    # for logging only, remove the overlapped keys
    num_dicts = len(arg_dicts)
    if num_dicts > 1:
        for id_dict in range(num_dicts - 1):
            higher_priority_keys = []
            for id_higher in range(id_dict + 1, num_dicts):
                higher_priority_keys += list(key_mapping[id_higher].keys())
            key_mapping[id_dict] = {
                k: v
                for k, v in key_mapping[id_dict].items()
                if k not in higher_priority_keys
            }

    # debug info
    logging.debug(f"search for {key} with prefix {prefix}")
    for t in key_mapping:
        for k, v in key_mapping[t].items():
            string = f" {str(t):>10.10}_args :  {k:>50s}"
            if k != v:
                string += f" <- {v:>50s}"
            logging.debug(string)

    return config.get(key, *kwargs)

