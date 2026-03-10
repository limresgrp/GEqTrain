""" Adapted from https://github.com/mir-group/nequip
"""

from typing import Optional, Union, List
import inspect
import logging
import re

from geqtrain.utils.savenload import load_callable

from .config import Config


def _camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def instantiate_from_cls_name(
    module,
    class_name: str,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: Optional[dict] = None,
    all_args: Optional[dict] = None,
    remove_kwargs: bool = True,
    _dry_run_mode: bool = False,
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
    Automatically initializes a class instance by matching keys from configuration
    dictionaries to the constructor arguments.

    The function constructs a list of potential prefixes for parameters by looking
    at the full context of the class being instantiated. This includes:
    - The top-level instance name (e.g., `head_del` from the YAML key).
    - The class name of the component itself (e.g., `ScalarMLPFunction`).
    - The argument name the component is assigned to (e.g., `readout_latent`).
    - The class names of all parent modules.

    It checks for keys in the YAML matching these prefixes and their combinations.
    More specific combinations have higher precedence.

    Precedence Order (from lowest to highest):
    Using the example of instantiating `ScalarMLPFunction` for the `readout_latent`
    argument inside an `AttentionReadoutModule` instance named `head_energy`.

    1. Global default:
       `mlp_latent_dimensions: [1,1,1]`
    2. By child class name:
       `scalar_mlp_function_mlp_latent_dimensions: [2,2,2]`
    3. By parent class name (applies to any valid submodule in the parent):
       `attention_readout_module_mlp_latent_dimensions: [3,3,3]`
    4. By argument name:
       `readout_latent_mlp_latent_dimensions: [4,4,4]`
    5. By instance name (applies as a default to any valid submodule in the instance):
       `head_energy_mlp_latent_dimensions: [5,5,5]`
    6. By instance name + parent class name:
       `head_energy_attention_readout_module_mlp_latent_dimensions: [6,6,6]`
    7. By instance name + child class name:
       `head_energy_scalar_mlp_function_mlp_latent_dimensions: [7,7,7]`
    8. By instance name + argument name (most specific):
       `head_energy_readout_latent_mlp_latent_dimensions: [8,8,8]`
    """

    builder_name = builder.__name__ if inspect.isclass(builder) else ""
    builder_name_snake = _camel_to_snake(builder_name) if builder_name else ""

    class_prefixes = [p for p in [builder_name, builder_name_snake] if p]

    parent_context_prefixes = []
    if isinstance(prefix, str):
        parent_context_prefixes.append(prefix)
    elif isinstance(prefix, list):
        parent_context_prefixes.extend(prefix)
        
    # Build the search list for the CURRENT builder. Later items have higher precedence.
    prefix_list = []
    prefix_list.extend(class_prefixes)
    prefix_list.extend(parent_context_prefixes)

    instance_prefix = None
    for p in parent_context_prefixes:
        # Heuristic: the instance name is the first context prefix that isn't a class name
        if p and not any(c.isupper() for c in p):
             instance_prefix = p
             break
    
    if instance_prefix:
        for cp in class_prefixes:
            if cp == builder_name_snake:
                prefix_list.append(f"{instance_prefix}_{cp}")

    prefix_list = list(dict.fromkeys(p for p in prefix_list if p))

    if inspect.isclass(builder):
        config = Config.from_class(builder, remove_kwargs=remove_kwargs)
    else:
        config = Config.from_callable(builder, remove_kwargs=remove_kwargs)

    allow = config.allow_list()
    for key in allow:
        bname = key[:-7]
        if key.endswith("_kwargs") and bname not in allow:
            raise KeyError(
                f"Instantiating {builder.__name__}: found kwargs argument `{key}`, but no parameter `{bname}` for the corresponding builder. Either add `{bname}` as a parameter or rename `{key}`."
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
        key_mapping["all"] = {k: v for k, v in key_mapping["all"].items() if k not in key_mapping["optional"]}

    final_optional_args = dict(config)

    if len(parent_builders) > 0:
        positional_args = {k: v for k, v in positional_args.items() if k in config.allow_list()}
    
    # Get the dry run mode from the global config
    _dry_run_mode = all_args.get("_dry_run_mode", False) if all_args is not None else False

    if "irreps_in" in positional_args and positional_args["irreps_in"] is not None:
        positional_args["irreps_in"]["_dry_run_mode"] = _dry_run_mode


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

        if callable(sub_builder) and sub_builder not in parent_builders and key + "_kwargs" not in positional_args:
            
            # This block constructs the prefix list for the recursive call.
            # It is built from lowest to highest priority, as later items will override earlier ones.
            sub_prefix_list = []

            # Heuristic: Find the top-level instance name from the parent's full context.
            instance_prefix = None
            for p in (prefix if isinstance(prefix, list) else [prefix]):
                if p and not any(c.isupper() for c in p):
                    instance_prefix = p
                    break

            parent_class_snake = _camel_to_snake(builder_name)
            child_class_snake = _camel_to_snake(sub_builder.__name__)

            # Level 3: Parent Class Name
            if parent_class_snake:
                sub_prefix_list.append(parent_class_snake)
            
            # Level 4: Argument Name
            sub_prefix_list.append(key)

            if instance_prefix:
                # Level 5: Instance Name (as a default for the submodule)
                sub_prefix_list.append(instance_prefix)
                
                # Level 6: Instance + Parent Class Name
                if parent_class_snake:
                    sub_prefix_list.append(f"{instance_prefix}_{parent_class_snake}")
                
                # Level 7: Instance + Child Class Name
                if child_class_snake:
                    sub_prefix_list.append(f"{instance_prefix}_{child_class_snake}")
                
                # Level 8: Instance + Argument Name
                sub_prefix_list.append(f"{instance_prefix}_{key}")


            nested_km, nested_kwargs = instantiate(
                sub_builder,
                prefix=sub_prefix_list,
                positional_args=positional_args,
                optional_args=optional_args,
                all_args=all_args,
                remove_kwargs=remove_kwargs,
                return_args_only=True,
                parent_builders=[builder] + parent_builders,
            )
            
            found_keys = set(nested_km.get('all', {}).keys()) | set(nested_km.get('optional', {}).keys())
            filtered_nested_kwargs = {k: v for k, v in nested_kwargs.items() if k in found_keys}

            filtered_nested_kwargs.update(final_optional_args.get(key + "_kwargs", {}))
            final_optional_args[key + "_kwargs"] = filtered_nested_kwargs

            for t in key_mapping:
                key_mapping[t].update({f"{key}_kwargs.{k}": v for k, v in nested_km.get(t, {}).items()})
        elif sub_builder in parent_builders:
            raise RuntimeError(f"cyclic recursion in builder {parent_builders} {sub_builder}")

    for key in positional_args:
        final_optional_args.pop(key, None)
        for t in key_mapping:
            key_mapping[t].pop(key, None)

    if return_args_only:
        return key_mapping, final_optional_args

    logging.debug(f"instantiate {builder.__name__}")
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
