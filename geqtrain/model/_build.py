import inspect
from typing import Optional
from torch.nn import Module
from geqtrain.data import AtomicDataset
from geqtrain.utils import load_callable, Config, add_tags_to_parameter
from typing import Tuple, Dict

def add_suffix(_str: str, i: int) -> str:
    return _str.__add__(f".__{i}__")

def remove_suffix(_str: str, i: int) -> str:
    suff = f".__{i}__"
    return _str[:-len(suff)]

def parse_model_builders(config) -> Dict[str, Dict[str, str]]:

    builders = config.get("model_builders", None)
    if builders is None:
        raise ValueError("No model_builders found in config")

    builders_and_params = dict()
    for i, builder in enumerate(builders):
        if isinstance(builder, dict):
            if "cls" in builder:
                """ Option 1
                    model_builders:
                        - cls: InteractionModule
                          name: my_interaction
                          weights: tune # load|tune|freeze
                          ...other params

                    Load single module class 'InteractionModule' (found by default inside geqtrain.nn)
                    and register in the SequentialGraphNetwork with name 'my_interaction'.
                    If weights are provided, try to load them into this module.
                    Use specified 'fine_tune_lr' learning rate for weights of this module.
                """
                _callable = add_suffix("Module", i)
                params = builder
            else:
                """ Option 2
                    model_builders:
                        - Heads: tune # load|tune|freeze

                    Load packaged model function 'Heads' (found by default inside geqtrain.model)
                    If weights are provided, try to load them into this module.
                    Use specified 'fine_tune_lr' learning rate for weights of modules packaged into this model.
                """
                assert len(builder) == 1
                _callable = add_suffix(list(builder.keys())[0], i)
                params = {"weights": list(builder.values())[0]}
            builders_and_params[_callable] = params
        elif isinstance(builder, str):
            """ Option 3
                model_builders:
                    - Heads

                Load packaged model function 'Heads' (found by default inside geqtrain.model)
                Initialize weights from scratch
            """
            _callable = add_suffix(builder, i)
            builders_and_params[_callable] = dict()

    return builders_and_params

def flatten_list(nested_list):
    """
    Flattens a nested list of arbitrary depth, including handling empty lists and None values.

    Args:
        nested_list (list): A potentially nested list to be flattened.

    Returns:
        list: A single flattened list with no nested elements.
    """
    flattened = []

    for element in nested_list:
        if isinstance(element, list):
            # Recursive call to handle nested lists
            flattened.extend(flatten_list(element))
        elif element is not None:
            # Include non-None elements
            flattened.append(element)

    return flattened

def model_from_config(
    config: Config,
    initialize: bool = False,
    dataset: Optional[AtomicDataset] = None,
    deploy: bool = False,
) -> Tuple[Module, Dict]:
    """Build a model based on `config`.

    step 1: create list of ordered factory methods
    step 2: iterate over factory methods and call them with the appropriate kwargs,
    step3: the return of the previous factory method is passed to the next one: model is a sequentially-wrapped-forwardable object

    Model builders (`model_builders`) can have arguments:
     - ``config``: the config. Always present.
     - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
     - ``initialize``: whether to initialize the model
     - ``dataset``: if ``initialize`` is True, the dataset
     - ``deploy``: whether the model object is for deployment / inference

    Args:
        config
        initialize (bool): whether ``model_builders`` should be instructed to initialize the model
        dataset: dataset for initializers if ``initialize`` is True.
        deploy (bool): whether ``model_builders`` should be told the model is for deployment / inference

    Returns:
        built Model
    """

    # parse builders and extract their params
    model_builders = parse_model_builders(config)
    builders, prms, weights_prms = [], [], []
    for i, (k, v) in enumerate(model_builders.items()):
        _callable = remove_suffix(k, i)
        builders.append(load_callable(_callable, prefix="geqtrain.model"))
        weights_prms.append(v.pop("weights", None))
        prms.append(v)

    # checks
    model_for_fine_tuning = config.get("fine_tune", False) # if present, pointed .pth has been already validated

    if 'progress' in config and model_for_fine_tuning:
        raise ValueError("cannot restart and fine-tune at the same time, if you want to fine-tune, do a fresh start")

    weights_prms_provided = any(flatten_list(weights_prms))
    if not model_for_fine_tuning and weights_prms_provided:
        raise ValueError("weights_params provided in model_builders but fine_tune model is not provided")

    if model_for_fine_tuning:
        assert weights_prms_provided, f"Fine-tuning {model_for_fine_tuning} provided, but not weights_params provided in model_builders"

    weights_to_drop_from_model_state: set[str] = set() # used in case of fine-tuning
    weights_already_loaded: set[str] = set()

    # build model
    model = None
    for builder, _prms, _w_prm in zip(builders, prms, weights_prms):
        params = {}
        if _prms.get("cls", None) is not None:
            params["cls"]  = load_callable(_prms.pop("cls"), prefix="geqtrain.nn")
            params["name"] = _prms.pop("name")
            for _p, _v in _prms.items():
                config.update({f'{params["name"]}_{_p}': _v})

        pnames = inspect.signature(builder).parameters # get kwargs of factory method signature
        if "initialize" in pnames:
            params["initialize"] = initialize
        if "deploy" in pnames:
            params["deploy"] = deploy
        if "config" in pnames:
            params["config"] = config
        if "dataset" in pnames:
            if "initialize" not in pnames:
                raise ValueError("Cannot request dataset without requesting initialize")
            if (
                initialize
                and pnames["dataset"].default == inspect.Parameter.empty
                and dataset is None
            ):
                raise RuntimeError(f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`.")
            params["dataset"] = dataset

        # Wrap return of previous builder (every module listed in model_builders - except the first - must require this param)
        if "model" in pnames:
            if model is None:
                raise RuntimeError(f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model")
            params["model"] = model
        else:
            if model is not None:
                raise RuntimeError(f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't")

        model: Module = builder(**params)
        if 'progress' in config or deploy:
            continue

        current_model_param_names = set(k for k, _ in model.named_parameters())
        params_just_added = current_model_param_names - weights_already_loaded
        if _w_prm is None: # if no fine-tuning params, then params of module must be dropped when loading model params; in train-from-scratch scenario all params have to be dropped from state dict
            weights_to_drop_from_model_state.update(params_just_added)
        else:
            for n, p in model.named_parameters():
                if n in params_just_added:
                    # opt1: freeze
                    if _w_prm == "freeze":
                        p.requires_grad = False
                    # opt2: tune with initial lr: fine_tune_lr
                    elif _w_prm == "tune":
                        add_tags_to_parameter(p, "tune")
                    # opt3: load weights if they exist
                    else:
                        assert _w_prm == "load", f"'weights' param must be one among freeze|tune|load. Got {_w_prm}."

        weights_already_loaded.update(current_model_param_names)

    assert model is not None, "Model is not loaded, check model_builders in yaml"
    return model, weights_to_drop_from_model_state