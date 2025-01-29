import inspect
from typing import Optional
from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataset
from geqtrain.utils import load_callable, Config, add_tags_to_parameter
from typing import List


def parse_model_builders(config):

    builders = config.get("model_builders", None)
    if builders is None:
        raise ValueError("No model_builders found in config")

    builders_and_params = {}
    for builder in builders:
        if isinstance(builder, dict):
            assert len(builder) == 1, "Only one builder can be specified at a time"
            _callable = list(builder.keys())[0] # get the only key
            assert len(builder.values()) == 1, "Only one set of parameters can be specified at a time for now"
            builders_and_params[_callable] = list(builder.values()) # assert list(builder.values())
        elif isinstance(builder, str):
            builders_and_params[builder] = []

    return builders_and_params #! todo do list k: [k:{load:T, lr:x, freeze:y}]


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
    config:Config,
    initialize: bool = False,
    dataset: Optional[AtomicDataset] = None,
    deploy: bool = False,
) -> GraphModuleMixin:
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
    model_builders = parse_model_builders(config)
    builders, fine_tune_params = [], []
    for k, v in model_builders.items():
        builders.append(load_callable(k, prefix="geqtrain.model"))
        fine_tune_params.append(v)

    model_for_fine_tuning = config.get("fine_tune", False) # if present, pointed .pth has been already validated
    fine_tune_params_provided = any(flatten_list(fine_tune_params))
    if not model_for_fine_tuning and fine_tune_params_provided:
        raise ValueError("fine_tune_params provided in model_builders but fine_tune model is not provided")

    if model_for_fine_tuning:
        assert fine_tune_params_provided, f"Fine-tuning {model_for_fine_tuning} provided, but not fine_tune_params provided in model_builders"

    model = None
    weights_to_drop_from_model_state: List[str] = set() # used in case of fine-tuning
    weights_already_loaded = set()
    for builder, ft_params in zip(builders, fine_tune_params):
        pnames = inspect.signature(builder).parameters # get kwargs of factory method signature
        params = {}
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

        model = builder(**params)
        current_model_param_names = set(k for k, _ in model.named_parameters())

        params_just_added = current_model_param_names - weights_already_loaded
        if not ft_params: # if no fine-tuning params, then params of module must be dropped when loading model params
            weights_to_drop_from_model_state.update(params_just_added)
        else:
            for n, p in model.named_parameters():
                if n in params_just_added:
                    # opt1: freeze
                    if "freeze" in ft_params:
                        p.requires_grad = False
                    # opt2: tune with initial lr: fine_tune_lr
                    elif "tune" in ft_params:
                        add_tags_to_parameter(p, "tune")

        weights_already_loaded.update(current_model_param_names)

    assert model is not None, "Model is not loaded, check model_builders in yaml"
    return model, weights_to_drop_from_model_state