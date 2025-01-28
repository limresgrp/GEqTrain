import inspect
from typing import Optional
from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataset
from geqtrain.utils import load_callable
from geqtrain.utils import Config
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
            builders_and_params[_callable] = list(builder.values()) # for now a list but l8r will be dict for lr, freeze etc
        elif isinstance(builder, str):
            builders_and_params[builder] = {}

    return builders_and_params #! todo do list k: [k:{load:T, lr:x, freeze:y}]



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

        if not ft_params:
            weights_to_drop_from_model_state.update(current_model_param_names - weights_already_loaded)

        weights_already_loaded.update(current_model_param_names)

    return model, weights_to_drop_from_model_state