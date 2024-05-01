import inspect
from typing import Optional
from geqtrain.nn import GraphModuleMixin
from geqtrain.data import AtomicDataset
from geqtrain.utils import load_callable


def model_from_config(
    config,
    initialize: bool = False,
    dataset: Optional[AtomicDataset] = None,
    deploy: bool = False,
) -> GraphModuleMixin:
    """Build a model based on `config`.

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
        The build model.
    """
    # Pre-process config

    # Build
    builders = [
        load_callable(b, prefix="geqtrain.model")
        for b in config.get("model_builders", [])
    ]

    model = None

    for builder in builders:
        pnames = inspect.signature(builder).parameters
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
                raise RuntimeError(
                    f"Builder {builder.__name__} requires the dataset, initialize is true, but no dataset was provided to `model_from_config`."
                )
            params["dataset"] = dataset
        if "model" in pnames:
            if model is None:
                raise RuntimeError(
                    f"Builder {builder.__name__} asked for the model as an input, but no previous builder has returned a model"
                )
            params["model"] = model
        else:
            if model is not None:
                raise RuntimeError(
                    f"All model_builders after the first one that returns a model must take the model as an argument; {builder.__name__} doesn't"
                )
        model = builder(**params)
        # if model is not None and not isinstance(model, GraphModuleMixin):
        #     raise TypeError(
        #         f"Builder {builder.__name__} didn't return a GraphModuleMixin, got {type(model)} instead"
        #     )

    return model