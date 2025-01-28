import torch

def add_tags_to_parameter(p: torch.nn.Parameter, tag: str):
    """
    Adds a tag to the 'tags' attribute of parameter p.

    Args:
        p (torch.nn.Parameter): The parameter.
        tag (str): The tag to add to the parameters.
    """
    tags = getattr(p, 'tags', [])
    tags.append(tag)
    p.tags = tags


def add_tags_to_module(model: torch.nn.Module, tag: str):
    """
    Adds a tag to the 'tags' attribute of each parameter in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        tag (str): The tag to add to the parameters.
    """
    for p in model.parameters():
        add_tags_to_parameter(p, tag)
