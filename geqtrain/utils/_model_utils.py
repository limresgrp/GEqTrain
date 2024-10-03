import torch

def add_tags_to_parameters(model: torch.nn.Module, tag: str):
    """
    Adds a tag to the 'tags' attribute of each parameter in the model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        tag (str): The tag to add to the parameters.
    """
    for p in model.parameters():
        tags = getattr(p, 'tags', [])
        tags.append(tag)
        p.tags = tags