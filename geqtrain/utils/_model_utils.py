from typing import List, Optional, Union, Set
import torch
from e3nn.o3 import Irreps


def add_tags_to_parameter(p: torch.nn.Parameter, tag: str):
    """
    Adds a tag to the 'tags' attribute of parameter p.
    The 'tags' attribute is stored as a set to ensure uniqueness.
    """
    if not hasattr(p, 'tags'):
        p.tags = set()
    p.tags.add(tag)


def add_tags_to_module(model: torch.nn.Module, tag: str):
    """
    Adds a tag to the 'tags' attribute of each parameter in the model.
    """
    for p in model.parameters():
        add_tags_to_parameter(p, tag)

def process_out_irreps(
    out_irreps: Optional[Union[Irreps, str]]        = None,
    output_ls : Optional[List[int]]                 = None,
    output_mul: Optional[int]                       = None,
    latent_dim: Optional[int]                       = None,
    edge_attrs_irreps: Optional[Union[Irreps, str]] = None,
):
    # if not out_irreps is specified, default to hidden irreps with degree of spharms and multiplicity of latent
    if out_irreps is None:
        out_irreps = Irreps([(latent_dim, ir) for _, ir in edge_attrs_irreps])
    else:
        out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)
    
    # - [optional] filter out_irreps l degrees
    if output_ls is None:
        output_ls = out_irreps.ls
    assert isinstance(output_ls, List)
    assert all([(l in edge_attrs_irreps.ls) for l in output_ls]), \
        f"Required output ls {output_ls} cannot be computed using l={edge_attrs_irreps.ls}"
    
    # [optional] set out_irreps multiplicity
    if output_mul is None:
        output_mul = out_irreps[0].mul
    if isinstance(output_mul, str):
        if output_mul == 'hidden':
            output_mul = latent_dim

    out_irreps = Irreps([(output_mul, ir) for _, ir in edge_attrs_irreps if ir.l in output_ls])
    return out_irreps, output_ls, output_mul
