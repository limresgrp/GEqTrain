from typing import List, Optional, Tuple, Union, Set
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

def build_concatenation_permutation(
    irreps_list: List[Irreps], device: torch.device = torch.device("cpu")
) -> Tuple[Optional[torch.Tensor], Irreps]:
    """
    Computes the permutation needed to sort a naively concatenated feature tensor.

    When features from different sources (e.g., edge and node equivariant attrs) are
    concatenated, their irreps need to be sorted for compatibility with e3nn layers.
    This function calculates the permutation indices to achieve this sorting.

    Args:
        irreps_list (List[o3.Irreps]): A list of Irreps objects corresponding to the
            features that will be concatenated, in order.
        device (torch.device): The device to create the permutation tensor on.

    Returns:
        A tuple containing:
        - Optional[torch.Tensor]: The permutation tensor. None if only one irrep is given.
        - o3.Irrereps: The final, sorted irreps of the concatenated tensor.
    """
    if not irreps_list:
        return None, Irreps("")
    if len(irreps_list) == 1:
        return None, irreps_list[0]

    # 1. Create the Irreps object for the naively concatenated features
    unsorted_irreps_list = [item for irreps in irreps_list for item in irreps]
    unsorted_irreps = Irreps(unsorted_irreps_list)

    # 2. Get the sorted irreps and the BLOCK permutation
    sorted_irreps, p_blocks, _ = unsorted_irreps.sort()

    # 3. Get the dimensions of each block in the ORIGINAL unsorted order
    dims = torch.tensor([mul * ir.dim for mul, ir in unsorted_irreps], device=device)

    # 4. Get the starting indices (offsets) of each block in the original tensor.
    offsets = torch.cumsum(torch.cat((torch.tensor([0], device=device), dims[:-1])), dim=0)

    # 5. Compute the inverse of the block permutation (argsort).
    #    This tells us which original block should go into each new position.
    arg_p_blocks = sorted(range(len(p_blocks)), key=p_blocks.__getitem__)

    # 6. Build the full element-wise permutation using the inverse block permutation.
    p_elements = torch.cat(
        [torch.arange(dims[i], device=device) + offsets[i] for i in arg_p_blocks]
    )

    return p_elements, sorted_irreps