from typing import List, Optional, Tuple, Union, Set
import torch
from e3nn.o3 import Irreps

from geqtrain.data import AtomicDataDict


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
    out_irreps: Optional[Union[Irreps, str]] = None,
    output_ls: Optional[List[int]] = None,
    output_mul: Optional[Union[int, List[int]]] = None,
    default_irreps: Optional[Irreps] = None,
):
    """
    Processes and validates output irreps configuration.

    1.  If `out_irreps` is provided, it is used as the base.
    2.  If `out_irreps` is None, `default_irreps` is used as a fallback.
    3.  If `output_ls` is provided, the irreps are filtered to only include those `l` values.
    4.  `output_mul` can be an integer or a list to flexibly set multiplicities.
    """
    # 1. Determine the base irreps
    if out_irreps is None:
        if default_irreps is None:
            raise ValueError("Either `out_irreps` or `default_irreps` must be provided.")
        out_irreps = default_irreps
    else:
        out_irreps = out_irreps if isinstance(out_irreps, Irreps) else Irreps(out_irreps)

    # 2. Filter by `l` degrees if `output_ls` is specified
    if output_ls is not None:
        out_irreps = Irreps([(mul, ir) for mul, ir in out_irreps if ir.l in output_ls])

    # 3. Set multiplicities based on `output_mul`
    if output_mul is not None:
        new_irreps_list = []
        if isinstance(output_mul, int):
            # Overwrite all multiplicities with a single integer value
            for _, ir in out_irreps:
                new_irreps_list.append((output_mul, ir))
        elif isinstance(output_mul, list):
            if len(output_mul) == len(out_irreps.ls):
                # Assign multiplicity based on the list, matching `l` values
                for i, (mul, ir) in enumerate(out_irreps):
                    new_irreps_list.append((output_mul[i], ir))
            elif len(output_mul) == 2:
                # Assign first for l=0, second for l>0
                for _, ir in out_irreps:
                    new_mul = output_mul[0] if ir.l == 0 else output_mul[1]
                    new_irreps_list.append((new_mul, ir))
            else:
                raise ValueError(f"Length of `output_mul` list ({len(output_mul)}) is not compatible with the number of irreps `l`s ({len(out_irreps.ls)}).")
        else:
            raise ValueError(f"Unsupported type for `output_mul`: {type(output_mul)}. Allowed types are int or list.")
        out_irreps = Irreps(new_irreps_list)

    return out_irreps

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

    return p_elements, sorted_irreps.simplify()

def prepare_conditioning_tensors(
    data: AtomicDataDict.Type,
    conditioning_fields: List[str],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Prepares conditioning tensors for node and edge operations.

    Args:
        data (AtomicDataDict.Type): The input data dictionary.
        conditioning_fields (List[str]): A list of field names to use for conditioning.
        node_fields (List[str]): A list of keys that are considered node fields.
        edge_fields (List[str]): A list of keys that are considered edge fields.

    Returns:
        A tuple containing:
        - node_conditioning_tensor (Optional[torch.Tensor]): Concatenated tensor of node-level conditioning fields.
        - edge_conditioning_tensor (Optional[torch.Tensor]): Concatenated tensor of edge-level conditioning fields.
    """
    node_cond_tensors = []
    edge_cond_tensors = []

    if not conditioning_fields:
        return None, None

    edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
    edge_neigh = data[AtomicDataDict.EDGE_INDEX_KEY][1]
    num_nodes = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    num_edges = edge_center.shape[0]
    batch = torch.jit.annotate(Optional[torch.Tensor], None)
    if AtomicDataDict.BATCH_KEY in data:
        batch = data[AtomicDataDict.BATCH_KEY].to(device=edge_center.device, dtype=torch.long).squeeze(-1)
    else:
        batch = torch.zeros((num_nodes,), device=edge_center.device, dtype=torch.long)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

    for field in conditioning_fields:
        # Not jittable if uncommented
        # if field not in irreps_in:
        #     raise ValueError(f"Conditioning field '{field}' not found in irreps_in.")
        
        # cond_irreps = irreps_in[field]
        # if not all(ir.l == 0 for _, ir in cond_irreps):
        #     raise ValueError(f"Conditioning field '{field}' must have scalar (l=0) irreps, but got {cond_irreps}.")

        tensor = data[field]
        is_default_graph_field = (
            field == AtomicDataDict.GRAPH_ATTRS_KEY
            or field == AtomicDataDict.GRAPH_FEATURES_KEY
        )

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0) if is_default_graph_field else tensor.unsqueeze(-1)

        if is_default_graph_field or len(tensor) == num_graphs or len(tensor) == 1:
            graph_tensor = tensor
            if graph_tensor.shape[0] == 1 and num_graphs > 1:
                graph_tensor = graph_tensor.expand(num_graphs, -1)
            if graph_tensor.shape[0] != num_graphs:
                raise ValueError(
                    f"Graph conditioning field '{field}' must have shape [num_graphs, D] "
                    f"or [1, D]. Got {graph_tensor.shape}, num_graphs={num_graphs}."
                )
            node_cond_tensors.append(graph_tensor[batch])
            edge_cond_tensors.append(graph_tensor[batch[edge_center]])
        elif len(tensor) == num_nodes:
            node_cond_tensors.append(tensor)
            edge_cond_tensors.append(tensor[edge_center])
            edge_cond_tensors.append(tensor[edge_neigh])
        elif len(tensor) == num_edges:
            edge_cond_tensors.append(tensor)
        else:
            raise ValueError(
                f"Conditioning field '{field}' has incompatible shape {tensor.shape}. "
                f"Expected node-level ({num_nodes}), edge-level ({num_edges}), "
                f"or graph-level ({num_graphs}) conditioning."
            )

    node_conditioning = torch.cat(node_cond_tensors, dim=-1) if node_cond_tensors else None
    edge_conditioning = torch.cat(edge_cond_tensors, dim=-1) if edge_cond_tensors else None
    
    return node_conditioning, edge_conditioning
