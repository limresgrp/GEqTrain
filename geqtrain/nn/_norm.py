import torch
from typing import Optional
from e3nn.o3 import Irreps
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


class Norm(GraphModuleMixin, torch.nn.Module):
    """
    Module for computing the norm of features associated with irreducible representations (irreps).
    This class takes input features corresponding to various irreps (typically from atomic or graph data),
    and computes the norm for each irrep. For scalar irreps (l=0), the value is kept as is. For higher-order
    irreps (l>0), the norm is computed over the corresponding components (2l+1 for each l). The output is a
    tensor where each chunk corresponds to the norm of the respective irrep, resulting in a scalar output
    for each input irrep.
    Args:
        field (str): The key in the input data dictionary containing the features to normalize.
        out_field (Optional[str], optional): The key to store the output in the data dictionary. If None,
            defaults to the value of `field`.
        irreps_in (dict, optional): Dictionary mapping field names to their associated irreps.
    Attributes:
        field (str): The input field name.
        out_field (str): The output field name.
        ls (list): List of degrees (l) for each irrep in the input.
    Forward Args:
        data (AtomicDataDict.Type): Input data dictionary containing feature tensors.
    Returns:
        AtomicDataDict.Type: The input data dictionary with an additional field containing the norms
            of the input features, stored under `out_field`.
    """
    

    out_field: str

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        irreps_in={},
    ):
        """
        Initialize the Norm module.

        Args:
            field (str): The key in the input data dictionary containing the features to normalize.
            out_field (Optional[str], optional): The key to store the output in the data dictionary.
                If None, defaults to the value of `field`.
            irreps_in (dict, optional): Dictionary mapping field names to their associated irreps.

        This constructor sets up the module to compute the norm for each irrep in the input features.
        For scalar irreps (l=0), the value is kept as is. For higher-order irreps (l>0), the norm is
        computed over the corresponding components (2l+1 for each l). The output irreps are set to "0e"
        for each input irrep, indicating a scalar output for each.
        """
        super().__init__()
        self.field = field
        # Set the output field name; default to input field if not specified
        self.out_field = field if out_field is None else out_field
        # Get the input irreps for the specified field
        input_irreps = irreps_in[self.field]
        # Store the list of degrees (l) for each irrep
        self.ls = input_irreps.ls
        # For each input irrep, output is a scalar ("0e")
        output_irreps = Irreps([(m, "0e") for (m, _) in input_irreps]).simplify()

        # Initialize irreps for input and output fields
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: output_irreps}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Compute the norm of features for each irrep in the input field.

        For scalar irreps (l=0), the value is kept as is.
        For higher-order irreps (l>0), the norm is computed over the corresponding components (2l+1 for each l).
        The output is a tensor where each chunk corresponds to the norm of the respective irrep,
        resulting in a scalar output for each input irrep.

        Args:
            data (AtomicDataDict.Type): Input data dictionary containing feature tensors.

        Returns:
            AtomicDataDict.Type: The input data dictionary with an additional field containing the norms
            of the input features, stored under `out_field`.
        """
        feat: torch.Tensor = data[self.field]
        # Prepare a list to hold the output chunks for each irrep
        chunks = []
        idx = 0  # Current index in the feature tensor
        for l in self.ls:
            if l == 0:
                # For scalar irreps (l=0), take the value as is (single component)
                chunks.append(feat[..., idx:idx+1])
                idx += 1
            else:
                # For higher-order irreps (l>0), compute the norm over the 2l+1 components
                dim = 2 * l + 1
                vec = feat[..., idx:idx+dim]  # Extract the vector components
                norm = torch.norm(vec, dim=-1, keepdim=True)  # Compute the norm, keep last dim for concatenation
                chunks.append(norm)
                idx += dim
        # Concatenate all chunks along the last dimension to form the output tensor
        data[self.out_field] = torch.cat(chunks, dim=-1)
        return data