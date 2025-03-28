import functools
from typing import List, Optional
import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction


@compile_mode("script")
class PerNodeAttrsScaleModule(GraphModuleMixin, torch.nn.Module):
    """
    """

    def __init__(
        self,
        field: str,
        out_field: str,
        readout_latent        = ScalarMLPFunction,
        readout_latent_kwargs = {},
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field
        out_irreps = Irreps("0e")
        
        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.NODE_ATTRS_KEY],
            my_irreps_in={self.field: out_irreps},
        )

        readout_latent = functools.partial(readout_latent, **readout_latent_kwargs)
        self.latent = readout_latent(
            mlp_input_dimension=irreps_in[AtomicDataDict.NODE_ATTRS_KEY].num_irreps,
            mlp_latent_dimensions = [64, 64],
            mlp_output_dimension=out_irreps.num_irreps,
        )

        self.irreps_out.update({self.out_field: out_irreps})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        node_features = data[self.field]
        bias = self.latent(data[AtomicDataDict.NODE_ATTRS_KEY])
        data[self.out_field] = node_features + bias
        return data

@compile_mode("script")
class PerTypeScaleModule(GraphModuleMixin, torch.nn.Module):
    """
        ScaleModule applies a scaling transformation to a specified field of the input data.
        
        This module can add a per-type bias and scale by per-type standard deviation.
        The result is stored in an output field.

        Args:
            func (GraphModuleMixin): The function to apply to the data.
            field (str): The field in the input data to be scaled.
            out_field (str): The field where the output data will be stored.
            num_types (int): The number of types for the per-type bias and std.
            per_type_bias (Optional[List], optional): The per-type bias values. Defaults to None.
            per_type_std (Optional[List], optional): The per-type standard deviation values. Defaults to None.
    """

    def __init__(
        self,
        field: str,
        out_field: str,
        num_types: int,
        per_type_bias: Optional[List] = None,
        per_type_std:  Optional[List] = None,
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field

        if per_type_bias is not None:
            assert len(per_type_bias) == num_types, (
                f"Expected per_type_bias to have length {num_types}, "
                f"but got {len(per_type_bias)}"
            )
            per_type_bias = torch.tensor(per_type_bias, dtype=torch.float32)
            self.per_type_bias = torch.nn.Parameter(per_type_bias.reshape(num_types, -1))
        else:
            self.per_type_bias = None
        
        if per_type_std is not None:
            assert len(per_type_std) == num_types, (
                f"Expected per_type_std to have length {num_types}, "
                f"but got {len(per_type_std)}"
            )
            per_type_std = torch.tensor(per_type_std, dtype=torch.float32)
            self.per_type_std = torch.nn.Parameter(per_type_std.reshape(num_types, -1))
        else:
            self.per_type_std = None
        
        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                AtomicDataDict.POSITIONS_KEY: Irreps("1o"),
                self.field: Irreps("0e"),
                },
        )
        self.irreps_out.update({self.out_field: Irreps("0e")})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])
        center_species = data[AtomicDataDict.NODE_TYPE_KEY][edge_center].squeeze(dim=-1)
        node_features = data[self.field]

        # Apply per-type std scaling if available
        if self.per_type_std is not None:
            node_features[edge_center] *= self.per_type_std[center_species]

        # Apply per-type bias if available
        if self.per_type_bias is not None:
            node_features[edge_center] += self.per_type_bias[center_species]

        data[self.out_field] = node_features

        return data