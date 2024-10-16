import torch

from typing import Optional
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from torch_scatter import scatter

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


@compile_mode("script")
class DipoleMomentModule(GraphModuleMixin, torch.nn.Module):
    r"""
    """

    def __init__(
        self,
        func: GraphModuleMixin,
        field: str,
        out_field: str,
    ):
        super().__init__()
        self.func = func
        self.field = field
        self.out_field = out_field

        # check and init irreps
        self._init_irreps(
            irreps_in=func.irreps_out,
            my_irreps_in={
                AtomicDataDict.POSITIONS_KEY: Irreps("1o"),
                #self.field: Irreps("0e"),
                },
            irreps_out={self.out_field: Irreps("0e")},
        )

        out_dims = self.irreps_in[self.field]

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.func(data)
        data = AtomicDataDict.with_batch(data)

        # position: torch.Tensor = data[AtomicDataDict.POSITIONS_KEY]
        # partial_charges: torch.Tensor = data[self.field]
        # batch: Optional[torch.Tensor] = data.get(AtomicDataDict.BATCH_KEY)

        # data[self.out_field] = scatter(torch.einsum('nx,nc->nx', position, partial_charges), batch, dim=0).norm(dim=-1, keepdim=True)

        per_node_dipole_moment: torch.Tensor = data[self.field]
        batch: Optional[torch.Tensor] = data.get(AtomicDataDict.BATCH_KEY)

        out = per_node_dipole_moment
        if str(self.irreps_in[self.field]) == '1x0e+1x1o':
            direction = per_node_dipole_moment[:, 1:]
            row_norms = torch.norm(direction, dim=1, keepdim=True)
            versor = direction / row_norms
            scaling = per_node_dipole_moment[:, 0].unsqueeze(1)  # Shape: [N, 1]
            out = scaling * versor

        # if already 1x1o
        data[self.out_field] = scatter(out, batch, dim=0).norm(dim=-1, keepdim=True) # Atoms, 3

        return data