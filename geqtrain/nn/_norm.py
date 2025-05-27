import torch
from typing import Optional
from e3nn.o3 import Irreps
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


class Norm(GraphModuleMixin, torch.nn.Module):
    """Sum nodewise features.

    """

    out_field: str

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = field if out_field is None else out_field
        input_irreps = irreps_in[self.field]
        self.ls = input_irreps.ls
        # For l > 0 irreps, the norm is a scalar, so output irreps is "0e"
        output_irreps = Irreps([(m, "0e") for (m, _) in input_irreps]).simplify()

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: output_irreps}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        feat: torch.Tensor = data[self.field]
        # Split feat into chunks according to self.ls (degree of each irrep)
        chunks = []
        idx = 0
        for l in self.ls:
            if l == 0:
                # Scalar, just keep as is
                chunks.append(feat[..., idx:idx+1])
                idx += 1
            else:
                # Vector, take norm over the l*2+1 components
                dim = 2 * l + 1
                vec = feat[..., idx:idx+dim]
                norm = torch.norm(vec, dim=-1, keepdim=True)
                chunks.append(norm)
                idx += dim
        data[self.out_field] = torch.cat(chunks, dim=-1)
        return data