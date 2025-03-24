import torch
from typing import Optional
from torch_scatter import scatter

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


class NodewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Sum nodewise features.

    """

    out_field: str

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        irreps_in={},
        residual_field: Optional[str] = None,
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = f"sum_{field}" if out_field is None else out_field
        input_irreps = irreps_in[self.field]

        self.residual_field = residual_field
        if residual_field is not None:
            input_irreps = (input_irreps + irreps_in[residual_field]).regroup() # requires e3nn 0.5.5

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: input_irreps}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        node_feat: torch.Tensor = data[self.field]
        batch: Optional[torch.Tensor] = data.get(AtomicDataDict.BATCH_KEY)

        if self.residual_field is not None:
            node_feat = torch.cat((node_feat, data[self.residual_field]), dim=-1)

        if batch is not None:
            data[self.out_field] = scatter(node_feat, batch, dim=0)
        return data