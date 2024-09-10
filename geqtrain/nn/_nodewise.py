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
    ):
        """Sum edges into nodes."""
        super().__init__()
        self.field = field
        self.out_field = f"sum_{field}" if out_field is None else out_field

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)
        node_feat: torch.Tensor = data[self.field]
        batch: Optional[torch.Tensor] = data.get(AtomicDataDict.BATCH_KEY)
        if batch is not None:
            data[self.out_field] = scatter(node_feat, batch, dim=0)
        return data