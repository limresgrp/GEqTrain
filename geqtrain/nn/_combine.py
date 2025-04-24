from typing import List
import torch

from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


@compile_mode("script")
class CombineModule(GraphModuleMixin, torch.nn.Module):
    """
    """

    def __init__(
        self,
        fields: List[str],
        out_field: str,
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        assert len(fields) > 1
        
        self.fields = fields
        self.out_field = out_field
        
        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=fields,
            irreps_out={self.out_field: irreps_in[fields[0]]},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        combined = data[self.fields[0]]
        for field in self.fields[1:]:
            combined += data[field]
        data[self.out_field] = combined
        return data