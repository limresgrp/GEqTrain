import math
import torch

# ShiftedSoftPlus
def ShiftedSoftPlus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)

class ShiftedSoftPlusModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - math.log(2.0)

# SwiGLU
def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * x

class SwiGLUModule(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x, gate = x.chunk(2, dim=-1)
		return torch.nn.functional.silu(gate) * x