import math
import torch


def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)

class ShiftedSoftPlusModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.softplus(x) - math.log(2.0)