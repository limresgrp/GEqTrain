import math
import torch

# ShiftedSoftPlus
def ShiftedSoftPlus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)

class ShiftedSoftPlusModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ShiftedSoftPlus(x)

# SwiGLU
def SwiGLU(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * x

class SwiGLUModule(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return SwiGLU(x)

# Swish
def Swish(x):
    return x * torch.nn.functional.sigmoid(x)

class SwishModule(torch.nn.Module):
    def forward(self,x):
        return Swish(x)


def select_nonlinearity(nonlinearity: str):
    nonlinearities_dict = {
        None: None,
        "silu": torch.nn.functional.silu,
        "ssp": ShiftedSoftPlus,
        "selu": torch.nn.functional.selu,
        "relu": torch.nn.functional.relu,
        "swiglu": SwiGLU,
        "sigmoid": torch.nn.functional.sigmoid,
        "swish": Swish
    }
    non_lin_instance = nonlinearities_dict[nonlinearity]

    from e3nn.math import normalize2mom
    nonlin_const = None
    if nonlinearity == "ssp":
        nonlin_const = normalize2mom(ShiftedSoftPlus).cst
    elif nonlinearity == "swish":
        nonlin_const = normalize2mom(Swish).cst
    elif nonlinearity in ["selu", "relu", "sigmoid"]:
        nonlin_const = torch.nn.init.calculate_gain(non_lin_instance, param=None)
    elif nonlinearity == "silu":
        nonlin_const = normalize2mom(non_lin_instance).cst
    elif nonlinearity == "swiglu":
        nonlin_const = 1.55
    
    return non_lin_instance, nonlin_const

def select_nonlinearity_module(nonlinearity: str):
    non_lin_instance = None
    if nonlinearity == 'ssp': non_lin_instance = ShiftedSoftPlusModule()
    elif nonlinearity == "silu": non_lin_instance = torch.nn.SiLU()
    elif nonlinearity == "selu": non_lin_instance = torch.nn.SELU()
    elif nonlinearity == "relu": non_lin_instance = torch.nn.ReLU()
    elif nonlinearity == "swiglu": non_lin_instance = SwiGLUModule()
    elif nonlinearity == "sigmoid": non_lin_instance = torch.nn.Sigmoid()
    elif nonlinearity == "swish": non_lin_instance = SwishModule()
    elif nonlinearity: raise ValueError(f'Nonlinearity {nonlinearity} is not supported')
    return non_lin_instance