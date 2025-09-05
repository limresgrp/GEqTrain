# geqtrain/nn/_grad_output.py
from typing import List, Union, Optional, Dict

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


ORIG_GRAD_STATE = "_original_grad_state"

@compile_mode("script")
class EnableGradients(GraphModuleMixin, torch.nn.Module):
    """
    Prepares a data dictionary for gradient computation.

    This module identifies the tensors to be differentiated, stores their
    original `requires_grad` state, and then enables gradient tracking on them.
    It is designed to be used in a sequence before the core model.
    """
    def __init__(
        self,
        gradient_of: str,
        gradient_wrt: Union[str, List[str]],
        irreps_in = None,
    ):
        super().__init__()
        self.gradient_of = gradient_of

        if isinstance(gradient_wrt, str):
            gradient_wrt = [gradient_wrt]
        self.gradient_wrt = gradient_wrt
        
        self._init_irreps(irreps_in=irreps_in)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        original_requires_grad: List[bool] = []
        for key in self.gradient_wrt:
            original_requires_grad.append(data[key].requires_grad)
            data[key].requires_grad_(True)

        # Store the original states to be restored by the ComputeGradient module
        data[ORIG_GRAD_STATE] = torch.tensor(
            original_requires_grad, dtype=torch.bool, device=data[self.gradient_wrt[0]].device
        )
        return data


@compile_mode("script")
class ComputeGradient(GraphModuleMixin, torch.nn.Module):
    """
    Computes a gradient and restores the original `requires_grad` state.
    The scaling factors are now trainable `torch.nn.Parameter`s.
    """
    sign: float
    _negate: bool

    def __init__(
        self,
        gradient_of: str,
        gradient_wrt: Union[str, List[str]],
        out_field: Optional[List[str]] = None,
        sign: float = 1.0,
        scales: Optional[Dict[str, float]] = None,
        irreps_in = None,
    ):
        super().__init__()
        sign = float(sign)
        assert sign in (1.0, -1.0)
        self.sign = sign
        self._negate = sign == -1.0
        self.gradient_of = gradient_of

        if isinstance(gradient_wrt, str):
            gradient_wrt = [gradient_wrt]
        if isinstance(out_field, str):
            out_field = [out_field]
        self.gradient_wrt = gradient_wrt
        
        if out_field is None:
            self.out_field = [f"d({gradient_of})/d({e})" for e in self.gradient_wrt]
        else:
            assert len(out_field) == len(self.gradient_wrt)
            self.out_field = out_field

        # Create trainable parameters for scales
        if scales is None:
            scales = {} # Default to an empty dictionary
        
        # A ParameterDict holds trainable parameters in a JIT-safe dictionary
        self.trainable_scales = torch.nn.ParameterDict()
        for key in self.gradient_wrt:
            # Initialize with the user-provided value, or default to 1.0 (no scaling)
            initial_value = scales.get(key, 1.0)
            self.trainable_scales[key] = torch.nn.Parameter(
                torch.as_tensor(initial_value, dtype=torch.float32)
            )
        # =================================================

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={gradient_of: Irreps("0e")},
        )
        self.irreps_out.update(
            {f: self.irreps_in[wrt_key] for f, wrt_key in zip(self.out_field, self.gradient_wrt)}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        wrt_tensors = [data[k] for k in self.gradient_wrt]
        
        grads = torch.autograd.grad(
            [data[self.gradient_of].sum()],
            wrt_tensors,
            create_graph=self.training,
        )
        
        for i, grad in enumerate(grads):
            if grad is None:
                raise RuntimeError(f"Gradient of '{self.gradient_of}' wrt '{self.gradient_wrt[i]}' could not be computed.")

            if self._negate:
                grad = torch.neg(grad)
            
            wrt_field = self.gradient_wrt[i]
            grad = grad * self.trainable_scales[wrt_field]

            data[self.out_field[i]] = grad

        # Restore original requires_grad state
        original_requires_grad = data.pop(ORIG_GRAD_STATE)
        for i, required in enumerate(original_requires_grad):
            wrt_tensors[i].requires_grad_(required.item())

        return data