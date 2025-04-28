from typing import List, Union, Optional
import warnings

import torch

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin


@compile_mode("script")
class SetRequireGradsOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its gradient.

    Args:
        of: the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
        wrt: the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
    """

    def __init__(
        self,
        of: str,
        wrt: Union[str, List[str]],
        requires_grad_keys_field: str = "requires_grad_keys",
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        self.of = of
        self.requires_grad_keys_field = requires_grad_keys_field

        if isinstance(wrt, str):
            wrt = [wrt]
        self.wrt = wrt

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={of: Irreps("0e")},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # set req grad
        old_requires_grad: List[bool] = []
        for k in self.wrt:
            old_requires_grad.append(data[k].requires_grad)
            data[k].requires_grad_(True)

        old_requires_grad_tensor = torch.tensor(old_requires_grad, dtype=torch.bool, device=data[k].device)
        data[self.requires_grad_keys_field] = old_requires_grad_tensor

        return data

@compile_mode("script")
class GradientOutput(GraphModuleMixin, torch.nn.Module):
    r"""Wrap a model and include as an output its gradient.

    Args:
        of: the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
        wrt: the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
        out_field: the field in which to return the computed gradients. Defaults to ``f"d({of})/d({wrt})"`` for each field in ``wrt``.
        sign: either 1 or -1; the returned gradient is multiplied by this.
    """
    sign: float
    _negate: bool

    def __init__(
        self,
        of: str,
        wrt: Union[str, List[str]],
        out_field: Optional[List[str]] = None,
        sign: float = 1.0,
        requires_grad_keys_field: str = "requires_grad_keys",
        # Other:
        irreps_in = None,
    ):
        super().__init__()
        sign = float(sign)
        assert sign in (1.0, -1.0)
        self.sign = sign
        self._negate = sign == -1.0
        self.of = of
        self.requires_grad_keys_field = requires_grad_keys_field

        # TO DO: maybe better to force using list?
        if isinstance(wrt, str):
            wrt = [wrt]
        if isinstance(out_field, str):
            out_field = [out_field]
        self.wrt = wrt
        if out_field is None:
            self.out_field = [f"d({of})/d({e})" for e in self.wrt]
        else:
            assert len(out_field) == len(
                self.wrt
            ), "Out field names must be given for all w.r.t tensors"
            self.out_field = out_field

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={of: Irreps("0e")},
        )

        # The gradient of a single scalar w.r.t. something of a given shape and irrep just has that shape and irrep
        # Ex.: gradient of energy (0e) w.r.t. position vector (L=1) is also an L = 1 vector
        self.irreps_out.update(
            {f: self.irreps_in[wrt] for f, wrt in zip(self.out_field, self.wrt)}
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        wrt_tensors = []
        for k in self.wrt:
            wrt_tensors.append(data[k])

        # Get grads
        grads = torch.autograd.grad(
            # TODO:
            # This makes sense for scalar batch-level or batch-wise outputs, specifically because d(sum(batches))/d wrt = sum(d batch / d wrt) = d my_batch / d wrt
            # for a well-behaved example level like energy where d other_batch / d wrt is always zero. (In other words, the energy of example 1 in the batch is completely unaffect by changes in the position of atoms in another example.)
            # This should work for any gradient of energy, but could act suspiciously and unexpectedly for arbitrary gradient outputs, if they ever come up
            [data[self.of].sum()],
            wrt_tensors,
            create_graph=self.training,  # needed to allow gradients of this output during training
        )
        
        for out, grad in zip(self.out_field, grads):
            if grad is None:
                # From the docs: "If an output doesn’t require_grad, then the gradient can be None"
                raise RuntimeError("Something is wrong, gradient couldn't be computed")

            if self._negate:
                grad = torch.neg(grad)
            data[out] = grad

        # unset requires_grad_
        old_requires_grad: List[bool] = data.get(self.requires_grad_keys_field)
        for req_grad, k in zip(old_requires_grad, self.wrt):
            data[k].requires_grad_(req_grad.item())

        return data