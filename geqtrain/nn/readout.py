import contextlib
from typing import Optional, Union, List
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict, _NODE_FIELDS
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.nn.allegro import Linear
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.utils import add_tags_to_module

import re

def is_valid_irreps_string(irrep_string):
    r'''
    Examples:
    print(is_valid_irreps_string('64x0e+64x1o+64x2e'))  # Should return True
    print(is_valid_irreps_string('64x0e+64x1o+64x2e+128x3o'))  # Should return True
    print(is_valid_irreps_string('something'))  # Should return False
    print(is_valid_irreps_string('64x0e+64x1o+64x'))  # Should return False
    '''
    pattern = re.compile(r'^(\d+x\d+[eo](\+\d+x\d+[eo])*)$')
    return bool(pattern.match(irrep_string))


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, torch.nn.Module):
    '''
    out_irreps options evaluated in the following order:
        1) o3.Irreps obj
        2) str castable to o3.Irreps obj (eg: 1x0e)
        3) a irreps_in key (and get its o3.Irreps)
        4) same irreps of out_field (if out_field in GraphModuleMixin.irreps_in dict)
        if out_irreps=None is passed, then option 4 is triggered is valid, else 5)
        5) if none of the above: outs irreps of same size of field
        if out_irreps is not provided it takes out_irreps from yaml
    '''

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        out_irreps: Union[o3.Irreps, str] = None,
        strict_irreps: bool = True,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        eq_has_internal_weights: bool = False,
        resnet: bool = False,
        dampen: bool = False,
        input_ls:  Optional[List[int]]              = None,
        input_mul: Optional[Union[str, int]]        = None,
        output_ls:  Optional[List[int]]             = None,
        output_mul: Optional[Union[str, int]]       = None,
        irreps_in=None, # if output is only scalar, this is required
        ignore_amp: bool = False, # wheter to adopt amp or not
    ):
        super().__init__()

        # --- start definition of input/output irreps --- #

        # define input irreps
        self.field = field
        in_irreps = irreps_in[field]

        # --- start in_irreps ls --- #
        # default behavior: do not modify

        # - [optional] filter in_irreps l degrees if needed:
        if input_ls is None:
            input_ls = in_irreps.ls # all ls if none are passed
        assert isinstance(input_ls, List)

        # [optional] set in_irreps multiplicity
        if input_mul is None:
            input_mul = in_irreps[0].mul # take l=0 mul; ok since all ls have same mul

        in_irreps = o3.Irreps(
            [
                (input_mul, ir)
                for _, ir in in_irreps
                if ir.l in input_ls
            ]
        )

        # update dict
        irreps_in[field] = in_irreps

        # --- end in_irreps ls --- #

        # define output irreps
        self.out_field = out_field or field
        if isinstance(out_irreps, o3.Irreps):
            out_irreps = out_irreps # leave as it is
        elif isinstance(out_irreps, str):
            if is_valid_irreps_string(out_irreps):
                out_irreps = o3.Irreps(out_irreps) # elif eg "1x0e" has been passed, cast it
            else:
                assert out_irreps in irreps_in, f"'out_irreps' param is behaving like a key, but '{out_irreps}' is missing from irreps_in"
                out_irreps = o3.Irreps(irreps_in[out_irreps]) # othewise we expect it to be key for irreps_in[key]
        elif self.out_field in irreps_in:
            out_irreps = irreps_in[self.out_field] # outs same irreps of irreps_in[out_field]
        else:
           out_irreps = in_irreps # outs same irreps of irreps_in[field]

        # --- start out_irreps ls --- #
        # default behavior: do not modify

        # - [optional] filter out_irreps l degrees if needed:
        if output_ls is None:
            output_ls = out_irreps.ls # all ls if none are passed
        assert isinstance(output_ls, List)

        # [optional] set out_irreps multiplicity
        if output_mul is None:
            output_mul = out_irreps[0].mul # take l=0 mul; ok since all ls have same mul

        out_irreps = o3.Irreps([(output_mul, ir) for _, ir in out_irreps if ir.l in output_ls])

        # --- end out_irreps ls --- #

        self.out_irreps = out_irreps
        self.out_irreps_muls = [ir.mul for ir in out_irreps]

        # check and init irreps dict
        my_irreps_in = {field: in_irreps}
        if resnet:
            my_irreps_in.update({self.out_field: out_irreps})
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in=my_irreps_in,
            irreps_out={self.out_field: out_irreps},
        )

        # --- end definition of input/output irreps --- #


        # --- start layer construction --- #

        self.eq_has_internal_weights = eq_has_internal_weights
        self.has_invariant_output = False # whether self outs scalars
        self.has_equivariant_output = False # whether self outs l>0

        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0

        self.n_scalars_out = out_irreps.ls.count(0)
        if self.n_scalars_out > 0:
            self.has_invariant_output = True
            self.inv_readout = readout_latent( # mlp on scalars ONLY
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **readout_latent_kwargs,
            )

        # if the out irreps requested has more elements then the request number of scalars to be provided in output
        self.reshape_in: Optional[reshape_irreps] = None
        if out_irreps.dim > self.n_scalars_out:
            self.has_equivariant_output = True
            eq_linear_input_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps  if ir.l>0])
            eq_linear_output_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l>0])
            self.reshape_in = reshape_irreps(eq_linear_input_irreps)
            self.eq_readout = Linear( # equivariant MLP acting on l>0 ONLY
                    eq_linear_input_irreps,
                    eq_linear_output_irreps,
                    shared_weights=self.eq_has_internal_weights,
                    internal_weights=self.eq_has_internal_weights,
                    pad_to_alignment=1,
            )
            self.reshape_back_features = inverse_reshape_irreps(eq_linear_output_irreps)
        elif strict_irreps:
            assert in_irreps.dim == self.n_scalars_in, (
                    f"Module input contains features with irreps that are not scalars ({in_irreps}). " +
                    f"However, the irreps of the output is composed of scalars only ({out_irreps}). "   +
                    "Please remove non-scalar features from the input, which otherwise would remain unused." +
                    f"If features come from InteractionModule, you can add the parameter 'output_ls=[0]' in the constructor." +
                    "If you want to allow this behavior, set 'strict_irreps=False'."
                )

        if self.has_equivariant_output and not self.eq_has_internal_weights and self.n_scalars_in > 0:
            self.weights_emb = readout_latent( # mlp on scalars, used to compute the weights of the self.eq_readout
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.eq_readout.weight_numel,
                **readout_latent_kwargs,
            )

        # --- end layer construction --- #

        self.resnet = resnet
        self._resnet_update_coeff: Optional[torch.nn.Parameter] = None # init to None for jit
        if resnet:
            assert irreps_in[self.out_field] == out_irreps
            self._resnet_update_coeff = torch.nn.Parameter(torch.tensor([0.0]))
        self.out_irreps_dim = self.out_irreps.dim

        self.act_on_nodes = self.field in _NODE_FIELDS
        self.ignore_amp = ignore_amp

        if dampen:
            add_tags_to_module(self, 'dampen')

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.ignore_amp:
            with torch.amp.autocast('cuda', enabled=False):
                return self._forward_impl(data)
        return self._forward_impl(data)
    
    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get features from input and create empty tensor to store output
        features = data[self.field]
        out_features = torch.zeros(
            (features.shape[0], self.out_irreps_dim),
            dtype=torch.float32,
            device=features.device
        )

        if self.act_on_nodes:
            active_nodes = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])
        else:
            active_nodes = torch.arange(len(features), device=features.device)

        if self.has_invariant_output: # invariant output may be present or not
            out_features[active_nodes, :self.n_scalars_out] += self.inv_readout(features[active_nodes, :self.n_scalars_in]) # normal mlp on scalar component (if any)

        # vectorial handling
        if self.has_equivariant_output and self.reshape_in is not None:
            eq_features = self.reshape_in(features[active_nodes, self.n_scalars_in:])
            if self.eq_has_internal_weights: # eq linear layer with its own inner weights
                eq_features = self.eq_readout(eq_features)
            else:
                # else the weights are computed via mlp using scalars
                weights = self.weights_emb(features[active_nodes, :self.n_scalars_in])
                eq_features = self.eq_readout(eq_features, weights)
            out_features[active_nodes, self.n_scalars_out:] += self.reshape_back_features(eq_features) # set features for l>=1

        if self.resnet: # eq. 2 from https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36329-y/MediaObjects/41467_2023_36329_MOESM1_ESM.pdf
            assert self._resnet_update_coeff is not None
            old_features = data[self.out_field]
            _coeff = self._resnet_update_coeff.sigmoid()
            coefficient_old = torch.rsqrt(_coeff.square() + 1)
            coefficient_new = _coeff * coefficient_old
            # Residual update
            out_features = coefficient_old * old_features + coefficient_new * out_features

        data[self.out_field] = out_features
        return data