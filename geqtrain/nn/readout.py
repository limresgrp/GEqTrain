from typing import Optional, Tuple, Union, List
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import (
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from geqtrain.nn import GraphModuleMixin, ScalarMLPFunction
from geqtrain.nn._film import FiLMFunction
from geqtrain.nn.allegro import Linear
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from geqtrain.utils import add_tags_to_module

from geqtrain.nn._heads import L0IndexedAttention, L1Scalarizer

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
        scalar_attnt: bool = True,
        num_heads: int = 32,
        dataset_mode: str = 'single', # single|ensemble
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

        in_irreps = o3.Irreps([(input_mul, ir) for _, ir in in_irreps if ir.l in input_ls])

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

        # self.use_l1_scalarizer = self.irreps_in[self.field].lmax >=1
        # if self.field == AtomicDataDict.EDGE_FEATURES_KEY: # self.field == AtomicDataDict.GRAPH_FEATURES_KEY or
        #     self.use_l1_scalarizer = False

        # --- start layer construction --- #

        self.eq_has_internal_weights = eq_has_internal_weights
        self.has_invariant_output = False # whether self outs scalars
        self.has_equivariant_output = False # whether self outs l>0

        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0

        self.n_scalars_out = out_irreps.ls.count(0)
        self.inv_readout = None
        if self.n_scalars_out > 0:
            self.has_invariant_output = True
            self.inv_readout = readout_latent( # mlp on scalars ONLY
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **readout_latent_kwargs,
            )

        # if the out irreps requested has more elements then the request number of scalars to be provided in output
        self.reshape_in: Optional[reshape_irreps] = None
        self.eq_readout = None
        if out_irreps.dim > self.n_scalars_out:
            self.has_equivariant_output = True
            eq_linear_input_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps  if ir.l>0])
            eq_linear_output_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l>0])
            self.reshape_in = reshape_irreps(eq_linear_input_irreps)
            self.eq_readout_iw, self.eq_readout = None, None
            if self.eq_has_internal_weights:
                self.eq_readout_iw = Linear( # equivariant MLP acting on l>0 ONLY
                        eq_linear_input_irreps,
                        eq_linear_output_irreps,
                        shared_weights=self.eq_has_internal_weights,
                        internal_weights=self.eq_has_internal_weights,
                        pad_to_alignment=1,
                )
                w_embs = self.eq_readout_iw.weight_numel
            else:
                self.eq_readout = Linear( # equivariant MLP acting on l>0 ONLY
                        eq_linear_input_irreps,
                        eq_linear_output_irreps,
                        shared_weights=self.eq_has_internal_weights,
                        internal_weights=self.eq_has_internal_weights,
                        pad_to_alignment=1,
                )
                w_embs = self.eq_readout.weight_numel
            self.reshape_back_features = inverse_reshape_irreps(eq_linear_output_irreps)
        elif strict_irreps:
            assert in_irreps.dim == self.n_scalars_in, (
                    f"Module input contains features with irreps that are not scalars ({in_irreps}). " +
                    f"However, the irreps of the output is composed of scalars only ({out_irreps}). "   +
                    "Please remove non-scalar features from the input, which otherwise would remain unused." +
                    f"If features come from InteractionModule, you can add the parameter 'output_ls: [0]' in the yaml config file." +
                    "If you want to allow this behavior, set 'strict_irreps=False'."
                )

        if self.has_equivariant_output and not self.eq_has_internal_weights and self.n_scalars_in > 0:
            self.weights_emb = readout_latent( # mlp on scalars, used to compute the weights of the self.eq_readout
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=w_embs,
                **readout_latent_kwargs,
            )

        # --- end layer construction --- #

        self.resnet = resnet
        self._resnet_update_coeff: Optional[torch.nn.Parameter] = None # init to None for jit
        if self.resnet:
            assert irreps_in[self.out_field] == out_irreps
            self._resnet_update_coeff = torch.nn.Parameter(torch.tensor([0.0]))
        self.out_irreps_dim = self.out_irreps.dim

        self.act_on_nodes = self.field in _NODE_FIELDS
        self.ignore_amp = ignore_amp

        if dampen:
            add_tags_to_module(self, 'dampen')

        # if self.use_l1_scalarizer:
        #     self.l1_scalarizer = L1Scalarizer(irreps_in, field=field)

        if self.field in _NODE_FIELDS:
            idx_key = 'batch'
        elif self.field in _EDGE_FIELDS: # edge can't attention; node/ensemble can
            # todo: if self.field == AtomicDataDict.edge then use edge_centers
            scalar_attnt = False
        elif self.field in _GRAPH_FIELDS:
            if dataset_mode == 'ensemble':
                idx_key = 'ensemble_index'
            else:
                scalar_attnt = False
        else:
            raise ValueError(f"Field '{self.field}' is not recognized as a valid node, edge, or graph field. Did you forget registering this field?")

        if self.n_scalars_in == 0:
            scalar_attnt = False

        if not self.has_invariant_output and self.has_equivariant_output and self.eq_has_internal_weights:
            scalar_attnt = False

        self.scalar_attnt = scalar_attnt
        self.split_index = [mul for mul, _ in irreps_in[self.field]][0]
        if self.scalar_attnt:
            assert self.n_scalars_in > 0, 'No scalars recieved for readout but scalar_attnt = True'
            self.ensemble_attnt1 = L0IndexedAttention(irreps_in=irreps_in, field=field, out_field=field, num_heads=num_heads, idx_key=idx_key, update_mlp=True)
            self.ensemble_attnt2 = L0IndexedAttention(irreps_in=irreps_in, field=field, out_field=field, num_heads=num_heads, idx_key=idx_key)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        #! ----------- COMMENT TO JIT COMPILE --------------- #
        if self.ignore_amp:
            with torch.amp.autocast('cuda', enabled=False):
                return self._forward_impl(data)
        # --------------------------------------------------- #
        return self._forward_impl(data)

    def _forward_impl(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features, out_features = self._initialize_features(data)

        if self.act_on_nodes:
            active_nodes = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])
        else:
            active_nodes = torch.arange(len(features), device=features.device)

        if self.scalar_attnt:
            features = self._apply_scalar_attnt(features, data)

        if self.has_invariant_output:
            out_features = self._handle_invariant_output(features, data, active_nodes, out_features)
        if self.has_equivariant_output and self.reshape_in is not None:
            out_features = self._handle_vectorial_output(features, data, active_nodes, out_features)
        if self.resnet: # eq. 2 from https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36329-y/MediaObjects/41467_2023_36329_MOESM1_ESM.pdf
            out_features = self._apply_residual_update(out_features, data)

        data[self.out_field] = out_features
        return data

    def _initialize_features(self, data: AtomicDataDict.Type) -> Tuple[torch.Tensor, torch.Tensor]:
        # get features from input and create empty tensor to store output
        features = data[self.field]
        out_features = torch.zeros(
            (features.shape[0], self.out_irreps_dim),
            dtype=torch.float32,
            device=features.device
        )
        return features, out_features

    def _apply_scalar_attnt(self, features, data: AtomicDataDict.Type):
        scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)
        scalars = self.ensemble_attnt1(scalars, data)
        scalars = self.ensemble_attnt2(scalars, data)
        features = torch.cat((scalars, equiv), dim=-1)
        return features

    def _handle_invariant_output(self, features, data: AtomicDataDict.Type, active_nodes, out_features) -> torch.Tensor:
        # invariant output may be present or not
        # scalarize norms and cos_similarity between l1s
        # if self.use_l1_scalarizer:
        #     data = self.l1_scalarizer(data)
        assert self.inv_readout is not None
        out_features[active_nodes, :self.n_scalars_out] += self.inv_readout(features[active_nodes, :self.n_scalars_in]) # normal mlp on scalar component (if any)
        return out_features

    def _handle_vectorial_output(self, features, data: AtomicDataDict.Type, active_nodes, out_features) -> torch.Tensor:
        assert self.eq_readout is not None
        eq_features = self.reshape_in(features[active_nodes, self.n_scalars_in:])
        if self.eq_has_internal_weights: # eq linear layer with its own inner weights
            assert self.eq_readout_iw is not None
            eq_features = self.eq_readout_iw(eq_features)
        else:
            # else the weights are computed via mlp using scalars
            weights = self.weights_emb(features[active_nodes, :self.n_scalars_in])
            assert self.eq_readout is not None
            eq_features = self.eq_readout(eq_features, weights)
        out_features[active_nodes, self.n_scalars_out:] += self.reshape_back_features(eq_features) # set features for l>=1
        return out_features

    def _apply_residual_update(self, out_features, data: AtomicDataDict.Type) -> torch.Tensor:
        assert self._resnet_update_coeff is not None
        old_features = data[self.out_field]
        _coeff = self._resnet_update_coeff.sigmoid()
        coefficient_old = torch.rsqrt(_coeff.square() + 1)
        coefficient_new = _coeff * coefficient_old
        # Residual update
        return coefficient_old * old_features + coefficient_new * out_features


@compile_mode("script")
class ReadoutModuleWithConditioning(ReadoutModule):
    '''
    '''
    def __init__(
        self,
        field: str,
        conditioning_field: str,
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
        scalar_attnt: bool = True,
        num_heads: int = 32,
    ):
        super().__init__(
            field,
            out_field,
            out_irreps,
            strict_irreps,
            readout_latent,
            readout_latent_kwargs,
            eq_has_internal_weights,
            resnet,
            dampen,
            input_ls,
            input_mul,
            output_ls,
            output_mul,
            irreps_in,
            ignore_amp,
            scalar_attnt,
            num_heads,
        )

        self.conditioning_field = conditioning_field

        # Shared MLP (pre-FiLM)
        self.film1 = FiLMFunction(
            mlp_input_dimension=self.irreps_in[self.conditioning_field].dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self.n_scalars_in,
            mlp_nonlinearity=None,
        )

        self.fc1 = ScalarMLPFunction(
            mlp_input_dimension=self.n_scalars_in,
            mlp_output_dimension=self.n_scalars_in,
            **readout_latent_kwargs,
        )

        self.film2 = FiLMFunction(
            mlp_input_dimension=self.irreps_in[self.conditioning_field].dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self.n_scalars_in,
            mlp_nonlinearity=None,
        )

        if self.has_invariant_output:
            self.film_scalar = FiLMFunction(
                mlp_input_dimension=self.irreps_in[self.conditioning_field].dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self.n_scalars_out,
                mlp_nonlinearity=None,
            )

        if self.has_equivariant_output and not self.eq_has_internal_weights and self.n_scalars_in > 0:
            self.film_vectorial = FiLMFunction(
                mlp_input_dimension=self.irreps_in[self.conditioning_field].dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self.eq_readout.weight_numel,
                mlp_nonlinearity=None,
            )

    def _initialize_features(self, data: AtomicDataDict.Type):
        # get features from input and create empty tensor to store output
        features, out_features = super()._initialize_features(data)
        conditioning = data[self.conditioning_field]
        features = self.film1(features, conditioning)
        features = self.fc1(features)
        features = self.film2(features, conditioning)
        return features, out_features

    def _handle_invariant_output(self, features, data: AtomicDataDict.Type, active_nodes, out_features):
        super()._handle_invariant_output(features, data, active_nodes, out_features)
        out_features[active_nodes, :self.n_scalars_out] = self.film_scalar(
            out_features[active_nodes, :self.n_scalars_out],
            data[self.conditioning_field],
        )
        return out_features

    def _handle_vectorial_output(self, features, data: AtomicDataDict.Type, active_nodes, out_features):
        eq_features = self.reshape_in(features[active_nodes, self.n_scalars_in:])
        if self.eq_has_internal_weights: # eq linear layer with its own inner weights
            eq_features = self.eq_readout(eq_features)
        else:
            # else the weights are computed via mlp using scalars
            weights = self.weights_emb(features[active_nodes, :self.n_scalars_in])
            weights = self.film_vectorial(weights, data[self.conditioning_field])
            eq_features = self.eq_readout(eq_features, weights)
        out_features[active_nodes, self.n_scalars_out:] += self.reshape_back_features(eq_features) # set features for l>=1
        return out_features