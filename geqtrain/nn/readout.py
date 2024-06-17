from typing import Optional, Union
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.nn.allegro import Linear
from geqtrain.nn.allegro._fc import ScalarMLPFunction
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        num_types: int,
        out_irreps: Union[o3.Irreps, str],
        field: str,
        out_field: Optional[str] = None,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        per_type_bias=None,
        has_bias=False,
        eq_has_internal_weights=False,
        irreps_in=None,
    ):
        super().__init__()

        self.DTYPE = torch.get_default_dtype()

        self.field = field
        self.out_field = out_field or field
        self.has_inv_out = False
        self.has_eq_out = False
        self.has_bias = has_bias
        self.eq_has_internal_weights = eq_has_internal_weights

        in_irreps = irreps_in[field]
        out_irreps = (
            out_irreps if isinstance(out_irreps, o3.Irreps)
            else (
                o3.Irreps(out_irreps) if isinstance(out_irreps, str)
                else in_irreps
            )
        )
        self.out_irreps = out_irreps
        self.out_irreps_muls = [ir.mul for ir in out_irreps]

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                field: in_irreps,
                },
            irreps_out={
                self.out_field: out_irreps,
            }
        )

        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0

        self.n_scalars_out = out_irreps.ls.count(0)
        if self.n_scalars_out > 0:
            self.has_inv_out = True
            self.inv_readout = readout_latent(
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **readout_latent_kwargs,
            )

            if self.has_bias:
                if per_type_bias is not None:
                    assert len(per_type_bias) == num_types
                    per_type_bias = torch.tensor(per_type_bias, dtype=torch.get_default_dtype())
                else:
                    per_type_bias = torch.zeros(num_types, dtype=torch.get_default_dtype())
                self.per_type_bias = torch.nn.Parameter(per_type_bias.reshape(num_types, -1))
            else:
                self.per_type_bias = None

        if out_irreps.dim > self.n_scalars_out:
            self.has_eq_out = True
            eq_linear_input_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps  if ir.l>0])
            eq_linear_output_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l>0])
            self.reshape_in = reshape_irreps(eq_linear_input_irreps)
            self.eq_readout = Linear(
                    eq_linear_input_irreps,
                    eq_linear_output_irreps,
                    shared_weights=self.eq_has_internal_weights,
                    internal_weights=self.eq_has_internal_weights,
                    pad_to_alignment=1,
                )
            
            if not self.eq_has_internal_weights:
                self.weights_emb = readout_latent(
                    mlp_input_dimension=self.n_scalars_in,
                    mlp_output_dimension=self.eq_readout.weight_numel,
                    **readout_latent_kwargs,
                )
            self.reshape_back_features = inverse_reshape_irreps(eq_linear_output_irreps)
        else:
            assert in_irreps.dim == self.n_scalars_in, (
                f"Module input contains features with irreps which are not scalars ({in_irreps})." +
                f"However, the irreps of the output is composed of scalars only ({out_irreps})."   +
                 "Please remove non-scalar features from the input, which otherwise would remain unused."
            )
            self.reshape_in = None
        
        self.out_irreps_dim = self.out_irreps.dim

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        out_features = torch.zeros(
            (len(features), self.out_irreps_dim),
            dtype=self.DTYPE,
            device=features.device
        )

        if self.has_inv_out:
            inv_features = self.inv_readout(features[:, :self.n_scalars_in])

            if self.has_bias and self.per_type_bias is not None:
                edge_center = torch.unique(data[AtomicDataDict.EDGE_INDEX_KEY][0])
                center_species = data[AtomicDataDict.NODE_TYPE_KEY][edge_center].squeeze(dim=-1)
                inv_features[edge_center] += self.per_type_bias[center_species]
            out_features[:, :self.n_scalars_out] += inv_features

        if self.has_eq_out and self.reshape_in is not None:
            eq_features = self.reshape_in(features[:, self.n_scalars_in:])
            if self.eq_has_internal_weights:
                eq_features = self.eq_readout(eq_features)
            else:
                weights = self.weights_emb(features[:, :self.n_scalars_in])
                eq_features = self.eq_readout(eq_features, weights)
            out_features[:, self.n_scalars_out:] += self.reshape_back_features(eq_features)
        
        data[self.out_field] = out_features
        return data


'''
from typing import Optional, Union
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.nn.allegro import Linear
from geqtrain.nn.allegro._fc import ScalarMLPFunction
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        num_types: int,
        out_irreps: Union[o3.Irreps, str],
        field: str,
        out_field: Optional[str] = None,
        readout_latent=ScalarMLPFunction,
        readout_latent_kwargs={},
        per_type_bias=None,
        has_bias=True,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field or field
        self.has_inv_out = False
        self.has_eq_out = False
        self.has_bias = has_bias

        in_irreps = irreps_in[field]
        in_irreps_muls = []
        n_l_in = {}
        for mul, ir in in_irreps:
            in_irreps_muls.append(mul)
            n_l_in[ir.l] = n_l_in.get(ir.l, 0)  + 1
        assert all([in_irreps_mul == in_irreps_muls[0] for in_irreps_mul in in_irreps_muls])
        
        self.in_irreps_mul = in_irreps_muls[0]
        self.n_l_in = n_l_in
        
        assert self.n_l_in.get(0, 0) > 0
        assert self.in_irreps_mul * self.n_l_in.get(0, 0) == in_irreps.ls.count(0)
        
        out_irreps = (
            out_irreps if isinstance(out_irreps, o3.Irreps)
            else (
                o3.Irreps(out_irreps) if isinstance(out_irreps, str)
                else in_irreps
            )
        )
        out_irreps_muls = []
        n_l_out = {}
        n_out = 0
        for mul, ir in out_irreps:
            out_irreps_muls.append(mul)
            n_l_out[ir.l] = n_l_out.get(ir.l, 0) + 1
            n_out += ir.dim
        assert all([out_irreps_mul == out_irreps_muls[0] for out_irreps_mul in out_irreps_muls])
        
        self.out_irreps_mul = out_irreps_muls[0]
        self.n_l_out = n_l_out
        self.n_out = n_out
        
        assert self.out_irreps_mul * self.n_l_out.get(0, 0) == out_irreps.ls.count(0)

        # check and init irreps
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={
                field: in_irreps,
                },
            irreps_out={
                self.out_field: out_irreps,
            }
        )

        if self.n_l_out.get(0, 0) > 0:
            self.has_inv_out = True
            self.inv_readout = readout_latent(
                mlp_input_dimension=self.in_irreps_mul * self.n_l_in[0],
                mlp_output_dimension=self.out_irreps_mul * self.n_l_out[0],
                **readout_latent_kwargs,
            )

            if per_type_bias is not None:
                self.has_bias = True
                assert len(per_type_bias) == num_types
                per_type_bias = torch.tensor(per_type_bias, dtype=torch.get_default_dtype())
            elif self.has_bias:
                per_type_bias = torch.zeros(num_types, dtype=torch.get_default_dtype())
            if self.has_bias:
                self.per_type_bias = torch.nn.Parameter(per_type_bias.reshape(num_types, -1))

        if out_irreps.dim > self.out_irreps_mul * self.n_l_out.get(0, 0):
            self.has_eq_out = True
            eq_linear_input_irreps = o3.Irreps([(mul, ir) for mul, ir in in_irreps  if ir.l>0])
            eq_linear_output_irreps = o3.Irreps([(mul, ir) for mul, ir in out_irreps if ir.l>0])
            self.reshape_in = reshape_irreps(eq_linear_input_irreps)
            self.eq_readout = Linear(
                    eq_linear_input_irreps,
                    eq_linear_output_irreps,
                    shared_weights=False,
                    internal_weights=False,
                    pad_to_alignment=1,
                )
            
            self.weights_emb = readout_latent(
                mlp_input_dimension=self.in_irreps_mul * self.n_l_in[0],
                mlp_output_dimension=self.eq_readout.weight_numel,
                **readout_latent_kwargs,
            )
            self.reshape_back_features = inverse_reshape_irreps(eq_linear_output_irreps)
        else:
            assert in_irreps.dim == self.in_irreps_mul * self.n_l_in[0], (
                f"Module input contains features with irreps which are not scalars ({in_irreps})." +
                f"However, the irreps of the output is composed of scalars only ({out_irreps})."   +
                 "Please remove non-scalar features from the input, which otherwise would remain unused."
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        features = data[self.field]
        out_features = torch.zeros(
            (len(features), self.out_irreps_mul, self.n_out),
            dtype=torch.get_default_dtype(),
            device=features.device
        )

        if self.has_inv_out:
            inv_features = self.inv_readout(features[:, :, : self.n_l_in[0]].reshape(-1, self.in_irreps_mul * self.n_l_in[0]))

            if self.has_bias:
                edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
                center_species = data[AtomicDataDict.NODE_TYPE_KEY][edge_center].squeeze(dim=-1)
                inv_features[edge_center] += self.per_type_bias[center_species]
            out_features[:, :, :self.n_l_out[0]] += inv_features.reshape(-1, self.out_irreps_mul, self.n_l_out[0])

        if self.has_eq_out:
            # weights = self.weights_emb(features[:, :self.n_scalars_in])
            weights = self.weights_emb(features[:, :, :self.n_l_in[0]].reshape(-1, self.in_irreps_mul * self.n_l_in[0]))
            # eq_features = self.reshape_in(features[:, self.n_scalars_in:])
            eq_features = self.eq_readout(features[:, :, self.n_l_in[0]:], weights)
            # eq_features = self.reshape_back_features(eq_features)
            # out_features[:, self.n_scalars_out:] += eq_features
            out_features[:, :, self.n_l_out.get(0, 0):] += eq_features
        
        data[self.out_field] = out_features
        return data
'''