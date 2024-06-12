from typing import Optional, Union
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.nn.allegro import Linear
from geqtrain.nn.allegro._fc import ScalarMLPFunction
# from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps
from torch_scatter import scatter


@compile_mode("script")
class Head(GraphModuleMixin, torch.nn.Module):

    '''
    I want to create a module that takes as input [B,N,k]
    passes thru scalr mlp the kdimensional feat vect of each node N
    and outs  a scalar. this is a single head,

    then i need multiple heads

    '''

    def __init__(
        self,

        field: str,

        out_irreps: Union[o3.Irreps, str], # only if ur output is a geometric tensor

        irreps_in={}, # due to super ctor call
        out_field: Optional[str] = None, # on which key of the AtomicDataDict to place output

        head_function=ScalarMLPFunction,
        head_function_kwargs={},
    ):
        super().__init__()
        self.field = field
        irreps = irreps_in[field]

        # self.out_irreps = out_irreps
        # self.irreps_in = irreps_in
        self.out_field = out_field
        self.head_function = head_function

        # here take irreps_in of the data that u want to use i.e. field
        # irreps = irreps_in[field]
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={field: irreps},
            irreps_out={out_field: out_irreps},
        )
        
        irreps_muls = []
        n_l = {}
        n_dim = 0
        for mul, ir in irreps:
            irreps_muls.append(mul)
            n_l[ir.l] = n_l.get(ir.l, 0) + 1
            n_dim += ir.dim
        assert all([irreps_mul == irreps_muls[0] for irreps_mul in irreps_muls])

        self.irreps_mul = irreps_muls[0]
        self.n_l = n_l

        self.head = head_function(
                mlp_input_dimension=self.irreps_mul * self.n_l[0],
                mlp_latent_dimensions= [], #
                mlp_output_dimension=1,
                **head_function_kwargs,
            )


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # get a single feature vector by summing node features
        features = data[self.field]
        graph_feature = scatter(features, data[AtomicDataDict.BATCH_KEY], dim=0)

        # pass thru head
        data[self.out_field] = self.head(graph_feature)

        return data


