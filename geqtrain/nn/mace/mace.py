from abc import abstractmethod
import math
import torch

from typing import Optional, List, Tuple, Type, Union

from e3nn import nn, o3
from e3nn.util.jit import compile_mode
# from torch_scatter import scatter_sum
from geqtrain.utils.pytorch_scatter import scatter_sum

from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    GraphModuleMixin,
)
from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps
from geqtrain.utils.tp_utils import complete_parities


@compile_mode("script")
class MACEModule(GraphModuleMixin, torch.nn.Module):
    '''
    '''
    
    def __init__(
        self,
        # required params
        num_layers: int,
        correlation: int,
        # alias:
        node_invariant_field   = AtomicDataDict.NODE_ATTRS_KEY,
        node_equivariant_field = AtomicDataDict.NODE_EQ_ATTRS_KEY,
        edge_invariant_field   = AtomicDataDict.EDGE_RADIAL_EMB_KEY,
        edge_equivariant_field = AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
        out_field              = AtomicDataDict.NODE_FEATURES_KEY,
        # hyperparams:
        latent_dim:                 int  = 64,
        mlp_latent_dimensions: List[int] = [64, 64, 64],
        avg_num_neighbors:         float = 10.0,
        # Other:
        irreps_in = None,
        name:str = "",
    ):
        super().__init__()
        self.name = name
        assert (num_layers >= 1)
        # save parameters
        self.num_layers             = num_layers
        self.node_invariant_field   = node_invariant_field
        self.node_equivariant_field = node_equivariant_field
        self.edge_invariant_field   = edge_invariant_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field              = out_field
        self.latent_dim             = latent_dim

        # init irreps
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=[self.node_invariant_field, self.edge_invariant_field, self.edge_equivariant_field])

        # compute irreps
        edge_attrs_irreps  = self.irreps_in[self.edge_equivariant_field]
        edge_attrs_irreps  = self.include_eq_input_irreps(edge_attrs_irreps)
        hidden_irreps      = o3.Irreps([(self.latent_dim, ir) for _, ir in edge_attrs_irreps]) # complete_parities(o3.Irreps([(self.latent_dim, ir) for _, ir in edge_attrs_irreps]))
        num_features       = hidden_irreps.count(o3.Irrep(0, 1)) + hidden_irreps.count(o3.Irrep(0, -1))
        node_feats_irreps  = o3.Irreps([(num_features, (0, 1))])
        node_attrs_irreps  = self.irreps_in[self.node_invariant_field]
        edge_feats_irreps  = (2 * node_feats_irreps + self.irreps_in[self.edge_invariant_field]).simplify()
        num_elements       = node_attrs_irreps.num_irreps
        interaction_irreps = hidden_irreps

        self.node_embedding = o3.Linear(
            irreps_in=node_attrs_irreps,
            irreps_out=node_feats_irreps,
        )

        inter = RealAgnosticInteractionBlock(
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=edge_attrs_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            mlp_latent_dimensions=mlp_latent_dimensions,
            sc=True,
        )
        self.interactions = torch.nn.ModuleList([inter])
        
        # Use the appropriate self connection at the first layer for proper E0
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            sc=True,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_layers):
            if i == num_layers - 1:
                hidden_irreps_out = hidden_irreps[1:2]  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = RealAgnosticInteractionBlock(
                node_attrs_irreps=node_attrs_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=edge_attrs_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                mlp_latent_dimensions=mlp_latent_dimensions,
                sc=True,
            )
            self.interactions.append(inter)
            
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                sc=True,
            )
            self.products.append(prod)
            if i == num_layers - 1:
                self.readouts.append(LinearReadoutBlock(hidden_irreps_out))
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

        # - End build modules - #
        self.irreps_out.update({self.out_field: o3.Irreps("0e")})
    
    def include_eq_input_irreps(self, edge_attrs_irreps):
        # concat node eq_input features of each atom pair to spharms vector
        if AtomicDataDict.NODE_EQ_ATTRS_KEY in self.irreps_in:
            node_eq_irreps = self.irreps_in[AtomicDataDict.NODE_EQ_ATTRS_KEY]
            
            # 1. Create the Irreps object for the naively concatenated features
            unsorted_irreps_list = list(edge_attrs_irreps) + list(node_eq_irreps) + list(node_eq_irreps)
            unsorted_irreps = o3.Irreps(unsorted_irreps_list)

            # 2. Get the sorted irreps and the BLOCK permutation
            sorted_irreps, p_blocks, _ = unsorted_irreps.sort()

            # 3. Get the dimensions of each block in the ORIGINAL unsorted order,
            #    correctly accounting for multiplicity.
            dims = torch.tensor([mul * ir.dim for mul, ir in unsorted_irreps])

            # 4. Get the starting indices (offsets) of each block in the original tensor.
            offsets = torch.cumsum(torch.cat((torch.tensor([0]), dims[:-1])), dim=0)

            # 5. Compute the inverse of the block permutation (argsort).
            #    This tells us which original block should go into each new position.
            arg_p_blocks = sorted(range(len(p_blocks)), key=p_blocks.__getitem__)
            
            # 6. Build the full element-wise permutation using the inverse block permutation.
            p_elements = torch.cat([
                torch.arange(dims[i]) + offsets[i] for i in arg_p_blocks
            ])

            # 7. Store the final, correct, element-wise permutation as a buffer
            self.register_buffer('concatenation_permutation', p_elements)
            
            # 8. Store the final sorted irreps description for later use
            self._has_node_eq_attrs = True
            return sorted_irreps
        # If there are no node features to concatenate, no permutation is needed.
        self.concatenation_permutation = None
        self._has_node_eq_attrs = False
        return edge_attrs_irreps

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_index      = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_center, edge_neighbor = edge_index

        edge_attrs      = data[self.edge_equivariant_field]
        edge_invariants = data[self.edge_invariant_field]
        node_attrs      = data[self.node_invariant_field]

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        node_feats = self.node_embedding(node_attrs)
        edge_feats = [node_feats[edge_center], node_feats[edge_neighbor], edge_invariants]
        if AtomicDataDict.EDGE_FEATURES_KEY in data:
            edge_feats += [data[AtomicDataDict.EDGE_FEATURES_KEY]]
        edge_feats = torch.cat(edge_feats, dim=-1)

        # The nonscalar features
        if self._has_node_eq_attrs and AtomicDataDict.NODE_EQ_ATTRS_KEY in data:
            node_equivariants = data[AtomicDataDict.NODE_EQ_ATTRS_KEY]

            # 1. Perform the naive concatenation.
            unsorted_features = torch.cat([
                edge_attrs, 
                node_equivariants[edge_center], 
                node_equivariants[edge_neighbor],
            ], dim=-1)

            # 2. Apply the pre-computed element-wise permutation.
            #    This `edge_attr` tensor now correctly corresponds to `edge_attrs_irreps`.
            edge_attrs = unsorted_features[:, self.concatenation_permutation]

        node_energies_list = []
        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                node_attrs=node_attrs,
                sc=sc,
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies_list.append(node_energies)
        node_energy_contributions = torch.stack(node_energies_list, dim=-1).sum(dim=-1, keepdim=True)

        data[self.out_field] = node_energy_contributions
        return data


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irrep_out: o3.Irreps = o3.Irreps("0e")):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irrep_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        mlp_latent_dimensions: List[int],
        sc: bool = False,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps     = target_irreps
        self.hidden_irreps     = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.mlp_latent_dimensions = mlp_latent_dimensions
        self.sc = sc

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.mlp_latent_dimensions + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.linear = o3.Linear(irreps_mid, self.target_irreps, internal_weights=True, shared_weights=True)

        # Selector TensorProduct
        if self.sc:
            skip_tp_irreps_in  = self.node_feats_irreps
            skip_tp_irreps_out = self.hidden_irreps
        else:
            skip_tp_irreps_in  = self.target_irreps
            skip_tp_irreps_out = self.target_irreps
            self.register_parameter("update_coeff", torch.nn.Parameter(torch.tensor([0.])))
        self.skip_tp = o3.FullyConnectedTensorProduct(skip_tp_irreps_in, self.node_attrs_irreps, skip_tp_irreps_out)
        self.reshape = reshape_irreps(self.target_irreps)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender        = edge_index[0]
        receiver      = edge_index[1]
        num_nodes     = node_feats.shape[0]
        node_feats    = self.linear_up(node_feats)
        tp_weights    = self.conv_tp_weights(edge_feats)
        mji           = self.conv_tp(node_feats[sender], edge_attrs, tp_weights)  # [n_edges, irreps]
        message       = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)  # [n_nodes, irreps]
        message       = self.linear(message) / self.avg_num_neighbors
        skip_tp_input = node_feats if self.sc else message
        sc            = self.skip_tp(skip_tp_input, node_attrs)
        if not self.sc:
            update_coeff  = torch.nn.functional.sigmoid(self.update_coeff)
            message       = apply_residual_stream(message, sc, update_coeff)
            sc = None
        return self.reshape(message), sc  # [n_nodes, channels, (lmax + 1)**2]


def apply_residual_stream(x, x_new, update_coeff):
    # At init, we assume new and old to be approximately uncorrelated
    # Thus their variances add
    # we always want the latent space to be normalized to variance = 1.0,
    # because it is critical for learnability. Still, we want to preserve
    # the _relative_ magnitudes of the current latent and the residual update
    # to be controled by `this_layer_update_coeff`
    # Solving the simple system for the two coefficients:
    #   a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
    # gives
    #   a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
    # rsqrt is reciprocal sqrt
    coefficient_old = torch.rsqrt(update_coeff.square() + 1)
    coefficient_new = update_coeff * coefficient_old
    # Residual update
    # Note that it only runs when there are latents to resnet with, so not at the first layer
    # index_add adds only to the edges for which we have something to contribute
    return coefficient_old * x + coefficient_new * x_new


# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions