import math
from functools import partial
import torch
import torch.nn.functional as F

from typing import Callable, Optional, List, Union
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils.tp_utils import tp_path_exists

from geqtrain.nn.allegro._fc import ScalarMLPFunction
from geqtrain.nn.allegro import Contracter, MakeWeightedChannels, Linear
from geqtrain.nn.cutoffs import polynomial_cutoff

from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps

from torch.nn import ModuleList, LayerNorm

SCALAR = o3.Irrep("0e")  # define for convinience


def pick_mpl_function(func):
    if isinstance(func, Callable):
        return func
    assert isinstance(func, str)
    if func.lower() == "ScalarMLPFunction".lower():
        return ScalarMLPFunction
    raise Exception(f"MLP Funciton {func} not implemented.")


@compile_mode("script")
class InteractionModule(GraphModuleMixin, torch.nn.Module):

    '''when the ctor of this class is called, it takes as input all the stuff that is listed in the yaml
    all the keys that are both in the arg list here and in the yaml are taken as input in the ctor of this class
    posititon is irrelevant in ctor arg
    concept from paper: this layer works on edge features: it splits edge info in 1) invariant descriptors 2) equivariant descriptors
    it processes 1 and 2 separately but
    conditions the operations in 2 using processed info coming from 1
    and conditions the operations in 1 using invariant info coming from
    idea: 2 tracks 1 handle invariants properties of sys, the other handles equivariant properties of sys
    these 2 tracks talk to each other
    the cutoff acts a weight that scales edgefeature wrt source/dist
    the angular comonent is based on displacement vectr -> it thus implices a center/cental node
    with this we have the ik weights selection for the angular track/tp
    scatter on nodes nb ij != ji
    readout

    Nomenclature and dims:

    "node_attrs"            [n_nodes, dim]      node_invariant_field            atom types (embedded?)
    "edge_radial_attrs"     [n_edge, dim]       edge_invariant_field            radial embedding of displacement vectors BESSEL
    "edge_angular_attrs"    [n_edge, dim]       edge_equivariant_field          angular embedding of displacement vectors SH
    "edge_features"         [n_edge, dim]       out_field                       edge_features are the output of interaction block
    '''


    # saved params
    num_layers: int
    node_invariant_field: str
    edge_invariant_field: str
    edge_equivariant_field: str
    out_field: str
    env_embed_mul: int
    weight_numel: int

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _input_pad: int

    def __init__(
        self,

        # required params
        num_layers: int,
        r_max: float,
        out_irreps: Optional[Union[o3.Irreps, str]] = None, # this defines the output irreps
        avg_num_neighbors: Optional[float] = None,

        # cutoffs
        PolynomialCutoff_p: float = 6,

        # general hyperparameters:
        node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
        edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
        edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
        out_field=AtomicDataDict.EDGE_FEATURES_KEY,

        env_embed_multiplicity: int = 64,
        head_dim: int = 32,
        product_correlation: int = 3,

        # * MLP parameters:
        # the {$NAME}_kwargs are taken from the yaml:
        # in the instanciate() all the keys listed in the yaml are read and compared with the names of the kwargs of self.__init__()
        # if some key of the yaml begins with {$NAME}_param_name, then param_name is loaded in one of the dict below (eg: env_embed_kwargs, latent_kwargs)
        # they are then used to call the ctor of the associated classes (eg env_embed kwdict and latent kwd are used to create their own instance of ScalarMLPFunction)
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},

        # Performance parameters:
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,

        # Other:
        irreps_in=None,
    ):
        assert (num_layers >= 1) # zero layers is "two body", but we don't need to support that fallback case
        super().__init__()
        self.num_layers, self.avg_num_neighbors = num_layers, avg_num_neighbors
        self.node_invariant_field, self.edge_invariant_field = node_invariant_field, edge_invariant_field
        self.edge_equivariant_field, self.out_field, self.head_dim = edge_equivariant_field, out_field, head_dim
        self.env_embed_mul = env_embed_multiplicity
        self.polynomial_cutoff_p = float(PolynomialCutoff_p)

        # precondition on input shapes, set up irreps, defines what are our input irreps and thus what are the info that this module needs from the irreps_dict
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.node_invariant_field,
                self.edge_invariant_field,
                self.edge_equivariant_field,
            ],
        )

        # for normalization of features: one per layer, eg: torch.as_tensor([5.] * 2) = tensor([5., 5.,)]
        self.register_buffer("env_sum_normalizations", torch.as_tensor([avg_num_neighbors] * num_layers)) #! IMPO: SHOULD BE SQRT(AVG NUM OF ATOMS IN NEIGH)
        self.register_buffer("per_layer_cutoffs",      torch.full((num_layers + 1,), r_max))

        self.register_buffer("_zero",                  torch.as_tensor(0.0))
        self.register_buffer("_coefficient_old",       torch.as_tensor(0.0))
        self.register_buffer("_coefficient_new",       torch.as_tensor(0.0))

        def _init_layer(func, _kwards):
            return partial(pick_mpl_function(func), **_kwards)

        two_body_latent = _init_layer(two_body_latent, two_body_latent_kwargs)
        env_embed       = _init_layer(env_embed, env_embed_kwargs)
        latent          = _init_layer(latent, latent_kwargs)

        self.latents        = ModuleList([]) # list of traditional MLPs that act on scalars, thus no need for them to be equivariant, acts on scalar in step 1 updates invariant edge_feature
        self.env_embed_mlps = ModuleList([]) # list of traditional MLPs that act on scalars, thus no need for them to be equivariant, acts on scalar in step 2 embeds invariant edge_feature to tp weights
        self.env_linears    = ModuleList([]) # list of equivariant linear layers to embed current scalars in, used to update node feature in local env descriptor
        self.tps            = ModuleList([]) # list of tensor products modules, the actual tensor product module
        self.linears        = ModuleList([]) # list of equivariant linear layers to embed current geom tensors, acts on geom tens that are output of tp

        # Embed to the spharm * it as mul
        input_edge_eq_irreps = self.irreps_in[self.edge_equivariant_field]
        assert all(mul == 1 for mul, _ in input_edge_eq_irreps)

        env_embed_irreps = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in input_edge_eq_irreps])
        assert (env_embed_irreps[0].ir ==SCALAR), "env_embed_irreps must start with scalars"

        # compute out_irreps
        if out_irreps is None:
            out_irreps = env_embed_irreps
        else:
            out_irreps = out_irreps if isinstance(
                out_irreps, o3.Irreps) else o3.Irreps(out_irreps)
            out_irreps = o3.Irreps([(env_embed_multiplicity, ir)
                                   for _, ir in env_embed_irreps if ir.l in [0] + out_irreps.ls])

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges, (possibly embedded)
        arg_irreps = env_embed_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]
        for layer_idx in range(num_layers):
            ir_out = env_embed_irreps
            # Create higher order terms cause there are more TPs coming
            if layer_idx == self.num_layers - 1:
                # No more TPs follow this, so only need ls that are present in out_irreps
                ir_out = o3.Irreps(
                    [ir for ir in env_embed_irreps if ir.ir.l in out_irreps.ls])

            # Prune impossible paths
            ir_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in ir_out
                    if tp_path_exists(arg_irreps, env_embed_irreps, ir)
                ]
            )

            # the argument to the next tensor product is the output of this one
            arg_irreps = ir_out
            tps_irreps.append(ir_out)
        # - end build irreps -

        # - Remove unneeded paths -
        temp_out_irreps = tps_irreps[-1]
        new_tps_irreps = [temp_out_irreps]
        for arg_irreps in reversed(tps_irreps[:-1]):
            new_arg_irreps = []
            for mul, arg_ir in arg_irreps:
                for _, env_ir in env_embed_irreps:
                    if any(i in temp_out_irreps for i in arg_ir * env_ir):
                        # arg_ir is useful: arg_ir * env_ir has a path to something we want
                        new_arg_irreps.append((mul, arg_ir))
                        # once its useful once, we keep it no matter what
                        break
            new_arg_irreps = o3.Irreps(new_arg_irreps)
            new_tps_irreps.append(new_arg_irreps)
            temp_out_irreps = new_arg_irreps

        assert len(new_tps_irreps) == len(tps_irreps)
        tps_irreps = list(reversed(new_tps_irreps))
        del new_tps_irreps

        assert tps_irreps[-1].lmax == out_irreps.lmax
        tps_irreps_in = tps_irreps[:-1]
        tps_irreps_out = tps_irreps[1:]
        del tps_irreps

        # Environment builder:
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_edge_eq_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment)

        self._n_scalar_outs: List[int] = []

        # - Build Products and TPs -
        for layer_idx, (arg_irreps, out_irreps) in enumerate(zip(tps_irreps_in, tps_irreps_out)):
            # Make the environment embed linear (not present in paper's fig 1)
            self.env_linears.append(
                Linear(
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps],
                    shared_weights=True,
                    internal_weights=True
                )
            )

            # Make TP
            tmp_i_out: int = 0
            instr = []
            n_scalar_outs: int = 0
            full_out_irreps = []
            for i_out, (_, ir_out) in enumerate(out_irreps):
                for i_1, (_, ir_1) in enumerate(arg_irreps):
                    for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                        if ir_out in ir_1 * ir_2:
                            if ir_out == SCALAR:
                                n_scalar_outs += 1
                            instr.append((i_1, i_2, tmp_i_out))
                            full_out_irreps.append(
                                (env_embed_multiplicity, ir_out))
                            tmp_i_out += 1
            full_out_irreps = o3.Irreps(full_out_irreps)
            self._n_scalar_outs.append(n_scalar_outs)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])

            tp = Contracter(
                irreps_in1=o3.Irreps([(env_embed_multiplicity, ir) for _, ir in arg_irreps]),
                irreps_in2=o3.Irreps([(env_embed_multiplicity, ir)for _, ir in env_embed_irreps]),
                irreps_out=o3.Irreps([(env_embed_multiplicity, ir) for _, ir in full_out_irreps]),
                instructions=instr,
                connection_mode=("uuu"), # non fully connected  ma tp slo tra stesse l
                shared_weights=False,
                has_weight=False,
                pad_to_alignment=pad_to_alignment,
                sparse_mode=sparse_mode,
            )

            self.tps.append(tp)

            generate_n_weights = (self._env_weighter.weight_numel) # Make env embed mlp, the weight for the edge embedding
            generate_n_weights += self.env_embed_mul # + the weights for the edge attention

            if layer_idx == 0:
                generate_n_weights += self._env_weighter.weight_numel # need weights to embed the edge itself, this is because the 2 body latent is mixed in with the first layer in terms of code

            # the linear acts after the extractor (red Linear in paper's fig1)
            self.linears.append(
                Linear(
                    full_out_irreps,
                    full_out_irreps if layer_idx == self.num_layers - 1 else env_embed_irreps,
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=pad_to_alignment
                )
            )

            if layer_idx == 0: # at layer0 no invariants from previous TPs
                self.latents.append(
                    two_body_latent( #* the call to the ctor of ScalarMLPFunction that define this module
                        mlp_input_dimension=((
                            2 * self.irreps_in[self.node_invariant_field].num_irreps  # Node invariants for center and neighbor (chemistry): Zi,Zi
                            + self.irreps_in[self.edge_invariant_field].num_irreps    # Plus edge invariants for the edge (radius): bessel(r_ij)
                            )
                        ),
                        mlp_output_dimension=None,
                        # weight_norm=True,
                        # dim=0,
                    )
                )
                self._latent_dim = self.latents[-1].out_features
            else:
                self.latents.append(
                    latent(
                        mlp_input_dimension=(
                            self.latents[-1].out_features                                   # the embedded latent invariants from the previous layer 448
                            + env_embed_multiplicity * self._n_scalar_outs[layer_idx - 1]), # and the invariants extracted from the last layer's TP:
                        mlp_output_dimension=None,
                        # weight_norm=True,
                        # dim=0,
                    )
                )

            # the env embed MLP takes the last latent's output as input and outputs enough weights for the env embedder
            self.env_embed_mlps.append(
                env_embed(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_output_dimension=generate_n_weights,
                    # weight_norm=True,
                    # dim=0,
                ))

        # -- end loop ---

        self.final_latent = latent(
            mlp_input_dimension=(
                 self.latents[-1].out_features                            # the embedded latent invariants from the previous layer(s)
                + env_embed_multiplicity * self._n_scalar_outs[layer_idx] # and the invariants extracted from the last layer's TP:
            ),
            mlp_output_dimension=env_embed_multiplicity * self._n_scalar_outs[layer_idx],
            # weight_norm=True,
            # dim=0,
        )

        self.reshape_back_features = inverse_reshape_irreps(full_out_irreps)

        # - end build modules -
        self.out_irreps = full_out_irreps

        # - layer resnet update weights -
        # We initialize to zeros, which under the sigmoid() become 0.5
        # so 1/2 * layer_1 + 1/4 * layer_2 + ...
        # note that the sigmoid of these are the factor _between_ layers
        # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
        # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
        self._latent_resnet_update_params = torch.nn.Parameter(torch.zeros(self.num_layers, dtype=torch.get_default_dtype()))

        self.irreps_out.update(  # set self output shape
            {
                self.out_field: self.out_irreps # 64x0e
            }
        )

        self.norm_layer = LayerNorm(self.latents[-1].out_features + env_embed_multiplicity * self._n_scalar_outs[layer_idx])


    def apply_residual_connection(self, layer_index, new_latents, latents, layer_update_coefficients):
        '''applies residual path
        At init, we assume new and old to be approximately uncorrelated
        Thus their variances add
        we always want the latent space to be normalized to variance = 1.0,
        because it is critical for learnability. Still, we want to preserve
        the _relative_ magnitudes of the current latent and the residual update
        to be controled by `this_layer_update_coeff`
        Solving the simple system for the two coefficients:
            a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
        gives:
            a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
        rsqrt is reciprocal sqrt
        The residual update at layer L is computed as a weighted sum above
        Note that it only runs when there are latents to resnet with, so not at the first layer'''

        this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
        self._coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
        self._coefficient_new = this_layer_update_coeff * self._coefficient_old
        return (self._coefficient_old * latents) + (self._coefficient_new * new_latents)


    def normalize_weights(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                normalized_param = F.normalize(param, p=2, dim=0)
                # Assign normalized parameter back to the model
                param.data.copy_(normalized_param)#


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # * ---- SETUP ---- #

        edge_center: int        = data[AtomicDataDict.EDGE_INDEX_KEY][0] # starting nodes idxs; shape: ([num_edges])
        edge_neighbor: int      = data[AtomicDataDict.EDGE_INDEX_KEY][1] # ending nodes idxs; shape: ([num_edges])
        edge_length: float      = data[AtomicDataDict.EDGE_LENGTH_KEY]   # edge lengths; shape: ([num_edges])
        edge_attr: float        = data[self.edge_equivariant_field]      # angular embedding of displacement vectors: SH enc Lmax=2; shape: ([num_edges, 9]) (eg 9 if Lmax=2)
        edge_invariants: float  = data[self.edge_invariant_field]        # radial embedding of displacement vectors: BESSEL(8) enc; shape: ([num_edges, 8]) 8= number of bessel for encoding
        node_invariants: int    = data[self.node_invariant_field]        # 1-hot atom types; shape ([2385, 375]), 375 number of type of atoms (look at yaml for more info)
        features: float         = edge_attr                              # The non-scalar features. Initially, the edge sh; shape: ([num_edges, 9])
        num_edges: int          = len(edge_invariants)
        num_nodes: int          = len(node_invariants)

        # Vectorized precompute per-layer treshold cutoffs, must be called in frwd cuz batch-dependant in edge_length
        cutoff_coeffs_all = polynomial_cutoff(edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p) # shape: ([num_layers x num_edges])
        # in this case the cutoff is the same for every layer -> all the num_layers are equal

        # compute the sigmoids vectorized instead of each loop, weights used of the lc of resnet update
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid() # shape: ([num_layers]), learnable, initialized as 0s -> w1-1/2, w2=1/2

        # Initialize state/container for output
        out_features = torch.zeros((num_edges, self.env_embed_mul, sum([irr.ir.dim for irr in self.out_irreps])), dtype=torch.get_default_dtype(), device=edge_attr.device) # self.env_embed_mul=64; out: 64x1o

        # * ---- FORWARD ---- #
        # input data: cat(Z_i, Z_j, ||r_ij||), only invariant info, first radial input
        latent_inputs_to_cat = torch.cat([node_invariants[edge_center], node_invariants[edge_neighbor], edge_invariants], dim = -1) # (num_edges, 8+375*2)

        layer_index: int = 0
        # iterable where idx:i returns a set of nn.Modules used in layer i
        _layers = zip(self.latents, self.env_embed_mlps, self.env_linears, self.linears, self.tps) # must be reinstanciated at each frwd, can't be stored as dmember cuz it gets "consumed"
        for latent, env_embed_mlp, env_linear, linear, tp in _layers: # iters through layer0, layer1, ..., layer_max-1

            # * step 1 : mlp -> scalings -> residual
            # latent updating (scalar part of) edge representations
            new_latents = latent(latent_inputs_to_cat) # process latents/scalars (blue track); latent: nn.ModuleList of normal MLPs (ok since it acts only on scalars) # at layer0: torch.Size([num_edges, 256])

            # Apply layer-wise cutoff scaling and normalization wrt num_neighbours scaling : feature are scaled wrt distance, and scaled wrt sqrt(avg(num_neighbours))
            cutoff_coeffs = cutoff_coeffs_all[layer_index] # proximity inductive bias
            norm_const = self.env_sum_normalizations[layer_index] # hyperparam: usually sqrt(avg(num_neighbours))
            new_latents = cutoff_coeffs.unsqueeze(-1) * new_latents * norm_const

            if layer_index == 0:
                latents = new_latents
            else:
                latents = self.apply_residual_connection(layer_index, new_latents, latents, layer_update_coefficients) # does not change shape of latents

            # * step 2 -> get tp weigths
            # weights is doubled lenght vector: 1/2 x tp weights ; 1/2 x embedding for SH
            # for each SH take a weight and weighted lc across them, computed using out of two body mlp that actually outs a doubled lenght vector
            weights = env_embed_mlp(latents) # torch.Size([num_edges, 448]): at l 0: 192 for encoding and 256 for tp

            w_index: int = 0
            if layer_index == 0:
                # split weights
                # part 1 = weights for tp
                env_w_part1 = weights.narrow(-1, w_index, self._env_weighter.weight_numel) # on last dim, selects from:w_index=0 to self._env_weighter.weight_numel elements (l0:192 els for embedding and )
                w_index += self._env_weighter.weight_numel # use later to get the part2 of weights
                features = self._env_weighter(features, env_w_part1) # used to weight the features=edge_attr=SH encodings, eq.9 of paper -> V^{ijL=0}_{nlp}; initially just angular info of input

            # part 2 = weights for the environment builder
            env_w_part2 = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            emb_latent = self._env_weighter(edge_attr, env_w_part2)  # emb of sh; edge_attr is the Y in tp, env_w_part2 is the w in tp eq.13

            # Pool over all weighted edge features to build node *local environment embedding*
            local_env_per_node = scatter(emb_latent, edge_center, dim=0, dim_size=num_nodes) # sum for k in N(i) of w_ik * SH(ik; shape: torch.Size([num_nodes, Ms, Lmax]); act as equivariant node feature vectors

            active_node_centers = edge_center.unique() # out of all the atoms in selected local env, only few are considered as source, cuz many nodes on the boundary of cutoff are not considered as source nodes
            local_env_per_active_atom = env_linear(local_env_per_node[active_node_centers]) # equivariant lin layer to update node features

            expanded_features_per_node = torch.zeros_like(local_env_per_node) # recreate initial tensor with torch.Size([num_nodes, Ms, Lmax]);
            expanded_features_per_node[active_node_centers] = local_env_per_active_atom # set into above the values of updated node features at correct (active) idxs

            # Copy to get per-edge, Large allocation, but no better way to do this
            local_env_per_active_edge = expanded_features_per_node[edge_center] # expanded_features_per_node.shape: torch.Size([2385, 64, 9]), edge_center: torch.Size([65657]) of idxs
            # copy of each equiv. node feature onto its outgoing edges. Used to: (features, local_env_per_active_edge)

            # * ---- TP ---- #
            # recursively tp current features with the environment embeddings
            features = tp(features, local_env_per_active_edge) # mixes info across different Ls between the V/equivariant updated descriptors of the sys and local env descriptors

            # * ---- GEOM TENSORs to SCALARS ---- #
            # Get invariants, features has shape [z][mul][k], scalars are first
            scalars = features[..., :self._n_scalar_outs[layer_index]].reshape(features.shape[0], -1) # this must be done BEFORE linear!

            # * ---- LINEAR ON GEOM_TENSR ---- #
            features = linear(features) # linear: Equivariant linear layer, mixes info between Ls of same freq, input features have been updated/agumented via local env descriptors

            # * ---- NEXT LAYER INPT ---- #
            # For layer2+, use the previous latents and scalars, This makes it deep
            latent_inputs_to_cat = torch.cat([latents, scalars], dim =-1)
            layer_index += 1


        # * ---- OUT OF LOOP ---- #

        # * ---- FINAL LAYER ---- #
        n_scalars = self._n_scalar_outs[layer_index- 1]

        # - output non-scalar values
        out_features[..., n_scalars:] = features[..., n_scalars:]
        out_features = self.reshape_back_features(out_features)

        # - output scalar values
        scalars = self.norm_layer(latent_inputs_to_cat)
        scalars = self.final_latent(scalars)

        out_features[:, :n_scalars * self.env_embed_mul] = scalars

        data[self.out_field] = out_features
        return data
