import math
import functools
import torch
import torch.nn.functional as F

from typing import Callable, Optional, List, Tuple, Union
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax

from einops import rearrange

from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import GraphModuleMixin
from geqtrain.utils.tp_utils import tp_path_exists

from geqtrain.nn.allegro._not_a_shitty_fc import ScalarMLPFunction
from geqtrain.nn.allegro import Contracter, MakeWeightedChannels, Linear
from geqtrain.nn.cutoffs import polynomial_cutoff

from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


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

    '''
    when the ctor of this class is called, it takes as input all the stuff that is listed in the yaml
    all the keys that are both in the arg list here and in the yaml are taken as input in the ctor of this class
    posititon is irrelevant in ctor arg
    concept from paper: this layer works on edge features: it splits edge info in 1) invariant descriptors 2) equivariant descriptors
    it processes 1 and 2 separately but
    conditions the operations in 2 using processed info coming from 1
    and conditions the operations in 1 using invariant info coming from
    idea: 2 tracks 1 handle invariants properties of sys, the other handles equivariant properties of sys
    these 2 tracks talk to each other
    the cutoff acts a weight that scales edgefeature wrt source/dist
    the angular comonent is based on displacement vectr -> it thus implies a center/cental node
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

    # internal values
    _env_builder_w_index: List[int]
    _env_builder_n_irreps: int
    _input_pad: int

    def __init__(
        self,
        # required params
        num_layers: int,
        r_max: float,
        out_irreps: Optional[Union[o3.Irreps, str]] = None,
        output_hidden_irreps: bool = False,
        avg_num_neighbors: Optional[float] = 5.0,
        # cutoffs
        PolynomialCutoff_p: float = 6,
        # general hyperparameters:
        node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
        edge_invariant_field=AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
        edge_equivariant_field=AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
        out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        env_embed_mul: int = 64,
        head_dim: int = 64,
        use_mace_product: bool = True,
        product_correlation: int = 2,
        use_interaction_attention: bool = True,

        # MLP parameters:
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},

        # Performance parameters:
        pad_to_alignment: int = 1,

        # Other:
        irreps_in=None,
    ):
        super().__init__()
        self.DTYPE = torch.get_default_dtype()

        self.use_mace_product = use_mace_product
        self.use_interaction_attention = use_interaction_attention

        # save parameters
        assert (
            num_layers >= 1
        )
        self.num_layers             = num_layers
        self.node_invariant_field   = node_invariant_field
        self.edge_invariant_field   = edge_invariant_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field              = out_field
        self.env_embed_mul          = env_embed_mul
        self.head_dim               = head_dim
        self.isqrtd                 = math.isqrt(head_dim)
        self.polynomial_cutoff_p    = float(PolynomialCutoff_p)

        # performance
        self.pad_to_alignment       = pad_to_alignment

        env_embed       = pick_mpl_function(env_embed)
        two_body_latent = pick_mpl_function(two_body_latent)
        latent          = pick_mpl_function(latent)

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.node_invariant_field,
                self.edge_invariant_field,
                self.edge_equivariant_field,
            ],
        )

        two_body_latent = functools.partial(two_body_latent,    **two_body_latent_kwargs)
        latent =          functools.partial(latent,    **latent_kwargs)
        env_embed =       functools.partial(env_embed, **env_embed_kwargs)

        self.interaction_layers = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_edge_eq_irreps = self.irreps_in[self.edge_equivariant_field]
        assert all(mul == 1 for mul, _ in input_edge_eq_irreps)

        env_embed_irreps = o3.Irreps([(self.env_embed_mul, ir) for _, ir in input_edge_eq_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / self.pad_to_alignment)) * self.pad_to_alignment
        ) - env_embed_irreps.dim

        if out_irreps is None:
            out_irreps = env_embed_irreps
        else:
            out_irreps = out_irreps if isinstance(out_irreps, o3.Irreps) else o3.Irreps(out_irreps)
        if output_hidden_irreps:
            out_irreps = o3.Irreps(
                [(self.env_embed_mul, ir) for _, ir in env_embed_irreps if ir.l in [0] + out_irreps.ls]
            )

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges
        # (possibly embedded)
        arg_irreps = env_embed_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]

        for layer_index in range(num_layers):
            ir_out = env_embed_irreps
            # Create higher order terms cause there are more TPs coming
            if layer_index == self.num_layers - 1:
                # No more TPs follow this, so only need ls that are present in out_irreps
                ir_out = out_irreps

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
        env_weighter = MakeWeightedChannels(
            irreps_in=input_edge_eq_irreps,
            multiplicity_out=self.env_embed_mul,
            pad_to_alignment=self.pad_to_alignment,
        )

        self._tp_n_scalar_outs: List[int] = []
        self._features_n_scalar_outs: List[int] = []

        # - Start Interaction Layers - #

        for layer_index, tps_irreps in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            self.interaction_layers.append(
                InteractionLayer(
                    layer_index=layer_index,
                    parent=self,
                    env_embed_irreps=env_embed_irreps,
                    tps_irreps=tps_irreps,
                    out_irreps=out_irreps,
                    product_correlation=product_correlation,
                    previous_latent_dim=self.interaction_layers[-1].latent_mlp.out_features if layer_index > 0 else None,

                    env_weighter=env_weighter,
                    two_body_latent=two_body_latent,
                    latent=latent,
                    env_embed=env_embed,

                    avg_num_neighbors=avg_num_neighbors,
                )
            )

        # - End Interaction Layers - #

        self._latent_dim = self.interaction_layers[-1].latent_mlp.out_features
        self.out_multiplicity = out_irreps[0].mul

        self.final_latent_mlp = latent(
            mlp_input_dimension=(
                # the embedded latent invariants from the previous layer(s)
                self._latent_dim
                # and the invariants extracted from the last layer's TP:
                + self.env_embed_mul * self._tp_n_scalar_outs[layer_index]
            ),
            mlp_output_dimension=self._latent_dim,
        )

        self.final_readout_mlp = latent(
            mlp_input_dimension=self._latent_dim,
            mlp_output_dimension=self.out_multiplicity * self._features_n_scalar_outs[layer_index],
        )

        self.reshape_back_features = inverse_reshape_irreps(out_irreps)

        # - End build modules - #

        out_feat_elems = []
        for irr in out_irreps:
            out_feat_elems.append(irr.ir.dim)
        self.out_feat_elems = sum(out_feat_elems)
        self.out_irreps = out_irreps

        # - layer resnet update weights -
        # We initialize to zeros, which under the sigmoid() become 0.5
        # so 1/2 * layer_1 + 1/4 * layer_2 + ...
        # note that the sigmoid of these are the factor _between_ layers
        # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
        # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
        self._latent_resnet_update_params  = torch.nn.Parameter(torch.zeros(self.num_layers, dtype=self.DTYPE))

        self.register_buffer("per_layer_cutoffs", torch.full((num_layers + 1,), r_max))
        self.register_buffer("_zero", torch.as_tensor(0.0))

        self.irreps_out.update(
            {
                self.out_field: self.out_irreps
            }
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]

        edge_attr = data[self.edge_equivariant_field]
        # pad edge_attr
        if self._input_pad > 0:
            edge_attr = torch.cat(
                (
                    edge_attr,
                    self._zero.expand(len(edge_attr), self._input_pad),
                ),
                dim=-1,
            )

        edge_invariants = data[self.edge_invariant_field]
        node_invariants = data[self.node_invariant_field]

        num_edges: int = len(edge_invariants)
        num_nodes: int = len(node_invariants)

        # Initialize state
        out_features = torch.zeros(
            (num_edges, self.out_multiplicity, self.out_feat_elems),
            dtype=self.DTYPE,
            device=edge_attr.device
        )
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=self.DTYPE,
            device=edge_attr.device,
        )
        active_edges = torch.arange(
            num_edges,
            device=edge_attr.device,
        )

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        inv_latent_cat = torch.cat(
            [
                node_invariants[edge_center],
                node_invariants[edge_neighbor],
                edge_invariants,
            ],
            dim=-1
        )
        # The nonscalar features. Initially, the edge data.
        eq_features = edge_attr

        layer_index: int = 0
        # compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        cutoff_coeffs_all = polynomial_cutoff(
            edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p
        )

        # This goes through layer0, layer1, ..., layer_max-1
        for layer_idx, layer in enumerate(self.interaction_layers):

            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            prev_mask = cutoff_coeffs[active_edges] > 0
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

            latents, inv_latent_cat, eq_features = layer(
                active_edges=active_edges,
                num_nodes=num_nodes,
                latents=latents,
                inv_latent_cat=inv_latent_cat[prev_mask],
                eq_features=eq_features[prev_mask],
                cutoff_coeffs=cutoff_coeffs[active_edges],
                edge_attr=edge_attr[active_edges],
                node_invariants=node_invariants,
                edge_invariants=edge_invariants[active_edges],
                edge_center=edge_center[active_edges],
                edge_neighbor=edge_neighbor[active_edges],

                this_layer_update_coeff=layer_update_coefficients[layer_idx - 1] if layer_idx > 0 else None
            )

        # - final layer -
        features_n_scalars = self._features_n_scalar_outs[layer_index- 1]

        # - Output equivariant values - #
        out_features[active_edges, :, features_n_scalars:] = eq_features[..., features_n_scalars:]
        out_features = self.reshape_back_features(out_features)

        # - Output invariant values - #
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = cutoff_coeffs[active_edges] > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

        # Compute latents
        new_latents = self.final_latent_mlp(inv_latent_cat[prev_mask])
        # Apply cutoff, which propagates through to everything else
        new_latents = cutoff_coeffs.unsqueeze(-1) * new_latents

        coefficient_old = torch.rsqrt(layer_update_coefficients[layer_idx].square() + 1)
        coefficient_new = layer_update_coefficients[layer_idx] * coefficient_old
        latents = torch.index_add(
            coefficient_old * latents,
            0,
            active_edges,
            coefficient_new * new_latents,
        )

        out_features[..., :self.out_multiplicity * features_n_scalars] = self.final_readout_mlp(latents)

        data[self.out_field] = out_features
        return data

    def normalize_weights(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                normalized_param = F.normalize(param, p=2, dim=0)
                # Assign normalized parameter back to the model
                param.data.copy_(normalized_param)


@compile_mode("script")
class InteractionLayer(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        layer_index: int,
        parent: InteractionModule,
        env_embed_irreps: o3.Irreps,
        tps_irreps: Tuple[o3.Irreps],
        out_irreps: o3.Irreps,
        product_correlation: int,
        previous_latent_dim: int,

        env_weighter: MakeWeightedChannels,
        two_body_latent: torch.nn.Module,
        latent: torch.nn.Module,
        env_embed: torch.nn.Module,

        avg_num_neighbors: float,
    ) -> None:
        super().__init__()

        self.layer_index = layer_index
        self.DTYPE = torch.get_default_dtype()
        self.env_embed_mul = parent.env_embed_mul
        self.head_dim = parent.head_dim
        self.isqrtd = math.isqrt(self.head_dim)

        # Make the env embed linear, which mixes eq. feats after edges scatter over nodes
        env_linear = Linear(
            env_embed_irreps,
            env_embed_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        # Perform eq. Atomic Cluster Expansion
        product = EquivariantProductBasisBlock(
            node_feats_irreps=env_embed_irreps,
            target_irreps=env_embed_irreps,
            correlation=product_correlation,
            num_elements=parent.irreps_in[parent.node_invariant_field].num_irreps,
        ) if parent.use_mace_product else None

        # Reshape back product so that you can perform tp: n m d -> n (m d)
        reshape_in_module = reshape_irreps(env_embed_irreps)

        # Make TP
        l_arg_irreps, l_out_irreps = tps_irreps
        tmp_i_out: int = 0
        instr = []
        tp_n_scalar_outs: int = 0
        full_out_irreps = []
        for i_out, (_, ir_out) in enumerate(l_out_irreps):
            for i_1, (_, ir_1) in enumerate(l_arg_irreps):
                for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                    if ir_out in ir_1 * ir_2:
                        if ir_out == SCALAR:
                            tp_n_scalar_outs += 1
                        instr.append((i_1, i_2, tmp_i_out))
                        full_out_irreps.append((self.env_embed_mul, ir_out))
                        tmp_i_out += 1
        parent._tp_n_scalar_outs.append(tp_n_scalar_outs)
        full_out_irreps = o3.Irreps(full_out_irreps)
        assert all(ir == SCALAR for _, ir in full_out_irreps[:tp_n_scalar_outs])

        # Build tensor product between env-aware node feats and edge attrs
        tp = Contracter(
            irreps_in1=o3.Irreps(
                [(self.env_embed_mul, ir) for _, ir in l_arg_irreps]
            ),
            irreps_in2=o3.Irreps(
                [(self.env_embed_mul, ir) for _, ir in env_embed_irreps]
            ),
            irreps_out=o3.Irreps(
                [(self.env_embed_mul, ir) for _, ir in full_out_irreps]
            ),
            instructions=instr,
            connection_mode=("uuu"),
            shared_weights=False,
            has_weight=False,
            normalization='norm', # 'norm' or 'component'
            pad_to_alignment=parent.pad_to_alignment,
        )

        # Make env embed mlp
        generate_n_weights = (env_weighter.weight_numel)  # the weight for the edge embedding
        generate_n_weights += self.env_embed_mul        # + the weights for the edge attention
        if self.layer_index == 0:
            # also need weights to embed the edge itself
            # this is because the 2 body latent is mixed in with the first layer
            # in terms of code
            generate_n_weights += env_weighter.weight_numel

        # the linear acts after the extractor
        linear_out_irreps = out_irreps if self.layer_index == parent.num_layers - 1 else env_embed_irreps

        _features_n_scalar_outs = linear_out_irreps.count(SCALAR) // linear_out_irreps[0].mul
        parent._features_n_scalar_outs.append(_features_n_scalar_outs)

        linear = Linear(
            full_out_irreps,
            linear_out_irreps,
            shared_weights=True,
            internal_weights=True,
            pad_to_alignment=parent.pad_to_alignment,
        )

        if self.layer_index == 0:
            # at the first layer, we have no invariants from previous TPs
            latent_mlp = two_body_latent(
                mlp_input_dimension=(
                    (
                        # Node invariants for center and neighbor (chemistry)
                        2 * parent.irreps_in[parent.node_invariant_field].num_irreps
                        # Plus edge invariants for the edge (radius).
                        + parent.irreps_in[parent.edge_invariant_field].num_irreps
                    )
                ),
                mlp_output_dimension=None,
            )
            self._latent_dim = latent_mlp.out_features
        else:
            assert previous_latent_dim is not None
            self._latent_dim = previous_latent_dim
            latent_mlp = latent(
                mlp_input_dimension=(
                    # the embedded latent invariants from the previous layer(s)
                    self._latent_dim
                    # and the invariants extracted from the last layer's TP:
                    + self.env_embed_mul * parent._tp_n_scalar_outs[self.layer_index - 1]
                ),
                mlp_output_dimension=self._latent_dim,
            )

        # the env embed MLP takes the last latent's output as input
        # and outputs enough weights for the env embedder
        env_embed_mlp = env_embed(
            mlp_input_dimension=self._latent_dim,
            mlp_output_dimension=generate_n_weights,
        )

        self.env_embed_mlp_norm = torch.nn.LayerNorm(generate_n_weights)


        # Take the node attrs and obtain a query matrix
        node_attr_to_query = ScalarMLPFunction(
            mlp_input_dimension=(
                 # Node invariants for center and neighbor (chemistry)
                2 * parent.irreps_in[parent.node_invariant_field].num_irreps
                # Plus edge invariants for the edge (radius).
                + parent.irreps_in[parent.edge_invariant_field].num_irreps
            ),
            mlp_latent_dimensions = [],
            mlp_output_dimension=self.env_embed_mul * self.head_dim,
            mlp_nonlinearity = None,
            use_norm_layer = True,
        ) if parent.use_interaction_attention else None


        # Take the node attrs and obtain a query matrix
        dot = o3.FullyConnectedTensorProduct(
            env_embed_irreps,
            env_embed_irreps,
            f"{self.env_embed_mul}x0e",
            path_normalization='path',
        ) if parent.use_interaction_attention else None
        reshape_back_tp = inverse_reshape_irreps(env_embed_irreps)

        edge_feat_to_key = ScalarMLPFunction(
            mlp_input_dimension=dot.irreps_out.dim,
            mlp_latent_dimensions = [],
            mlp_output_dimension=self.env_embed_mul * self.head_dim,
            mlp_nonlinearity = None,
            use_norm_layer = True,
        ) if parent.use_interaction_attention else None


        self.latent_mlp = latent_mlp
        self.env_embed_mlp = env_embed_mlp
        self._env_weighter = env_weighter
        self.node_attr_to_query = node_attr_to_query
        self.edge_feat_to_key = edge_feat_to_key
        self.dot = dot
        self.env_linear = env_linear
        self.product = product
        self.reshape_in_module = reshape_in_module
        self.tp = tp
        self.tp_n_scalar_out = parent._tp_n_scalar_outs[self.layer_index]
        self.linear = linear
        self.reshape_back_tp = reshape_back_tp

        self.use_interaction_attention = parent.use_interaction_attention
        self.use_mace_product = parent.use_mace_product

        self.register_buffer(
            "env_sum_normalization",
            torch.as_tensor([avg_num_neighbors]).rsqrt(),
        )

    def forward(
        self,
        active_edges,
        num_nodes,
        latents,
        inv_latent_cat,
        eq_features,
        cutoff_coeffs: torch.Tensor,
        edge_attr: torch.Tensor,
        node_invariants,
        edge_invariants,
        edge_center,
        edge_neighbor,
        this_layer_update_coeff: Optional[torch.Tensor],
    ):

        # Compute latents
        new_latents = self.latent_mlp(inv_latent_cat)

        # Apply cutoff, which propagates through to everything else
        new_latents = cutoff_coeffs.unsqueeze(-1) * new_latents

        if self.layer_index > 0:
            assert this_layer_update_coeff is not None
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
            coefficient_old = torch.rsqrt(this_layer_update_coeff.square() + 1)
            coefficient_new = this_layer_update_coeff * coefficient_old
            # Residual update
            # Note that it only runs when there are latents to resnet with, so not at the first layer
            # index_add adds only to the edges for which we have something to contribute
            latents = torch.index_add(
                coefficient_old * latents,
                0,
                active_edges,
                coefficient_new * new_latents,
            )
        else:
            # Normal (non-residual) update
            # index_copy replaces, unlike index_add
            latents = torch.index_copy(latents, 0, active_edges, new_latents)

        # From the latents, compute the weights for active edges:
        weights = self.env_embed_mlp(latents[active_edges])
        weights = self.env_embed_mlp_norm(weights)

        w_index: int = 0
        if self.layer_index == 0:
            # embed initial edge
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel
            eq_features = self._env_weighter(
                eq_features, env_w
            )  # eq_features is edge_attr

        # Extract weights for the edge attrs
        env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
        w_index += self._env_weighter.weight_numel
        emb_latent = self._env_weighter(edge_attr, env_w)

        if self.use_interaction_attention:
            # Apply attention on features
            edge_full_attr = torch.cat([
                node_invariants[edge_center],
                node_invariants[edge_neighbor],
                edge_invariants,
            ], dim=-1)

            Q = self.node_attr_to_query(edge_full_attr)
            Q = rearrange(Q, 'e (m d) -> e m d', m=self.env_embed_mul, d=self.head_dim)

            K = self.edge_feat_to_key(self.dot(self.reshape_back_tp(emb_latent), self.reshape_back_tp(eq_features)))
            K = rearrange(K, 'e (m d) -> e m d', m=self.env_embed_mul, d=self.head_dim)

            W = torch.einsum('emd,emd -> em', Q, K) * self.isqrtd

            emb_latent = torch.einsum('emd,em->emd', emb_latent, scatter_softmax(W, edge_center, dim=0))

        # Pool over all attention-weighted edge features to build node local environment embedding

        local_env_per_node = scatter(
            emb_latent,
            edge_center,
            dim=0,
            dim_size=num_nodes,
        )
        if not self.use_interaction_attention:
            local_env_per_node = local_env_per_node * self.env_sum_normalization

        active_node_centers = torch.unique(edge_center)
        local_env_per_active_atom = self.env_linear(local_env_per_node[active_node_centers])

        if self.use_mace_product:
            expanded_features_per_active_atom: torch.Tensor = self.product(
                node_feats=local_env_per_active_atom,
                node_attrs=node_invariants[active_node_centers],
            )
            expanded_features_per_active_atom = self.reshape_in_module(expanded_features_per_active_atom)

            expanded_features_per_node = torch.zeros_like(local_env_per_node, dtype=expanded_features_per_active_atom.dtype)
            expanded_features_per_node[active_node_centers] = expanded_features_per_active_atom
        else:
            expanded_features_per_node = torch.zeros_like(local_env_per_node, dtype=local_env_per_active_atom.dtype)
            expanded_features_per_node[active_node_centers] = local_env_per_active_atom


        # Copy to get per-edge
        # Large allocation, but no better way to do this:
        local_env_per_active_edge = expanded_features_per_node[edge_center]

        # Now do the TP
        # recursively tp current features with the environment embeddings
        eq_features = self.tp(eq_features, local_env_per_active_edge)

        # Get invariants
        # features has shape [z][mul][k]
        # we know scalars are first
        scalars = rearrange(eq_features[:, :, :self.tp_n_scalar_out], 'e m s -> e (m s)')

        inv_latent = torch.cat(
            [
                latents[active_edges],
                scalars,
            ],
            dim=-1
        )

        # do the linear for eq. features
        eq_features = self.linear(eq_features)
        return latents, inv_latent, eq_features

