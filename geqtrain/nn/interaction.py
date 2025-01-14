import math
import functools
import torch
import wandb

from typing import Optional, List, Tuple, Union
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from einops.layers.torch import Rearrange

from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn import (
    GraphModuleMixin,
    SO3_LayerNorm,
    ScalarMLPFunction,
)
from geqtrain.nn.allegro import (
    Linear,
    Contracter,
    MakeWeightedChannels,
)
from geqtrain.utils.tp_utils import SCALAR, tp_path_exists
from geqtrain.nn.cutoffs import tanh_cutoff
from geqtrain.nn._film import FiLMFunction
from geqtrain.nn._edge import EdgeRadialAttrsEmbedder

from geqtrain.nn.mace.blocks import EquivariantProductBasisBlock
from geqtrain.nn.mace.irreps_tools import reshape_irreps, inverse_reshape_irreps


def log_feature_on_wandb(debug: bool, name:str, t:torch.tensor):
    if debug and wandb.run is not None:
        #todo: do we need to differenciate scalars with geom tensors?
        wandb.log({
            f"activations_dists/{name}.mean": t.mean().item(),
            f"activations_dists/{name}.std":  t.std().item(),})


@compile_mode("script")
class InteractionModule(GraphModuleMixin, torch.nn.Module):
    '''
    ctor args: match yaml keys with keys in ctor kwargs
    always outputs scalars
    Nomenclature and dims:
        "node_attrs"            [n_nodes, dim]      node_invariant_field            atom types (embedded?)
        "edge_radial_attrs"     [n_edge, dim]       edge_invariant_field            radial embedding of displacement vectors BESSEL
        "edge_angular_attrs"    [n_edge, dim]       edge_equivariant_field          angular embedding of displacement vectors SH
        "edge_features"         [n_edge, dim]       out_field                       edge_features are the output of interaction block
    '''
    num_layers: int
    node_invariant_field: str
    edge_invariant_field: str
    edge_equivariant_field: str
    env_embed_multiplicity: int
    out_field: str
    def __init__(
        self,
        # required params
        num_layers: int,
        r_max:      float,
        # optional params
        out_irreps: Optional[Union[o3.Irreps, str]] = None, #! out_irreps: if None: (yaml.latent_dim x lmax), else yaml.out_irreps
        output_ls:  Optional[List[int]]             = None,
        output_mul: Optional[Union[str, int]]       = None, #! 3 options: 1) None: don't change out_irreps mul, 2) 'hidden': mul=yaml.latent_dim, 3) int
        avg_num_neighbors: Optional[float]          = None,
        # cutoffs
        TanhCutoff_n: float = 6.,
        # alias:
        node_invariant_field   = AtomicDataDict.NODE_ATTRS_KEY,
        edge_invariant_field   = AtomicDataDict.EDGE_RADIAL_ATTRS_KEY,
        edge_equivariant_field = AtomicDataDict.EDGE_ANGULAR_ATTRS_KEY,
        out_field              = AtomicDataDict.EDGE_FEATURES_KEY,
        # hyperparams:
        latent_dim:             int  = 64,
        env_embed_multiplicity: int  = 64,
        use_attention:          bool = False,
        head_dim:               int  = 16,
        use_mace_product:       bool = False,
        product_correlation:    int  = 2,
        # MLP parameters:
        two_body_latent        = ScalarMLPFunction,
        two_body_latent_kwargs = {},
        env_embed              = ScalarMLPFunction,
        env_embed_kwargs       = {},
        latent                 = ScalarMLPFunction,
        latent_kwargs          = {},
        # Graph conditioning
        graph_conditioning_field=AtomicDataDict.GRAPH_ATTRS_KEY,
        # Other:
        irreps_in=None,
        debug: bool = False,
        name:str = "",
        learn_cutoff_bias: bool = True,
    ):
        super().__init__()
        self.name = name
        assert (num_layers >= 1)
        self.debug = debug
        # save parameters
        self.num_layers             = num_layers
        self.node_invariant_field   = node_invariant_field
        self.edge_invariant_field   = edge_invariant_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field              = out_field
        self.env_embed_multiplicity = env_embed_multiplicity
        self.latent_dim             = latent_dim
        self.head_dim               = head_dim
        self.isqrtd                 = math.isqrt(head_dim)
        self.tanh_cutoff_n          = float(TanhCutoff_n)
        self.learn_cutoff_bias      = learn_cutoff_bias
        # architectural choices
        self.use_attention          = use_attention
        self.use_mace_product       = use_mace_product
        # set up irreps
        self._init_irreps(irreps_in=irreps_in, required_irreps_in=[self.node_invariant_field, self.edge_invariant_field, self.edge_equivariant_field])

        two_body_latent = functools.partial(two_body_latent, **two_body_latent_kwargs)
        latent          = functools.partial(latent,          **latent_kwargs)
        env_embed       = functools.partial(env_embed,       **env_embed_kwargs)

        # Embed to the spharm * it as mul
        input_edge_eq_irreps = self.irreps_in[self.edge_equivariant_field]
        assert all(mul == 1 for mul, _ in input_edge_eq_irreps)

        env_embed_irreps = o3.Irreps([(self.env_embed_multiplicity, ir) for _, ir in input_edge_eq_irreps])
        assert (env_embed_irreps[0].ir == SCALAR), "env_embed_irreps must start with scalars"

        # if not out_irreps is specified, default to hidden irreps with degree of spharms and multiplicity of latent
        if out_irreps is None: out_irreps = o3.Irreps([(self.latent_dim, ir) for _, ir in input_edge_eq_irreps])
        else: out_irreps = out_irreps if isinstance(out_irreps, o3.Irreps) else o3.Irreps(out_irreps)
        if 0 not in out_irreps.ls: # add scalar (l=0) if missing from out_irreps
            out_irreps = o3.Irreps([(mul, o3.Irrep('0e')) for mul, _ in out_irreps[:1]]) + out_irreps

        # - [optional] filter out_irreps l degrees
        if output_ls is None:
            output_ls = out_irreps.ls + [0]
        assert isinstance(output_ls, List)
        assert all([(l in input_edge_eq_irreps.ls) for l in output_ls]), \
            f"Required output ls {output_ls} cannot be computed using l_max={max(input_edge_eq_irreps.ls)}"

        # [optional] set out_irreps multiplicity
        if output_mul is None: output_mul = out_irreps[0].mul
        if isinstance(output_mul, str):
            if output_mul == 'hidden':
                output_mul = self.latent_dim

        #! the interaction layer always keeps l=0, even if requested out is: l>0
        out_irreps = o3.Irreps([(output_mul, ir) for _, ir in input_edge_eq_irreps if ir.l in output_ls])
        self.out_multiplicity = output_mul

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges (possibly embedded)
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
            ir_out = o3.Irreps([(mul, ir) for mul, ir in ir_out if tp_path_exists(arg_irreps, env_embed_irreps, ir)])
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
        env_weighter = MakeWeightedChannels(irreps_in=input_edge_eq_irreps, multiplicity_out=self.env_embed_multiplicity)

        # - Layer resnet update weights - #
        # We initialize to zeros, which under the sigmoid() become 0.5
        # so 1/2 * layer_1 + 1/4 * layer_2 + ...
        # note that the sigmoid of these are the factor _between_ layers
        # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
        # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
        self._latent_resnet_update_params = torch.nn.Parameter(torch.zeros(self.num_layers, dtype=torch.float32))
        self.register_buffer("per_layer_cutoffs", torch.full((num_layers + 1,), r_max))
        self.register_buffer("_zero", torch.as_tensor(0.0))

        # - Start Interaction Layers - #
        self.interaction_layers = torch.nn.ModuleList([])
        self._tp_n_scalar_outs: List[int] = []

        for layer_index, tps_irreps in enumerate(zip(tps_irreps_in, tps_irreps_out)):
            is_last_layer = layer_index == self.num_layers - 1

            self.linear_out_irreps = o3.Irreps(
                [(mul, ir) for mul, ir in out_irreps if ir.l > 0]
            ) if is_last_layer else env_embed_irreps

            self.interaction_layers.append(
                InteractionLayer(
                    layer_index=layer_index,
                    is_last_layer=is_last_layer,
                    parent=self,
                    env_embed_irreps=env_embed_irreps,
                    tps_irreps=tps_irreps,
                    linear_out_irreps=self.linear_out_irreps,
                    product_correlation=product_correlation,
                    previous_latent_dim=self.interaction_layers[-1].latent_mlp.out_features if layer_index > 0 else None,
                    graph_conditioning_field=graph_conditioning_field,

                    env_weighter=env_weighter,
                    two_body_latent=two_body_latent,
                    latent=latent,
                    env_embed=env_embed,

                    avg_num_neighbors=avg_num_neighbors,
                )
            )

        if self.linear_out_irreps: self.last_eq_norm = SO3_LayerNorm(self.linear_out_irreps)

        # - End Interaction Layers - #

        # Equivariant out features
        self.reshape_back_features = inverse_reshape_irreps(out_irreps)
        self.has_scalar_output, self.final_latent_mlp, self.final_readout_mlp = False, None, None
        self.out_n_scalars = out_irreps.count(SCALAR) // self.out_multiplicity
        if self.out_n_scalars > 0:
            self.has_scalar_output = True
            # Invariant out features
            self.final_latent_mlp = latent(
                # embedded latent invariants from the previous layer(s) + invariants extracted from the last layer's TP
                mlp_input_dimension=(self.latent_dim + self.env_embed_multiplicity * self._tp_n_scalar_outs[layer_index]),
                mlp_output_dimension=self.latent_dim,
            )

            self.final_readout_mlp = latent(
                mlp_input_dimension=self.latent_dim,
                mlp_output_dimension=self.out_multiplicity * self.out_n_scalars,
            )

            if self.learn_cutoff_bias:
                # self.rbf_embedder = EdgeRadialAttrsEmbedder(
                #     in_dim=self.irreps_in[self.edge_invariant_field].num_irreps,
                #     out_dim=self.final_latent_mlp.out_features
                # )
                self.rbf_embedder = FiLMFunction(
                    mlp_input_dimension=self.irreps_in[self.edge_invariant_field].num_irreps,
                    mlp_latent_dimensions=[2*self.irreps_in[self.edge_invariant_field].num_irreps],
                    mlp_output_dimension=self.final_latent_mlp.out_features,
                    mlp_nonlinearity='swiglu',
                    zero_init_last_layer_weights=False,
                    has_bias=False,
                    final_non_lin='sigmoid'
                )

        # - End build modules - #
        out_feat_elems = []
        for irr in out_irreps: out_feat_elems.append(irr.ir.dim)
        self.out_feat_elems = sum(out_feat_elems)
        self.out_irreps = out_irreps
        self.irreps_out.update({self.out_field: self.out_irreps})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center     = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor   = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_length     = data[AtomicDataDict.EDGE_LENGTH_KEY]
        edge_attr       = data[self.edge_equivariant_field]
        edge_invariants = data[self.edge_invariant_field]
        node_invariants = data[self.node_invariant_field]
        num_edges: int  = len(edge_invariants)
        num_nodes: int  = len(node_invariants)

        # Initialize state
        out_features = torch.zeros((num_edges, self.out_multiplicity, self.out_feat_elems), dtype=torch.float32, device=edge_attr.device)
        latents = torch.zeros((num_edges, self.latent_dim), dtype=torch.float32, device=edge_attr.device)
        active_edges = torch.arange(num_edges,device=edge_attr.device)

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        inv_latent_cat = torch.cat([node_invariants[edge_center], node_invariants[edge_neighbor], edge_invariants], dim=-1)
        # The nonscalar features. Initially, the edge data.
        eq_features = edge_attr

        # Compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        cutoff_coeffs_all = tanh_cutoff(edge_length, self.per_layer_cutoffs, n=self.tanh_cutoff_n)

        for layer_index, layer in enumerate(self.interaction_layers):

            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]

            latents, inv_latent_cat, eq_features = layer(
                data=data,
                active_edges=active_edges,
                num_nodes=num_nodes,
                latents=latents,
                inv_latent_cat=inv_latent_cat,
                eq_features=eq_features,
                cutoff_coeffs=cutoff_coeffs,
                edge_attr=edge_attr,
                node_invariants=node_invariants,
                edge_invariants=edge_invariants,
                edge_center=edge_center,
                edge_neighbor=edge_neighbor,
                this_layer_update_coeff=layer_update_coefficients[layer_index - 1] if layer_index > 0 else None,
            )

        # --- final layer --- #

        # - Output equivariant values - #
        if eq_features is not None:
            eq_features = self.last_eq_norm(eq_features)
            out_features[..., self.out_n_scalars:] = eq_features

        out_features = self.reshape_back_features(out_features)

        # - Output invariant values - #
        if self.has_scalar_output:
            # update latents
            new_latents = self.final_latent_mlp(inv_latent_cat)

            # apply cutoff bias
            if self.learn_cutoff_bias:
                new_latents = self.rbf_embedder(new_latents, data[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY])
            else:
                cutoff_coeffs = cutoff_coeffs_all[layer_index + 1]
                new_latents[:, :new_latents.size(1)//2] = cutoff_coeffs.unsqueeze(-1) * new_latents[:, :new_latents.size(1)//2]

            # apply residual stream normalization
            coefficient_old = torch.rsqrt(layer_update_coefficients[layer_index].square() + 1)
            coefficient_new = layer_update_coefficients[layer_index] * coefficient_old
            latents_old = coefficient_old * latents
            latents_new = coefficient_new * new_latents
            latents = torch.index_add(
                latents_old,
                0,
                active_edges,
                latents_new,
            )

            # last update on residued features
            updated_latents_scalars_only = self.final_readout_mlp(latents)
            out_features[..., :self.out_multiplicity * self.out_n_scalars] = updated_latents_scalars_only

        data[self.out_field] = out_features

        log_feature_on_wandb(self.debug, f"{self.name}.out_features.prev_layer", latents_old)
        log_feature_on_wandb(self.debug, f"{self.name}.out_features.this_layer", latents_new)
        log_feature_on_wandb(self.debug, f"{self.name}.out_features.updated_latents_scalars_only", updated_latents_scalars_only)
        if eq_features is not None: log_feature_on_wandb(self.debug, f"{self.name}.out_features.equiv_only", eq_features)
        log_feature_on_wandb(self.debug, f"{self.name}.out_features", out_features)

        return data


@compile_mode("script")
class InteractionLayer(torch.nn.Module):
    def __init__(
        self,
        layer_index: int,
        is_last_layer: bool,
        parent: InteractionModule,
        env_embed_irreps: o3.Irreps,
        tps_irreps: Tuple[o3.Irreps],
        linear_out_irreps: o3.Irreps,
        product_correlation: int,
        previous_latent_dim: Optional[int],
        graph_conditioning_field: str,
        env_weighter: MakeWeightedChannels,
        two_body_latent: torch.nn.Module,
        latent: torch.nn.Module,
        env_embed: torch.nn.Module,
        avg_num_neighbors: float,
        avg_num_neighbors_is_learnable: bool = True,
    ) -> None:
        super().__init__()
        #! cannot store self.parent = parent due to nn recursive loops
        self.parent_name = parent.name
        self.debug = parent.debug
        self.layer_index = layer_index
        self.is_last_layer = is_last_layer
        self.env_embed_multiplicity = parent.env_embed_multiplicity
        self.latent_dim = parent.latent_dim
        self.head_dim = parent.head_dim
        self.isqrtd = math.isqrt(self.head_dim)
        self.use_attention = parent.use_attention
        self.use_mace_product = parent.use_mace_product
        self.learn_cutoff_bias = parent.learn_cutoff_bias

        # Make the env embed linear, which mixes eq. feats after edges scatter over nodes
        self.env_norm = SO3_LayerNorm(env_embed_irreps)
        self.env_linear = Linear(
            env_embed_irreps,
            env_embed_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        self.product, self.reshape_in_module = None, None
        if self.use_mace_product:
            # Perform eq. Atomic Cluster Expansion
            self.product = EquivariantProductBasisBlock(
                node_feats_irreps=env_embed_irreps,
                target_irreps=env_embed_irreps,
                correlation=product_correlation,
                num_elements=parent.irreps_in[parent.node_invariant_field].num_irreps,
            )

            # Reshape back product so that you can perform tp: n m d -> n (m d)
            self.reshape_in_module = reshape_irreps(env_embed_irreps)

        # Make TP
        l_arg_irreps, l_out_irreps = tps_irreps
        tmp_i_out: int = 0
        instr = []
        tp_n_scalar_outs: int = 0
        full_out_irreps = []
        for i_out, (_, ir_out) in enumerate(l_out_irreps):
            for i_1, (_, ir_1) in enumerate(l_arg_irreps):
                for i_2, (_, ir_2) in enumerate(env_embed_irreps):
                    if ir_out in ir_1 * ir_2: # checks if this L can be obtained via tp between the 2 considered irreps
                        if ir_out == SCALAR:
                            tp_n_scalar_outs += 1 # count number of scalars
                        instr.append((i_1, i_2, tmp_i_out))
                        full_out_irreps.append((self.env_embed_multiplicity, ir_out))
                        tmp_i_out += 1
        parent._tp_n_scalar_outs.append(tp_n_scalar_outs)
        full_out_irreps = o3.Irreps(full_out_irreps)
        assert all(ir == SCALAR for _, ir in full_out_irreps[:tp_n_scalar_outs])

        # Build tensor product between env-aware node feats and edge attrs
        self.eq_features_irreps = o3.Irreps([(self.env_embed_multiplicity, ir) for _, ir in l_arg_irreps])
        self.eq_features_irreps_norm = SO3_LayerNorm(self.eq_features_irreps)
        self.tp = Contracter(
            irreps_in1=self.eq_features_irreps,
            irreps_in2=o3.Irreps([(self.env_embed_multiplicity, ir) for _, ir in env_embed_irreps]),
            irreps_out=o3.Irreps([(self.env_embed_multiplicity, ir) for _, ir in full_out_irreps]),
            instructions=instr,
            connection_mode=("uuu"),
            shared_weights=False,
            has_weight=False,
            normalization='component', # 'norm' or 'component'
        )
        self.tp_norm = SO3_LayerNorm(o3.Irreps([(self.env_embed_multiplicity, ir) for _, ir in full_out_irreps]))
        self.rearrange_scalars = Rearrange('e m s -> e (m s)')

        # Make env embed mlp
        generate_n_weights = (env_weighter.weight_numel)  # the weight for the edge embedding
        generate_n_weights += self.env_embed_multiplicity # + the weights for the edge attention

        if self.layer_index == 0:
            # also need weights to embed the edge itself
            # this is because the 2 body latent is mixed in with the first layer
            # in terms of code
            generate_n_weights += env_weighter.weight_numel

        # FiLM layer for conditioning on graph input features
        self.film = None
        self.graph_conditioning_field = graph_conditioning_field
        if self.graph_conditioning_field in parent.irreps_in:
            self.film = FiLMFunction(
                mlp_input_dimension=parent.irreps_in[self.graph_conditioning_field].dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=generate_n_weights,
                mlp_nonlinearity=None,
            )

        self.linear = Linear(
            irreps_in=o3.Irreps([(mul, ir) for mul, ir in full_out_irreps if ir.l in linear_out_irreps.ls]),
            irreps_out=linear_out_irreps,
            internal_weights=True,
            shared_weights=True,
        ) if len(set(linear_out_irreps.ls)) > 0 else None
        self.latest_linear_out_irreps = linear_out_irreps

        if self.layer_index == 0:
            assert previous_latent_dim is None
            # at the first layer, we have no invariants from previous TPs
            self.latent_mlp = two_body_latent(
                mlp_input_dimension=(
                    (
                        # Node invariants for center and neighbor (chemistry)
                        2 * parent.irreps_in[parent.node_invariant_field].num_irreps
                        # Plus edge invariants for the edge (radius).
                        + parent.irreps_in[parent.edge_invariant_field].num_irreps
                    )
                ),
                mlp_output_dimension=self.latent_dim,
            )
        else:
            assert previous_latent_dim is not None
            self.latent_dim = previous_latent_dim
            self.latent_mlp = latent(
                mlp_input_dimension=(
                    # the embedded latent invariants from the previous layer(s)
                    self.latent_dim
                    # and the invariants extracted from the last layer's TP:
                    + self.env_embed_multiplicity * parent._tp_n_scalar_outs[self.layer_index - 1]
                ),
                mlp_output_dimension=self.latent_dim,
            )

        self.post_norm = torch.nn.LayerNorm(self.latent_dim)
        # the env embed MLP takes the last latent's output as input and outputs enough weights for the env embedder
        self.env_embed_mlp = env_embed(
          mlp_input_dimension=self.latent_dim,
          mlp_output_dimension=generate_n_weights,
        )

        # Take the node attrs and obtain a query matrix
        self.edge_attr_to_query = ScalarMLPFunction(
            mlp_input_dimension=(
                # Node invariants for center and neighbor (chemistry)
                2 * parent.irreps_in[parent.node_invariant_field].num_irreps
                # Plus edge invariants for the edge (radius).
                + parent.irreps_in[parent.edge_invariant_field].num_irreps
            ),
            mlp_latent_dimensions = [],
            mlp_output_dimension=self.env_embed_multiplicity * self.head_dim,
            mlp_nonlinearity = None,
            use_layer_norm=False,
            zero_init_last_layer_weights= True,
        ) if self.use_attention else None

        # Take the node attrs and obtain a query matrix
        self.latent_to_key = ScalarMLPFunction(
            mlp_input_dimension = self.latent_dim,
            mlp_latent_dimensions = [],
            mlp_output_dimension = self.env_embed_multiplicity * self.head_dim,
            mlp_nonlinearity = None,
            use_layer_norm=False,
            zero_init_last_layer_weights = True,
        ) if self.use_attention else None

        self.rearrange_qk = Rearrange('e (m d) -> e m d', m=self.env_embed_multiplicity, d=self.head_dim) if self.use_attention else None
        self._env_weighter = env_weighter
        self.tp_n_scalar_out = parent._tp_n_scalar_outs[self.layer_index]

        if not self.use_attention:
            if avg_num_neighbors_is_learnable:
                self.env_sum_normalization = torch.nn.Parameter(torch.as_tensor([avg_num_neighbors]).rsqrt())
            else:
                self.register_buffer("env_sum_normalization", torch.as_tensor([avg_num_neighbors]).rsqrt())

        if self.learn_cutoff_bias:
            # self.rbf_embedder = EdgeRadialAttrsEmbedder(
            #     in_dim=parent.irreps_in[parent.edge_invariant_field].num_irreps,
            #     out_dim=self.latent_mlp.out_features
            # )
            self.rbf_embedder = FiLMFunction(
                mlp_input_dimension=parent.irreps_in[parent.edge_invariant_field].num_irreps,
                mlp_latent_dimensions=[2*parent.irreps_in[parent.edge_invariant_field].num_irreps],
                mlp_output_dimension=self.latent_mlp.out_features,
                mlp_nonlinearity='swiglu',
                zero_init_last_layer_weights=False,
                has_bias=False,
                final_non_lin='sigmoid'
            )


    def apply_attention(self, node_invariants, edge_invariants, edge_center, edge_neighbor, latents, emb_latent):
        edge_full_attr = torch.cat([
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ], dim=-1)

        # Asserts needed for JIT
        assert self.edge_attr_to_query is not None
        assert self.latent_to_key is not None
        assert self.rearrange_qk is not None

        Q = self.edge_attr_to_query(edge_full_attr)
        Q = self.rearrange_qk(Q)

        K = self.latent_to_key(latents)
        K = self.rearrange_qk(K)

        W = torch.einsum('emd,emd -> em', Q, K) * self.isqrtd
        # updated emb_latent
        return torch.einsum('emd,em->emd', emb_latent, scatter_softmax(W, edge_center, dim=0))


    def apply_mace(self, local_env_per_active_atom, node_invariants, active_node_centers):
        # Asserts needed for JIT
        assert self.product is not None
        assert self.reshape_in_module is not None
        expanded_features_per_active_atom: torch.Tensor = self.product(
            node_feats=local_env_per_active_atom,
            node_attrs=node_invariants[active_node_centers],
        )
        # updated local_env_per_active_atom
        return self.reshape_in_module(expanded_features_per_active_atom)


    def forward(
        self,
        data: AtomicDataDict.Type,
        active_edges: torch.Tensor,
        num_nodes: int,
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
        new_latents = self.latent_mlp(inv_latent_cat)
        new_latents = self.post_norm(new_latents)
        # Apply cutoff, which propagates through to everything else
        if self.learn_cutoff_bias:
            new_latents = self.rbf_embedder(new_latents, data[AtomicDataDict.EDGE_RADIAL_ATTRS_KEY])
        else:
            new_latents[:, :new_latents.size(1)//2] = cutoff_coeffs.unsqueeze(-1) * new_latents[:, :new_latents.size(1)//2]

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
            latents = torch.index_add(coefficient_old * latents, 0, active_edges, coefficient_new * new_latents)
        else:
            # Normal (non-residual) update
            # index_copy replaces, unlike index_add
            latents = torch.index_copy(latents, 0, active_edges, new_latents)

        # From the latents, compute the weights for active edges:
        weights = self.env_embed_mlp(latents)

        if self.film is not None: weights = self.film(weights, data[self.graph_conditioning_field], data[AtomicDataDict.BATCH_KEY][edge_center])

        w_index: int = 0
        if self.layer_index == 0:
            # embed initial edge
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel
            eq_features = self._env_weighter(eq_features, env_w) # eq_features is edge_attr
            eq_features = self.eq_features_irreps_norm(eq_features) # TODO it's better if this comes before

        # Extract weights for the edge attrs
        env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
        w_index += self._env_weighter.weight_numel
        emb_latent = self._env_weighter(edge_attr, env_w) # emb_latent is normalized below

        if self.use_attention: emb_latent = self.apply_attention(node_invariants, edge_invariants, edge_center, edge_neighbor, latents, emb_latent)

        # Pool over all attention-weighted edge features to build node local environment embedding
        local_env_per_node = scatter(emb_latent, edge_center, dim=0, dim_size=num_nodes)
        if not self.use_attention: local_env_per_node = local_env_per_node * self.env_sum_normalization

        active_node_centers = torch.unique(edge_center)
        local_env_per_node_active_node_centers = local_env_per_node[active_node_centers]

        local_env_per_node_active_node_centers = self.env_norm(local_env_per_node_active_node_centers)
        local_env_per_active_atom = self.env_linear(local_env_per_node_active_node_centers)

        if self.use_mace_product: local_env_per_active_atom = self.apply_mace(local_env_per_active_atom, node_invariants, active_node_centers)

        expanded_features_per_node = torch.zeros_like(local_env_per_node, dtype=local_env_per_active_atom.dtype)
        expanded_features_per_node[active_node_centers] = local_env_per_active_atom

        # Copy to get per-edge
        # Large allocation, but no better way to do this:
        local_env_per_active_edge = expanded_features_per_node[edge_center]

        # Now do the TP
        # recursively tp current features with the environment embeddings
        eq_features = self.tp(eq_features, local_env_per_active_edge)
        eq_features = self.tp_norm(eq_features)

        # Get invariants
        # features has shape [z][mul][k]
        # we know scalars are first
        scalars, equivariant = torch.split(eq_features, [self.tp_n_scalar_out, eq_features.size(-1) - self.tp_n_scalar_out], dim=-1)
        scalars = self.rearrange_scalars(scalars)

        inv_latent = torch.cat([latents, scalars],dim=-1) # scalars.shape (E, 2*sum(embedding_dimensionality in yaml))

        log_feature_on_wandb(self.debug, f"{self.parent_name}.{self.layer_index}.latents", latents)
        log_feature_on_wandb(self.debug, f"{self.parent_name}.{self.layer_index}.inv_latent", inv_latent)
        if self.linear is None: return latents, inv_latent, None

        # do the linear for eq. features
        eq_features = self.linear(equivariant if self.is_last_layer else eq_features)
        log_feature_on_wandb(self.debug, f"{self.parent_name}.{self.layer_index}.eq_features", eq_features)
        return latents, inv_latent, eq_features