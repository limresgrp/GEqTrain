import math
import functools
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

from torch.nn import LayerNorm


def exists(val):
    return val is not None


def pick_mpl_function(func):
    if isinstance(func, Callable):
        return func
    assert isinstance(func, str)
    if func.lower() == "ScalarMLPFunction".lower():
        return ScalarMLPFunction
    raise Exception(f"MLP Funciton {func} not implemented.")


@compile_mode("script")
class InteractionModule(GraphModuleMixin, torch.nn.Module):
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
        out_irreps: Optional[Union[o3.Irreps, str]] = None,
        output_hidden_irreps: bool = False,
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

        # MLP parameters:
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
        use_norms: bool = False,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")  # define for convinience
        self.DTYPE = torch.get_default_dtype()

        # save parameters
        assert (
            num_layers >= 1
        )  # zero layers is "two body", but we don't need to support that fallback case
        self.num_layers = num_layers

        self.node_invariant_field = node_invariant_field
        self.edge_invariant_field = edge_invariant_field
        self.edge_equivariant_field = edge_equivariant_field
        self.out_field = out_field
        self.env_embed_mul = env_embed_multiplicity
        self.head_dim = head_dim
        self.isqrtd = math.isqrt(head_dim)
        self.polynomial_cutoff_p = float(PolynomialCutoff_p)
        self.avg_num_neighbors = avg_num_neighbors

        env_embed = pick_mpl_function(env_embed)
        two_body_latent = pick_mpl_function(two_body_latent)
        latent = pick_mpl_function(latent)

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.node_invariant_field,
                self.edge_invariant_field,
                self.edge_equivariant_field,
            ],
        )

        # for normalization of features
        # one per layer
        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor([5.] * num_layers),
        )

        latent =          functools.partial(latent,    **latent_kwargs)
        two_body_latent = functools.partial(latent,    **two_body_latent_kwargs)
        env_embed =       functools.partial(env_embed, **env_embed_kwargs)

        self.latents = torch.nn.ModuleList([])
        self.env_embed_mlps = torch.nn.ModuleList([])
        self.node_attr_to_queries = torch.nn.ModuleList([])
        self.edge_feat_to_keys = torch.nn.ModuleList([])
        self.tps = torch.nn.ModuleList([])
        self.products = torch.nn.ModuleList([])
        self.reshape_in_modules = torch.nn.ModuleList([])
        self.linears = torch.nn.ModuleList([])
        self.env_linears = torch.nn.ModuleList([])

        # Embed to the spharm * it as mul
        input_edge_eq_irreps = self.irreps_in[self.edge_equivariant_field]
        assert all(mul == 1 for mul, _ in input_edge_eq_irreps)

        env_embed_irreps = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in input_edge_eq_irreps])
        assert (
            env_embed_irreps[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"
        self._input_pad = (
            int(math.ceil(env_embed_irreps.dim / pad_to_alignment)) * pad_to_alignment
        ) - env_embed_irreps.dim

        if out_irreps is None:
            out_irreps = env_embed_irreps
        else:
            out_irreps = out_irreps if isinstance(out_irreps, o3.Irreps) else o3.Irreps(out_irreps)
        if output_hidden_irreps:
            out_irreps = o3.Irreps(
                [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps if ir.l in [0] + out_irreps.ls]
            )

        # Initially, we have the B(r)Y(\vec{r})-projection of the edges
        # (possibly embedded)
        arg_irreps = env_embed_irreps

        # - begin irreps -
        # start to build up the irreps for the iterated TPs
        tps_irreps = [arg_irreps]

        for layer_idx in range(num_layers):
            ir_out = env_embed_irreps
            # Create higher order terms cause there are more TPs coming
            if layer_idx == self.num_layers - 1:
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
        self._env_weighter = MakeWeightedChannels(
            irreps_in=input_edge_eq_irreps,
            multiplicity_out=env_embed_multiplicity,
            pad_to_alignment=pad_to_alignment,
        )

        self._tp_n_scalar_outs: List[int] = []
        self._features_n_scalar_outs: List[int] = []

        # - Build Products and TPs -
        for layer_idx, (l_arg_irreps, l_out_irreps) in enumerate(
            zip(tps_irreps_in, tps_irreps_out)
        ):
            # Make the env embed linear
            self.env_linears.append(
                Linear(
                    env_embed_irreps,
                    env_embed_irreps,
                    shared_weights=True,
                    internal_weights=True,
                )
            )

            # Make product
            self.products.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=env_embed_irreps,
                    target_irreps=env_embed_irreps,
                    correlation=product_correlation,
                    num_elements=self.irreps_in[self.node_invariant_field].dim,
                )
            )

            # Reshape back product so that you can perform tp
            self.reshape_in_modules.append(reshape_irreps(env_embed_irreps))

            # Make TP
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
                            full_out_irreps.append((env_embed_multiplicity, ir_out))
                            tmp_i_out += 1
            self._tp_n_scalar_outs.append(tp_n_scalar_outs)
            full_out_irreps = o3.Irreps(full_out_irreps)
            assert all(ir == SCALAR for _, ir in full_out_irreps[:tp_n_scalar_outs])
            tp = Contracter(
                irreps_in1=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in l_arg_irreps]
                ),
                irreps_in2=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in env_embed_irreps]
                ),
                irreps_out=o3.Irreps(
                    [(env_embed_multiplicity, ir) for _, ir in full_out_irreps]
                ),
                instructions=instr,
                connection_mode=("uuu"),
                shared_weights=False,
                has_weight=False,
                pad_to_alignment=pad_to_alignment,
                sparse_mode=sparse_mode,
            )
            self.tps.append(tp)

            # Make env embed mlp
            generate_n_weights = (self._env_weighter.weight_numel)  # the weight for the edge embedding
            generate_n_weights += self.env_embed_mul                # + the weights for the edge attention
            if layer_idx == 0:
                # also need weights to embed the edge itself
                # this is because the 2 body latent is mixed in with the first layer
                # in terms of code
                generate_n_weights += self._env_weighter.weight_numel

            # the linear acts after the extractor
            linear_out_irreps = out_irreps if layer_idx == self.num_layers - 1 else env_embed_irreps

            _features_n_scalar_outs = linear_out_irreps.count(SCALAR) // linear_out_irreps[0].mul
            self._features_n_scalar_outs.append(_features_n_scalar_outs)

            self.linears.append(
                Linear(
                    full_out_irreps,
                    linear_out_irreps,
                    shared_weights=True,
                    internal_weights=True,
                    pad_to_alignment=pad_to_alignment,
                )
            )

            if layer_idx == 0:
                # at the first layer, we have no invariants from previous TPs
                self.latents.append(
                    two_body_latent(
                        mlp_input_dimension=(
                            (
                                # Node invariants for center and neighbor (chemistry)
                                2 * self.irreps_in[self.node_invariant_field].num_irreps
                                # Plus edge invariants for the edge (radius).
                                + self.irreps_in[self.edge_invariant_field].num_irreps
                            )
                        ),
                        mlp_output_dimension=None,
                        weight_norm=False,
                    )
                )
                self._latent_dim = self.latents[-1].out_features
            else:
                self.latents.append(
                    latent(
                        mlp_input_dimension=(
                            # the embedded latent invariants from the previous layer(s)
                            self.latents[-1].out_features
                            # and the invariants extracted from the last layer's TP:
                            + env_embed_multiplicity * self._tp_n_scalar_outs[layer_idx - 1]
                        ),
                        mlp_output_dimension=None,
                        weight_norm=False,
                    )
                )

            # the env embed MLP takes the last latent's output as input
            # and outputs enough weights for the env embedder
            self.env_embed_mlps.append(
                env_embed(
                    mlp_input_dimension=self.latents[-1].out_features,
                    mlp_output_dimension=generate_n_weights,
                    weight_norm=False,
                )
            )

            # Take the node attrs and obtain a query matrix
            self.node_attr_to_queries.append(
                env_embed(
                    mlp_input_dimension=irreps_in[AtomicDataDict.NODE_ATTRS_KEY].dim,
                    mlp_output_dimension=env_embed_multiplicity * head_dim,
                )
            )

            # Take the node attrs and obtain a query matrix
            self.edge_feat_to_keys.append(
                env_embed(
                    mlp_input_dimension=self.env_embed_mul,
                    mlp_output_dimension=env_embed_multiplicity * head_dim,
                )
            )

        self.out_multiplicity = out_irreps[0].mul
        self.final_latent = latent(
            mlp_input_dimension=(
                # the embedded latent invariants from the previous layer(s)
                self.latents[-1].out_features
                # and the invariants extracted from the last layer's TP:
                + env_embed_multiplicity * self._tp_n_scalar_outs[layer_idx]
            ),
            mlp_output_dimension=self.out_multiplicity * self._features_n_scalar_outs[layer_idx],
            weight_norm=False,
        )

        self.reshape_back_features = inverse_reshape_irreps(out_irreps)

        # - end build modules -
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

        self.final_norm_layer = LayerNorm(self.latents[-1].out_features + env_embed_multiplicity * self._tp_n_scalar_outs[layer_idx]) if use_norms else None

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

        # pre-declare variables as Tensors for TorchScript
        scalars = self._zero
        coefficient_old = scalars
        coefficient_new = scalars

        # Initialize state
        out_features = torch.zeros(
            (num_edges, self.out_multiplicity, self.out_feat_elems),
            dtype=self.DTYPE,
            device=edge_attr.device
        )
        latents = torch.zeros(
            (num_edges, self._latent_dim),
            dtype=edge_attr.dtype,
            device=edge_attr.device,
        )
        active_edges = torch.arange(
            num_edges,
            device=edge_attr.device,
        )

        # For the first layer, we use the input invariants:
        # The center and neighbor invariants and edge invariants
        latent_inputs_to_cat = [
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ]
        # The nonscalar features. Initially, the edge data.
        features = edge_attr

        layer_index: int = 0
        # compute the sigmoids vectorized instead of each loop
        layer_update_coefficients = self._latent_resnet_update_params.sigmoid()

        # Vectorized precompute per layer cutoffs
        cutoff_coeffs_all = polynomial_cutoff(
            edge_length, self.per_layer_cutoffs, p=self.polynomial_cutoff_p
        )

        # This goes through layer0, layer1, ..., layer_max-1
        for (latent,
            env_embed_mlp,
            node_attr_to_query,
            edge_feat_to_key,
            env_linear,
            linear,
            prod,
            reshape_in,
            tp
        ) in zip(
            self.latents,
            self.env_embed_mlps,
            self.node_attr_to_queries,
            self.edge_feat_to_keys,
            self.env_linears,
            self.linears,
            self.products,
            self.reshape_in_modules,
            self.tps
        ):
            # Determine which edges are still in play
            cutoff_coeffs = cutoff_coeffs_all[layer_index]
            prev_mask = cutoff_coeffs[active_edges] > 0
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

            # Compute latents
            new_latents = latent(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])
            # Apply cutoff, which propagates through to everything else
            norm_const = self.env_sum_normalizations[layer_index]
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents * norm_const

            if layer_index > 0:
                this_layer_update_coeff = layer_update_coefficients[layer_index - 1]
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
            weights = env_embed_mlp(latents[active_edges])
            w_index: int = 0

            if layer_index == 0:
                # embed initial edge
                env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
                w_index += self._env_weighter.weight_numel
                features = self._env_weighter(
                    features[prev_mask], env_w
                )  # features is edge_attr
            else:
                # just take the previous features that we still need
                features = features[prev_mask]

            # Extract weights for the environment builder
            env_w = weights.narrow(-1, w_index, self._env_weighter.weight_numel)
            w_index += self._env_weighter.weight_numel
            emb_latent = self._env_weighter(edge_attr[active_edges], env_w)

            # Apply attention on emb_latent
            Q = node_attr_to_query(data[AtomicDataDict.NODE_ATTRS_KEY])
            Q = Q.reshape(-1, self.env_embed_mul, self.head_dim)[edge_center[active_edges]]

            key_w = weights.narrow(-1, w_index, self.env_embed_mul)
            w_index += self.env_embed_mul
            K = edge_feat_to_key(key_w)
            K = K.reshape(-1, self.env_embed_mul, self.head_dim)

            A = torch.einsum('ijk,ijk -> ij', Q, K) * self.isqrtd
            emb_latent = torch.einsum('emd,em->emd', emb_latent, scatter_softmax(A, edge_center[active_edges], dim=0))

            # Pool over all attention-weighted edge features to build node local environment embedding
            local_env_per_node = scatter(
                emb_latent,
                edge_center[active_edges],
                dim=0,
                dim_size=num_nodes,
            )

            active_node_centers = torch.unique(edge_center[active_edges])
            local_env_per_active_atom = env_linear(local_env_per_node[active_node_centers])

            expanded_features_per_active_atom: torch.Tensor = prod(
                node_feats=local_env_per_active_atom,
                node_attrs=node_invariants[active_node_centers],
            )
            expanded_features_per_active_atom = reshape_in(expanded_features_per_active_atom)

            expanded_features_per_node = torch.zeros_like(local_env_per_node)
            expanded_features_per_node[active_node_centers] = expanded_features_per_active_atom

            # Copy to get per-edge
            # Large allocation, but no better way to do this:
            local_env_per_active_edge = expanded_features_per_node[edge_center[active_edges]]

            # Now do the TP
            # recursively tp current features with the environment embeddings
            features = tp(features, local_env_per_active_edge)

            # Get invariants
            # features has shape [z][mul][k]
            # we know scalars are first
            scalars = features[:, :, :self._tp_n_scalar_outs[layer_index]].reshape(
                features.shape[0], -1
            )

            # do the linear
            features = linear(features)

            # For layer2+, use the previous latents and scalars
            # This makes it deep
            latent_inputs_to_cat = [
                latents[active_edges],
                scalars,
            ]

            # increment counter
            layer_index += 1

        # - final layer -
        features_n_scalars = self._features_n_scalar_outs[layer_index- 1]

        # - output non-scalar values
        out_features[active_edges, :, features_n_scalars:] = features[..., features_n_scalars:]
        out_features = self.reshape_back_features(out_features)

        # - output scalar values
        cutoff_coeffs = cutoff_coeffs_all[layer_index]
        prev_mask = cutoff_coeffs[active_edges] > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

        # norm

        if exists(self.final_norm_layer):
            scalars = self.final_norm_layer(torch.cat(latent_inputs_to_cat, dim=-1)[prev_mask])

        # final MLP

        scalars = self.final_latent(scalars)

        out_features[active_edges, :self.out_multiplicity * features_n_scalars] = scalars

        data[self.out_field] = out_features
        return data

    def normalize_weights(self) -> None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                normalized_param = F.normalize(param, p=2, dim=0)
                # Assign normalized parameter back to the model
                param.data.copy_(normalized_param)
