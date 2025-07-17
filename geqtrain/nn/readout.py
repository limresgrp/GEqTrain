from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
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

from geqtrain.nn._heads import L0IndexedAttention

from geqtrain.utils.tp_utils import PSEUDO_SCALAR, SCALAR


@compile_mode("script")
class ReadoutModule(GraphModuleMixin, nn.Module):
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
        irreps_in=None, # if output is only scalar, this is required
        ignore_amp: bool = False, # wheter to adopt amp or not
        scalar_attnt: bool = True,
        num_heads: int = 32,
        dataset_mode: str = 'single', # single|ensemble
    ):
        super().__init__()
        
        self.field = field
        self.ignore_amp = ignore_amp
        self.out_field = out_field or self.field
        irreps_out = {self.out_field: out_irreps}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out=irreps_out,
        )
        in_irreps:  o3.Irreps = self.irreps_in[field]
        out_irreps: o3.Irreps = self.irreps_out[self.out_field]
        
        # --- start definition of input/output irreps --- #
        
        self.resnet = resnet
        self.eq_has_internal_weights = eq_has_internal_weights
        self.has_invariant_output   = False
        self.has_equivariant_output = False
        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0
        self.inv_readout = None
        self.n_scalars_out = out_irreps.ls.count(0)

        # check irreps
        if self.resnet:
            assert in_irreps == out_irreps        

        # --- end definition of input/output irreps --- #

        # --- start layer construction --- #
        # scalar
        if self.n_scalars_out > 0:
            self.has_invariant_output = True
            self.inv_readout = readout_latent( # mlp on scalars ONLY
                mlp_input_dimension=self.n_scalars_in,
                mlp_output_dimension=self.n_scalars_out,
                **readout_latent_kwargs,
            )

        # l>0
        self.reshape_in: Optional[reshape_irreps] = None
        self.eq_readout = None
        self.weights_emb = None
        if out_irreps.dim > self.n_scalars_out:
            self.has_equivariant_output = True
            eq_linear_input_irreps  = o3.Irreps([(mul, ir) for mul, ir in in_irreps  if ir.l>0])
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
            else:
                self.eq_readout = Linear( # equivariant MLP acting on l>0 ONLY
                        eq_linear_input_irreps,
                        eq_linear_output_irreps,
                        shared_weights=self.eq_has_internal_weights,
                        internal_weights=self.eq_has_internal_weights,
                        pad_to_alignment=1,
                )
                self.weights_emb = readout_latent( # mlp on scalars, used to compute the weights of the self.eq_readout
                    mlp_input_dimension=self.n_scalars_in,
                    mlp_output_dimension=self.eq_readout.weight_numel,
                    **readout_latent_kwargs,
                )
            self.reshape_back_features = inverse_reshape_irreps(eq_linear_output_irreps)
        elif strict_irreps:
            assert in_irreps.dim == self.n_scalars_in, (
                    f"Module input contains features with irreps that are not scalars ({in_irreps}). " +
                    f"However, the irreps of the output is composed of scalars only ({out_irreps}). "   +
                    "Please remove non-scalar features from the input, which otherwise would remain unused." +
                    f"If features come from InteractionModule, you can add the parameter 'output_ls: [0]' in the yaml config file." +
                    "If you want to allow this behavior, set 'strict_irreps=False'."
                )

        # --- end layer construction --- #
        self._resnet_update_coeff: Optional[nn.Parameter] = None # init to None for jit
        if self.resnet:
            assert irreps_in[self.out_field] == out_irreps
            self._resnet_update_coeff = nn.Parameter(torch.tensor([0.0]))
        self.out_irreps_dim = out_irreps.dim
        self.act_on_nodes = self.field in _NODE_FIELDS

        if dampen:
            add_tags_to_module(self, 'dampen')

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
        irps = irreps_in[self.field]
        self.split_index = sum([mul for mul, ir in irps if ir in [SCALAR, PSEUDO_SCALAR]])
        self.ensemble_attnt1 = None
        self.ensemble_attnt2 = None
        if self.scalar_attnt:
            assert self.n_scalars_in > 0, 'No scalars recieved for readout but scalar_attnt = True'
            self.ensemble_attnt1 = L0IndexedAttention(irreps_in=irreps_in, field=field, out_field=field, num_heads=num_heads, idx_key=idx_key, update_mlp=True)
            self.ensemble_attnt2 = L0IndexedAttention(irreps_in=irreps_in, field=field, out_field=field, num_heads=num_heads, idx_key=idx_key)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        #! ----------- COMMENT TO JIT COMPILE --------------- #
        # if self.ignore_amp:
        #     with torch.amp.autocast('cuda', enabled=False):
        #         return self._forward_impl(data)
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

    def _apply_scalar_attnt(self, features, data: AtomicDataDict.Type) -> torch.Tensor:
        assert self.ensemble_attnt1 is not None
        assert self.ensemble_attnt2 is not None
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
        irreps_in=None, # if output is only scalar, this is required
        ignore_amp: bool = False, # wheter to adopt amp or not
        scalar_attnt: bool = True,
        num_heads: int = 32,
        dataset_mode: str = 'single', # single|ensemble
    ):
        super().__init__(
            field=field,
            out_field=out_field,
            out_irreps=out_irreps,
            strict_irreps=strict_irreps,
            readout_latent=readout_latent,
            readout_latent_kwargs=readout_latent_kwargs,
            eq_has_internal_weights=eq_has_internal_weights,
            resnet=resnet,
            dampen=dampen,
            irreps_in=irreps_in,
            ignore_amp=ignore_amp,
            scalar_attnt=scalar_attnt,
            num_heads=num_heads,
            dataset_mode=dataset_mode,
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
        features = data[self.field]
        out_features = torch.zeros(
            (features.shape[0], self.out_irreps_dim),
            dtype=torch.float32,
            device=features.device
        )
        conditioning = data[self.conditioning_field]
        scalars, equiv = torch.split(features, [self.split_index, features.shape[-1] - self.split_index], dim=-1)
        scalars = self.film1(scalars, conditioning)
        scalars = self.fc1(scalars)
        scalars = self.film2(scalars, conditioning)
        return torch.cat((scalars, equiv), dim=-1), out_features

    def _handle_invariant_output(self, features, data: AtomicDataDict.Type, active_nodes, out_features) -> torch.Tensor:
        assert self.inv_readout is not None
        out_features[active_nodes, :self.n_scalars_out] += self.inv_readout(features[active_nodes, :self.n_scalars_in])
        out_features[active_nodes, :self.n_scalars_out] = self.film_scalar(
            out_features[active_nodes, :self.n_scalars_out],
            data[self.conditioning_field],
        )
        return out_features

    def _handle_vectorial_output(self, features, data: AtomicDataDict.Type, active_nodes, out_features) -> torch.Tensor:
        eq_features = self.reshape_in(features[active_nodes, self.n_scalars_in:])
        if self.eq_has_internal_weights: # eq linear layer with its own inner weights
            assert self.eq_readout_iw is not None
            eq_features = self.eq_readout_iw(eq_features)
        else:
            # else the weights are computed via mlp using scalars
            weights = self.weights_emb(features[active_nodes, :self.n_scalars_in])
            weights = self.film_vectorial(weights, data[self.conditioning_field])
            assert self.eq_readout is not None
            eq_features = self.eq_readout(eq_features, weights)
        out_features[active_nodes, self.n_scalars_out:] += self.reshape_back_features(eq_features) # set features for l>=1
        return out_features


@compile_mode("script")
class ReadoutModuleWithSimilarity(GraphModuleMixin, nn.Module):
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
        num_heads: int = 32,
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field or self.field
        in_irreps: o3.Irreps = irreps_in[field]

        irreps_out = {self.out_field: out_irreps}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out=irreps_out,
        )
        out_irreps = self.irreps_out[self.out_field]

        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0
        self.n_scalars_out = out_irreps.ls.count(0)
        assert self.n_scalars_out > 0

        self.register_parameter("linear_proj", nn.Parameter(torch.empty(self.n_scalars_in, self.n_scalars_in, dtype=torch.get_default_dtype())))
        nn.init.xavier_uniform_(self.linear_proj)

        self.register_parameter("similarity", nn.Parameter(torch.empty(num_heads, self.n_scalars_out, self.n_scalars_in, dtype=torch.get_default_dtype())))
        nn.init.xavier_uniform_(self.similarity)

        self.register_buffer("tau", torch.tensor(2.7172))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[self.field][:, :self.n_scalars_in]
        h = torch.einsum('ni,io -> no', x, self.linear_proj).unsqueeze(-2).unsqueeze(-2)

        # Compute similarity between h and shape_matcher (cosine similarity)
        similarity = nn.functional.cosine_similarity(h, self.similarity, dim=-1)
        w = nn.functional.softmax(similarity * self.tau, dim=-2)

        data[self.out_field] = (w * similarity).sum(dim=-2) * 2 * self.tau # output logits
        return data


@compile_mode("script")
class ReadoutModuleWithVQ(GraphModuleMixin, nn.Module):
    '''
    A readout module that uses a VQ-VAE-like mechanism.

    For each input feature, it finds the `num_selected_codes` (N) closest vectors
    from a learnable `codebook` of size `num_codes` (BigN). The average of these
    selected vectors is then projected to produce the final output logits.
    '''
    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        out_irreps: Union[o3.Irreps, str] = None,
        num_codes: int = 512,           # BigN: Total number of vectors in the codebook
        num_selected_codes: int = 4,    # N: Number of closest vectors to select
        irreps_in=None,
    ):
        super().__init__()
        self.field = field
        self.out_field = out_field or self.field
        self.num_codes = num_codes
        self.num_selected_codes = num_selected_codes

        # --- Irreps Initialization ---
        in_irreps: o3.Irreps = irreps_in[field]
        irreps_out = {self.out_field: out_irreps}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[field],
            irreps_out=irreps_out,
        )
        out_irreps = self.irreps_out[self.out_field]

        self.n_scalars_in = in_irreps.ls.count(0)
        assert self.n_scalars_in > 0
        self.n_scalars_out = out_irreps.ls.count(0)
        assert self.n_scalars_out > 0

        # --- Learnable Parameters ---

        # 1. Optional initial projection of input features
        self.linear_proj = nn.Parameter(torch.empty(self.n_scalars_in, self.n_scalars_in))
        nn.init.xavier_uniform_(self.linear_proj)

        # 2. The codebook (replaces 'similarity' parameter)
        self.codebook = nn.Parameter(torch.empty(self.num_codes, self.n_scalars_in))
        nn.init.xavier_uniform_(self.codebook)

        # 3. Final projection head to produce logits
        self.logit_head = nn.Linear(self.n_scalars_in, self.n_scalars_out)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Select scalar features from the input field
        x = data[self.field][:, :self.n_scalars_in]

        # Project input features
        h = torch.einsum('ni,io -> no', x, self.linear_proj)

        # --- VQ-style selection ---

        # 1. Compute pairwise squared Euclidean distances between inputs and codebook vectors
        # dist^2 = ||h - c||^2 = ||h||^2 - 2*h*c^T + ||c||^2
        h_squared = torch.sum(h**2, dim=1, keepdim=True)
        codebook_squared = torch.sum(self.codebook**2, dim=1)
        h_dot_codebook = torch.matmul(h, self.codebook.t()) # (batch, num_codes)

        distances_sq = h_squared - 2 * h_dot_codebook + codebook_squared

        # 2. Find the indices of the N closest codebook vectors for each input
        # We use `topk` with `largest=False` to find the smallest distances
        _, top_k_indices = torch.topk(
            distances_sq,
            k=self.num_selected_codes,
            dim=-1,
            largest=False
        ) # Shape: (batch, num_selected_codes)

        # 3. Gather the N closest codebook vectors
        selected_codes = self.codebook[top_k_indices] # Shape: (batch, num_selected_codes, n_scalars_in)

        # 4. Aggregate the selected vectors (e.g., by averaging)
        quantized_h = torch.mean(selected_codes, dim=1) # Shape: (batch, n_scalars_in)

        # 5. Project to the final logit dimension
        logits = self.logit_head(quantized_h) # Shape: (batch, n_scalars_out)

        data[self.out_field] = logits
        return data