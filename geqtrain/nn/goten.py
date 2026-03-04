import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqtrain.data import AtomicDataDict
from geqtrain.nn._equivariant_scalar_mlp import EquivariantScalarMLP
from geqtrain.nn._fc import ScalarMLPFunction
from geqtrain.nn._graph_mixin import GraphModuleMixin
from geqtrain.nn._embedding import BaseEmbedding
from geqtrain.nn.so3 import SO3_Linear
from geqtrain.utils._model_utils import process_out_irreps
from geqtrain.utils.pytorch_scatter import scatter_sum, scatter_softmax


# ---- helpers (same logic as official) ----
def _get_split_sizes_from_lmax(lmax: int, start: int = 1) -> List[int]:
    return [2 * l + 1 for l in range(start, lmax + 1)]


def _split_to_components(tensor: torch.Tensor, lmax: int, start: int = 1, dim: int = -1) -> List[torch.Tensor]:
    return list(torch.split(tensor, _get_split_sizes_from_lmax(lmax, start=start), dim=dim))


class _CosineCutoff(torch.nn.Module):
    """Matches official CosineCutoff behavior."""
    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # distances: [..., 1] or [...]
        d = distances
        cut = 0.5 * (torch.cos(d * math.pi / self.cutoff) + 1.0)
        cut = cut * (d < self.cutoff).to(d.dtype)
        return cut


# ==============================================================================
# 1. NODE EMBEDDING
# ==============================================================================
@compile_mode("script")
class GotenNodeEmbedding(BaseEmbedding):
    """
    Initializes scalar node features based on atom types and their local environment.
    This version removes the final LayerNorm, deferring normalization to subsequent layers.
    """
    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        node_irreps = self.irreps_in[self.node_field]
        node_dim = node_irreps.dim
        edge_attr_dim = self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY].dim
        
        self.W_neighbor_embed = torch.nn.Parameter(torch.randn(node_dim, latent_dim))
        self.W_center_embed = torch.nn.Parameter(torch.randn(node_dim, latent_dim))
        self.W_radial_embed = torch.nn.Parameter(torch.randn(edge_attr_dim, latent_dim))
        self.W_update = torch.nn.Parameter(torch.randn(2 * latent_dim, latent_dim))
        self.norm = torch.nn.LayerNorm(latent_dim)

        self.out_irreps = o3.Irreps(f'{latent_dim}x0e')

    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_attr = data[AtomicDataDict.EDGE_RADIAL_EMB_KEY]
        
        node_field = self.node_field
        assert isinstance(node_field, str)
        node_attr = data[node_field]
        num_nodes = len(node_attr)

        neighbor_embeddings = torch.einsum('ni,id -> nd', node_attr, self.W_neighbor_embed)
        radial_features = torch.einsum('ej,jd -> ed', edge_attr, self.W_radial_embed)
        messages_source_features = neighbor_embeddings[edge_neighbor]
        messages = messages_source_features * radial_features
        aggregated_messages = scatter_sum(messages, edge_center, dim=0, dim_size=num_nodes)
        center_node_features = torch.einsum('ni,id -> nd', node_attr, self.W_center_embed)
        concatenated_features = torch.cat([center_node_features, aggregated_messages], dim=-1)
    
        # Apply final update and layer normalization
        updated_features = torch.einsum('nk,kd -> nd', concatenated_features, self.W_update)
        return self.norm(updated_features)

# ==============================================================================
# 2. EDGE EMBEDDING
# ==============================================================================
@compile_mode("script")
class GotenEdgeEmbedding(BaseEmbedding):
    """
    Initializes scalar edge features (`t_ij`) based on node features and radial embeddings.
    This version implements the original GotenNet EdgeInit logic: (h_i + h_j) * W_erp(phi_ij).
    Optionally, scalar edge attributes can be concatenated with radial embeddings
    before the projection.
    Inherits from BaseEmbedding to be compatible with the EmbeddingAttrs class.
    """
    def __init__(
        self,
        include_edge_field: bool = True,
        include_edge_radial: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_edge_field = bool(include_edge_field)
        self.include_edge_radial = bool(include_edge_radial)
        self._use_edge_field = False
        self._use_edge_radial = False

        node_field = torch.jit._unwrap_optional(self.node_field)
        node_irreps = self.irreps_in[node_field]

        edge_in_dim = 0
        if self.include_edge_radial and AtomicDataDict.EDGE_RADIAL_EMB_KEY in self.irreps_in:
            self._use_edge_radial = True
            edge_in_dim += self.irreps_in[AtomicDataDict.EDGE_RADIAL_EMB_KEY].dim

        if (
            self.include_edge_field
            and self.edge_field is not None
            and self.edge_field in self.irreps_in
            and self.irreps_in[self.edge_field] is not None
        ):
            self._use_edge_field = True
            edge_in_dim += self.irreps_in[self.edge_field].dim

        if edge_in_dim <= 0:
            raise ValueError(
                "GotenEdgeEmbedding needs at least one scalar edge input. "
                "Enable `include_edge_radial` and/or provide `edge_field` with `include_edge_field=True`."
            )

        # Linear layer to project edge scalar inputs, similar to W_erp in the paper.
        self.W_erp = torch.nn.Linear(edge_in_dim, node_irreps.dim)
        
        self.out_irreps = node_irreps
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_erp.weight)

    def forward(self, data: AtomicDataDict.Type) -> torch.Tensor:
        edge_center   = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        node_field = torch.jit._unwrap_optional(self.node_field)
        h = data[node_field]

        # Gather features from center and neighbor nodes
        h_i = h[edge_center]
        h_j = h[edge_neighbor]

        edge_scalar_inputs = []
        if self._use_edge_radial:
            edge_scalar_inputs.append(data[AtomicDataDict.EDGE_RADIAL_EMB_KEY])
        if self._use_edge_field:
            edge_field = torch.jit._unwrap_optional(self.edge_field)
            edge_scalar_inputs.append(data[edge_field])

        if len(edge_scalar_inputs) == 1:
            edge_input = edge_scalar_inputs[0]
        else:
            edge_input = torch.cat(edge_scalar_inputs, dim=-1)

        # Project edge scalar inputs.
        radial_proj = self.W_erp(edge_input)
        
        # Compute t_ij by combining node and edge features
        t_ij = (h_i + h_j) * radial_proj
        
        # Return the tensor directly, as per BaseEmbedding contract
        return t_ij

class _TensorLayerNorm(torch.nn.Module):
    """
    Matches official TensorLayerNorm (VisNet-style max-min norm per degree).
    See official implementation. :contentReference[oaicite:6]{index=6}
    """
    def __init__(self, hidden_channels: int, lmax: int, eps: float = 1e-12):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.eps = float(eps)
        self.register_buffer("weight", torch.ones(self.hidden_channels))

    def _max_min_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: [N, (2l+1), H]
        dist = torch.norm(tensor, dim=1, keepdim=True)  # [N, 1, H]
        if (dist == 0).all():
            return torch.zeros_like(tensor)
        dist = dist.clamp(min=self.eps)
        direct = tensor / dist
        max_val, _ = torch.max(dist, dim=-1)  # [N, 1]
        min_val, _ = torch.min(dist, dim=-1)  # [N, 1]
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        return F.relu(dist) * direct

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: [N, equi_dim, H], equi_dim = sum_{l=1..lmax} (2l+1)
        parts = torch.split(X, _get_split_sizes_from_lmax(self.lmax, start=1), dim=1)
        parts = [self._max_min_norm(p) for p in parts]
        out = torch.cat(parts, dim=1)
        return out * self.weight.view(1, 1, -1)


class _MLP2(torch.nn.Module):
    """A simple 2-layer MLP with activation on the hidden layer."""
    def __init__(self, in_dim: int, out_dim: int, activation=F.silu, bias: bool = True):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim, in_dim, bias=bias)
        self.lin2 = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.activation(self.lin1(x)))


@compile_mode("script")
class _EQFF_Official(torch.nn.Module):
    """Matches official EQFF math and shapes. :contentReference[oaicite:7]{index=7}"""
    def __init__(self, n_atom_basis: int, epsilon: float = 1e-8):
        super().__init__()
        self.n_atom_basis = int(n_atom_basis)
        self.epsilon = float(epsilon)
        # W_vu: linear map on channel dim (last dim)
        self.W_vu = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis, bias=False)
        torch.nn.init.xavier_uniform_(self.W_vu.weight)

        # gamma_m: ctx_dim=2H -> H -> 2H
        self.gamma_m_1 = torch.nn.Linear(2 * self.n_atom_basis, self.n_atom_basis)
        self.gamma_m_2 = torch.nn.Linear(self.n_atom_basis, 2 * self.n_atom_basis)
        torch.nn.init.xavier_uniform_(self.gamma_m_1.weight)
        torch.nn.init.zeros_(self.gamma_m_1.bias)
        torch.nn.init.xavier_uniform_(self.gamma_m_2.weight)
        torch.nn.init.zeros_(self.gamma_m_2.bias)

    def forward(self, h: torch.Tensor, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [N, 1, H], X: [N, equi_dim, H]
        X_p = self.W_vu(X)
        X_pn = torch.sqrt(torch.sum(X_p ** 2, dim=-2, keepdim=True) + self.epsilon)  # [N, 1, H]
        ctx = torch.cat([h, X_pn], dim=-1)  # [N, 1, 2H]
        x = self.gamma_m_2(F.silu(self.gamma_m_1(ctx)))  # [N, 1, 2H]
        m1, m2 = torch.split(x, self.n_atom_basis, dim=-1)
        h = h + m1
        X = X + m2 * X_p
        return h, X


@compile_mode("script")
class _GATA_Official(torch.nn.Module):
    """
    Implements the official GATA forward/message/edge_update logic, but in plain torch + scatter.
    See official message() and edge_update() blocks. 
    """
    def __init__(
        self,
        n_atom_basis: int,
        lmax: int,
        cutoff: float,
        num_heads: int = 8,
        dropout: float = 0.0,
        layer_norm: bool = False,
        steerable_norm: bool = False,
        edge_updates: Union[bool, str] = True,
        last_layer: bool = False,
        scale_edge: bool = True,
        sep_htr: bool = True,
        sep_dir: bool = False,
        sep_tensor: bool = False,
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        edge_ln: str = "",
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.n_atom_basis = int(n_atom_basis)
        self.lmax = int(lmax)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.last_layer = bool(last_layer)
        self.scale_edge = bool(scale_edge)
        self.sep_htr = bool(sep_htr)
        self.sep_dir = bool(sep_dir)
        self.sep_tensor = bool(sep_tensor)
        self.epsilon = float(epsilon)

        # multiplier logic (matches official)
        multiplier = 3
        if self.sep_dir:
            multiplier += self.lmax - 1
        if self.sep_tensor:
            multiplier += self.lmax - 1
        self.multiplier = int(multiplier)

        # norms
        self.layernorm = torch.nn.LayerNorm(self.n_atom_basis) if layer_norm else torch.nn.Identity()
        self.tensor_layernorm = _TensorLayerNorm(self.n_atom_basis, self.lmax) if steerable_norm else torch.nn.Identity()

        # attention projections
        self.W_q = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis)
        self.W_k = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis)
        torch.nn.init.xavier_uniform_(self.W_q.weight); torch.nn.init.zeros_(self.W_q.bias)
        torch.nn.init.xavier_uniform_(self.W_k.weight); torch.nn.init.zeros_(self.W_k.bias)

        # gamma_s, gamma_v (2-layer)
        self.gamma_s_1 = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis)
        self.gamma_s_2 = torch.nn.Linear(self.n_atom_basis, self.multiplier * self.n_atom_basis)
        self.gamma_v_1 = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis)
        self.gamma_v_2 = torch.nn.Linear(self.n_atom_basis, self.multiplier * self.n_atom_basis)
        for lin in [self.gamma_s_1, self.gamma_s_2, self.gamma_v_1, self.gamma_v_2]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

        # edge transforms
        self.W_re = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis)
        self.W_rs = torch.nn.Linear(self.n_atom_basis, self.multiplier * self.n_atom_basis)
        torch.nn.init.xavier_uniform_(self.W_re.weight); torch.nn.init.zeros_(self.W_re.bias)
        torch.nn.init.xavier_uniform_(self.W_rs.weight); torch.nn.init.zeros_(self.W_rs.bias)

        self.cutoff_fn = _CosineCutoff(cutoff)

        # ---- edge update config parsing (matches official behavior) ----
        update_info = {"gated": False, "rej": True, "mlp": False, "mlpa": False, "lin_w": 0, "lin_ln": 0}
        if isinstance(edge_updates, str):
            parts = edge_updates.split("_")
            allowed = ["gated","gatedt","norej","norm","mlp","mlpa","act","linw","linwa","ln","postln"]
            if not all([p in allowed for p in parts]):
                raise ValueError(f"Invalid edge_updates='{edge_updates}'. Allowed parts: {allowed}")
            if "gated" in parts:  update_info["gated"] = "gated"
            if "gatedt" in parts: update_info["gated"] = "gatedt"
            if "act" in parts:    update_info["gated"] = "act"
            if "norej" in parts:  update_info["rej"] = False
            if "mlp" in parts:    update_info["mlp"] = True
            if "mlpa" in parts:   update_info["mlpa"] = True
            if "linw" in parts:   update_info["lin_w"] = 1
            if "linwa" in parts:  update_info["lin_w"] = 2
            if "ln" in parts:     update_info["lin_ln"] = 1
            if "postln" in parts: update_info["lin_ln"] = 2
            self.edge_updates = True
        else:
            self.edge_updates = bool(edge_updates)
        self.update_info = update_info

        # edge update modules
        self.edge_vec_dim = self.n_atom_basis if evec_dim is None else int(evec_dim)
        self.edge_mlp_dim = self.n_atom_basis if emlp_dim is None else int(emlp_dim)

        if (not self.last_layer) and self.edge_updates:
            # gamma_t: apply on t_ij
            # official uses an MLP; here: 2-layer, with optional hidden emlp_dim if mlp/mlpa
            if self.update_info["mlp"] or self.update_info["mlpa"]:
                self.gamma_t_1 = torch.nn.Linear(self.n_atom_basis, self.edge_mlp_dim)
                self.gamma_t_2 = torch.nn.Linear(self.edge_mlp_dim, self.n_atom_basis)
                torch.nn.init.xavier_uniform_(self.gamma_t_1.weight); torch.nn.init.zeros_(self.gamma_t_1.bias)
                torch.nn.init.xavier_uniform_(self.gamma_t_2.weight); torch.nn.init.zeros_(self.gamma_t_2.bias)
            else:
                self.gamma_t_1 = torch.nn.Linear(self.n_atom_basis, self.n_atom_basis)
                self.gamma_t_2 = None
                torch.nn.init.xavier_uniform_(self.gamma_t_1.weight); torch.nn.init.zeros_(self.gamma_t_1.bias)

            # W_vq / W_vk: apply on X (last dim)
            self.W_vq = torch.nn.Linear(self.n_atom_basis, self.edge_vec_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.W_vq.weight)
            if self.sep_htr:
                self.W_vk = torch.nn.ModuleList([torch.nn.Linear(self.n_atom_basis, self.edge_vec_dim, bias=False) for _ in range(self.lmax)])
                for w in self.W_vk:
                    torch.nn.init.xavier_uniform_(w.weight)
            else:
                self.W_vk = torch.nn.Linear(self.n_atom_basis, self.edge_vec_dim, bias=False)
                torch.nn.init.xavier_uniform_(self.W_vk.weight)

            # gamma_w: optional linear map to H + gating
            self._gamma_w_ln = torch.nn.LayerNorm(self.edge_vec_dim) if self.update_info["lin_ln"] == 1 else None
            self.W_edp = None
            if self.update_info["lin_w"] > 0:
                self.W_edp = torch.nn.Linear(self.edge_vec_dim, self.n_atom_basis, bias=True)
                torch.nn.init.xavier_uniform_(self.W_edp.weight); torch.nn.init.zeros_(self.W_edp.bias)

    @staticmethod
    def _vector_rejection(rep: torch.Tensor, rl_ij: torch.Tensor) -> torch.Tensor:
        # official: vec_proj = (rep * rl_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        # here rl_ij: [E, d, 1], rep: [E, d, C]
        vec_proj = (rep * rl_ij).sum(dim=1, keepdim=True)  # [E, 1, C]
        return rep - vec_proj * rl_ij  # broadcast to [E, d, C]

    def _gamma_s(self, h: torch.Tensor) -> torch.Tensor:
        return self.gamma_s_2(F.silu(self.gamma_s_1(h)))

    def _gamma_v(self, h: torch.Tensor) -> torch.Tensor:
        return self.gamma_v_2(F.silu(self.gamma_v_1(h)))

    def _gamma_t(self, t_ij: torch.Tensor) -> torch.Tensor:
        # t_ij: [E, 1, H]
        if self.gamma_t_2 is None:
            return self.gamma_t_1(t_ij)
        return self.gamma_t_2(F.silu(self.gamma_t_1(t_ij)))

    def _gamma_w(self, w_ij: torch.Tensor) -> torch.Tensor:
        # w_ij: [E, edge_vec_dim]
        x = w_ij
        if self._gamma_w_ln is not None:
            x = self._gamma_w_ln(x)
        if self.update_info["lin_w"] % 10 == 2:
            x = F.silu(x)
        if self.W_edp is not None:
            x = self.W_edp(x)  # -> [E, H]

        # gated variants
        if self.update_info["gated"] == "gatedt":
            x = torch.tanh(x)
        elif self.update_info["gated"] == "gated":
            x = torch.sigmoid(x)
        elif self.update_info["gated"] == "act":
            x = F.silu(x)
        return x  # [E, H] (or [E, edge_vec_dim] if lin_w==0)

    def _message(self, edge_center, edge_neighbor, x, q, k, v, X, t_ij_filter, t_ij_attn, r_ij, rl_ij, n_edges):
        # gather per-edge tensors
        x_j = x[edge_neighbor]                      # [E, 1, mult*H]
        q_i = q[edge_center]                        # [E, heads, H/heads]
        k_j = k[edge_neighbor]                      # [E, heads, H/heads]
        v_j = v[edge_neighbor]                      # [E, 1, mult*H]
        X_j = X[edge_neighbor]                      # [E, equi_dim, H]

        # attention
        t_attn = t_ij_attn.reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)  # [E, heads, H/heads]
        attn = (q_i * k_j * t_attn).sum(dim=-1)  # [E, heads]
        attn = scatter_softmax(attn, edge_center, dim=0)  # softmax over incoming edges per node, per head

        if self.scale_edge:
            norm = torch.sqrt(n_edges) / math.sqrt(self.n_atom_basis)  # [E, 1]
        else:
            norm = 1.0 / math.sqrt(self.n_atom_basis)
        attn = attn * norm  # broadcasts: [E, heads] * [E, 1]
        attn = F.dropout(attn, p=self.dropout, training=self.training)  # [E, heads]

        v_jh = v_j.reshape(-1, self.num_heads, (self.n_atom_basis * self.multiplier) // self.num_heads)  # [E, heads, mult*H/heads]
        sea_ij = (attn.unsqueeze(-1) * v_jh).reshape(-1, 1, self.n_atom_basis * self.multiplier)        # [E, 1, mult*H]

        # spatial
        cutoff = self.cutoff_fn(r_ij).view(-1, 1, 1)  # [E, 1, 1]
        spatial_attn = t_ij_filter * x_j * cutoff  # [E, 1, mult*H]

        outputs = spatial_attn + sea_ij  # [E, 1, mult*H]
        chunks = list(torch.split(outputs, self.n_atom_basis, dim=-1))

        o_s_ij = chunks[0]  # [E, 1, H]
        chunks = chunks[1:]

        # direction part
        if self.sep_dir:
            o_d_l_ij = chunks[: self.lmax]
            chunks = chunks[self.lmax :]
            rl_split = _split_to_components(rl_ij[..., None], self.lmax, dim=1)  # list of [E, 2l+1, 1]
            dir_parts = [rl_split[i] * o_d_l_ij[i] for i in range(self.lmax)]    # -> [E, 2l+1, H]
            dX_R = torch.cat(dir_parts, dim=1)                                   # [E, equi_dim, H]
        else:
            o_d_ij = chunks[0]; chunks = chunks[1:]
            dX_R = o_d_ij * rl_ij[..., None]  # [E, equi_dim, H]

        # tensor part
        if self.sep_tensor:
            o_t_l_ij = chunks[: self.lmax]
            X_split = _split_to_components(X_j, self.lmax, dim=1)
            ten_parts = [X_split[i] * o_t_l_ij[i] for i in range(self.lmax)]
            dX_X = torch.cat(ten_parts, dim=1)
        else:
            o_t_ij = chunks[0]
            dX_X = o_t_ij * X_j

        dX = dX_R + dX_X
        return o_s_ij, dX

    def _edge_update(self, EQ_i, EK_j, rl_ij, t_ij) -> torch.Tensor:
        # EQ_i, EK_j: [E, equi_dim, edge_vec_dim], rl_ij: [E, equi_dim, 1], t_ij: [E,1,H]
        if self.sep_htr:
            EQ_split = _split_to_components(EQ_i, self.lmax, dim=1)
            EK_split = _split_to_components(EK_j, self.lmax, dim=1)
            rl_split = _split_to_components(rl_ij, self.lmax, dim=1)
            pairs = []
            for l in range(len(EQ_split)):
                if self.update_info["rej"]:
                    EQ_l = self._vector_rejection(EQ_split[l], rl_split[l])
                    EK_l = self._vector_rejection(EK_split[l], -rl_split[l])
                else:
                    EQ_l, EK_l = EQ_split[l], EK_split[l]
                pairs.append((EQ_l, EK_l))
        elif not self.update_info["rej"]:
            pairs = [(EQ_i, EK_j)]
        else:
            EQr = self._vector_rejection(EQ_i, rl_ij)
            EKr = self._vector_rejection(EK_j, -rl_ij)
            pairs = [(EQr, EKr)]

        w_ij = None
        for EQ_l, EK_l in pairs:
            w_l = (EQ_l * EK_l).sum(dim=1)  # sum over geom dim -> [E, edge_vec_dim]
            w_ij = w_l if w_ij is None else (w_ij + w_l)

        gw = self._gamma_w(w_ij)  # [E, H] (if lin_w>0) else [E, edge_vec_dim]
        if gw.shape[-1] != self.n_atom_basis:
            # If lin_w==0, official expects this to already be mapped to H; enforce mapping here.
            # (Most configs in practice use lin_w>0.)
            raise ValueError("gamma_w did not produce H-dim outputs. Set edge_updates to include 'linw' or 'linwa'.")

        gt = self._gamma_t(t_ij)  # [E, 1, H]
        return gt * gw.unsqueeze(1)  # [E, 1, H]

    def forward(self, edge_index, h, X, rl_ij, t_ij, r_ij, n_edges, num_nodes: int):
        # normalize
        h = self.layernorm(h)
        X = self.tensor_layernorm(X)

        # q,k: computed from h (node-wise)
        q = self.W_q(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)
        k = self.W_k(h).reshape(-1, self.num_heads, self.n_atom_basis // self.num_heads)

        # inter-atomic terms
        x = self._gamma_s(h)  # [N, 1, mult*H]
        v = self._gamma_v(h)  # [N, 1, mult*H]
        t_ij_attn = self.W_re(t_ij)      # [E, 1, H]
        t_ij_filter = self.W_rs(t_ij)    # [E, 1, mult*H]

        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        # message + aggregate (add)
        o_s_ij, dX_ij = self._message(
            edge_center, edge_neighbor, x, q, k, v, X, t_ij_filter, t_ij_attn, r_ij, rl_ij, n_edges
        )
        d_h = scatter_sum(o_s_ij, edge_center, dim=0, dim_size=num_nodes)
        d_X = scatter_sum(dX_ij, edge_center, dim=0, dim_size=num_nodes)

        h = h + d_h
        X = X + d_X

        # edge updates (HTR)
        if (not self.last_layer) and self.edge_updates:
            EQ = self.W_vq(X)  # [N, equi_dim, edge_vec_dim]
            if self.sep_htr:
                X_split = torch.split(X, _get_split_sizes_from_lmax(self.lmax, start=1), dim=1)
                EK = torch.cat([self.W_vk[i](X_split[i]) for i in range(self.lmax)], dim=1)
            else:
                EK = self.W_vk(X)

            EQ_i = EQ[edge_center]
            EK_j = EK[edge_neighbor]
            dt_ij = self._edge_update(EQ_i, EK_j, rl_ij[..., None], t_ij)  # rl needs [...,1]
            t_ij = t_ij + dt_ij

        return h, X, t_ij


@compile_mode("script")
class GotenInteractionModule(GraphModuleMixin, torch.nn.Module):
    """
    A GotenInteractionModule that mirrors the official GotenNet interaction loop:
      init X=0
      for each layer: GATA -> EQFF
    and writes scalar/vector reps back into AtomicDataDict.
    """
    def __init__(
        self,
        num_layers: int,
        r_max: float,
        num_heads: int = 16,
        attn_dropout: float = 0.0,
        layer_norm: bool = False,
        steerable_norm: bool = False,
        edge_updates: Union[bool, str] = True,
        scale_edge: bool = True,
        sep_htr: bool = True,
        sep_dir: bool = False,
        sep_tensor: bool = False,
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        edge_ln: str = "",
        epsilon: float = 1e-8,
        out_field_node: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field_node_eq: str = "vector_representation",
        out_field_edge: str = AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=None,
        out_irreps_node: Optional[Union[o3.Irreps, str]] = None,
        output_ls: Optional[List[int]] = None,
        output_mul: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_layers = int(num_layers)

        self.out_field_node = out_field_node
        self.out_field_node_eq = out_field_node_eq
        self.out_field_edge = out_field_edge
        cutoff = r_max

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.NODE_ATTRS_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.EDGE_SPHARMS_EMB_KEY,
            ],
        )

        # infer dims from irreps
        H = self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY].dim
        sph_irreps = self.irreps_in[AtomicDataDict.EDGE_SPHARMS_EMB_KEY]
        lmax = max(ir.l for _, ir in sph_irreps)
        self.lmax = int(lmax)
        self.equi_dim = ((self.lmax + 1) ** 2) - 1

        self.gata_list = torch.nn.ModuleList([
            _GATA_Official(
                n_atom_basis=H,
                lmax=self.lmax,
                cutoff=cutoff,
                num_heads=num_heads,
                dropout=attn_dropout,
                layer_norm=layer_norm,
                steerable_norm=steerable_norm,
                edge_updates=edge_updates,
                last_layer=(i == num_layers - 1),
                scale_edge=scale_edge,
                sep_htr=sep_htr,
                sep_dir=sep_dir,
                sep_tensor=sep_tensor,
                evec_dim=evec_dim,
                emlp_dim=emlp_dim,
                edge_ln=edge_ln,
                epsilon=epsilon,
            )
            for i in range(num_layers)
        ])

        self.eqff_list = torch.nn.ModuleList([
            _EQFF_Official(n_atom_basis=H, epsilon=epsilon)
            for _ in range(num_layers)
        ])

        # ---- declare outputs / irreps ----
        eq_irreps = o3.Irreps([(H, ir) for _, ir in sph_irreps if ir.l > 0])
        default_out_irreps = o3.Irreps([(H, ir) for _, ir in sph_irreps])
        out_irreps_node = process_out_irreps(
            out_irreps=out_irreps_node,
            output_ls=output_ls,
            output_mul=output_mul,
            default_irreps=default_out_irreps,
        )
        self.has_equivariant_output = any(ir.l > 0 for _, ir in out_irreps_node)

        eq_input_irreps = eq_irreps if self.has_equivariant_output else o3.Irreps("")
        self.node_output_projection = EquivariantScalarMLP(
            in_irreps=(o3.Irreps(f"{H}x0e"), eq_input_irreps),
            out_irreps=out_irreps_node,
            latent_module=ScalarMLPFunction,
            latent_kwargs={
                "mlp_latent_dimensions": [],
                "mlp_nonlinearity": None,
                "use_layer_norm": False,
                "use_weight_norm": False,
                "has_bias": True,
            },
            equiv_linear_module=SO3_Linear,
            output_shape_spec="flat",
        )

        irreps_out = {
            self.out_field_node: out_irreps_node,
            self.out_field_edge: o3.Irreps(f"{H}x0e"),
        }
        if self.out_field_node_eq is not None:
            irreps_out[self.out_field_node_eq] = eq_irreps
        self.irreps_out.update(irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_center = edge_index[0]
        num_nodes = data[AtomicDataDict.NODE_ATTRS_KEY].shape[0]

        # inputs (already embedded by your EmbeddingAttrs stack)
        h0 = data[AtomicDataDict.NODE_ATTRS_KEY]      # [N, H]
        t0 = data[AtomicDataDict.EDGE_ATTRS_KEY]      # [E, H]
        sph = data[AtomicDataDict.EDGE_SPHARMS_EMB_KEY]  # [E, (lmax+1)^2] typically

        # distances (need for cosine cutoff in GATA)
        if AtomicDataDict.EDGE_LENGTH_KEY in data:
            r_ij = data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1, 1)  # [E, 1]
        else:
            # fallback: compute if edge vectors exist
            if AtomicDataDict.EDGE_VECTORS_KEY not in data:
                data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
            r_ij = data[AtomicDataDict.EDGE_LENGTH_KEY].view(-1, 1)

        # rl_ij = spherical harmonics without l=0 (drop the first component)
        # official: rl_ij = sphere(edge_vec)[:, 1:] :contentReference[oaicite:9]{index=9}
        if sph.shape[-1] == (self.lmax + 1) ** 2:
            rl_ij = sph[:, 1:]
        else:
            # If user provided only l>0 already, accept it.
            rl_ij = sph

        # n_edges per edge_center
        ones = torch.ones((r_ij.shape[0], 1), device=r_ij.device, dtype=r_ij.dtype)
        num_edges_per_node = scatter_sum(ones, edge_center, dim=0, dim_size=num_nodes)  # [N, 1]
        n_edges = num_edges_per_node[edge_center]  # [E, 1]

        # init (official)
        H = h0.shape[-1]
        h = h0.unsqueeze(1)                   # [N, 1, H]
        t_ij = t0.unsqueeze(1)                # [E, 1, H]
        X = torch.zeros((num_nodes, rl_ij.shape[1], H), device=h0.device, dtype=h0.dtype)  # [N, equi_dim, H]

        for gata, eqff in zip(self.gata_list, self.eqff_list):
            h, X, t_ij = gata(edge_index, h, X, rl_ij, t_ij, r_ij, n_edges, num_nodes=num_nodes)
            h, X = eqff(h, X)

        # write outputs
        node_scalars = h.squeeze(1)  # [N, H]
        node_equiv_flat = X.permute(0, 2, 1).reshape(num_nodes, -1)  # [N, H*equi_dim]
        if self.has_equivariant_output:
            data[self.out_field_node] = self.node_output_projection((node_scalars, node_equiv_flat))
        else:
            data[self.out_field_node] = self.node_output_projection((node_scalars, None))
        data[self.out_field_edge] = t_ij.squeeze(1)  # [E, H]
        if self.out_field_node_eq is not None:
            data[self.out_field_node_eq] = node_equiv_flat
        return data
