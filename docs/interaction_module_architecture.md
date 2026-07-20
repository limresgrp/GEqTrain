# InteractionModule Architecture (Current)

This note describes the current `geqtrain.nn.InteractionModule` implementation in `geqtrain/nn/interaction.py`.

## 1) What the module computes

`InteractionModule` builds edge-level latent states from node/edge inputs and updates them through equivariant message passing.
It outputs `data[out_field]` (usually `edge_features`) with requested output irreps.

Core idea:
- keep a scalar latent stream (`[E, latent_dim]`)
- keep an equivariant latent stream (`[E, mul, feat_per_mul]` in channel-wise layout)
- iteratively update both streams with local edge->node->edge context and tensor products

---

## 2) Inputs and feature families

Main input families (all optional except radial/spherical embeddings):
- Node invariant (scalar) features: `node_invariant_field` (default `node_attrs`)
- Node equivariant features: `node_equivariant_field` (default `node_eq_attrs`)
- Edge invariant features: `edge_invariant_field` (default `edge_attrs`)
- Edge equivariant features: `edge_equivariant_field` (default `edge_eq_attrs`)
- Edge radial invariant embedding: `edge_radial_emb_field` (default `radial_emb`)
- Edge spherical equivariant embedding: `edge_spharm_emb_field` (default `spharms_emb`)

Graph connectivity:
- `edge_center = edge_index[0]`
- `edge_neigh = edge_index[1]`

---

## 3) Forward flow (non-recycling mode)

### Step A: Build initial edge latents

Scalar latent input is concatenated from:
- edge radial embedding
- edge invariant input (if present)
- node invariant at edge center and neighbor (if present)

Equivariant latent input is concatenated from:
- edge spherical embedding
- edge equivariant input (if present)
- node equivariant at edge center and neighbor (if present)

Then a fixed permutation (`eq_concat_permutation`) reorders equivariant blocks to match expected irreps layout.

### Step B: Initial latent projection

`initial_latent_generator` (`EquivariantScalarMLP`) maps:
- scalar input -> scalar latent `[E, latent_dim]`
- equivariant input -> equivariant latent `[E, mul, feat_per_mul]`

### Step C: Interaction stack

Each `InteractionLayer` performs:
1. `env_embed_mlp((scalar_latent, equiv_latent))` -> edge equivariant environment features
2. Optional attention:
   - queries from node invariants at edge centers
   - keys from scalar latent
   - scaled dot-product over heads
   - optional logit clipping (`attention_logit_clip`)
   - softmax grouped by edge center
3. Aggregate edge environments to nodes with `scatter_sum(..., edge_center)`
4. Optional MACE product on node equivariant context
5. Normalize node context (`SO3_LayerNorm`)
6. Broadcast node context back to edges (`env_nodes[edge_center]`)
7. Tensor product (`Contracter`) of edge equivariant latent with local node context
8. Normalize TP output (`SO3_LayerNorm`)
9. Project TP output to new scalar/equivariant latents
10. Residual update:
    - scalar stream always residual-mixed (when coeff available)
    - equivariant stream optionally residual-mixed (`use_equivariant_residual`)

Residual coefficients are controlled by learned params (`_latent_resnet_update_params`) and bounded by `residual_update_max`.

### Step D: Final projection

After the stack, `final_projection` maps latent streams to requested output irreps (`out_irreps` / `output_ls` / `output_mul`).

---

## 4) Fixed-point recycling mode (`use_fixed_point_recycling`)

When enabled, the stack is treated as a recurrent map `F(s, e)` and solved iteratively.

### Why layer behavior changes

To keep a consistent recurrent state:
- per-layer "last layer" shrink/filtering is disabled inside the loop
- latent in/out irreps remain constant during recycling
- output irreps filtering is done only once in the final projection

### Solver structure

`_fixed_point_refine` does:
1. No-grad iterative solve (`fp_max_iter`, `fp_tol`) with damped update
2. Optional short grad-enabled refinement (`fp_grad_steps`) from detached fixed point

Update form:
- `state_next = state + alpha * (F(state) - state)`

Convergence control knobs:
- `fp_alpha`
- `fp_adaptive_damping`, `fp_alpha_min`, `fp_residual_growth_tol`
- `fp_first_layer_update_coeff`
- `fp_state_clip_value`
- `fp_enforce_inverse_quadratic`
- `fp_use_static_context`, `fp_static_context_strength`

### Static-context recycling (new)

When `fp_use_static_context: true`, the module keeps a **fixed anchor state** built from the original node/edge attrs (the output of `initial_latent_generator` before recycling starts).

At every recycle iteration:
- dynamic scalar latent and static scalar anchor are concatenated and fused back to `latent_dim`
- dynamic equivariant latent and static equivariant anchor are flattened to irreps-flat layout, concatenated, fused with an equivariant linear map, then reshaped back
- the fused state is then passed to the interaction stack

So the recurrent map becomes effectively `F(dynamic, static_anchor)`, where only `dynamic` evolves and `static_anchor` stays fixed.
This gives each recycle step direct access to stable input information and reduces dilution/drift over many iterations.

`fp_static_context_strength` controls blending:
- `1.0`: fully use fused (concat-projected) state
- `<1.0`: interpolate between current dynamic state and fused state

If inverse-quadratic mode is enabled, applied residual is capped by an envelope proportional to `1 / (k+1)^2`.

### Returned solver stats (internal API)

`_fixed_point_refine` returns:
- refined scalar latent
- refined equivariant latent
- `iters_used`
- `last_residual`

`forward()` currently uses the refined latents and discards stats.

---

## 5) Shape summary

Assuming `E = #edges`, common multiplicity `M`:
- Scalar latent: `[E, latent_dim]`
- Equivariant latent (channel-wise): `[E, M, feat_per_mul]`
- Final output (`out_field`): `[E, out_irreps.dim]`

---

## 6) Practical interpretation

- Node features influence edge updates in two ways:
  - direct center/neighbor injection into initial edge latent
  - indirect node context aggregation from edge messages (`edge -> node -> edge`)
- Edge features carry most of the recurrent state.
- Recycling mode turns the interaction stack into an iterative solver over edge latent states and is useful when deeper effective context is needed without manually stacking many blocks.
