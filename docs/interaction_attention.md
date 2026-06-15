# Interaction Attention

`InteractionModule` supports two attention modes when `use_attention: true`.

## Scalar-only attention weights

Attention logits are always computed from invariant scalar channels only.

- Equivariant features may be weighted by attention.
- Equivariant features are not used directly to form dot-product logits.
- This keeps attention weights invariant scalars, so multiplying equivariant values by those weights preserves equivariance.

## `attention_mode: edge_self`

This is grouped edge self-attention.

```yaml
- _target_: geqtrain.nn.InteractionModule
  num_layers: 2
  latent_dim: 256
  eq_latent_multiplicity: 8
  use_attention: true
  attention_mode: edge_self
  attention_head_dim: 64
```

For each center node, every outgoing edge attends to every other outgoing edge sharing that center.

- Query: scalar edge latent.
- Key: scalar edge latent.
- Value: equivariant edge environment.
- Complexity: roughly `sum_center degree(center)^2`.

Use this when the graph degree is modest and explicit edge-edge comparison is desired.

## `attention_mode: node_feature_query`

This is recurrent node-state query attention.

```yaml
- _target_: geqtrain.nn.InteractionModule
  num_layers: 2
  latent_dim: 256
  eq_latent_multiplicity: 8
  use_attention: true
  attention_mode: node_feature_query
  attention_head_dim: 64
  node_state_field: node_features
  node_state_pooling: mean
  node_state_use_residual: true
```

The module maintains a scalar node state with irreps `latent_dim x 0e`.

- At the start, if `node_state_field` exists and has dimension `latent_dim`, it is used.
- Otherwise, the node state is initialized from `node_invariant_field` (`node_attrs` by default).
- After each interaction layer, the updated scalar edge latent is pooled over center nodes and saved to `node_state_field`.
- The next layer uses this recurrent node state as the query.
- By default, repeated node-state poolings use the same variance-preserving residual update form used by the interaction latent streams.

Attention details:

- Query: scalar recurrent node state at the edge center.
- Key: scalar edge latent.
- Value: equivariant edge environment.
- Softmax group: outgoing edges with the same center node.
- Complexity: linear in the number of edges.

Use this for larger graphs or when the query should depend on previous interaction context rather than static node attributes.

Node-state pooling options:

- `node_state_pooling: mean`: average updated scalar edge states per center node.
- `node_state_pooling: sum`: sum updated scalar edge states per center node.
- `node_state_pooling: attention`: compute one scalar logit per edge from the updated scalar edge state, softmax over edges with the same center, and use the result for weighted pooling.

Node-state residual options:

- `node_state_use_residual: true` is the default.
- `node_state_residual_update_max` defaults to `residual_update_max`.
- Set `node_state_use_residual: false` to overwrite the node state after each pooling.

## Relation to `EdgewiseReduce`

`EdgewiseReduce(use_attention: true)` uses scalar edge features for grouped edge self-attention before reducing edges to nodes. It does not use static `node_attrs` as attention queries.

For full equivariant node features after an `InteractionModule`, keep the usual explicit post-interaction pooling:

```yaml
- _target_: geqtrain.nn.EdgewiseReduce
  field: edge_features
  out_field: node_features
  use_attention: false
```

The internal `node_feature_query` state is scalar-only and exists to provide dynamic attention queries inside the interaction stack.

If downstream heads need full equivariant node features, keep an explicit post-interaction `EdgewiseReduce` from `edge_features` to `node_features`. In that case, use a distinct internal state field such as `node_attention_features`:

```yaml
- _target_: geqtrain.nn.InteractionModule
  use_attention: true
  attention_mode: node_feature_query
  node_state_field: node_attention_features

- _target_: geqtrain.nn.EdgewiseReduce
  field: edge_features
  out_field: node_features
```
