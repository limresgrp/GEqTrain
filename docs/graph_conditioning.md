# Scalar Graph Conditioning

GEqTrain supports scalar graph-level inputs as optional conditioning for modules
that expose `conditioning_fields`.

The recommended pattern is explicit:

```yaml
model:
  stack:
    - ${stack_blocks.node_input_attrs}
    - ${stack_blocks.graph_input_attrs}
    - ${stack_blocks.edge_radial_attrs}
    - ${stack_blocks.edge_angular_attrs}
    - _target_: geqtrain.nn.EmbeddingAttrs
      name: attrs
      node_out_irreps: 512x0e
      edge_out_irreps: 512x0e
    - _target_: geqtrain.nn.InteractionModule
      name: interaction
      conditioning_fields: [graph_attrs]
      # ...
    - _target_: geqtrain.nn.ReadoutModule
      name: head
      field: node_features
      out_field: target
      out_irreps: 1x0e
      conditioning_fields: [graph_attrs]

graph_attributes:
  temperature:
    attribute_type: numerical
    embedding_dimensionality: 1
  condition_id:
    attribute_type: categorical
    embedding_mode: embedding
    num_types: 8
    embedding_dimensionality: 16

eq_graph_attributes: {}
```

`EmbeddingInputAttrs` handles graph attributes the same way it handles scalar
node and edge attributes:

- numerical graph attributes are concatenated directly into `graph_attrs`;
- categorical graph attributes can use `one_hot`, `embedding`, or `positional`;
- equivariant graph attributes are stored in `graph_eq_attrs`, but they are not
  valid conditioning inputs for scalar MLP FiLM paths.

Only scalar `0e` fields can be used as conditioning. If `graph_attrs` is listed
in `conditioning_fields`, modules broadcast it to the needed granularity:

- node-level MLPs receive `graph_attrs[batch]`;
- edge-level MLPs receive `graph_attrs[batch[edge_center]]`;
- graph-level readouts receive graph rows directly.

This is deliberately opt-in. Adding a graph attribute to a dataset should not
silently change model architecture or checkpoint compatibility. Add
`conditioning_fields: [graph_attrs]` to each module where the graph context
should affect computation.

Currently supported consumers:

- `InteractionModule`: graph conditioning is passed to its edge-level
  `EquivariantScalarMLP` blocks, including initial latent generation,
  interaction projections, and final projection.
- `ReadoutModule`: graph conditioning is broadcast to node readouts or used
  directly for graph readouts.
- `GotenInteractionModule`: graph conditioning is broadcast internally to both
  node and edge conditioning tensors.
