# Architecture IR v1 construction contract

The parent document is an example, not a template restriction. You may change
widths, depth, topology, normalization, feed-forward type, positional mechanism,
residual algebra, routing, composition, attention head count, and readout
within this registered vocabulary:

- `input`: rank-2 token IDs; no attributes.
- `token_embedding`: `vocab: 15`; output `[Batch, Time, H]`.
- `positional`: required `mechanism` in `learned|sinusoidal`.
- `attention`: required `causal: true` and positive `heads`; optional Boolean
  `bias`; `H` must be divisible by `heads`.
- `normalization`: required `mechanism` in `layer_norm|rms_norm`; optional
  positive `epsilon` and Boolean `affine`.
- `feed_forward`: required `mechanism` in `gelu|gated` and positive
  `hidden_dimension`; optional Boolean `bias`; gated blocks may set
  `activation` to `gelu|silu`.
- `algebraic`: required `mechanism` in `add|learned_gate|sigmoid_gate`.
- `routing`: required `mechanism` in `softmax_mix|fixed_mix`; softmax routing
  may set positive `temperature`; fixed routing requires one finite float32
  `weights` value per input with a positive absolute-value sum.
- `composition`: required `mechanism` in `identity|concat_project`; concat
  projection may set Boolean `bias`.
- `readout`: `vocab: 15`; optional Boolean `bias`; optional `tie_embedding`
  names a token-embedding node with matching width.

Every non-input port must have exactly one incoming edge. Edge kinds are
`data`, `residual`, or `routing`; state and recurrent edges are unavailable in
v1. Input shapes must exactly match the source output shape. Every node must be
on the input-to-output path, the graph must be acyclic, and every declared
causal attention node must have a measurable effect on logits at runtime.

Graph metadata must preserve `max_sequence_length` and `vocab_size: 15`.
`graph_id` and every `node_id` must start with a letter and contain only letters,
digits, `_`, `.`, `:`, or `-`. Use concrete hidden widths in rank-3 shapes.
Resource ceilings control executable compute and memory; they are validity
constraints, never an architecture-size objective.

Changing only `graph_id`, metadata, node names/order, or non-executable edge
labels is not an architectural proposal. Native controllers compare the
normalized executable-structure `architecture_hash` and reject such no-op
rewrites before training.

Return a complete valid graph. Do not limit proposals to merely resizing the
parent: use the registered primitives compositionally to test genuinely
different information-flow mechanisms.
