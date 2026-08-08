# Trusted architecture IR

The IR is a declarative tensor and module graph. Candidate data can specify
shapes, topology, attention, normalization, feed-forward transformations,
position mechanisms, routing, algebraic composition, recurrence, and state.
It cannot contain Python source, callables, imports, commands, checkpoints, or
state dictionaries.

Novel mechanisms are not limited to a fixed template. A mechanism can be added
as a versioned `CustomPrimitiveSpec`, implemented by trusted evaluator code,
and then referenced by candidate graphs. The spec freezes input arity, tensor
ranks, state-cycle behavior, and allowed attributes. Unregistered custom
primitives fail validation.

Validation checks bounded JSON size and nesting, ports, tensor shapes,
instantaneous cycles, state-edge targets, causal attention declarations,
input-to-output reachability, dead nodes, and a conservative training-workspace
estimate before any candidate tensors are allocated. Parameter count may appear
in metadata but has no effect on validity.

`RuntimeBindings` maps validated attention node IDs to modules constructed by
the trusted interpreter. Runtime probes then measure attention execution,
future-token causality, sequence dependence, per-attention-node intervention
effects, gradient-based parameter influence, mask-buffer evidence, and device
placement. Every declared attention node must individually affect logits.
Class names and source words are never sufficient evidence.

`load_and_build_ir_candidate(path, seed)` is the execution entry point. It
reads bounded UTF-8, rejects duplicate/unknown JSON fields, applies graph and
interpreter validation before allocation, and deterministically constructs a
fresh CPU model. It returns an `InterpretedCandidate` containing the canonical
graph, validation result, model, trusted metadata, and `RuntimeBindings`.
`validate_ir_candidate_path(path)` performs the same validation without model
allocation. `validate_ir_candidate_json(text)` is available to controllers for
strict response preflight and canonicalization.

## Version 1 primitive contract

All sequence tensors have shape `[batch, time, concrete_hidden_width]`. The
graph metadata must declare `max_sequence_length`; Phase 1 vocabulary metadata,
when present, must be `vocab_size: 15`.

| Kind | Required attributes | Optional attributes | Supported mechanisms |
|---|---|---|---|
| `input` | none | none | rank-2 token IDs |
| `token_embedding` | `vocab: 15` | none | learned lookup |
| `positional` | `mechanism` | none | `learned`, `sinusoidal` |
| `attention` | `causal: true`, `heads` | `bias` | dense causal self-attention |
| `normalization` | `mechanism` | `epsilon`, `affine` | `layer_norm`, `rms_norm` |
| `feed_forward` | `mechanism`, `hidden_dimension` | `bias`, gated-only `activation` | `gelu`, `gated` (`gelu` or `silu` activation) |
| `algebraic` | `mechanism` | none | `add`, `learned_gate`, `sigmoid_gate` |
| `routing` | `mechanism` | mechanism-specific `temperature` or `weights` | `softmax_mix`, `fixed_mix` |
| `composition` | `mechanism` | concat-only `bias` | `identity`, `concat_project` |
| `readout` | `vocab: 15` | `bias`, `tie_embedding` | linear logits readout |

`recurrent`, `state`, `custom`, state edges, and recurrent edges are explicit
fail-closed cases in interpreter version 1. A novel operation becomes
executable only after trusted evaluator code and a new versioned primitive
contract are reviewed and installed; candidates cannot install implementations.

Hard default safety ceilings are 128 nodes, 512 edges, fan-in 16, sequence
length 512, hidden width 2,048, feed-forward width 8,192, 128 attention heads,
64 million parameters, 16 million buffer elements, and a conservative 4 GiB
training-workspace estimate at the frozen full-profile batch and sequence
dimensions. These are pre-allocation resource controls. Parameter count remains
descriptive metadata and is never a fitness, parent-selection, archive, or
tie-break objective.

`graph_hash` identifies the entire canonical document, including provenance
metadata. `architecture_hash` identifies executable architecture structure and
therefore excludes `graph_id`, metadata, node naming/order, and v1 edge labels
that have no distinct runtime behavior. Native controllers reject already-seen
`architecture_hash` values, so a descriptive or identifier-only rewrite cannot
consume candidate training compute or be selected as an architectural
discovery.

The checked-in `common/initial_candidate.ir.json` represents the former
conventional Python seed: two pre-normalized causal decoder blocks, learned
token and position embeddings, GELU feed-forward layers, residual additions,
final layer normalization, and a token-embedding-tied readout. The interpreter
build contains 6,080 trainable parameters, matching that control architecture.
