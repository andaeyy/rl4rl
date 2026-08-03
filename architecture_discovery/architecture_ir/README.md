# Architecture IR prototype

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

Validation checks ports, tensor shapes, instantaneous cycles, state-edge
targets, causal attention declarations, input-to-output reachability, and dead
nodes. Parameter count may appear in metadata but has no effect on validity.

`RuntimeBindings` maps validated attention node IDs to modules constructed by
the trusted interpreter. Runtime probes then measure attention execution,
future-token causality, sequence dependence, attention intervention effects,
gradient-based parameter influence, mask-buffer evidence, and device
placement. Class names and source words are never sufficient evidence.

This package does not yet provide a complete trusted PyTorch interpreter for
every primitive. Until that interpreter exists, the IR is a validated schema
and runtime-evidence contract, not a replacement execution path.
