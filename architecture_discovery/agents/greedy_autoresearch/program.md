# Greedy Autoresearch Protocol

This controller uses the declarative architecture IR. This IR-only protocol
supersedes any legacy instruction in a shared prompt to emit Python,
`build_untrained_model`, or SEARCH/REPLACE blocks. Never return executable
source code.

Maintain one accepted candidate.

For each iteration:

1. Read the accepted candidate and append-only ledger.
2. State one architectural hypothesis in the IR `metadata` object under the
   `mechanism_hypothesis` key.
3. Return one complete replacement architecture as strict JSON matching
   `architecture_tensor_graph` schema version `1.0`. A single `json` fenced
   block is permitted, but prose outside the JSON is forbidden.
4. Let the shared evaluator initialize and train the child from scratch.
5. Accept the child when it passes the validity gate and the frozen robustness floor.
6. Preserve rejected and crashed response artifacts before restoring the incumbent.
7. Stop at the configured candidate limit.

The control prompt does not provide architecture-family labels. Use your own
analysis of the typed graph and observed failures. Only evaluator-owned,
versioned primitive names are permitted. Python callbacks, import paths,
commands, file paths, checkpoints, and arbitrary expressions are forbidden.

Parameter count stays in metadata. Do not use it in proposals or decisions.
