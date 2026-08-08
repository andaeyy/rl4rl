# Semantic Autoresearch Protocol

This controller uses the declarative architecture IR. This IR-only protocol
supersedes any legacy instruction in a shared prompt to emit Python,
`build_untrained_model`, or SEARCH/REPLACE blocks. Never return executable
source code.

Maintain a frozen categorical archive of evaluator-reported architecture
signatures. Only successfully evaluated candidates that satisfy the public
parent-eligibility rule may become future parents.

For each proposal opportunity:

1. Select an eligible parent from the least-used occupied archive cell.
2. State one architectural mechanism hypothesis in the IR `metadata` object
   under the `mechanism_hypothesis` key.
3. Return one complete replacement architecture as strict JSON matching
   `architecture_tensor_graph` schema version `1.0`. A single `json` fenced
   block is permitted, but prose outside the JSON is forbidden.
4. Let the trusted evaluator initialize, train, checkpoint, and evaluate it.
5. Preserve the candidate in a new categorical cell, or replace the occupant of
   the same cell only when public search accuracy is higher.
6. Preserve every rejected, malformed, and failed proposal in the append-only
   lineage without making it a parent.

Categorical coverage is an exploratory search-diversity device. A new cell is
not evidence of scientific novelty, mechanism validity, or external validity.
Category codes are labels, not ordered fitness values. Parameter count remains
descriptive metadata and must never affect proposals, retention, parent
selection, tie-breaking, or stopping.

Only evaluator-owned, versioned primitive names are permitted. Python
callbacks, import paths, commands, file paths, checkpoints, and arbitrary
expressions are forbidden.
