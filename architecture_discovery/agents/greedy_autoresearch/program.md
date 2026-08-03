# Greedy Autoresearch Protocol

Maintain one accepted candidate.

For each iteration:

1. Read the accepted candidate and append-only ledger.
2. State one architectural hypothesis before editing.
3. Produce SEARCH/REPLACE blocks for the architecture-only candidate.
4. Let the shared evaluator initialize and train the child from scratch.
5. Accept the child when it passes the validity gate and the frozen robustness floor.
6. Preserve rejected and crashed code artifacts before restoring the incumbent.
7. Stop at the configured candidate limit.

The control prompt does not provide architecture-family labels. Use your own analysis of the code and observed failures.

Parameter count stays in metadata. Do not use it in proposals or decisions.
