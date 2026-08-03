# Shadow evaluation boundary

This directory holds the shadow, edge-case, and carry-chain evaluation split. The
controller prompts explicitly prohibit reading it.

This is an experimental separation, not a security boundary: every process runs
under the same local user and can technically read the workspace. A defensible
study should run the shadow evaluator in a separate service or account and expose
only aggregate validity and score fields to the search controllers.

When `DISCOVERY_SHADOW_SEED` is unset, setup smoke tests use an unlogged,
process-local random seed. Controlled study runs should inject a fixed secret seed
into the isolated evaluator service, not into the controller environment.
