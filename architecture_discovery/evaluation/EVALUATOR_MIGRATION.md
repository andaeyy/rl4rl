# Evaluator integration required

This package establishes the Layer A, B, and C contracts, but it does not edit
the existing shared evaluator or controller entrypoints. Those files are owned
by the integration coordinator.

## Current leak

`common/evaluator.py` imports `private_eval.shadow_evaluator` and computes
`robustness_score`, `qualifies`, and `combined_score` from shadow, edge, and
carry outcomes. Controllers import `evaluate_candidate` from that module and
then use those fields for retention. The transitive dependency audit therefore
correctly reports a route from each controller to `private_eval`.

## Required shared-file patch

1. Split public Layer A generation and scoring into a module with no import of
   `private_eval` or `sealed_eval`.
2. Make the online evaluator return `evaluation.SearchEvaluationRecord`.
3. Compute `public_accuracy`, `search_score`, and `eligible_for_parent` only
   from the frozen Layer A plan. Do not read shadow, edge, carry, Layer B, or
   Layer C results.
4. Pass only `record.controller_view()` or a
   `ControllerEvaluationInbox` view into selection, prompting, repair,
   stopping, lineage-visible metrics, and semantic archive code.
5. Move shadow, edge, and carry evaluation into a trusted Layer B adapter. The
   adapter must accept only a `FrozenRunSnapshot` created after a terminal run
   event.
6. Replace generic `DiscoveryEvaluation` use in primary C0-C3 controllers. It
   may remain temporarily in explicitly labeled legacy regression commands,
   but those commands must fail the scientific-readiness dependency audit.
7. Update OpenEvolve fitness policy to use only the explicit Layer A fields.
8. Stop flattening sealed records into controller lineage JSON.

After integration, run the transitive audit against every primary controller:

```python
from pathlib import Path

from evaluation.dependency_audit import assert_controller_dependencies_clean

root = Path(__file__).resolve().parents[1]
entries = tuple((root / "agents").glob("*/run.py"))
assert_controller_dependencies_clean(entries, project_root=root)
```

Do not suppress that failure by adding an allowlist for `common.evaluator`.
Remove the transitive sealed dependency instead.

