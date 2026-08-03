# End-to-End Adversarial Readiness Audit

Date: 2026-07-31  
Scope: offline code and synthetic-fixture audit only  
Scientific launch decision: **BLOCKED**

No provider API, network request, candidate training, or full MPS run was made
during this audit. The six adversarial regressions that initially exposed five
code defects are now ordinary passing tests; their assertions were not weakened.

## Executive finding

The new primary path has materially better separation and accounting than the
legacy controller plumbing. Layer A exposes a narrow record, C0-C3 share a study
engine, failures stay in run-level ITT records, MPS work is serialized, and
ordinary artifact edits are detected. These properties are testable offline.

The system is not ready for a scientific run. Arbitrary candidate Python still
has no proven OS boundary, the typed IR is not the evaluator's scientific
execution path, no real MPS `full_train_v1` run has completed, and checkpoint and
event-chain roots are not anchored outside the mutable local study directory.

## Attack results

| Attack | Result | Evidence or blocker |
| --- | --- | --- |
| Shadow, Layer B, or Layer C feedback reaches a primary controller | Pass for current static and typed boundaries | Transitive controller audit is clean; injected `shadow_accuracy` is rejected; controller view contains only `CONTROLLER_SEARCH_FIELDS`. Dynamic string construction is outside the static audit and remains part of containment. |
| String values spoof trusted booleans | Pass after repair | Untrusted Layer A records and frozen study, novelty, mechanism, replication, analysis, and research-protocol records now require exact booleans. The adversarial suite rejects `"false"` instead of treating it as true. |
| Boolean or fractional values spoof scientific counts | Pass after final hardening | Study budgets, training resources, mechanism/replication seeds, primary outcomes, time-to-first records, and resume checkpoints now require exact non-boolean integers; negative fractions can no longer truncate to zero. |
| Resumed active state changes parents or transition exposure | Pass after final hardening | Active fields use an exact schema; parents and scheduled transition exposure are recomputed from the frozen run, while provider-attempt and repair counts must reconstruct from the budget ledger. |
| Provider retries or parse failures create extra proposal opportunities | Pass | All C0-C3 arms retain one seed evaluation and three descendant opportunities under the same hostile behavior. One provider retry plus one capped parse-repair produces five provider attempts, two parse failures, one repair, and equal training attempts in every arm without adding a proposal opportunity. |
| Initial seed is counted differently by treatment | Pass | The common engine evaluates one seed before proposal opportunity 1 in every C0-C3 run. Seed work remains visible in total resources and outside the descendant opportunity count. |
| Resume changes a plainly edited randomization table | Pass | Existing per-run and plan hashes reject edits that are not rehashed. |
| Resume accepts a rewritten and fully rehashed randomization table | Pass after repair | The loader regenerates the deterministic assignment from `StudySpec` and rejects a self-consistent but changed order in `test_rehashed_randomization_cannot_change_order_on_resume`. This does not replace an external pre-run anchor. |
| Multiple candidates in one run inflate one canonical mechanism | Pass on canonical clustering path | Renaming nodes/classes, adding novelty wording, and changing width still creates one cluster and one run contribution. |
| Relabelling the same mechanism key as two cluster IDs inflates outcome | Pass after repair | `novelty.unique_cluster_counts_by_run` and reconstruction deduplicate the canonical `mechanism_cluster_key`; relabelled presentation IDs do not add an outcome. |
| Descriptor or class-name wording creates scientific novelty | Pass | Scientific signatures ignore presentation attributes and have no dependency on online descriptor extraction. |
| New architecture loads a checkpoint from another candidate | Pass for identity mismatch | Resume validation binds candidate hash, profile, task, and seed bundle. Same-identity checkpoint integrity still needs an externally anchored digest. |
| Scientific profile silently runs on CPU | Pass | `full_train_v1` rejects CPU even with the engineering CPU flag. |
| PyTorch silently falls back from MPS | Pass at launch guard | A truthy `PYTORCH_ENABLE_MPS_FALLBACK` blocks strict MPS worker creation. Real operation-level MPS fallback behavior still needs the full hardware run. |
| A syntactic MPS receipt uses `1/0` as booleans or points at modified artifacts | Pass at receipt validator | The audit requires exact booleans and step integers, rehashes the training manifest, summary, and candidate source, and cross-checks the full profile, MPS device, fallback, containment, cleanup, and hardware-match fields. No real MPS receipt exists yet. |
| Two MPS training leases coexist | Pass | A second lease fails closed while the first exists. Crash-stale leases still require deliberate operator recovery. |
| Worker inherits provider credentials or shadow seed | Pass for environment inheritance | Worker environment is allowlisted and test secrets are absent. This does not prevent reading credentials from unrestricted files. |
| Candidate opens files, sockets, or child processes through direct syntax | Detected, but not safely contained | Static risk scanner catches direct fixtures. Static detection is defense in depth, not an OS boundary. |
| Candidate recovers capabilities through `globals`, `getattr`, and an indirect `__import__` | Pass at source gate after repair | The executed candidate contract now consumes the containment risk scan and rejects the indirect-capability fixture before import. This remains defense in depth, not scientific OS containment. |
| No-search condition receives parent, score, transition, repair, or history data | Pass for provider-visible input | Rich contexts produce the same constant model input; leaked parent/history markers never enter messages. A real injected-client provider adapter now exists. Scientific assignment through the scheduler and scientific artifact emission remain open. |
| Failed assigned run is removed before ITT analysis | Pass at `RunOutcomeTable`; pass in reconstruction after external roster integration | The analysis table requires an exact frozen assignment roster. Callers must obtain that roster from the frozen randomization authority, not infer it from observed runs. |
| Candidate rows are treated as independent statistical replicates | Pass | Candidate schemas are rejected and duplicate run rows fail. |
| Smoke case counts enter a scientific evaluation profile | Pass | A 64-case Layer B plan is rejected below the 10,000-case scientific floor. PI-approved scientific case counts remain unresolved. |
| Artifact payload or content-addressed object is modified in place | Pass | Payload-hash/envelope-hash verification and object-digest verification reject both attacks. |
| Entire local event chain is rewritten and rehashed | **External integrity blocker** | Local hashes provide tamper evidence only while at least one trusted digest is retained elsewhere. There is no signed, WORM, or independently stored study anchor yet. |
| Candidate/scientific failure is automatically rerun | Pass | Only preregistered infrastructure failure classes authorize linked reruns. |
| Non-finite MPS compute ceiling bypasses budget comparison | Pass after repair | Budget and resource validation require finite floating-point values; NaN and infinity are rejected by hard regression tests. |

## Resolved shared-file defects

### Resolved: bind randomization to deterministic authority

`study/randomization.py::load_or_create_plan` now regenerates the expected plan
from the supplied `StudySpec` and output root and requires the stored canonical
assignment to match. A hash stored inside the mutable plan still cannot
authenticate the full local study against an attacker who can rewrite everything.

Minimal behavior:

```python
expected = generate_plan(spec, output_root)
if plan.assignment_hash != expected.assignment_hash:
    raise ValueError("frozen randomization differs from deterministic StudySpec assignment")
```

The remaining action is to retain the frozen plan hash in a signed, WORM, or
independently controlled manifest before the first run.

### Resolved: reject type-confused evaluation records

`evaluation/records.py` now uses an exact boolean validator rather than coercing
untrusted values with `bool(payload[field])`.

```python
def require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be boolean")
    return value
```

### Resolved: count canonical mechanism keys

`novelty/clustering.py::unique_cluster_counts_by_run` now adds
`mechanism_cluster_key` to each run's set, so mutable presentation labels cannot
inflate the scientific outcome.

### Resolved at source gate: scan executed exploratory Python

`common/candidate_contract.py` now combines the structural contract with
`containment.source_scan.scan_python_source` and rejects risky source before
module import. This closes the tested indirect-builtin gap for engineering runs,
but it does not replace the scientific OS boundary.

The scientific path must continue to fail closed until either:

1. A candidate-bound, authenticated OS attestation proves filesystem, network,
   credential, process, resource, identity, and sandbox isolation on MPS; or
2. The evaluator executes a validated architecture IR through a trusted
   interpreter that cannot recover Python capabilities.

### Resolved: reject non-finite budgets and resource actuals

`study/budget.py` and `study/interfaces.py` now require finite MPS seconds,
scores, and floating-point resource values before comparison or serialization.

## Open P1: externally anchor checkpoints and event chains

Resume metadata prevents accidental cross-candidate reuse but does not prove that
the stored model state is the previously committed state. After each resume
checkpoint write:

- hash the checkpoint;
- append its digest to the immutable event ledger;
- anchor the run-chain head and frozen randomization hash outside the mutable run
  directory;
- verify both before resume;
- preserve the old checkpoint object by digest instead of overwriting the only
  copy.

## External launch blockers

These cannot be cleared by offline unit tests:

- No real strong-containment attestation exists for arbitrary Python on macOS/MPS.
- Architecture IR is not yet the evaluator-owned scientific construction path.
- `full_train_v1` has not completed on an MPS-available machine.
- No populated, reviewed, frozen reference corpus exists.
- Independent blinded reviewers and final adjudication policy are not in place.
- Pilot run outcomes, dispersion estimate, smallest effect, power, alpha, and
  primary contrast are not frozen.
- C0-C3 plus no-search have not completed a paid pilot through the integrated
  study scheduler and artifact ledger.
- Layer C has not and should not be released during infrastructure work.
- An external-validity task or scaling grid remains a PI decision.

## Readiness decision

Keep `readiness_evidence.yaml` status `blocked`. Passing this audit means only that
the offline guards behave as described. Scientific launch still requires the
external checkpoint/event-chain anchor, every external gate above, full repository
tests, and an independently verified MPS pilot artifact.

Final coordinator verification: **all 285 collected repository tests passed**.
The integrated offline C0-C3 plus no-search study completed and exactly resumed
at `/private/tmp/architecture-discovery-final-offline-20260731-v3`; it recorded
zero provider calls and zero Torch training runs. Provider-free reproducibility
report generation also completed with report digest
`2eeaec354df138baf679719d44e968feb027dd056781b772f7bd47cb4ed25a0d`.
This verification made no provider, network, candidate-training, or full-MPS
call.
