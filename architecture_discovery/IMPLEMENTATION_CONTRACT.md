# Architecture Discovery Implementation Contract

Status: engineering contract for the July 2026 readiness work. This file does
not authorize a paid pilot or a scientific run.

## Scientific boundary

The primary experiment uses one project-owned engine with a two-by-two design:

| Condition | Parent policy | Proposal policy |
| --- | --- | --- |
| C0 | single | ordinary |
| C1 | single | scheduled transition |
| C2 | portfolio | ordinary |
| C3 | portfolio | scheduled transition |

Only those two treatment fields may vary. Native Greedy Autoresearch and
OpenEvolve remain secondary system replications. All candidate training is
evaluator-owned, from scratch, and sequential on MPS. Parameter count is
descriptive metadata and never a reward or tie-breaker.

## Evaluation boundary

- Layer A is controller-visible public search evaluation.
- Layer B is post-run sealed qualification of a frozen snapshot.
- Layer C is one-shot confirmation under a release authorization.
- Layer B or C fields, cases, thresholds, paths, artifacts, and outcomes must
  never enter controller imports, prompts, metrics, retention, repair, or
  stopping decisions.
- Current shadow, edge, and carry cases are treated as sealed until the PI
  explicitly designates separate public Layer A generators.

The controller accepts only an explicit `SearchEvaluationRecord` view. It must
not receive a generic dictionary containing arbitrary evaluation artifacts.

## Stable record envelope

Scientific records carry these common fields where applicable:

- `schema_name` and `schema_version`
- stable `record_id`
- `study_id`, `block_id`, `run_id`, and `condition_id`
- UTC creation time
- writer component and code/config/environment hashes
- parent, ancestry, training, snapshot, and artifact references
- payload hash and previous-event hash for event records

Required logical records are:

- `StudySpec`, `BlockSpec`, `RunSpec`, `ConditionSpec`
- `BudgetSpec`, `BudgetLedger`
- `ProposalRecord`, `CandidateRecord`, `TrainingRecord`
- `SearchEvaluationRecord`, `QualificationEvaluationRecord`,
  `ConfirmationEvaluationRecord`
- `MechanismClusterRecord`, `NoveltyReviewRecord`
- `RunState`, `ArtifactIndex`

Schemas may be split across focused packages, but serialized records must use
canonical JSON and version identifiers. Cross-package references use stable
IDs and content hashes instead of embedded mutable objects.

Frozen scientific records use exact JSON types. Boolean strings, booleans in
integer fields, fractional counts, numeric strings, unknown fields, and
schema-version changes fail closed before normalization or hashing. Resume
recomputes active parents and transition exposure from the frozen assignment;
stored treatment context is not authoritative.

## Budget semantics

One proposal opportunity is one preregistered chance to request one child.
Provider retries stay inside that opportunity. Parse failures, invalid code,
candidate-caused failures, and low accuracy consume the opportunity.
Infrastructure retries consume resources but never create a new scientific
opportunity. Repairs use a separate capped budget.
Each unparsable provider response is counted. A format repair stays inside the
same opportunity, receives no evaluation feedback, and is bounded by both a
run-wide ceiling and a per-opportunity ceiling.

The frozen ceilings and reconstructed actuals separately record:

- seed evaluations
- proposal opportunities and provider attempts
- prompt and completion tokens, including unknown-usage flags
- parse failures and unique candidate sources
- candidate training attempts, steps, examples, and MPS seconds
- evaluation cases, repairs, and infrastructure retries
- accepted, rejected, invalid, scientific-failure, and infrastructure-failure
  outcomes

Every primary C0-C3 run evaluates the initial seed identically. The seed
evaluation is excluded from the descendant proposal budget but included in
total resource reporting.

## Scheduling and resume semantics

- Each block contains one C0, C1, C2, and C3 assignment.
- The full randomization table is generated, canonically serialized, and
  hashed before execution.
- Resume never regenerates or changes assignments.
- A run receives a unique directory and stable run ID.
- Opportunity `t + 1` cannot start until opportunity `t` has a terminal event.
- At most one MPS lease may exist across the study.
- Stored provider responses and terminal evaluations are never repeated.
- Only predeclared infrastructure failures may create a linked rerun attempt.
- Scientific failures remain in the intent-to-treat record.

Every durable primary-engine state transition is mirrored into the immutable
event sink. The sink is idempotent across resume, retains provider responses
and candidate source as content-addressed objects, and freezes a search-complete
index. Before a scientific launch, the assignment hash, index digest, and event
chain head must also be retained outside the mutable run directory.

## Containment contract

Arbitrary generated Python is an exploratory format until a real OS boundary
passes adversarial tests. The scientific lane must fail closed when strong
containment is unavailable. The preferred path is an extensible typed tensor
and module graph, with an OS sandbox as defense in depth. The IR must support
new routing, recurrence, attention, state, algebraic, sharing, and composition
mechanisms without any parameter-count preference.

Static source inspection is defense in depth only. Transformer validity needs
typed graph evidence, fresh-build reproducibility, runtime attention and device
evidence, causal metamorphic tests, and intervention evidence that attention
affects outputs.

The typed graph and probes are implemented, but the trusted interpreter and
evaluation-record integration are not. This contract therefore continues to
block scientific arbitrary Python and does not treat the current runtime
heuristics as sufficient evidence.

## Evidence flow after search

1. Freeze a completed run snapshot at a preregistered budget checkpoint.
2. Run Layer B without a return channel to the controller.
3. Canonicalize qualified candidates and cluster equivalent mechanisms.
4. Count each qualifying mechanism cluster at most once per run.
5. Generate treatment-blinded review packets.
6. Freeze mechanism, replication, and analysis plans before outcome access.
7. Perform paired from-scratch ablations, interventions, rescue tests,
   replications, counterfactuals, and scaling studies.
8. Release Layer C only through a one-shot authorization.

The independent statistical unit is a complete assigned run, never a candidate.

## Unresolved PI decisions

Scientific manifests remain invalid until the decision ledger resolves at
least: portfolio size K, transition schedule, primary budget and token policy,
proposal/repair/provider-attempt limits, public Layer A fields, A/B/C case
counts and disjoint generators, promotion thresholds, replication seeds and
success rules, per-candidate compute caps, corpus cutoff and inclusion policy,
clustering rule, smallest effect of interest, target power and alpha, primary
contrasts, multiplicity family, and external-validity plan.

Toy fixtures may use clearly labeled values. They are not scientific defaults.
The decision ledger also requires an explicit PI identity, UTC approval time,
attestation, `approved` status, and separate pilot/main launch switches. MPS,
pilot, protocol, mechanism, replication, and analysis receipts must cross-link
to the same decision, manifest, study, candidate, and artifact hashes. Local
hash syntax is not an external integrity proof; the signed/WORM custodian
verification path remains a launch blocker until implemented and tested.
