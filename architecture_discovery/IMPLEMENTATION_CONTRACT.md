# Architecture Discovery Implementation Contract

Status: engineering contract, amended in August 2026 for the Modal/CUDA
execution condition. This file does not authorize a paid pilot or a scientific
run. Historical version-1 MPS profiles and records remain reproducible, but
they are not evidence of equivalence with CUDA.

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
evaluator-owned, from scratch, and sequential under one accelerator lease.
Modal with NVIDIA CUDA is canonical for new remote runs; MPS is a historical
and local compatibility backend. Parameter count is descriptive metadata and
never a reward or tie-breaker.

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

Active version-2 records serialize run, output, checkpoint, and artifact
locations as logical project-relative or run-relative POSIX paths paired with
content hashes. Executor-absolute host or container paths are runtime state and
must not enter those records. Compatibility readers may accept historical
version-1 MPS absolute paths without rewriting the source record or changing its
hash; active version-2 writers reject absolute executor paths.

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
- candidate training attempts, steps, examples, accelerator kind, and
  accelerator seconds
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
- At most one accelerator lease may exist across the study. Version-2 leases
  bind the remote call ID and credential-free artifact location.
- Stored provider responses and terminal evaluations are never repeated.
- Only predeclared infrastructure failures may create a linked rerun attempt.
- Scientific failures remain in the intent-to-treat record.

Every durable primary-engine state transition is mirrored into the immutable
event sink. The sink is idempotent across resume, retains provider responses
and candidate source as content-addressed objects, and freezes a search-complete
index. Before a scientific launch, the assignment hash, index digest, and event
chain head must also be retained outside the mutable run directory.

## Modal execution boundary

The optional Modal SDK is isolated to `modal_app.py`; core imports and all
provider-free tests remain usable without it. New remote runs use an immutable
Python 3.12 image whose allowlisted source set and dependency lock are hashed.
Before Modal resolves local upload paths, every allowlisted byte is copied into
a manifest-verified read-only temporary snapshot. The minimal
`modal_image_build.py` entrypoint is copied from that snapshot before its
source-disabled build Function is imported. That CPU-only build Function has
two requested CPUs, 8192 MiB requested memory, no GPU, a 600-second user-code
timeout, and a credential-stripped child environment. The image-build API does
not expose repository-controlled CPU or memory tuple limits. Its Debian `git`
installation and frozen Python package
installation require build-time network; `git` is verified before the Python
environment is installed.
GPU-backed engineering-validation functions request two CPUs with a two-core
soft throttle threshold and 8192 MiB memory with an 8192 MiB hard limit, plus
one T4, one container, zero warm containers, zero retries, and a five-minute
function timeout. The provider-free offline-smoke and artifact verification
functions omit the GPU while retaining the same request, soft-CPU, hard-memory,
container, retry, and timeout settings. These Functions are preemptible; Modal
may restart an input after platform preemption independently of `retries=0`,
and GPU Functions cannot opt into nonpreemptible execution. Before any action
probe, subprocess, provider request, training, staged publication, or artifact-
verifier source read, the Function reloads the Volume, constructs its execution
context, exclusively creates the fresh run directory, writes its two provenance
records with create-once no-follow writes, and explicitly commits them. That
commit is the durable pre-action lease: a restarted input reloads it and refuses
to reuse the run ID before it can repeat the action. Every paid action passes
through the provider-free `scripts/launch_modal.py` gate. The launcher pins
Modal 1.5.3, profile `scalingintelligence`, and environment `main`; validates the
action shape, current source-tree/image/cohort identity, globally fresh remote
run IDs, and every action-specific predecessor receipt path and raw SHA-256; and
then starts one synchronous, ephemeral `modal run`. Direct `modal run`,
`modal deploy`, `--detach`, `.spawn()`, and unbounded `.map()` invocations are
never operator surfaces. The launcher binds the boot with
`ModalLocalHostAnchor/1.0`, creates every global
`ModalRemoteRunReservation/1.2`, and only then creates the owned cohort
`ModalActionIntent/1.6`. After `Popen(start_new_session=True)` returns, it
immediately creates `ModalLocalProcessStart/1.1`, binding the intent digest,
host/boot identity, PID, expected PGID and SID, and process birth identity. The
terminal is `ModalActionAttemptReceipt/3.6`; a started terminal is unsealable
unless it binds that marker and proves the process group closed. Cohort intent
and terminal attempt-ID sets must match exactly. Successful or partially
successful aggregate canary actions additionally use
`ProviderCanaryAggregateOutcomeReceipt/1.1`.

A failure before durable cohort-intent ownership writes its version-3.6
terminal under the global `modal_launch_rejections/` namespace and is not a
cohort-roster member. The `action_intent_persistence` and
`action_intent_persistence_uncertain` failure kinds are global-only. A global
rejection that owns published run reservations remains unresolved and requires
recovery before those names can be considered closed.

All launch, action-orphan recovery, cleanup-snapshot, prior-accounting,
lineage, roster, resource-cleanup, migration-bundle, and held-lock journal-scan
work shares `outputs/readiness/.modal_action.lock`. The launcher requires a
resolved workspace-global scan under that lock before approval consumption,
reservation publication, or `Popen`. Readiness publishers scan before
derivation and publication; final lineage binds an immutable global rejection
seal and rejects later journal changes.

Action-orphan recovery is frozen as `ModalActionRecoveryRequest/1.0` plus the
create-only `ModalActionRecoveryIntent/1.0`,
`ModalActionRecoveryHostContainment/1.0`, and
`ModalActionRecoveryResolution/1.0` triplet. The local provider-free CLI
supports only exact pre-`Popen` proof or a may-have-started action contained by
same-boot process-group absence or a strictly later boot and an existing
complete six-read cleanup snapshot. It never captures or queries external
state. Incomplete/tampered stages remain blocking; byte-identical stages are
crash-resumable. Every resolution is quarantined, reserves unresolved Modal
and provider exposure conservatively, requires a named fresh attempt, and is
excluded from final acceptance. That fresh attempt passes its exact 32-hex ID
through the ordinary launcher without a recovery bypass.

Each action also requires a fresh source-bound `ModalPriceBasis/1.0` and one
explicit per-action operator cap before the subprocess may start.
The bound local estimate includes request-rate runtime, a cache-miss build
estimate, and one month of paid-rate bounded storage for every possible new
run. The operator cap must cover that estimate, but its exact scope is
`local_pre_popen_request_rate_and_one_gib_month_storage_estimate_not_platform_billing_cap`:
it is not a Modal billing/runtime ceiling. The outer CLI deadline is likewise a
local client deadline; uncertain remote execution requires cleanup inspection.
Direct local imports expose no App or Function objects.
Deployed endpoints, detached calls, spawned calls, and unbounded maps are
outside this contract.

Provider actions require a second approval chain. The cost-free
`ProviderCanaryApprovalPlan/1.2` binds the source tree, image, cohort, exact
candidate/resume preflight receipt, four frozen configs, and one opportunity per
harness. `ModalProviderCanaryRunSelector/2.0` binds the four accepted downloaded
runs to the same source/image/cohort. The provider request timeout is 180
seconds, controller timeout is 240 seconds, Function timeout is 300 seconds,
and provider retries and retry delay are both zero. A Modal approval never
implies provider approval, and neither approval is reusable for a later action.

Checkpoint resume uses one shared 240-second action deadline. Contract probes,
resumed training, and progression verification have bounded allocations from
that shared clock and fail before starting when the later-stage reserve cannot
be preserved. The enclosing 300-second Function timeout leaves approximately
60 seconds beyond the action deadline for failure/result recording,
manifest-last finalization, Volume commit, and shutdown.

Run artifacts live in a fresh directory under the single named Volume mounted
at `/mnt/discovery`. `ExecutionContext/1.0` records the logical app/function,
exact executing bound Function ID under the pinned runtime, Modal call ID,
image source digest, run ID, and credential-free
Volume URI. The named Volume is operator-managed and must already exist;
runtime lookup uses `create_if_missing=False`. All mutable provider-free,
candidate, checkpoint-resume, and provider-canary outputs are staged in a
private temporary directory outside the mount. Before exclusive publication,
the boundary snapshots stable regular-file bytes and accounts the retained
provenance, staged bytes, and exact reserved final-result bytes against the
256 MiB aggregate cap. Preemption or uncertain process-group closure before
publication leaves only the committed provenance lease, never a background-
committed partial training tree. The prospective exact path roster is also
serialized against the manifest's 2 MiB cap before publication. Downloads are
accepted only after the remote manifest and every
file hash verify locally. The manifest is rejected before transfer when one
file exceeds 64 MiB or the declared aggregate exceeds 256 MiB; those ceilings,
and each file's declared size, are enforced again while streaming; the manifest
has a separate 2 MiB cap and is committed last. Provider
credentials enter only the four controller canary functions through the named
Modal Secret and never enter candidate training child environments. Before a
canary subprocess starts, the Modal boundary requires the exact official OpenAI
API base and frozen `gpt-5.6-sol` model without echoing rejected Secret values;
downloaded canary validation independently checks that provider identity and
the frozen request ceilings recorded in each controller manifest. Modal 1.5.3
Secret metadata proves required-key presence but does not expose the complete
key roster, so the controller child environment is constructed from an exact
allowlist and forwards only the three frozen provider variables.

Final engineering acceptance is cohort-scoped and create-only. It requires
exactly eight distinct accepted primary execution contexts and eight distinct
artifact-verifier contexts, `ModalMigrationCohortRoster/4.0`, one
`ModalCleanupSnapshotCaptureManifest/1.0` covering the App, container,
endpoint, Volume, `/runs`, and complete-hour billing snapshots, and one
`ModalMigrationLineage/1.1`. Snapshot capture holds the shared action lock from
before its first remote read through validation and create-only manifest
publication. Every earlier live cohort is either the selected
final cohort or has a validated
`ModalPriorCohortQuarantineAccounting/1.1`; omitted or invented cohorts fail
closed. The final component receipts are
`ModalCUDAEnvironmentValidationReceipt/2.0`,
`ModalArtifactRoundTripValidationReceipt/3.0`,
`ModalResourceCleanupValidationReceipt/4.0`, and
`ModalMigrationValidationBundleReceipt/4.0`.

Modal accounting distinguishes measured and estimated quantities. The reported
`migration_total_usd` is final-cohort App-attributed compute from the bound
billing snapshot plus every validated prior-quarantined cohort App-compute
subtotal plus the preserved legacy measurement `$0.00643852`. No billing row is
counted twice. Modal exposure accounting separately reports measured App
billing, a full-cap reserve for unresolved starts, a remaining-cap reserve for
failed or zero-measured starts while billing may lag, their conservative sum,
and any measured overage above the local authorization cap. Those reserves and
overage checks do not change `migration_total_usd`; the local authorization cap
is not a Modal platform hard bound. Retained Volume storage is a separate
conservative monthly estimate based on the bound price basis and artifact byte
caps; it is not a measured billed charge and is not added to
`migration_total_usd`. Provider usage is likewise a separately labeled
approval-bound estimate, not a provider invoice.

Provider-produced canary artifacts additionally pass the provider-credential
scan before publication. Failed or unsafe staged provider files are not
published to the Volume. A failed child with one terminal, context-bound ledger
record proves exactly one attempt. A missing or initialized-but-empty ledger
does not prove zero because the child can die while transport is in flight; it
publishes `ProviderRequestStartUncertainEvidence/1.0`, reserves one full
approved request, and blocks aggregate acceptance. Multiple or context-
mismatched records fail closed as impossible for the one-opportunity,
zero-retry canary contract.

The vendored OpenEvolve commit and every reviewed patch input are named in
`experiment_manifest.yaml`; configuration validation rehashes the retry patch,
the process-pool credential-isolation implementation, and its adversarial test.
After initializing a fresh clone, the operator validates the integrated reviewed
state with `.venv/bin/python scripts/openevolve_patch_bundle.py`. The paid
launcher must never repair it implicitly and must fail closed before invoking
Modal unless the frozen commit, patch inputs, and integrated-file hashes match.
The Modal image source manifest independently binds the patched runtime bytes.

`full_train_cuda_v2` and `smoke_train_cuda_v2` are distinct deterministic
float32 profiles. They require CUDA, deterministic algorithms, deterministic
cuDNN, disabled cuDNN benchmarking, disabled TF32, and a pinned
`CUBLAS_WORKSPACE_CONFIG`; CPU fallback is forbidden. Version-1 readers retain
the exact historical `mps_seconds`, MPS telemetry, profile hashes, and receipts.

The current five-minute Modal functions are engineering-validation surfaces and
expose only `smoke_train_cuda_v2` for candidate training. They do not expose or
authorize `full_train_cuda_v2`. A future full-profile function requires resolved
scientific gates, its own explicitly reviewed resource and timeout contract,
and separate operator approval; it must not silently inherit the engineering
timeout.

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

The typed graph, trusted interpreter, and Layer A runtime probes are integrated
for the native harnesses. The primary C0-C3 scientific adapter and arbitrary
Python OS boundary remain unresolved. A Modal container alone is not proof of
strong arbitrary-Python containment.

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
attestation, `approved` status, and separate pilot/main launch switches.
Accelerator, pilot, protocol, mechanism, replication, and analysis receipts
must cross-link to the same decision, manifest, study, candidate, and artifact
hashes. Local
hash syntax is not an external integrity proof; the signed/WORM custodian
verification path remains a launch blocker until implemented and tested.
