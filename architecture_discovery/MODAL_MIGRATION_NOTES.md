# Modal CUDA migration notes

Status: the migration implementation is present, but the current source tree
has no accepted live Modal cohort. All dated hashes, test counts, Apps, runs,
costs, and approvals below are historical checkpoints only. The two recorded
live environment attempts predate the current source-bound contracts and remain
superseded, non-authorizing evidence. Derive fresh source and image identities,
rerun provider-free validation, and obtain a new action-specific approval before
any live action. This document is an engineering log, not scientific
authorization or evidence of completed Modal validation.

## Static manifest and external live-evidence boundary

`experiment_manifest.yaml` remains a static input to both the Modal image
source digest and the local engineering source-tree digest. Its
`remote_execution.image_source_sha256` therefore stays null/pending: embedding
the digest computed from that file back into the file would create an
unresolvable self-reference and invalidate the cohort. The manifest instead
freezes `external_create_only_receipts` as the live evidence authority and the
exact four readiness receipt paths/schema versions:

- `ModalCUDAEnvironmentValidationReceipt/2.0`;
- `ModalArtifactRoundTripValidationReceipt/3.0`;
- `ModalResourceCleanupValidationReceipt/4.0`; and
- `ModalMigrationValidationBundleReceipt/4.0`.

Dynamic run/app/function/call/image IDs, timestamps, measured costs, computed
image digests, approval plans, selectors, and receipts live only under the
excluded `outputs/readiness` tree, `readiness_evidence.yaml`, README, and this
notes file. Recording those facts cannot change the image/source identity that
produced them. Tests bind `experiment_manifest.yaml` into both hashes and prove
that edits to these three root evidence/docs files affect neither.

Provider request timeouts are now 180 seconds inside the 240-second controller
subprocess and 300-second Function ceilings, leaving 60 seconds for a graceful
provider-timeout path to flush its terminal sanitized attempt record. Aggregate
canaries attempt all four harnesses once in frozen order, continue after an
ordinary per-harness exception, emit exact outcomes, and exit nonzero if any
failed; they never catch process-control `BaseException`s. Mixed aggregate and
recovery downloads are selected by a create-only
`ModalProviderCanaryRunSelector/2.0`, not copied or symlinked into a fabricated
four-directory parent.

## Current source-bound completion contract

Use Modal SDK `1.5.3`, profile `scalingintelligence`, and environment `main`.
Every remote action must go through `scripts/launch_modal.py`; direct
`modal run`, `modal deploy`, detached calls, spawned calls, and unbounded maps
are outside the operator contract. The launcher binds a freshly derived source
tree SHA-256, approved image-source SHA-256, final cohort ID, globally fresh run
ID, exact outer timeout, fresh price basis, per-action cap, and every required
predecessor receipt path plus raw SHA-256. The durable owned-attempt order is
`ModalLocalHostAnchor/1.0`, global `ModalRemoteRunReservation/1.2` record(s),
cohort `ModalActionIntent/1.6`, `Popen(start_new_session=True)`, immediate
`ModalLocalProcessStart/1.1`, and terminal
`ModalActionAttemptReceipt/3.6`. The process marker binds the intent digest,
host/boot identity, PID, expected PGID and SID, and process birth identity. A
started terminal cannot seal without the marker and proven process-group
closure. Cohort intent and terminal attempt-ID sets must match exactly.
Aggregate canary actions returning 0 or 2 additionally bind
`ProviderCanaryAggregateOutcomeReceipt/1.1`.

Pre-ownership failures publish their version-3.6 terminals in the global
`modal_launch_rejections/` namespace, not the cohort roster. The
`action_intent_persistence` and `action_intent_persistence_uncertain` kinds are
global-only; a rejection with published reservations remains blocking pending
recovery. All launcher, action-orphan recovery, snapshot, prior-accounting, lineage, roster,
resource-cleanup, bundle, and held-lock scan work uses
`outputs/readiness/.modal_action.lock`.

`modal_action_journal.py` provides the workspace-global scanner and resolved
gate. The launcher invokes it under the shared lock before any ownership write
or process start, and readiness/final-sealer callers use pre/post scans plus an
immutable global rejection seal. Action-orphan recovery is operational through
the provider-free `recover_modal_action_journal.py` CLI with exact v1 request,
intent, host-containment, and resolution schemas. Only scanner-proven
pre-`Popen` or externally snapshotted contained-start branches are permitted;
the CLI makes no external request. It resumes byte-identical crash stages,
rejects changed evidence, conservatively reserves unresolved exposure, and
keeps every recovered attempt quarantined. A fresh candidate uses the
request-named 32-hex attempt ID through the ordinary launcher gate, with fresh
run/cohort IDs and a new approval. Its manifest status is
`operational_exact_v1_cli_scanner_validated`.

The provider approval chain is `ProviderCanaryApprovalPlan/1.2`, bound to the
current source, image, cohort, four configs, and exact candidate/resume
preflight receipt, followed by `ModalProviderCanaryRunSelector/2.0`. Provider
request, controller, and Function timeouts are respectively 180, 240, and 300
seconds; provider retries and retry delay are zero. Neither a historical Modal
approval nor a provider-free approval authorizes a provider request.

Final acceptance requires exactly eight accepted primary execution contexts and
eight accepted artifact-verifier contexts under one source/image/cohort root.
Capture the App, container, endpoint, Volume, `/runs`, and complete-hour billing
snapshots with `ModalCleanupSnapshotCaptureManifest/1.0`; do not hand-author
those files. Snapshot capture holds the shared lock across all six remote reads,
validation, and create-only manifest publication. Seal every real earlier
quarantined cohort with `ModalPriorCohortQuarantineAccounting/1.1`, seal the
cross-cohort `ModalMigrationLineage/1.1`, and then seal the complete
`ModalMigrationCohortRoster/4.0` before creating the version-4 cleanup and
bundle receipts. The cleanup snapshot, lineage, prior-accounting, roster,
component receipts, and bundle are create-only and fail closed on unbound or
invented identities.

`migration_total_usd` is the final cohort's App-attributed compute plus every
validated prior-quarantined cohort App-compute subtotal plus the preserved
legacy measurement `$0.00643852`, with no billing row counted twice. Retained
Volume storage is a separately labeled conservative monthly estimate; provider
usage is a separately labeled approval-bound estimate. Modal compute exposure
separately records measured App billing, full-cap unresolved-start reserves,
remaining-cap failure/billing-lag reserves, their conservative sum, and any
measured overage above the local authorization cap. The local cap is not a
Modal platform hard bound. Neither reserves nor either separate estimate is
added to `migration_total_usd`.

All four live readiness gates remain false with null paths, hashes, and selected
cohort identity until their current create-only receipts exist and independently
revalidate. The independent scientific, governance, custody, and launch gates
also remain pending. No historical checkpoint below may be copied into those
fields.

## Historical migration log

Everything from this point onward is a point-in-time record retained for audit
continuity. Commands and identities in this log are not current operator input.

### Pre-change baseline

Recorded on 2026-08-08 before migration edits:

- Root pytest: 8 tests passed.
- Architecture pytest: all 421 collected tests passed. The suite is
  provider-free but includes bounded CPU test training.
- Root-owned Ruff scope (`src tests`): passed.
- Documented root `make check`: failed before migration because Ruff traversed
  the nested architecture project and vendored sources, producing 5,045
  findings. The Makefile scope must be corrected and the command rerun.
- Configuration validation and compileall: passed.
- Synthetic C0-C3 study, no-search control, reconstruction, reporting, and the
  four static controller-surface checks: passed with zero provider calls and
  zero candidate-training runs.
- Scientific readiness audit: expected exit 2, `pilot_ready=false`,
  `main_study_ready=false`, nine of thirty gates passed. MPS was unavailable in
  the Codex process. PI, protocol, custody, corpus, Layer B, pilot, and other
  scientific blockers remained open.

Baseline outputs were written under
`/private/tmp/rl4rl-modal-baseline.4LnkN3`. Existing ignored MPS evidence and
smoke outputs were not modified.

### MPS assumptions generalized at that checkpoint

- [x] Typed CPU/MPS/CUDA selection, strict fallback rules, synchronization,
  cleanup, memory telemetry, hardware fingerprinting, CUDA determinism, and
  CPU/MPS/CUDA RNG checkpointing.
- [x] New CUDA v2 full and smoke profiles; retain exact v1 MPS profile hashes.
- [x] Generic v2 training results and relative artifact paths; retain v1 MPS
  readers and evidence recorder.
- [x] CUDA-capable child worker environment with one-device visibility and no
  provider, Modal, Tinker, Hugging Face, GitHub, or unrelated secrets.
- [x] CUDA options and v2 profiles in direct training, retraining, Greedy,
  Semantic Autoresearch, and both OpenEvolve entrypoints.
- [x] Accelerator-neutral budget, evaluation, event, lease, scheduler,
  reconstruction, and reporting schemas with v1 `mps_seconds` readers.
- [x] Accelerator capability/containment audit and CUDA/Modal infrastructure
  failure taxonomy without treating a Modal container as arbitrary-Python
  containment evidence.
- [x] Modal-only adapter module, allowlisted immutable Python 3.12 image source,
  one T4/one-container/five-minute functions, synchronous calls, one Volume,
  named Secret only on provider canaries, and safe artifact verification.
- [x] Versioned accelerator validation receipt binding code, lock, image source,
  GPU, profile, candidate, seeds, outputs, runtime, and cleanup.
- [x] Active manifest, readiness, validators, reports, environment example,
  implementation contract, and operator documentation.

### Historical compatibility

- Preserve `full_train_v1` and `smoke_train_v1` as MPS profiles with hashes
  `046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e`
  and `1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61`.
- Preserve `record_mps_validation.py`, MPS failure enum values, dated MPS
  reports, and raw/hash-linked local MPS artifacts without rewriting them.
- Readers must accept v1 MPS budgets, ledgers, run states, training summaries,
  events, receipts, and absolute paths. New writers use v2 accelerator-neutral
  fields and portable relative paths.
- CUDA is a new execution condition. Do not reuse MPS hashes or claim
  cross-device scientific equivalence.

### Historical paid validation attempts

The first provider-free Modal invocation was explicitly approved with a $0.25
operator authorization amount and no application retry. That amount was not a
Modal-enforced billing ceiling. It built image source SHA-256
`46c2a95415f3247e17deddc098dce0aa0986cd6e540fdddecd54577d55c4c24b`,
started the single T4 `cuda_environment` function, and failed before creating a
run directory because Modal exposed `/mnt/discovery` as a symlink while the
local boundary rejected every symlinked mount root. App
`ap-hRTRnEPeSwUGDaaTnIUW0z` then stopped with zero tasks. Post-run inventories
showed zero containers and zero endpoints, and the newly created
`rl4rl-architecture-artifacts` Volume was empty. No provider or Tinker call was
made. Modal's completed 16:00-17:00 CST billing interval attributes
$0.00164157 CPU, $0.00100956 memory, and $0 GPU to that App, for $0.00265113
metered compute total. The invocation will not be retried under that approval.

After the compatibility fix passes local adversarial tests, report the new
source hash, exact resource request, source-bound request-rate/storage estimate,
preemption and billing uncertainty, artifacts, and cleanup procedure, then
obtain a fresh explicit approval. Provider-backed canaries still require their
own later approval.

#### First-attempt remediation

The mount compatibility fix now requires an explicit opt-in at the sole Modal
call site, resolves the trusted mount alias once, and creates `runs/` plus the
fresh run ID through no-follow directory descriptors. Default callers and every
descendant symlink remain fail-closed. The image recipe was also consolidated
from 31 source COPY layers to two manifest-exact directory layers: one frozen
dependency subset before the bounded builder and one complete verified source
snapshot afterward. This keeps dependency caching while making backend image
work materially smaller and accurately disclosable.

After these changes, 691 architecture tests and the 8 root tests passed, Ruff
and `git diff --check` passed, and the cost-free Modal plan reported 215 files,
2,402,737 bytes across two COPY layers with a 4,805,474-byte aggregate source
upper bound, dependency lock
SHA-256 `91efea8a7afd18eb94a22b17d70a08956dd8d8d5ab253c80baef0175c5e9e01d`,
image-source SHA-256
`c39f89ec973b80d43f104420b65c369390a97b77ec27bc1f2f3d6d1b6f9747fc`,
and zero remote calls.

#### Successful CUDA environment attempt

With a fresh explicit $0.25/no-retry approval, run
`modal-cuda-env-20260809-02` completed on a Tesla T4 using image source
`c39f89ec973b80d43f104420b65c369390a97b77ec27bc1f2f3d6d1b6f9747fc`.
Its artifact manifest SHA-256 is
`5100129b79030dbef067bce8d5aca9d56e2d6a9aa173a7d76cb3977e86c5d909`.
App `ap-BbPU2boNFr7Y6CVFvUdI7T` stopped with zero tasks, no migration endpoint or
container remained active, and the Volume run directory contained exactly the
five declared files. The unrelated active shared-workspace container was not
touched.

A separately approved CPU-only verifier/download then ran once as App
`ap-lPfOUHrfbnIV3LIo3FNtEr`. It remotely reverified the manifest, downloaded the
five-file run without overwriting local state, and saved the immutable verifier
capture. The App stopped with zero tasks and no migration container or endpoint
remained active. The cost-free local CUDA recorder then correctly failed before
receipt creation because its exact schema omitted the runtime's intentional
`git_version` field. No partial or canonical CUDA receipt was created.

The completed 17:00-18:00 CST billing interval attributes `$0.00369927` to the
successful environment App (`$0.00146304` CPU, `$0.00092507` memory, and
`$0.00131116` T4) and `$0.00008812` to the verifier App (`$0.00005256` CPU and
`$0.00003556` memory). Together with the first attempt's `$0.00265113`, measured
migration compute through this superseded cohort is `$0.00643852`.

The recorder now requires and strictly validates `git_version`; the real
downloaded Tesla T4 artifact passes a no-write replay. A cross-stage audit found
no analogous mismatch in offline smoke, candidate smoke, checkpoint resume, the
four canaries, verification, or download. Because the recorder is baked into
the image, the fix changed the image digest. The `c39f89ec...` run cannot be
mixed with later executions in the strict eight-context cohort and remains
read-only superseded evidence.

### Local validation checkpoint

Recorded on 2026-08-09 after the corrected recorder and then-final local review:

- Root `make check`: Ruff passed and 8 tests passed.
- Architecture Discovery: 702 tests passed; the immutable
  local receipt binds source tree
  `01040bee7092220ab1617dea8b5001e269636f8bf7c352117e856e6874bc5c6d`.
- Fresh provider-free C0-C3 plus no-search smoke: four completed runs, zero
  provider calls, and zero Torch training runs.
- Configuration validation, migration-specific Ruff, static four-controller
  validation, synthetic reconstruction/reporting, locked offline dependency
  resolution, and the Modal plan all passed.
- Frozen dependency lock SHA-256:
  `91efea8a7afd18eb94a22b17d70a08956dd8d8d5ab253c80baef0175c5e9e01d`.
- Frozen image-source SHA-256 after readiness evidence was recorded:
  `646c3b660b9bea24cf02597d1e6b855edb8b657db6084207aceca43d5307dcb6`.
  The manifest contains 215 files totaling 2,403,537 bytes; its two source-copy
  layers have a 4,807,074-byte aggregate input upper bound.
- The fail-closed readiness audit reports local unit and offline-smoke levels as
  passed. Its four live Modal receipt gates and the independent scientific,
  PI, custody, protocol, and authorization gates remain false.

#### Final provider-boundary freeze checkpoint

Before starting a new same-image live cohort, the Modal canary boundary was
hardened to require the literal official endpoint
`https://api.openai.com/v1` and model `gpt-5.6-sol` before any controller
subprocess starts. The downloaded validator now requires the exact 15-field
provider/request contract, rejects missing or unknown fields, and rejects
credential-bearing metadata. A cost-free parity regression loads all four real
controller YAML files and proves that their normalized emitter settings equal
that validator contract. Provider-free functions remain isolated from all
three provider variables.

After that hardening, the architecture suite passed 720 tests. Fresh
create-only unit and offline-smoke receipts both bind source tree SHA-256
`fb7efc1d481c092b477f23ad0af981dc526f8b74c766f35ad04139701bd60e63`;
the offline smoke again completed C0-C3 plus no-search with zero provider calls
and zero Torch training runs. The then-final cost-free Modal plan contained 215 files
totaling 2,406,404 bytes, with a 4,812,808-byte two-copy-layer upper bound and
image-source SHA-256
`5e799e187e673b696af8706ea5583e426a602e3c2e326c991022ff22f8bcd5c3`.
This supersedes the pre-hardening `646c3b66...` image before any run used it.

At that checkpoint, the four first-opportunity provider message contents were 12,375,
13,201, 20,517, and 21,414 UTF-8 bytes. With conservative input approval
bounds of 16,384, 16,384, 24,576, and 24,576 tokens and explicit
`retries=0`, a four-canary approval must disclose up to four provider requests,
81,920 input tokens, and 65,536 requested completion tokens. Modal retries
remain zero. No provider canary is authorized by the provider-free approvals.
The provider-free `scripts/provider_canary_plan.py` command now derives those
bounds from the actual native and vendored OpenEvolve first-prompt
constructors. The current command emits `ProviderCanaryApprovalPlan/1.2`, binds
the fresh source/image/cohort and candidate/resume preflight receipt in addition
to the config hashes, and initializes no provider client. Prompt byte counts
must be derived again; the dated native counts were exact across bounded
seed-metric endpoints, while the OpenEvolve counts were explicit upper bounds
because its actual outcome phrase varies with the live seed metrics.

Before publication, each paid canary now validates its private off-Volume
staging tree against an exact harness roster, exact controller manifest and
terminal-result schemas, exact successful native lineage or OpenEvolve trace
and checkpoint-metadata schemas, two distinct trained Architecture IRs, and
the one-attempt provider ledger. Download validation independently rejects
re-manifested outer or controller extras. If a post-request validation fails,
only the sanitized provider-attempt ledger is retained; invalid prompts,
responses, and controller artifacts remain unpublished.

#### Final early-path receipt hardening

A producer-to-consumer audit then found four cost-free fail-closed gaps before
the replacement live cohort began. The current CUDA environment receipt uses
version 2.0, exact-validates and SHA-binds `remote_action_result.json`, and
cross-checks the frozen Python version plus CUDA report scalars against the
strict accelerator fingerprint. Artifact round-trip evidence now accepts only
a `candidate_smoke` execution context. The downloaded offline validator now
rejects Boolean or non-integer subprocess return codes. Resume, checkpoint
manifest normalization, final-manifest, eight-context bundle, and remote/local
verification schemas were also audited end to end without another mismatch.

After these fixes, 737 architecture tests passed. Fresh create-only unit and
offline-smoke receipts bind source tree SHA-256
`052cb8ea04e2be54c9d042a96184b0acf48b7e7b6b7b660f7e0b57e1c0f5aa86`;
the offline receipt again proves four completed C0-C3 runs, zero provider
calls, and zero Torch training runs. The then-final cost-free image plan contained
215 files totaling 2,409,039 bytes, with a 4,818,078-byte two-copy-layer upper
bound and image-source SHA-256
`7087ef1716550555112b27227776a56ba4db7e855e26e49fd7e0a5acf4f12114`.
The intermediate `5e799e18...` and `b8237975...` hashes were never run and are
superseded. No paid Modal or provider call was made during this audit.
