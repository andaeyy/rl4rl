# Architecture Discovery on AdderBoard

This repository is an offline-tested research-infrastructure system for studying
novel transformer architecture discovery. AdderBoard is used only as a
correctness and accuracy environment. Parameter count is descriptive metadata,
not an optimization target: it is never a reward, selection criterion,
tie-breaker, or stopping rule. Neutral pre-allocation compute and memory
ceilings still reject candidates that are unsafe to execute on the host.

## Teammate quickstart for the shared Modal workspace

Clone the repository with its pinned submodules, then run the provider-free
bootstrap:

```bash
git clone --recurse-submodules https://github.com/andaeyy/rl4rl.git
cd rl4rl/architecture_discovery
sh scripts/bootstrap_teammate.sh
```

The bootstrap installs the locked Python 3.12 Modal environment, verifies the
OpenEvolve commit containing the two reviewed changes, and runs the local
environment and configuration checks. It never asks for, reads, or stores a
credential.

After it succeeds, use Modal's interactive credential prompt. Enter the shared
Modal token ID and token secret only when that prompt requests them:

```bash
.venv/bin/modal token set --profile scalingintelligence --activate
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal token info
```

Never put Modal credentials in this repository, `.env`, a command-line flag,
or a GitHub issue. `DISCOVERY_API_KEY` is a separate model-provider credential;
provider-free CUDA and artifact-verification actions do not need it. Because the
Modal App, Volume, and billing account are shared, only one teammate may execute
a paid launcher command at a time. Every paid action still requires its own
explicit cost approval.

## Current status

The four native controller harnesses have an explicitly non-scientific,
IR-only engineering-pilot path. New remote engineering work targets a pinned
Python 3.12 Modal image and NVIDIA CUDA through `smoke_train_cuda_v2`; all four
scientific configurations point at the distinct `full_train_cuda_v2` profile.
The Modal boundary is synchronous, single-container, and opt-in. Local tests do
not require the Modal SDK, and no live Modal run is authorized by this document.
The current five-minute Modal entrypoints are engineering validation only; they
do not expose `full_train_cuda_v2`. A future full-profile entrypoint requires
resolved scientific gates, a separate resource/timeout contract, and separate
operator approval.

### Convenient autoresearch and OpenEvolve runs

Use the top-level `evolve` command for configurable engineering runs. Planning
is cost-free and is the default:

```bash
./evolve openevolve -n 60
./evolve autoresearch -n 12
./evolve semantic-openevolve -n 40
./evolve semantic-autoresearch -n 8
```

The command accepts 1–345 iterations. The upper bound is the largest single
run whose dynamically calculated Function timeout fits Modal's 24-hour limit.
It fixes seed 1, `smoke_train_cuda_v2`, `smoke_eval_v1`, 24 cases, one T4,
zero retries, one provider request per iteration, and non-scientific status.
The printed plan includes the exact request and timeout ceilings and starts no
paid work.

To launch, repeat the command with explicit Modal and provider dollar caps:

```bash
./evolve openevolve -n 60 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --provider-cost-cap-usd "$APPROVED_PROVIDER_ACTION_CAP_USD" \
  --execute
```

To approve the exact source-bound estimates automatically instead of entering
the two values manually:

```bash
./evolve openevolve -n 60 --execute --accept-estimated-cost
```

On macOS, use `caffeinate` for an attached paid run so the machine does not
sleep while the local Modal client is supervising the remote Function:

```bash
LANG=en_US.UTF-8 LC_ALL=C \
  caffeinate -dimsu ./evolve openevolve -n 60 \
  --execute --accept-estimated-cost
```

Keep that Terminal open and maintain a stable network connection until the
command finishes. `caffeinate` prevents macOS sleep; it cannot prevent a Wi-Fi,
VPN, or DNS interruption. This command starts a fresh run and does not resume a
previous OpenEvolve checkpoint.

This is still a local approval bound, not a platform-enforced billing limit.
The command prints both approved estimates before delegating to the launcher.

`evolve` automatically uses `.venv`, selects the sole ready cohort for the
current frozen source, validates the current preflight and price bases, creates
or reuses the immutable source-bound approval plan, and delegates to the paid
launcher. Pass `--cohort-id` when more than one current cohort is ready and
`--run-id` when a specific run identity is desired. Direct `modal run` remains
unsupported because it bypasses these approval and journal checks.

The command works on teammate Macs after each teammate completes the bootstrap
above, authenticates a Modal profile that can use the `scalingintelligence`
workspace and `main` environment, and creates the current local engineering
freeze and live-cohort readiness evidence described below. Those receipts are
bound to the source, executable environment, and local machine; do not copy
another teammate's `outputs/readiness` directory. The shared provider Secret
must also exist in the authorized Modal workspace before a provider-backed run.

### Legacy bounded 60-iteration OpenEvolve run

The dedicated `openevolve-generic-60` compatibility action runs exactly 60
generic OpenEvolve iterations at seed 1 on one T4 container. It uses
`smoke_train_cuda_v2`, `smoke_eval_v1`, 24 evaluation cases, zero provider and
Modal retries, and at most 60 provider requests. It is an engineering smoke run,
not scientific evidence or a final ranking. It does not resume after an
interruption; a platform preemption can still restart the one logical Modal
call, so the committed run lease prevents silently repeating the run ID.

The action has a separate 15,300-second Function timeout, a 15,000-second
controller timeout, and a 16,200-second local launcher timeout. Every provider
request is rejected before transport if its canonical JSON encoding exceeds
1,048,576 UTF-8 bytes. Each request retains the existing 16,384 completion-token
limit. These are conservative approval ceilings, not expected usage.

After creating the current source/image freeze and accepted candidate-resume
preflight receipt, create the source-bound, create-only provider plan without
reading the provider secret:

```bash
.venv/bin/python scripts/openevolve_60_plan.py \
  --source-tree-sha256 "$SOURCE_TREE_SHA256" \
  --cohort-id "$COHORT_ID" \
  --candidate-resume-preflight-receipt-path "$PREFLIGHT_PATH" \
  --candidate-resume-preflight-receipt-sha256 "$PREFLIGHT_SHA256" \
  --output outputs/readiness/openevolve_60_provider_approval_plan.json
```

Review that plan and current Modal/provider price bases before obtaining explicit
cost approval. The only accepted launch shape is:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action openevolve-generic-60 \
  --run-id "$RUN_ID" \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 16200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --provider-approved \
  --provider-cost-cap-usd "$APPROVED_PROVIDER_ACTION_CAP_USD" \
  --provider-approval-plan-path outputs/readiness/openevolve_60_provider_approval_plan.json \
  --approval-plan-sha256 "$OPENEVOLVE_60_PLAN_SHA256" \
  --provider-price-basis-path "$PROVIDER_PRICE_BASIS_PATH" \
  --provider-price-basis-sha256 "$PROVIDER_PRICE_BASIS_SHA256" \
  --candidate-resume-preflight-receipt-path "$PREFLIGHT_PATH" \
  --candidate-resume-preflight-receipt-sha256 "$PREFLIGHT_SHA256" \
  --approved
```

Do not replace this with a direct `modal run`. The launcher validates the frozen
source, image, predecessor receipt, request ceilings, prices, cost caps, and
global action journal before it exposes the provider secret. Successful output
is privately validated for all 60 terminal iterations before publication to the
artifact Volume.

### What to run next: bounded exploratory Modal pilot

The repository now includes a separate `exploratory_non_scientific` lane. It
uses one block with one opportunity per C0-C3, a single provider attempt, zero
provider/Modal retries, the CUDA-only `exploratory_train_cuda_v2` profile, and
explicit Modal/provider cost ceilings. It is for plumbing, hypothesis
generation, and learning how the search behaves; it is not a ranking run and
does not change `scientific_decisions.yaml` or unlock `study_scientific_run.py`.

After the teammate bootstrap above, run the cost-free preflight (it reads no
provider secret and makes no network call):

```bash
.venv/bin/python exploratory_pilot.py preflight \
  --run-id exploratory-team-20260815-01 --print-approval
```

The first live action must still be launched through `scripts/launch_modal.py`
and requires a fresh local freeze, current Modal price basis, provider price
basis, candidate-resume preflight receipt, and a checked-in approval plan. The
approval text is intentionally explicit:

```text
I approve exactly one provider-backed Modal exploratory_c0c3_pilot for run/cohort exploratory-team-20260815-01, with a $0.25 Modal cap and $0.25 provider cap, zero retries, provider access, and stop after first success or failure.
```

Once those receipt paths and hashes have been produced by the existing
readiness workflow, the bounded launch shape is:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action exploratory_c0c3_pilot \
  --run-id exploratory-team-20260815-01 \
  --cohort-id exploratory-team-20260815-01 \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --provider-approved \
  --provider-cost-cap-usd 0.25 \
  --provider-approval-plan-path "$EXPLORATORY_PROVIDER_PLAN" \
  --approval-plan-sha256 "$EXPLORATORY_PROVIDER_PLAN_SHA256" \
  --provider-price-basis-path "$PROVIDER_PRICE_BASIS_PATH" \
  --provider-price-basis-sha256 "$PROVIDER_PRICE_BASIS_SHA256" \
  --candidate-resume-preflight-receipt-path "$PREFLIGHT_PATH" \
  --candidate-resume-preflight-receipt-sha256 "$PREFLIGHT_SHA256" \
  --approved
```

The launcher refuses missing approval, stale price bases, nonzero retries, a
stale source/image freeze, or a missing predecessor receipt before starting
Modal. After success, use the existing separately approved `download` action
to retrieve the Volume run, then run `exploratory_pilot.py verify` against the
downloaded `exploratory-<RUN_ID>` directory. Finish with the existing
read-only inventory/cleanup command from the Modal readiness runbook; cleanup
is never implicit.

Create the provider approval plan after the current source/image/cohort values
are known, without reading the provider Secret:

```bash
.venv/bin/python exploratory_pilot.py approval-plan \
  --source-tree-sha256 SOURCE_TREE_SHA256 \
  --image-source-sha256 IMAGE_SOURCE_SHA256 \
  --cohort-id exploratory-team-20260815-01 \
  --output outputs/readiness/exploratory_provider_approval_plan.json
```

For a completely provider-free end-to-end check, use a fresh output directory:

```bash
.venv/bin/python exploratory_pilot.py fake-run \
  --output-dir /private/tmp/rl4rl-exploratory-fake \
  --run-id exploratory-local-fake-1
.venv/bin/python exploratory_pilot.py verify \
  --run-directory /private/tmp/rl4rl-exploratory-fake/exploratory-exploratory-local-fake-1 \
  --run-id exploratory-exploratory-local-fake-1
```

The verifier checks the content-addressed artifact manifest and rejects any
tampering. It expects the required summary, randomization/assignment plan,
sanitized provider-attempt ledger, candidate-source index, training/evaluation
summary, terminal receipt, and cost ceiling records. After an approved live
run, use the existing `download` action and then run the same local verifier;
perform cleanup/inventory only with a separate explicit approval.

Historical MPS evidence is retained unchanged. A one-opportunity Greedy canary
and the paid 10-by-4 mechanics pilot completed sequentially on MPS on 2026-08-08
UTC. Public smoke accuracy was 0.0 throughout. Those artifacts remain useful
for reproduction and schema-compatibility tests, but they do not validate CUDA
or rank architectures.

A scientific pilot and the main C0-C3 study remain **blocked**. The primary
causal engine still uses its legacy Python proposal adapter, and the repository
still lacks resolved principal-investigator decisions, real
`full_train_cuda_v2` accelerator evidence, a populated frozen literature corpus and reviewer roster,
production Layer B orchestration, a scheduled scientific no-search path,
revision-bound externally attested validation receipts, a cryptographically
authenticated external ledger anchor, completed pilot evidence, and explicit
PI launch authorization.

`scripts/study_scientific_run.py` audits those gates before reading provider
credentials or constructing an API client. It cannot presently start a paid
run.

## Primary causal design

All primary conditions execute through the same `CommonStudyEngine`:

| Condition | Parent policy | Proposal policy |
| --- | --- | --- |
| C0 | one parent | ordinary proposal |
| C1 | one parent | fixed scheduled transition |
| C2 | K-parent portfolio | ordinary proposal |
| C3 | K-parent portfolio | fixed scheduled transition |

Only those two treatment fields may differ. Every block contains C0–C3 once,
with deterministic blocked randomization and a frozen order. Runs are isolated,
and a study-wide lease permits only one accelerator run at a time. The independent
statistical unit is one complete assigned run, not one candidate.

The no-search GPT-5.6 baseline is a separate control. Its provider-visible input
is constant across opportunities and contains no parents, history, scores,
transition state, or repair feedback. The four named native Autoresearch and
OpenEvolve harnesses remain secondary system replications, not the primary
causal comparison.

## Evaluation firewall

- Layer A is public, online search feedback. Controllers receive only a typed
  allowlisted view.
- Layer B is sealed post-run qualification over a frozen run snapshot. It cannot
  affect proposals, retention, repair, or stopping.
- Layer C is disabled by default and requires an explicit one-shot release
  authorization after Layer B.

Scientific A/B/C profiles have no implicit case count and reject fewer than
10,000 cases. The actual scientific counts and disjoint B/C sources remain PI
decisions. Legacy official/shadow regression evaluation is isolated under
`private_eval/` and is not part of online search fitness.

## Candidate training

GPT-5.6 Sol proposes a complete declarative Architecture IR document; it does
not train the arithmetic model or supply executable Python. Trusted evaluator
code owns:

- schema, primitive, topology, shape, and resource validation;
- deterministic construction and fresh seeded initialization;
- deterministic public training data and order;
- optimizer, schedule, steps, examples, and wall-time ceiling;
- public-development-only checkpoint selection;
- generic autoregressive decoding and Layer A evaluation;
- runtime transformer-validity probes; and
- checkpoint/artifact/profile/task/seed/trusted-code identity verification.

The vendor `best.pt` is used only in an explicitly named pretrained regression
path. New candidates never load it. Best-model and resume checkpoints use
restricted `weights_only=True` loading; resume identity is bound to the exact
candidate, profile, task, and seed bundle. External checkpoint and event-chain
anchoring is still an open scientific gate.

`full_train_cuda_v2` is a new profile: CUDA, float32, deterministic algorithms,
deterministic cuDNN, benchmarking and TF32 disabled, a pinned
`CUBLAS_WORKSPACE_CONFIG`, 30,000
optimizer steps, batch 512, AdamW at 0.001 with betas 0.9/0.98 and weight decay
0.1, 300 warmup steps, cosine decay, gradient clipping at 1.0, 2,000 public
development examples every 1,000 steps, and a 1,800-second safety ceiling. It
has no CPU fallback. It does not reuse or claim equivalence with the historical
`full_train_v1` MPS profile or hash.

`smoke_train_cuda_v2` is a ten-step engineering check only. It is not valid for
ranking architectures or making scientific claims.

## Containment and transformer validity

Provider-backed candidates in the four native harnesses are JSON data and are
never imported or executed as Python. The trusted interpreter has a fixed
primitive vocabulary, strict shape/topology/resource limits, and runtime
causality, sequence-dependence, parameter-influence, and attention-intervention
probes. Runtime evidence is retained as a hash-linked Layer A artifact.

Legacy `.py` loading remains for checked-in regression fixtures and the
not-yet-migrated primary causal adapter. It is not a safe provider-generated
lane: scientific arbitrary-Python training still fails closed unless an exact
candidate-bound OS attestation proves filesystem, network, credential,
process, resource, identity, and sandbox isolation on the actual execution
condition. A Modal container alone is not such proof.

## Budgets and artifacts

The common budget ledger separately accounts for the seed evaluation,
scientific proposal opportunities, provider attempts, prompt/completion tokens,
parse failures, repairs, training attempts/steps/examples, accelerator kind and
accelerator seconds,
evaluation cases, infrastructure retries, and terminal outcomes. Provider
retries and format repairs stay inside the original opportunity. Repairs have
both total and per-opportunity ceilings.

Every integrated C0–C3 persistence transition is mirrored into an append-only,
hash-linked event ledger. Candidate source, provider responses, and indexes use
content-addressed storage. Reconstruction verifies the chain and recovers run
state, budgets, ancestry, failures, canonical mechanism clusters, and one ITT
row per frozen assignment. A local chain can still be rewritten by an attacker
who controls the whole directory, so the scientific run also requires an
independently retained or WORM chain-head receipt.

The novelty, blinded-review, mechanism, replication, analysis, research-ledger,
and reporting packages are implemented with synthetic fixtures. Their
scientific corpus, reviewers, policies, thresholds, seeds, and effect-size
choices are intentionally not invented by the code.

## Set up the environment

```bash
git submodule update --init --recursive
cd architecture_discovery
uv sync --python 3.12 --group modal
.venv/bin/python scripts/openevolve_patch_bundle.py
source .venv/bin/activate
```

`uv sync --frozen` or equivalent dependency downloads belong in this setup and
cache-hydration phase, before the immutable cost-free freeze. The final freeze
uses `uv lock --check` and the already-created environment; it does not require
an offline sync whose cached build requirements may be incomplete.

The submodule checkout in a fresh clone contains the frozen OpenEvolve commit
with this repository's reviewed changes integrated. The validation command
verifies that commit, the historical patch inputs, and the resulting files. Paid
launches never apply or repair patches implicitly: `scripts/launch_modal.py`
must fail closed before invoking Modal unless the frozen commit, patch inputs,
and integrated OpenEvolve file hashes all match the manifest.

Offline checks do not need an API key.

## Provider-free validation

Run these first:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/validate_configs.py
../.venv/bin/ruff check --isolated --select E4,E7,E9,F --ignore E402 --target-version py312 --line-length 88 agents analysis architecture_ir artifacts audits baselines common containment evaluation mechanism novelty private_eval reconstruction replication reporting research_ledger review sealed_eval scripts study tests modal_action_journal.py modal_app.py modal_boundary.py modal_image_build.py
.venv/bin/python -m compileall -q common agents scripts tests study evaluation sealed_eval containment architecture_ir novelty review mechanism replication baselines analysis artifacts private_eval reconstruction research_ledger reporting modal_action_journal.py modal_app.py modal_boundary.py modal_image_build.py
.venv/bin/python -m pytest
```

Run the complete fake C0–C3 study plus feedback-free no-search control. Use a
new output directory:

```bash
.venv/bin/python scripts/study_offline_smoke.py \
  --output-dir /private/tmp/architecture-discovery-offline-check \
  --study-id offline-check-v1 \
  --blocks 1 \
  --opportunities 2
```

Run the one-command reconstruction/reporting exercise, again with a new
directory:

```bash
.venv/bin/python -m reporting.synthetic \
  --output /private/tmp/architecture-discovery-report-check
```

Statically validate the four named controller surfaces and a deterministic,
complete Architecture IR response fixture:

```bash
.venv/bin/python scripts/validate_engineering_canaries.py \
  --output /private/tmp/four-harness-controller-canary.json
```

This command makes zero provider calls, starts zero training runs, and neither
imports nor executes controller entrypoints or the fixed child graph. For
Normal Autoresearch, Semantic Autoresearch, OpenEvolve, and Semantic OpenEvolve,
it statically checks the CLI declaration, configuration, and prompt presence.
It validates one bounded full-document JSON fixture per named surface, but does
not inject that fixture into a live controller. Success means only that static
surface metadata and the IR boundary are internally consistent; it does not
prove provider connectivity, CUDA execution, or scientific readiness.

After separately completing and downloading the trusted ten-step CUDA smoke,
check its `seed_1` directory for internal consistency without retraining:

```bash
.venv/bin/python scripts/validate_engineering_canaries.py \
  --cuda-smoke-output outputs/development/modal_downloads/RUN_ID/candidate_smoke/seed_1 \
  --require-cuda-smoke \
  --output /private/tmp/four-harness-plus-cuda-canary.json
```

The artifact checker accepts only self-consistent `smoke_train_cuda_v2` output for
the checked-in `common/initial_candidate.ir.json`. It requires the immutable
`candidate_graph.json`, rejects a different graph, CPU/fallback declarations,
partial step counts, unchanged initialization, visible credential names, and
scientific-profile artifacts. These files are self-authored: consistency does
not prove that real CUDA execution occurred and is not an execution-origin
attestation.

Audit both pilot and main-study readiness without provider or training calls:

```bash
.venv/bin/python scripts/audit_scientific_readiness.py
```

Exit status 2 is currently expected: it means the fail-closed audit found open
gates. Read `scientific_decisions.yaml`, `readiness_evidence.yaml`, and the JSON
audit output; do not bypass the missing evidence.

## Modal/CUDA operator runbook

Modal is paid infrastructure. Never use a direct `modal run` command; the only
paid operator surface is `scripts/launch_modal.py`. Do not invoke that launcher
until the operator has reviewed the exact command, requested resources,
ten-minute image-build user-code timeout, five-minute Function timeout, cost
estimate, artifact plan, and cleanup plan and has explicitly approved it. Never
add `--detach`. The first invocation builds the pinned image as needed
but cannot create the named Volume: the runtime uses
`create_if_missing=False` and fails closed unless that operator-managed Volume
already exists. It is still a paid/remote action even if the selected
entrypoint is small. Do not use Tinker anywhere in this migration.

Every entrypoint documented in this runbook is an engineering-validation
entrypoint. The 300-second Function timeout is not a scientific full-profile
runtime allowance and cannot be reused for `full_train_cuda_v2`.

Provider canaries have three nested frozen deadlines: a 180-second provider
request timeout, a 240-second controller-subprocess timeout, and the
300-second Modal Function timeout. The 60-second gap after a provider timeout
is reserved for the controller to append and flush the sanitized terminal
`ProviderAttemptRecord` before outer-process finalization. Do not increase the
provider timeout independently; an in-flight request killed by the outer
deadline cannot produce reconcilable spend evidence.

Checkpoint resume has one shared 240-second action deadline inside that
300-second Function timeout. Contract probes, resumed training, and progression
verification receive bounded slices of that one deadline rather than three
independent clocks; each stage fails before starting if the time reserved for
later stages is unavailable. The gap between the action deadline and Function
timeout leaves roughly 60 seconds for result/failure recording, manifest-last
finalization, Volume commit, and Function shutdown.

The safe local plan imports no Modal SDK and makes no network or remote call:

```bash
.venv/bin/python scripts/modal_plan.py
.venv/bin/python -m pytest \
  tests/test_modal_boundary.py \
  tests/test_modal_import_boundary.py \
  tests/test_runtime_context.py
```

Before the first paid action, freeze the exact current source with the three
create-only, source-revision-bound recorder commands below. Run them only after
all source edits are final. They select a deterministic directory named by a
`validation_identity_sha256` over the complete execution-source manifest, Git
revision, pre-live validation-input manifest, and exact local execution
environment. A change to any of those inputs selects a new directory and
leaves every prior freeze as historical evidence. `private_eval/` and the
starting checkpoint are part of the complete local source digest but remain
excluded from the Modal image.

```bash
.venv/bin/python scripts/record_local_engineering_evidence.py \
  unit_tested --run-command
.venv/bin/python scripts/record_local_engineering_evidence.py \
  offline_smoke_tested --run-command
.venv/bin/python scripts/record_local_engineering_evidence.py \
  aggregate --run-command
```

Each frozen command revalidates the complete root and Architecture Discovery
source manifests, Git revision, dynamic readiness documents, Python/toolchain,
installed dependency versions, and exact sanitized environment before and
after execution. It refuses to publish if any identity changes. The unit
component records the complete Architecture Discovery pytest count. The
offline component records and securely manifests every file in a fresh
provider-free C0-C3 and no-search smoke, including its run/index/ledger
consistency. The aggregate command runs root `make check`, migration Ruff, both Git
diff checks, configuration, compile/import, offline `uv lock --check`, the
focused Modal/secret/device/resume/timeout suite, synthetic sealed Layer B/C,
synthetic reconstruction/reporting, four-controller static validation, and the
cost-free Modal plan. Its immutable Phase-2 receipt stores each exact command,
working-directory role, resolved argv, exact sanitized environment, timeout,
return code, bounded UTF-8 output, output hashes, and pass count against one
source, revision, validation-input, execution-environment, lock, and image
identity. `UV_OFFLINE=1` is fixed for lock validation and root `make check`.

`LocalEngineeringFreezeReceipt/1.0` binds the unit, offline-smoke, and Phase-2
receipt paths and raw SHA-256 values plus the complete persisted manifests and
current image-source SHA-256. The zero counters refer specifically to remote
actions, provider calls, remote training, and scientific runs; bounded local
fixture training is permitted during engineering tests.
Every paid action independently reopens this complete local freeze. Live actions
also carry the action-specific predecessor receipt path and raw SHA-256 described
below. Any missing, failed, reordered, or changed component blocks launch. Any
later source, Git revision, validation document, Python/toolchain, dependency,
or sanitized-environment change selects a new freeze and invalidates the
provider approval plan; do not edit, delete, or overwrite prior freeze evidence.

Every paid action must go through `scripts/launch_modal.py`. It validates the
action shape, separate approval flags, run IDs, and literal current source hash
before it starts the Modal CLI. A direct local import of `modal_app.py` exposes
no App or Function objects, so `modal run modal_app.py` and direct
`modal_app.py::FUNCTION` forms fail locally before image/App hydration. Never
set or forward the launcher's private process marker by hand.

Copy the freeze/plan's exact source-tree and image-source SHA-256 values into the
approval record. Select one new cohort ID; the first CUDA-environment run ID must
equal it. Set all three as literal shell values before any paid command:

```bash
SOURCE_TREE_SHA256=REVIEWED_64_HEX_FREEZE_VALUE
APPROVED_IMAGE_SOURCE_SHA256=REVIEWED_64_HEX_PLAN_VALUE
COHORT_ID=modal-cuda-env-YYYYMMDD
LIVE_COHORT_ROOT="outputs/readiness/modal_only_final/modal_live_cohorts/${SOURCE_TREE_SHA256}/${APPROVED_IMAGE_SOURCE_SHA256}/${COHORT_ID}"
```

For each individual paid invocation, also set a fresh literal
`APPROVED_MODAL_ACTION_CAP_USD` to that command's reviewed positive dollar
cap. Do not reuse either the human approval or an attempt ID for a retry. The
launcher first binds the current boot to one private
`ModalLocalHostAnchor/1.0`, then publishes a create-only
`ModalRemoteRunReservation/1.2` for every concrete remote run ID. Once all
reservations exist, the durable owned-attempt sequence is
reservation(s) -> `ModalActionIntent/1.6` -> `Popen` ->
`ModalLocalProcessStart/1.1` -> `ModalActionAttemptReceipt/3.6`. An aggregate
`canaries` launch with return code 0 or 2 also writes
`ProviderCanaryAggregateOutcomeReceipt/1.1`.

The intent and terminal for an owned attempt are written automatically under
`$LIVE_COHORT_ROOT/action_attempts`; their attempt-ID sets must match exactly.
The process-start marker is written immediately after `Popen` and binds the
intent digest, host and boot identity, PID, expected PGID and SID, and process
birth identity. A started attempt cannot be sealed unless that durable marker
is bound by its terminal receipt and the process group is proven closed.

A failure before cohort-intent ownership publishes its version-3.6 terminal
under `outputs/readiness/modal_only_final/modal_launch_rejections`, not in a
cohort roster. The `action_intent_persistence` and
`action_intent_persistence_uncertain` failure kinds are global-rejection-only.
If such a rejection owns any published reservations, those reservations remain
globally occupied and the attempt remains blocking pending recovery. Record the
printed terminal path and its raw SHA-256 before any dependent action.

`modal_action_journal.py` implements a lock-held global scanner and a gate
helper across cohort journals, global rejections, reservations, process-start
markers, recovery paths, the immutable global rejection seal, and migration
lineage. The paid launcher invokes the resolved-journal gate while holding the
shared lock, before containment publication, approval consumption,
reservations, or `Popen`. Readiness publishers scan before deriving evidence,
rescan before publication, and seal the exact global rejection roster before
the final migration lineage. Unresolved, incomplete, mutated, or post-seal
journal state therefore fails closed. This wiring is an engineering control;
it does not itself authorize a paid action.

Do not derive this value dynamically inside a paid command. Every paid
entrypoint requires it through `--expected-image-source-sha256`. The image
recipe copies the allowlisted source into a hash-verified temporary snapshot
before Modal resolves local paths, bakes the digest into the image, and rejects
the remote action if the executing source differs. Any source change requires
a new local plan, a new approval packet, and a new literal value.

`experiment_manifest.yaml` is deliberately included in both the image-source
manifest and the local engineering source-tree hash. Consequently,
`remote_execution.image_source_sha256` remains null with pending status in that
static manifest: embedding the digest derived from the manifest back into the
manifest would change the digest again and invalidate the approved cohort. The
stable `modal_engineering_evidence` section instead fixes the four receipt
paths and schema versions. Actual Modal IDs, run IDs, timestamps, measured
costs, and the computed image digest belong only in create-only files under
`outputs/readiness`, `readiness_evidence.yaml`, and these operator documents.
Those evidence/docs paths are excluded from both source hashes, so recording a
completed run does not retroactively change the image that produced it.

The shared local action lock is
`outputs/readiness/.modal_action.lock`. It serializes the paid launcher,
action-orphan recovery, cleanup-snapshot capture, prior-cohort accounting,
lineage, cohort-roster, resource-cleanup, migration-bundle, and held-lock
global-journal scan surfaces.
This is a local serialization and snapshot-consistency boundary, not a Modal
platform lock or cost ceiling.

After explicit approval, but before executing the approved paid command, check
the configured account without displaying a token secret:

```bash
.venv/bin/modal profile activate scalingintelligence
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal profile current
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal token info
```

The paid launcher and cleanup-capture helper ignore ambient Modal selection and
force profile `scalingintelligence` and environment `main`. Use the same two
literal values for every manual read-only inventory, Secret-name check, billing
query, or exact emergency stop. The paid launcher also requires the canonical
account-owned `$HOME/.modal.toml`, the active project virtual environment, and
exactly Modal 1.5.3.

### Approval and cost worksheet

The `--approved` and `--provider-approved` flags are assertions that the human
operator already approved that exact command; they are not interactive prompts
and they do not authorize a later command. Before each paid unit, copy the
exact command into the approval record and fill in a positive operator
authorization amount. That local threshold is not a platform maximum.
Increasing the action count, retry count, timeout, resource request, or
retention period requires a new approval.

Use a fresh snapshot of the official Modal pricing page at approval time. The
launcher accepts only `ModalPriceBasis/1.0`, bound to the approved image-source
SHA-256, no more than 48 hours old, and containing these currently reviewed
base rates: CPU `$0.0000131` per core-second, memory `$0.00000222` per
GiB-second, T4 `$0.000164` per second, and Volume storage `$0.09` per
GiB-month. The price page lists 1,024 included Volume GiB/month, but the local
estimate deliberately ignores that account-wide allowance. The page does not
separately list download-transfer pricing, so the exact disclosure is
`not_separately_listed_on_official_pricing_page`; the estimate records a
bounded transfer amount and `$0` for that component only under this basis.

At those rates, the source-bound local pre-launch estimates are:

| Approval unit | Runtime request-rate estimate | Cache-miss build estimate | One-month new-run storage estimate | Total local action estimate |
| --- | ---: | ---: | ---: | ---: |
| CPU-only `offline-smoke`, `download`, or `verify` | `$0.013188` | `$0.026376` | `$0.02267578125` | `$0.06223978125` |
| one T4 action, including one recovery `canary` | `$0.062388` | `$0.026376` | `$0.02267578125` | `$0.11143978125` |
| four-harness `canaries` | `$0.249552` | `$0.026376` | `$0.090703125` | `$0.366631125` |

Each possible new run reserves one month of storage for the full bounded
artifact set: `(256 MiB + 2 MiB) / GiB = 0.251953125 GiB`. A `download`
additionally discloses `0.50390625 GiB` of bounded remote verification/read
transfer and estimates that component at `$0` only because the official price
page does not list it separately. The launcher requires the operator's exact
`--modal-cost-cap-usd` to be greater than or equal to the applicable total and
rederives that total twice before `Popen`.

This gate has scope
`local_pre_popen_request_rate_and_one_gib_month_storage_estimate_not_platform_billing_cap`.
It is an operator authorization check, not a Modal billing cap or a guaranteed
floor/ceiling. Modal bills CPU and memory from requested or actual consumption
as applicable. Runtime CPU tuples use the second value as a soft throttle
threshold, while the memory tuple's second value is a hard limit. Functions
remain preemptible, GPU Functions cannot be made nonpreemptible, and Modal may
restart the same input after platform preemption independently of `retries=0`.
CPU bursting, preemption/restarts, backend image work, retained storage, and
pricing/account changes can therefore make actual cost differ from the local
estimate.

Before any action probe, subprocess, provider request, training, staged
publication, or artifact-verifier source read, a Function reloads the Volume,
constructs its execution context, creates the fresh run directory, writes only
`execution_context.json` and `image_source_manifest.json`, and explicitly
commits that provenance. This is the durable pre-action lease. A platform
restart reloads the committed lease and refuses the already-consumed run ID
before repeating the action. All mutable provider-free, candidate, resume, and
provider-canary outputs are written to a private temporary tree outside the
Volume. Successful outputs are captured as stable regular-file bytes and
exclusively published only after the existing provenance, staged files, and
exact final-result bytes fit the 256 MiB aggregate cap and the prospective
path roster fits the manifest's separate 2 MiB cap. The manifest is committed
last. Termination before publication leaves no partial training tree on the
Volume; the provenance-only run ID remains quarantined.

The dependency layer uses one Modal image-build function requesting 2 CPU
cores, 8192 MiB, no GPU, and a 600-second user-code timeout. Its subprocess/native
build concurrency is capped at two threads where supported, but neither that
setting nor the timeout is a CPU or memory hard limit. Its build-time network
is limited to installing and verifying Debian `git`, the pinned `uv` release,
and the frozen Python lock. The four primary provider-free runtime Functions
separately use `block_network=True` and publish an empirical denial probe; do
not describe the entire `modal run` command as network-blocked. The artifact
verifier has a separate static source-bound proof that its provider-secret
policy is false and therefore its function options derive
`block_network=True`; the four-run empirical-probe claim does not silently
expand to verifier executions. The recipe also has exactly two manifest-bound `copy=True`
image layers: the dependency subset before that build Function and the complete
verified source snapshot afterward. These platform image-layer operations are
separate from user-Function compute and do not expose repository-controlled
CPU, memory, or whole-recipe timeout knobs; disclose their exact plan-reported
file/byte counts and do not claim that the local cache-miss estimate bounds any
applicable backend image-build charge. The outer launcher deadline is a local
client deadline, not a platform runtime or billing ceiling. If it expires or
the client is interrupted, remote execution is treated as possibly started;
inspect and stop only the exact migration App, preserve the terminal receipt,
and obtain a new approval before another attempt.

All runtime actions use `max_containers=1`, `min_containers=0`, and `retries=0`.
The readiness actions retain their 300-second Function timeout; the separate
`openevolve-generic-60` action uses the explicitly disclosed 15,300-second
timeout above. The canary aggregate runs four logical calls sequentially.
Neither logical call counts nor local deadlines cap the number of container
attempts that platform preemption may cause. JSON resource/billing snapshots
and local receipt recorders start no Function, although ongoing Volume storage
can still accrue.

Create the Modal rate snapshot locally after reviewing the official pricing
page and within 48 hours of the paid launch. This command performs no lookup and
refuses to overwrite an existing record:

```bash
.venv/bin/python scripts/record_modal_readiness.py modal-price-basis \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --official-source-url https://modal.com/pricing \
  --retrieved-at-utc 2026-MM-DDTHH:MM:SSZ \
  --cpu-usd-per-core-second 0.0000131 \
  --memory-usd-per-gib-second 0.00000222 \
  --t4-usd-per-gpu-second 0.000164 \
  --volume-storage-usd-per-gib-month 0.09 \
  --included-volume-storage-gib-per-month 1024 \
  --download-transfer-pricing not_separately_listed_on_official_pricing_page

MODAL_PRICE_BASIS_PATH="outputs/readiness/modal_only_final/modal_price_bases/${APPROVED_IMAGE_SOURCE_SHA256}/2026MMDDTHHMMSSZ.json"
shasum -a 256 "$MODAL_PRICE_BASIS_PATH"
MODAL_PRICE_BASIS_SHA256=REVIEWED_RAW_FILE_SHA256
```

The path must contain the exact approved image digest and compact form of the
recorded timestamp. Review the file and raw hash before assigning the two
variables. A source, rate, disclosure, or freshness change requires a new
create-only record. Every paid command below binds both values into its intent
and terminal receipt.

Each harness has one opportunity, `retries=0`, and a maximum of 16,384
completion tokens per attempt. If `R_IN` and `R_OUT` are the provider's current
uncached rates per token and `I_h` is the approved input-token bound for
harness `h` and `R_REQUEST` is the non-negative per-request fee, the
four-harness provider cap is
`sum_h(I_h)*R_IN + 65536*R_OUT + 4*R_REQUEST`; a one-harness recovery cap is
`I_h*R_IN + 16384*R_OUT + R_REQUEST`. Fill the input bounds and rates before approval.
Provider approval is distinct from Modal approval.

Do not create a provider plan yet. `ProviderCanaryApprovalPlan/1.2` is bound to
the accepted candidate/resume preflight receipt as well as the current source
tree, image, and cohort. Its exact create-only command appears after that
preflight below. Hard-coded prompt byte counts are not approval evidence; use
the plan derived from the current constructors and configs.

The normal eight-run evidence bundle below has seven logical T4 executions,
one logical CPU-only offline execution, and eight logical CPU-only download
verifiers. Sum the exact per-action estimates for planning, but do not call the
sum a platform compute or billing ceiling: platform preemption, backend image
work, storage duration, and unattributed shared charges remain outside that
claim. Write the resulting operator authorization amount here before starting:
`APPROVED_MODAL_MAX_USD = ____`; `APPROVED_PROVIDER_MAX_USD = ____`;
`APPROVED_VOLUME_RETENTION_UNTIL_UTC = ____`.

A recovered harness requires a fresh one-T4 estimate and, if downloaded, a
fresh CPU-only estimate, as well as the one-harness provider cap. A new resume
attempt likewise requires a new T4 approval and a separate download approval.
Charges from failed, preempted, or uncertain attempts remain real.

Immediately after every launcher-started `modal run` below, inspect the selected
account/environment before advancing to the next paid unit:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal app list --env main --json
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal container list --env main --json
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal endpoint list --env main --json
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal volume list --env main --json
```

Do not advance while a migration-created app, container, or endpoint remains
active or while the expected artifact Volume is absent.

The first approved live command is one pre-hydration launcher invocation that
starts one ephemeral app and one T4 environment
function. It requests two CPU cores with a two-core soft throttle threshold,
8192 MiB memory with an 8192 MiB hard limit, one T4, at most one
container, no warm container, no retry, and a 300-second function timeout. On
an image-cache miss, that invocation first uses one build function with two CPU
cores, 8192 MiB memory, no GPU, Debian/Python package network access, and a
600-second user-code timeout; its API does not expose the same tuple limits.
The `$0.11143978125` local estimate includes the cache-miss request-rate amount
and one month of bounded new-run storage. It is not platform-enforced. After
each source-producing launch,
copy the exact terminal receipt path that the launcher created into the
corresponding `*_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH` variable below. A download
is bound to that terminal receipt, its paired intent, and, for an aggregate
canary source, the launcher's typed aggregate outcome receipt. Do not infer
or hand-edit these paths:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action cuda-environment \
  --run-id "$COHORT_ID" \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --approved

CUDA_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$CUDA_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
CUDA_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id "$COHORT_ID" \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-cuda-env-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$CUDA_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$CUDA_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/record_modal_readiness.py cuda-environment \
  --run-id "$COHORT_ID" \
  --cohort-id "$COHORT_ID"

CUDA_RECEIPT_PATH="$LIVE_COHORT_ROOT/components/modal_cuda_environment_validation_receipt.v2.0.json"
shasum -a 256 "$CUDA_RECEIPT_PATH"
CUDA_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256
```

The validated launcher starts `modal run`, which builds the image from
`pyproject.toml`, `uv.lock`, the pinned
OpenEvolve source, and the allowlisted trusted source. The output receipt binds
the Modal image ID when available and always binds the complete image-source
SHA-256. The command creates an ephemeral app; Modal normally stops it when the
client exits. The named Volume persists until deliberately removed. The
environment run expects exactly these six accepted files under
`/runs/RUN_ID`: `execution_context.json`, `image_source_manifest.json`,
`provider_free_network_denial_probe.json`, `cuda_environment.json`,
`remote_action_result.json`, and `artifact_manifest.json`. The
download uses one separately approved CPU-only verifier and the local recorder
then creates and revalidates the create-only CUDA component receipt without a
network call. Do not spend on the later T4 or provider units unless all three
commands above and their immediate resource checks pass.

After that environment receipt passes, use new run IDs for the provider-free
offline smoke and CUDA train/evaluate smoke. Download and locally validate each
stage before advancing. The candidate round-trip receipt is deliberately
created before resume or provider spend:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action offline-smoke \
  --run-id modal-offline-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --cuda-receipt-path "$CUDA_RECEIPT_PATH" \
  --cuda-receipt-sha256 "$CUDA_RECEIPT_SHA256" \
  --approved

OFFLINE_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$OFFLINE_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
OFFLINE_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-offline-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-offline-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$OFFLINE_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$OFFLINE_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/validate_downloaded_offline_study.py \
  outputs/development/modal_downloads/modal-offline-YYYYMMDD

.venv/bin/python scripts/record_modal_readiness.py offline-smoke \
  --run-id modal-offline-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-offline-YYYYMMDD \
  --verifier-attempt-id REVIEWED_OFFLINE_VERIFIER_ATTEMPT_ID

OFFLINE_RECEIPT_PATH="$LIVE_COHORT_ROOT/components/modal_offline_smoke_validation_receipt.v2.0.json"
shasum -a 256 "$OFFLINE_RECEIPT_PATH"
OFFLINE_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/launch_modal.py \
  --action candidate-smoke \
  --run-id modal-candidate-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --cuda-receipt-path "$CUDA_RECEIPT_PATH" \
  --cuda-receipt-sha256 "$CUDA_RECEIPT_SHA256" \
  --offline-smoke-receipt-path "$OFFLINE_RECEIPT_PATH" \
  --offline-smoke-receipt-sha256 "$OFFLINE_RECEIPT_SHA256" \
  --approved

CANDIDATE_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$CANDIDATE_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
CANDIDATE_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-candidate-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-candidate-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$CANDIDATE_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$CANDIDATE_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/record_modal_readiness.py artifact-round-trip \
  --source-run-id modal-candidate-YYYYMMDD \
  --verifier-run-id modal-verifier-candidate-YYYYMMDD \
  --verifier-attempt-id REVIEWED_CANDIDATE_VERIFIER_ATTEMPT_ID \
  --cohort-id "$COHORT_ID"

ARTIFACT_ROUND_TRIP_RECEIPT_PATH="$LIVE_COHORT_ROOT/components/modal_artifact_round_trip_validation_receipt.v3.0.json"
shasum -a 256 "$ARTIFACT_ROUND_TRIP_RECEIPT_PATH"
ARTIFACT_ROUND_TRIP_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/validate_engineering_canaries.py \
  --cuda-smoke-output outputs/development/modal_downloads/modal-candidate-YYYYMMDD/candidate_smoke/seed_1 \
  --require-cuda-smoke \
  --output /private/tmp/modal-candidate-YYYYMMDD-validation.json

.venv/bin/python scripts/launch_modal.py \
  --action checkpoint-resume \
  --source-run-id modal-candidate-YYYYMMDD \
  --run-id modal-resume-attempt-1-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --artifact-round-trip-receipt-path "$ARTIFACT_ROUND_TRIP_RECEIPT_PATH" \
  --artifact-round-trip-receipt-sha256 "$ARTIFACT_ROUND_TRIP_RECEIPT_SHA256" \
  --approved

RESUME_1_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$RESUME_1_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
RESUME_1_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-resume-attempt-1-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-resume-attempt-1-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$RESUME_1_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$RESUME_1_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved
```

A resume never writes into the candidate-smoke run. `--source-run-id` names
that immutable successful source; `--run-id` names a fresh resume attempt. The
final aggregate recorder independently recomputes the resume negative probes
and progression check before accepting the attempt.

The resume function first verifies the source run's checkpoint manifest. It
copies the single retained step-5 `partial_resume_checkpoint.pt`, immutable
candidate graph, and event prefix through step 5 into the fresh attempt. It
keeps the partial checkpoint, initializes `latest_resume_checkpoint.pt` from
the same bytes, runs the read-only mismatch probes, resumes steps 6 through 10,
and then writes `resume_progression_verification.json`. That evidence binds the
source manifest and hashes, optimizer and scheduler progression, examples,
final summary, best checkpoint, and contiguous event chain. The source run is
verified again before the attempt is finalized.

If an attempt fails, retain its failure manifest and retry from the same source
with another unused attempt ID; never reuse either run directory:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action checkpoint-resume \
  --source-run-id modal-candidate-YYYYMMDD \
  --run-id modal-resume-attempt-2-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --artifact-round-trip-receipt-path "$ARTIFACT_ROUND_TRIP_RECEIPT_PATH" \
  --artifact-round-trip-receipt-sha256 "$ARTIFACT_ROUND_TRIP_RECEIPT_SHA256" \
  --approved

RESUME_2_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$RESUME_2_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
RESUME_2_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-resume-attempt-2-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-resume-attempt-2-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$RESUME_2_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$RESUME_2_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved
```

If attempt 2 is the accepted fresh resume, substitute its run ID for attempt 1
in every later cleanup/aggregate command. Never name a failed attempt as the
accepted `--resume-run-id`.

The download entrypoint first invokes the bounded remote artifact verifier,
then writes into the ignored local output root and rechecks every size and
SHA-256. Keep the checkpoint, training summary, event log, execution context,
CUDA environment, image-source manifest, source binding, negative-probe and
progression evidence, artifact manifest, and resume result together. Candidate
smoke runs finalized with `artifact_manifest.checkpoint.json` and resume
attempts finalized with `artifact_manifest.json` are both downloadable. Never
overwrite a previous run directory. Before transfer, the declared manifest is
rejected if any file exceeds 64 MiB or their aggregate exceeds 256 MiB.
The same per-file and aggregate ceilings, plus each file's smaller declared
size, are enforced while streaming; a violation removes the new incomplete
local destination.

`--source-evidence-recovery` is reserved for a separately approved verifier of
a failed or uncertain source attempt when typed evidence may need preservation.
For an ordinary Modal action, the source must be a closed failed launcher
attempt with its original intent and terminal receipt, a complete manifest-last
failure directory, a sanitized failure/result pair, an exact downloaded
`ExecutionContext`, and one fresh successful verifier capture. Classify the
failed source `failed` and `quarantined`, classify the verifier
`artifact_verifier` and `validation_only`, attribute both exact Apps in billing,
and link the failed attempt to the later accepted replacement. A partial,
hard-preempted, unmanifested, or unverifiable source remains blocking. Provider
sources additionally require a typed provider outcome. Recovery never turns
the failed source into an accepted primary run, and the original
intent/terminal/aggregate chain is retained. Never add the flag to an ordinary
successful download.

That verifier/source-evidence recovery is distinct from local action-journal
orphan recovery. The latter is operational through
`scripts/recover_modal_action_journal.py` and uses these create-only path shapes
under a cohort's `action_recoveries/` directory:
`ATTEMPT.intent.v1.0.json`, `ATTEMPT.host-containment.v1.0.json`, and
`ATTEMPT.resolution.v1.0.json`. Their exact schemas are respectively
`ModalActionRecoveryIntent/1.0`,
`ModalActionRecoveryHostContainment/1.0`, and
`ModalActionRecoveryResolution/1.0`; the operator input is
`ModalActionRecoveryRequest/1.0`. Never hand-author, edit, delete, copy, or
replace any of the three journal stages. An incomplete triplet remains a
global scanner blocker, and the CLI resumes only byte-identical stages in the
fixed intent -> host-containment -> resolution order.

Put the request in a source-excluded operator directory inside this project but
outside `outputs/readiness/modal_only_final/modal_live_cohorts`. It must be an
explicit normalized absolute path to an account-owned, single-link,
non-symlink regular file with mode `0600`. Its exact field roster is:

```json
{
  "schema_name": "ModalActionRecoveryRequest",
  "schema_version": "1.0",
  "attempt_id": "ORPHAN_32_LOWERCASE_HEX_ATTEMPT_ID",
  "fresh_candidate_attempt_id": "FRESH_32_LOWERCASE_HEX_ATTEMPT_ID",
  "expected_branch": "definitely_not_started",
  "snapshot_manifest_path": null,
  "initial_reservation_bindings": [
    {
      "path": "outputs/readiness/modal_only_final/modal_remote_run_reservations/RUN_ID.json",
      "sha256": "EXACT_RAW_FILE_SHA256",
      "size_bytes": 1
    }
  ]
}
```

The reservation roster must contain the exact sorted path/SHA-256/size binding
for every reservation present when the request is frozen; use the real positive
size, not the illustrative `1`. Select `definitely_not_started` only when the
scanner proves a reservation-only or global-rejection pre-`Popen` orphan. This
branch forbids a snapshot and may create only missing members of the canonical
reservation roster. Select `may_have_started_contained` only with an existing
six-read `ModalCleanupSnapshotCaptureManifest/1.0`; set
`snapshot_manifest_path` to its absolute `0600` path. That branch requires
either the same boot with the exact bound marker/terminal and proven absent
process group, or a strictly later trusted boot session. A live group, PID
reuse, ambiguous boot evidence, active target resource, missing target Volume,
incomplete billing hour, changed reservation, or changed input blocks recovery.

Run the cost-free inspection before publication:

```bash
RECOVERY_REQUEST="$PWD/outputs/operator/modal_action_recovery_requests/ORPHAN_ATTEMPT.request.v1.0.json"
chmod 600 "$RECOVERY_REQUEST"
.venv/bin/python scripts/recover_modal_action_journal.py inspect \
  --project-root "$PWD" \
  --request "$RECOVERY_REQUEST"

.venv/bin/python scripts/recover_modal_action_journal.py resolve \
  --project-root "$PWD" \
  --request "$RECOVERY_REQUEST"
```

`inspect` performs no persistent write and neither operation imports Modal,
contacts a provider, captures a snapshot, queries billing or prices, signals a
process, or deletes evidence. `resolve` consumes only already-frozen local
evidence under the shared lock. For a may-have-started orphan it reports the
complete App-name/environment-main billing snapshot plus the full local Modal
authorization as an additional unresolved-start reserve; that authorization is
not a platform hard bound. If provider ledgers are incomplete, it reserves the
entire frozen provider approval. Only complete successful known-usage ledgers
plus the intent-bound price basis permit exact provider accounting.

Every recovered attempt remains quarantined, is ineligible for final
acceptance, and requires a fresh attempt, fresh run IDs, and a fresh cohort ID
under the unchanged source/image identity. Preserve the recovered cohort in a
prior-quarantine accounting receipt and include that receipt in final lineage.
Use the request-named attempt ID on the newly approved launcher command:

```bash
# Append every ordinary source, image, predecessor, timeout, and approval flag.
.venv/bin/python scripts/launch_modal.py \
  --attempt-id FRESH_32_LOWERCASE_HEX_ATTEMPT_ID \
  --action REVIEWED_ACTION \
  --run-id FRESH_REVIEWED_RUN_ID \
  --cohort-id FRESH_REVIEWED_COHORT_ID \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --approved
```

`--attempt-id` accepts exactly 32 lowercase hexadecimal characters and gets no
recovery bypass: the same fresh global scan, ownership checks, cost gates, and
zero-retry launcher path still apply. A prior approval does not authorize this
fresh action.

The approved `download` action also makes a bounded create-only local capture
of the remote verifier's success receipt before source transfer, or its typed
failure receipt when the verifier raises. Validate a retained capture with
`record_modal_readiness.py validate-verifier-capture`. Fetching the verifier
directory later with `modal volume get` is not intrinsically cost-free: it may
incur storage or transfer charges and therefore requires its own explicit
bounded download approval. Never describe a Volume evidence fetch as a local
free validation step.

Modal Volumes can background-commit files every few seconds and commit again
when a container shuts down. The atomic file writes plus manifest-last rule are
therefore the acceptance boundary, not a transactional rollback mechanism. A
timeout, uncertain process-group closure, or other failed action can leave a
partial `/runs/RUN_ID` directory without a valid final artifact manifest. Treat
that run ID as quarantined: record its App and call IDs, inspect it read-only
with `MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main .venv/bin/modal
volume ls --env main --json rl4rl-architecture-artifacts /runs/RUN_ID`, never
download, accept, resume from, or reuse it, and retry only under a fresh approved
run ID. Preserve the quarantined directory through resource and billing
accounting. Removing it later is a separate destructive operation requiring an
exact target and explicit approval; no failure path deletes it automatically.

Before confirming the provider Secret or approving any canary spend, run the
strict local preflight. It recomputes candidate Layer A validity, resume source
binding, negative probes and progression, offline reconstruction/reporting,
the four download captures, both early component receipts, unique call IDs, and
the shared image-source identity:

```bash
.venv/bin/python scripts/record_modal_readiness.py candidate-resume-preflight \
  --environment-run-id "$COHORT_ID" \
  --environment-verifier-run-id modal-verifier-cuda-env-YYYYMMDD \
  --environment-verifier-attempt-id REVIEWED_ENVIRONMENT_VERIFIER_ATTEMPT_ID \
  --offline-run-id modal-offline-YYYYMMDD \
  --offline-verifier-run-id modal-verifier-offline-YYYYMMDD \
  --offline-verifier-attempt-id REVIEWED_OFFLINE_VERIFIER_ATTEMPT_ID \
  --candidate-run-id modal-candidate-YYYYMMDD \
  --candidate-verifier-run-id modal-verifier-candidate-YYYYMMDD \
  --candidate-verifier-attempt-id REVIEWED_CANDIDATE_VERIFIER_ATTEMPT_ID \
  --resume-run-id modal-resume-attempt-1-YYYYMMDD \
  --resume-verifier-run-id modal-verifier-resume-attempt-1-YYYYMMDD \
  --resume-verifier-attempt-id REVIEWED_RESUME_VERIFIER_ATTEMPT_ID \
  --cohort-id "$COHORT_ID"
```

The cohort-scoped `ModalOfflineSmokeValidationReceipt/2.0` must already exist
before `candidate-smoke`. Its recorder reopens the receipt and rederives the
full offline study, network-denial proof, downloaded manifest, execution
identity, and remote verifier evidence. The create-only
`CandidateResumePreflightReceipt/2.0` binds that receipt and all four exact
verifier attempt IDs while revalidating the artifacts directly.

This command is read-only, provider-free, and starts zero remote calls or
training runs. Do not proceed unless it exits 0 and reports `valid: true`,
`remote_calls_started: 0`, `provider_calls_started: 0`, and
`training_runs_started: 0`. If a later resume attempt was accepted, substitute
that exact fresh ID here and everywhere below.

Copy `binding_sha256` from the preflight output, derive its canonical path, and
hash the raw file. Then create the provider plan and provider price basis. These
commands initialize no provider client and make no network or Modal call:

```bash
CANDIDATE_RESUME_PREFLIGHT_BINDING_SHA256=REVIEWED_OUTPUT_BINDING_SHA256
CANDIDATE_RESUME_PREFLIGHT_RECEIPT_PATH="$LIVE_COHORT_ROOT/components/candidate_resume_preflight_receipts/v2.0/${CANDIDATE_RESUME_PREFLIGHT_BINDING_SHA256}.json"
shasum -a 256 "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_PATH"
CANDIDATE_RESUME_PREFLIGHT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256

PROVIDER_APPROVAL_PLAN_PATH="$LIVE_COHORT_ROOT/provider_canary_approval/provider_canary_plan.v1.2.json"
.venv/bin/python scripts/provider_canary_plan.py \
  --output "$PROVIDER_APPROVAL_PLAN_PATH" \
  --source-tree-sha256 "$SOURCE_TREE_SHA256" \
  --cohort-id "$COHORT_ID" \
  --candidate-resume-preflight-receipt-path "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_PATH" \
  --candidate-resume-preflight-receipt-sha256 "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_SHA256"
PROVIDER_APPROVAL_PLAN_SHA256=REVIEWED_PLAN_SELF_HASH

.venv/bin/python scripts/record_modal_readiness.py provider-price-basis \
  --source-tree-sha256 "$SOURCE_TREE_SHA256" \
  --image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --cohort-id "$COHORT_ID" \
  --official-source-url https://openai.com/api/pricing/ \
  --retrieved-at-utc 2026-MM-DDTHH:MM:SSZ \
  --uncached-input-usd-per-million-tokens INPUT_RATE \
  --output-usd-per-million-tokens OUTPUT_RATE \
  --per-request-fee-usd REQUEST_FEE
PROVIDER_PRICE_BASIS_PATH="$LIVE_COHORT_ROOT/resource_cleanup/provider_price_basis.json"
shasum -a 256 "$PROVIDER_PRICE_BASIS_PATH"
PROVIDER_PRICE_BASIS_SHA256=REVIEWED_RAW_FILE_SHA256
APPROVED_PROVIDER_ACTION_CAP_USD=REVIEWED_POSITIVE_USD_CAP
```

`ProviderCanaryApprovalPlan/1.2` derives exact current constructor payloads and
the frozen one-opportunity token ceilings. Do not replace its source-bound
values with hard-coded byte counts. The provider cap must cover every
plan-selected request, including the conservative full-request reserve for a
failed or request-start-uncertain attempt. It is an authorization gate, not a
provider billing ceiling.

Provider canaries require a separate cost approval and the dedicated named
`rl4rl-discovery-provider` Secret with the three required keys
`DISCOVERY_API_KEY`, `DISCOVERY_API_BASE=https://api.openai.com/v1`, and
`DISCOVERY_MODEL=gpt-5.6-sol`; do not add other keys. Modal 1.5.3
`required_keys` metadata proves that these keys are present but does not expose
the complete key roster, so a metadata check cannot prove the absence of extra
keys. The runtime therefore constructs the controller child environment from
an exact allowlist and forwards only these three values. Prefer creating the
Secret in the Modal dashboard. Check only its name from the CLI; do not print
or copy its values:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal secret list --env main --json
```

If the dashboard is unavailable, the create-only CLI form below reads the API
key interactively instead of placing it in shell history. Do not add `--force`:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal secret create --env main rl4rl-discovery-provider \
  DISCOVERY_API_KEY=- \
  DISCOVERY_API_BASE=https://api.openai.com/v1 \
  DISCOVERY_MODEL=gpt-5.6-sol
```

The Modal canary boundary rejects any other literal endpoint or model before
starting a controller subprocess. After download, the canary validator also
requires the exact official provider identity, endpoint, model, and complete
request-setting contract recorded by every harness.

After the operator confirms that Secret and separately approves provider cost,
repeat the current plan's four-request, 81,920-input-token, and
65,536-completion-token ceilings plus a numeric USD limit in the approval
packet, then run the four one-opportunity harnesses in their frozen sequence:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action canaries \
  --run-id modal-canary-YYYYMMDD \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 2100 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --provider-cost-cap-usd "$APPROVED_PROVIDER_ACTION_CAP_USD" \
  --provider-approval-plan-path "$PROVIDER_APPROVAL_PLAN_PATH" \
  --approval-plan-sha256 "$PROVIDER_APPROVAL_PLAN_SHA256" \
  --provider-price-basis-path "$PROVIDER_PRICE_BASIS_PATH" \
  --provider-price-basis-sha256 "$PROVIDER_PRICE_BASIS_SHA256" \
  --candidate-resume-preflight-receipt-path "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_PATH" \
  --candidate-resume-preflight-receipt-sha256 "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_SHA256" \
  --approved \
  --provider-approved

AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256
```

The aggregate attempts every harness exactly once in `CANARY_ORDER`. An
ordinary failure is reduced to its exception class, later harnesses still run,
and the aggregate prints all four exact `success`/`failed` outcomes before
exiting nonzero if any failed. It never retries a harness. Process-control
exceptions such as `KeyboardInterrupt` are not caught.

`--approved` asserts approval for Modal compute and storage; the distinct
`--provider-approved` flag asserts the separately reviewed model-provider
spend. The local entrypoint refuses both aggregate and single-harness provider
actions unless both flags are present. Neither flag is persisted for a later
command.

Each provider canary writes provider-produced artifacts to a private temporary
directory outside the mounted Volume. After the action returns, the runner
validates the exact harness-specific roster and evidence schemas, rejects
provider credential material, symlinks, non-regular or over-limit files, and
only then exclusively publishes the scanned safe files to the fresh Volume run
directory. Provider output is never generated directly inside the Volume. If a
failure follows a paid request, only the sanitized append-only provider-attempt
ledger is preserved; prompts, responses, and invalid staged evidence are not
published.

This produces `modal-canary-YYYYMMDD-greedy-ar`,
`modal-canary-YYYYMMDD-semantic-ar`,
`modal-canary-YYYYMMDD-openevolve-generic`, and
`modal-canary-YYYYMMDD-openevolve-semantic`. The canonical migration receipt
uses the flat download layout below. Each download performs exactly one bounded
CPU-only remote manifest verifier, locally rechecks every file size and SHA-256,
and exclusively saves that verifier result under
`outputs/readiness/modal_artifact_round_trip/RUN_ID/VERIFIER_RUN_ID/remote_verification.json`.
Do not run a separate `--action verify` in the normal flow.

The environment, offline, candidate, and resume runs are already flat under
`outputs/development/modal_downloads`. Download the four accepted canaries into
that same flat parent. Every `RUN_ID` directory and remote-verification capture
must be new:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-canary-YYYYMMDD-greedy-ar \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-canary-greedy-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-canary-YYYYMMDD-semantic-ar \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-canary-semantic-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-canary-YYYYMMDD-openevolve-generic \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-canary-oe-generic-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-canary-YYYYMMDD-openevolve-semantic \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-canary-oe-semantic-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$AGGREGATE_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved

.venv/bin/python scripts/validate_engineering_canaries.py \
  --modal-canary-download-root outputs/development/modal_downloads/modal-canary-YYYYMMDD \
  --require-modal-canaries \
  --output /private/tmp/modal-canary-YYYYMMDD-validation.json
```

The selector path `outputs/development/modal_downloads/modal-canary-YYYYMMDD`
must **not** itself exist. It is a prefix selector for the four flat sibling
directories with the frozen terminal suffixes. The validator rejects any
missing or additional sibling for that prefix, re-verifies the outer manifests
and execution contexts, requires four unique Modal call IDs with one shared
image/source identity, rejects credential-bearing or executor-absolute JSON
fields, and checks each bounded CUDA smoke and controller result without
contacting Modal or the provider.

For each aggregate harness reported failed or left unstarted by an operator
interrupt, run at most one separately approved recovery with a fresh run ID.
This bounded recovery path makes one synchronous, one-opportunity call and
does not resume, overwrite, or rerun the earlier attempt.

```bash
.venv/bin/python scripts/launch_modal.py \
  --action canary \
  --harness openevolve_generic \
  --run-id modal-canary-recovery-YYYYMMDD-openevolve-generic \
  --cohort-id "$COHORT_ID" \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --provider-cost-cap-usd "$APPROVED_PROVIDER_ACTION_CAP_USD" \
  --provider-approval-plan-path "$PROVIDER_APPROVAL_PLAN_PATH" \
  --approval-plan-sha256 "$PROVIDER_APPROVAL_PLAN_SHA256" \
  --provider-price-basis-path "$PROVIDER_PRICE_BASIS_PATH" \
  --provider-price-basis-sha256 "$PROVIDER_PRICE_BASIS_SHA256" \
  --candidate-resume-preflight-receipt-path "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_PATH" \
  --candidate-resume-preflight-receipt-sha256 "$CANDIDATE_RESUME_PREFLIGHT_RECEIPT_SHA256" \
  --approved \
  --provider-approved

RECOVERY_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH=REVIEWED_PRINTED_TERMINAL_PATH
shasum -a 256 "$RECOVERY_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH"
RECOVERY_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256=REVIEWED_RAW_FILE_SHA256
```

The accepted frozen IDs are `greedy_autoresearch`, `semantic_autoresearch`,
`openevolve_generic`, and `openevolve_semantic`. Use this recovery action only
after the same explicit Modal and provider-cost disclosures and approvals as
the aggregate command. Omitting `--harness`, inventing another ID, or passing a
harness to `--action canaries` fails locally before a remote call. A recovery
run ID must end in that harness's frozen download suffix: `-greedy-ar`,
`-semantic-ar`, `-openevolve-generic`, or `-openevolve-semantic`.
The recovery approval must name the selected harness and its current `I_h`:
16,384 input tokens for either native harness or 24,576 for either OpenEvolve
harness, plus at most 16,384 completion tokens and any per-request fee for its
single permitted request.

Download the recovery into the same flat parent under its fresh run ID. Do not
delete or overwrite the failed attempt. For the `openevolve_generic` example
above:

```bash
.venv/bin/python scripts/launch_modal.py \
  --action download \
  --run-id modal-canary-recovery-YYYYMMDD-openevolve-generic \
  --cohort-id "$COHORT_ID" \
  --verifier-run-id modal-verifier-canary-recovery-oe-generic-YYYYMMDD \
  --local-output outputs/development/modal_downloads \
  --expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --outer-cli-timeout-seconds 1200 \
  --modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD" \
  --modal-price-basis-path "$MODAL_PRICE_BASIS_PATH" \
  --modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256" \
  --source-action-attempt-receipt-path "$RECOVERY_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_PATH" \
  --source-action-attempt-receipt-sha256 "$RECOVERY_CANARY_SOURCE_ACTION_ATTEMPT_RECEIPT_SHA256" \
  --approved
```

Once every accepted run is downloaded, create one explicit selector instead
of copying or symlinking four directories into a synthetic bundle. Substitute
the accepted aggregate or recovery run ID for each harness:

```bash
.venv/bin/python scripts/validate_engineering_canaries.py \
  --create-modal-canary-selector provider-canary-cohort-YYYYMMDD \
  --source-tree-sha256 "$SOURCE_TREE_SHA256" \
  --image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --cohort-id "$COHORT_ID" \
  --modal-canary-run greedy_autoresearch=ACCEPTED_GREEDY_RUN_ID \
  --modal-canary-run semantic_autoresearch=ACCEPTED_SEMANTIC_RUN_ID \
  --modal-canary-run openevolve_generic=ACCEPTED_GENERIC_RUN_ID \
  --modal-canary-run openevolve_semantic=ACCEPTED_SEMANTIC_OE_RUN_ID

.venv/bin/python scripts/validate_engineering_canaries.py \
  --modal-canary-selector \
    "$LIVE_COHORT_ROOT/provider_canary_selection/provider-canary-cohort-YYYYMMDD/canary_run_selector.json" \
  --require-modal-canaries \
  --output /private/tmp/modal-canary-selected-validation.json
```

`ModalProviderCanaryRunSelector/2.0` is create-only and binds the source tree,
image, cohort, and each canonical
`outputs/development/modal_downloads/RUN_ID` path, raw artifact-manifest
SHA-256, execution-context file SHA-256, image-source SHA-256, and Modal image
ID. Validation rejects unknown fields, symlinks, duplicate evidence, hash
drift, harness substitutions, and mixed image/source identity while ignoring
unselected failed directories in the flat download parent.

The final recorders consume one explicit source/image/cohort-bound
`ModalMigrationCohortRoster/4.0`; they never derive a run ID from a prefix. The
roster names all eight accepted primary runs and their launcher attempt IDs,
all eight artifact-verifier run/attempt IDs and immutable captures, the
create-only provider-canary selector, exact paired intent/terminal attempt-ID
sets and applicable aggregate receipts under
`$LIVE_COHORT_ROOT/action_attempts`, accepted, failed,
completed-but-unaccepted, and provider-request-start-uncertain canary outcomes,
per-attempt billing object IDs, failed/quarantined/recovery links and run IDs,
validation-only classifications, the completed billing query window, the
cleanup snapshot-capture manifest, `ModalMigrationLineage/1.1`, provider price
basis, and fixed superseded usage record. Global launch-rejection terminals are
not cohort journal members and must not be inserted into this roster. A
recovery changes the explicit accepted entry and adds a failure-to-recovery
link; it never removes the failed receipt, partial outcome, or run ID.

No migration-created container, running app, endpoint, or detached call should
remain after the inspections above. If an ephemeral app remains, first preserve
the launcher's terminal receipt and exact App ID, classify all possibly started
run IDs as quarantined, and identify the exact `ap-...` ID from the selected
`main` environment. Stop only that App:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal app stop --env main ap-EXACT_MIGRATION_APP_ID
```

Re-run the read-only inventories until the exact App is stopped and no migration
container or endpoint remains. Do not reuse any affected run ID, do not delete
its Volume directory, and do not infer zero cost. Capture the completed-hour
billing evidence and quarantine accounting described below. Any replacement
action needs a new run ID, per-action cap, and explicit approval.

Modal 1.5.3 exposes neither a standalone historical task/function-call list nor
a detached-call list. `modal container list --env main --json` is the available active
container/task surface. The receipts therefore record both unavailable
inventories as `unavailable_in_modal_cli_1_5_3`; they do not claim a directly
observed detached-call count. Static source validation prohibits `--detach`,
`.spawn()`, and unbounded `.map()`, while synchronous completion and empty
migration container/app inventories provide the runtime evidence. Confirm the
migration app in the Modal dashboard as an additional operator check.

### Immutable migration receipts and final audit

After all eight accepted executions have been downloaded and every relevant UTC
billing hour is complete, capture all six final Modal 1.5.3 snapshots as one
bounded create-only cohort. Do not use shell redirection or `--for today`; those
forms do not bind a complete explicit billing window or an immutable capture
manifest:

```bash
.venv/bin/python scripts/capture_modal_cleanup_snapshots.py \
  --source-tree-sha256 "$SOURCE_TREE_SHA256" \
  --image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256" \
  --cohort-id "$COHORT_ID" \
  --capture-id final-cleanup-YYYYMMDD \
  --billing-window-start-utc 2026-MM-DDTHH:00:00Z \
  --billing-window-end-utc 2026-MM-DDTHH:00:00Z

SNAPSHOT_CAPTURE_MANIFEST_PATH="$LIVE_COHORT_ROOT/resource_cleanup/snapshot_captures/final-cleanup-YYYYMMDD/capture_manifest.v1.0.json"
shasum -a 256 "$SNAPSHOT_CAPTURE_MANIFEST_PATH"
SNAPSHOT_CAPTURE_MANIFEST_SHA256=REVIEWED_RAW_FILE_SHA256
```

The helper starts no Function and performs no write operation against Modal. It
uses the exact Modal 1.5.3 executable, profile `scalingintelligence`, environment
`main`, 45-second per-command limits, a 300-second outer deadline, and zero
retries. `ModalCleanupSnapshotCaptureManifest/1.0` binds the ordered App,
container, endpoint, Volume, `/runs`, and hourly billing snapshots and is
published only after all six validate. The helper holds the shared
`outputs/readiness/.modal_action.lock` from before the first remote read through
validation and create-only manifest publication, so a paid launch or sealer
cannot interleave with the six-read snapshot.

The app snapshot must include every App ID attributed by the cohort roster:
accepted primary actions, download/verifier actions, failed remote starts, and
recoveries. Each must be stopped with zero tasks. It must not contain an
unattributed App named `rl4rl-architecture-discovery`. Container and endpoint
snapshots must contain no migration-created active resource, and the Volume
snapshot must include exactly one `rl4rl-architecture-artifacts` row. The
`/runs` snapshot must include every accepted, verifier, explicitly failed or
quarantined, recovery, and preserved superseded run directory; unrelated run
directories are retained and counted rather than rejected. The recorders
reject duplicate IDs and rows, extra/missing fields, and Boolean values
masquerading as integers. If a migration app remains active, stop only its
exact `ap-...` ID and create a new capture ID after cleanup; never edit or
overwrite a completed capture.

Billing data includes only complete intervals and can lag. Every prior
quarantined cohort has its own six-read snapshot and hourly billing window, and
all prior captures must finish before the first final-cohort action starts. The
final helper's `--billing-window-start-utc` is inclusive and
`--billing-window-end-utc` is exclusive; both are hour-aligned and cover only
the final cohort's attempts. Prior receipts retain and account their own
windows. All returned rows lie inside their selected completed window; Modal
may omit idle hours and the recorders never invent them. A zero-row or
zero-measured started attempt is allowed only with the conservative billing-lag
reserve described below.

Each billed attempt explicitly names its App object ID; preflight/lock failures
with no remote start use `no_remote_start`, while an uncontained started process
blocks sealing and is recovered before accounting. Every selected row is
counted once in its owner window. Later zero-cost rows are inert unless the
exact owner snapshot selected them; they neither move an earlier row into the
final window nor erase its prior ownership. Modal call IDs come from bound
`ExecutionContext/1.0` evidence, not the billing report. The shared/recent App
list must contain every final App and validates any prior App still visible,
but may omit a historical stopped App already bound by its prior snapshot.
Unrelated Apps, run directories, and cached images are retained; an active or
unattributed App named `rl4rl-architecture-discovery` is rejected. Reuse of a
cached Modal image ID is valid only for the same image-source digest.

Cleanup sums rows uniquely attributed to each App-backed attempt and reports,
rather than suppresses, any amount over its local authorization. The local cap
is not a platform hard bound. `migration_total_usd` is exactly the final-only
App-attributed compute total plus every validated prior-quarantined cohort
`app_compute_subtotal_usd` plus preserved legacy usage `$0.00643852` for
`modal-cuda-env-20260809-02`. No row may be counted twice.

The cleanup receipt also reports Modal compute exposure without conflating its
categories. Complete-hour App rows are `measured_app_billing_usd`. An
unresolved started attempt reserves its full local authorization cap; a failed
or zero-measured started attempt reserves the nonnegative cap remainder for
billing lag. Measured billing plus that reserve is
`conservative_compute_exposure_usd`, while any measured amount above the local
cap is reported separately as `measured_over_local_authorization_cap_usd`.
These reserves and overage checks do not change `migration_total_usd`, and the
local authorization cap is explicitly not a Modal platform hard bound.

Retained Volume storage is reported separately as
`conservative_retained_Volume_storage_estimate_not_billed_cost`. For every
retained run, the recorder uses
`MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES`, sums prior
immutable accounting estimates, applies the bound current Volume USD/GiB-month
rate to final retained bytes, and subtracts no shared included quota. This is an
estimate, not measured billed storage, and it is never added to
`migration_total_usd`. Backend image work and other shared charges are likewise
not silently allocated. None of these records turns an operator cap into a
Modal platform ceiling.

The provider price basis was already created and hash-bound before provider
spend; do not replace it with a later price. The aggregate reopens the exact
provider attempt ledgers, requires unique request and response IDs, and
reconciles successful response token counts. Known successes are estimated
from response usage plus the request fee. Each failed request conservatively
reserves its approval-plan input ceiling, requested completion ceiling, and
request fee because no response usage may exist. The result remains an
approval-bound estimate, not an authoritative provider invoice or billed-cost
claim. An empty published provider-attempt ledger is not zero-start evidence:
it must be paired with exact `ProviderRequestStartUncertainEvidence/1.0`, is
reported with a 0--1 request-count bound, and reserves one full approved
request. If the controller ledger was absent, the runtime publishes that same
empty ledger and uncertainty evidence with ledger state `missing`; readiness
still reserves the request and never infers a zero start. If the published
ledger itself is missing, readiness rejects the cohort as unresolved.

Prepare a strict local input for the canonical
`$LIVE_COHORT_ROOT/cohort_roster.v4.0.json` seal. `accepted_primary_runs`,
`accepted_attempt_ids`, and
`artifact_verifiers` must each contain these eight literal keys:
`cuda_environment`, `offline_smoke`, `candidate_smoke`, `resume_attempt`,
`canary_greedy_autoresearch`, `canary_semantic_autoresearch`,
`canary_openevolve_generic`, and `canary_openevolve_semantic`. Each primary
verifier record has exactly `source_label`, `source_run_id`, `verifier_run_id`,
`attempt_id`, `remote_verification_path`, `remote_verification_sha256`,
`verifier_execution_context`, and `expected_remote_receipt_roster`.
Additionally provide:

- `action_intent_receipts`, `action_attempt_receipts`, and
  `provider_canary_aggregate_outcome_receipts`: the sorted project-relative
  paths that exactly classify every `.intent.json`, terminal `.json`, and
  `.aggregate.json` in the canonical journal directory. An aggregate is
  mandatory exactly when a started `canaries` terminal returns 0 or 2. Intent
  records use `ModalActionIntent/1.6`; terminal records use
  `ModalActionAttemptReceipt/3.6`; the intent and terminal attempt-ID sets must
  be identical. Started terminals transitively validate their
  `ModalRemoteRunReservation/1.2` records and bound
  `ModalLocalProcessStart/1.1` marker. Both intent and terminal bind the raw
  Modal price-basis SHA and independently rederived estimate;
- `attempt_classifications`: one sorted record per attempt with roles drawn
  from `accepted_primary`, `artifact_verifier`, `failed`, `quarantined`,
  `recovery`, and `validation_only`;
- `billing_attributions`: one sorted record per attempt with disposition
  `billed` and explicit unique App object IDs, or `no_remote_start` and an empty
  object list; `start_uncertain` is representable for audit but blocks final
  acceptance;
- `additional_artifact_verifiers`: every paid verifier outside the eight final
  successful captures, including a typed Volume success/failure capture or an
  explicit unresolved identity. Every failed verifier names a distinct fresh
  successful verifier attempt for the same source, the retry must start no
  earlier than the failed verifier finishes, and both executions retain their
  exact four-file capture, billing, and execution identity;
- `provider_canary_outcomes`: the exact per-child outcome and launcher,
  provider ledger and successful verifier link. Only
  `provider_request_start_uncertain` may bind a missing ledger (null ledger
  SHA-256 and evidence state `missing`) or an exactly empty ledger (the empty
  file SHA-256 and evidence state `present`); both use
  `ProviderRequestStartUncertainEvidence/1.0`, represent the closed 0..1 range,
  reserve one full approved request, and cannot be classified as measured zero
  or enter the accepted aggregate. Every ledger record has
  `attempt_ordinal == 1`, the exact approved generation-settings digest, and
  input/output usage within the approved ceilings. If provider transport
  succeeds and IR validation, training, or finalization then fails, retain that
  successful record, charge its known usage, classify the child and aggregate
  child as failed/quarantined, and recover with a fresh run; never retry the
  already successful request within that child. A successful aggregate child
  is `accepted` or `completed_unaccepted` and is never provider-recovery
  eligible. Other unaccepted outcomes require an explicit recovery link;
- `provider_canary_selector_path` and `provider_canary_selector_sha256` for the
  create-only final four-run selector;
- `recovery_links`: every failed attempt ID, distinct accepted recovery attempt
  ID, and exact recovered run IDs; the failed attempt must finish no later than
  the recovery starts, and their concrete remote run identities must be
  disjoint;
- sorted `declared_failed_run_ids`, `declared_quarantined_run_ids`, and
  `declared_recovery_run_ids` lists, including empty lists when none occurred;
- the exact hourly `billing_window_start_utc`, `billing_window_end_utc`, later
  `snapshot_captured_at_utc`, and the canonical snapshot-capture manifest path
  and raw SHA-256;
- the canonical `ModalMigrationLineage/1.1` path and raw SHA-256;
- `provider_price_basis_path`; and
- `superseded_usage` fixed to run `modal-cuda-env-20260809-02`, amount
  `0.00643852`, and accounting basis
  `preserved_prior_measurement_excluded_from_cohort_billing_snapshot`.

The top-level schema name/version are `ModalMigrationCohortRoster`/`4.0`, and
`cleanup_run_id` must equal the explicitly accepted CUDA-environment run. The
cleanup receipt hashes this roster; the migration bundle hashes it again and
includes every unique Modal price basis referenced by a bound attempt in its
required-artifact roster. Cleanup and bundle receipts both use version `4.0`.
After either receipt exists, do not edit the roster, price basis, attempt
receipts, verifier captures, snapshots, or downloaded runs.

The download actions preserve one immutable capture for every verifier attempt;
the eight accepted-source captures use
`modal_artifact_round_trip/SOURCE_RUN_ID/VERIFIER_RUN_ID/remote_verification.json`,
and failed or locally interrupted verifier attempts retain their typed Volume
captures alongside the successful retry evidence. The fail-fast steps
already created the CUDA-environment and candidate round-trip component
receipts. Do not rerun those create-only commands.

Finalization order is strict: seal every prior quarantined cohort that exists,
seal the complete cross-cohort lineage, seal the final cohort roster, create the
cleanup component, and then create the bundle. Every `--input` file must be an
absolute-path, strict-JSON, account-owned, single-link, non-symlink regular file
with mode `0600`; the commands refuse looser input. They are local,
provider-free, and make no Modal or provider call. Each prior-accounting input
is the complete candidate `ModalPriorCohortQuarantineAccounting/1.1` payload,
not a partial patch. The roster input is likewise the complete candidate
`ModalMigrationCohortRoster/4.0` payload. Both recorders independently rederive
their source/image/cohort bindings, sorted identities, time order, and monetary
totals before create-only publication:

```bash
# Repeat only for each real prior quarantined cohort; never fabricate one.
# Capture every prior six-read snapshot before the first final-cohort action.
PRIOR_OPERATOR_DIR="$PWD/outputs/operator/prior_quarantine_accounting"
PRIOR_REQUEST="$PRIOR_OPERATOR_DIR/PRIOR_COHORT.request.v1.0.json"
PRIOR_CANDIDATE="$PRIOR_OPERATOR_DIR/PRIOR_COHORT.candidate.v1.1.json"
mkdir -p "$PRIOR_OPERATOR_DIR"

.venv/bin/python scripts/record_modal_readiness.py \
  prior-quarantine-accounting-template \
  --source-tree-sha256 PRIOR_SOURCE_TREE_SHA256 \
  --image-source-sha256 PRIOR_IMAGE_SOURCE_SHA256 \
  --cohort-id PRIOR_COHORT_ID \
  --recorded-at-utc 2026-MM-DDTHH:MM:SSZ \
  --snapshot-capture-manifest /absolute/path/to/prior/capture_manifest.v1.0.json \
  --output "$PRIOR_REQUEST"

# Read-only derivation: print the complete candidate or its exact blockers.
.venv/bin/python scripts/record_modal_readiness.py \
  prior-quarantine-accounting-inspect \
  --input "$PRIOR_REQUEST"

# Create a noncanonical, validator-clean candidate and review its bytes.
.venv/bin/python scripts/record_modal_readiness.py \
  prior-quarantine-accounting-scaffold \
  --input "$PRIOR_REQUEST" \
  --output "$PRIOR_CANDIDATE"
shasum -a 256 "$PRIOR_CANDIDATE"

# Publication independently rederives and revalidates every claim.
.venv/bin/python scripts/record_modal_readiness.py \
  prior-quarantine-accounting \
  --input "$PRIOR_CANDIDATE"

# ModalMigrationLineageInput/1.0 contains only the final source/image/cohort,
# exact eight accepted run IDs, exact eight accepted attempt IDs, and the sorted
# unique list of canonical prior quarantine-accounting paths (possibly empty).
chmod 600 /absolute/path/lineage-input.json
.venv/bin/python scripts/record_modal_readiness.py migration-lineage \
  --input /absolute/path/lineage-input.json

MIGRATION_LINEAGE_PATH="$LIVE_COHORT_ROOT/migration_lineage.v1.1.json"
shasum -a 256 "$MIGRATION_LINEAGE_PATH"
MIGRATION_LINEAGE_SHA256=REVIEWED_RAW_FILE_SHA256

chmod 600 /absolute/path/cohort-roster.json
.venv/bin/python scripts/record_modal_readiness.py cohort-roster \
  --input /absolute/path/cohort-roster.json

COHORT_ROSTER_PATH="$LIVE_COHORT_ROOT/cohort_roster.v4.0.json"
shasum -a 256 "$COHORT_ROSTER_PATH"
COHORT_ROSTER_SHA256=REVIEWED_RAW_FILE_SHA256

.venv/bin/python scripts/record_modal_readiness.py resource-cleanup \
  --cohort-roster "$COHORT_ROSTER_PATH"

.venv/bin/python scripts/record_modal_readiness.py migration-bundle \
  --cohort-roster "$COHORT_ROSTER_PATH"
```

Template and scaffold outputs are create-only normalized absolute paths inside
the authenticated project but outside the canonical live-cohort receipt
namespace. `inspect` writes nothing; `scaffold` writes only the noncanonical
operator candidate. The final publish command does not trust the scaffold: it
reruns the resolved global scan and exact evidence derivation before creating
the canonical receipt. A prior receipt owns only its selected billing window,
App lifecycle evidence, attempts, and run directories. The final receipt owns
the final window; shared recent resource listings and same-digest image-cache
reuse do not transfer historical compute between them.

The aggregate recorder independently revalidates eight distinct run IDs and
Modal call IDs, eight distinct verifier run/call IDs and contexts, one shared
image identity, the CUDA environment, candidate Layer A smoke, fresh resume
progression and negative probes, all four selected canaries, offline
reconstruction/reporting, all remote/local downloads, every exact paired cohort
intent/terminal, its transitive reservations and started-process marker, every
applicable aggregate outcome and recovery link, zero active migration
resources, complete hourly Modal billing attribution, the preserved superseded
amount, and the separately labeled provider usage estimate. Receipts are
create-only. Global launch rejections remain globally owned rather than roster
members. Every readiness publisher scans the global journal while holding the
shared lock, requires all attempts resolved, and rescans before publication.
Final lineage creates or reuses the immutable global rejection seal and then
rescans again; any changed, unresolved, or unaccounted recovery state blocks.

Only after all four component receipts exist and independently revalidate may
the operator update the four existing Modal gates in `readiness_evidence.yaml`.
Until then they remain `passed: false` with null `receipt_path`,
`receipt_sha256`, and `selected_cohort_identity`. A passed record requires a
nonempty evidence description, the canonical cohort-scoped path, its raw
SHA-256, the exact `{source_tree_sha256, image_source_sha256, cohort_id}` object,
and these contracts:

```yaml
modal_cuda_environment_validated:
  receipt_path_template: "$LIVE_COHORT_ROOT/components/modal_cuda_environment_validation_receipt.v2.0.json"
  receipt_contract: {schema_name: ModalCUDAEnvironmentValidationReceipt, schema_version: "2.0"}
modal_artifact_round_trip_validated:
  receipt_path_template: "$LIVE_COHORT_ROOT/components/modal_artifact_round_trip_validation_receipt.v3.0.json"
  receipt_contract: {schema_name: ModalArtifactRoundTripValidationReceipt, schema_version: "3.0"}
modal_resource_cleanup_validated:
  receipt_path_template: "$LIVE_COHORT_ROOT/components/modal_resource_cleanup_validation_receipt.v4.0.json"
  receipt_contract: {schema_name: ModalResourceCleanupValidationReceipt, schema_version: "4.0"}
modal_migration_validation_bundle_validated:
  receipt_path_template: "$LIVE_COHORT_ROOT/components/modal_migration_validation_bundle_receipt.v4.0.json"
  receipt_contract: {schema_name: ModalMigrationValidationBundleReceipt, schema_version: "4.0"}
```

The `$LIVE_COHORT_ROOT` strings above are documentation templates, not literal
values accepted in `readiness_evidence.yaml`; write the resolved project-relative
paths only after the corresponding create-only files exist.

Then revalidate configuration and run the provider-free readiness audit:

```bash
.venv/bin/python scripts/validate_configs.py
.venv/bin/python scripts/audit_scientific_readiness.py \
  --json-output outputs/readiness/post-modal-migration-audit.json
```

The audit reopens and revalidates every receipt and committed artifact. Exit
status 2 remains expected until the separate scientific/governance gates are
resolved; the Modal receipt gates themselves must be reported as passing. An
engineering migration receipt does not authorize `full_train_cuda_v2`, a paid
scientific pilot, or the main study.

Do not stop another user's app and do not delete the shared Volume. A specific
non-scientific smoke directory may be removed only after its verified local
copy is accepted and the operator explicitly chooses to remove it:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal volume rm -r \
  rl4rl-architecture-artifacts /runs/EXACT_SMOKE_RUN_ID
```

For ongoing spend monitoring after the frozen receipt snapshot, use:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal billing report \
  --for today --resolution h --show-resources --json
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  .venv/bin/modal billing summary --for "this month" --json
```

## Historical MPS compatibility checks

Check what the current process can see:

```bash
.venv/bin/python scripts/check_environment.py
```

An explicitly non-scientific historical MPS smoke can validate compatibility on
the ordinary Mac Terminal:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/train_candidate.py \
  --candidate common/initial_candidate.ir.json \
  --profile smoke_train_v1 \
  --device mps \
  --seed 1 \
  --output-dir /private/tmp/architecture-training-mps-smoke
```

Do not interpret that smoke as `full_train_v1` validation. After the smoke
passes on the real Mac Terminal, the trusted IR seed can be exercised with the
full frozen training profile using:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/train_candidate.py \
  --candidate common/initial_candidate.ir.json \
  --profile full_train_v1 \
  --device mps \
  --seed 1 \
  --output-dir outputs/readiness/full_train_v1_seed_1
```

This historical path is expensive and is not the canonical execution condition
for new runs. Do not run it merely to validate controller plumbing.

After a future full-profile run completes successfully in an MPS-available
process, create the hash-linked evidence receipt without retraining:

```bash
.venv/bin/python scripts/record_mps_validation.py \
  --training-output-dir outputs/readiness/full_train_v1_seed_1 \
  --output outputs/readiness/full_train_v1_mps_evidence.json
```

The recorder rejects CPU execution, partial step counts, fallback-enabled runs,
unmatched candidate/profile hashes, weak-containment manifests, and modified
training artifacts. It creates the receipt once and will not overwrite it.

## Scientific launch sequence

For the scientific entrypoint, do not export provider credentials until the
readiness audit is otherwise clean. The required order is:

1. Resolve every null in `scientific_decisions.yaml`, complete its PI approval
   record, and change its status to `approved`; placeholders and empty values
   fail the audit.
2. Populate matching manifest values and freeze an executable
   `study/scientific_study.json` bound to the manifest hash.
3. Freeze the primary C0-C3 candidate format. Migrate its proposal/store path to
   the trusted IR interpreter, or produce a real candidate-bound OS containment
   attestation for its legacy Python lane.
4. Complete the versioned Modal CUDA validation and retain its hash-linked
   accelerator receipt. Preserve historical MPS receipts separately.
5. Populate, independently review, and freeze the novelty corpus and reviewer
   custody record.
6. Freeze and cross-link the research protocol, mechanism plan, replication
   policy, and analysis inputs; establish a cryptographically verified external
   artifact anchor; then rerun the readiness audit.
7. Record explicit PI pilot authorization only after all non-pilot gates pass.
   Run a paid pilot only when the audit reports `pilot_ready: true`.
8. Use pilot estimates to freeze the final power/analysis plan. Run the main
   study only when `main_study_ready` is true.

The gated future entrypoint requires an explicit phase:

```bash
.venv/bin/python scripts/study_scientific_run.py \
  --phase pilot \
  --study-spec study/scientific_study.json \
  --initial-candidate common/initial_candidate.py \
  --output-root outputs/scientific
```

It exits before provider initialization while any required gate is open. The
legacy `.py` argument shown here reflects the still-blocked primary adapter; do
not confuse it with the IR-only native engineering canaries in the Modal
runbook.

## Direct-run API environment (not Modal)

The active Modal operator path above does not read provider credentials from the
local shell; it injects the three provider values through the named Modal
Secret. Only use these exports for a deliberately selected direct runner on a
separately provisioned CUDA host. Keep secrets in the current shell or a local
ignored secret manager; never add them to YAML, Markdown, source, or git:

```bash
export DISCOVERY_API_KEY="YOUR_KEY"
export DISCOVERY_API_BASE="https://api.openai.com/v1"
export DISCOVERY_MODEL="gpt-5.6-sol"
export DISCOVERY_TRAIN_DEVICE="cuda"
export DISCOVERY_ALLOW_CPU_TRAINING="0"
export PYTORCH_ENABLE_MPS_FALLBACK="0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

The API key belongs to an OpenAI platform project; the ChatGPT subscription is
separate. Worker environments omit provider credentials. These exports are not
needed by any paid launcher command above and do not authorize a direct provider
call. The gated scientific entrypoint must still wait until its readiness audit
passes.

## Historical MPS engineering-pilot record

The direct local MPS commands used on 2026-08-08 are historical, not the active
operator path. New provider-backed engineering canaries use the bounded Modal
sequence above with `smoke_train_cuda_v2`. The historical runs exercised the
real provider, trusted IR interpreter, from-scratch training, public smoke
evaluation, controller lineage, and artifact paths. They were exploratory
mechanics tests, not scientifically valid architecture rankings.

The recorded 2026-08-08 UTC run under `outputs/engineering_10x4/` completed all
40 proposal opportunities and all 44 permitted candidate trainings. Every
training summary reports ten completed steps on MPS with unsupported-operation
fallback disabled, and every runtime-validity record passed. All public smoke
accuracies were 0.0; do not interpret this mechanics result as evidence that
one harness or proposed architecture is better than another.

The four runs request 40 proposal opportunities and permit 44 candidate
trainings in total because each harness evaluates the shared seed once. A
malformed or invalid proposal consumes its opportunity without training. Safe
native-run resume was not implemented for those artifacts. Do not point a
rerun at any of their non-empty directories.
