# Architecture Discovery on AdderBoard

This repository is an offline-tested research-infrastructure system for studying
novel transformer architecture discovery. AdderBoard is used only as a
correctness and accuracy environment. Parameter count is descriptive metadata,
not an optimization target: it is never a reward, selection criterion,
tie-breaker, or stopping rule. Neutral pre-allocation compute and memory
ceilings still reject candidates that are unsafe to execute on the host.

## Current status

The four native controller harnesses now have an explicitly non-scientific,
IR-only engineering-pilot path. Their prompts produce complete JSON
Architecture IR documents; a trusted evaluator-owned interpreter constructs and
trains each model from scratch. Static four-harness checks and CPU integration
tests pass. After a one-opportunity Greedy Autoresearch canary, the full paid
10-by-4 engineering pilot completed sequentially on MPS on 2026-08-08 UTC. All
40 proposal opportunities terminated, and all 44 seed/proposal candidates
completed ten-step from-scratch training with fallback disabled and passed
runtime validity. Public smoke accuracy was 0.0 throughout, so this validates
controller and training mechanics only and provides no useful architecture
ranking.

A scientific pilot and the main C0-C3 study remain **blocked**. The primary
causal engine still uses its legacy Python proposal adapter, and the repository
still lacks resolved principal-investigator decisions, real `full_train_v1`
MPS evidence, a populated frozen literature corpus and reviewer roster,
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
and a study-wide lease permits only one MPS run at a time. The independent
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

`full_train_v1` remains frozen: MPS, float32, deterministic algorithms, 30,000
optimizer steps, batch 512, AdamW at 0.001 with betas 0.9/0.98 and weight decay
0.1, 300 warmup steps, cosine decay, gradient clipping at 1.0, 2,000 public
development examples every 1,000 steps, and a 1,800-second safety ceiling. It
has no CPU fallback.

`smoke_train_v1` is a ten-step engineering check only. It is not valid for
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
process, resource, identity, and sandbox isolation on the real MPS host.

## Budgets and artifacts

The common budget ledger separately accounts for the seed evaluation,
scientific proposal opportunities, provider attempts, prompt/completion tokens,
parse failures, repairs, training attempts/steps/examples, MPS seconds,
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
uv sync --python 3.12
source .venv/bin/activate
```

Offline checks do not need an API key.

## Provider-free validation

Run these first:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/validate_configs.py
.venv/bin/python -m compileall -q common agents scripts tests study evaluation sealed_eval containment architecture_ir novelty review mechanism replication baselines analysis artifacts reconstruction research_ledger reporting
.venv/bin/python -m pytest -q
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
prove provider connectivity, MPS execution, or scientific readiness.

After separately completing the trusted ten-step MPS smoke, check its existing
artifacts for internal consistency without retraining:

```bash
.venv/bin/python scripts/validate_engineering_canaries.py \
  --mps-smoke-output /private/tmp/architecture-training-mps-smoke \
  --require-mps-smoke \
  --output /private/tmp/four-harness-plus-mps-canary.json
```

The artifact checker accepts only self-consistent `smoke_train_v1` output for
the checked-in `common/initial_candidate.ir.json`. It requires the immutable
`candidate_graph.json`, rejects a different graph, CPU/fallback declarations,
partial step counts, unchanged initialization, visible credential names, and
scientific-profile artifacts. These files are self-authored: consistency does
not prove that real MPS execution occurred and is not an execution-origin
attestation.

Audit both pilot and main-study readiness without provider or training calls:

```bash
.venv/bin/python scripts/audit_scientific_readiness.py
```

Exit status 2 is currently expected: it means the fail-closed audit found open
gates. Read `scientific_decisions.yaml`, `readiness_evidence.yaml`, and the JSON
audit output; do not bypass the missing evidence.

## MPS checks

Check what the current process can see:

```bash
.venv/bin/python scripts/check_environment.py
```

An explicitly non-scientific MPS smoke can validate basic device mechanics on
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

This is expensive and is not a substitute for the other scientific-readiness
gates. Do not run it merely to validate controller plumbing.

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
4. Complete real MPS validation and retain its hashed evidence.
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
not confuse it with the IR-only native engineering pilots below.

## API environment

Keep secrets in the current shell or a local ignored secret manager; never add
them to YAML, Markdown, source, or git:

```bash
export DISCOVERY_API_KEY="YOUR_KEY"
export DISCOVERY_API_BASE="https://api.openai.com/v1"
export DISCOVERY_MODEL="gpt-5.6-sol"
export DISCOVERY_TRAIN_DEVICE="mps"
export DISCOVERY_ALLOW_CPU_TRAINING="0"
export PYTORCH_ENABLE_MPS_FALLBACK="0"
```

The API key belongs to an OpenAI platform project; the ChatGPT subscription is
separate. Worker environments omit provider credentials. These exports may be
used for the explicitly non-scientific engineering canary below. The gated
scientific entrypoint must still wait until its readiness audit passes.

## Four native engineering pilots

These runs exercise the real provider, trusted IR interpreter, from-scratch
training, public smoke evaluation, controller lineage, and artifact paths. They
are exploratory mechanics tests, not scientifically valid architecture
rankings. `--engineering-pilot` is mandatory and forces `smoke_train_v1`,
`smoke_eval_v1`, and a mechanics-only eligibility threshold. It cannot create
authoritative scientific evidence.

From the repository root, verify MPS and run one paid, one-opportunity canary
in an ordinary Mac Terminal. Use a fresh output path:

```bash
cd architecture_discovery
source .venv/bin/activate
export DISCOVERY_API_KEY="YOUR_KEY"
export DISCOVERY_API_BASE="https://api.openai.com/v1"
export DISCOVERY_MODEL="gpt-5.6-sol"
export PYTORCH_ENABLE_MPS_FALLBACK=0
export DISCOVERY_TRAIN_DEVICE=mps
export DISCOVERY_ALLOW_CPU_TRAINING=0

python scripts/check_environment.py
python agents/greedy_autoresearch/run.py \
  --engineering-pilot \
  --iterations 1 \
  --seed 1 \
  --evaluation-cases 64 \
  --device mps \
  --output-dir outputs/engineering_canary/greedy_seed_1
```

Do not continue if the environment reports `mps_available: false`, the canary
fails, or its `run_summary.json` does not report one terminal proposal
opportunity. A successful canary performs two candidate trainings: the shared
seed and one proposal.

Then run the four 10-opportunity pilots **sequentially**, each into a fresh
directory. Running them concurrently would compete for the Mac's unified GPU
memory and invalidate the intended compute treatment.

```bash
python agents/greedy_autoresearch/run.py \
  --engineering-pilot --iterations 10 --seed 1 --evaluation-cases 64 --device mps \
  --output-dir outputs/engineering_10x4/greedy_seed_1

python agents/semantic_autoresearch/run.py \
  --engineering-pilot --iterations 10 --seed 1 --evaluation-cases 64 --device mps \
  --output-dir outputs/engineering_10x4/semantic_autoresearch_seed_1

python agents/openevolve_generic/run.py \
  --engineering-pilot --iterations 10 --seed 1 --evaluation-cases 64 --device mps \
  --output-dir outputs/engineering_10x4/openevolve_generic_seed_1

python agents/openevolve_semantic/run.py \
  --engineering-pilot --iterations 10 --seed 1 --evaluation-cases 64 --device mps \
  --output-dir outputs/engineering_10x4/openevolve_semantic_seed_1
```

The recorded 2026-08-08 UTC run under `outputs/engineering_10x4/` completed all
40 proposal opportunities and all 44 permitted candidate trainings. Every
training summary reports ten completed steps on MPS with unsupported-operation
fallback disabled, and every runtime-validity record passed. All public smoke
accuracies were 0.0; do not interpret this mechanics result as evidence that
one harness or proposed architecture is better than another.

The four runs request 40 proposal opportunities and permit 44 candidate
trainings in total because each harness evaluates the shared seed once. A
malformed or invalid proposal consumes its opportunity without training. Safe
native resume is not implemented, so never point a rerun at a non-empty output
directory; choose a new directory instead.
