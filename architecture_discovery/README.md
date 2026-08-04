# Architecture Discovery on AdderBoard

This repository is an offline-tested research-infrastructure system for studying
novel transformer architecture discovery. AdderBoard is used only as a
correctness and accuracy environment. Parameter count is unrestricted,
descriptive metadata; it is never a reward, selection criterion, tie-breaker,
or stopping rule.

## Current status

The engineering infrastructure is implemented and tested offline. A scientific
pilot and the main study are both **blocked**. The blockers are deliberate:
unresolved principal-investigator decisions, no trusted architecture-IR
interpreter, no proven arbitrary-Python OS boundary on MPS, no real
`full_train_v1` MPS evidence, no populated frozen literature corpus or reviewer
roster, no production Layer B orchestration or scheduled scientific no-search path, no
cryptographically authenticated external ledger anchor, no completed pilot,
and no explicit PI launch authorization.

`scripts/study_scientific_run.py` audits those gates before reading provider
credentials or constructing an API client. It cannot presently start a paid
run.

CUDA support is additive and does not change those MPS gates. Apptainer 1.4.1
is available through the cluster module system, but there is no authenticated,
candidate-bound adversarial containment receipt. Therefore a trusted
non-scientific CUDA smoke is permitted on an allocated GPU, while arbitrary
generated Python remains blocked from formal scientific CUDA execution.

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
transition state, or repair feedback. Native Greedy Autoresearch and OpenEvolve
entrypoints remain secondary system replications, not the primary causal
comparison.

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

GPT-5.6 Sol proposes architecture source; it does not train the arithmetic
model. Trusted evaluator code owns:

- fresh seeded initialization through `build_untrained_model(seed)`;
- deterministic public training data and order;
- optimizer, schedule, steps, examples, and wall-time ceiling;
- public-development-only checkpoint selection;
- generic autoregressive decoding and Layer A evaluation;
- checkpoint/source/profile/task/seed identity verification.

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

Two separately hashed CUDA profiles are also available:

- `full_train_cuda_a40_v1` retains the complete 30,000-step algorithmic budget,
  batch 512, AdamW/schedule/validation/checkpoint rules, float32, and all
  no-fallback constraints of `full_train_v1`. It is a separate hardware
  condition and is not presently authorized for formal arbitrary-Python runs.
- `smoke_train_cuda_v1` is exactly ten optimizer steps and explicitly
  non-scientific. It may be used only with `common/initial_candidate.py` for
  machinery validation; that path is additionally bound to the reviewed
  candidate SHA-256 so modified or alternate Python fails before import.

CUDA and MPS trajectories must never be pooled merely because their profiles
share seeds or algorithmic settings. Cross-backend analysis requires an
explicit analysis plan. Parameter count remains descriptive metadata only on
both backends and is never an optimization objective.

## Containment and transformer validity

Generated Python runs in a credential-scrubbed worker and is statically scanned
for direct and indirect capability recovery. That is defense in depth, not a
security boundary. Scientific arbitrary-Python training fails closed unless a
candidate-bound trusted attestation proves filesystem, network, credential,
process, resource, identity, and sandbox isolation on the real MPS host.

The repository includes an extensible typed architecture graph and runtime,
causality, sequence-dependence, parameter-influence, and attention-intervention
probes. There is not yet a trusted evaluator-owned IR interpreter, and those
probes are not yet retained by the scientific Layer A record. That gate remains
open rather than falling back to class names or source keywords.

## Budgets and artifacts

The common budget ledger separately accounts for the seed evaluation,
scientific proposal opportunities, provider attempts, prompt/completion tokens,
parse failures, repairs, training attempts/steps/examples, MPS seconds,
evaluation cases, infrastructure retries, and terminal outcomes. Provider
retries and format repairs stay inside the original opportunity. Repairs have
both total and per-opportunity ceilings.

The v1 `mps_seconds` budget and ledger records retain their original schema and
round-trip behavior. New accelerator resource records use a v2 backend-neutral
schema keyed by backend and hardware condition; it rejects accidental MPS/CUDA
pooling.

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

Audit both pilot and main-study readiness without provider or training calls:

```bash
.venv/bin/python scripts/audit_scientific_readiness.py
```

Exit status 2 is currently expected: it means the fail-closed audit found open
gates. Read `scientific_decisions.yaml`, `readiness_evidence.yaml`, and the JSON
audit output; do not bypass the missing evidence.

The audit also validates an optional `CudaA40ValidationReceipt` v1.0 as a
separate engineering readiness level. Even a passing CUDA smoke receipt does
not set `pilot_ready`, `main_study_ready`, or formal scientific authorization.

## MPS checks

Check what the current process can see:

```bash
.venv/bin/python scripts/check_environment.py
```

An explicitly non-scientific MPS smoke can validate basic device mechanics on
the ordinary Mac Terminal:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/train_candidate.py \
  --candidate common/initial_candidate.py \
  --profile smoke_train_v1 \
  --device mps \
  --seed 1 \
  --output-dir /private/tmp/architecture-training-mps-smoke
```

Do not interpret that smoke as `full_train_v1` validation. The full profile
must remain blocked until trusted IR execution or real candidate-bound OS
containment is in place. Once that gate is implemented, an ordinary Terminal
full-profile validation uses:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/train_candidate.py \
  --candidate common/initial_candidate.py \
  --profile full_train_v1 \
  --device mps \
  --seed 1 \
  --output-dir outputs/readiness/full_train_v1_seed_1
```

At present this command should fail at the containment gate before training;
that is correct behavior.

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

## NVIDIA A40 CUDA smoke

Never run candidate training on `login1`. The inspected cluster uses Slurm and
advertises A40 resources on partition `gpu` with `--gres=gpu:a40:1`. If account
or QOS flags are required, use only values authorized for the project; none are
invented in the template.

Interactive allocation:

```bash
salloc --partition=gpu --gres=gpu:a40:1 --nodes=1 --ntasks=1 --time=00:20:00
srun --ntasks=1 --gres=gpu:a40:1 scripts/slurm_cuda_smoke.sh
```

Parameterized batch submission:

```bash
sbatch scripts/slurm_cuda_smoke.sbatch
# If required by the scheduler:
sbatch --account=<approved-account> --qos=<approved-qos> \
  scripts/slurm_cuda_smoke.sbatch
```

The launcher refuses the login host and missing/non-single-GPU allocations,
uses the pinned Python 3.12/PyTorch 2.7.1 environment, clears provider
credential variables, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and writes only
to `/scratch/maandrew-rl4rl-runs`. It does not make API calls or generate a
candidate.

After a successful run, validate the hash-linked receipt without retraining:

```bash
/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python \
  scripts/record_cuda_a40_validation.py \
  --training-output-dir /scratch/maandrew-rl4rl-runs/<run>/training \
  --output /scratch/maandrew-rl4rl-runs/<run>/cuda_a40_validation_receipt.json \
  --expected-seed 1
```

The recorder requires an A40, Slurm allocation evidence, exactly ten completed
steps, validation, changed parameters, synchronized timing, non-null CUDA
memory telemetry, strict deterministic settings, the trusted checked-in source,
and matching checkpoint/artifact hashes. It also rejects credential marker
names in retained text artifacts. See `CUDA_A40_MIGRATION_RUNBOOK.md` for the
server evidence and reproducibility procedure.

## Scientific launch sequence

Do not export provider credentials until the readiness audit is otherwise
clean. The required order is:

1. Resolve every null in `scientific_decisions.yaml`, complete its PI approval
   record, and change its status to `approved`; placeholders and empty values
   fail the audit.
2. Populate matching manifest values and freeze an executable
   `study/scientific_study.json` bound to the manifest hash.
3. Implement and validate the trusted IR interpreter/runtime-evidence path, or
   produce a real candidate-bound OS containment attestation.
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

It exits before provider initialization while any required gate is open.

## API environment, only after readiness

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
separate. Worker environments omit provider credentials.

## Secondary native-controller checks

The legacy native entrypoints remain useful only after the primary causal
system is configured, and their results are secondary replications:

```bash
.venv/bin/python agents/greedy_autoresearch/run.py --iterations 5 --seed 1
.venv/bin/python agents/openevolve_generic/run.py --iterations 5 --seed 1
.venv/bin/python agents/openevolve_semantic/run.py --iterations 5 --seed 1
```

Run them one at a time. They make paid model calls and candidate-training
attempts, so they are not part of offline readiness validation.
