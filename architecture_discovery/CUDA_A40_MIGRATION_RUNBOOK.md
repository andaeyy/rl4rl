# NVIDIA A40 CUDA migration runbook

This is an engineering and research-infrastructure record. It does not fill in
principal-investigator decisions, authorize paid generation, attest arbitrary
Python containment, or claim formal scientific readiness.

## Protocol identities

The CUDA condition is additive. Existing MPS profiles and artifacts retain
their names and semantics.

| Profile | Purpose | Steps | Hardware | Current profile hash |
| --- | --- | ---: | --- | --- |
| `full_train_v1` | frozen scientific MPS protocol | 30,000 | MPS | `046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e` |
| `smoke_train_v1` | engineering MPS smoke | 10 | MPS | `1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61` |
| `full_train_cuda_a40_v1` | full-budget CUDA condition | 30,000 | NVIDIA A40 CUDA | `21d2406108443d48d964c7d216a1b912ebe9a758868027b9f7ccbe3a72df6dc8` |
| `smoke_train_cuda_v1` | non-scientific machinery smoke | exactly 10 | NVIDIA A40 CUDA | `87c1fd454cd0141687b4bf5ffc2cc84835f7a7af532024d1f7fa37f13cd2f948` |

Recheck these identities after every integration change:

```bash
/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python - <<'PY'
from common.training_config import PROFILES
for name in sorted(PROFILES):
    print(name, PROFILES[name].profile_hash)
PY
```

`full_train_cuda_a40_v1` retains the full algorithmic budget: batch 512,
AdamW with the existing learning rate/betas/weight decay, 300-step warmup,
cosine decay, existing validation and checkpoint selection, float32, no mixed
precision, no automatic batch reduction, no CPU fallback, and no
`torch.compile`. Parameter count is descriptive metadata only; it must not
affect ranking, selection, tie-breaking, or stopping.

The CUDA smoke lane also pins `common/initial_candidate.py` to SHA-256
`fee5f0784fbc3bf2c51d450e24064a4a6a988b92658043ce19b40101db294007`.
Another path, or modified bytes at that path, fails before candidate import.

CUDA and MPS are different hardware conditions. Identical seeds do not imply
identical trajectories. Do not pool or compare results across backends without
an explicit analysis plan.

## Inspected server environment

The lead inspection on 2026-08-03 recorded:

- login host `login1`; candidate training is forbidden there;
- Slurm 25.05.8 with `select/cons_tres`, `task/cgroup`, and
  `ConstrainDevices=yes`;
- partition `gpu` advertising four A40 nodes as `gpu:a40:4` and an observed
  one-GPU request spelling of `--gres=gpu:a40:1`;
- CUDA modules 11.8 and 12.4;
- a pinned compute environment at
  `/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271` with
  Python 3.12.12, PyTorch 2.7.1+cu126, CUDA runtime 12.6, cuDNN 9.5.1, and
  pytest 8.4.1;
- output workspace `/scratch/maandrew-rl4rl-runs`, outside the Git checkout.

Apptainer 1.4.1 is module-available with setuid and user namespaces enabled.
Tool discovery is not containment proof. There is no authenticated,
candidate-bound receipt showing that filesystem, network, credentials, child
processes, resources, identity, and sandbox escape tests passed. This remains a
formal scientific-readiness blocker for arbitrary generated Python. Do not
self-attest or weaken the gate. The trusted checked-in candidate may be used for
the non-scientific smoke only.

## Offline checks

Run configuration validation and tests in the pinned Linux environment on a
compute/permitted node. Do not modify the Mac `.venv`.

```bash
PY=/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python
"$PY" scripts/validate_configs.py
"$PY" -m pytest -q
"$PY" scripts/check_environment.py
```

The normal/login environment checker may report CUDA unavailable and no GPU
allocation; that is expected. It must not treat login-node visibility as CUDA
validation and must never fall back to CPU.

Final verification job `6499616` ran in the pinned environment on `compute085`
and completed `0:0` on 2026-08-03. Configuration invariants passed and all 368
tests passed. The ordinary non-GPU checker reported
`cuda_unallocated_no_fallback`; the readiness audit accepted the CUDA receipt
as engineering evidence while remaining fail-closed for pilot/main science.
Its retained stdout/stderr are
`/scratch/maandrew-rl4rl-runs/final-tests-6499616.{out,err}`. A final checker
run on `login1` also reported `cuda_unallocated_no_fallback` and did not call
`nvidia-smi` or perform training.

The three initial read-only reviews reported no critical findings. Every high
finding was fixed: A40 matching is token-exact, deterministic controls are
reasserted around arbitrary candidate execution, the smoke candidate path and
hash are pinned, v2 CUDA accounting is wired to the worker, CUDA failures map
to explicit classes, and the semantic comparator validates each run rather
than merely comparing two mutually wrong inputs. Medium findings were fixed or
kept fail-closed, and the failure-specific fixes received their required
independent read-only reviews.

## Allocate exactly one A40

Interactive:

```bash
salloc --partition=gpu --gres=gpu:a40:1 --nodes=1 --ntasks=1 --time=00:20:00
srun --ntasks=1 --gres=gpu:a40:1 scripts/slurm_cuda_smoke.sh
```

Batch:

```bash
sbatch scripts/slurm_cuda_smoke.sbatch
```

Supply `--account` or `--qos` only when the scheduler requires an authorized
value. The template deliberately contains neither.

Inside the job retain hostname, Slurm job ID, logical GPU index, GPU name/UUID,
total memory, driver, PyTorch CUDA runtime, cuDNN, PyTorch/Python/OS versions,
`CUDA_VISIBLE_DEVICES`, Git commit, and profile hash. The launcher writes its
environment, command, output, error, exit code, acceptance record, and hashes
under `/scratch/maandrew-rl4rl-runs` with private permissions.

## Trusted smoke and receipt

The equivalent direct command inside the allocation is:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python \
  scripts/train_candidate.py \
  --candidate common/initial_candidate.py \
  --profile smoke_train_cuda_v1 \
  --seed 1 \
  --device cuda \
  --output-dir /scratch/maandrew-rl4rl-runs/<fresh-run>/training
```

Do not export a provider key. Do not use another candidate. Do not reduce the
profile below ten steps.

Create and independently revalidate the engineering-only receipt:

```bash
PY=/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python
"$PY" scripts/record_cuda_a40_validation.py \
  --training-output-dir /scratch/maandrew-rl4rl-runs/<fresh-run>/training \
  --output /scratch/maandrew-rl4rl-runs/<fresh-run>/cuda_a40_validation_receipt.json \
  --expected-seed 1
"$PY" scripts/audit_scientific_readiness.py \
  --cuda-a40-evidence /scratch/maandrew-rl4rl-runs/<fresh-run>/cuda_a40_validation_receipt.json
```

The readiness audit still exits 2 while formal gates are unresolved. A passing
`cuda_a40_engineering_validation_receipt` gate is deliberately excluded from
MPS pilot/main authorization and cannot make the study scientifically ready.

## Reproducibility repeat

Run the same trusted command in a second fresh directory with seed 1. Compare
model-state tensors after restricted checkpoint loading, optimizer/global step,
the per-step loss sequence in `training_events.jsonl`, validation metrics, and selected
checkpoint step. Semantic equality is required; archive byte hashes alone are
not sufficient because container metadata can differ.

```bash
PY=/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python
"$PY" scripts/compare_cuda_smoke_runs.py \
  --first /scratch/maandrew-rl4rl-runs/<repeat-a>/training \
  --second /scratch/maandrew-rl4rl-runs/<repeat-b>/training \
  --output /scratch/maandrew-rl4rl-runs/cuda-smoke-reproducibility.json
```

Record actual run evidence here after execution:

| Run | Slurm job | Host/GPU UUID | Output directory | Receipt hash | Result |
| --- | --- | --- | --- | --- | --- |
| seed-1 repeat A | `6499613` (`0:0`) | `gpu01` / `GPU-30c0665d-1160-c247-7d9f-25db14cde51f` | `/scratch/maandrew-rl4rl-runs/cuda-smoke-seed-1-repeat-a-20260803-v2` | `83e0cf36166283a74a3f21674e09b1711dfef61c49067bf55857bbbe4fc25900` | all smoke acceptance checks passed |
| seed-1 repeat B | `6499614` (`0:0`) | `gpu01` / `GPU-30c0665d-1160-c247-7d9f-25db14cde51f` | `/scratch/maandrew-rl4rl-runs/cuda-smoke-seed-1-repeat-b-20260803` | `6d8aa488fb59b2f1b4034bd1db24c1b24a461926fe8a359b06428865dfb9c1af` | all smoke acceptance checks passed |

Semantic comparison job `6499615` completed `0:0`. Every comparison check
passed, including model-state tensors, global and per-parameter optimizer
steps, loss sequence, validation metrics, selected checkpoint step, input
identities, and the preserved MPS profile hash. The retained comparison is
`/scratch/maandrew-rl4rl-runs/cuda-smoke-reproducibility-20260803.json`
(SHA-256 `f37d7e154ae8dc66bcf7b1eaefe4e730d84930834c87fd47964f54c31749fbd8`).
The serialized checkpoint hashes differ while their semantic tensor/state
contents match, which is expected and is why archive hashes were not used as
the reproducibility criterion.

## Failure-driven iteration record

For every failed job retain the exact command, exit code, Slurm job ID, stdout,
stderr, summary, and environment report. Classify it as scheduler/allocation,
environment/dependency, CUDA driver/runtime, deterministic-operation rejection,
code defect, checkpoint/resume, containment, timeout, OOM, or candidate defect.
Record at least two hypotheses before changing code. Use a fresh output
directory after each fix.

| Iteration | Job/command | Classification | Hypotheses | Smallest fix | Tests/review | Result |
| --- | --- | --- | --- | --- | --- | --- |
| targeted review-fix tests | `sbatch /tmp/architecture_cuda_review_tests.sbatch`; job `6499606` on `compute085` | code/test-contract defect | (1) the CUDA runtime configurator assumed the worker had already set `CUBLAS_WORKSPACE_CONFIG`; (2) the generalized login-host guard either made an exact safety assertion stale or lost an intentional `login1` refusal contract | retain cuBLAS as an early child-process precondition and test missing-state rejection; restore explicit `login1` plus general login/head guards; parse scheduler TRES counts as exact tokens | two diagnostic creators, then two read-only fix reviewers; `sbatch /tmp/architecture_cuda_failure_fix_tests.sbatch`, job `6499608`; stdout/stderr `/scratch/maandrew-rl4rl-runs/failure-fix-tests-6499608.{out,err}` | original failed `1:0` with 2 of 89 failures and no training; fix job completed `0:0`, 31/31 tests passed |
| A40 smoke attempt 1 | `sbatch --export=CUDA_SMOKE_OUTPUT_DIR=/scratch/maandrew-rl4rl-runs/cuda-smoke-seed-1-repeat-a-20260803 scripts/slurm_cuda_smoke.sbatch`; job `6499610` on `gpu01` | scheduler/launcher | (1) Slurm executes a spooled wrapper copy, so `BASH_SOURCE[0]` is not the repository script path; (2) `--export=NONE` or submission-directory assumptions removed context needed by a relative handoff | absolute `/bin/bash` and repository launcher handoff; private wrapper umask and stream permissions | two diagnostic creators and two read-only fix reviewers; stdout/stderr `/scratch/maandrew-rl4rl-runs/slurm-cuda-smoke-6499610.{out,err}` | failed `127:0` before environment inspection/import/training; exact one-A40 allocation was retained and the next fresh smoke passed |
| launcher-fix targeted tests | `sbatch /tmp/architecture_cuda_failure_fix_tests.sbatch`; job `6499611` on `compute085` | code/test-contract defect | (1) the test was only partially updated from `exec bash` to `exec /bin/bash`; (2) the wrapper may have accidentally lost its exec handoff | compare ordering against the actual `exec /bin/bash` token | two diagnostic creators and two read-only fix reviewers; rerun job `6499612`; stdout/stderr `/scratch/maandrew-rl4rl-runs/failure-fix-tests-6499612.{out,err}` | original failed `1:0` with 1 of 31 failures and no training; rerun completed `0:0`, 31/31 passed |
| real A40 repeat A | job `6499613`, command recorded in its run directory | none | not applicable | no corrective fix | launcher acceptance plus independently validated CUDA receipt | completed `0:0`; all 16 launcher acceptance checks and all receipt checks passed |
| real A40 repeat B | job `6499614`, same profile/device/seed in a fresh directory | none | not applicable | no corrective fix | launcher acceptance plus independently validated CUDA receipt | completed `0:0`; all acceptance/receipt checks passed; semantic comparator job `6499615` also completed `0:0` |

Never respond to failure by using CPU, disabling determinism, enabling mixed
precision, shrinking the ten-step smoke, reducing full-training budget,
changing the architecture, exposing credentials, or bypassing containment.

## Remaining readiness blockers

At minimum, the following remain until genuine evidence or PI authorization is
provided:

- authenticated candidate-bound strong containment for arbitrary generated
  Python (Apptainer availability alone does not satisfy this);
- unresolved PI decisions and explicit pilot/main launch authorization;
- any other fail-closed gates reported by
  `scripts/audit_scientific_readiness.py`;
- full-profile CUDA evidence if CUDA is ever proposed as a formal scientific
  condition; a ten-step smoke can never substitute for it;
- an explicit analysis plan before any MPS/CUDA comparison or pooling.

## Future one-proposal CUDA canaries

These commands are paid, generated-candidate controller runs. They are examples
for after formal authorization and containment are satisfied; do not run them
during the trusted smoke or this migration without explicit user approval.

```bash
PY=/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python
salloc --partition=gpu --gres=gpu:a40:1 --nodes=1 --ntasks=1 --time=01:00:00

srun --ntasks=1 --gres=gpu:a40:1 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PY" agents/greedy_autoresearch/run.py \
  --iterations 1 --seed 1 --profile full_train_cuda_a40_v1 --device cuda \
  --output-dir /scratch/maandrew-rl4rl-runs/greedy-cuda-canary

srun --ntasks=1 --gres=gpu:a40:1 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PY" agents/openevolve_generic/run.py \
  --iterations 1 --seed 1 --profile full_train_cuda_a40_v1 --device cuda \
  --output-dir /scratch/maandrew-rl4rl-runs/openevolve-generic-cuda-canary

srun --ntasks=1 --gres=gpu:a40:1 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PY" agents/openevolve_semantic/run.py \
  --iterations 1 --seed 1 --profile full_train_cuda_a40_v1 --device cuda \
  --output-dir /scratch/maandrew-rl4rl-runs/openevolve-semantic-cuda-canary
```
