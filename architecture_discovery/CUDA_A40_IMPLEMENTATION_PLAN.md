# CUDA A40 migration: implementation plan

This document records the lead-agent baseline and non-overlapping ownership
plan before CUDA implementation began. It is an engineering record, not PI
authorization and not a scientific-readiness attestation.

## Pre-edit baseline

- Recorded on 2026-08-03 on host `login1`.
- Starting Git commit: `91186cf663799e1d5b4948cc0bc1c6f1f5fbfdb2`.
- Starting branch: `codex/architecture-discovery-infrastructure`.
- Migration branch created before edits: `codex/cuda-a40-migration`.
- Working tree before branch creation: clean (`git status --short` emitted no
  paths; the starting branch was even with its upstream).
- Applicable `AGENTS.md`: none found under `/scratch/maandrew-rl4rl` or the
  project subtree.
- Frozen `FULL_TRAIN_V1.profile_hash`:
  `046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e`.
- Frozen `SMOKE_TRAIN_V1.profile_hash`:
  `1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61`.
- Project requirement: Python `>=3.12,<3.13`; lock requirement Python
  `==3.12.*`; PyTorch `==2.7.1`; pytest `==8.4.1`.
- The checked-in `architecture_discovery/.venv` is the Mac environment. Its
  interpreter symlink targets `/opt/homebrew/opt/python@3.12/bin/python3.12`
  and is intentionally not modified on this Linux server.
- Initial Linux test attempt used the only pre-existing usable environment:
  Python 3.10.18, PyTorch 2.9.0+cu128, pytest 9.1.1. Configuration validation
  and pytest collection both exited nonzero before tests ran because Python
  3.10 lacks `enum.StrEnum` and `datetime.UTC`. This is an environment
  mismatch, not a repository-test regression.
- Slurm environment job `6499601` completed successfully on `compute127` and
  created a separate lockfile-driven environment at
  `/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271` with
  Python 3.12.12, PyTorch 2.7.1+cu126, CUDA runtime 12.6, cuDNN 9.5.1, and
  pytest 8.4.1.
- Pinned pre-edit baseline job `6499602` ran on `compute127` and exited `0:0`.
  `scripts/validate_configs.py` reported `configuration invariants: PASS`, and
  the complete pytest suite reached 100% (285 tests) with no failures. Its
  retained stdout/stderr are `/scratch/maandrew-rl4rl-runs/baseline-6499602.out`
  and `/scratch/maandrew-rl4rl-runs/baseline-6499602.err`.

## Cluster and isolation observations

- Login hostname: `login1`. Candidate training is forbidden there.
- Slurm 25.05.8 is installed. `select/cons_tres`, `task/cgroup`, and
  `ConstrainDevices=yes` are configured.
- Partition `gpu` permits all accounts/QOS and advertises four A40 nodes with
  `gpu:a40:4`; the observed request spelling is `--gres=gpu:a40:1`.
- A40 node `gpu01` reports 64 CPUs, approximately 1 TB RAM, and four A40 GRES.
- CUDA modules 11.8 and 12.4 are available.
- Apptainer 1.4.1 is available as a module; its configuration reports setuid
  and user namespaces enabled. No candidate-bound, authenticated containment
  receipt or adversarial isolation proof exists, so arbitrary generated Python
  must remain blocked from formal scientific execution.
- `/scratch/maandrew-rl4rl-runs` was created as a separate 30-day scratch
  workspace for server outputs outside the Git checkout.
- No `nvidia-smi` command was used on the login node as GPU proof.

## Non-overlapping creator ownership

Creator A owns CUDA device/profile/trainer runtime only:

- `common/device.py`
- `common/training_config.py`
- `common/trainer.py`
- `tests/test_cuda_device_runtime.py` (new)
- `tests/test_cuda_training_profiles.py` (new)
- `tests/test_cuda_trainer_runtime.py` (new)

Creator B owns worker/CLI/native harness/server launch only:

- `common/training_client.py`
- `common/training_worker.py`
- `common/evaluator.py`
- `common/openevolve_runner.py`
- `agents/greedy_autoresearch/run.py`
- all three `agents/*/config.yaml` training sections
- `scripts/train_candidate.py`
- `scripts/retrain_candidate.py`
- `scripts/check_environment.py`
- `scripts/slurm_cuda_smoke.sh` (new)
- `scripts/slurm_cuda_smoke.sbatch` (new parameterized template)
- `tests/test_cuda_worker_cli.py` (new)
- `tests/test_cuda_harness_selection.py` (new)

Creator C owns policy/evidence/resource schema/documentation only:

- `containment/audit.py`
- `containment/policy.py`
- `study/scheduling.py`
- `study/budget.py`
- `artifacts/failures.py`
- `artifacts/study_sink.py` only if required for the new resource schema
- `scripts/audit_scientific_readiness.py`
- a new CUDA validation-receipt script
- `.env.example`
- `README.md`
- `CUDA_A40_MIGRATION_RUNBOOK.md` (new)
- CUDA policy/evidence/budget/scheduling tests in new, uniquely named test
  files plus directly corresponding existing policy tests where necessary

`CUDA_A40_IMPLEMENTATION_PLAN.md`, `experiment_manifest.yaml`,
`IMPLEMENTATION_CONTRACT.md`, and any cross-owner integration edits remain
lead-owned. If a creator needs another owner's file, it must report the need
instead of editing that file.

## Integration invariants

1. `FULL_TRAIN_V1` and `SMOKE_TRAIN_V1` remain byte-for-byte equivalent at
   serialization and retain their hashes.
2. CUDA profiles are additive, separately named, and hash-distinct.
3. CUDA smoke is exactly 10 steps, non-scientific, trusted-candidate-only, and
   provider-free.
4. No CPU fallback, mixed precision, automatic batch-size reduction, TF32, or
   `torch.compile` is introduced.
5. CUDA and MPS records remain separate hardware conditions; v1 MPS readers
   continue to work.
6. Worker environments remain explicit allowlists and never contain provider
   credentials or credential names.
7. Scientific arbitrary-Python execution stays fail-closed without genuine
   strong-containment evidence.
8. Parameter count remains descriptive metadata and never influences ranking.
