#!/bin/bash
# Default cluster spelling was inspected read-only on 2026-08-03. The request
# below is exactly one A40. Override partition/GRES only with scheduler-confirmed
# values. Account and QOS are deliberately unset; when required, supply them at
# submission time instead of editing this file:
#
#   sbatch --account=<approved-account> --qos=<approved-qos> \
#     scripts/slurm_cuda_smoke.sbatch
#
# Optional non-secret overrides can be supplied explicitly:
#   sbatch --export=CUDA_SMOKE_OUTPUT_DIR=/scratch/maandrew-rl4rl-runs/<fresh-name> \
#     scripts/slurm_cuda_smoke.sbatch

#SBATCH --job-name=architecture-cuda-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/maandrew-rl4rl-runs/slurm-cuda-smoke-%j.out
#SBATCH --error=/scratch/maandrew-rl4rl-runs/slurm-cuda-smoke-%j.err
#SBATCH --export=NONE

set -Eeuo pipefail
umask 077

# Slurm opens these streams before this wrapper starts. Tighten them before
# any handoff so even a pre-launch failure does not leave group-readable logs.
chmod 600 \
    "/scratch/maandrew-rl4rl-runs/slurm-cuda-smoke-${SLURM_JOB_ID}.out" \
    "/scratch/maandrew-rl4rl-runs/slurm-cuda-smoke-${SLURM_JOB_ID}.err" \
    2>/dev/null || true

# Slurm executes a controller-spooled copy of this file, whose source directory
# is under /var/spool rather than the repository containing the companion script.
exec /bin/bash "/scratch/maandrew-rl4rl/architecture_discovery/scripts/slurm_cuda_smoke.sh"
