#!/bin/bash
# Run only inside an allocated Slurm GPU step. This script never submits work
# itself and therefore cannot accidentally train on a login node.
#
# Interactive example (append only an account/QOS authorized for your project):
#   salloc --partition=gpu --gres=gpu:a40:1 --nodes=1 --ntasks=1 --time=00:20:00
#   srun --ntasks=1 --gres=gpu:a40:1 scripts/slurm_cuda_smoke.sh
#
# Batch example:
#   sbatch scripts/slurm_cuda_smoke.sbatch

set -Eeuo pipefail
umask 077

PROJECT_ROOT="/scratch/maandrew-rl4rl/architecture_discovery"
RUNS_ROOT="/scratch/maandrew-rl4rl-runs"
PYTHON_BIN="${DISCOVERY_PYTHON:-/scratch/maandrew-rl4rl/env/architecture-discovery-py312-torch271/bin/python}"
PROFILE="smoke_train_cuda_v1"
SEED="1"
CANDIDATE="${PROJECT_ROOT}/common/initial_candidate.py"

short_hostname="$(hostname -s)"
if [[ "${short_hostname}" == "login1" ]]; then
    echo "refusing CUDA smoke training on login1" >&2
    exit 2
fi
if [[ "${short_hostname}" == login* || "${short_hostname}" == head* ]]; then
    echo "refusing CUDA smoke training on a login/head node: ${short_hostname}" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "CUDA smoke requires an active Slurm allocation" >&2
    exit 2
fi
chmod 600 \
    "${RUNS_ROOT}/slurm-cuda-smoke-${SLURM_JOB_ID}.out" \
    "${RUNS_ROOT}/slurm-cuda-smoke-${SLURM_JOB_ID}.err" \
    2>/dev/null || true
if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${SLURM_GPUS_ON_NODE:-}" ]]; then
    echo "Slurm job has no GPU-allocation marker" >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose exactly one allocated GPU" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "pinned Python environment is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -f "${CANDIDATE}" ]]; then
    echo "trusted checked-in candidate is unavailable" >&2
    exit 2
fi

resolved_runs_root="$(realpath -m "${RUNS_ROOT}")"
default_run_dir="${resolved_runs_root}/cuda-smoke-seed-1-job-${SLURM_JOB_ID}"
run_dir="$(realpath -m "${CUDA_SMOKE_OUTPUT_DIR:-${default_run_dir}}")"
case "${run_dir}" in
    "${resolved_runs_root}"/*) ;;
    *)
        echo "CUDA smoke output must remain under ${resolved_runs_root}" >&2
        exit 2
        ;;
esac
if [[ -e "${run_dir}" ]]; then
    echo "fresh CUDA smoke output already exists: ${run_dir}" >&2
    exit 2
fi

mkdir -p "${run_dir}"
training_dir="${run_dir}/training"
environment_report="${run_dir}/environment.json"
training_stdout="${run_dir}/training.stdout.json"
training_stderr="${run_dir}/training.stderr.log"
current_stage="launcher_initialization"

record_overall_exit() {
    status=$?
    printf '%s\n' "${status}" >"${run_dir}/overall_exit_code.txt"
    printf '%s\n' "${current_stage}" >"${run_dir}/final_stage.txt"
}
trap record_overall_exit EXIT

printf '%s\n' "${SLURM_JOB_ID}" >"${run_dir}/slurm_job_id.txt"
printf '%s\n' "${short_hostname}" >"${run_dir}/hostname.txt"

# The trusted machinery smoke requires no provider configuration. Remove it
# before either the checker or the credential-scrubbed training client starts.
unset DISCOVERY_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY
unset GEMINI_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
unset GITHUB_TOKEN GH_TOKEN HF_TOKEN HUGGINGFACE_TOKEN
unset DISCOVERY_API_BASE DISCOVERY_MODEL DISCOVERY_REASONING_EFFORT
while IFS='=' read -r variable_name _value; do
    upper_name="${variable_name^^}"
    case "${upper_name}" in
        *API_KEY*|*ACCESS_TOKEN*|*AUTH_TOKEN*|*PASSWORD*|*PRIVATE_KEY*|*SECRET*)
            unset "${variable_name}"
            ;;
    esac
done < <(env)

export DISCOVERY_TRAIN_DEVICE="cuda"
export DISCOVERY_TRAINING_PROFILE="${PROFILE}"
export DISCOVERY_ALLOW_CPU_TRAINING="0"
export PYTORCH_ENABLE_MPS_FALLBACK="0"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

cd "${PROJECT_ROOT}"
command=(
    "${PYTHON_BIN}"
    "scripts/train_candidate.py"
    "--candidate"
    "${CANDIDATE}"
    "--profile"
    "${PROFILE}"
    "--seed"
    "${SEED}"
    "--device"
    "cuda"
    "--output-dir"
    "${training_dir}"
)
printf '%q ' "${command[@]}" >"${run_dir}/command.txt"
printf '\n' >>"${run_dir}/command.txt"

checker_command=(
    "${PYTHON_BIN}"
    "scripts/check_environment.py"
    "--device"
    "cuda"
    "--profile"
    "${PROFILE}"
    "--require-ready"
    "--require-a40"
)
printf '%q ' "${checker_command[@]}" >"${run_dir}/checker_command.txt"
printf '\n' >>"${run_dir}/checker_command.txt"

current_stage="scheduler_preflight"
scontrol show job "${SLURM_JOB_ID}" -o >"${run_dir}/scheduler_job.txt"
"${PYTHON_BIN}" - "${run_dir}/scheduler_job.txt" "${SLURM_JOB_ID}" \
    "${short_hostname}" "${CANDIDATE}" >"${run_dir}/scheduler_preflight.json" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from common.cuda_contract import slurm_tres_has_exact_count
from common.trusted_candidate import validate_trusted_initial_candidate

job_report = Path(sys.argv[1]).read_text(encoding="utf-8").strip()
expected_job = sys.argv[2]
hostname = sys.argv[3]
candidate = Path(sys.argv[4])
fields = {
    key: value
    for item in shlex.split(job_report)
    if "=" in item
    for key, value in (item.split("=", 1),)
}
allocated_tres = fields.get("AllocTRES", "")
requested_tres = fields.get("ReqTRES", "")
checks = {
    "job_id_matches": fields.get("JobId") == expected_job,
    "job_running": fields.get("JobState") == "RUNNING",
    "current_host_allocated": hostname in fields.get("NodeList", "").split(","),
    "exactly_one_gpu_allocated": slurm_tres_has_exact_count(
        allocated_tres, "gres/gpu", 1
    ),
    "one_a40_allocated": slurm_tres_has_exact_count(
        allocated_tres, "gres/gpu:a40", 1
    ),
    "one_a40_requested": slurm_tres_has_exact_count(
        requested_tres, "gres/gpu:a40", 1
    ),
}
candidate_sha256 = validate_trusted_initial_candidate(candidate)
result = {
    "schema_version": "cuda_scheduler_preflight_v1",
    "passed": all(checks.values()),
    "checks": checks,
    "trusted_candidate_sha256": candidate_sha256,
}
print(json.dumps(result, indent=2, sort_keys=True))
if not result["passed"]:
    raise SystemExit(2)
PY

current_stage="environment_preflight"
set +e
"${checker_command[@]}" >"${environment_report}" 2>"${run_dir}/checker.stderr.log"
checker_exit_code=$?
set -e
printf '%s\n' "${checker_exit_code}" >"${run_dir}/checker_exit_code.txt"
if [[ "${checker_exit_code}" -ne 0 ]]; then
    echo "CUDA environment preflight failed; retained evidence at ${run_dir}" >&2
    exit "${checker_exit_code}"
fi

current_stage="training"
set +e
"${command[@]}" >"${training_stdout}" 2>"${training_stderr}"
training_exit_code=$?
set -e
printf '%s\n' "${training_exit_code}" >"${run_dir}/training_exit_code.txt"
sed -n '1,240p' "${training_stdout}"
if [[ -s "${training_stderr}" ]]; then
    sed -n '1,240p' "${training_stderr}" >&2
fi
if [[ "${training_exit_code}" -ne 0 ]]; then
    echo "CUDA smoke training failed; retained evidence at ${run_dir}" >&2
    exit "${training_exit_code}"
fi

current_stage="acceptance_validation"
set +e
"${PYTHON_BIN}" - "${training_stdout}" "${training_dir}" "${CANDIDATE}" \
    >"${run_dir}/smoke_acceptance.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
training_dir = Path(sys.argv[2]).resolve()
candidate = Path(sys.argv[3]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
manifest = json.loads(
    (training_dir / "training_manifest.json").read_text(encoding="utf-8")
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks = {
    "training_success": summary.get("success") is True,
    "exactly_ten_optimizer_steps": summary.get("steps_completed") == 10,
    "examples_processed_positive": int(summary.get("examples_processed", 0)) > 0,
    "parameters_changed": summary.get("parameters_changed") is True,
    "requested_device_cuda": manifest.get("requested_device") == "cuda",
    "selected_device_cuda": manifest.get("selected_device") == "cuda:0",
    "summary_device_cuda": summary.get("device") == "cuda:0",
    "no_cpu_fallback": summary.get("unsupported_operation_fallback") is False,
    "timing_synchronized": summary.get("timing_synchronized") is True,
    "validation_completed": int(summary.get("best_development_step", -1)) == 10,
    "profile_hash_matches": (
        summary.get("profile_hash") == manifest.get("profile_hash")
        and summary.get("profile_name") == "smoke_train_cuda_v1"
    ),
    "source_hash_matches": (
        summary.get("candidate_source_hash") == sha256(candidate)
        and manifest.get("candidate_source_hash") == sha256(candidate)
    ),
}

checkpoint = Path(str(summary.get("checkpoint_path", "")))
checks["checkpoint_created"] = checkpoint.is_file()
checks["checkpoint_hash_valid"] = bool(
    checkpoint.is_file()
    and summary.get("checkpoint_sha256")
    and sha256(checkpoint) == summary.get("checkpoint_sha256")
)
telemetry = summary.get("accelerator_telemetry") or {}
memory = telemetry.get("memory") or {}
checks["cuda_memory_telemetry"] = bool(
    telemetry.get("backend") == "cuda"
    and memory.get("allocated_bytes") is not None
    and memory.get("reserved_bytes") is not None
    and memory.get("peak_allocated_bytes") is not None
)

forbidden_markers = (
    b"DISCOVERY_API_KEY",
    b"OPENAI_API_KEY",
    b"ANTHROPIC_API_KEY",
    b"GOOGLE_API_KEY",
    b"GEMINI_API_KEY",
    b"AWS_SECRET_ACCESS_KEY",
    b"GITHUB_TOKEN",
    b"HF_TOKEN",
)
credential_clean = True
for artifact in training_dir.parent.rglob("*"):
    if not artifact.is_file() or artifact.stat().st_size > 20_000_000:
        continue
    payload = artifact.read_bytes()
    if any(marker in payload for marker in forbidden_markers):
        credential_clean = False
        break
checks["credential_markers_absent"] = credential_clean

result = {
    "schema_version": "cuda_smoke_acceptance_v1",
    "passed": all(checks.values()),
    "checks": checks,
}
print(json.dumps(result, indent=2, sort_keys=True))
if not result["passed"]:
    raise SystemExit(3)
PY
acceptance_exit_code=$?
set -e
printf '%s\n' "${acceptance_exit_code}" >"${run_dir}/acceptance_exit_code.txt"
if [[ "${acceptance_exit_code}" -ne 0 ]]; then
    echo "CUDA smoke acceptance validation failed; evidence at ${run_dir}" >&2
    exit "${acceptance_exit_code}"
fi

current_stage="receipt_validation"
set +e
"${PYTHON_BIN}" scripts/record_cuda_a40_validation.py \
    --training-output-dir "${training_dir}" \
    --output "${run_dir}/cuda_a40_validation_receipt.json" \
    --expected-seed 1 >"${run_dir}/receipt_recording.stdout.json"
receipt_exit_code=$?
set -e
printf '%s\n' "${receipt_exit_code}" >"${run_dir}/receipt_exit_code.txt"
if [[ "${receipt_exit_code}" -ne 0 ]]; then
    echo "CUDA smoke receipt validation failed; evidence at ${run_dir}" >&2
    exit "${receipt_exit_code}"
fi

sha256sum "${training_dir}/best_checkpoint.pt" \
    >"${run_dir}/best_checkpoint.sha256"
current_stage="complete"
echo "CUDA smoke acceptance passed; retained evidence at ${run_dir}"
