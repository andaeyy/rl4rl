from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agents.greedy_autoresearch.run import build_parser as greedy_parser
from common.openevolve_runner import build_parser as openevolve_parser
from common.training_client import resolve_harness_training_selection


ROOT = Path(__file__).resolve().parents[1]
HARNESS_NAMES = (
    "greedy_autoresearch",
    "openevolve_generic",
    "openevolve_semantic",
)


def _training_config(name: str = "greedy_autoresearch") -> dict:
    return yaml.safe_load(
        (ROOT / "agents" / name / "config.yaml").read_text(encoding="utf-8")
    )["training"]


def test_all_native_harnesses_keep_mps_defaults_and_advertise_cuda() -> None:
    configurations = [_training_config(name) for name in HARNESS_NAMES]
    assert configurations[0] == configurations[1] == configurations[2]
    training = configurations[0]
    assert training["profile"] == "full_train_v1"
    assert training["profile_version"] == "1"
    assert training["device"] == "mps"
    assert training["allow_cpu_for_tests"] is False
    assert set(training["selectable_profiles"]) == {
        "full_train_v1",
        "smoke_train_v1",
        "full_train_cuda_a40_v1",
        "smoke_train_cuda_v1",
    }
    assert training["selectable_devices"] == ["mps", "cuda"]


def test_harness_selection_is_cli_then_environment_then_configuration() -> None:
    training = _training_config()
    default = resolve_harness_training_selection(
        training,
        profile_override=None,
        device_override=None,
        environment={},
    )
    assert default.profile.name == "full_train_v1"
    assert default.requested_device == "mps"
    assert default.profile_source == "configuration"
    assert default.device_source == "configuration"

    environment = {
        "DISCOVERY_TRAINING_PROFILE": "smoke_train_cuda_v1",
        "DISCOVERY_TRAIN_DEVICE": "cuda",
    }
    selected = resolve_harness_training_selection(
        training,
        profile_override=None,
        device_override=None,
        environment=environment,
    )
    assert selected.profile.name == "smoke_train_cuda_v1"
    assert selected.requested_device == "cuda"
    assert selected.profile_source == "environment"
    assert selected.device_source == "environment"

    command_line = resolve_harness_training_selection(
        training,
        profile_override="full_train_cuda_a40_v1",
        device_override="cuda",
        environment={
            "DISCOVERY_TRAINING_PROFILE": "full_train_v1",
            "DISCOVERY_TRAIN_DEVICE": "mps",
        },
    )
    assert command_line.profile.name == "full_train_cuda_a40_v1"
    assert command_line.requested_device == "cuda"
    assert command_line.profile_source == "command_line"
    assert command_line.device_source == "command_line"


def test_harness_rejects_cross_backend_profile_substitution() -> None:
    with pytest.raises(ValueError, match="cannot substitute"):
        resolve_harness_training_selection(
            _training_config(),
            profile_override="smoke_train_cuda_v1",
            device_override="mps",
            environment={},
        )


def test_harness_manifest_records_actual_resolved_selection() -> None:
    selection = resolve_harness_training_selection(
        _training_config(),
        profile_override="smoke_train_cuda_v1",
        device_override="cuda",
        environment={},
    )
    manifest = selection.manifest_fields()
    assert manifest["profile"] == "smoke_train_cuda_v1"
    assert manifest["profile_hash"] == selection.profile.profile_hash
    assert manifest["requested_device"] == "cuda"
    assert manifest["device"] == "cuda"
    assert manifest["selection"] == {
        "profile_source": "command_line",
        "device_source": "command_line",
    }


def test_all_three_native_entrypoints_parse_cuda_overrides() -> None:
    greedy = greedy_parser().parse_args(
        ["--profile", "smoke_train_cuda_v1", "--device", "cuda"]
    )
    assert greedy.profile == "smoke_train_cuda_v1"
    assert greedy.device == "cuda"

    for _kind in ("generic", "semantic"):
        openevolve = openevolve_parser().parse_args(
            ["--profile", "smoke_train_cuda_v1", "--device", "cuda"]
        )
        assert openevolve.profile == "smoke_train_cuda_v1"
        assert openevolve.device == "cuda"


def test_slurm_template_requests_one_a40_and_has_no_invented_account_or_qos() -> None:
    template = (ROOT / "scripts" / "slurm_cuda_smoke.sbatch").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "slurm_cuda_smoke.sh").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=gpu" in template
    assert template.count("#SBATCH --gres=gpu:a40:1") == 1
    assert "#SBATCH --export=NONE" in template
    assert "#SBATCH --account=" not in template
    assert "#SBATCH --qos=" not in template
    assert "BASH_SOURCE" not in template
    assert "umask 077" in template
    assert "slurm-cuda-smoke-${SLURM_JOB_ID}.out" in template
    assert template.index("chmod 600") < template.index("exec /bin/bash")
    assert (
        'exec /bin/bash "/scratch/maandrew-rl4rl/architecture_discovery/'
        'scripts/slurm_cuda_smoke.sh"'
    ) in template
    assert "common/initial_candidate.py" in launcher
    assert 'PROFILE="smoke_train_cuda_v1"' in launcher
    assert 'SEED="1"' in launcher
    assert 'DISCOVERY_ALLOW_CPU_TRAINING="0"' in launcher
    assert "/scratch/maandrew-rl4rl-runs" in launcher
    assert "refusing CUDA smoke training on login1" in launcher
