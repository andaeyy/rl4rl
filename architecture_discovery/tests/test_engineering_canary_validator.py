from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch
import yaml

from architecture_ir import load_and_build_ir_candidate, validate_ir_candidate_json
from common.training_config import SMOKE_TRAIN_V1
from scripts.validate_engineering_canaries import (
    DeterministicFakeProvider,
    HARNESSES,
    MAX_FAKE_RESPONSE_BYTES,
    build_report,
    validate_controller_surfaces,
    validate_existing_mps_smoke,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_fixture(tmp_path: Path, *, omit: str | None = None) -> Path:
    project = tmp_path / "project"
    (project / "common").mkdir(parents=True)
    shutil.copy2(
        ROOT / "common" / "initial_candidate.ir.json",
        project / "common" / "initial_candidate.ir.json",
    )
    (project / "common" / "openevolve_runner.py").write_text(
        """import argparse

def run_controller(kind):
    parser = argparse.ArgumentParser()
    parser.add_argument('--engineering-pilot', action='store_true')
    parser.add_argument('--iterations')
    parser.add_argument('--seed')
    parser.add_argument('--output-dir')
    parser.add_argument('--training-profile')
    parser.add_argument('--evaluation-profile')
    parser.add_argument('--evaluation-cases')
    parser.add_argument('--device')
    return parser.parse_args()
""",
        encoding="utf-8",
    )
    for spec in HARNESSES:
        if spec.harness_id == omit:
            continue
        agent = project / "agents" / spec.agent_directory
        agent.mkdir(parents=True)
        if spec.delegated_controller_kind is None:
            entrypoint = """import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--engineering-pilot', action='store_true')
parser.add_argument('--iterations')
parser.add_argument('--seed')
parser.add_argument('--output-dir')
parser.add_argument('--training-profile')
parser.add_argument('--evaluation-profile')
parser.add_argument('--evaluation-cases')
parser.add_argument('--device')
if __name__ == '__main__':
    parser.parse_args()
"""
        else:
            entrypoint = f"""from common.openevolve_runner import run_controller

if __name__ == '__main__':
    run_controller({spec.delegated_controller_kind!r})
"""
        (agent / "run.py").write_text(entrypoint, encoding="utf-8")
        (agent / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "condition": spec.config_condition or spec.harness_id,
                    "training": {
                        "profile": "full_train_v1",
                        "profile_version": "1",
                        "device": "mps",
                        "allow_cpu_for_tests": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        (agent / "program.md").write_text(
            f"# {spec.display_name}\n\nFixture prompt.\n",
            encoding="utf-8",
        )
    return project


def _synthetic_mps_smoke(project: Path, tmp_path: Path) -> Path:
    output = tmp_path / "mps-smoke"
    output.mkdir()
    candidate = project / "common" / "initial_candidate.ir.json"
    stored_candidate = output / "candidate_graph.json"
    shutil.copy2(candidate, stored_candidate)

    initialization_seed = 17
    interpreted = load_and_build_ir_candidate(candidate, initialization_seed)
    model = interpreted.model
    trained_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    changed_name = next(
        name for name, value in trained_state.items() if value.is_floating_point()
    )
    trained_state[changed_name] = trained_state[changed_name] + 0.001
    checkpoint = output / "best_checkpoint.pt"
    torch.save({"model_state": trained_state}, checkpoint)

    events = output / "training_events.jsonl"
    events.write_text(
        "".join(
            json.dumps(
                {
                    "optimizer_step": step,
                    "loss": 1.0 / step,
                },
                sort_keys=True,
            )
            + "\n"
            for step in range(1, SMOKE_TRAIN_V1.max_steps + 1)
        ),
        encoding="utf-8",
    )
    candidate_hash = _sha256(candidate)
    candidate_validation = validate_ir_candidate_json(
        candidate.read_text(encoding="utf-8")
    )
    assert candidate_validation.valid
    manifest = {
        "allow_cpu_for_tests": False,
        "hardware_matched_scientific_run": False,
        "candidate_source_hash": candidate_hash,
        "candidate_artifact_hash": candidate_hash,
        "candidate_format": "architecture_ir",
        "candidate_graph_hash": candidate_validation.graph_hash,
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "requested_device": "mps",
        "selected_device": "mps",
        "parameter_count_role": "descriptive_metadata_only",
        "isolation_level": "engineering_only_or_scientific_gate_blocked",
        "runtime": {
            "mps_built": True,
            "mps_available": True,
            "pytorch_enable_mps_fallback": "0",
        },
        "containment_decision": {"allowed": True, "scientific": False},
        "containment_audit": {"visible_credential_names": []},
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    summary = {
        "success": True,
        "scientific": False,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
        "profile_name": SMOKE_TRAIN_V1.name,
        "profile_version": SMOKE_TRAIN_V1.version,
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "candidate_source_hash": candidate_hash,
        "device": "mps",
        "dtype": "float32",
        "steps_completed": SMOKE_TRAIN_V1.max_steps,
        "examples_processed": (
            SMOKE_TRAIN_V1.max_steps * SMOKE_TRAIN_V1.global_batch_size
        ),
        "best_development_loss": 0.1,
        "final_training_loss": 0.1,
        "train_seconds": 1.0,
        "initialization_seed": initialization_seed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "event_log_path": str(events),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return output


def test_static_controller_surfaces_are_four_harness_and_nonexecuting(
    tmp_path,
):
    project = _project_fixture(tmp_path)
    report = validate_controller_surfaces(project)

    assert report["passed"]
    assert report["real_provider_calls"] == 0
    assert report["local_fixture_calls"] == 4
    assert report["entrypoint_execution_runs"] == 0
    assert report["candidate_execution_runs"] == 0
    assert report["training_runs"] == 0
    assert [item["harness_id"] for item in report["harnesses"]] == [
        "normal_autoresearch",
        "semantic_autoresearch",
        "openevolve",
        "semantic_openevolve",
    ]
    assert report["candidate_format"] == "architecture_ir"
    assert all(
        item["candidate_executable_structure_unchanged"]
        for item in report["harnesses"]
    )
    assert all(item["candidate_graph_hash_changed"] for item in report["harnesses"])
    assert all(item["candidate_ir_valid"] for item in report["harnesses"])
    assert all(
        item["fixed_response_format"] == "complete_architecture_ir_json"
        for item in report["harnesses"]
    )
    assert all(
        0 < item["fixed_response_bytes"] <= MAX_FAKE_RESPONSE_BYTES
        for item in report["harnesses"]
    )
    assert all(item["static_cli_contract_passed"] for item in report["harnesses"])
    assert not any(item["entrypoint_executed"] for item in report["harnesses"])
    assert not any(item["candidate_executed"] for item in report["harnesses"])


def test_fake_provider_rejects_an_unbounded_response_before_ir_decoding():
    with pytest.raises(ValueError, match="byte limit"):
        DeterministicFakeProvider("x" * (MAX_FAKE_RESPONSE_BYTES + 1))


def test_static_controller_surfaces_fail_when_one_named_harness_is_missing(tmp_path):
    project = _project_fixture(tmp_path, omit="semantic_autoresearch")
    report = validate_controller_surfaces(project)

    assert not report["passed"]
    semantic = next(
        item
        for item in report["harnesses"]
        if item["harness_id"] == "semantic_autoresearch"
    )
    assert not semantic["passed"]
    assert any("missing entrypoint" in error for error in semantic["errors"])


def test_static_controller_surface_requires_explicit_engineering_pilot_flag(tmp_path):
    project = _project_fixture(tmp_path)
    entrypoint = project / "agents" / "greedy_autoresearch" / "run.py"
    entrypoint.write_text(
        entrypoint.read_text(encoding="utf-8").replace(
            "parser.add_argument('--engineering-pilot', action='store_true')\n",
            "",
        ),
        encoding="utf-8",
    )

    report = validate_controller_surfaces(project)

    assert not report["passed"]
    greedy = next(
        item for item in report["harnesses"] if item["harness_id"] == "normal_autoresearch"
    )
    assert not greedy["static_cli_contract_passed"]
    assert any("--engineering-pilot" in error for error in greedy["errors"])


def test_static_surface_inspection_never_executes_entrypoint_code(tmp_path):
    project = _project_fixture(tmp_path)
    marker = project / "entrypoint-executed.txt"
    entrypoint = project / "agents" / "greedy_autoresearch" / "run.py"
    entrypoint.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('unsafe')\n"
        + entrypoint.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = validate_controller_surfaces(project)

    assert report["passed"]
    assert not marker.exists()
    assert report["entrypoint_execution_runs"] == 0


def test_report_never_claims_scientific_or_generated_candidate_readiness(tmp_path):
    project = _project_fixture(tmp_path)
    report = build_report(project_root=project)

    assert report["static_controller_surfaces_passed"]
    assert report["status"] == "static_controller_surfaces_passed"
    assert report["provider_calls"] == 0
    assert report["training_runs"] == 0
    assert report["scientific"] is False
    assert report["scientific_pilot_ready"] is False
    assert report["autonomous_generated_candidate_execution_ready"] is False
    assert report["mps_smoke_artifacts_self_consistent"] is False
    assert report["mps_execution_origin_attested"] is False


def test_existing_smoke_artifacts_are_consistent_without_attesting_execution(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert evidence["valid"]
    assert evidence["artifact_self_consistent"]
    assert evidence["execution_origin_attested"] is False
    assert evidence["claim_scope"] == "self_authored_artifact_consistency_only"
    assert evidence["training_started_by_validator"] is False
    assert evidence["parameters_changed"]
    assert evidence["scientific"] is False


def test_existing_smoke_accepts_controller_canonicalized_trusted_seed(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    graph_path = output / "candidate_graph.json"
    validation = validate_ir_candidate_json(graph_path.read_text(encoding="utf-8"))
    assert validation.valid and validation.graph is not None
    graph_path.write_text(validation.graph.canonical_json, encoding="utf-8")
    canonical_hash = _sha256(graph_path)

    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["candidate_source_hash"] = canonical_hash
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    manifest_path = output / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_source_hash"] = canonical_hash
    manifest["candidate_artifact_hash"] = canonical_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert evidence["valid"]
    assert evidence["artifact_self_consistent"]
    assert evidence["parameters_changed"]


def test_smoke_artifact_check_rejects_wrong_checkpoint_shape(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    checkpoint_path = output / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    name = next(iter(checkpoint["model_state"]))
    checkpoint["model_state"][name] = checkpoint["model_state"][name].reshape(-1)
    torch.save(checkpoint, checkpoint_path)
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_sha256"] = _sha256(checkpoint_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert any("wrong shape" in error for error in evidence["errors"])


def test_mps_smoke_rejects_a_candidate_other_than_the_trusted_seed(tmp_path):
    project = _project_fixture(tmp_path)
    output = _synthetic_mps_smoke(project, tmp_path)
    graph_path = output / "candidate_graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["metadata"]["untrusted_change"] = True
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = validate_existing_mps_smoke(output, project_root=project)

    assert not evidence["valid"]
    assert not evidence["artifact_self_consistent"]
    assert any("trusted initial candidate" in error for error in evidence["errors"])
