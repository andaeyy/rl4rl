import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.task_adapter import DEFAULT_TASK
from common.trainer import training_manifest, validate_training_request
from common.training_config import (
    FULL_TRAIN_CUDA_V2,
    FULL_TRAIN_V1,
    SEED_DERIVATION_METHOD,
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
    TrainingSeedBundle,
    get_training_profile,
)
from containment.policy import CandidateFormat
from scripts.retrain_candidate import _build_aggregate_report

ROOT = Path(__file__).resolve().parents[1]


def test_all_controllers_resolve_the_same_frozen_training_profile():
    names = (
        "greedy_autoresearch",
        "openevolve_generic",
        "openevolve_semantic",
    )
    references = [
        yaml.safe_load(
            (ROOT / "agents" / name / "config.yaml").read_text()
        )["training"]
        for name in names
    ]
    assert references[0] == references[1] == references[2]
    profile = get_training_profile(references[0]["profile"])
    assert profile.version == references[0]["profile_version"]
    assert references[0]["task_adapter"] == DEFAULT_TASK.version
    assert references[0]["seed_derivation"] == SEED_DERIVATION_METHOD
    assert profile.max_steps == 30_000
    assert profile.device_requirement == "cuda"


def test_seed_bundle_is_condition_independent_and_stable():
    assert TrainingSeedBundle.from_run_seed(1) == TrainingSeedBundle.from_run_seed(1)
    assert (
        TrainingSeedBundle.from_run_seed(1).bundle_hash
        == "20e9cf13691d96bfe09725776965e6dcce315328f6f81b4671b31b31b1b5482f"
    )


def test_training_code_never_reads_official_or_shadow_seeds():
    for name in ("trainer.py", "training_data.py"):
        source = (ROOT / "common" / name).read_text()
        assert "private_eval" not in source
        assert "DISCOVERY_SHADOW_SEED" not in source
        assert "2025" not in source


def test_historical_mps_profile_hashes_are_stable_and_cuda_is_versioned():
    assert (
        FULL_TRAIN_V1.profile_hash
        == "046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e"
    )
    assert (
        SMOKE_TRAIN_V1.profile_hash
        == "1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61"
    )
    assert FULL_TRAIN_CUDA_V2.version == "2"
    assert FULL_TRAIN_CUDA_V2.device_requirement == "cuda"
    assert FULL_TRAIN_CUDA_V2.profile_hash != FULL_TRAIN_V1.profile_hash
    assert SMOKE_TRAIN_CUDA_V2.profile_hash != SMOKE_TRAIN_V1.profile_hash


def test_v2_nonterminal_checkpoints_coincide_with_validation():
    assert (
        SMOKE_TRAIN_CUDA_V2.checkpoint_interval
        % SMOKE_TRAIN_CUDA_V2.validation_interval
        == 0
    )
    invalid = replace(
        SMOKE_TRAIN_CUDA_V2,
        validation_interval=10,
        checkpoint_interval=5,
    )
    with pytest.raises(
        ValueError,
        match="nonterminal checkpoints must coincide with validation",
    ):
        invalid.validate()


def test_v2_preflight_serializes_only_portable_logical_paths(tmp_path):
    candidate = tmp_path / "controller" / "artifacts" / "candidate.ir.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(
        (ROOT / "common" / "initial_candidate.ir.json").read_bytes()
    )
    output = tmp_path / "controller" / "candidate_training" / "0000_seed"

    payload = validate_training_request(
        candidate_path=candidate,
        profile=SMOKE_TRAIN_CUDA_V2,
        seeds=TrainingSeedBundle.from_run_seed(5),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        output_dir=output,
    )

    assert payload["candidate"] == "artifacts/candidate.ir.json"
    assert payload["output_dir"] == "candidate_training/0000_seed"
    assert payload["resume"] is None
    assert not Path(payload["candidate"]).is_absolute()
    assert not Path(payload["output_dir"]).is_absolute()


def test_v1_preflight_retains_historical_absolute_path_shape(tmp_path):
    candidate = ROOT / "common" / "initial_candidate.py"
    output = tmp_path / "legacy-training"
    payload = validate_training_request(
        candidate_path=candidate,
        profile=SMOKE_TRAIN_V1,
        seeds=TrainingSeedBundle.from_run_seed(5),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        output_dir=output,
    )

    assert payload["candidate"] == str(candidate.resolve())
    assert payload["output_dir"] == str(output.resolve())


def test_v2_training_manifest_is_explicit_and_v1_path_shape_is_unchanged():
    seeds = TrainingSeedBundle.from_run_seed(9)
    common = {
        "candidate_hash": "a" * 64,
        "seeds": seeds,
        "requested_device": "cpu",
        "selected_device": "cpu",
        "task": DEFAULT_TASK,
        "allow_cpu_for_tests": True,
        "containment_audit": {},
        "containment_decision": {"allowed": True, "scientific": False},
        "candidate_graph_hash": None,
        "component_hashes": {"test.component": "b" * 64},
        "dependency_lock_hash": "c" * 64,
    }
    legacy_path = ROOT / "common" / "initial_candidate.py"
    legacy = training_manifest(
        candidate_path=legacy_path,
        profile=SMOKE_TRAIN_V1,
        candidate_format=CandidateFormat.ARBITRARY_PYTHON,
        **common,
    )
    portable = training_manifest(
        candidate_path=ROOT / "common" / "initial_candidate.ir.json",
        profile=SMOKE_TRAIN_CUDA_V2,
        candidate_format=CandidateFormat.ARCHITECTURE_IR,
        **common,
    )

    assert "schema_name" not in legacy
    assert "schema_version" not in legacy
    assert legacy["candidate_path"] == str(legacy_path)
    assert portable["schema_name"] == "TrainingManifest"
    assert portable["schema_version"] == "2.0"
    assert portable["candidate_path"] == "candidate_graph.json"
    assert not Path(portable["candidate_path"]).is_absolute()


def test_v2_retraining_aggregate_has_only_portable_candidate_artifacts():
    candidate = ROOT / "common" / "initial_candidate.ir.json"
    plan = resolve_evaluation_plan(
        "smoke_eval_v1",
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=24,
    )
    report = _build_aggregate_report(
        candidate=candidate,
        profile=SMOKE_TRAIN_CUDA_V2,
        evaluation_plan=plan,
        requested_device="cuda",
        seeds=[1],
        runs=[
            {
                "seed": 1,
                "success": True,
                "public_accuracy": 0.0,
                "eligible_for_parent": False,
                "failure_stage": "public_accuracy",
                "evaluation_record": {"fixture": "typed-layer-a-record"},
            }
        ],
    )

    assert report["schema_name"] == "AggregateRetrainingReport"
    assert report["schema_version"] == "2.0"
    assert report["candidate_format"] == "architecture_ir"
    assert len(report["candidate_source_hash"]) == 64
    assert len(report["candidate_graph_hash"]) == 64
    assert report["candidate_artifact_paths"] == [
        "seed_1/candidate_graph.json"
    ]
    assert all(
        not Path(path).is_absolute() and not path.startswith("/opt/")
        for path in report["candidate_artifact_paths"]
    )
    encoded = json.dumps(report, sort_keys=True)
    assert str(ROOT.resolve()) not in encoded
    assert "/opt/architecture_discovery" not in encoded


def test_v1_retraining_aggregate_retains_historical_candidate_path_shape():
    candidate = ROOT / "common" / "initial_candidate.py"
    plan = resolve_evaluation_plan(
        "smoke_eval_v1",
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=24,
    )
    runs = [
        {
            "seed": 1,
            "success": False,
            "public_accuracy": 0.0,
            "eligible_for_parent": False,
            "failure_stage": "public_accuracy",
        }
    ]
    report = _build_aggregate_report(
        candidate=candidate,
        profile=SMOKE_TRAIN_V1,
        evaluation_plan=plan,
        requested_device="mps",
        seeds=[1],
        runs=runs,
    )

    assert report == {
        "candidate": str(candidate.resolve()),
        "profile": SMOKE_TRAIN_V1.name,
        "profile_version": SMOKE_TRAIN_V1.version,
        "profile_hash": SMOKE_TRAIN_V1.profile_hash,
        "device": "mps",
        "sequential": True,
        "success_count": 0,
        "layer_a_eligibility_rate": 0.0,
        "mean_public_accuracy": 0.0,
        "population_stddev_public_accuracy": 0.0,
        "sealed_qualification_performed": False,
        "runs": runs,
    }
