import json
from pathlib import Path

import common.evaluator as evaluator_module
from common.evaluation_profiles import EvaluationLayer, resolve_evaluation_plan
from common.evaluator import (
    SearchEvaluationContext,
    evaluate_candidate,
    evaluate_trained_candidate_in_process,
)
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.trainer import trusted_component_hashes, trusted_component_set_sha256
from common.training_config import TrainingSeedBundle
from evaluation.records import search_evaluation_from_dict


ROOT = Path(__file__).resolve().parents[1]


def _plan():
    return resolve_evaluation_plan(
        "smoke_eval_v1",
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=8,
    )


def _context(run_id: str) -> SearchEvaluationContext:
    return SearchEvaluationContext(
        study_id="reproducibility-test",
        block_id="reproducibility-block",
        run_id=run_id,
        condition_id="C0",
    )


def test_trusted_component_set_hash_is_named_order_independent() -> None:
    hashes = trusted_component_hashes()
    required = {
        "common.candidate_artifact",
        "architecture_ir.interpreter",
        "architecture_ir.graph",
        "architecture_ir.codec",
        "architecture_ir.runtime_evidence",
        "common.descriptor_extractor",
        "common.training_config",
        "common.device",
        "common.task_adapter",
        "common.training_data",
        "common.evaluator",
        "common.trainer",
        "containment.source_scan",
    }
    assert required.issubset(hashes)
    assert trusted_component_set_sha256(hashes) == trusted_component_set_sha256(
        dict(reversed(list(hashes.items())))
    )
    changed = dict(hashes)
    changed["common.trainer"] = "0" * 64
    assert trusted_component_set_sha256(changed) != trusted_component_set_sha256(
        hashes
    )


def test_training_manifest_records_reviewable_component_hashes(cpu_smoke_training):
    _training, output = cpu_smoke_training
    manifest = json.loads((output / "training_manifest.json").read_text())
    expected_hashes = trusted_component_hashes()
    expected_set_hash = trusted_component_set_sha256(expected_hashes)
    assert manifest["trusted_executable_component_hashes"] == expected_hashes
    assert manifest["trusted_component_set_sha256"] == expected_set_hash
    assert manifest["controller_source_hash"] == expected_set_hash
    assert manifest["immutable_candidate_relative_path"] == "candidate_source.py"


def test_evaluator_builds_only_the_immutable_training_copy(
    cpu_smoke_training, monkeypatch
):
    training, output = cpu_smoke_training
    real_builder = evaluator_module.build_candidate_artifact
    built_paths: list[Path] = []

    def tracked_builder(path, *, seed):
        built_paths.append(Path(path).resolve())
        return real_builder(path, seed=seed)

    monkeypatch.setattr(evaluator_module, "build_candidate_artifact", tracked_builder)
    record = evaluate_trained_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.py",
        training=training,
        seeds=TrainingSeedBundle.from_run_seed(17),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        evaluation_plan=_plan(),
        context=_context("immutable-copy"),
        eligibility_threshold=0.0,
    )

    assert record.execution_ok
    assert built_paths == [(output / "candidate_source.py").resolve()]
    assert record.envelope.code_sha256 == trusted_component_set_sha256(
        trusted_component_hashes()
    )

    # The runtime-evidence field is an additive optional field in schema v1;
    # records written before it existed must remain readable.
    old_v1_payload = record.to_dict()
    old_v1_payload.pop("runtime_validity_artifact")
    restored = search_evaluation_from_dict(old_v1_payload)
    assert restored.runtime_validity_artifact is None
    assert restored.controller_view().as_dict() == record.controller_view().as_dict()


def test_same_shape_candidate_swap_is_rejected_before_checkpoint_or_build(
    cpu_smoke_training, tmp_path, monkeypatch
):
    training, _output = cpu_smoke_training
    original = ROOT / "common" / "initial_candidate.py"
    swapped = tmp_path / "same_shape_candidate.py"
    swapped.write_bytes(
        original.read_bytes()
        + b"\n# Same architecture and parameter shapes; deliberately different identity.\n"
    )

    checkpoint_loaded = False
    candidate_built = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal checkpoint_loaded
        checkpoint_loaded = True
        raise AssertionError("checkpoint must not load after candidate identity mismatch")

    def forbidden_build(*_args, **_kwargs):
        nonlocal candidate_built
        candidate_built = True
        raise AssertionError("candidate must not build after candidate identity mismatch")

    monkeypatch.setattr(evaluator_module.torch, "load", forbidden_load)
    monkeypatch.setattr(evaluator_module, "build_candidate_artifact", forbidden_build)
    record = evaluate_trained_candidate_in_process(
        candidate_path=swapped,
        training=training,
        seeds=TrainingSeedBundle.from_run_seed(17),
        requested_device="cpu",
        allow_cpu_for_tests=True,
        evaluation_plan=_plan(),
        context=_context("candidate-swap"),
        eligibility_threshold=0.0,
    )

    assert not checkpoint_loaded
    assert not candidate_built
    assert not record.execution_ok
    assert not record.transformer_valid
    assert record.failure_stage == "reproducibility_binding"
    assert record.infrastructure_failure


def test_ir_runtime_evidence_is_bound_to_training_and_code_identity(tmp_path):
    output = tmp_path / "ir-training"
    record = evaluate_candidate(
        ROOT / "common" / "initial_candidate.ir.json",
        training_profile="smoke_train_v1",
        training_seed=31,
        training_output_dir=output,
        device="cpu",
        allow_cpu_for_tests=True,
        evaluation_profile="smoke_eval_v1",
        evaluation_case_count=8,
        eligibility_threshold=0.0,
        context=_context("ir-runtime-binding"),
    )
    assert record.execution_ok
    assert record.transformer_valid
    assert record.runtime_validity_artifact is not None

    runtime_path = output / record.runtime_validity_artifact.relative_path
    runtime = json.loads(runtime_path.read_text())
    manifest = json.loads((output / "training_manifest.json").read_text())
    checkpoint_hash = evaluator_module.file_hash(output / "best_checkpoint.pt")
    assert runtime["candidate_artifact_hash"] == manifest["candidate_source_hash"]
    assert runtime["candidate_graph_hash"] == manifest["candidate_graph_hash"]
    assert runtime["checkpoint_sha256"] == checkpoint_hash
    assert runtime["training_profile_hash"] == manifest["profile_hash"]
    assert runtime["seed_bundle_hash"] == manifest["seed_bundle_hash"]
    assert runtime["selected_device"] == "cpu"
    assert (
        runtime["trusted_component_set_sha256"]
        == manifest["trusted_component_set_sha256"]
        == record.envelope.code_sha256
    )
