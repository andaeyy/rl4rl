from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import openevolve_adapter
from common.descriptor_schema import SEMANTIC_METRIC_NAMES
from common.evaluator import EvaluationBindingError
from evaluation.records import ControllerSearchView, SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _write_ir_candidate(path):
    payload = json.loads(
        (ROOT / "common" / "initial_candidate.ir.json").read_text(encoding="utf-8")
    )
    payload["graph_id"] = "tests.openevolve.candidate"
    payload["metadata"]["mechanism_hypothesis"] = "adapter test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _view(candidate, *, eligible, score, descriptors=()):
    source_hash = openevolve_adapter.file_hash(candidate)
    return ControllerSearchView(
        schema_name="search_evaluation",
        schema_version=SCHEMA_VERSION,
        record_id=f"record-{source_hash}",
        run_id="native-openevolve",
        condition_id="native-openevolve",
        candidate_id=f"candidate-{source_hash}",
        execution_ok=eligible,
        transformer_valid=True,
        public_accuracy=score,
        search_score=score,
        eligible_for_parent=eligible,
        failure_stage="" if eligible else "device_unavailable",
        infrastructure_failure=not eligible,
        online_descriptor_codes=descriptors,
    )


def _set_native_context(monkeypatch):
    monkeypatch.setenv("DISCOVERY_RUN_ID", "native-openevolve")
    monkeypatch.setenv("DISCOVERY_CONDITION_ID", "native-openevolve")


def _architecture_hash(path):
    validation = openevolve_adapter.validate_ir_candidate_path(path)
    assert validation.valid and validation.graph is not None
    return validation.graph.architecture_hash


def test_failed_evaluation_emits_all_unknown_semantic_metrics(monkeypatch, tmp_path):
    _set_native_context(monkeypatch)
    candidate = _write_ir_candidate(tmp_path / "candidate.json")
    view = _view(candidate, eligible=False, score=0.0)
    monkeypatch.setattr(
        openevolve_adapter,
        "evaluate_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(
            controller_view=lambda: view
        ),
    )
    monkeypatch.setenv("DISCOVERY_TRAINING_OUTPUT_ROOT", str(tmp_path / "runs"))

    result = openevolve_adapter.evaluate_for_openevolve(str(candidate))

    for metric_name in SEMANTIC_METRIC_NAMES.values():
        assert result.metrics[metric_name] == 0.0
    assert result.metrics["combined_score"] == 0.0
    assert result.metrics["eligible_for_parent"] == 0.0


def test_combined_score_uses_only_eligibility_and_search_score(monkeypatch, tmp_path):
    _set_native_context(monkeypatch)
    candidate = _write_ir_candidate(tmp_path / "candidate.json")
    view = _view(
        candidate,
        eligible=True,
        score=0.8,
        descriptors=(("semantic_token_representation", 4.0),),
    )
    monkeypatch.setattr(
        openevolve_adapter,
        "evaluate_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(
            controller_view=lambda: view
        ),
    )

    result = openevolve_adapter.evaluate_for_openevolve(str(candidate))

    assert result.metrics["combined_score"] == 2.8
    assert result.metrics["semantic_token_representation"] == 4.0
    assert result.artifacts["candidate_graph_hash"]
    assert result.artifacts["candidate_architecture_hash"]


def test_stale_evaluator_result_is_rejected_before_metrics_reach_database(
    monkeypatch, tmp_path
):
    _set_native_context(monkeypatch)
    candidate = _write_ir_candidate(tmp_path / "candidate.json")
    stale = _view(candidate, eligible=True, score=1.0)
    stale = ControllerSearchView(
        **{**stale.__dict__, "candidate_id": "candidate-" + "0" * 64}
    )
    monkeypatch.setattr(
        openevolve_adapter,
        "evaluate_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(
            controller_view=lambda: stale
        ),
    )

    with pytest.raises(EvaluationBindingError, match="candidate source hash"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))


def test_python_candidate_is_rejected_before_shared_evaluator(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("raise RuntimeError('must never execute')\n")
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", unexpected)

    with pytest.raises(ValueError, match="generated Python is never executed"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
    assert not called


def test_malformed_ir_is_rejected_before_shared_evaluator(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text('{"schema_name":"architecture_tensor_graph"}')
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", unexpected)

    with pytest.raises(ValueError, match="invalid Architecture IR candidate"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
    assert not called


def test_metadata_only_parent_rewrite_is_rejected_before_training(
    monkeypatch, tmp_path
):
    candidate = _write_ir_candidate(tmp_path / "metadata-only.json")
    parent = ROOT / "common" / "initial_candidate.ir.json"
    monkeypatch.setenv("DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE", "1")
    monkeypatch.setenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", _architecture_hash(parent))
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", unexpected)

    with pytest.raises(ValueError, match="executable architecture no-op"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
    assert not called


def test_node_renaming_parent_rewrite_is_rejected_before_training(
    monkeypatch, tmp_path
):
    parent = ROOT / "common" / "initial_candidate.ir.json"
    payload = json.loads(parent.read_text(encoding="utf-8"))
    renames = {
        node["node_id"]: f"renamed_{index}"
        for index, node in enumerate(payload["nodes"])
    }
    payload["graph_id"] = "tests.openevolve.identifier.only"
    payload["input_node_id"] = renames[payload["input_node_id"]]
    payload["output_node_id"] = renames[payload["output_node_id"]]
    for node in payload["nodes"]:
        node["node_id"] = renames[node["node_id"]]
        tied_embedding = node["attributes"].get("tie_embedding")
        if tied_embedding is not None:
            node["attributes"]["tie_embedding"] = renames[tied_embedding]
    for edge in payload["edges"]:
        edge["source"] = renames[edge["source"]]
        edge["target"] = renames[edge["target"]]
    candidate = tmp_path / "identifier-only.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    parent_hash = _architecture_hash(parent)
    assert _architecture_hash(candidate) == parent_hash

    monkeypatch.setenv("DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE", "1")
    monkeypatch.setenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", parent_hash)
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", unexpected)

    with pytest.raises(ValueError, match="executable architecture no-op"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
    assert not called


def test_changed_child_is_bound_to_parent_and_reaches_training(monkeypatch, tmp_path):
    _set_native_context(monkeypatch)
    parent = ROOT / "common" / "initial_candidate.ir.json"
    payload = json.loads(parent.read_text(encoding="utf-8"))
    payload["graph_id"] = "tests.openevolve.changed"
    normalization = next(
        node for node in payload["nodes"] if node["kind"] == "normalization"
    )
    normalization["attributes"]["epsilon"] = 2e-5
    candidate = tmp_path / "changed.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    parent_hash = _architecture_hash(parent)
    assert _architecture_hash(candidate) != parent_hash

    monkeypatch.setenv("DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE", "1")
    monkeypatch.setenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", parent_hash)
    registry = openevolve_adapter.ArchitectureHashRegistry(tmp_path / "registry")
    assert registry.claim(parent_hash)
    monkeypatch.setenv(
        "DISCOVERY_ARCHITECTURE_HASH_REGISTRY",
        str(registry.directory),
    )
    view = _view(candidate, eligible=True, score=0.5)
    calls = []

    def fake_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(controller_view=lambda: view)

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", fake_evaluate)

    result = openevolve_adapter.evaluate_for_openevolve(str(candidate))

    assert len(calls) == 1
    assert result.artifacts["parent_architecture_hash"] == parent_hash
    assert result.artifacts["candidate_architecture_hash"] != parent_hash


def test_run_wide_duplicate_is_rejected_before_training(monkeypatch, tmp_path):
    parent = ROOT / "common" / "initial_candidate.ir.json"
    payload = json.loads(parent.read_text(encoding="utf-8"))
    payload["graph_id"] = "tests.openevolve.duplicate"
    normalization = next(
        node for node in payload["nodes"] if node["kind"] == "normalization"
    )
    normalization["attributes"]["epsilon"] = 2e-5
    candidate = tmp_path / "duplicate.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    parent_hash = _architecture_hash(parent)
    candidate_hash = _architecture_hash(candidate)
    registry = openevolve_adapter.ArchitectureHashRegistry(tmp_path / "registry")
    assert registry.claim(parent_hash)
    assert registry.claim(candidate_hash)
    monkeypatch.setenv("DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE", "1")
    monkeypatch.setenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", parent_hash)
    monkeypatch.setenv(
        "DISCOVERY_ARCHITECTURE_HASH_REGISTRY",
        str(registry.directory),
    )
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", unexpected)

    with pytest.raises(ValueError, match="duplicates an architecture"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
    assert not called


def test_missing_parent_binding_fails_closed_for_noninitial_candidate(
    monkeypatch, tmp_path
):
    parent = ROOT / "common" / "initial_candidate.ir.json"
    payload = json.loads(parent.read_text(encoding="utf-8"))
    normalization = next(
        node for node in payload["nodes"] if node["kind"] == "normalization"
    )
    normalization["attributes"]["epsilon"] = 2e-5
    candidate = tmp_path / "unbound-child.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE", "1")
    monkeypatch.setenv("DISCOVERY_INITIAL_ARCHITECTURE_HASH", _architecture_hash(parent))
    monkeypatch.delenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", raising=False)
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(openevolve_adapter, "evaluate_candidate", unexpected)

    with pytest.raises(ValueError, match="missing for a non-initial candidate"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
    assert not called


def test_initial_candidate_requires_and_consumes_one_time_authorization(
    monkeypatch, tmp_path
):
    _set_native_context(monkeypatch)
    candidate = _write_ir_candidate(tmp_path / "initial.json")
    initial_hash = _architecture_hash(candidate)
    monkeypatch.setenv("DISCOVERY_ENFORCE_PARENT_ARCHITECTURE_CHANGE", "1")
    monkeypatch.setenv("DISCOVERY_INITIAL_ARCHITECTURE_HASH", initial_hash)
    monkeypatch.delenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", raising=False)
    monkeypatch.setenv(
        "DISCOVERY_OPENEVOLVE_INITIAL_EVALUATION_AUTH",
        initial_hash,
    )
    view = _view(candidate, eligible=True, score=0.5)
    monkeypatch.setattr(
        openevolve_adapter,
        "evaluate_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(controller_view=lambda: view),
    )

    openevolve_adapter.evaluate_for_openevolve(str(candidate))

    assert (
        "DISCOVERY_OPENEVOLVE_INITIAL_EVALUATION_AUTH"
        not in openevolve_adapter.os.environ
    )
    with pytest.raises(ValueError, match="one-time initial evaluation authorization"):
        openevolve_adapter.evaluate_for_openevolve(str(candidate))
