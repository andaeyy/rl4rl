from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from common import openevolve_runner
from common.training_config import SMOKE_TRAIN_V1, TrainingSeedBundle


def _offline_provider(monkeypatch):
    monkeypatch.setenv("DISCOVERY_API_KEY", "offline-test-key")
    monkeypatch.setenv("DISCOVERY_API_BASE", "https://api.openai.com/v1")
    monkeypatch.setenv("DISCOVERY_MODEL", "gpt-5.6-sol")
    for name in (
        "DISCOVERY_REASONING_EFFORT",
        "DISCOVERY_MAX_COMPLETION_TOKENS",
        "DISCOVERY_REQUEST_TIMEOUT_SECONDS",
        "DISCOVERY_REQUEST_RETRIES",
        "DISCOVERY_RETRY_DELAY_SECONDS",
        "DISCOVERY_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def _record_terminal_iterations(iterations, *, error=False):
    for iteration in range(1, iterations + 1):
        result = openevolve_runner.SerializableResult(
            error="offline worker error" if error else None,
            child_program_dict=None if error else {"id": f"child-{iteration}"},
            iteration=iteration,
        )
        openevolve_runner._record_terminal_outcome(iteration, result)


def _stub_initial_ir(monkeypatch, tmp_path):
    candidate = tmp_path / "initial_candidate.ir.json"
    candidate.write_text(
        (
            openevolve_runner.ROOT
            / "common"
            / "initial_candidate.ir.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(openevolve_runner, "INITIAL_IR_CANDIDATE", candidate)
    return candidate


def test_preflight_resolves_smoke_plan_without_provider_or_training(
    monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        lambda **kwargs: captured.update(kwargs),
    )

    plan = openevolve_runner._preflight_runtime(
        candidate_path=tmp_path / "candidate.json",
        training_profile=SMOKE_TRAIN_V1,
        training_seeds=TrainingSeedBundle.from_run_seed(1),
        training_device="mps",
        allow_cpu_for_tests=False,
        evaluation_profile_name="smoke_eval_v1",
        evaluation_case_count=64,
        pi_decision_record_id=None,
    )

    assert plan.profile_name == "smoke_eval_v1"
    assert plan.case_count == 64
    assert captured["profile"] is SMOKE_TRAIN_V1
    assert captured["requested_device"] == "mps"


def test_iteration_worker_binds_validated_parent_architecture_hash(
    monkeypatch,
):
    parent_code = (
        openevolve_runner.ROOT / "common" / "initial_candidate.ir.json"
    ).read_text(encoding="utf-8")
    expected_hash = openevolve_runner._architecture_hash_for_ir_text(parent_code)
    captured = {}

    def fake_vendor(iteration, snapshot, parent_id, inspiration_ids):
        captured["hash"] = openevolve_runner.os.environ[
            "DISCOVERY_PARENT_ARCHITECTURE_HASH"
        ]
        return openevolve_runner.SerializableResult(iteration=iteration)

    monkeypatch.setattr(
        openevolve_runner,
        "_VENDOR_RUN_ITERATION_WORKER",
        fake_vendor,
    )
    monkeypatch.delenv("DISCOVERY_PARENT_ARCHITECTURE_HASH", raising=False)

    result = openevolve_runner._parent_bound_iteration_worker(
        3,
        {"programs": {"parent": {"code": parent_code}}},
        "parent",
        [],
    )

    assert result.error is None
    assert captured["hash"] == expected_hash
    assert "DISCOVERY_PARENT_ARCHITECTURE_HASH" not in openevolve_runner.os.environ


def test_iteration_worker_fails_closed_for_invalid_parent(monkeypatch):
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        openevolve_runner,
        "_VENDOR_RUN_ITERATION_WORKER",
        unexpected,
    )

    result = openevolve_runner._parent_bound_iteration_worker(
        4,
        {"programs": {"parent": {"code": "{}"}}},
        "parent",
        [],
    )

    assert result.error is not None
    assert "parent architecture binding failed" in result.error
    assert not called


def test_invalid_scientific_plan_fails_before_training_preflight(monkeypatch, tmp_path):
    called = False

    def unexpected(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        unexpected,
    )

    with pytest.raises(SystemExit, match="before provider initialization"):
        openevolve_runner._preflight_runtime(
            candidate_path=tmp_path / "candidate.json",
            training_profile=SMOKE_TRAIN_V1,
            training_seeds=TrainingSeedBundle.from_run_seed(1),
            training_device="mps",
            allow_cpu_for_tests=False,
            evaluation_profile_name="scientific_layer_a_v1",
            evaluation_case_count=None,
            pi_decision_record_id=None,
        )
    assert not called


def test_training_and_evaluation_scientific_status_must_match(
    monkeypatch, tmp_path
):
    called = False

    def unexpected(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        unexpected,
    )

    with pytest.raises(SystemExit, match="matching scientific status"):
        openevolve_runner._preflight_runtime(
            candidate_path=tmp_path / "candidate.json",
            training_profile=SMOKE_TRAIN_V1,
            training_seeds=TrainingSeedBundle.from_run_seed(1),
            training_device="mps",
            allow_cpu_for_tests=False,
            evaluation_profile_name="scientific_layer_a_v1",
            evaluation_case_count=10_000,
            pi_decision_record_id="frozen-decision",
        )
    assert not called


@pytest.mark.parametrize("value", ["0", "-1"])
def test_iterations_must_be_positive(value):
    with pytest.raises(SystemExit):
        openevolve_runner.run_controller("generic", ["--iterations", value])


def test_preflight_runs_before_provider_configuration(monkeypatch, tmp_path):
    for name in ("DISCOVERY_API_KEY", "DISCOVERY_API_BASE", "DISCOVERY_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        openevolve_runner,
        "_preflight_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(SystemExit("preflight sentinel")),
    )

    with pytest.raises(SystemExit, match="preflight sentinel"):
        openevolve_runner.run_controller(
            "generic",
            [
                "--engineering-pilot",
                "--iterations",
                "1",
                "--training-profile",
                "smoke_train_v1",
                "--evaluation-profile",
                "smoke_eval_v1",
                "--evaluation-cases",
                "64",
                "--device",
                "mps",
                "--output-dir",
                str(tmp_path / "fresh"),
            ],
        )


def test_nonempty_output_and_resume_are_refused(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "record.json").write_text("{}")

    with pytest.raises(SystemExit, match="non-empty"):
        openevolve_runner._require_fresh_output(output)
    with pytest.raises(SystemExit, match="resume is not implemented"):
        openevolve_runner.run_controller(
            "generic",
            ["--resume-checkpoint", str(output / "checkpoint")],
        )


def test_one_iteration_mock_controller_writes_two_candidate_budget_manifest(
    monkeypatch, tmp_path
):
    output = tmp_path / "pilot"
    initial_ir = _stub_initial_ir(monkeypatch, tmp_path)
    _offline_provider(monkeypatch)
    monkeypatch.setenv("DISCOVERY_SCIENTIFIC_DECISION_RECORD", "test-decision")
    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        lambda **_kwargs: {},
    )
    calls = {}

    class FakeController:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs

        async def run(self, *, iterations):
            calls["iterations"] = iterations
            _record_terminal_iterations(iterations)
            return SimpleNamespace(id="eligible-best")

    monkeypatch.setattr(openevolve_runner, "OpenEvolve", FakeController)

    openevolve_runner.run_controller(
        "generic",
        [
            "--iterations",
            "1",
            "--seed",
            "7",
            "--training-profile",
            "full_train_v1",
            "--evaluation-profile",
            "scientific_layer_a_v1",
            "--evaluation-cases",
            "10000",
            "--device",
            "mps",
            "--output-dir",
            str(output),
        ],
    )

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert calls["iterations"] == 1
    assert manifest["candidate_budget"] == 2
    assert manifest["mutation_budget"] == 1
    assert manifest["training"]["profile"] == "full_train_v1"
    assert manifest["training"]["device"] == "mps"
    assert manifest["evaluation"]["profile"] == "scientific_layer_a_v1"
    assert manifest["evaluation"]["case_count"] == 10_000
    assert manifest["evidence_scope"] == "secondary_native_replication"
    assert manifest["authoritative_scientific_evidence"] is False
    assert manifest["candidate_format"] == "architecture_ir_json"
    assert manifest["proposal_format"] == "strict_full_document_json"
    assert manifest["generated_python_execution"] is False
    assert (
        manifest["trusted_executable_component_hashes"]
        == openevolve_runner.trusted_component_hashes()
    )
    assert (
        manifest["trusted_component_set_sha256"]
        == openevolve_runner.trusted_component_set_sha256(
            manifest["trusted_executable_component_hashes"]
        )
    )
    assert calls["constructor"]["initial_program_path"] == str(initial_ir)
    assert calls["constructor"]["config"].language == "json"
    assert calls["constructor"]["config"].file_suffix == ".json"
    assert calls["constructor"]["config"].diff_based_evolution is False

    result = json.loads((output / "run_result.json").read_text())
    assert result["completed"] is True
    assert result["eligible_best_program_found"] is True
    assert result["best_program_id"] == "eligible-best"


def test_smoke_profile_requires_explicit_engineering_pilot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        lambda **_kwargs: {},
    )
    monkeypatch.delenv("DISCOVERY_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="require.*--engineering-pilot"):
        openevolve_runner.run_controller(
            "generic",
            [
                "--iterations",
                "1",
                "--training-profile",
                "smoke_train_v1",
                "--evaluation-profile",
                "smoke_eval_v1",
                "--evaluation-cases",
                "64",
                "--device",
                "mps",
                "--output-dir",
                str(tmp_path / "engineering-blocked"),
            ],
        )


@pytest.mark.parametrize("kind", ["generic", "semantic"])
def test_ten_opportunity_engineering_pilot_is_provider_free_and_non_authoritative(
    kind, monkeypatch, tmp_path
):
    output = tmp_path / kind
    initial_ir = _stub_initial_ir(monkeypatch, tmp_path)
    _offline_provider(monkeypatch)
    monkeypatch.setenv("DISCOVERY_LAYER_A_PROFILE", "scientific_layer_a_v1")
    monkeypatch.setenv("DISCOVERY_LAYER_A_CASES", "10000")
    monkeypatch.setenv("DISCOVERY_SCIENTIFIC_DECISION_RECORD", "ignored-in-engineering")
    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        lambda **_kwargs: {},
    )
    calls = {}

    class ProviderFreeController:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs

        async def run(self, *, iterations):
            import openevolve.process_parallel as process_parallel

            calls["iterations"] = iterations
            calls["parent_bound_worker_installed"] = (
                process_parallel._run_iteration_worker
                is openevolve_runner._parent_bound_iteration_worker
            )
            _record_terminal_iterations(iterations)
            return SimpleNamespace(id=f"{kind}-eligible-best")

    monkeypatch.setattr(openevolve_runner, "OpenEvolve", ProviderFreeController)

    openevolve_runner.run_controller(
        kind,
        [
            "--engineering-pilot",
            "--iterations",
            "10",
            "--seed",
            "7",
            "--output-dir",
            str(output),
        ],
    )

    manifest = json.loads((output / "run_manifest.json").read_text())
    result = json.loads((output / "run_result.json").read_text())
    config = calls["constructor"]["config"]

    assert calls["iterations"] == 10
    assert calls["constructor"]["initial_program_path"] == str(initial_ir)
    assert manifest["proposal_opportunities"] == 10
    assert manifest["candidate_budget"] == 11
    assert manifest["candidate_training_budget"] == 11
    assert manifest["parent_relative_architecture_change_required"] is True
    assert len(manifest["initial_architecture_hash"]) == 64
    assert manifest["architecture_hash_schema"] == "architecture_executable_v2"
    assert calls["parent_bound_worker_installed"] is True
    assert manifest["maximum_provider_attempts"] == 30
    assert manifest["engineering_pilot"] is True
    assert manifest["evidence_scope"] == "exploratory_engineering_pilot"
    assert manifest["authoritative_scientific_evidence"] is False
    assert manifest["training"]["profile"] == "smoke_train_v1"
    assert manifest["training"]["device"] == "mps"
    assert manifest["training"]["allow_cpu_for_tests"] is False
    assert manifest["evaluation"]["profile"] == "smoke_eval_v1"
    assert manifest["evaluation"]["case_count"] == 64
    assert manifest["evaluation"]["pi_decision_record_id"] is None
    assert config.language == "json"
    assert config.file_suffix == ".json"
    assert config.diff_based_evolution is False
    assert config.prompt.template_dir.endswith(f"openevolve_{kind}/templates")
    assert "parent document is an example" in config.prompt.system_message.lower()
    template = (
        openevolve_runner.ROOT
        / "agents"
        / f"openevolve_{kind}"
        / "templates"
        / "full_rewrite_user.txt"
    ).read_text(encoding="utf-8")
    assert '"nodes": []' not in template
    assert result["proposal_opportunities_requested"] == 10
    assert result["proposal_opportunities_completed"] == 10
    assert result["authoritative_scientific_evidence"] is False


def test_terminal_outcome_summary_counts_error_opportunities(monkeypatch, tmp_path):
    ledger = tmp_path / "terminal.jsonl"
    monkeypatch.setenv("DISCOVERY_OPENEVOLVE_TERMINAL_LEDGER", str(ledger))
    _record_terminal_iterations(2, error=True)

    summary = openevolve_runner._terminal_outcome_summary(ledger, 2)

    assert summary["terminal_count"] == 2
    assert summary["terminal_status_counts"] == {"error": 2}
    assert summary["all_requested_terminal"] is True


def test_seed_best_does_not_mask_partial_shutdown(monkeypatch, tmp_path):
    output = tmp_path / "partial"
    _stub_initial_ir(monkeypatch, tmp_path)
    _offline_provider(monkeypatch)
    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        lambda **_kwargs: {},
    )

    class PartiallyCompletedController:
        def __init__(self, **_kwargs):
            pass

        async def run(self, *, iterations):
            assert iterations == 3
            _record_terminal_iterations(1)
            # This mirrors the vendor controller returning the retained seed
            # after a graceful shutdown with later opportunities unsubmitted.
            return SimpleNamespace(id="retained-seed")

    monkeypatch.setattr(
        openevolve_runner,
        "OpenEvolve",
        PartiallyCompletedController,
    )

    with pytest.raises(SystemExit, match="1/3"):
        openevolve_runner.run_controller(
            "generic",
            [
                "--engineering-pilot",
                "--iterations",
                "3",
                "--output-dir",
                str(output),
            ],
        )

    result = json.loads((output / "run_result.json").read_text())
    assert result["completed"] is False
    assert result["eligible_best_program_found"] is True
    assert result["proposal_opportunities_completed"] == 1
    assert result["failure_stage"] == "incomplete_proposal_opportunities"


@pytest.mark.parametrize("kind", ["generic", "semantic"])
def test_config_requires_full_document_json_ir(kind):
    config_path = (
        openevolve_runner.ROOT / "agents" / f"openevolve_{kind}" / "config.yaml"
    )
    payload = yaml.safe_load(config_path.read_text())

    assert payload["diff_based_evolution"] is False
    assert payload["language"] == "json"
    assert payload["file_suffix"] == ".json"


def test_engineering_pilot_rejects_conflicting_profiles(tmp_path):
    with pytest.raises(SystemExit, match="fixes --training-profile"):
        openevolve_runner.run_controller(
            "generic",
            [
                "--engineering-pilot",
                "--training-profile",
                "full_train_v1",
                "--output-dir",
                str(tmp_path / "conflict"),
            ],
        )


def test_no_eligible_best_is_a_failed_run_without_best_artifact(monkeypatch, tmp_path):
    output = tmp_path / "no-best"
    _stub_initial_ir(monkeypatch, tmp_path)
    _offline_provider(monkeypatch)
    monkeypatch.setenv("DISCOVERY_SCIENTIFIC_DECISION_RECORD", "test-decision")
    monkeypatch.setattr(
        openevolve_runner,
        "validate_training_request",
        lambda **_kwargs: {},
    )

    class NoEligibleController:
        def __init__(self, **_kwargs):
            pass

        async def run(self, *, iterations):
            assert iterations == 1
            return None

    monkeypatch.setattr(openevolve_runner, "OpenEvolve", NoEligibleController)

    with pytest.raises(SystemExit, match="without an eligible candidate"):
        openevolve_runner.run_controller(
            "generic",
            [
                "--iterations",
                "1",
                "--training-profile",
                "full_train_v1",
                "--evaluation-profile",
                "scientific_layer_a_v1",
                "--evaluation-cases",
                "10000",
                "--device",
                "mps",
                "--output-dir",
                str(output),
            ],
        )

    result = json.loads((output / "run_result.json").read_text())
    assert result["completed"] is False
    assert result["failure_stage"] == "no_eligible_candidate"
    assert not (output / "best_program.json").exists()
