from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
from scripts import study_scientific_run
from study.contracts import StudySpec


def test_scientific_entrypoint_blocks_before_provider_initialization(
    monkeypatch, tmp_path
) -> None:
    provider_constructed = False

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_constructed
        provider_constructed = True
        raise AssertionError("provider must not be initialized behind a failed gate")

    monkeypatch.setattr(
        study_scientific_run,
        "audit_readiness",
        lambda **_kwargs: {
            "ready": False,
            "pilot_ready": False,
            "main_study_ready": False,
            "decision_ledger_sha256": "0" * 64,
            "gates": [],
        },
    )
    monkeypatch.setattr(study_scientific_run, "OpenAI", forbidden_provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study_scientific_run.py",
            "--study-spec",
            str(tmp_path / "missing-study.json"),
            "--phase",
            "main",
            "--output-root",
            str(tmp_path / "outputs"),
            "--initial-candidate",
            str(tmp_path / "missing-candidate.py"),
        ],
    )

    assert study_scientific_run.main() == 2
    assert not provider_constructed


def test_scientific_entrypoint_configures_cuda_scheduler(
    monkeypatch, tmp_path
) -> None:
    class SchedulerObserved(RuntimeError):
        pass

    observed: dict[str, object] = {}
    study_spec = replace(StudySpec.toy(), scientific=True)
    study_spec_path = tmp_path / "study.json"
    study_spec_path.write_text(
        json.dumps(study_spec.to_dict()),
        encoding="utf-8",
    )
    initial_candidate = tmp_path / "initial.ir.json"
    initial_candidate.write_text(
        (study_scientific_run.ROOT / "common" / "initial_candidate.ir.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        study_scientific_run,
        "audit_readiness",
        lambda **_kwargs: {
            "ready": True,
            "pilot_ready": True,
            "main_study_ready": True,
            "decision_ledger_sha256": "0" * 64,
            "gates": [],
        },
    )
    monkeypatch.setattr(
        study_scientific_run,
        "_validated_runtime_contract",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        study_scientific_run,
        "load_or_create_plan",
        lambda *_args, **_kwargs: object(),
    )

    def observe_scheduler(_plan, **kwargs):
        observed.update(kwargs)
        raise SchedulerObserved

    monkeypatch.setattr(
        study_scientific_run,
        "SequentialAcceleratorScheduler",
        observe_scheduler,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study_scientific_run.py",
            "--study-spec",
            str(study_spec_path),
            "--phase",
            "main",
            "--output-root",
            str(tmp_path / "outputs"),
            "--initial-candidate",
            str(initial_candidate),
        ],
    )

    with pytest.raises(SchedulerObserved):
        study_scientific_run.main()
    assert observed["accelerator_kind"] == "cuda"


def test_scientific_runtime_contract_rejects_non_cuda_budget() -> None:
    spec = replace(StudySpec.toy(), scientific=True)

    with pytest.raises(SystemExit, match="budget.accelerator_kind='cuda'"):
        study_scientific_run._validated_runtime_contract(
            spec,
            initial_source="{}\n",
            readiness={"decision_ledger_sha256": "0" * 64},
        )


def test_scientific_entrypoint_has_no_fallthrough_to_engineering_modal_action() -> None:
    with pytest.raises(SystemExit, match="no frozen full-profile Modal action"):
        study_scientific_run._require_modal_full_profile_launch_contract()


def test_scientific_preflight_blocks_before_provider_when_readiness_is_mocked_true(
    monkeypatch, tmp_path
) -> None:
    provider_constructed = False

    def forbidden_provider(**_kwargs):
        nonlocal provider_constructed
        provider_constructed = True
        raise AssertionError("provider must remain unreachable")

    study_spec = replace(StudySpec.toy(), scientific=True)
    study_spec_path = tmp_path / "study.json"
    study_spec_path.write_text(json.dumps(study_spec.to_dict()), encoding="utf-8")
    initial_candidate = tmp_path / "initial.ir.json"
    initial_candidate.write_text(
        (study_scientific_run.ROOT / "common" / "initial_candidate.ir.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        study_scientific_run,
        "audit_readiness",
        lambda **_kwargs: {
            "pilot_ready": True,
            "main_study_ready": True,
            "decision_ledger_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(
        study_scientific_run,
        "_validated_runtime_contract",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        study_scientific_run,
        "load_or_create_plan",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        study_scientific_run,
        "SequentialAcceleratorScheduler",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(study_scientific_run, "OpenAI", forbidden_provider)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study_scientific_run.py",
            "--study-spec",
            str(study_spec_path),
            "--phase",
            "pilot",
            "--output-root",
            str(tmp_path / "outputs"),
            "--initial-candidate",
            str(initial_candidate),
        ],
    )

    with pytest.raises(SystemExit, match="no frozen full-profile Modal action"):
        study_scientific_run.main()
    assert not provider_constructed
