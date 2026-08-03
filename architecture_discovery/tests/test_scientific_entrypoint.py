from __future__ import annotations

import sys

from scripts import study_scientific_run


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
