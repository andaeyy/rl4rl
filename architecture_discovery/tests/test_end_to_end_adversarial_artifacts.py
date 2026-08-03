from __future__ import annotations

import json

import pytest

from artifacts.failures import (
    FailureClass,
    FailureRecord,
    RerunNotAuthorized,
    RerunPolicy,
    authorize_rerun,
)
from artifacts.records import ArtifactContext, EventKind
from artifacts.store import ArtifactIntegrityError, RunArtifactStore


def _context() -> ArtifactContext:
    return ArtifactContext(
        study_id="study",
        block_id="block",
        run_id="run",
        condition_id="C0",
        writer_component="adversarial-test",
        code_sha256="a" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        run_seed=7,
        assignment_sha256="d" * 64,
    )


def test_payload_tamper_and_content_object_tamper_are_detected(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    store.append(EventKind.PROPOSAL, {"proposal_id": "proposal-one"})
    event_path = next((store.raw_events / "00000000000000000001").iterdir())
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    payload["payload"]["proposal_id"] = "attacker-rewrite"
    event_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError) as error:
        store.scan()
    assert error.value.report.findings[0].code == "INVALID_EVENT_HASH_OR_SCHEMA"

    reference = store.objects.put_bytes(b"original-object")
    object_path = store.objects.path_for(reference.sha256)
    object_path.write_bytes(b"tampered-object")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.objects.read_bytes(reference)


def test_scientific_or_candidate_failure_cannot_authorize_a_rerun() -> None:
    policy = RerunPolicy()
    for failure_class in (
        FailureClass.CANDIDATE_IMPORT,
        FailureClass.NONQUALIFYING_RESULT,
    ):
        failure = FailureRecord.create(
            attempt_id="attempt-one",
            failure_class=failure_class,
            stage="evaluation",
        )
        with pytest.raises(RerunNotAuthorized, match="remain ITT outcomes"):
            authorize_rerun(
                assigned_run_id="run",
                previous_attempt_id="attempt-one",
                attempt_number=1,
                failure=failure,
                policy=policy,
            )
