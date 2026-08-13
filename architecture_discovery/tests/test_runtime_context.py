from __future__ import annotations

import pytest
from common.runtime_context import ExecutionContextV1


def _modal_context() -> ExecutionContextV1:
    return ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-run-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id="ap-abc123",
        modal_function_id="fu-def456",
        modal_call_id="fc-ghi789",
        modal_image_id="im-jkl012",
        image_source_sha256="a" * 64,
        artifact_uri=("volume://rl4rl-architecture-artifacts/runs/modal-run-1"),
    )


def test_execution_context_round_trip_uses_an_exact_schema() -> None:
    context = _modal_context()
    payload = context.to_dict()

    assert set(payload) == ExecutionContextV1.FIELDS
    assert ExecutionContextV1.from_dict(payload) == context

    with pytest.raises(ValueError, match="unexpected or missing"):
        ExecutionContextV1.from_dict({**payload, "api_key": "secret"})
    with pytest.raises(ValueError, match="unexpected or missing"):
        ExecutionContextV1.from_dict(
            {key: value for key, value in payload.items() if key != "run_id"}
        )


def test_execution_context_rejects_secret_bearing_or_unbounded_values() -> None:
    payload = _modal_context().to_dict()
    with pytest.raises(ValueError, match="contain no credentials"):
        ExecutionContextV1.from_dict(
            {
                **payload,
                "artifact_uri": "volume://user:secret@volume/runs/run-1",
            }
        )
    with pytest.raises(ValueError, match="safe logical name"):
        ExecutionContextV1.from_dict(
            {**payload, "function_name": "Bearer sk-sensitive"}
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ExecutionContextV1.from_dict({**payload, "image_source_sha256": "not-a-digest"})


def test_local_context_cannot_claim_modal_identifiers() -> None:
    local = ExecutionContextV1.local(
        run_id="local-1",
        artifact_uri="local://workspace/runs/local-1",
    )
    assert ExecutionContextV1.from_dict(local.to_dict()) == local

    with pytest.raises(ValueError, match="may not contain Modal IDs"):
        ExecutionContextV1(
            **{
                **local.__dict__,
                "modal_call_id": "fc-forbidden",
            }
        )


def test_modal_context_requires_bound_source_and_volume() -> None:
    values = _modal_context().__dict__
    with pytest.raises(ValueError, match="image source digest"):
        ExecutionContextV1(**{**values, "image_source_sha256": None})
    with pytest.raises(ValueError, match="volume URI"):
        ExecutionContextV1(
            **{**values, "artifact_uri": "local://workspace/runs/modal-run-1"}
        )
