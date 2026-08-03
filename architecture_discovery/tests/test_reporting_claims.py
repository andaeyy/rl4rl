from __future__ import annotations

import pytest

from reporting import (
    ArithmeticClaim,
    ArithmeticClaimKind,
    ExternalValidityRecord,
    ExternalValidityStatus,
)


def test_arithmetic_claim_template_renders_a_hard_scope_disclaimer() -> None:
    claim = ArithmeticClaim(
        claim_id="claim-1",
        kind=ArithmeticClaimKind.MECHANISM,
        result_summary="the paired arithmetic intervention changed the tested outcome.",
        evidence_artifact_ids=("analysis-1",),
        limitations=("Only the preregistered integer-addition generators were tested.",),
    )

    assert claim.claim_scope == "autoregressive_arithmetic_only"
    assert "does not establish a general language-model improvement" in claim.rendered_text


@pytest.mark.parametrize(
    "summary",
    (
        "the architecture improves all tasks.",
        "the result advances general intelligence.",
        "the method is state-of-the-art.",
        "language models generally benefit from the mechanism.",
    ),
)
def test_arithmetic_claim_template_rejects_common_external_overclaims(summary) -> None:
    with pytest.raises(ValueError, match="arithmetic-only"):
        ArithmeticClaim(
            claim_id="overclaim",
            kind=ArithmeticClaimKind.SEARCH_YIELD,
            result_summary=summary,
            evidence_artifact_ids=("analysis-1",),
            limitations=("A limitation is retained.",),
        )


def test_external_validity_status_cannot_claim_untested_evidence() -> None:
    arithmetic_only = ExternalValidityRecord(
        status=ExternalValidityStatus.ARITHMETIC_ONLY,
        primary_task_id="integer-addition",
        tested_task_ids=("integer-addition",),
        second_task_evidence_ids=(),
        scaling_evidence_ids=(),
        limitation="No second task or scale was tested.",
    )
    assert arithmetic_only.status is ExternalValidityStatus.ARITHMETIC_ONLY

    with pytest.raises(ValueError, match="scaling evidence"):
        ExternalValidityRecord(
            status=ExternalValidityStatus.ARITHMETIC_SCALING_TESTED,
            primary_task_id="integer-addition",
            tested_task_ids=("integer-addition",),
            second_task_evidence_ids=(),
            scaling_evidence_ids=(),
            limitation="Scaling was not actually measured.",
        )
