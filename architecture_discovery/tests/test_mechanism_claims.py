import pytest

from mechanism.claims import ClaimEvidence, ClaimVerdict, assess_claim
from mechanism.fakes import toy_mechanism_claim


def _evidence(
    evidence_id: str,
    requirement_id: str,
    test_id: str,
    *,
    supports: bool = True,
) -> ClaimEvidence:
    return ClaimEvidence(
        evidence_id=evidence_id,
        requirement_id=requirement_id,
        discriminating_test_ids=(test_id,),
        artifact_sha256=evidence_id[-1] * 64,
        supports_prediction=supports,
        summary="Synthetic preregistered evidence.",
    )


def test_claim_cannot_be_supported_without_every_declared_test_and_requirement():
    claim = toy_mechanism_claim()
    partial = (
        _evidence(
            "evidence-record:a",
            "evidence:ablation",
            "test:ablation",
        ),
    )
    with pytest.raises(ValueError, match="cannot be supported"):
        assess_claim(claim, partial, requested_verdict=ClaimVerdict.SUPPORTED)

    assessment = assess_claim(
        claim,
        partial,
        requested_verdict=ClaimVerdict.INCONCLUSIVE,
    )
    assert assessment.missing_test_ids == ("test:rescue",)
    assert assessment.missing_requirement_ids == ("evidence:rescue",)


def test_supported_claim_requires_positive_discriminating_evidence():
    claim = toy_mechanism_claim()
    complete = (
        _evidence(
            "evidence-record:a",
            "evidence:ablation",
            "test:ablation",
        ),
        _evidence(
            "evidence-record:b",
            "evidence:rescue",
            "test:rescue",
        ),
    )
    assessment = assess_claim(
        claim,
        complete,
        requested_verdict=ClaimVerdict.SUPPORTED,
    )
    assert assessment.verdict is ClaimVerdict.SUPPORTED
    assert not assessment.missing_test_ids
    assert not assessment.missing_requirement_ids

    contradicted = complete[:-1] + (
        _evidence(
            "evidence-record:c",
            "evidence:rescue",
            "test:rescue",
            supports=False,
        ),
    )
    with pytest.raises(ValueError, match="cannot be supported"):
        assess_claim(
            claim,
            contradicted,
            requested_verdict=ClaimVerdict.SUPPORTED,
        )


def test_claim_hash_is_stable_and_bound_to_snapshot():
    first = toy_mechanism_claim()
    second = toy_mechanism_claim()
    assert first.claim_hash == second.claim_hash
    assert len(first.claim_hash) == 64
