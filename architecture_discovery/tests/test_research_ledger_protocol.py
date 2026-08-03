from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_ledger import (
    ConfidenceLevel,
    DiscriminatingTestSpec,
    HypothesisSpec,
    HypothesisStatus,
    PredictionSpec,
    ResearchProtocol,
    freeze_protocol,
    load_frozen_protocol,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def toy_protocol() -> ResearchProtocol:
    hypothesis = HypothesisSpec(
        hypothesis_id="hypothesis-routing",
        hypothesis="A routing operation changes how carry state reaches later positions.",
        causal_claim="The routed path causes improved propagation of carry state.",
        predictions=(
            PredictionSpec(
                prediction_id="prediction-carry",
                statement="Disabling the route removes the carry-state signature.",
                falsification_condition="The signature remains after the route is disabled.",
            ),
        ),
        nearest_alternative="The result comes only from added capacity.",
        discriminating_tests=(
            DiscriminatingTestSpec(
                test_id="test-route-zeroing",
                description="Zero the route while preserving the capacity control.",
                prediction_if_claim_true="The carry-state signature decreases.",
                prediction_if_alternative_true="The signature remains unchanged.",
            ),
        ),
        initial_confidence=ConfidenceLevel.MODERATE,
        initial_status=HypothesisStatus.ACTIVE,
    )
    return ResearchProtocol(
        protocol_id="toy-research-protocol",
        study_id="offline-toy-study",
        research_scope="Autoregressive integer-addition architecture discovery only.",
        hypotheses=(hypothesis,),
        code_sha256=digest("code"),
        config_sha256=digest("config"),
        environment_sha256=digest("environment"),
        pi_decision_sha256=digest("toy decisions"),
        scientific=False,
    )


def test_protocol_freeze_is_exclusive_and_hash_verified(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    protocol = toy_protocol()
    receipt = freeze_protocol(protocol, path)

    receipt.verify()
    assert load_frozen_protocol(path).protocol_sha256 == protocol.protocol_hash
    with pytest.raises(FileExistsError):
        freeze_protocol(protocol, path)


def test_protocol_mutation_is_detected_before_ledger_use(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    receipt = freeze_protocol(toy_protocol(), path)
    payload = json.loads(path.read_text())
    payload["protocol"]["hypotheses"][0]["causal_claim"] = "Mutated claim."
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        receipt.verify()


def test_protocol_scientific_flag_rejects_string_truthiness() -> None:
    payload = toy_protocol().to_dict()
    payload["scientific"] = "false"

    with pytest.raises(ValueError, match="scientific must be boolean"):
        ResearchProtocol.from_dict(payload)
