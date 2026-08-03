from __future__ import annotations

import pytest

from reporting import (
    MeasurementStatus,
    QuantityDisclosure,
    ResourceDisclosure,
)
from reporting.cards import ModelCard


def _quantity(name: str) -> QuantityDisclosure:
    return QuantityDisclosure(
        name=name,
        unit="units",
        status=MeasurementStatus.UNKNOWN,
        value=None,
        method="The quantity was not instrumented in this record.",
    )


def test_resource_disclosure_requires_compute_cost_and_energy() -> None:
    with pytest.raises(ValueError, match="resource disclosure"):
        ResourceDisclosure(
            quantities=(_quantity("mps_compute"), _quantity("cpu_compute")),
            prompt_tokens=0,
            completion_tokens=0,
            provider_usage_complete=False,
            notes=("Provider usage is incomplete.",),
        )


def test_parameter_count_is_fixed_to_descriptive_metadata() -> None:
    card = ModelCard(
        model_card_id="card-1",
        candidate_id="candidate-1",
        architecture_signature_sha256="a" * 64,
        training_configuration_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        parameter_count_metadata=9_999_999,
        intended_use="Record one frozen arithmetic candidate.",
        evaluation_scope="Preregistered arithmetic evaluation only.",
        limitations=("No broader task evidence is available.",),
    )

    assert card.parameter_count_role == "descriptive_metadata_only"
