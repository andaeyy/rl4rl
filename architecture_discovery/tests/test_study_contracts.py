from dataclasses import fields

import pytest

from study.contracts import (
    ConditionId,
    ConditionSpec,
    ParentPolicy,
    ProposalPolicy,
    StudySpec,
)


def test_condition_cells_have_exactly_two_treatment_fields() -> None:
    conditions = ConditionSpec.primary()
    assert {condition.condition_id for condition in conditions} == set(ConditionId)
    assert ConditionSpec.TREATMENT_FIELDS == ("parent_policy", "proposal_policy")
    assert {field.name for field in fields(ConditionSpec)} == {
        "condition_id",
        "parent_policy",
        "proposal_policy",
    }
    assert len(
        {
            (condition.parent_policy, condition.proposal_policy)
            for condition in conditions
        }
    ) == 4


def test_condition_identifier_cannot_hide_an_extra_configuration_difference() -> None:
    with pytest.raises(ValueError, match="expected"):
        ConditionSpec(
            ConditionId.C0,
            ParentPolicy.PORTFOLIO,
            ProposalPolicy.ORDINARY,
        )


def test_common_configuration_lives_once_on_study_spec() -> None:
    study = StudySpec.toy(proposal_opportunities=4)
    serialized = study.to_dict()
    assert "conditions" not in serialized
    assert serialized["portfolio_size"] == 2
    assert serialized["transition_opportunities"] == [2]
    for condition in study.conditions:
        assert set(condition.to_dict()) == {
            "schema_name",
            "schema_version",
            "condition_id",
            "parent_policy",
            "proposal_policy",
        }
