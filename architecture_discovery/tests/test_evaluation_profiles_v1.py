import pytest

from common.evaluation_profiles import (
    DEVELOPMENT_EVAL_V1,
    EVALUATION_PROFILES,
    SCIENTIFIC_CASE_FLOOR,
    EvaluationLayer,
    resolve_evaluation_plan,
    validate_disjoint_scientific_plans,
)


def _hash(character: str) -> str:
    return character * 64


def test_all_evaluation_profiles_are_versioned_and_self_validating():
    assert set(EVALUATION_PROFILES) == {
        "unit_eval_v1",
        "smoke_eval_v1",
        "development_eval_v1",
        "scientific_layer_a_v1",
        "scientific_layer_b_v1",
        "scientific_layer_c_v1",
    }
    for profile in EVALUATION_PROFILES.values():
        profile.validate_definition()
        assert profile.version == "1"
        assert len(profile.profile_hash) == 64


def test_engineering_profile_is_explicitly_synthetic():
    plan = resolve_evaluation_plan(
        DEVELOPMENT_EVAL_V1.name,
        layer=EvaluationLayer.QUALIFICATION,
        case_source_id="synthetic-layer-b-fixture",
        case_source_sha256=_hash("a"),
    )
    assert plan.case_count == DEVELOPMENT_EVAL_V1.default_case_count
    assert plan.synthetic
    assert plan.sealed
    assert not plan.controller_visible
    assert not plan.scientific


def test_scientific_profile_has_no_implicit_case_count():
    with pytest.raises(ValueError, match="case_count must be supplied"):
        resolve_evaluation_plan(
            "scientific_layer_a_v1",
            layer=EvaluationLayer.SEARCH,
            case_source_id="public-a-v1",
            case_source_sha256=_hash("a"),
            pi_decision_record_id="decision-1",
        )


def test_smoke_sized_case_count_cannot_enter_scientific_profile():
    with pytest.raises(ValueError, match="requires at least"):
        resolve_evaluation_plan(
            "scientific_layer_b_v1",
            layer=EvaluationLayer.QUALIFICATION,
            case_source_id="sealed-b-v1",
            case_source_sha256=_hash("b"),
            case_count=64,
            pi_decision_record_id="decision-1",
        )


def test_scientific_plan_requires_frozen_pi_decision():
    with pytest.raises(ValueError, match="PI decision"):
        resolve_evaluation_plan(
            "scientific_layer_c_v1",
            layer=EvaluationLayer.CONFIRMATION,
            case_source_id="sealed-c-v1",
            case_source_sha256=_hash("c"),
            case_count=SCIENTIFIC_CASE_FLOOR,
        )


def test_scientific_profiles_are_fixed_to_their_layers():
    with pytest.raises(ValueError, match="fixed to layer_a"):
        resolve_evaluation_plan(
            "scientific_layer_a_v1",
            layer=EvaluationLayer.QUALIFICATION,
            case_source_id="wrong-layer",
            case_source_sha256=_hash("d"),
            case_count=SCIENTIFIC_CASE_FLOOR,
            pi_decision_record_id="decision-1",
        )


def test_scientific_layers_require_disjoint_case_sources():
    plans = tuple(
        resolve_evaluation_plan(
            profile,
            layer=layer,
            case_source_id=source_id,
            case_source_sha256=source_hash,
            case_count=SCIENTIFIC_CASE_FLOOR,
            pi_decision_record_id="decision-1",
        )
        for profile, layer, source_id, source_hash in (
            (
                "scientific_layer_a_v1",
                EvaluationLayer.SEARCH,
                "public-a-v1",
                _hash("a"),
            ),
            (
                "scientific_layer_b_v1",
                EvaluationLayer.QUALIFICATION,
                "sealed-b-v1",
                _hash("b"),
            ),
            (
                "scientific_layer_c_v1",
                EvaluationLayer.CONFIRMATION,
                "sealed-c-v1",
                _hash("c"),
            ),
        )
    )
    validate_disjoint_scientific_plans(plans)

    duplicate = (plans[0], plans[1], plans[2].__class__(
        **{
            **plans[2].__dict__,
            "case_source_id": plans[1].case_source_id,
            "case_source_sha256": plans[1].case_source_sha256,
        }
    ))
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_disjoint_scientific_plans(duplicate)

