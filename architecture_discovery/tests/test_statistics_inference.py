import json
import math

import pytest

from analysis.count_models import (
    blocked_count_log_likelihood,
    estimate_blocked_rate_ratio,
    estimate_nb2_dispersion,
    negative_binomial_log_likelihood,
    poisson_log_likelihood,
)
from analysis.multiplicity import adjust_pvalues
from analysis.outcomes import RunOutcome, RunOutcomeTable, RunTerminalStatus
from analysis.plan import (
    AnalysisPlan,
    MultiplicityMethod,
    freeze_analysis_plan,
    load_frozen_analysis_plan,
)
from analysis.randomization_inference import (
    Alternative,
    blocked_randomization_test,
)
from analysis.time_to_first import (
    DiscoveryRecord,
    ExposureUnit,
    derive_time_to_first,
    kaplan_meier,
)


def _paired_table(
    target_counts: tuple[int, ...],
    reference_counts: tuple[int, ...],
) -> RunOutcomeTable:
    assert len(target_counts) == len(reference_counts)
    rows = []
    for index, (target, reference) in enumerate(
        zip(target_counts, reference_counts, strict=True)
    ):
        for condition, count in (("C1", target), ("C0", reference)):
            rows.append(
                RunOutcome(
                    study_id="study",
                    block_id=f"block-{index}",
                    run_id=f"run-{index}-{condition}",
                    condition_id=condition,
                    run_seed=index,
                    terminal_status=RunTerminalStatus.COMPLETED,
                    qualifying_cluster_count=count,
                    proposal_exposure=20,
                    token_exposure=2_000,
                )
            )
    return RunOutcomeTable(tuple(rows), tuple(row.run_id for row in rows))


def test_dependency_free_poisson_and_nb2_likelihoods() -> None:
    poisson = poisson_log_likelihood((0, 2), (0.5, 2.0))
    expected = -0.5 + 2 * math.log(2.0) - 2.0 - math.lgamma(3)
    assert poisson == pytest.approx(expected)
    assert negative_binomial_log_likelihood(
        (0, 2), (0.5, 2.0), dispersion=0
    ) == pytest.approx(poisson)
    assert math.isfinite(
        negative_binomial_log_likelihood(
            (0, 2), (0.5, 2.0), dispersion=1.2
        )
    )
    assert estimate_nb2_dispersion((0, 0, 1, 8)) > 0


def test_blocked_model_and_rate_ratio_use_run_counts() -> None:
    table = _paired_table((2, 3, 4), (1, 1, 2))
    likelihood = blocked_count_log_likelihood(
        table,
        condition_log_rates={"C0": 0.0, "C1": math.log(2.0)},
        block_log_effects={f"block-{index}": 0.0 for index in range(3)},
        dispersion=0.5,
    )
    assert math.isfinite(likelihood)
    estimate = estimate_blocked_rate_ratio(
        table, target_condition="C1", reference_condition="C0"
    )
    assert estimate.blocks == 3
    assert estimate.rate_ratio == pytest.approx(9 / 4)
    assert estimate.continuity_correction == 0


def test_blocked_randomization_inference_is_exact_and_reproducible() -> None:
    table = _paired_table((1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0))
    first = blocked_randomization_test(
        table,
        target_condition="C1",
        reference_condition="C0",
        alternative=Alternative.GREATER,
        seed=99,
    )
    second = blocked_randomization_test(
        table,
        target_condition="C1",
        reference_condition="C0",
        alternative=Alternative.GREATER,
        seed=1,
    )
    assert first.exact is True
    assert first.p_value == pytest.approx(1 / 64)
    assert first == second.__class__(**{**second.__dict__, "seed": 99})


def test_time_to_first_uses_both_proposal_and_token_exposure() -> None:
    table = _paired_table((1, 0), (0, 0))
    discoveries = (
        DiscoveryRecord(
            run_id="run-0-C1",
            mechanism_cluster_id="cluster-late",
            opportunity_index=7,
            cumulative_generator_tokens=700,
            qualifies=True,
        ),
        DiscoveryRecord(
            run_id="run-0-C1",
            mechanism_cluster_id="cluster-first",
            opportunity_index=3,
            cumulative_generator_tokens=250,
            qualifies=True,
        ),
    )
    records = derive_time_to_first(table, discoveries)
    event = next(row for row in records if row.run_id == "run-0-C1")
    assert (event.proposal_time, event.token_time, event.event) == (3, 250, True)
    assert len(records) == len(table.rows)
    proposal_curve = kaplan_meier(records, exposure_unit=ExposureUnit.PROPOSALS)
    token_curve = kaplan_meier(records, exposure_unit=ExposureUnit.TOKENS)
    assert proposal_curve[0].exposure == 3
    assert token_curve[0].exposure == 250
    assert proposal_curve[-1].at_risk >= proposal_curve[-1].events


def test_multiplicity_adjustments_are_monotone_and_bounded() -> None:
    values = {"h1": 0.01, "h2": 0.04, "h3": 0.20}
    holm = {
        item.hypothesis_id: item.adjusted_p_value
        for item in adjust_pvalues(values, method=MultiplicityMethod.HOLM)
    }
    bh = {
        item.hypothesis_id: item.adjusted_p_value
        for item in adjust_pvalues(
            values, method=MultiplicityMethod.BENJAMINI_HOCHBERG
        )
    }
    assert holm == pytest.approx({"h1": 0.03, "h2": 0.08, "h3": 0.20})
    assert all(values[key] <= adjusted <= 1 for key, adjusted in bh.items())


def test_frozen_analysis_plan_rejects_overwrite_and_tampering(tmp_path) -> None:
    plan = AnalysisPlan.toy()
    path = tmp_path / "analysis-plan.json"
    frozen = freeze_analysis_plan(path, plan)
    assert load_frozen_analysis_plan(path) == frozen

    with pytest.raises(FileExistsError):
        freeze_analysis_plan(path, plan)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["plan"]["alpha"] = 0.10
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_analysis_plan(path)


def test_scientific_plan_cannot_reuse_toy_pi_decisions() -> None:
    payload = AnalysisPlan.toy().to_dict()
    payload["scientific"] = True
    with pytest.raises(ValueError, match="toy PI decisions"):
        AnalysisPlan.from_dict(payload)
