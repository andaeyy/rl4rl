import pytest

from analysis.intent_to_treat import (
    failure_class_counts,
    infrastructure_failure_sensitivity,
    intent_to_treat_summary,
)
from analysis.outcomes import (
    PilotDataset,
    RunOutcome,
    RunOutcomeTable,
    RunTerminalStatus,
)
from analysis.power import estimate_pilot_count_parameters


def _row(
    run_id: str,
    condition: str,
    *,
    block: str = "block-1",
    status: RunTerminalStatus = RunTerminalStatus.COMPLETED,
    count: int | None = 1,
    failure_class: str = "",
) -> RunOutcome:
    return RunOutcome(
        study_id="study",
        block_id=block,
        run_id=run_id,
        condition_id=condition,
        run_seed=1,
        terminal_status=status,
        qualifying_cluster_count=count,
        proposal_exposure=10,
        token_exposure=1_000,
        failure_class=failure_class,
    )


def test_candidate_records_cannot_enter_the_independent_run_table() -> None:
    candidate_record = {
        "schema_name": "CandidateRecord",
        "study_id": "study",
        "run_id": "run",
        "candidate_id": "candidate-99",
    }
    with pytest.raises(ValueError, match="candidate rows are not replicates"):
        RunOutcomeTable.from_records(
            [candidate_record], assigned_run_ids=("run",)
        )


def test_duplicate_run_rows_are_rejected_as_pseudoreplication() -> None:
    row = _row("run-1", "C0")
    with pytest.raises(ValueError, match="pseudoreplication"):
        RunOutcomeTable((row, row), ("run-1",))


def test_missing_failed_assignment_cannot_be_dropped_from_analysis() -> None:
    completed = _row("completed", "C0")
    with pytest.raises(ValueError, match=r"missing=\['failed-assignment'\]"):
        RunOutcomeTable(
            (completed,),
            ("completed", "failed-assignment"),
        )


def test_failed_assigned_runs_remain_in_intent_to_treat() -> None:
    rows = (
            _row("c0-good", "C0", block="b1", count=2),
            _row(
                "c0-science-failure",
                "C0",
                block="b2",
                status=RunTerminalStatus.SCIENTIFIC_FAILURE,
                count=None,
                failure_class="candidate_invalid",
            ),
            _row("c1-good", "C1", block="b1", count=4),
            _row(
                "c1-infra-failure",
                "C1",
                block="b2",
                status=RunTerminalStatus.INFRASTRUCTURE_FAILURE,
                count=None,
                failure_class="host_crash",
            ),
        )
    table = RunOutcomeTable(rows, tuple(row.run_id for row in rows))
    summaries = {item.condition_id: item for item in intent_to_treat_summary(table)}
    assert summaries["C0"].assigned_runs == 2
    assert summaries["C0"].scientific_failures == 1
    assert summaries["C0"].mean_clusters_per_assigned_run == 1.0
    assert summaries["C1"].assigned_runs == 2
    assert summaries["C1"].infrastructure_failures == 1
    assert summaries["C1"].mean_clusters_per_assigned_run == 2.0

    failures = {(item.failure_class, item.count) for item in failure_class_counts(table)}
    assert failures == {("candidate_invalid", 1), ("host_crash", 1)}
    sensitivity = infrastructure_failure_sensitivity(table)
    assert sensitivity.excluded_infrastructure_run_ids == ("c1-infra-failure",)
    retained = {item.condition_id: item for item in sensitivity.summaries}
    assert retained["C0"].assigned_runs == 2  # scientific failure stays
    assert retained["C1"].assigned_runs == 1


def test_pilot_schema_uses_run_rows_and_estimates_overdispersion() -> None:
    rows = tuple(
            _row(f"run-{index}", "C0", block=f"b{index}", count=count)
            for index, count in enumerate((0, 0, 1, 7), start=1)
        )
    table = RunOutcomeTable(rows, tuple(row.run_id for row in rows))
    pilot = PilotDataset(
        pilot_id="pilot-v1",
        outcomes=table,
        extraction_rule_hash="frozen-extraction",
    )
    restored = PilotDataset.from_dict(pilot.to_dict())
    assert restored.dataset_hash == pilot.dataset_hash
    estimate = estimate_pilot_count_parameters(restored, condition_id="C0")
    assert estimate.assigned_runs == 4
    assert estimate.zero_fraction == 0.5
    assert estimate.dispersion_nb2 > 0
