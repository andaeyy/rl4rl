import pytest

from analysis import RunTerminalStatus
from artifacts import (
    ArtifactContext,
    EventKind,
    FailureClass,
    FailureRecord,
    RerunPolicy,
    RunArtifactStore,
    authorize_rerun,
)
from reconstruction import (
    ReconstructionError,
    build_analysis_table,
    reconstruct_run,
    reconstruct_runs,
)


def _store(tmp_path, *, run_id: str = "run-1", condition_id: str = "C0"):
    context = ArtifactContext(
        study_id="study-reconstruct",
        block_id="block-1",
        run_id=run_id,
        condition_id=condition_id,
        writer_component="reconstruction-tests",
        code_sha256="a" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        run_seed=29,
        assignment_sha256="d" * 64,
    )
    return RunArtifactStore(tmp_path / run_id, context)


def _complete_run(store: RunArtifactStore) -> None:
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    store.append(
        EventKind.PROPOSAL,
        {"proposal_id": "proposal-1", "opportunity_index": 1},
    )
    store.append(
        EventKind.PARENT_SELECTION,
        {"selected_candidate_ids": ["seed-candidate"]},
    )
    store.append(
        EventKind.CANDIDATE,
        {
            "candidate_id": "candidate-1",
            "proposal_id": "proposal-1",
            "parent_candidate_ids": ["seed-candidate"],
        },
    )
    store.append(
        EventKind.TRAINING,
        {"training_id": "training-1", "candidate_id": "candidate-1"},
    )
    store.append(
        EventKind.SEARCH_EVALUATION,
        {"evaluation_id": "evaluation-1", "candidate_id": "candidate-1"},
    )
    store.append(
        EventKind.BUDGET,
        {
            "totals": {
                "proposal_opportunities": 1,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "training_steps": 8,
            }
        },
    )
    cluster_payload = {
        "cluster_id": "cluster-novel",
        "mechanism_cluster_key": "e" * 64,
        "run_ids": [store.context.run_id],
        "candidate_ids": ["candidate-1"],
        "qualifies_for_primary": True,
    }
    store.append(EventKind.MECHANISM_CLUSTER, cluster_payload)
    store.append(
        EventKind.MECHANISM_CLUSTER,
        {**cluster_payload, "cluster_id": "relabeled-same-mechanism"},
    )
    store.append(EventKind.PROMOTION, {"promotion_id": "promotion-1"})
    store.append(EventKind.REVIEW, {"review_id": "review-1"})
    store.append(EventKind.RUN_STATUS, {"status": "completed"})


def test_reconstructs_state_budget_ancestry_clusters_and_analysis_row(tmp_path) -> None:
    store = _store(tmp_path)
    _complete_run(store)

    run = reconstruct_run(store)

    assert run.status == "completed"
    assert run.budget_totals == {
        "completion_tokens": 40,
        "prompt_tokens": 100,
        "proposal_opportunities": 1,
        "training_steps": 8,
    }
    assert run.ancestry == {"candidate-1": ("seed-candidate",)}
    assert run.parent_selection_history == (("seed-candidate",),)
    assert run.qualifying_mechanism_cluster_keys == ("e" * 64,)
    assert run.promotion_ids == ("promotion-1",)
    assert run.review_ids == ("review-1",)
    assert run.outcome is not None
    assert run.outcome.terminal_status is RunTerminalStatus.COMPLETED
    assert run.outcome.qualifying_cluster_count == 1
    assert run.outcome.proposal_exposure == 1
    assert run.outcome.token_exposure == 140
    assert len(run.analysis_ready_rows()) == 1

    table = build_analysis_table((run,), assigned_run_ids=(store.context.run_id,))
    assert table.rows == (run.outcome,)
    assert table.rows[0].run_id == store.context.run_id


def test_candidate_failure_remains_an_itt_row_and_cannot_gain_a_rerun(tmp_path) -> None:
    store = _store(tmp_path)
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    store.append(
        EventKind.BUDGET,
        {"delta": {"proposal_opportunities": 1, "prompt_tokens": 10}},
    )
    failure = FailureRecord.create(
        attempt_id="attempt-1",
        failure_class=FailureClass.INVALID_TRANSFORMER,
        stage="runtime_validity",
    )
    store.append(EventKind.FAILURE, failure.to_event_payload())

    run = reconstruct_run(store)
    assert run.status == "candidate_failure"
    assert run.outcome is not None
    assert run.outcome.terminal_status is RunTerminalStatus.SCIENTIFIC_FAILURE
    assert run.outcome.itt_cluster_count == 0
    assert run.outcome.failure_class == FailureClass.INVALID_TRANSFORMER.value

    store.append(EventKind.RUN_STATUS, {"status": "running"})
    with pytest.raises(ReconstructionError, match="cannot return to running"):
        reconstruct_run(store)


def test_authorized_infrastructure_attempt_stays_linked_to_assigned_run(tmp_path) -> None:
    store = _store(tmp_path)
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    failure = FailureRecord.create(
        attempt_id="attempt-original",
        failure_class=FailureClass.WORKER_CRASH,
        stage="training_worker",
    )
    store.append(EventKind.FAILURE, failure.to_event_payload())
    authorization = authorize_rerun(
        assigned_run_id=store.context.run_id,
        previous_attempt_id="attempt-original",
        attempt_number=1,
        failure=failure,
        policy=RerunPolicy(),
    )
    store.append(EventKind.RERUN_ATTEMPT, authorization.to_event_payload())
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    store.append(EventKind.RUN_STATUS, {"status": "completed"})

    run = reconstruct_run(store)
    assert run.status == "completed"
    assert run.rerun_attempt_ids == (authorization.rerun_attempt_id,)
    assert run.failure_class == ""
    assert run.outcome is not None
    assert run.outcome.run_id == store.context.run_id
    assert run.outcome.terminal_status is RunTerminalStatus.COMPLETED


def test_reconstruction_rejects_decreasing_budgets_and_ancestry_cycles(tmp_path) -> None:
    budget_store = _store(tmp_path, run_id="budget-run")
    budget_store.append(EventKind.BUDGET, {"totals": {"prompt_tokens": 20}})
    budget_store.append(EventKind.BUDGET, {"totals": {"prompt_tokens": 19}})
    with pytest.raises(ReconstructionError, match="decreased"):
        reconstruct_run(budget_store)

    cycle_store = _store(tmp_path, run_id="cycle-run")
    cycle_store.append(
        EventKind.CANDIDATE,
        {"candidate_id": "candidate-a", "parent_candidate_ids": ["candidate-b"]},
    )
    cycle_store.append(
        EventKind.CANDIDATE,
        {"candidate_id": "candidate-b", "parent_candidate_ids": ["candidate-a"]},
    )
    with pytest.raises(ReconstructionError, match="cycle"):
        reconstruct_run(cycle_store)


def test_study_table_has_exactly_one_row_per_assigned_run(tmp_path) -> None:
    first_store = _store(tmp_path, run_id="run-a", condition_id="C0")
    second_store = _store(tmp_path, run_id="run-b", condition_id="C1")
    _complete_run(first_store)
    _complete_run(second_store)
    runs = reconstruct_runs((second_store, first_store))

    table = build_analysis_table(runs, assigned_run_ids=("run-a", "run-b"))
    assert {row.run_id for row in table.rows} == {"run-a", "run-b"}
    assert len(table.rows) == 2

    with pytest.raises(ReconstructionError, match="duplicate assigned runs"):
        reconstruct_runs((first_store, first_store))


def test_analysis_table_refuses_nonterminal_runs(tmp_path) -> None:
    store = _store(tmp_path)
    store.append(EventKind.RUN_STATUS, {"status": "running"})
    run = reconstruct_run(store)
    assert run.outcome is None
    with pytest.raises(ValueError, match="terminal assigned runs"):
        build_analysis_table((run,), assigned_run_ids=(store.context.run_id,))


def test_analysis_table_detects_an_omitted_assigned_run(tmp_path) -> None:
    store = _store(tmp_path, run_id="observed-run")
    _complete_run(store)
    observed = reconstruct_run(store)

    with pytest.raises(ValueError, match="frozen assignment roster"):
        build_analysis_table(
            (observed,), assigned_run_ids=("observed-run", "omitted-failed-run")
        )
