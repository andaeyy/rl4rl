from pathlib import Path

import pytest

from artifacts import (
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    EventKind,
    ImmutableStudyEventSink,
    RunArtifactStore,
    StudyEventSinkError,
)
from reconstruction import build_analysis_table, reconstruct_run
from study.contracts import ConditionId, StudySpec
from study.fakes import DeterministicFakeEvaluator, DeterministicFakeGenerator
from study.randomization import generate_plan


def _fixture(tmp_path, *, study_id: str, opportunities: int = 2):
    study = StudySpec.toy(
        study_id=study_id,
        block_count=1,
        proposal_opportunities=opportunities,
    )
    plan = generate_plan(study, tmp_path / "study")
    run = next(
        item
        for item in plan.runs
        if item.condition.condition_id is ConditionId.C0
    )
    context = ArtifactContext(
        study_id=study.study_id,
        block_id=run.block_id,
        run_id=run.run_id,
        condition_id=run.condition.condition_id.value,
        writer_component="artifact-study-sink-tests",
        code_sha256="a" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        run_seed=run.run_seed,
        assignment_sha256=run.assignment_hash,
    )
    store = RunArtifactStore(Path(run.run_directory) / "immutable_artifacts", context)
    sink = ImmutableStudyEventSink(store)
    return study, run, store, sink


def test_instrumented_engine_maps_irreversible_transitions_and_freezes_indexes(
    tmp_path,
) -> None:
    study, run, store, sink = _fixture(
        tmp_path, study_id="artifact-sink-complete", opportunities=2
    )
    generator = DeterministicFakeGenerator()
    state = ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=generator,
        evaluator=DeterministicFakeEvaluator(accepted_opportunities={1}),
        artifact_sink=sink,
    ).execute()

    assert state.status == "completed"
    report = store.scan()
    transition_keys = [
        str(event.payload["transition_key"])
        for event in report.events
        if "transition_key" in event.payload
    ]
    assert len(transition_keys) == len(set(transition_keys))
    kinds = {event.event_kind for event in report.events}
    assert {
        EventKind.RUN_STATUS,
        EventKind.PROPOSAL,
        EventKind.CANDIDATE,
        EventKind.TRAINING,
        EventKind.SEARCH_EVALUATION,
        EventKind.PARENT_SELECTION,
        EventKind.BUDGET,
        EventKind.PROMOTION,
    } <= kinds

    search_pointer, search_index = store.load_frozen_index("search_completion")
    assert search_pointer.event_count == len(report.events)
    assert search_index.last_event_sha256 == report.last_event_sha256
    assert store.verify_against_index(search_index).valid

    final_pointer = sink.freeze_final_index()
    loaded_final_pointer, final_index = store.load_frozen_index("final")
    assert loaded_final_pointer == final_pointer
    assert final_pointer.object_reference.sha256 == search_pointer.object_reference.sha256
    assert store.verify_against_index(final_index).valid

    reconstructed = reconstruct_run(store)
    assert reconstructed.status == "completed"
    assert reconstructed.budget_totals["proposal_opportunities"] == 2
    assert reconstructed.budget_totals["candidate_training_attempts"] == 3
    assert study.initial_candidate_id in reconstructed.ancestry
    assert len(reconstructed.ancestry) == 3
    table = build_analysis_table(
        (reconstructed,), assigned_run_ids=(run.run_id,)
    )
    assert table.rows[0].qualifying_cluster_count == 0


def test_completed_resume_is_idempotent_and_does_not_repeat_generator(tmp_path) -> None:
    study, run, store, sink = _fixture(
        tmp_path, study_id="artifact-sink-resume", opportunities=1
    )
    generator = DeterministicFakeGenerator()
    evaluator = DeterministicFakeEvaluator()
    engine = ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=generator,
        evaluator=evaluator,
        artifact_sink=sink,
    )
    engine.execute()
    first_report = store.scan()
    first_pointer, _ = store.load_frozen_index("search_completion")

    reopened_store = RunArtifactStore.open(store.root)
    resumed = ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=generator,
        evaluator=evaluator,
        artifact_sink=ImmutableStudyEventSink(reopened_store),
    ).execute()

    assert resumed.status == "completed"
    assert len(generator.calls) == 1
    assert reopened_store.scan().events == first_report.events
    second_pointer, _ = reopened_store.load_frozen_index("search_completion")
    assert second_pointer == first_pointer

    reopened_store.append(
        EventKind.REVIEW,
        {"review_id": "post-search-review", "phase": "sealed_post_search"},
    )
    observation = ImmutableStudyEventSink(reopened_store).observe(resumed)
    assert observation.search_completion_index == first_pointer


def test_sink_records_parse_repair_and_nonterminal_candidate_failure(tmp_path) -> None:
    study, run, store, sink = _fixture(
        tmp_path, study_id="artifact-sink-repair", opportunities=1
    )
    state = ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=DeterministicFakeGenerator(parse_failures={1}),
        evaluator=DeterministicFakeEvaluator(),
        artifact_sink=sink,
    ).execute()

    assert state.status == "completed"
    report = store.scan()
    proposals = [event for event in report.events if event.event_kind is EventKind.PROPOSAL]
    repairs = [event for event in report.events if event.event_kind is EventKind.REPAIR]
    failures = [event for event in report.events if event.event_kind is EventKind.FAILURE]
    assert len([event for event in proposals if event.payload["phase"] == "provider_attempt_started"]) == 2
    assert len([event for event in proposals if event.payload["phase"] == "provider_response"]) == 2
    assert len(repairs) == 1
    assert repairs[0].payload["used_evaluation_feedback"] is False
    assert len(failures) == 1
    assert failures[0].payload["failure_domain"] == "candidate"
    assert failures[0].payload["terminal"] is False
    assert reconstruct_run(store).status == "completed"


def test_process_interruption_resumes_from_state_without_duplicate_events(tmp_path) -> None:
    class CrashOnceEvaluator(DeterministicFakeEvaluator):
        def __init__(self) -> None:
            super().__init__()
            self.crashed = False

        def evaluate_candidate(self, *args, **kwargs):
            if not self.crashed:
                self.crashed = True
                raise RuntimeError("synthetic process interruption")
            return super().evaluate_candidate(*args, **kwargs)

    study, run, store, sink = _fixture(
        tmp_path, study_id="artifact-sink-interruption", opportunities=1
    )
    generator = DeterministicFakeGenerator()
    evaluator = CrashOnceEvaluator()
    engine = ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=generator,
        evaluator=evaluator,
        artifact_sink=sink,
    )

    with pytest.raises(RuntimeError, match="interruption"):
        engine.execute()
    interrupted = store.scan()
    assert any(
        event.payload.get("transition_key") == "opportunity-1:candidate"
        for event in interrupted.events
    )

    completed = engine.execute()
    assert completed.status == "completed"
    assert len(generator.calls) == 1
    keys = [
        event.payload.get("transition_key")
        for event in store.scan().events
        if event.payload.get("transition_key") is not None
    ]
    assert len(keys) == len(set(keys))


def test_sink_fails_closed_on_conflicting_transition_or_premature_final_freeze(
    tmp_path,
) -> None:
    study, run, store, sink = _fixture(
        tmp_path, study_id="artifact-sink-conflict", opportunities=1
    )
    with pytest.raises(StudyEventSinkError, match="before search completion"):
        sink.freeze_final_index()

    store.append(
        EventKind.REVIEW,
        {
            "transition_key": "run:started",
            "review_id": "spoofed-transition",
        },
    )
    engine = ArtifactEmittingStudyEngine(
        study=study,
        run=run,
        generator=DeterministicFakeGenerator(),
        evaluator=DeterministicFakeEvaluator(),
        artifact_sink=sink,
    )
    with pytest.raises(StudyEventSinkError, match="conflicts"):
        engine.execute()
