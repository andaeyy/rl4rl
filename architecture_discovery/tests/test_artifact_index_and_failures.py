import pytest

from artifacts import (
    INDEX_CATEGORIES,
    ArtifactContext,
    EventKind,
    FailureClass,
    FailureDomain,
    FailureRecord,
    RerunNotAuthorized,
    RerunPolicy,
    RunArtifactStore,
    authorize_rerun,
)


def _context() -> ArtifactContext:
    return ArtifactContext(
        study_id="study-index",
        block_id="block-0",
        run_id="run-index",
        condition_id="C3",
        writer_component="index-tests",
        code_sha256="1" * 64,
        config_sha256="2" * 64,
        environment_sha256="3" * 64,
        assignment_sha256="4" * 64,
    )


def test_artifact_index_covers_every_required_record_category(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    object_reference = store.objects.put_bytes(b"candidate source")
    for index, kind in enumerate(EventKind):
        store.append(
            kind,
            {
                "synthetic_id": f"record-{index}",
                "object_sha256s": [object_reference.sha256],
            },
        )

    artifact_index, index_reference = store.build_index()

    assert tuple(artifact_index.categories) == INDEX_CATEGORIES
    assert artifact_index.event_count == len(EventKind)
    assert sum(len(items) for items in artifact_index.categories.values()) == len(
        EventKind
    )
    assert all(
        entry.object_sha256s == (object_reference.sha256,)
        for entries in artifact_index.categories.values()
        for entry in entries
    )
    assert store.objects.read_json(index_reference) == artifact_index.to_dict()
    loaded_index = RunArtifactStore.open(store.root).load_index(index_reference)
    assert loaded_index == artifact_index
    assert store.verify_against_index(loaded_index).valid


def test_failure_taxonomy_rejects_a_mismatched_domain() -> None:
    with pytest.raises(ValueError, match="declared domain"):
        FailureRecord(
            failure_id="failure-1",
            attempt_id="attempt-1",
            failure_class=FailureClass.INVALID_TRANSFORMER,
            failure_domain=FailureDomain.INFRASTRUCTURE,
            stage="validity",
            terminal=True,
        )


def test_only_predeclared_infrastructure_failures_authorize_linked_reruns() -> None:
    policy = RerunPolicy(max_linked_attempts=2)
    failure = FailureRecord.create(
        attempt_id="attempt-original",
        failure_class=FailureClass.WORKER_CRASH,
        stage="training_worker",
    )

    authorization = authorize_rerun(
        assigned_run_id="assigned-run",
        previous_attempt_id="attempt-original",
        attempt_number=1,
        failure=failure,
        policy=policy,
    )

    assert authorization.assigned_run_id == "assigned-run"
    assert authorization.previous_attempt_id == "attempt-original"
    assert authorization.rerun_attempt_id.startswith("assigned-run-rerun-1-")
    assert authorization.triggering_failure_id == failure.failure_id


@pytest.mark.parametrize(
    "failure_class",
    [
        FailureClass.NONQUALIFYING_RESULT,
        FailureClass.PROPOSAL_PARSE,
        FailureClass.MPS_UNAVAILABLE,
        FailureClass.CONTAINMENT_UNAVAILABLE,
    ],
)
def test_scientific_candidate_and_unapproved_infrastructure_failures_do_not_rerun(
    failure_class: FailureClass,
) -> None:
    failure = FailureRecord.create(
        attempt_id="attempt-original",
        failure_class=failure_class,
        stage="synthetic-stage",
    )
    with pytest.raises(RerunNotAuthorized):
        authorize_rerun(
            assigned_run_id="assigned-run",
            previous_attempt_id="attempt-original",
            attempt_number=1,
            failure=failure,
            policy=RerunPolicy(),
        )


def test_rerun_must_link_the_failure_attempt_and_obey_attempt_cap() -> None:
    failure = FailureRecord.create(
        attempt_id="attempt-original",
        failure_class=FailureClass.POWER_INTERRUPTION,
        stage="host",
    )
    with pytest.raises(RerunNotAuthorized, match="previous attempt"):
        authorize_rerun(
            assigned_run_id="assigned-run",
            previous_attempt_id="unrelated-attempt",
            attempt_number=1,
            failure=failure,
            policy=RerunPolicy(),
        )
    with pytest.raises(RerunNotAuthorized, match="frozen policy"):
        authorize_rerun(
            assigned_run_id="assigned-run",
            previous_attempt_id="attempt-original",
            attempt_number=3,
            failure=failure,
            policy=RerunPolicy(max_linked_attempts=2),
        )
