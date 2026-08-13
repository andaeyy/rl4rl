import json
from concurrent.futures import ThreadPoolExecutor
import pytest

from artifacts import (
    ArtifactContext,
    ArtifactIntegrityError,
    EventKind,
    RunArtifactStore,
)


def _context(run_id: str = "run-1") -> ArtifactContext:
    return ArtifactContext(
        study_id="study-1",
        block_id="block-1",
        run_id=run_id,
        condition_id="C0",
        writer_component="artifact-tests",
        code_sha256="a" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        run_seed=17,
        assignment_sha256="d" * 64,
    )


def test_append_builds_canonical_hash_chain_and_reopens(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    first = store.append(EventKind.PROPOSAL, {"proposal_id": "proposal-1"})
    second = store.append(
        EventKind.CANDIDATE,
        {
            "candidate_id": "candidate-1",
            "parent_candidate_ids": ["seed-candidate"],
        },
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert second.previous_event_sha256 == first.event_sha256
    report = RunArtifactStore.open(store.root).scan()
    assert report.valid
    assert report.events == (first, second)
    assert report.last_event_sha256 == second.event_sha256
    assert not report.findings


def test_content_addressed_objects_are_deduplicated_and_verified(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    first = store.objects.put_json({"candidate": "source", "version": 1})
    second = store.objects.put_json({"version": 1, "candidate": "source"})

    assert first.sha256 == second.sha256
    assert store.objects.read_json(first) == {"candidate": "source", "version": 1}
    object_path = store.objects.path_for(first.sha256)
    assert object_path.read_bytes() == store.objects.read_bytes(first)


def test_frozen_content_addressed_index_detects_trailing_chain_deletion(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    store.append(EventKind.PROPOSAL, {"proposal_id": "proposal-1"})
    store.append(EventKind.CANDIDATE, {"candidate_id": "candidate-1"})
    frozen_index, _ = store.build_index()

    trailing = store.raw_events / f"{2:020d}" / "attempt-000001.event"
    trailing.unlink()
    trailing.parent.rmdir()

    assert len(store.scan().events) == 1
    with pytest.raises(ArtifactIntegrityError, match="FROZEN_INDEX_MISMATCH"):
        store.verify_against_index(frozen_index)


@pytest.mark.parametrize("partial", [b"", b"{", b'{"almost":"complete"}'])
def test_resume_preserves_and_recovers_one_trailing_incomplete_attempt(
    tmp_path, partial: bytes
) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    first = store.append(EventKind.PROPOSAL, {"proposal_id": "proposal-1"})
    interrupted_directory = store.raw_events / f"{2:020d}"
    interrupted_directory.mkdir()
    interrupted_path = interrupted_directory / "attempt-000001.event"
    interrupted_path.write_bytes(partial)

    interrupted_report = store.scan()
    assert interrupted_report.events == (first,)
    assert [item.code for item in interrupted_report.findings] == [
        "TRAILING_INCOMPLETE_EVENT"
    ]

    recovered = store.append(EventKind.CANDIDATE, {"candidate_id": "candidate-1"})
    assert recovered.sequence == 2
    assert interrupted_path.read_bytes() == partial
    assert (interrupted_directory / "attempt-000002.event").exists()
    final_report = store.scan()
    assert len(final_report.events) == 2
    assert [item.code for item in final_report.findings] == [
        "RECOVERED_TRAILING_INCOMPLETE_EVENT"
    ]


def test_complete_malformed_or_hash_tampered_event_is_never_tolerated(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    store.append(EventKind.PROPOSAL, {"proposal_id": "proposal-1"})
    event_path = next((store.raw_events / f"{1:020d}").iterdir())
    event = json.loads(event_path.read_text("utf-8"))
    event["payload"]["proposal_id"] = "tampered"
    event_path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="INVALID_EVENT_HASH_OR_SCHEMA"):
        store.scan()


def test_only_one_incomplete_attempt_is_tolerated(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())
    directory = store.raw_events / f"{1:020d}"
    directory.mkdir()
    (directory / "attempt-000001.event").write_bytes(b"{")
    (directory / "attempt-000002.event").write_bytes(b"{")

    with pytest.raises(ArtifactIntegrityError, match="MULTIPLE_INCOMPLETE"):
        store.scan()


def test_per_run_lock_serializes_concurrent_appenders(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "run", _context())

    def append(index: int):
        return store.append(EventKind.PROPOSAL, {"proposal_id": f"proposal-{index}"})

    with ThreadPoolExecutor(max_workers=6) as executor:
        records = tuple(executor.map(append, range(24)))

    report = store.scan()
    assert len(records) == 24
    assert [record.sequence for record in report.events] == list(range(1, 25))
    assert len({record.record_id for record in report.events}) == 24


def test_run_context_is_immutable(tmp_path) -> None:
    root = tmp_path / "run"
    RunArtifactStore(root, _context("run-original"))

    with pytest.raises(ValueError, match="different run context"):
        RunArtifactStore(root, _context("run-collision"))
