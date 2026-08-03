"""Crash-visible append-only event storage and content-addressed objects."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from artifacts.records import (
    GENESIS_EVENT_SHA256,
    ArtifactContext,
    EventKind,
    EventRecord,
    canonical_json,
    require_identifier,
    require_sha256,
    sha256_bytes,
)


_SEQUENCE_DIRECTORY = re.compile(r"^[0-9]{20}$")
_ATTEMPT_FILE = re.compile(r"^attempt-([0-9]{6})\.event$")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial file is intentionally retained as interruption evidence.
        raise
    _fsync_directory(path.parent)


class IntegritySeverity(StrEnum):
    WARNING = "warning"
    FATAL = "fatal"


@dataclass(frozen=True)
class IntegrityFinding:
    severity: IntegritySeverity
    code: str
    relative_path: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "relative_path": self.relative_path,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class IntegrityReport:
    events: tuple[EventRecord, ...]
    findings: tuple[IntegrityFinding, ...]
    last_event_sha256: str

    @property
    def valid(self) -> bool:
        return not any(
            finding.severity is IntegritySeverity.FATAL for finding in self.findings
        )


class ArtifactIntegrityError(RuntimeError):
    def __init__(self, report: IntegrityReport) -> None:
        self.report = report
        fatal = next(
            finding
            for finding in report.findings
            if finding.severity is IntegritySeverity.FATAL
        )
        super().__init__(f"{fatal.code}: {fatal.detail} ({fatal.relative_path})")


class RunLock(AbstractContextManager["RunLock"]):
    """Advisory per-run lock shared by every ledger writer and reader."""

    def __init__(self, path: Path, *, exclusive: bool) -> None:
        self.path = path
        self.exclusive = exclusive
        self._descriptor: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        operation = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


@dataclass(frozen=True)
class ObjectReference:
    sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("object reference requires a lowercase SHA-256 digest")
        if self.size_bytes < 0:
            raise ValueError("object size cannot be negative")
        if not self.media_type.strip():
            raise ValueError("object media type cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }


@dataclass(frozen=True)
class FrozenIndexReference:
    name: str
    run_id: str
    index_sha256: str
    event_count: int
    last_event_sha256: str
    object_reference: ObjectReference
    schema_name: str = "FrozenArtifactIndexReference"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        require_identifier(self.name, "name")
        require_identifier(self.run_id, "run_id")
        require_sha256(self.index_sha256, "index_sha256")
        require_sha256(self.last_event_sha256, "last_event_sha256")
        if self.event_count < 0:
            raise ValueError("frozen index event count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "name": self.name,
            "run_id": self.run_id,
            "index_sha256": self.index_sha256,
            "event_count": self.event_count,
            "last_event_sha256": self.last_event_sha256,
            "object_reference": self.object_reference.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenIndexReference":
        if payload.get("schema_name") != "FrozenArtifactIndexReference":
            raise ValueError("expected a frozen artifact-index reference")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported frozen index-reference schema")
        return cls(
            name=str(payload["name"]),
            run_id=str(payload["run_id"]),
            index_sha256=str(payload["index_sha256"]),
            event_count=int(payload["event_count"]),
            last_event_sha256=str(payload["last_event_sha256"]),
            object_reference=ObjectReference(**dict(payload["object_reference"])),
        )


class ContentAddressedObjectStore:
    """Immutable objects committed atomically under their SHA-256 digest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("object digest must be lowercase SHA-256")
        return self.root / "sha256" / sha256[:2] / sha256

    def put_bytes(
        self, payload: bytes, *, media_type: str = "application/octet-stream"
    ) -> ObjectReference:
        digest = sha256_bytes(payload)
        destination = self.path_for(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ArtifactIntegrityError(
                    IntegrityReport(
                        events=(),
                        findings=(
                            IntegrityFinding(
                                IntegritySeverity.FATAL,
                                "OBJECT_DIGEST_COLLISION",
                                str(destination),
                                "existing object bytes do not match their digest",
                            ),
                        ),
                        last_event_sha256=GENESIS_EVENT_SHA256,
                    )
                )
            return ObjectReference(digest, len(payload), media_type)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".pending", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.read_bytes() != payload:
                    raise RuntimeError("content-addressed object collision")
            _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return ObjectReference(digest, len(payload), media_type)

    def put_json(self, payload: Any) -> ObjectReference:
        return self.put_bytes(
            canonical_json(payload).encode("utf-8"),
            media_type="application/json",
        )

    def read_bytes(self, reference: ObjectReference | str) -> bytes:
        digest = reference.sha256 if isinstance(reference, ObjectReference) else reference
        payload = self.path_for(digest).read_bytes()
        if sha256_bytes(payload) != digest:
            raise ValueError("stored object digest mismatch")
        return payload

    def read_json(self, reference: ObjectReference | str) -> Any:
        return json.loads(self.read_bytes(reference))


class RunArtifactStore:
    """One immutable raw ledger plus object store for a single assigned run."""

    def __init__(self, root: str | Path, context: ArtifactContext) -> None:
        self.root = Path(root)
        self.context = context
        self.raw_events = self.root / "raw_events"
        self.lock_path = self.root / ".artifact-ledger.lock"
        self.objects = ContentAddressedObjectStore(self.root / "objects")
        self._initialize_context()

    def _initialize_context(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        context_path = self.root / "artifact_context.json"
        expected = (canonical_json(self.context.to_dict()) + "\n").encode("utf-8")
        try:
            _create_exclusive(context_path, expected)
        except FileExistsError:
            if context_path.read_bytes() != expected:
                raise ValueError("artifact directory belongs to a different run context")
        self.raw_events.mkdir(parents=True, exist_ok=True)
        _fsync_directory(self.root)

    @classmethod
    def open(cls, root: str | Path) -> "RunArtifactStore":
        root_path = Path(root)
        payload = json.loads((root_path / "artifact_context.json").read_text("utf-8"))
        return cls(root_path, ArtifactContext.from_dict(payload))

    def append(
        self, event_kind: EventKind | str, payload: Mapping[str, Any]
    ) -> EventRecord:
        with RunLock(self.lock_path, exclusive=True):
            report = self._scan_unlocked(tolerate_trailing_incomplete=True)
            if not report.valid:
                raise ArtifactIntegrityError(report)
            sequence = len(report.events) + 1
            sequence_directory = self.raw_events / f"{sequence:020d}"
            sequence_directory.mkdir(parents=True, exist_ok=True)
            existing_attempts = [
                int(match.group(1))
                for path in sequence_directory.iterdir()
                if (match := _ATTEMPT_FILE.match(path.name)) is not None
            ]
            attempt = max(existing_attempts, default=0) + 1
            record = EventRecord.create(
                context=self.context,
                sequence=sequence,
                event_kind=event_kind,
                payload=payload,
                previous_event_sha256=report.last_event_sha256,
            )
            destination = sequence_directory / f"attempt-{attempt:06d}.event"
            _create_exclusive(
                destination,
                (canonical_json(record.to_dict()) + "\n").encode("utf-8"),
            )
            _fsync_directory(self.raw_events)
            return record

    def scan(
        self, *, tolerate_trailing_incomplete: bool = True
    ) -> IntegrityReport:
        with RunLock(self.lock_path, exclusive=False):
            report = self._scan_unlocked(
                tolerate_trailing_incomplete=tolerate_trailing_incomplete
            )
        if not report.valid:
            raise ArtifactIntegrityError(report)
        return report

    def _scan_unlocked(
        self, *, tolerate_trailing_incomplete: bool
    ) -> IntegrityReport:
        events: list[EventRecord] = []
        findings: list[IntegrityFinding] = []
        last_hash = GENESIS_EVENT_SHA256
        incomplete_count = 0
        directories: list[tuple[int, Path]] = []
        for path in self.raw_events.iterdir():
            if path.is_dir() and _SEQUENCE_DIRECTORY.match(path.name):
                directories.append((int(path.name), path))
            else:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "UNEXPECTED_RAW_EVENT_ENTRY",
                        str(path.relative_to(self.root)),
                        "raw event storage may contain only numbered sequence directories",
                    )
                )
        directories.sort()
        for position, (sequence, directory) in enumerate(directories):
            expected_sequence = len(events) + 1
            if sequence != expected_sequence:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "EVENT_SEQUENCE_GAP",
                        str(directory.relative_to(self.root)),
                        f"expected sequence {expected_sequence}, found {sequence}",
                    )
                )
                break
            valid_records: list[EventRecord] = []
            incomplete_paths: list[Path] = []
            attempts = sorted(directory.iterdir())
            if not attempts:
                incomplete_paths.append(directory)
            attempt_numbers = [
                int(match.group(1))
                for attempt_path in attempts
                if (match := _ATTEMPT_FILE.match(attempt_path.name)) is not None
            ]
            if attempt_numbers and attempt_numbers != list(
                range(1, max(attempt_numbers) + 1)
            ):
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "EVENT_ATTEMPT_GAP",
                        str(directory.relative_to(self.root)),
                        "immutable event-attempt numbering is not contiguous",
                    )
                )
            for attempt_path in attempts:
                match = _ATTEMPT_FILE.match(attempt_path.name)
                relative = str(attempt_path.relative_to(self.root))
                if not attempt_path.is_file() or match is None:
                    findings.append(
                        IntegrityFinding(
                            IntegritySeverity.FATAL,
                            "UNEXPECTED_EVENT_ATTEMPT",
                            relative,
                            "sequence directories may contain only numbered event attempts",
                        )
                    )
                    continue
                raw = attempt_path.read_bytes()
                if not raw.endswith(b"\n"):
                    incomplete_paths.append(attempt_path)
                    continue
                try:
                    decoded = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    findings.append(
                        IntegrityFinding(
                            IntegritySeverity.FATAL,
                            "MALFORMED_COMPLETE_EVENT",
                            relative,
                            str(error),
                        )
                    )
                    continue
                try:
                    record = EventRecord.from_dict(decoded)
                    record.validate(context=self.context)
                except (TypeError, ValueError) as error:
                    findings.append(
                        IntegrityFinding(
                            IntegritySeverity.FATAL,
                            "INVALID_EVENT_HASH_OR_SCHEMA",
                            relative,
                            str(error),
                        )
                    )
                    continue
                valid_records.append(record)

            incomplete_count += len(incomplete_paths)
            if incomplete_count > 1:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "MULTIPLE_INCOMPLETE_EVENT_ATTEMPTS",
                        str(directory.relative_to(self.root)),
                        "only one crash-truncated event attempt can be tolerated",
                    )
                )
            if len(valid_records) > 1:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "DUPLICATE_COMMITTED_EVENT",
                        str(directory.relative_to(self.root)),
                        "a logical sequence has more than one complete event",
                    )
                )
            if not valid_records:
                is_last = position == len(directories) - 1
                if incomplete_paths and is_last and tolerate_trailing_incomplete:
                    findings.append(
                        IntegrityFinding(
                            IntegritySeverity.WARNING,
                            "TRAILING_INCOMPLETE_EVENT",
                            str(incomplete_paths[0].relative_to(self.root)),
                            "ignored one crash-truncated trailing event attempt",
                        )
                    )
                else:
                    findings.append(
                        IntegrityFinding(
                            IntegritySeverity.FATAL,
                            "MISSING_COMMITTED_EVENT",
                            str(directory.relative_to(self.root)),
                            "sequence has no valid committed event",
                        )
                    )
                break
            record = valid_records[0]
            if record.sequence != sequence:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "EVENT_SEQUENCE_MISMATCH",
                        str(directory.relative_to(self.root)),
                        "event envelope sequence differs from its directory",
                    )
                )
            if record.previous_event_sha256 != last_hash:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.FATAL,
                        "EVENT_CHAIN_BREAK",
                        str(directory.relative_to(self.root)),
                        "previous event hash does not match the accepted predecessor",
                    )
                )
            if incomplete_paths:
                findings.append(
                    IntegrityFinding(
                        IntegritySeverity.WARNING,
                        "RECOVERED_TRAILING_INCOMPLETE_EVENT",
                        str(incomplete_paths[0].relative_to(self.root)),
                        "a later immutable attempt recovered the interrupted append",
                    )
                )
            if any(item.severity is IntegritySeverity.FATAL for item in findings):
                break
            events.append(record)
            last_hash = record.event_sha256

        return IntegrityReport(tuple(events), tuple(findings), last_hash)

    def build_index(self):
        # Local import avoids an import cycle while keeping the public convenience API.
        from artifacts.index import ArtifactIndex

        report = self.scan()
        index = ArtifactIndex.from_events(self.context, report.events)
        for entries in index.categories.values():
            for entry in entries:
                for digest in entry.object_sha256s:
                    self.objects.read_bytes(digest)
        reference = self.objects.put_json(index.to_dict())
        return index, reference

    def verify_against_index(self, index) -> IntegrityReport:
        """Detect chain truncation or extension against a retained frozen index."""

        from artifacts.index import ArtifactIndex

        if not isinstance(index, ArtifactIndex):
            raise TypeError("index must be an ArtifactIndex")
        report = self.scan()
        rebuilt = ArtifactIndex.from_events(self.context, report.events)
        if rebuilt.to_dict() != index.to_dict():
            finding = IntegrityFinding(
                IntegritySeverity.FATAL,
                "FROZEN_INDEX_MISMATCH",
                ".",
                "raw ledger no longer matches the retained content-addressed index",
            )
            failed = IntegrityReport(
                report.events,
                (*report.findings, finding),
                report.last_event_sha256,
            )
            raise ArtifactIntegrityError(failed)
        return report

    def load_index(self, reference: ObjectReference | str):
        from artifacts.index import ArtifactIndex

        return ArtifactIndex.from_dict(self.objects.read_json(reference))

    def freeze_index(self, name: str = "final") -> FrozenIndexReference:
        """Freeze an exact index snapshot and an immutable discoverable CAS pointer."""

        if (
            not name
            or name != name.strip()
            or any(character in name for character in "/\\\x00")
        ):
            raise ValueError("frozen index name must be a safe identifier")
        index, object_reference = self.build_index()
        if object_reference.sha256 != index.index_sha256:
            raise RuntimeError("ArtifactIndex CAS digest differs from its canonical hash")
        frozen = FrozenIndexReference(
            name=name,
            run_id=self.context.run_id,
            index_sha256=index.index_sha256,
            event_count=index.event_count,
            last_event_sha256=index.last_event_sha256,
            object_reference=object_reference,
        )
        path = self.root / "frozen_indexes" / f"{name}.json"
        expected = (canonical_json(frozen.to_dict()) + "\n").encode("utf-8")
        try:
            _create_exclusive(path, expected)
        except FileExistsError:
            if path.read_bytes() != expected:
                raise ValueError(
                    f"frozen index {name!r} already refers to a different ledger head"
                )
        return frozen

    def load_frozen_index(
        self, name: str = "final"
    ) -> tuple[FrozenIndexReference, Any]:
        if (
            not name
            or name != name.strip()
            or any(character in name for character in "/\\\x00")
        ):
            raise ValueError("frozen index name must be a safe identifier")
        path = self.root / "frozen_indexes" / f"{name}.json"
        frozen = FrozenIndexReference.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if frozen.run_id != self.context.run_id:
            raise ValueError("frozen index reference belongs to another run")
        index = self.load_index(frozen.object_reference)
        if (
            index.index_sha256 != frozen.index_sha256
            or index.event_count != frozen.event_count
            or index.last_event_sha256 != frozen.last_event_sha256
        ):
            raise ValueError("frozen index reference does not match its CAS object")
        return frozen, index
