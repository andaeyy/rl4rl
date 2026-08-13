"""Provider-free orchestration for externally authorized post-search evaluation.

This module is intentionally outside every controller dependency graph. It
accepts only a frozen search ledger, explicit artifact bindings, sealed plans,
and already-produced trusted measurements. It has no provider client and no
return channel into online parent selection.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from artifacts.index import ArtifactIndex
from artifacts.records import EventKind, require_identifier
from artifacts.store import FrozenIndexReference, RunArtifactStore
from common.evaluation_profiles import EvaluationLayer, EvaluationPlan
from evaluation.artifacts import EvaluationArtifactRoots, JsonEvaluationArtifactStore
from evaluation.records import (
    ConfirmationEvaluationRecord,
    QualificationEvaluationRecord,
    RecordEnvelope,
    content_sha256,
    require_bool,
    require_sha256,
)

from sealed_eval.confirmation import (
    ConfirmationMeasurements,
    ConfirmationReleaseManifest,
    LayerCReleaseGate,
)
from sealed_eval.qualification import (
    LayerBQualificationRunner,
    QualificationMeasurements,
    QualificationPolicy,
)
from sealed_eval.snapshot import FrozenRunSnapshot, freeze_completed_run

_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "authorization_id",
        "enabled",
        "run_id",
        "search_index_sha256",
        "snapshot_id",
        "budget_checkpoint_id",
        "qualification_plan_sha256",
        "qualification_policy_sha256",
        "decision_record_id",
        "candidate_ids",
    }
)
_CANDIDATE_BINDING_FIELDS = frozenset(
    {
        "candidate_id",
        "source_object_sha256",
        "checkpoint_relative_path",
        "checkpoint_sha256",
    }
)


def _safe_relative_path(value: object, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field_name} must be a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field_name} must be a safe POSIX relative path")
    if path.as_posix() != value:
        raise ValueError(f"{field_name} must be normalized")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class QualificationAuthorizationManifest:
    """External authorization that binds one Layer B run and candidate set."""

    authorization_id: str
    enabled: bool
    run_id: str
    search_index_sha256: str
    snapshot_id: str
    budget_checkpoint_id: str
    qualification_plan_sha256: str
    qualification_policy_sha256: str
    decision_record_id: str
    candidate_ids: tuple[str, ...]
    schema_name: str = "QualificationAuthorizationManifest"
    schema_version: str = "1.0"

    def validate(self) -> None:
        if (
            self.schema_name != "QualificationAuthorizationManifest"
            or self.schema_version != "1.0"
        ):
            raise ValueError("unsupported Layer B authorization schema")
        require_bool(self.enabled, "Layer B authorization enabled")
        for field_name in (
            "authorization_id",
            "run_id",
            "snapshot_id",
            "budget_checkpoint_id",
            "decision_record_id",
        ):
            require_identifier(getattr(self, field_name), field_name)
        for field_name in (
            "search_index_sha256",
            "qualification_plan_sha256",
            "qualification_policy_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if not self.candidate_ids:
            raise ValueError("Layer B authorization requires candidates")
        for candidate_id in self.candidate_ids:
            require_identifier(candidate_id, "candidate_id")
        if tuple(sorted(set(self.candidate_ids))) != self.candidate_ids:
            raise ValueError("Layer B candidate IDs must be sorted and unique")

    @property
    def authorization_sha256(self) -> str:
        self.validate()
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "candidate_ids": list(self.candidate_ids)}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> QualificationAuthorizationManifest:
        if not isinstance(payload, Mapping) or set(payload) != _AUTHORIZATION_FIELDS:
            raise ValueError("Layer B authorization has invalid fields")
        raw_candidate_ids = payload["candidate_ids"]
        if not isinstance(raw_candidate_ids, list):
            raise ValueError("Layer B candidate_ids must be an array")
        authorization = cls(
            authorization_id=payload["authorization_id"],
            enabled=payload["enabled"],
            run_id=payload["run_id"],
            search_index_sha256=payload["search_index_sha256"],
            snapshot_id=payload["snapshot_id"],
            budget_checkpoint_id=payload["budget_checkpoint_id"],
            qualification_plan_sha256=payload["qualification_plan_sha256"],
            qualification_policy_sha256=payload["qualification_policy_sha256"],
            decision_record_id=payload["decision_record_id"],
            candidate_ids=tuple(raw_candidate_ids),
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
        )
        authorization.validate()
        return authorization


@dataclass(frozen=True)
class SealedCandidateArtifactBinding:
    """Explicit source/checkpoint binding checked before snapshot creation."""

    candidate_id: str
    source_object_sha256: str
    checkpoint_relative_path: str
    checkpoint_sha256: str

    def validate(self) -> None:
        require_identifier(self.candidate_id, "candidate_id")
        require_sha256(self.source_object_sha256, "source_object_sha256")
        require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        _safe_relative_path(
            self.checkpoint_relative_path, "checkpoint_relative_path"
        )

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> SealedCandidateArtifactBinding:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != _CANDIDATE_BINDING_FIELDS
        ):
            raise ValueError("sealed candidate binding has invalid fields")
        binding = cls(**dict(payload))
        binding.validate()
        return binding


@dataclass(frozen=True)
class QualificationBatchResult:
    snapshot: FrozenRunSnapshot
    records: tuple[QualificationEvaluationRecord, ...]
    search_index: FrozenIndexReference


@dataclass(frozen=True)
class ConfirmationResult:
    record: ConfirmationEvaluationRecord
    final_index: FrozenIndexReference


class SealedPostSearchOrchestrator:
    """Append sealed evaluation evidence without touching controller state."""

    def __init__(
        self,
        *,
        artifact_store: RunArtifactStore,
        evaluation_roots: EvaluationArtifactRoots,
        checkpoint_root: str | Path,
    ) -> None:
        self._store = artifact_store
        self._roots = evaluation_roots
        self._roots.prepare()
        raw_checkpoint_root = Path(checkpoint_root)
        if raw_checkpoint_root.is_symlink():
            raise ValueError("checkpoint root may not be a symbolic link")
        self._checkpoint_root = raw_checkpoint_root.resolve()
        if not self._checkpoint_root.is_dir():
            raise FileNotFoundError("checkpoint root is missing")

    def _verified_search_prefix(
        self,
    ) -> tuple[FrozenIndexReference, ArtifactIndex, tuple[Any, ...]]:
        frozen, index = self._store.load_frozen_index("search_completion")
        report = self._store.scan(tolerate_trailing_incomplete=False)
        if len(report.events) < frozen.event_count:
            raise ValueError("run ledger is shorter than its frozen search index")
        prefix = report.events[: frozen.event_count]
        rebuilt = ArtifactIndex.from_events(self._store.context, prefix)
        if rebuilt.to_dict() != index.to_dict():
            raise ValueError("run ledger search prefix differs from its frozen index")
        if not prefix:
            raise ValueError("search completion index cannot be empty")
        terminal = prefix[-1]
        if (
            terminal.event_kind is not EventKind.RUN_STATUS
            or terminal.payload.get("transition_key") != "run:completed"
            or terminal.payload.get("status") != "completed"
        ):
            raise ValueError("search index does not end in a completed run event")
        return frozen, index, prefix

    @staticmethod
    def _candidate_sources(prefix: tuple[Any, ...]) -> dict[str, str]:
        candidates: dict[str, str] = {}
        for event in prefix:
            if event.event_kind is not EventKind.CANDIDATE:
                continue
            candidate_id = event.payload.get("candidate_id")
            source_sha256 = event.payload.get("source_object_sha256")
            require_identifier(candidate_id, "candidate event candidate_id")
            require_sha256(source_sha256, "candidate event source_object_sha256")
            if candidate_id in candidates:
                raise ValueError("search ledger contains a duplicate candidate ID")
            candidates[candidate_id] = source_sha256
        return candidates

    def _checkpoint_path(self, binding: SealedCandidateArtifactBinding) -> Path:
        relative = _safe_relative_path(
            binding.checkpoint_relative_path, "checkpoint_relative_path"
        )
        current = self._checkpoint_root
        for component in relative.parts:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"sealed checkpoint is missing: {relative.as_posix()}"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("sealed checkpoint path may not traverse symlinks")
        if not stat.S_ISREG(current.lstat().st_mode):
            raise ValueError("sealed checkpoint must be a regular file")
        canonical = current.resolve()
        if (
            canonical.parent != self._checkpoint_root
            and self._checkpoint_root not in canonical.parents
        ):
            raise ValueError("sealed checkpoint escaped its trusted root")
        return current

    def freeze_authorized_snapshot(
        self,
        *,
        authorization: QualificationAuthorizationManifest,
        evaluation_plan: EvaluationPlan,
        policy: QualificationPolicy,
        candidate_bindings: tuple[SealedCandidateArtifactBinding, ...],
    ) -> tuple[FrozenRunSnapshot, FrozenIndexReference]:
        authorization.validate()
        if authorization.enabled is not True:
            raise PermissionError("Layer B authorization is disabled")
        evaluation_plan.validate()
        if evaluation_plan.layer is not EvaluationLayer.QUALIFICATION:
            raise ValueError("Layer B authorization requires a qualification plan")
        policy.validate()
        policy_sha256 = content_sha256(asdict(policy))
        if authorization.qualification_plan_sha256 != evaluation_plan.plan_hash:
            raise ValueError("Layer B authorization references another plan")
        if authorization.qualification_policy_sha256 != policy_sha256:
            raise ValueError("Layer B authorization references another policy")
        if authorization.decision_record_id != policy.decision_record_id:
            raise ValueError("Layer B authorization references another decision")
        if (
            evaluation_plan.scientific
            and evaluation_plan.pi_decision_record_id
            != authorization.decision_record_id
        ):
            raise ValueError("scientific Layer B inputs use different decisions")

        frozen, _index, prefix = self._verified_search_prefix()
        if authorization.run_id != self._store.context.run_id:
            raise ValueError("Layer B authorization belongs to another run")
        if authorization.search_index_sha256 != frozen.index_sha256:
            raise ValueError("Layer B authorization references another search index")
        binding_ids = tuple(
            sorted(binding.candidate_id for binding in candidate_bindings)
        )
        if binding_ids != authorization.candidate_ids:
            raise ValueError(
                "Layer B artifact bindings differ from authorized candidates"
            )
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("Layer B artifact bindings contain duplicate candidates")

        source_events = self._candidate_sources(prefix)
        artifact_hashes: dict[str, tuple[str, str]] = {}
        for binding in candidate_bindings:
            binding.validate()
            if source_events.get(binding.candidate_id) != binding.source_object_sha256:
                raise ValueError("candidate source binding differs from search ledger")
            source_object_path = self._store.objects.path_for(
                binding.source_object_sha256
            )
            source_metadata = os.stat(source_object_path, follow_symlinks=False)
            if not stat.S_ISREG(source_metadata.st_mode):
                raise ValueError("candidate source object must be a regular file")
            self._store.objects.read_bytes(binding.source_object_sha256)
            checkpoint = self._checkpoint_path(binding)
            if _sha256_file(checkpoint) != binding.checkpoint_sha256:
                raise ValueError("candidate checkpoint SHA-256 mismatch")
            artifact_hashes[binding.candidate_id] = (
                binding.source_object_sha256,
                binding.checkpoint_sha256,
            )

        snapshot = freeze_completed_run(
            snapshot_id=authorization.snapshot_id,
            run_id=authorization.run_id,
            budget_checkpoint_id=authorization.budget_checkpoint_id,
            terminal_event_sha256=frozen.last_event_sha256,
            candidate_artifacts=artifact_hashes,
            run_complete=True,
        )
        JsonEvaluationArtifactStore(
            self._roots, EvaluationLayer.QUALIFICATION
        ).write_json(snapshot.snapshot_id, snapshot.to_dict())
        return snapshot, frozen

    def qualify(
        self,
        *,
        authorization: QualificationAuthorizationManifest,
        evaluation_plan: EvaluationPlan,
        policy: QualificationPolicy,
        candidate_bindings: tuple[SealedCandidateArtifactBinding, ...],
        measurements: Mapping[str, QualificationMeasurements],
    ) -> QualificationBatchResult:
        snapshot, search_index = self.freeze_authorized_snapshot(
            authorization=authorization,
            evaluation_plan=evaluation_plan,
            policy=policy,
            candidate_bindings=candidate_bindings,
        )
        if set(measurements) != set(authorization.candidate_ids):
            raise ValueError("Layer B measurements differ from authorized candidates")
        runner = LayerBQualificationRunner(
            evaluation_plan=evaluation_plan,
            policy=policy,
        )
        artifact_writer = JsonEvaluationArtifactStore(
            self._roots, EvaluationLayer.QUALIFICATION
        )
        records: list[QualificationEvaluationRecord] = []
        context = self._store.context
        for candidate_id in authorization.candidate_ids:
            record = runner.evaluate_frozen_candidate(
                snapshot=snapshot,
                candidate_id=candidate_id,
                measurements=measurements[candidate_id],
                envelope=RecordEnvelope.create(
                    schema_name="qualification_evaluation",
                    record_id=(
                        f"qualification-{authorization.authorization_id}-{candidate_id}"
                    ),
                    study_id=context.study_id,
                    block_id=context.block_id,
                    run_id=context.run_id,
                    condition_id=context.condition_id,
                    writer_component="sealed_eval.orchestration",
                    code_sha256=context.code_sha256,
                    config_sha256=context.config_sha256,
                    environment_sha256=context.environment_sha256,
                ),
            )
            artifact_writer.write_json(record.envelope.record_id, record.to_dict())
            record_object = self._store.objects.put_json(record.to_dict())
            if record_object.sha256 != record.record_hash:
                raise RuntimeError("Layer B record object hash mismatch")
            self._store.append(
                EventKind.QUALIFICATION_EVALUATION,
                {
                    "authorization_id": authorization.authorization_id,
                    "authorization_sha256": authorization.authorization_sha256,
                    "candidate_id": candidate_id,
                    "frozen_snapshot_id": snapshot.snapshot_id,
                    "frozen_snapshot_sha256": snapshot.snapshot_sha256,
                    "qualification_record_id": record.envelope.record_id,
                    "qualification_record_sha256": record.record_hash,
                    "evaluation_plan_sha256": evaluation_plan.plan_hash,
                    "qualifies": record.qualifies,
                    "evaluation_complete": record.evaluation_complete,
                    "object_sha256s": [record_object.sha256],
                },
            )
            records.append(record)
        return QualificationBatchResult(snapshot, tuple(records), search_index)

    def confirm_once(
        self,
        *,
        snapshot: FrozenRunSnapshot,
        qualification: QualificationEvaluationRecord,
        evaluation_plan: EvaluationPlan,
        release_manifest: ConfirmationReleaseManifest,
        release_token: str,
        measurements: ConfirmationMeasurements,
    ) -> ConfirmationResult:
        evaluation_plan.validate()
        if evaluation_plan.layer is not EvaluationLayer.CONFIRMATION:
            raise ValueError("Layer C release requires a confirmation plan")
        context = self._store.context
        if (
            snapshot.run_id != context.run_id
            or qualification.envelope.run_id != context.run_id
        ):
            raise ValueError("Layer C inputs belong to another run")
        gate = LayerCReleaseGate(
            roots=self._roots,
            evaluation_plan=evaluation_plan,
            enabled=True,
        )
        receipt = gate.authorize_once(
            manifest=release_manifest,
            token=release_token,
            snapshot=snapshot,
            qualification=qualification,
        )
        record = gate.build_confirmation_record(
            receipt=receipt,
            manifest=release_manifest,
            snapshot=snapshot,
            qualification=qualification,
            measurements=measurements,
            envelope=RecordEnvelope.create(
                schema_name="confirmation_evaluation",
                record_id=f"confirmation-{release_manifest.authorization_id}",
                study_id=context.study_id,
                block_id=context.block_id,
                run_id=context.run_id,
                condition_id=context.condition_id,
                writer_component="sealed_eval.orchestration",
                code_sha256=context.code_sha256,
                config_sha256=context.config_sha256,
                environment_sha256=context.environment_sha256,
            ),
        )
        JsonEvaluationArtifactStore(
            self._roots, EvaluationLayer.CONFIRMATION
        ).write_json(record.envelope.record_id, record.to_dict())
        record_object = self._store.objects.put_json(record.to_dict())
        if record_object.sha256 != record.record_hash:
            raise RuntimeError("Layer C record object hash mismatch")
        self._store.append(
            EventKind.CONFIRMATION_EVALUATION,
            {
                "release_authorization_id": release_manifest.authorization_id,
                "release_manifest_sha256": release_manifest.manifest_sha256,
                "candidate_id": record.candidate_id,
                "frozen_snapshot_id": record.frozen_snapshot_id,
                "qualification_record_id": record.qualification_record_id,
                "confirmation_record_id": record.envelope.record_id,
                "confirmation_record_sha256": record.record_hash,
                "evaluation_plan_sha256": evaluation_plan.plan_hash,
                "confirmed": record.confirmed,
                "evaluation_complete": record.evaluation_complete,
                "object_sha256s": [record_object.sha256],
            },
        )
        final_index = self._store.freeze_index("final")
        return ConfirmationResult(record, final_index)
