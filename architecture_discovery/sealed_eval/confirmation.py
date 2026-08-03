"""Disabled-by-default, one-shot Layer C release authorization."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from common.evaluation_profiles import EvaluationLayer, EvaluationPlan
from evaluation.artifacts import EvaluationArtifactRoots
from evaluation.records import (
    ConfirmationEvaluationRecord,
    QualificationEvaluationRecord,
    RecordEnvelope,
    canonical_json,
    content_sha256,
    require_sha256,
    sha256_text,
    utc_now,
)
from sealed_eval.snapshot import FrozenRunSnapshot


@dataclass(frozen=True)
class ConfirmationReleaseManifest:
    authorization_id: str
    enabled: bool
    candidate_id: str
    frozen_snapshot_id: str
    frozen_candidate_sha256: str
    qualification_record_id: str
    qualification_record_sha256: str
    confirmation_plan_sha256: str
    confirmation_threshold: float
    token_sha256: str
    pi_release_record_id: str

    def validate(self) -> None:
        for name in (
            "authorization_id",
            "candidate_id",
            "frozen_snapshot_id",
            "qualification_record_id",
            "pi_release_record_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        for name in (
            "frozen_candidate_sha256",
            "qualification_record_sha256",
            "confirmation_plan_sha256",
            "token_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if not 0.0 <= self.confirmation_threshold <= 1.0:
            raise ValueError("confirmation_threshold must be in [0, 1]")

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class ConfirmationAuthorizationReceipt:
    authorization_id: str
    manifest_sha256: str
    consumed_at_utc: str
    consumption_marker_sha256: str


@dataclass(frozen=True)
class ConfirmationMeasurements:
    exact_match_accuracy: float
    metrics: tuple[tuple[str, float], ...] = ()
    complete: bool = True


class LayerCReleaseGate:
    """Persistently consume one authorization before Layer C can run."""

    def __init__(
        self,
        *,
        roots: EvaluationArtifactRoots,
        evaluation_plan: EvaluationPlan,
        enabled: bool = False,
    ) -> None:
        evaluation_plan.validate()
        if evaluation_plan.layer is not EvaluationLayer.CONFIRMATION:
            raise ValueError("LayerCReleaseGate requires a Layer C plan")
        if not evaluation_plan.sealed or evaluation_plan.controller_visible:
            raise ValueError("Layer C evaluation must be sealed and controller-hidden")
        roots.validate()
        self._roots = roots
        self._plan = evaluation_plan
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def authorize_once(
        self,
        *,
        manifest: ConfirmationReleaseManifest,
        token: str,
        snapshot: FrozenRunSnapshot,
        qualification: QualificationEvaluationRecord,
    ) -> ConfirmationAuthorizationReceipt:
        if not self._enabled:
            raise PermissionError("Layer C release is disabled")
        manifest.validate()
        if not manifest.enabled:
            raise PermissionError("release manifest is not enabled")
        if manifest.confirmation_plan_sha256 != self._plan.plan_hash:
            raise ValueError("release manifest references a different Layer C plan")
        if not token:
            raise PermissionError("Layer C confirmation token cannot be empty")
        if sha256_text(token) != manifest.token_sha256:
            raise PermissionError("invalid Layer C confirmation token")
        snapshot.validate(require_completed=True)
        candidate = snapshot.candidate(manifest.candidate_id)
        if snapshot.snapshot_id != manifest.frozen_snapshot_id:
            raise ValueError("release manifest snapshot mismatch")
        if candidate.artifact_sha256 != manifest.frozen_candidate_sha256:
            raise ValueError("release manifest candidate hash mismatch")
        if qualification.envelope.record_id != manifest.qualification_record_id:
            raise ValueError("release manifest qualification record mismatch")
        if qualification.record_hash != manifest.qualification_record_sha256:
            raise ValueError("qualification record content hash mismatch")
        if qualification.candidate_id != manifest.candidate_id:
            raise ValueError("qualification candidate mismatch")
        if qualification.frozen_snapshot_id != snapshot.snapshot_id:
            raise ValueError("qualification snapshot mismatch")
        if not qualification.evaluation_complete or not qualification.qualifies:
            raise PermissionError(
                "Layer C requires a completed, qualifying Layer B record"
            )

        marker_root = self._roots.layer_c / "release_consumed"
        marker_root.mkdir(parents=True, exist_ok=True)
        marker = (marker_root / f"{manifest.authorization_id}.json").resolve()
        if marker_root.resolve() not in marker.parents:
            raise ValueError("release authorization ID escaped the Layer C root")
        consumed_at = utc_now()
        marker_payload = {
            "authorization_id": manifest.authorization_id,
            "manifest_sha256": manifest.manifest_sha256,
            "consumed_at_utc": consumed_at,
        }
        marker_bytes = (canonical_json(marker_payload) + "\n").encode("utf-8")
        try:
            descriptor = os.open(
                marker,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise PermissionError("Layer C authorization was already consumed") from error
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(marker_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        import hashlib

        return ConfirmationAuthorizationReceipt(
            authorization_id=manifest.authorization_id,
            manifest_sha256=manifest.manifest_sha256,
            consumed_at_utc=consumed_at,
            consumption_marker_sha256=hashlib.sha256(marker_bytes).hexdigest(),
        )

    def build_confirmation_record(
        self,
        *,
        receipt: ConfirmationAuthorizationReceipt,
        manifest: ConfirmationReleaseManifest,
        snapshot: FrozenRunSnapshot,
        qualification: QualificationEvaluationRecord,
        measurements: ConfirmationMeasurements,
        envelope: RecordEnvelope,
    ) -> ConfirmationEvaluationRecord:
        if not self._enabled:
            raise PermissionError("Layer C release is disabled")
        manifest.validate()
        if receipt.authorization_id != manifest.authorization_id:
            raise ValueError("Layer C receipt does not match release manifest")
        if receipt.manifest_sha256 != manifest.manifest_sha256:
            raise ValueError("Layer C receipt manifest hash mismatch")
        marker = (
            self._roots.layer_c
            / "release_consumed"
            / f"{manifest.authorization_id}.json"
        ).resolve()
        if not marker.is_file():
            raise PermissionError("Layer C authorization has not been consumed")
        import hashlib

        if hashlib.sha256(marker.read_bytes()).hexdigest() != receipt.consumption_marker_sha256:
            raise PermissionError("Layer C authorization receipt is not authentic")
        snapshot.validate(require_completed=True)
        candidate = snapshot.candidate(manifest.candidate_id)
        if qualification.envelope.record_id != manifest.qualification_record_id:
            raise ValueError("Layer C qualification record mismatch")
        if qualification.record_hash != manifest.qualification_record_sha256:
            raise ValueError("Layer C qualification record was modified")
        if not qualification.evaluation_complete or not qualification.qualifies:
            raise PermissionError("Layer C requires a qualifying Layer B record")
        confirmed = (
            measurements.complete
            and measurements.exact_match_accuracy >= manifest.confirmation_threshold
        )
        record = ConfirmationEvaluationRecord(
            envelope=envelope,
            candidate_id=manifest.candidate_id,
            frozen_snapshot_id=snapshot.snapshot_id,
            frozen_candidate_sha256=candidate.artifact_sha256,
            qualification_record_id=qualification.envelope.record_id,
            release_authorization_id=receipt.authorization_id,
            evaluation_plan_sha256=self._plan.plan_hash,
            exact_match_accuracy=measurements.exact_match_accuracy,
            confirmed=confirmed,
            evaluation_complete=measurements.complete,
            sealed_metrics=measurements.metrics,
        )
        completion_root = self._roots.layer_c / "release_completed"
        completion_root.mkdir(parents=True, exist_ok=True)
        completion_marker = (
            completion_root / f"{manifest.authorization_id}.json"
        ).resolve()
        completion_payload = (
            canonical_json(
                {
                    "authorization_id": manifest.authorization_id,
                    "confirmation_record_id": record.envelope.record_id,
                    "confirmation_record_sha256": record.record_hash,
                }
            )
            + "\n"
        ).encode("utf-8")
        try:
            descriptor = os.open(
                completion_marker,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise PermissionError(
                "Layer C authorization already produced a confirmation record"
            ) from error
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(completion_payload)
            handle.flush()
            os.fsync(handle.fileno())
        return record


def write_release_manifest(
    path: str | Path, manifest: ConfirmationReleaseManifest
) -> None:
    """Write a release file with exclusive creation and owner-only permissions."""

    manifest.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(canonical_json(asdict(manifest)) + "\n")


def load_release_manifest(path: str | Path) -> ConfirmationReleaseManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Layer C release manifest must be a JSON object")
    manifest = ConfirmationReleaseManifest(**payload)
    manifest.validate()
    return manifest
