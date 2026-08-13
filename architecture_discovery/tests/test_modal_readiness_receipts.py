from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import modal_action_journal as modal_global_journal
import pytest
import scripts.capture_modal_cleanup_snapshots as cleanup_capture
import scripts.provider_canary_plan as provider_plan
import scripts.record_modal_readiness as modal_readiness
from common.provider_attempts import ProviderAttemptRecord
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    APP_NAME,
    ArtifactIntegrityError,
    ArtifactVerificationV1,
    ImageSourceManifestV1,
    ModalLiveCohortIdentity,
    SourceFileV1,
    build_artifact_manifest,
    load_raw_artifact_manifest,
    volume_artifact_uri,
    write_artifact_manifest,
)
from scripts.record_local_engineering_evidence import LOCAL_ENGINEERING_FREEZE_ROOT
from scripts.record_modal_readiness import (
    MODAL_READINESS_RECEIPT_CONTRACTS,
    capture_remote_verification,
    record_resource_cleanup,
    validate_modal_readiness_gate_record,
    validate_modal_readiness_receipt,
)
from scripts.record_modal_readiness import (
    record_artifact_round_trip as _record_artifact_round_trip,
)
from scripts.record_modal_readiness import (
    record_cuda_environment as _record_cuda_environment,
)

_FIXTURE_SOURCE_TREE_SHA256 = "9" * 64
_FIXTURE_COHORT_ID = "cleanup-check-1"


def _fixture_image_manifest(root: Path) -> ImageSourceManifestV1:
    return ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=_bound_policy_source_files(root),
    )


def _fixture_identity(root: Path) -> ModalLiveCohortIdentity:
    return ModalLiveCohortIdentity(
        source_tree_sha256=_FIXTURE_SOURCE_TREE_SHA256,
        image_source_sha256=_fixture_image_manifest(root).manifest_sha256,
        cohort_id=_FIXTURE_COHORT_ID,
    )


def _provider_plan_for_fixture(
    identity: ModalLiveCohortIdentity,
    *,
    preflight_path: str,
    preflight_sha256: str,
    dependency_lock_sha256: str | None = None,
) -> dict[str, object]:
    project = modal_readiness.ROOT
    current_identity = ModalLiveCohortIdentity(
        source_tree_sha256=provider_plan.compute_source_tree_sha256(project),
        image_source_sha256=(
            modal_readiness.build_image_source_manifest(project).manifest_sha256
        ),
        cohort_id="provider-plan-fixture-template",
    )
    current_preflight = (
        modal_readiness.modal_live_cohort_root(current_identity)
        / "components/candidate_resume_preflight_receipts/v2.0"
        / (("a" * 64) + ".json")
    )
    plan = modal_readiness.build_provider_canary_approval_plan(
        project,
        source_tree_sha256=current_identity.source_tree_sha256,
        cohort_id=current_identity.cohort_id,
        candidate_resume_preflight_receipt_path=current_preflight.as_posix(),
        candidate_resume_preflight_receipt_sha256="b" * 64,
    )
    plan.update(modal_readiness.modal_cohort_identity_dict(identity))
    plan["candidate_resume_preflight_receipt"] = {
        "path": preflight_path,
        "sha256": preflight_sha256,
    }
    if dependency_lock_sha256 is not None:
        plan["dependency_lock_sha256"] = dependency_lock_sha256
    plan.pop("approval_plan_sha256")
    plan["approval_plan_sha256"] = modal_readiness.canonical_sha256(plan)
    return plan


def _verifier_attempt_id(run_id: str) -> str:
    return hashlib.sha256(f"verifier:{run_id}".encode()).hexdigest()[:32]


def _attempt_root(root: Path) -> Path:
    return root / modal_readiness.modal_action_attempt_directory(
        _fixture_identity(root)
    )


def _intent_logical(root: Path, attempt_id: str) -> str:
    return modal_readiness.modal_action_intent_receipt_path(
        _fixture_identity(root), attempt_id
    ).as_posix()


def _terminal_logical(root: Path, attempt_id: str) -> str:
    return modal_readiness.modal_action_terminal_receipt_path(
        _fixture_identity(root), attempt_id
    ).as_posix()


def _fixture_local_containment() -> dict[str, object]:
    host_sha256 = hashlib.sha256(b"modal-readiness-fixture-host").hexdigest()
    boot_identity = bytes.fromhex("01" * 16)
    session_sha256 = hashlib.sha256(
        b"RL4RL ModalLocalBootSessionBinding v2\0"
        + bytes.fromhex(host_sha256)
        + boot_identity
    ).hexdigest()
    return {
        "local_host_anchor_path": (
            modal_readiness.modal_local_host_anchor_path().as_posix()
        ),
        "local_host_anchor_sha256": host_sha256,
        "local_boot_started_at_unix_microseconds": 1_735_689_599_000_000,
        "local_boot_session_sha256": session_sha256,
    }


def _remote_run_reservation_bindings(
    root: Path,
    identity: ModalLiveCohortIdentity,
    *,
    attempt_id: str,
    action: str,
    run_ids: list[str],
    created_at_utc: str,
    launch_capability_sha256: str,
) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for run_id in run_ids:
        logical = modal_readiness.modal_remote_run_reservation_path(
            run_id
        ).as_posix()
        payload = {
            "schema_name": "ModalRemoteRunReservation",
            "schema_version": "1.2",
            "remote_run_id": run_id,
            "owner_attempt_id": attempt_id,
            "action": action,
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "modal_environment": modal_readiness.MODAL_ENVIRONMENT,
            "created_at_utc": created_at_utc,
            "launch_capability_sha256": launch_capability_sha256,
            **_fixture_local_containment(),
        }
        path = root / logical
        modal_readiness.create_json_exclusive(path, payload)
        bindings.append(
            {
                "run_id": run_id,
                "path": logical,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return bindings


def _refresh_receipt_reservations(
    root: Path,
    receipt: dict[str, object],
    *,
    identity: ModalLiveCohortIdentity | None = None,
) -> None:
    selected_identity = identity or _fixture_identity(root)
    attempt_id = str(receipt["attempt_id"])
    action = str(receipt["action"])
    run_ids = list(receipt["concrete_remote_run_ids"])
    capability = hashlib.sha256(
        f"capability:{attempt_id}".encode()
    ).hexdigest()
    receipt["launch_capability_sha256"] = capability
    receipt["remote_run_reservations"] = _remote_run_reservation_bindings(
        root,
        selected_identity,
        attempt_id=attempt_id,
        action=action,
        run_ids=run_ids,
        created_at_utc=str(receipt["started_at_utc"]),
        launch_capability_sha256=capability,
    )


def record_cuda_environment(*, run_id: str, root: Path) -> dict[str, object]:
    return _record_cuda_environment(
        run_id=run_id,
        cohort_id=_FIXTURE_COHORT_ID,
        root=root,
    )


def record_artifact_round_trip(
    *,
    source_run_id: str,
    verifier_run_id: str,
    root: Path,
) -> dict[str, object]:
    return _record_artifact_round_trip(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        verifier_attempt_id=_verifier_attempt_id(source_run_id),
        cohort_id=_FIXTURE_COHORT_ID,
        root=root,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_snapshot_capture_manifest(snapshot_root: Path) -> None:
    """Refresh one synthetic capture and its test-only roster binding."""

    cohort_root = snapshot_root.parents[2]
    project_root = cohort_root.parents[6]
    roster_path = cohort_root / "cohort_roster.v4.0.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    capture_id = snapshot_root.name
    commands = cleanup_capture.build_modal_cleanup_snapshot_commands(
        modal_executable=Path("/dev/fd/123"),
        billing_window_start_utc=roster["billing_window_start_utc"],
        billing_window_end_utc=roster["billing_window_end_utc"],
    )
    snapshots = {}
    for name, command in zip(
        cleanup_capture.SNAPSHOT_NAMES,
        commands,
        strict=True,
    ):
        path = snapshot_root / f"{name}.json"
        os.chmod(path, 0o600)
        raw = path.read_bytes()
        snapshots[name] = {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "argv": list(command),
            "captured_at_utc": "2025-01-01T01:00:30Z",
        }
    manifest = {
        "schema_name": cleanup_capture.CAPTURE_MANIFEST_SCHEMA_NAME,
        "schema_version": cleanup_capture.CAPTURE_MANIFEST_SCHEMA_VERSION,
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
        "capture_id": capture_id,
        "modal_profile": cleanup_capture.MODAL_PROFILE,
        "modal_environment": cleanup_capture.MODAL_ENVIRONMENT,
        "modal_cli_version": modal_readiness.MODAL_VERSION,
        "billing_window_start_utc": roster["billing_window_start_utc"],
        "billing_window_end_utc": roster["billing_window_end_utc"],
        "started_at_utc": "2025-01-01T01:00:01Z",
        "finished_at_utc": roster["snapshot_captured_at_utc"],
        "command_timeout_seconds": cleanup_capture.COMMAND_TIMEOUT_SECONDS,
        "outer_timeout_seconds": cleanup_capture.OUTER_TIMEOUT_SECONDS,
        "command_retry_count": 0,
        "snapshots": snapshots,
    }
    manifest_path = snapshot_root / cleanup_capture.CAPTURE_MANIFEST_FILENAME
    _write_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o600)
    roster["snapshot_capture_manifest_path"] = manifest_path.relative_to(
        project_root
    ).as_posix()
    roster["snapshot_capture_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    _write_json(roster_path, roster)


def _refresh_migration_lineage(roster_path: Path) -> None:
    """Re-seal one synthetic cohort after a test fixture finishes mutating it."""

    project_root = roster_path.parents[7]
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    attribution_by_attempt = {
        item["attempt_id"]: item for item in roster["billing_attributions"]
    }
    volume_run_ids: set[str] = set()
    manifest_logical = roster.get("snapshot_capture_manifest_path")
    if isinstance(manifest_logical, str) and manifest_logical != "pending":
        capture_manifest = json.loads(
            (project_root / manifest_logical).read_text(encoding="utf-8")
        )
        run_directory_path = project_root / capture_manifest["snapshots"][
            "run_directory_list"
        ]["path"]
        for row in json.loads(run_directory_path.read_text(encoding="utf-8")):
            parts = PurePosixPath(
                row["filename"].removeprefix("/").removesuffix("/")
            ).parts
            if len(parts) == 2 and parts[0] == "runs":
                volume_run_ids.add(parts[1])
    terminal_dispositions = []
    for terminal_logical in roster["action_attempt_receipts"]:
        terminal = json.loads(
            (project_root / terminal_logical).read_text(encoding="utf-8")
        )
        attribution = attribution_by_attempt[terminal["attempt_id"]]
        for run_id in terminal["concrete_remote_run_ids"]:
            execution_evidence = any(
                modal_readiness._path_has_any_entry(project_root, logical)
                for logical in modal_readiness._prior_execution_evidence_candidates(
                    identity,
                    terminal,
                    run_id,
                )
            )
            if not terminal["modal_cli_process_started"]:
                execution_disposition = "definitely_not_started"
            elif execution_evidence:
                execution_disposition = "remote_execution_bound"
            else:
                execution_disposition = (
                    "may_have_started_unresolved_quarantined"
                )
            provider_action = terminal["action"] in {"canary", "canaries"}
            provider_evidence = any(
                modal_readiness._path_has_any_entry(project_root, logical)
                for logical in modal_readiness._prior_provider_evidence_candidates(
                    run_id
                )
            )
            provider_disposition = (
                "not_applicable"
                if not provider_action
                else "definitely_not_started"
                if not terminal["modal_cli_process_started"]
                else "evidence_bound"
                if execution_disposition == "remote_execution_bound"
                and provider_evidence
                else "start_unresolved_conservative"
            )
            app_ids = attribution["object_ids"]
            volume_present = run_id in volume_run_ids
            if not terminal["modal_cli_process_started"]:
                snapshot_disposition = "no_remote_resources_observed"
                volume_disposition = "absent"
            elif terminal["status"] == "succeeded":
                snapshot_disposition = "app_volume_and_billing_bound"
                volume_disposition = "present_bound"
            elif app_ids or volume_present:
                snapshot_disposition = "stopped_resources_bound"
                volume_disposition = (
                    "present_bound" if volume_present else "absent_after_failure"
                )
            else:
                snapshot_disposition = "no_remote_resources_observed"
                volume_disposition = "absent"
            terminal_dispositions.append(
                {
                    "attempt_id": terminal["attempt_id"],
                    "run_id": run_id,
                    "execution_disposition": execution_disposition,
                    "provider_disposition": provider_disposition,
                    "snapshot_disposition": snapshot_disposition,
                    "snapshot_app_ids": app_ids,
                    "volume_disposition": volume_disposition,
                }
            )
    roster["terminal_run_dispositions"] = sorted(
        terminal_dispositions,
        key=lambda item: (item["attempt_id"], item["run_id"]),
    )
    _write_json(roster_path, roster)
    lineage_path = project_root / modal_readiness.modal_migration_lineage_path(
        identity
    )
    lineage_path.unlink(missing_ok=True)
    modal_readiness.create_modal_migration_lineage(
        final_identity=identity,
        accepted_primary_runs=roster["accepted_primary_runs"],
        accepted_attempt_ids=roster["accepted_attempt_ids"],
        root=project_root,
    )
    roster["migration_lineage_path"] = lineage_path.relative_to(
        project_root
    ).as_posix()
    roster["migration_lineage_sha256"] = hashlib.sha256(
        lineage_path.read_bytes()
    ).hexdigest()
    _write_json(roster_path, roster)


def _remove_global_terminal_seals(
    project_root: Path,
    identity: ModalLiveCohortIdentity,
) -> tuple[Path, Path]:
    """Return a synthetic fixture to the pre-terminal-seal state."""

    lineage_path = project_root / modal_readiness.modal_migration_lineage_path(
        identity
    )
    rejection_path = (
        project_root
        / modal_readiness.modal_global_launch_rejection_seal_path()
    )
    lineage_path.unlink(missing_ok=True)
    rejection_path.unlink(missing_ok=True)
    return lineage_path, rejection_path


def _write_modal_price_basis(
    root: Path,
    image_source_sha256: str,
    *,
    retrieved_at_utc: str = "2025-01-01T00:00:00Z",
) -> tuple[str, str, dict[str, object]]:
    payload: dict[str, object] = {
        "schema_name": "ModalPriceBasis",
        "schema_version": "1.0",
        "image_source_sha256": image_source_sha256,
        "official_source_url": modal_readiness.MODAL_PRICE_BASIS_OFFICIAL_SOURCE_URL,
        "retrieved_at_utc": retrieved_at_utc,
        "region": None,
        "cpu_usd_per_core_second": "0.0000131",
        "memory_usd_per_gib_second": "0.00000222",
        "t4_usd_per_gpu_second": "0.000164",
        "volume_storage_usd_per_gib_month": "0.09",
        "included_volume_storage_gib_per_month": "1024",
        "download_transfer_pricing": (
            modal_readiness.MODAL_DOWNLOAD_TRANSFER_PRICING
        ),
    }
    logical = modal_readiness.modal_price_basis_logical_path(
        image_source_sha256,
        retrieved_at_utc,
    ).as_posix()
    path = root / logical
    _write_json(path, payload)
    return logical, hashlib.sha256(path.read_bytes()).hexdigest(), payload


def _provider_start_uncertain_payload(
    *,
    harness: str = "greedy_autoresearch",
    run_id: str = "provider-start-uncertain",
    modal_call_id: str = "fc-provider-start-uncertain",
    ledger_state: str = "present",
) -> dict[str, object]:
    return {
        "schema_name": "ProviderRequestStartUncertainEvidence",
        "schema_version": "1.0",
        "harness": harness,
        "action": "one_opportunity_engineering_canary",
        "execution_backend": "modal",
        "action_run_id": run_id,
        "modal_call_id": modal_call_id,
        "api_endpoint": modal_readiness.OFFICIAL_OPENAI_API_BASE,
        "model": modal_readiness.TARGET_MODEL,
        "provider_attempt_count_lower_bound": 0,
        "provider_attempt_count_upper_bound": 1,
        "provider_request_started": "unknown",
        "provider_attempt_ledger_state": ledger_state,
        "billing_treatment": "reserve_one_full_approved_request",
        "reason": "controller_terminated_without_terminal_attempt_record",
    }


def _local_freeze_predecessors(root: Path) -> list[dict[str, str]]:
    source_digest = "9" * 64
    directory = root / LOCAL_ENGINEERING_FREEZE_ROOT / source_digest
    records = []
    for gate, filename in (
        ("local_unit_tested", "unit_test_evidence_receipt.json"),
        ("local_offline_smoke_tested", "offline_smoke_evidence_receipt.json"),
        (
            "local_engineering_freeze_validated",
            "local_engineering_freeze_receipt.json",
        ),
    ):
        path = directory / filename
        if not path.exists():
            _write_json(path, {"fixture": gate})
        records.append(
            {
                "gate": gate,
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records


@pytest.fixture(autouse=True)
def _rederived_local_freeze(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        modal_readiness,
        "source_tree_sha256",
        lambda _root: _FIXTURE_SOURCE_TREE_SHA256,
    )
    identity = _fixture_identity(tmp_path)
    for gate, contract in MODAL_READINESS_RECEIPT_CONTRACTS.items():
        monkeypatch.setitem(
            contract,
            "receipt_path",
            modal_readiness.modal_component_receipt_path(identity, gate).as_posix(),
        )
    monkeypatch.setattr(
        modal_readiness,
        "MODAL_OFFLINE_SMOKE_VALIDATION_RECEIPT_PATH",
        modal_readiness.modal_component_receipt_path(
            identity, "modal_offline_smoke_validated"
        ).as_posix(),
        raising=False,
    )
    monkeypatch.setattr(
        modal_readiness,
        "historical_local_engineering_freeze_predecessor_bindings",
        lambda bindings, *, root, expected_image_source_sha256=None: tuple(
            bindings
        ),
    )


def _network_denial_probe(context: ExecutionContextV1) -> dict[str, object]:
    return {
        "schema_name": "ProviderFreeNetworkDenialProbe",
        "schema_version": "1.0",
        "attempted_endpoint": {"ip": "1.1.1.1", "port": 443},
        "timeout_seconds": 1.0,
        "denied": True,
        "exception_type": "PermissionError",
        "execution_context": context.to_dict(),
    }


@pytest.mark.parametrize(
    ("observed", "expected"),
    (
        (True, 1),
        (False, 0),
        (1, 1.0),
        ({"nested": [1, True]}, {"nested": [1.0, True]}),
        (["value"], ("value",)),
    ),
)
def test_exact_json_equal_rejects_type_aliases(
    observed: object,
    expected: object,
) -> None:
    assert modal_readiness.exact_json_equal(observed, expected) is False
    assert modal_readiness.exact_json_equal(expected, observed) is False


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "provider_attempt_count_lower_bound",
            True,
            "must be an integer",
        ),
        (
            "provider_attempt_count_upper_bound",
            1.0,
            "must be an integer",
        ),
        ("provider_attempt_ledger_state", "absent", "evidence is invalid"),
        ("action_run_id", "another-run", "evidence is invalid"),
        ("modal_call_id", "fc-another-call", "evidence is invalid"),
    ),
)
def test_provider_start_uncertain_evidence_rejects_type_and_binding_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    run_id = "provider-start-uncertain"
    logical = (
        f"outputs/development/modal_downloads/{run_id}/controller/"
        "provider_request_start_uncertain.json"
    )
    payload = _provider_start_uncertain_payload(run_id=run_id)
    payload[field] = replacement
    path = tmp_path / logical
    _write_json(path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match=message):
        modal_readiness._load_provider_start_uncertain_evidence(
            tmp_path,
            logical,
            digest,
            harness="greedy_autoresearch",
            run_id=run_id,
            expected_modal_call_id="fc-provider-start-uncertain",
        )


def test_provider_start_uncertain_evidence_rejects_schema_and_digest_drift(
    tmp_path: Path,
) -> None:
    run_id = "provider-start-uncertain"
    logical = (
        f"outputs/development/modal_downloads/{run_id}/controller/"
        "provider_request_start_uncertain.json"
    )
    path = tmp_path / logical
    payload = _provider_start_uncertain_payload(run_id=run_id)
    payload["extra"] = "forbidden"
    _write_json(path, payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="invalid exact schema"):
        modal_readiness._load_provider_start_uncertain_evidence(
            tmp_path,
            logical,
            digest,
            harness="greedy_autoresearch",
            run_id=run_id,
        )

    payload.pop("extra")
    _write_json(path, payload)
    with pytest.raises(ValueError, match="digest changed"):
        modal_readiness._load_provider_start_uncertain_evidence(
            tmp_path,
            logical,
            "0" * 64,
            harness="greedy_autoresearch",
            run_id=run_id,
        )


def test_action_predecessors_rederive_exact_local_freeze_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bindings = _local_freeze_predecessors(tmp_path)
    changed = tuple(dict(item) for item in bindings)
    changed[0]["sha256"] = "4" * 64
    monkeypatch.setattr(
        modal_readiness,
        "historical_local_engineering_freeze_predecessor_bindings",
        lambda *_args, **_kwargs: changed,
    )

    with pytest.raises(ValueError, match="freeze bindings changed"):
        modal_readiness._expected_predecessor_gates(
            "cuda-environment",
            bindings,
            root=tmp_path,
            image_source_sha256="b" * 64,
            identity=_fixture_identity(tmp_path),
        )


def test_modal_price_basis_is_source_bound_fresh_and_create_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_sha256 = "b" * 64
    retrieved = datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )
    monkeypatch.setattr(
        modal_readiness,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    arguments = {
        "expected_image_source_sha256": image_sha256,
        "official_source_url": modal_readiness.MODAL_PRICE_BASIS_OFFICIAL_SOURCE_URL,
        "retrieved_at_utc": retrieved,
        "cpu_usd_per_core_second": "0.0000131",
        "memory_usd_per_gib_second": "0.00000222",
        "t4_usd_per_gpu_second": "0.000164",
        "volume_storage_usd_per_gib_month": "0.09",
        "included_volume_storage_gib_per_month": "1024",
        "download_transfer_pricing": (
            modal_readiness.MODAL_DOWNLOAD_TRANSFER_PRICING
        ),
        "root": tmp_path,
    }

    payload = modal_readiness.create_modal_price_basis(**arguments)
    logical = modal_readiness.modal_price_basis_logical_path(
        image_sha256,
        retrieved,
    )

    assert payload["image_source_sha256"] == image_sha256
    assert (tmp_path / logical).is_file()
    with pytest.raises(FileExistsError):
        modal_readiness.create_modal_price_basis(**arguments)


def test_modal_price_basis_rejects_stale_or_incomplete_snapshot(
    tmp_path: Path,
) -> None:
    image_sha256 = "b" * 64
    payload = _write_modal_price_basis(
        tmp_path,
        image_sha256,
        retrieved_at_utc="2025-01-01T00:00:00Z",
    )[2]
    with pytest.raises(ValueError, match="older than 48 hours"):
        modal_readiness.validate_modal_price_basis_payload(
            payload,
            expected_image_source_sha256=image_sha256,
            now_utc=datetime(2025, 1, 4, tzinfo=UTC),
            require_freshness=True,
        )

    incomplete = dict(payload)
    incomplete.pop("download_transfer_pricing")
    with pytest.raises(ValueError, match="invalid exact schema"):
        modal_readiness.validate_modal_price_basis_payload(
            incomplete,
            expected_image_source_sha256=image_sha256,
            require_freshness=False,
        )


def test_receipt_reader_rejects_leaf_and_parent_symlinks(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    target = real_directory / "receipt.json"
    _write_json(target, {"schema_name": "Fixture"})

    leaf_alias = tmp_path / "leaf-alias.json"
    leaf_alias.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        modal_readiness._load_object(leaf_alias)

    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="traverse symlinks"):
        modal_readiness._load_object(parent_alias / target.name)


def test_receipt_reader_is_bounded_and_rejects_concurrent_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.touch()
    with oversized.open("r+b") as handle:
        handle.truncate(modal_readiness._MAX_JSON_OBJECT_BYTES + 1)
    with pytest.raises(ValueError, match="exceeds its size limit"):
        modal_readiness._load_object(oversized)

    changing = tmp_path / "changing.json"
    changing.write_bytes(b'{"value":"' + b"a" * (128 * 1024) + b'"}')
    real_read = modal_readiness.os.read
    changed = False

    def read_then_change(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, size)
        if chunk and not changed:
            changed = True
            with changing.open("ab") as handle:
                handle.write(b" ")
        return chunk

    monkeypatch.setattr(modal_readiness.os, "read", read_then_change)
    with pytest.raises(ValueError, match="changed while it was read"):
        modal_readiness._load_object(changing)

    replacing = tmp_path / "replacing.json"
    replacing.write_bytes(b'{"value":"' + b"a" * (128 * 1024) + b'"}')
    replacement = tmp_path / "replacement.json"
    _write_json(replacement, {"value": "replacement"})
    replaced = False

    def read_then_replace(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            replacement.replace(replacing)
        return chunk

    monkeypatch.setattr(modal_readiness.os, "read", read_then_replace)
    with pytest.raises(ValueError, match="changed while it was read"):
        modal_readiness._load_object(replacing)


def test_modal_and_provider_price_basis_bind_the_loaded_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_sha256 = "b" * 64
    logical, raw_sha256, expected = _write_modal_price_basis(
        tmp_path,
        image_sha256,
    )
    monkeypatch.setattr(
        modal_readiness,
        "_sha256_file",
        lambda _path: pytest.fail("price-basis load must not re-open for its digest"),
    )
    loaded, _rates, _path = modal_readiness.load_modal_price_basis(
        tmp_path,
        logical,
        expected_raw_sha256=raw_sha256,
        expected_image_source_sha256=image_sha256,
        require_freshness=False,
    )
    assert loaded == expected

    provider_logical = "outputs/readiness/provider-price-basis.json"
    provider_path = tmp_path / provider_logical
    _write_json(
        provider_path,
        {
            "schema_name": "ProviderPriceBasis",
            "schema_version": "1.0",
            "model": modal_readiness.TARGET_MODEL,
            "official_source_url": "https://openai.com/api/pricing/",
            "retrieved_at_utc": "2025-01-01T00:00:00Z",
            "uncached_input_usd_per_million_tokens": "1",
            "output_usd_per_million_tokens": "1",
            "per_request_fee_usd": "0",
        },
    )
    _provider, _provider_path, provider_sha256 = (
        modal_readiness._load_price_basis(tmp_path, provider_logical)
    )
    assert provider_sha256 == hashlib.sha256(provider_path.read_bytes()).hexdigest()


_BOUND_MODAL_APP_SOURCE = """\
from modal_boundary import FUNCTION_SPECS

def invoke_synchronously():
    return None

def _function_options(name):
    spec = FUNCTION_SPECS[name]
    return {"block_network": not spec.provider_secret}

@app.function(**_function_options("artifact_verify"))
def artifact_verify(source_run_id, verifier_run_id):
    return source_run_id, verifier_run_id
"""
_BOUND_MODAL_BOUNDARY_SOURCE = """\
class FunctionSpec:
    pass

FUNCTION_SPECS = {
    "artifact_verify": FunctionSpec("artifact_verify", None, False),
}
"""


def _bound_policy_source_files(root: Path) -> tuple[SourceFileV1, ...]:
    sources = {
        "modal_app.py": _BOUND_MODAL_APP_SOURCE,
        "modal_boundary.py": _BOUND_MODAL_BOUNDARY_SOURCE,
    }
    entries: list[SourceFileV1] = []
    for logical, content in sources.items():
        path = root / logical
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        entries.append(
            SourceFileV1(
                relative_path=logical,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(entries)


def _execution_stub(
    root: Path,
    *,
    run_id: str,
    function_name: str,
    index: int,
    image_source_sha256: str,
) -> tuple[dict[str, object], Path, SimpleNamespace, ExecutionContextV1]:
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=APP_NAME,
        function_name=function_name,
        modal_app_id=f"ap-{index}",
        modal_function_id=f"fu-{index}",
        modal_call_id=f"fc-{index}",
        modal_image_id=f"im-{index}",
        image_source_sha256=image_source_sha256,
        artifact_uri=volume_artifact_uri(run_id),
    )
    manifest = SimpleNamespace(
        image_source_sha256=image_source_sha256,
        manifest_sha256=f"{index:x}" * 64,
        files=(),
    )
    claim = {
        "run_id": run_id,
        "function_name": function_name,
        "modal_app_id": context.modal_app_id,
        "modal_function_id": context.modal_function_id,
        "modal_call_id": context.modal_call_id,
        "modal_image_id": context.modal_image_id,
        "image_source_sha256": image_source_sha256,
        "artifact_manifest_sha256": manifest.manifest_sha256,
    }
    return claim, root / run_id, manifest, context


def _image_binding_inputs(
    root: Path,
    manifest: ImageSourceManifestV1,
    *,
    image_ids: tuple[str | None, ...] | None = None,
) -> dict[str, tuple[Path, SimpleNamespace, ExecutionContextV1]]:
    selected_ids = image_ids or ("im-shared",) * 8
    assert len(selected_ids) == 8
    executions: dict[str, tuple[Path, SimpleNamespace, ExecutionContextV1]] = {}
    for index, image_id in enumerate(selected_ids):
        run_id = f"image-binding-run-{index}"
        run_root = root / run_id
        _write_json(run_root / "image_source_manifest.json", manifest.to_dict())
        context = ExecutionContextV1(
            execution_backend="modal",
            run_id=run_id,
            app_name=APP_NAME,
            function_name=f"binding_function_{index}",
            modal_app_id=f"ap-binding-{index}",
            modal_function_id=f"fu-binding-{index}",
            modal_call_id=f"fc-binding-{index}",
            modal_image_id=image_id,
            image_source_sha256=manifest.manifest_sha256,
            artifact_uri=volume_artifact_uri(run_id),
        )
        executions[f"execution_{index}"] = (
            run_root,
            SimpleNamespace(image_source_sha256=manifest.manifest_sha256),
            context,
        )
    return executions


def _downloaded_cuda_run(root: Path, run_id: str = "cuda-check-1") -> Path:
    run = root / "outputs/development/modal_downloads" / run_id
    run.mkdir(parents=True)
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=_bound_policy_source_files(root),
    )
    image_hash = image_manifest.manifest_sha256
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=APP_NAME,
        function_name="cuda_environment",
        modal_app_id="ap-test123",
        modal_function_id="fu-test123",
        modal_call_id="fc-test123",
        modal_image_id="im-test123",
        image_source_sha256=image_hash,
        artifact_uri=volume_artifact_uri(run_id),
    )
    fingerprint = {
        "requested_device": "cuda",
        "selected_device": "cuda:0",
        "accelerator_kind": "cuda",
        "gpu_name": "NVIDIA T4",
        "gpu_count": 1,
        "compute_capability": "7.5",
        "cuda_runtime": "12.8",
        "cuda_driver": "570.00",
        "torch_version": "2.7.1",
        "host_platform": "Linux-test",
    }
    _write_json(run / "execution_context.json", context.to_dict())
    _write_json(run / "image_source_manifest.json", image_manifest.to_dict())
    _write_json(
        run / "provider_free_network_denial_probe.json",
        _network_denial_probe(context),
    )
    _write_json(
        run / "cuda_environment.json",
        {
            "python": "3.12.13",
            "platform": "Linux-test",
            "torch": "2.7.1",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "NVIDIA T4",
            "cuda_compute_capability": [7, 5],
            "cuda_runtime": "12.8",
            "cuda_driver": "570.00",
            "cuda_total_memory_bytes": 16_000_000_000,
            "git_version": "git version 2.30.2",
            "accelerator_fingerprint": fingerprint,
            "execution_context": context.to_dict(),
        },
    )
    _write_json(
        run / "remote_action_result.json",
        {
            "success": True,
            "mode": "cuda_environment",
            "observed_gpu": "NVIDIA T4",
        },
    )
    manifest = build_artifact_manifest(
        run, run_id=run_id, image_source_sha256=image_hash
    )
    write_artifact_manifest(run, manifest)
    return run


def _downloaded_candidate_run(
    root: Path, run_id: str = "candidate-check-1"
) -> Path:
    run = root / "outputs/development/modal_downloads" / run_id
    run.mkdir(parents=True)
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=_bound_policy_source_files(root),
    )
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=APP_NAME,
        function_name="candidate_smoke",
        modal_app_id="ap-candidate123",
        modal_function_id="fu-candidate123",
        modal_call_id="fc-candidate123",
        modal_image_id="im-candidate123",
        image_source_sha256=image_manifest.manifest_sha256,
        artifact_uri=volume_artifact_uri(run_id),
    )
    empty_digest = hashlib.sha256(b"").hexdigest()
    _write_json(run / "execution_context.json", context.to_dict())
    _write_json(run / "image_source_manifest.json", image_manifest.to_dict())
    _write_json(
        run / "provider_free_network_denial_probe.json",
        _network_denial_probe(context),
    )
    (run / "candidate_smoke").mkdir()
    _write_json(
        run / "remote_action_result.json",
        {
            "success": True,
            "mode": "cuda_candidate_train_and_layer_a",
            "returncode": 0,
            "stdout_sha256": empty_digest,
            "stdout_size_bytes": 0,
            "stderr_sha256": empty_digest,
            "stderr_size_bytes": 0,
        },
    )
    manifest = build_artifact_manifest(
        run,
        run_id=run_id,
        image_source_sha256=image_manifest.manifest_sha256,
    )
    write_artifact_manifest(
        run,
        manifest,
        filename="artifact_manifest.checkpoint.json",
    )
    return run


def _verifier_run_id(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:12]
    return f"verify-{digest}"


def _remote_verification_payload(root: Path, run_id: str) -> dict[str, object]:
    run = root / "outputs/development/modal_downloads" / run_id
    manifest_path = run / "artifact_manifest.json"
    if not manifest_path.is_file():
        manifest_path = run / "artifact_manifest.checkpoint.json"
    raw_manifest = load_raw_artifact_manifest(manifest_path)
    verifier_run_id = _verifier_run_id(run_id)
    source_context = ExecutionContextV1.from_dict(
        json.loads((run / "execution_context.json").read_text(encoding="utf-8"))
    )
    verifier_suffix = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:10]
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=verifier_run_id,
        app_name=APP_NAME,
        function_name="artifact_verify",
        modal_app_id=f"ap-verifier-{verifier_suffix}",
        modal_function_id=f"fu-verifier-{verifier_suffix}",
        modal_call_id=f"fc-verifier-{verifier_suffix}",
        modal_image_id=source_context.modal_image_id,
        image_source_sha256=raw_manifest.manifest.image_source_sha256,
        artifact_uri=volume_artifact_uri(run_id),
    )
    return ArtifactVerificationV1(
        source_run_id=run_id,
        verifier_run_id=verifier_run_id,
        manifest_filename=raw_manifest.filename,
        raw_manifest_sha256=raw_manifest.raw_sha256,
        raw_manifest_size_bytes=raw_manifest.raw_size_bytes,
        canonical_manifest_sha256=raw_manifest.manifest.manifest_sha256,
        file_count=len(raw_manifest.manifest.files),
        verifier_execution_context=context,
    ).to_dict()


def _write_remote_verification(
    root: Path,
    run_id: str,
    *,
    attempt_id: str | None = None,
    identity: ModalLiveCohortIdentity | None = None,
) -> Path:
    verifier_run_id = _verifier_run_id(run_id)
    path = root / modal_readiness.modal_remote_verification_receipt_path(
        identity or _fixture_identity(root),
        run_id,
        verifier_run_id,
        attempt_id or _verifier_attempt_id(run_id),
    )
    _write_json(path, _remote_verification_payload(root, run_id))
    return path


def _rewrite_artifact_manifest(run: Path, run_id: str) -> None:
    manifest_path = run / "artifact_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.unlink()
    manifest = build_artifact_manifest(
        run,
        run_id=run_id,
        image_source_sha256=old_manifest["image_source_sha256"],
    )
    write_artifact_manifest(run, manifest)


def _refresh_source_manifest_and_verifier(
    roster_path: Path,
    run_id: str,
) -> None:
    """Rebind a fixture verifier after deliberately mutating a source run."""

    root = roster_path.parents[7]
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    run_root = root / f"outputs/development/modal_downloads/{run_id}"
    _rewrite_artifact_manifest(run_root, run_id)
    verifier = next(
        record
        for record in roster["artifact_verifiers"].values()
        if record["source_run_id"] == run_id
    )
    verification_path = _write_remote_verification(
        root,
        run_id,
        attempt_id=verifier["attempt_id"],
        identity=_fixture_identity(root),
    )
    verifier["remote_verification_sha256"] = hashlib.sha256(
        verification_path.read_bytes()
    ).hexdigest()
    selector_path = root / roster["provider_canary_selector_path"]
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    for entry in selector["runs"].values():
        if entry["run_id"] == run_id:
            entry["raw_artifact_manifest_sha256"] = hashlib.sha256(
                (run_root / "artifact_manifest.json").read_bytes()
            ).hexdigest()
            break
    else:
        raise AssertionError("fixture source run is absent from its selector")
    _write_json(selector_path, selector)
    roster["provider_canary_selector_sha256"] = hashlib.sha256(
        selector_path.read_bytes()
    ).hexdigest()
    _write_json(roster_path, roster)


def _ledger(gate_name: str, *, passed: bool, receipt_path: str | None) -> dict:
    contract = MODAL_READINESS_RECEIPT_CONTRACTS[gate_name]
    selected_identity = None
    if passed:
        parts = Path(contract["receipt_path"]).parts
        selected_identity = {
            "source_tree_sha256": parts[-5],
            "image_source_sha256": parts[-4],
            "cohort_id": parts[-3],
        }
    return {
        "schema_name": "scientific_readiness_evidence",
        "schema_version": "4",
        "gates": {
            gate_name: {
                "passed": passed,
                "evidence": "local receipt fixture",
                "receipt_path": receipt_path,
                "receipt_sha256": "0" * 64 if passed else None,
                "selected_cohort_identity": selected_identity,
                "receipt_contract": dict(contract["receipt_contract"]),
            }
        },
    }


def _cleanup_snapshots(
    root: Path,
    run_id: str,
    *,
    app_state: str = "stopped",
    tasks: str = "0",
) -> Path:
    snapshot_root = root / "outputs/readiness/modal_resource_cleanup" / run_id
    _write_json(
        snapshot_root / "app_list.json",
        [
            {
                "app_id": "ap-test123",
                "description": APP_NAME,
                "state": app_state,
                "tasks": tasks,
                "created_at": "2026-08-09 00:00:00+00:00",
                "stopped_at": (
                    "2026-08-09 00:02:00+00:00"
                    if app_state == "stopped"
                    else None
                ),
            }
        ],
    )
    _write_json(snapshot_root / "container_list.json", [])
    _write_json(snapshot_root / "endpoint_list.json", [])
    _write_json(
        snapshot_root / "volume_list.json",
        [
            {
                "name": "rl4rl-architecture-artifacts",
                "created_at": "2026-08-09 00:00:00+00:00",
                "created_by": "test-user",
            }
        ],
    )
    _write_json(
        snapshot_root / "run_directory_list.json",
        [
            {
                "filename": f"/runs/{run_id}",
                "type": "dir",
                "created_modified": "2026-08-09 00:30 UTC",
                "size": "0 B",
            },
            {
                "filename": "/runs/modal-cuda-env-20260809-02",
                "type": "dir",
                "created_modified": "2026-08-09 00:30 UTC",
                "size": "0 B",
            },
        ],
    )
    _write_json(
        snapshot_root / "billing_report.json",
        [
            {
                "object_id": "ap-test123",
                "description": APP_NAME,
                "environment": "main",
                "interval_start": "2026-08-09T00:00:00+00:00",
                "resource": "T4 GPU",
                "cost": "0.25",
            }
        ],
    )
    (root / "modal_app.py").write_text(
        "def invoke_synchronously():\n    return None\n",
        encoding="utf-8",
    )
    (root / "experiment_manifest.yaml").write_text(
        "remote_execution:\n  detached_calls: false\n",
        encoding="utf-8",
    )
    return snapshot_root


def _aggregate_cleanup_fixture(
    root: Path,
    cleanup_run_id: str = "cleanup-check-1",
) -> tuple[Path, Path]:
    environment_root = _downloaded_cuda_run(root, cleanup_run_id)
    raw_image_manifest = json.loads(
        (environment_root / "image_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256=raw_image_manifest["dependency_lock_sha256"],
        files=tuple(
            SourceFileV1(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
            )
            for item in raw_image_manifest["files"]
        ),
    )
    image_sha256 = image_manifest.manifest_sha256
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=_FIXTURE_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id=cleanup_run_id,
    )
    (
        modal_price_basis_path,
        modal_price_basis_sha256,
        modal_price_basis,
    ) = _write_modal_price_basis(root, image_sha256)
    image_id = "im-test123"
    primary_runs = {
        "cuda_environment": cleanup_run_id,
        "offline_smoke": "accepted-offline",
        "candidate_smoke": "accepted-candidate",
        "resume_attempt": "accepted-resume",
        "canary_greedy_autoresearch": "accepted-canary-greedy-ar",
        "canary_semantic_autoresearch": "accepted-canary-semantic-ar",
        "canary_openevolve_generic": "accepted-canary-openevolve-generic",
        "canary_openevolve_semantic": "accepted-canary-openevolve-semantic",
    }
    functions = {
        "cuda_environment": "cuda_environment",
        "offline_smoke": "offline_smoke",
        "candidate_smoke": "candidate_smoke",
        "resume_attempt": "checkpoint_resume",
        "canary_greedy_autoresearch": "canary_greedy_autoresearch",
        "canary_semantic_autoresearch": "canary_semantic_autoresearch",
        "canary_openevolve_generic": "canary_openevolve_generic",
        "canary_openevolve_semantic": "canary_openevolve_semantic",
    }
    primary_contexts: dict[str, ExecutionContextV1] = {
        "cuda_environment": ExecutionContextV1.from_dict(
            json.loads(
                (environment_root / "execution_context.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    }
    canary_labels = {
        f"canary_{harness}": harness for harness in modal_readiness.CANARY_ORDER
    }
    provider_plan_by_harness = {
        item["harness"]: item
        for item in _provider_plan_for_fixture(
            identity,
            preflight_path=modal_readiness.modal_candidate_resume_preflight_receipt_path(
                identity,
                "f" * 64,
            ).as_posix(),
            preflight_sha256="b" * 64,
        )["harnesses"]
    }
    for index, label in enumerate(tuple(primary_runs)[1:], start=1):
        run_id = primary_runs[label]
        run = root / "outputs/development/modal_downloads" / run_id
        run.mkdir(parents=True)
        context = ExecutionContextV1(
            execution_backend="modal",
            run_id=run_id,
            app_name=APP_NAME,
            function_name=functions[label],
            modal_app_id=f"ap-primary-{index}",
            modal_function_id=f"fu-primary-{index}",
            modal_call_id=f"fc-primary-{index}",
            modal_image_id=image_id,
            image_source_sha256=image_sha256,
            artifact_uri=volume_artifact_uri(run_id),
        )
        _write_json(run / "execution_context.json", context.to_dict())
        _write_json(run / "image_source_manifest.json", image_manifest.to_dict())
        if label in {"offline_smoke", "candidate_smoke", "resume_attempt"}:
            _write_json(
                run / "provider_free_network_denial_probe.json",
                _network_denial_probe(context),
            )
        if label == "offline_smoke":
            (run / "offline_study").mkdir()
            _write_json(run / "remote_action_result.json", {"fixture": True})
        elif label == "candidate_smoke":
            (run / "candidate_smoke").mkdir()
            _write_json(run / "remote_action_result.json", {"fixture": True})
        elif label == "resume_attempt":
            (run / "candidate_smoke").mkdir()
            for filename in (
                "resume_action_result.json",
                "resume_contract_verification.json",
                "resume_execution_context.json",
                "resume_progression_verification.json",
                "resume_source_binding.json",
            ):
                _write_json(run / filename, {"fixture": True})
        if label in canary_labels:
            harness = canary_labels[label]
            provider_record = {
                "schema_name": "ProviderAttemptRecord",
                "schema_version": "1.0",
                "harness": harness,
                "action": "one_opportunity_engineering_canary",
                "controller_run_id": f"controller-{run_id}",
                "execution_backend": "modal",
                "action_run_id": run_id,
                "modal_call_id": context.modal_call_id,
                "attempt_ordinal": 1,
                "started_at_utc": "2025-01-01T00:05:10Z",
                "ended_at_utc": "2025-01-01T00:05:20Z",
                "status": "success",
                "api_endpoint": modal_readiness.OFFICIAL_OPENAI_API_BASE,
                "model": modal_readiness.TARGET_MODEL,
                "generation_settings_sha256": provider_plan_by_harness[harness][
                    "generation_settings_sha256"
                ],
                "provider_response_id": f"response-{harness}",
                "provider_request_id": f"request-{harness}",
                "usage_known": True,
                "input_tokens": 100 + index,
                "output_tokens": 20 + index,
                "total_tokens": 120 + 2 * index,
                "error_class": None,
            }
            ledger = run / "controller/provider_attempts.jsonl"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(
                json.dumps(provider_record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifest = build_artifact_manifest(
            run, run_id=run_id, image_source_sha256=image_sha256
        )
        write_artifact_manifest(
            run,
            manifest,
            filename=(
                "artifact_manifest.checkpoint.json"
                if label == "candidate_smoke"
                else "artifact_manifest.json"
            ),
        )
        primary_contexts[label] = context

    verifier_records: dict[str, dict[str, object]] = {}
    verifier_contexts: dict[str, ExecutionContextV1] = {}
    for verifier_ordinal, (label, run_id) in enumerate(
        primary_runs.items(), start=len(primary_runs) + 1
    ):
        verifier_attempt_id = f"{verifier_ordinal:032x}"
        path = _write_remote_verification(
            root,
            run_id,
            attempt_id=verifier_attempt_id,
            identity=identity,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        context = ExecutionContextV1.from_dict(payload["verifier_execution_context"])
        verifier_contexts[label] = context
        remote_logical = modal_readiness._remote_verification_logical(
            identity,
            run_id,
            context.run_id,
            verifier_attempt_id,
        )
        verifier_records[label] = {
            "source_label": label,
            "source_run_id": run_id,
            "verifier_run_id": context.run_id,
            "attempt_id": verifier_attempt_id,
            "remote_verification_path": remote_logical,
            "remote_verification_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "verifier_execution_context": context.to_dict(),
            "expected_remote_receipt_roster": [
                "execution_context.json",
                "image_source_manifest.json",
                "artifact_verification_result.json",
                "artifact_manifest.json",
            ],
        }

    predecessor_paths = {
        "cuda": modal_readiness.modal_component_receipt_path(
            identity, "modal_cuda_environment_validated"
        ).as_posix(),
        "offline": modal_readiness.modal_component_receipt_path(
            identity, "modal_offline_smoke_validated"
        ).as_posix(),
        "round_trip": modal_readiness.modal_component_receipt_path(
            identity, "modal_artifact_round_trip_validated"
        ).as_posix(),
        "preflight": modal_readiness.modal_candidate_resume_preflight_receipt_path(
            identity, "f" * 64
        ).as_posix(),
    }
    predecessor_contracts = {
        "cuda": ("ModalCUDAEnvironmentValidationReceipt", "2.0"),
        "offline": ("ModalOfflineSmokeValidationReceipt", "2.0"),
        "round_trip": ("ModalArtifactRoundTripValidationReceipt", "3.0"),
        "preflight": ("CandidateResumePreflightReceipt", "2.0"),
    }
    for name, logical in predecessor_paths.items():
        schema_name, schema_version = predecessor_contracts[name]
        _write_json(
            root / logical,
            {
                "schema_name": schema_name,
                "schema_version": schema_version,
                **modal_readiness.modal_cohort_identity_dict(identity),
                "fixture": name,
            },
        )

    price_path = root / (
        modal_readiness.modal_cleanup_snapshot_directory(identity)
        / "provider_price_basis.json"
    )
    _write_json(
        price_path,
        {
            "schema_name": "ProviderPriceBasis",
            "schema_version": "1.0",
            "model": modal_readiness.TARGET_MODEL,
            "official_source_url": "https://openai.com/api/pricing/",
            "retrieved_at_utc": "2025-01-01T00:00:00Z",
            "uncached_input_usd_per_million_tokens": "1.25",
            "output_usd_per_million_tokens": "10",
            "per_request_fee_usd": "0",
        },
    )
    approval_plan_path = (
        root
        / "outputs/readiness/provider_canary_approval"
        / "provider-canary-fixture.json"
    )
    approval_plan = _provider_plan_for_fixture(
        identity,
        preflight_path=predecessor_paths["preflight"],
        preflight_sha256=hashlib.sha256(
            (root / predecessor_paths["preflight"]).read_bytes()
        ).hexdigest(),
        dependency_lock_sha256=image_manifest.dependency_lock_sha256,
    )
    _write_json(approval_plan_path, approval_plan)

    local_predecessors = _local_freeze_predecessors(root)

    attempt_directory_logical = modal_readiness.modal_action_attempt_directory(
        identity
    )
    attempt_directory = root / attempt_directory_logical
    attempt_directory.mkdir(parents=True)
    accepted_attempt_ids: dict[str, str] = {}
    classifications: list[dict[str, object]] = []
    attributions: list[dict[str, object]] = []
    receipt_paths: list[str] = []
    intent_paths: list[str] = []
    action_by_label = {
        "cuda_environment": "cuda-environment",
        "offline_smoke": "offline-smoke",
        "candidate_smoke": "candidate-smoke",
        "resume_attempt": "checkpoint-resume",
    }

    def write_attempt(
        ordinal: int,
        *,
        action: str,
        run_id: str,
        app_id: str,
        roles: list[str],
        source_run_id: str | None = None,
        verifier_run_id: str | None = None,
        harness: str | None = None,
        predecessor_receipts: list[dict[str, str]] | None = None,
    ) -> str:
        attempt_id = f"{ordinal:032x}"
        provider_action = action in {"canary", "canaries"}
        modal_resource_profile = modal_readiness._expected_modal_resource_profile(
            action,
            harness,
        )
        modal_cost_estimate = modal_readiness.derive_modal_action_cost_estimate(
            action=action,
            harness=harness,
            resource_profile=modal_resource_profile,
            price_basis=modal_price_basis,
        )
        concrete_remote_run_ids = (
            [verifier_run_id]
            if action in {"download", "verify"}
            else [run_id]
        )
        launch_capability_sha256 = hashlib.sha256(
            f"capability:{attempt_id}".encode()
        ).hexdigest()
        reservation_bindings = _remote_run_reservation_bindings(
            root,
            identity,
            attempt_id=attempt_id,
            action=action,
            run_ids=concrete_remote_run_ids,
            created_at_utc="2025-01-01T00:05:00Z",
            launch_capability_sha256=launch_capability_sha256,
        )
        receipt = {
            "schema_name": "ModalActionAttemptReceipt",
            "schema_version": "3.6",
            "attempt_id": attempt_id,
            "started_at_utc": "2025-01-01T00:05:00Z",
            "finished_at_utc": "2025-01-01T00:06:00Z",
            "status": "succeeded",
            "failure_kind": None,
            "action": action,
            "run_id": run_id,
            "concrete_remote_run_ids": concrete_remote_run_ids,
            "remote_run_reservations": reservation_bindings,
            **_fixture_local_containment(),
            "source_run_id": source_run_id,
            "verifier_run_id": verifier_run_id,
            "harness": harness,
            "source_tree_sha256": identity.source_tree_sha256,
            "cohort_id": identity.cohort_id,
            "approved_image_source_sha256": image_sha256,
            "modal_command_sha256": hashlib.sha256(
                attempt_id.encode("utf-8")
            ).hexdigest(),
            "launch_capability_sha256": launch_capability_sha256,
            "modal_profile": "scalingintelligence",
            "modal_environment": modal_readiness.MODAL_ENVIRONMENT,
            "outer_cli_timeout_seconds": modal_readiness._expected_attempt_timeout(
                action
            ),
            "modal_cost_cap_usd": "0.25",
            "modal_resource_profile": modal_resource_profile,
            "modal_price_basis_path": modal_price_basis_path,
            "modal_price_basis_sha256": modal_price_basis_sha256,
            "modal_cost_estimate": modal_cost_estimate,
            "modal_cost_approved": True,
            "provider_cost_approved": provider_action,
            "provider_cost_cap_usd": "2.00" if provider_action else None,
            "provider_approval_plan_path": (
                approval_plan_path.relative_to(root).as_posix()
                if provider_action
                else None
            ),
            "approval_plan_sha256": (
                approval_plan["approval_plan_sha256"] if provider_action else None
            ),
            "provider_price_basis_path": (
                price_path.relative_to(root).as_posix()
                if provider_action
                else None
            ),
            "provider_price_basis_sha256": (
                hashlib.sha256(price_path.read_bytes()).hexdigest()
                if provider_action
                else None
            ),
            "predecessor_receipts": [
                *local_predecessors,
                *(predecessor_receipts or []),
            ],
            "source_evidence_recovery": False,
            "local_process_start_receipt_path": None,
            "local_process_start_receipt_sha256": None,
            "local_process_id": None,
            "local_process_group_id": None,
            "local_session_id": None,
            "modal_cli_process_started": True,
            "remote_execution_state": "may_have_started",
            "returncode": 0,
            "process_group_closed": True,
        }
        receipt["modal_command_sha256"] = (
            modal_readiness._reconstructed_modal_command_sha256(
                receipt,
                root=root,
            )
        )
        intent_logical, terminal_logical = _write_attempt_intent_and_terminal(
            root,
            receipt,
        )
        intent_paths.append(
            intent_logical
        )
        receipt_paths.append(
            terminal_logical
        )
        classifications.append(
            {"attempt_id": attempt_id, "roles": sorted(roles)}
        )
        attributions.append(
            {
                "attempt_id": attempt_id,
                "disposition": "billed",
                "object_ids": [app_id],
            }
        )
        return attempt_id

    def predecessor(gate: str, logical: str) -> dict[str, str]:
        path = root / logical
        return {
            "gate": gate,
            "path": logical,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    ordinal = 1
    for label, context in primary_contexts.items():
        if label in canary_labels:
            action = "canary"
            harness = canary_labels[label]
        else:
            action = action_by_label[label]
            harness = None
        roles = ["accepted_primary"]
        if label in {"cuda_environment", "offline_smoke"}:
            roles.append("validation_only")
        primary_predecessors = {
            "cuda_environment": [],
            "offline_smoke": [
                predecessor(
                    "modal_cuda_environment_validated",
                    predecessor_paths["cuda"],
                )
            ],
            "candidate_smoke": [
                predecessor(
                    "modal_cuda_environment_validated",
                    predecessor_paths["cuda"],
                ),
                predecessor(
                    "modal_offline_smoke_validated",
                    predecessor_paths["offline"],
                ),
            ],
            "resume_attempt": [
                predecessor(
                    "modal_artifact_round_trip_validated",
                    predecessor_paths["round_trip"],
                )
            ],
        }
        selected_predecessors = (
            [
                predecessor(
                    "candidate_resume_preflight_validated",
                    predecessor_paths["preflight"],
                )
            ]
            if label in canary_labels
            else primary_predecessors[label]
        )
        attempt_id = write_attempt(
            ordinal,
            action=action,
            run_id=context.run_id,
            app_id=context.modal_app_id or "",
            roles=roles,
            source_run_id=(
                primary_runs["candidate_smoke"]
                if label == "resume_attempt"
                else None
            ),
            harness=harness,
            predecessor_receipts=selected_predecessors,
        )
        accepted_attempt_ids[label] = attempt_id
        ordinal += 1
    for label, context in verifier_contexts.items():
        source_attempt_id = accepted_attempt_ids[label]
        source_intent_logical = modal_readiness.modal_action_intent_receipt_path(
            identity, source_attempt_id
        ).as_posix()
        source_terminal_logical = modal_readiness.modal_action_terminal_receipt_path(
            identity, source_attempt_id
        ).as_posix()
        source_process_start_logical = (
            modal_readiness.modal_local_process_start_receipt_path(
                source_attempt_id
            ).as_posix()
        )
        attempt_id = write_attempt(
            ordinal,
            action="download",
            run_id=primary_runs[label],
            app_id=context.modal_app_id or "",
            roles=["artifact_verifier", "validation_only"],
            verifier_run_id=context.run_id,
            predecessor_receipts=[
                predecessor("source_action_intent", source_intent_logical),
                predecessor(
                    "source_action_attempt_terminal",
                    source_terminal_logical,
                ),
                predecessor(
                    "source_local_process_start",
                    source_process_start_logical,
                ),
            ],
        )
        assert verifier_records[label]["attempt_id"] == attempt_id
        ordinal += 1

    provider_outcomes: list[dict[str, object]] = []
    for harness in modal_readiness.CANARY_ORDER:
        label = f"canary_{harness}"
        run_id = primary_runs[label]
        launcher_attempt_id = accepted_attempt_ids[label]
        launcher_receipt = (
            attempt_directory / f"{launcher_attempt_id}.json"
        )
        downloaded_run = root / "outputs/development/modal_downloads" / run_id
        ledger = downloaded_run / "controller/provider_attempts.jsonl"
        verifier = verifier_records[label]
        provider_outcomes.append(
            {
                "launcher_attempt_id": launcher_attempt_id,
                "launcher_attempt_receipt_path": (
                    modal_readiness.modal_action_terminal_receipt_path(
                        identity, launcher_attempt_id
                    ).as_posix()
                ),
                "launcher_attempt_receipt_sha256": hashlib.sha256(
                    launcher_receipt.read_bytes()
                ).hexdigest(),
                "harness": harness,
                "concrete_run_id": run_id,
                "outcome": "accepted",
                "provider_attempt_ledger_path": (
                    f"outputs/development/modal_downloads/{run_id}/controller/"
                    "provider_attempts.jsonl"
                ),
                "provider_attempt_ledger_sha256": hashlib.sha256(
                    ledger.read_bytes()
                ).hexdigest(),
                "provider_start_uncertain_evidence_path": None,
                "provider_start_uncertain_evidence_sha256": None,
                "artifact_verifier": {
                    "attempt_id": verifier["attempt_id"],
                    "verifier_run_id": verifier["verifier_run_id"],
                    "remote_verification_path": verifier[
                        "remote_verification_path"
                    ],
                    "remote_verification_sha256": verifier[
                        "remote_verification_sha256"
                    ],
                },
            }
        )

    price_path = root / (
        modal_readiness.modal_cleanup_snapshot_directory(identity)
        / "provider_price_basis.json"
    )
    selector_path = (
        root
        / modal_readiness.modal_live_cohort_root(identity)
        / "provider_canary_selection/selector-check-1/canary_run_selector.json"
    )
    selector_runs = {}
    for harness in modal_readiness.CANARY_ORDER:
        run_id = primary_runs[f"canary_{harness}"]
        logical = f"outputs/development/modal_downloads/{run_id}"
        manifest_path = root / logical / "artifact_manifest.json"
        context_path = root / logical / "execution_context.json"
        context = json.loads(context_path.read_text(encoding="utf-8"))
        selector_runs[harness] = {
            "harness": harness,
            "run_id": run_id,
            "download_path": logical,
            "artifact_manifest_path": f"{logical}/artifact_manifest.json",
            "raw_artifact_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "execution_context_path": f"{logical}/execution_context.json",
            "execution_context_sha256": hashlib.sha256(
                context_path.read_bytes()
            ).hexdigest(),
            "image_source_sha256": context["image_source_sha256"],
            "modal_image_id": context["modal_image_id"],
        }
    _write_json(
        selector_path,
        {
            "schema_name": "ModalProviderCanaryRunSelector",
            "schema_version": "2.0",
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "selector_id": "selector-check-1",
            "harness_order": list(modal_readiness.CANARY_ORDER),
            "runs": selector_runs,
        },
    )
    roster_path = root / modal_readiness.modal_cohort_roster_path(identity)
    component_receipts = {
        gate: {
            "path": logical,
            "sha256": hashlib.sha256((root / logical).read_bytes()).hexdigest(),
        }
        for gate, logical in {
            "modal_cuda_environment_validated": predecessor_paths["cuda"],
            "modal_offline_smoke_validated": predecessor_paths["offline"],
            "modal_artifact_round_trip_validated": predecessor_paths["round_trip"],
            "candidate_resume_preflight_validated": predecessor_paths["preflight"],
        }.items()
    }
    attribution_by_attempt = {
        item["attempt_id"]: item for item in attributions
    }
    terminal_run_dispositions = []
    for receipt_logical in sorted(receipt_paths):
        receipt = json.loads((root / receipt_logical).read_text(encoding="utf-8"))
        for run_id in receipt["concrete_remote_run_ids"]:
            terminal_run_dispositions.append(
                {
                    "attempt_id": receipt["attempt_id"],
                    "run_id": run_id,
                    "execution_disposition": "remote_execution_bound",
                    "provider_disposition": (
                        "evidence_bound"
                        if receipt["action"] in {"canary", "canaries"}
                        else "not_applicable"
                    ),
                    "snapshot_disposition": "app_volume_and_billing_bound",
                    "snapshot_app_ids": attribution_by_attempt[
                        receipt["attempt_id"]
                    ]["object_ids"],
                    "volume_disposition": "present_bound",
                }
            )
    terminal_run_dispositions.sort(
        key=lambda item: (item["attempt_id"], item["run_id"])
    )
    roster = {
        "schema_name": "ModalMigrationCohortRoster",
        "schema_version": "4.0",
        **modal_readiness.modal_cohort_identity_dict(identity),
        "cleanup_run_id": cleanup_run_id,
        "component_receipts": component_receipts,
        "accepted_primary_runs": primary_runs,
        "accepted_attempt_ids": accepted_attempt_ids,
        "artifact_verifiers": verifier_records,
        "additional_artifact_verifiers": [],
        "provider_canary_outcomes": provider_outcomes,
        "provider_canary_selector_path": selector_path.relative_to(root).as_posix(),
        "provider_canary_selector_sha256": hashlib.sha256(
            selector_path.read_bytes()
        ).hexdigest(),
        "action_intent_receipts": sorted(intent_paths),
        "action_attempt_receipts": sorted(receipt_paths),
        "provider_canary_aggregate_outcome_receipts": [],
        "attempt_classifications": sorted(
            classifications, key=lambda item: item["attempt_id"]
        ),
        "billing_attributions": sorted(
            attributions, key=lambda item: item["attempt_id"]
        ),
        "terminal_run_dispositions": terminal_run_dispositions,
        "recovery_links": [],
        "declared_failed_run_ids": [],
        "declared_quarantined_run_ids": [],
        "declared_recovery_run_ids": [],
        "billing_window_start_utc": "2025-01-01T00:00:00Z",
        "billing_window_end_utc": "2025-01-01T01:00:00Z",
        "snapshot_captured_at_utc": "2025-01-01T01:01:00Z",
        "snapshot_capture_manifest_path": "pending",
        "snapshot_capture_manifest_sha256": "0" * 64,
        "migration_lineage_path": "pending",
        "migration_lineage_sha256": "0" * 64,
        "superseded_usage": {
            "run_id": "modal-cuda-env-20260809-02",
            "amount_usd": "0.00643852",
            "accounting_basis": (
                "preserved_prior_measurement_excluded_from_cohort_billing_snapshot"
            ),
        },
        "provider_price_basis_path": price_path.relative_to(root).as_posix(),
    }
    _write_json(roster_path, roster)

    app_ids = [
        context.modal_app_id
        for context in (*primary_contexts.values(), *verifier_contexts.values())
    ]
    snapshot_root = root / (
        cleanup_capture.modal_cleanup_snapshot_capture_manifest_path(
            identity,
            "capture-final-1",
        ).parent
    )
    _write_json(
        snapshot_root / "app_list.json",
        [
            {
                "app_id": app_id,
                "description": APP_NAME,
                "state": "stopped",
                "tasks": "0",
                "created_at": "2025-01-01 00:04:45+00:00",
                "stopped_at": "2025-01-01 00:06:15+00:00",
            }
            for app_id in app_ids
        ],
    )
    _write_json(snapshot_root / "container_list.json", [])
    _write_json(snapshot_root / "endpoint_list.json", [])
    _write_json(
        snapshot_root / "volume_list.json",
        [
            {
                "name": "rl4rl-architecture-artifacts",
                "created_at": "2025-01-01 00:00:00+00:00",
                "created_by": "test-user",
            }
        ],
    )
    volume_run_ids = sorted(
        {
            *primary_runs.values(),
            *(record["verifier_run_id"] for record in verifier_records.values()),
            "modal-cuda-env-20260809-02",
        }
    )
    _write_json(
        snapshot_root / "run_directory_list.json",
        [
            {
                "filename": f"/runs/{run_id}",
                "type": "dir",
                "created_modified": "2025-01-01 00:30 UTC",
                "size": "0 B",
            }
            for run_id in volume_run_ids
        ],
    )
    _write_json(
        snapshot_root / "billing_report.json",
        [
            {
                "object_id": app_id,
                "description": APP_NAME,
                "environment": "main",
                "interval_start": "2025-01-01T00:00:00+00:00",
                "resource": "T4 GPU" if index < 8 else "CPU",
                "cost": "0.01",
            }
            for index, app_id in enumerate(app_ids)
        ],
    )
    _refresh_snapshot_capture_manifest(snapshot_root)
    _refresh_migration_lineage(roster_path)
    (root / "experiment_manifest.yaml").write_text(
        "remote_execution:\n  detached_calls: false\n",
        encoding="utf-8",
    )
    return roster_path, snapshot_root


def _prior_quarantine_accounting_fixture(
    root: Path,
) -> tuple[dict[str, object], str]:
    """Build a real all-started prior-cohort 1.1 accounting seal."""

    roster_path, snapshot_root = _aggregate_cleanup_fixture(root)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    journal, attempts = modal_readiness._cohort_action_journal(root, identity)
    attempt_by_id = {attempt["attempt_id"]: attempt for attempt in attempts}

    verifier_by_attempt = {
        record["attempt_id"]: record
        for record in roster["artifact_verifiers"].values()
    }
    execution_records: list[dict[str, object]] = []
    execution_contexts: dict[tuple[str, str], ExecutionContextV1] = {}
    verifier_manifest_paths: dict[str, Path] = {}
    for attempt in attempts:
        attempt_id = attempt["attempt_id"]
        for run_id in attempt["concrete_remote_run_ids"]:
            if attempt["action"] in {"download", "verify"}:
                verifier = verifier_by_attempt[attempt_id]
                context = ExecutionContextV1.from_dict(
                    verifier["verifier_execution_context"]
                )
                capture_root = root / (
                    modal_readiness.modal_artifact_verifier_capture_directory_path(
                        identity,
                        attempt["run_id"],
                        run_id,
                        attempt_id,
                    )
                )
                context_path = capture_root / "execution_context.json"
                _write_json(context_path, context.to_dict())
                capture_manifest = build_artifact_manifest(
                    capture_root,
                    run_id=run_id,
                    image_source_sha256=identity.image_source_sha256,
                )
                manifest_path = write_artifact_manifest(
                    capture_root,
                    capture_manifest,
                )
                verifier_manifest_paths[run_id] = manifest_path
            else:
                context_path = (
                    root
                    / "outputs/development/modal_downloads"
                    / run_id
                    / "execution_context.json"
                )
                context = ExecutionContextV1.from_dict(
                    json.loads(context_path.read_text(encoding="utf-8"))
                )
            logical = context_path.relative_to(root).as_posix()
            raw = context_path.read_bytes()
            execution_records.append(
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "execution_context_path": logical,
                    "execution_context_sha256": hashlib.sha256(raw).hexdigest(),
                    "execution_context_size_bytes": len(raw),
                }
            )
            execution_contexts[(attempt_id, run_id)] = context
    execution_records.sort(key=lambda item: (item["attempt_id"], item["run_id"]))

    provider_records: list[dict[str, object]] = []
    provider_cost_evidence: dict[tuple[str, str], dict[str, object]] = {}
    for attempt in attempts:
        if attempt["action"] not in {"canary", "canaries"}:
            continue
        for run_id in attempt["concrete_remote_run_ids"]:
            logical = (
                f"outputs/development/modal_downloads/{run_id}/controller/"
                "provider_attempts.jsonl"
            )
            path = root / logical
            raw = path.read_bytes()
            records = modal_readiness._strict_provider_ledger(
                path,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
            )
            provider_records.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "run_id": run_id,
                    "state": "ledger",
                    "evidence_path": logical,
                    "evidence_sha256": hashlib.sha256(raw).hexdigest(),
                    "evidence_size_bytes": len(raw),
                    "provider_attempt_count": len(records),
                    "request_ids": sorted(
                        record.provider_request_id
                        for record in records
                        if record.provider_request_id is not None
                    ),
                    "response_ids": sorted(
                        record.provider_response_id
                        for record in records
                        if record.provider_response_id is not None
                    ),
                }
            )
            provider_cost_evidence[(attempt["attempt_id"], run_id)] = {
                "state": "ledger",
                "records": records,
                "harness": modal_readiness._provider_harness_for_run(
                    attempt,
                    run_id,
                ),
            }
    provider_records.sort(key=lambda item: (item["attempt_id"], item["run_id"]))

    snapshot_manifest_path = root / roster["snapshot_capture_manifest_path"]
    snapshot_manifest_raw = snapshot_manifest_path.read_bytes()
    billing_rows = json.loads((snapshot_root / "billing_report.json").read_text())
    volume_rows = json.loads(
        (snapshot_root / "run_directory_list.json").read_text()
    )
    app_to_attempt = {
        context.modal_app_id: attempt_id
        for (attempt_id, _run_id), context in execution_contexts.items()
    }
    app_lifecycles = [
        {
            "attempt_id": attempt_id,
            "app_id": app_id,
            "created_at_utc": "2025-01-01T00:04:45Z",
            "stopped_at_utc": "2025-01-01T00:06:15Z",
        }
        for app_id, attempt_id in sorted(app_to_attempt.items())
    ]
    selected_billing_rows = sorted(
        (
            {
                "attempt_id": app_to_attempt[row["object_id"]],
                "app_id": row["object_id"],
                "row_sha256": modal_readiness.canonical_sha256(row),
                "row": row,
            }
            for row in billing_rows
            if row["object_id"] in app_to_attempt
        ),
        key=lambda item: (item["attempt_id"], item["row_sha256"]),
    )
    measured_by_attempt = {
        attempt_id: modal_readiness.Decimal("0") for attempt_id in attempt_by_id
    }
    for record in selected_billing_rows:
        measured_by_attempt[record["attempt_id"]] += modal_readiness.Decimal(
            record["row"]["cost"]
        )

    volume_entry_by_run_id = {
        PurePosixPath(row["filename"].removeprefix("/")).parts[-1]: row
        for row in volume_rows
    }
    volume_dispositions: list[dict[str, object]] = []
    for run_id in sorted(
        run_id
        for attempt in attempts
        for run_id in attempt["concrete_remote_run_ids"]
    ):
        if run_id in verifier_manifest_paths:
            manifest_path = verifier_manifest_paths[run_id]
        else:
            run_root = root / "outputs/development/modal_downloads" / run_id
            raw_manifest = modal_readiness._select_downloaded_raw_manifest(run_root)
            manifest_path = run_root / raw_manifest.filename
        manifest_raw = manifest_path.read_bytes()
        entry = volume_entry_by_run_id[run_id]
        volume_dispositions.append(
            {
                "run_id": run_id,
                "entry_sha256": modal_readiness.canonical_sha256(entry),
                "entry": entry,
                "artifact_manifest_disposition": "bound",
                "artifact_manifest_path": manifest_path.relative_to(root).as_posix(),
                "artifact_manifest_sha256": hashlib.sha256(
                    manifest_raw
                ).hexdigest(),
                "artifact_manifest_size_bytes": len(manifest_raw),
            }
        )

    reservation_bindings = sorted(
        {
            record["path"]: {
                **record,
                "size_bytes": len((root / record["path"]).read_bytes()),
            }
            for attempt in attempts
            for record in attempt["remote_run_reservations"]
        }.values(),
        key=lambda item: item["path"],
    )
    price_logical = attempts[0]["modal_price_basis_path"]
    price_path = root / price_logical
    price_raw = price_path.read_bytes()
    _price, rates, _path = modal_readiness.load_modal_price_basis(
        root,
        price_logical,
        expected_raw_sha256=hashlib.sha256(price_raw).hexdigest(),
        expected_image_source_sha256=identity.image_source_sha256,
        require_freshness=False,
    )
    retained_count = len(volume_dispositions)
    bytes_per_run = (
        modal_readiness.MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES
        + modal_readiness.MAX_ARTIFACT_MANIFEST_BYTES
    )
    total_bytes = retained_count * bytes_per_run
    estimated_gib = modal_readiness.Decimal(total_bytes) / modal_readiness.Decimal(
        1024**3
    )
    remote_dispositions = roster["terminal_run_dispositions"]
    payload: dict[str, object] = {
        "schema_name": "ModalPriorCohortQuarantineAccounting",
        "schema_version": "1.1",
        **modal_readiness.modal_cohort_identity_dict(identity),
        "recorded_at_utc": "2025-01-01T02:00:00Z",
        "action_journal": journal,
        "remote_run_reservations": reservation_bindings,
        "attempt_dispositions": [
            {
                "attempt_id": attempt["attempt_id"],
                "action": attempt["action"],
                "status": attempt["status"],
                "concrete_remote_run_ids": attempt["concrete_remote_run_ids"],
                "disposition": "quarantined",
            }
            for attempt in sorted(attempts, key=lambda item: item["attempt_id"])
        ],
        "remote_run_dispositions": remote_dispositions,
        "remote_executions": execution_records,
        "provider_attempt_evidence": provider_records,
        "unbound_provider_evidence": [],
        "provider_spend_estimate": (
            modal_readiness._derive_journal_provider_spend_estimate(
                root,
                identity=identity,
                attempts=attempts,
                remote_run_dispositions={
                    (record["attempt_id"], record["run_id"]): record
                    for record in remote_dispositions
                },
                provider_evidence=provider_cost_evidence,
                accounting_label=(
                    "prior_quarantined_known_usage_plus_failed_and_"
                    "may_have_started_conservative_reserves_not_billed_cost"
                ),
            )
        ),
        "modal_compute_exposure": modal_readiness._derive_modal_compute_exposure(
            attempts,
            measured_by_attempt=measured_by_attempt,
            unresolved_attempt_ids=set(),
            accounting_label=(
                "prior_quarantined_measured_billing_plus_unresolved_or_lagged_"
                "compute_reserve_not_a_platform_hard_bound"
            ),
        ),
        "snapshot_capture_manifest_path": roster[
            "snapshot_capture_manifest_path"
        ],
        "snapshot_capture_manifest_sha256": hashlib.sha256(
            snapshot_manifest_raw
        ).hexdigest(),
        "snapshot_capture_manifest_size_bytes": len(snapshot_manifest_raw),
        "app_lifecycles": app_lifecycles,
        "selected_billing_rows": selected_billing_rows,
        "app_compute_subtotal_usd": format(
            sum(measured_by_attempt.values(), modal_readiness.Decimal("0")),
            "f",
        ),
        "volume_dispositions": volume_dispositions,
        "modal_price_basis": {
            "path": price_logical,
            "sha256": hashlib.sha256(price_raw).hexdigest(),
            "size_bytes": len(price_raw),
        },
        "active_app_count": 0,
        "active_container_count": 0,
        "active_endpoint_count": 0,
        "accepted_contexts": [],
        "retained_storage_estimate": {
            "retained_run_count": retained_count,
            "conservative_bytes_per_run": bytes_per_run,
            "conservative_total_bytes": total_bytes,
            "estimated_gib": format(estimated_gib, "f"),
            "volume_rate_usd_per_gib_month": format(rates["volume"], "f"),
            "estimated_monthly_usd": format(
                estimated_gib * rates["volume"],
                "f",
            ),
            "basis": (
                "retained_run_count_times_per_run_artifact_download_and_"
                "manifest_caps; included_shared_volume_quota_not_subtracted"
            ),
        },
        "validated": True,
    }
    logical = modal_readiness.modal_prior_quarantine_accounting_path(
        identity
    ).as_posix()
    return payload, logical


def _prior_quarantine_accounting_request(
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_name": "ModalPriorCohortQuarantineAccountingRequest",
        "schema_version": "1.0",
        "source_tree_sha256": payload["source_tree_sha256"],
        "image_source_sha256": payload["image_source_sha256"],
        "cohort_id": payload["cohort_id"],
        "recorded_at_utc": payload["recorded_at_utc"],
        "snapshot_capture_manifest": {
            "path": payload["snapshot_capture_manifest_path"],
            "sha256": payload["snapshot_capture_manifest_sha256"],
            "size_bytes": payload["snapshot_capture_manifest_size_bytes"],
        },
    }


def _refresh_prior_snapshot_binding(
    root: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    """Rebind a mutated prior snapshot in both payload and operator request."""

    manifest_path = root / str(payload["snapshot_capture_manifest_path"])
    _refresh_snapshot_capture_manifest(manifest_path.parent)
    manifest_raw = manifest_path.read_bytes()
    payload["snapshot_capture_manifest_sha256"] = hashlib.sha256(
        manifest_raw
    ).hexdigest()
    payload["snapshot_capture_manifest_size_bytes"] = len(manifest_raw)
    return _prior_quarantine_accounting_request(payload)


def _write_attempt_intent_and_terminal(
    root: Path,
    receipt: dict[str, object],
    *,
    identity: ModalLiveCohortIdentity | None = None,
) -> tuple[str, str]:
    attempt_id = str(receipt["attempt_id"])
    selected_identity = identity or _fixture_identity(root)
    for field, value in _fixture_local_containment().items():
        receipt.setdefault(field, value)
    terminal_logical = modal_readiness.modal_action_terminal_receipt_path(
        selected_identity,
        attempt_id,
    ).as_posix()
    intent_logical = modal_readiness.modal_action_intent_receipt_path(
        selected_identity,
        attempt_id,
    ).as_posix()
    intent = {
        "schema_name": "ModalActionIntent",
        "schema_version": "1.6",
        "attempt_id": attempt_id,
        "created_at_utc": receipt["started_at_utc"],
        **{
            field: receipt[field]
            for field in modal_readiness._INTENT_TERMINAL_SHARED_FIELDS
        },
    }
    intent_path = root / intent_logical
    _write_json(intent_path, intent)
    intent_path.chmod(0o600)
    if receipt["modal_cli_process_started"] is True:
        process_id = 10_000 + int(attempt_id[-6:], 16)
        marker_logical = (
            modal_readiness.modal_local_process_start_receipt_path(
                attempt_id
            ).as_posix()
        )
        marker = {
            "schema_name": "ModalLocalProcessStart",
            "schema_version": "1.1",
            "attempt_id": attempt_id,
            "created_at_utc": receipt["started_at_utc"],
            "action": receipt["action"],
            "run_id": receipt["run_id"],
            "intent_path": intent_logical,
            "intent_sha256": hashlib.sha256(
                (root / intent_logical).read_bytes()
            ).hexdigest(),
            "source_tree_sha256": receipt["source_tree_sha256"],
            "image_source_sha256": receipt[
                "approved_image_source_sha256"
            ],
            "cohort_id": receipt["cohort_id"],
            "modal_command_sha256": receipt["modal_command_sha256"],
            "launch_capability_sha256": receipt[
                "launch_capability_sha256"
            ],
            "modal_cost_cap_usd": receipt["modal_cost_cap_usd"],
            "provider_cost_cap_usd": receipt["provider_cost_cap_usd"],
            **{
                field: receipt[field]
                for field in _fixture_local_containment()
            },
            "process_id": process_id,
            "expected_process_group_id": process_id,
            "expected_session_id": process_id,
            "process_birth_identity_sha256": hashlib.sha256(
                f"fixture-process-birth:{attempt_id}".encode("ascii")
            ).hexdigest(),
        }
        marker_path = root / marker_logical
        _write_json(marker_path, marker)
        marker_path.chmod(0o600)
        marker_path.parent.chmod(0o700)
        marker_path.parent.parent.chmod(0o700)
        receipt.update(
            {
                "local_process_start_receipt_path": marker_logical,
                "local_process_start_receipt_sha256": hashlib.sha256(
                    marker_path.read_bytes()
                ).hexdigest(),
                "local_process_id": process_id,
                "local_process_group_id": process_id,
                "local_session_id": process_id,
            }
        )
    else:
        receipt.update(
            {
                "local_process_start_receipt_path": None,
                "local_process_start_receipt_sha256": None,
                "local_process_id": None,
                "local_process_group_id": None,
                "local_session_id": None,
            }
        )
    terminal_path = root / terminal_logical
    _write_json(terminal_path, receipt)
    terminal_path.chmod(0o600)
    return intent_logical, terminal_logical


def _refresh_process_start_intent_binding(
    root: Path,
    *,
    attempt_id: str,
    terminal_path: Path,
) -> None:
    """Rebind a fixture marker after deliberately rewriting its intent."""

    marker_path = (
        root
        / modal_readiness.modal_local_process_start_receipt_path(attempt_id)
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["intent_sha256"] = hashlib.sha256(
        (root / marker["intent_path"]).read_bytes()
    ).hexdigest()
    _write_json(marker_path, marker)
    marker_path.chmod(0o600)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["local_process_start_receipt_sha256"] = hashlib.sha256(
        marker_path.read_bytes()
    ).hexdigest()
    _write_json(terminal_path, terminal)


def _clear_process_start_evidence(receipt: dict[str, object]) -> None:
    receipt.update(
        {
            "local_process_start_receipt_path": None,
            "local_process_start_receipt_sha256": None,
            "local_process_id": None,
            "local_process_group_id": None,
            "local_session_id": None,
        }
    )


def _add_stopped_app_and_billing(
    snapshot_root: Path,
    app_id: str,
    *,
    resource: str,
    created_at: str,
    stopped_at: str,
) -> None:
    app_path = snapshot_root / "app_list.json"
    apps = json.loads(app_path.read_text(encoding="utf-8"))
    apps.append(
        {
            "app_id": app_id,
            "description": APP_NAME,
            "state": "stopped",
            "tasks": "0",
            "created_at": created_at,
            "stopped_at": stopped_at,
        }
    )
    _write_json(app_path, apps)
    billing_path = snapshot_root / "billing_report.json"
    billing = json.loads(billing_path.read_text(encoding="utf-8"))
    billing.append(
        {
            "object_id": app_id,
            "description": APP_NAME,
            "environment": "main",
            "interval_start": "2025-01-01T00:00:00+00:00",
            "resource": resource,
            "cost": "0.01",
        }
    )
    _write_json(billing_path, billing)
    _refresh_snapshot_capture_manifest(snapshot_root)


def _append_volume_run(snapshot_root: Path, run_id: str) -> None:
    path = snapshot_root / "run_directory_list.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "filename": f"/runs/{run_id}",
            "type": "dir",
            "created_modified": "2025-01-01 00:30 UTC",
            "size": "0 B",
        }
    )
    rows.sort(key=lambda item: item["filename"])
    _write_json(path, rows)
    _refresh_snapshot_capture_manifest(snapshot_root)


def _ordinary_failure_recovery_fixture(
    root: Path,
) -> tuple[Path, dict[str, str]]:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(root)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    accepted_attempt_id = roster["accepted_attempt_ids"]["offline_smoke"]
    attempt_root = _attempt_root(root)
    accepted_path = attempt_root / f"{accepted_attempt_id}.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_run = root / (
        "outputs/development/modal_downloads/"
        + roster["accepted_primary_runs"]["offline_smoke"]
    )
    raw_image_manifest = json.loads(
        (accepted_run / "image_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256=raw_image_manifest["dependency_lock_sha256"],
        files=tuple(
            SourceFileV1(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
            )
            for item in raw_image_manifest["files"]
        ),
    )

    failed_attempt_id = f"{17:032x}"
    failed_run_id = "failed-ordinary-cuda"
    failed_app_id = "ap-failed-ordinary-cuda"
    failed_root = root / f"outputs/development/modal_downloads/{failed_run_id}"
    failed_root.mkdir(parents=True)
    failed_context = ExecutionContextV1(
        execution_backend="modal",
        run_id=failed_run_id,
        app_name=APP_NAME,
        function_name="offline_smoke",
        modal_app_id=failed_app_id,
        modal_function_id="fu-failed-ordinary-cuda",
        modal_call_id="fc-failed-ordinary-cuda",
        modal_image_id="im-test123",
        image_source_sha256=image_manifest.manifest_sha256,
        artifact_uri=volume_artifact_uri(failed_run_id),
    )
    _write_json(failed_root / "execution_context.json", failed_context.to_dict())
    _write_json(
        failed_root / "image_source_manifest.json",
        image_manifest.to_dict(),
    )
    _write_json(
        failed_root / "remote_action_failure.json",
        {
            "error_type": "RuntimeError",
            "message": "remote action failed; details suppressed",
        },
    )
    _write_json(
        failed_root / "remote_action_result.json",
        {"success": False, "error_type": "RuntimeError"},
    )
    failed_manifest = build_artifact_manifest(
        failed_root,
        run_id=failed_run_id,
        image_source_sha256=image_manifest.manifest_sha256,
    )
    write_artifact_manifest(failed_root, failed_manifest)

    failed_receipt = dict(accepted)
    failed_receipt.update(
        {
            "attempt_id": failed_attempt_id,
            "started_at_utc": "2025-01-01T00:01:30Z",
            "finished_at_utc": "2025-01-01T00:02:00Z",
            "status": "failed",
            "failure_kind": "modal_cli_exit",
            "run_id": failed_run_id,
            "concrete_remote_run_ids": [failed_run_id],
            "returncode": 2,
        }
    )
    _refresh_receipt_reservations(root, failed_receipt)
    failed_receipt["modal_command_sha256"] = (
        modal_readiness._reconstructed_modal_command_sha256(
            failed_receipt,
            root=root,
        )
    )
    failed_intent, failed_terminal = _write_attempt_intent_and_terminal(
        root,
        failed_receipt,
    )

    recovery_verifier_attempt_id = f"{18:032x}"
    primary_verifier = roster["artifact_verifiers"]["offline_smoke"]
    verifier_template_path = (
        attempt_root / f"{primary_verifier['attempt_id']}.json"
    )
    verifier_receipt = json.loads(
        verifier_template_path.read_text(encoding="utf-8")
    )
    verifier_run_id = _verifier_run_id(failed_run_id)
    verifier_capture = root / (
        modal_readiness.modal_artifact_verifier_capture_directory_path(
            _fixture_identity(root),
            failed_run_id,
            verifier_run_id,
            recovery_verifier_attempt_id,
        )
    )
    verifier_capture.mkdir(parents=True)
    verification = _remote_verification_payload(root, failed_run_id)
    verifier_context = ExecutionContextV1.from_dict(
        verification["verifier_execution_context"]
    )
    _write_json(
        verifier_capture / "artifact_verification_result.json",
        verification,
    )
    _write_json(
        verifier_capture / "execution_context.json",
        verifier_context.to_dict(),
    )
    _write_json(
        verifier_capture / "image_source_manifest.json",
        image_manifest.to_dict(),
    )
    capture_manifest = build_artifact_manifest(
        verifier_capture,
        run_id=verifier_run_id,
        image_source_sha256=image_manifest.manifest_sha256,
    )
    write_artifact_manifest(verifier_capture, capture_manifest)
    result_path = verifier_capture / "artifact_verification_result.json"
    verifier_receipt.update(
        {
            "attempt_id": recovery_verifier_attempt_id,
            "started_at_utc": "2025-01-01T00:03:00Z",
            "finished_at_utc": "2025-01-01T00:04:00Z",
            "run_id": failed_run_id,
            "concrete_remote_run_ids": [verifier_run_id],
            "verifier_run_id": verifier_run_id,
            "predecessor_receipts": [
                *_local_freeze_predecessors(root),
                {
                    "gate": "source_action_intent",
                    "path": failed_intent,
                    "sha256": hashlib.sha256(
                        (root / failed_intent).read_bytes()
                    ).hexdigest(),
                },
                {
                    "gate": "source_action_attempt_terminal",
                    "path": failed_terminal,
                    "sha256": hashlib.sha256(
                        (root / failed_terminal).read_bytes()
                    ).hexdigest(),
                },
                {
                    "gate": "source_local_process_start",
                    "path": (
                        modal_readiness.modal_local_process_start_receipt_path(
                            failed_attempt_id
                        ).as_posix()
                    ),
                    "sha256": hashlib.sha256(
                        (
                            root
                            / modal_readiness.modal_local_process_start_receipt_path(
                                failed_attempt_id
                            )
                        ).read_bytes()
                    ).hexdigest(),
                },
            ],
            "source_evidence_recovery": True,
        }
    )
    _refresh_receipt_reservations(root, verifier_receipt)
    verifier_receipt["modal_command_sha256"] = (
        modal_readiness._reconstructed_modal_command_sha256(
            verifier_receipt,
            root=root,
        )
    )
    verifier_intent, verifier_terminal = _write_attempt_intent_and_terminal(
        root,
        verifier_receipt,
    )

    roster["action_intent_receipts"].extend([failed_intent, verifier_intent])
    roster["action_intent_receipts"].sort()
    roster["action_attempt_receipts"].extend(
        [failed_terminal, verifier_terminal]
    )
    roster["action_attempt_receipts"].sort()
    for classification in roster["attempt_classifications"]:
        if classification["attempt_id"] == accepted_attempt_id:
            classification["roles"] = sorted(
                [*classification["roles"], "recovery"]
            )
    roster["attempt_classifications"].extend(
        [
            {
                "attempt_id": failed_attempt_id,
                "roles": ["failed", "quarantined"],
            },
            {
                "attempt_id": recovery_verifier_attempt_id,
                "roles": ["artifact_verifier", "validation_only"],
            },
        ]
    )
    roster["attempt_classifications"].sort(
        key=lambda item: item["attempt_id"]
    )
    roster["billing_attributions"].extend(
        [
            {
                "attempt_id": failed_attempt_id,
                "disposition": "billed",
                "object_ids": [failed_app_id],
            },
            {
                "attempt_id": recovery_verifier_attempt_id,
                "disposition": "billed",
                "object_ids": [verifier_context.modal_app_id],
            },
        ]
    )
    roster["billing_attributions"].sort(key=lambda item: item["attempt_id"])
    roster["additional_artifact_verifiers"] = [
        {
            "source_run_id": failed_run_id,
            "verifier_run_id": verifier_run_id,
            "attempt_id": recovery_verifier_attempt_id,
            "status": "succeeded",
            "remote_verifier_outcome": "success",
            "remote_evidence_kind": "volume_success_capture",
            "billing_object_ids": [verifier_context.modal_app_id],
            "remote_verification_path": result_path.relative_to(root).as_posix(),
            "remote_verification_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
            "verifier_execution_context": verifier_context.to_dict(),
            "failure_receipt_path": None,
            "failure_receipt_sha256": None,
            "failure_execution_context": None,
            "recovery_verifier_attempt_id": None,
            "expected_remote_receipt_roster": list(
                modal_readiness._VERIFIER_REMOTE_RECEIPT_ROSTER
            ),
        }
    ]
    roster["recovery_links"] = [
        {
            "failed_attempt_id": failed_attempt_id,
            "recovery_attempt_id": accepted_attempt_id,
            "recovered_run_ids": [
                roster["accepted_primary_runs"]["offline_smoke"]
            ],
        }
    ]
    roster["declared_failed_run_ids"] = [failed_run_id]
    roster["declared_quarantined_run_ids"] = [failed_run_id]
    roster["declared_recovery_run_ids"] = [
        roster["accepted_primary_runs"]["offline_smoke"]
    ]
    _write_json(roster_path, roster)

    _add_stopped_app_and_billing(
        snapshot_root,
        failed_app_id,
        resource="T4 GPU",
        created_at="2025-01-01 00:01:15+00:00",
        stopped_at="2025-01-01 00:02:15+00:00",
    )
    _add_stopped_app_and_billing(
        snapshot_root,
        verifier_context.modal_app_id or "",
        resource="CPU",
        created_at="2025-01-01 00:02:45+00:00",
        stopped_at="2025-01-01 00:04:15+00:00",
    )
    _append_volume_run(snapshot_root, failed_run_id)
    _append_volume_run(snapshot_root, verifier_run_id)
    _refresh_migration_lineage(roster_path)
    return roster_path, {
        "accepted_attempt_id": accepted_attempt_id,
        "failed_attempt_id": failed_attempt_id,
        "failed_run_id": failed_run_id,
        "failed_app_id": failed_app_id,
        "verifier_attempt_id": recovery_verifier_attempt_id,
        "verifier_run_id": verifier_run_id,
        "verifier_app_id": verifier_context.modal_app_id or "",
    }


def _failed_verifier_retry_fixture(
    root: Path,
) -> tuple[Path, dict[str, str]]:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(root)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    source_run_id = roster["accepted_primary_runs"]["cuda_environment"]
    source_root = root / f"outputs/development/modal_downloads/{source_run_id}"
    raw_image_manifest = json.loads(
        (source_root / "image_source_manifest.json").read_text(encoding="utf-8")
    )
    image_manifest = ImageSourceManifestV1(
        dependency_lock_sha256=raw_image_manifest["dependency_lock_sha256"],
        files=tuple(
            SourceFileV1(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
            )
            for item in raw_image_manifest["files"]
        ),
    )
    primary_verifier = roster["artifact_verifiers"]["cuda_environment"]
    recovery_attempt_id = primary_verifier["attempt_id"]
    failed_attempt_id = f"{17:032x}"
    failed_verifier_run_id = "failed-paid-verifier-cuda"
    failed_verifier_app_id = "ap-failed-paid-verifier-cuda"
    failed_context = ExecutionContextV1(
        execution_backend="modal",
        run_id=failed_verifier_run_id,
        app_name=APP_NAME,
        function_name="artifact_verify",
        modal_app_id=failed_verifier_app_id,
        modal_function_id="fu-failed-paid-verifier-cuda",
        modal_call_id="fc-failed-paid-verifier-cuda",
        modal_image_id="im-test123",
        image_source_sha256=image_manifest.manifest_sha256,
        artifact_uri=volume_artifact_uri(source_run_id),
    )
    capture_root = root / (
        modal_readiness.modal_artifact_verifier_capture_directory_path(
            _fixture_identity(root),
            source_run_id,
            failed_verifier_run_id,
            failed_attempt_id,
        )
    )
    capture_root.mkdir(parents=True)
    failure_receipt = {
        "schema_name": "ModalArtifactVerificationFailure",
        "schema_version": "1.0",
        "source_run_id": source_run_id,
        "verifier_run_id": failed_verifier_run_id,
        "error_type": "ArtifactIntegrityError",
        "message": "artifact verification failed; details suppressed",
        "verifier_execution_context": failed_context.to_dict(),
    }
    _write_json(
        capture_root / "artifact_verification_failure.json",
        failure_receipt,
    )
    _write_json(capture_root / "execution_context.json", failed_context.to_dict())
    _write_json(
        capture_root / "image_source_manifest.json",
        image_manifest.to_dict(),
    )
    failure_manifest = build_artifact_manifest(
        capture_root,
        run_id=failed_verifier_run_id,
        image_source_sha256=image_manifest.manifest_sha256,
    )
    write_artifact_manifest(capture_root, failure_manifest)
    failure_path = capture_root / "artifact_verification_failure.json"

    template_path = _attempt_root(root) / f"{recovery_attempt_id}.json"
    failed_attempt = json.loads(template_path.read_text(encoding="utf-8"))
    failed_attempt.update(
        {
            "attempt_id": failed_attempt_id,
            "started_at_utc": "2025-01-01T00:02:00Z",
            "finished_at_utc": "2025-01-01T00:03:00Z",
            "status": "failed",
            "failure_kind": "modal_cli_exit",
            "concrete_remote_run_ids": [failed_verifier_run_id],
            "verifier_run_id": failed_verifier_run_id,
            "returncode": 2,
        }
    )
    _refresh_receipt_reservations(root, failed_attempt)
    failed_attempt["modal_command_sha256"] = (
        modal_readiness._reconstructed_modal_command_sha256(
            failed_attempt,
            root=root,
        )
    )
    failed_intent, failed_terminal = _write_attempt_intent_and_terminal(
        root,
        failed_attempt,
    )
    roster["action_intent_receipts"].append(failed_intent)
    roster["action_intent_receipts"].sort()
    roster["action_attempt_receipts"].append(failed_terminal)
    roster["action_attempt_receipts"].sort()
    roster["attempt_classifications"].append(
        {
            "attempt_id": failed_attempt_id,
            "roles": [
                "artifact_verifier",
                "failed",
                "quarantined",
                "validation_only",
            ],
        }
    )
    roster["attempt_classifications"].sort(
        key=lambda item: item["attempt_id"]
    )
    roster["billing_attributions"].append(
        {
            "attempt_id": failed_attempt_id,
            "disposition": "billed",
            "object_ids": [failed_verifier_app_id],
        }
    )
    roster["billing_attributions"].sort(key=lambda item: item["attempt_id"])
    roster["additional_artifact_verifiers"] = [
        {
            "source_run_id": source_run_id,
            "verifier_run_id": failed_verifier_run_id,
            "attempt_id": failed_attempt_id,
            "status": "failed",
            "remote_verifier_outcome": "failure",
            "remote_evidence_kind": "volume_failure_capture",
            "billing_object_ids": [failed_verifier_app_id],
            "remote_verification_path": None,
            "remote_verification_sha256": None,
            "verifier_execution_context": None,
            "failure_receipt_path": failure_path.relative_to(root).as_posix(),
            "failure_receipt_sha256": hashlib.sha256(
                failure_path.read_bytes()
            ).hexdigest(),
            "failure_execution_context": failed_context.to_dict(),
            "recovery_verifier_attempt_id": recovery_attempt_id,
            "expected_remote_receipt_roster": list(
                modal_readiness._FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER
            ),
        }
    ]
    roster["declared_failed_run_ids"] = [failed_verifier_run_id]
    roster["declared_quarantined_run_ids"] = [failed_verifier_run_id]
    _write_json(roster_path, roster)
    _add_stopped_app_and_billing(
        snapshot_root,
        failed_verifier_app_id,
        resource="CPU",
        created_at="2025-01-01 00:01:45+00:00",
        stopped_at="2025-01-01 00:03:15+00:00",
    )
    _append_volume_run(snapshot_root, failed_verifier_run_id)
    _refresh_migration_lineage(roster_path)
    return roster_path, {
        "source_run_id": source_run_id,
        "failed_attempt_id": failed_attempt_id,
        "failed_verifier_run_id": failed_verifier_run_id,
        "failed_verifier_app_id": failed_verifier_app_id,
        "recovery_attempt_id": recovery_attempt_id,
        "recovery_verifier_run_id": primary_verifier["verifier_run_id"],
        "recovery_remote_path": primary_verifier["remote_verification_path"],
    }


@pytest.mark.parametrize("gate_name", MODAL_READINESS_RECEIPT_CONTRACTS)
def test_pending_modal_receipt_paths_fail_closed(
    tmp_path: Path,
    gate_name: str,
) -> None:
    ledger = _ledger(gate_name, passed=False, receipt_path=None)

    with pytest.raises(FileNotFoundError, match="receipt path is pending"):
        validate_modal_readiness_gate_record(ledger, gate_name, root=tmp_path)


def test_gate_record_rejects_path_and_contract_drift(tmp_path: Path) -> None:
    gate = "modal_cuda_environment_validated"
    ledger = _ledger(gate, passed=True, receipt_path="outputs/readiness/wrong.json")
    with pytest.raises(ValueError, match="receipt path drifted"):
        validate_modal_readiness_gate_record(ledger, gate, root=tmp_path)

    ledger = _ledger(
        gate,
        passed=True,
        receipt_path=MODAL_READINESS_RECEIPT_CONTRACTS[gate]["receipt_path"],
    )
    ledger["gates"][gate]["receipt_contract"]["schema_version"] = "0.0"
    with pytest.raises(ValueError, match="receipt contract drifted"):
        validate_modal_readiness_gate_record(ledger, gate, root=tmp_path)


def test_cuda_and_round_trip_recorders_revalidate_downloaded_artifacts(
    tmp_path: Path,
) -> None:
    run_id = "cuda-check-1"
    run = _downloaded_cuda_run(tmp_path, run_id)

    cuda = record_cuda_environment(run_id=run_id, root=tmp_path)
    assert cuda["observed_gpu_name"] == "NVIDIA T4"
    assert cuda["files_verified"] == 5
    assert cuda["schema_version"] == "2.0"
    assert cuda["remote_action_result_sha256"] == hashlib.sha256(
        (run / "remote_action_result.json").read_bytes()
    ).hexdigest()
    assert json.loads((run / "cuda_environment.json").read_text())["git_version"] == (
        "git version 2.30.2"
    )
    cuda_gate = "modal_cuda_environment_validated"
    cuda_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[cuda_gate][
        "receipt_path"
    ]
    assert "gpu=NVIDIA T4" in validate_modal_readiness_receipt(
        cuda_gate, cuda_path, root=tmp_path
    )

    candidate_run_id = "candidate-check-1"
    _downloaded_candidate_run(tmp_path, candidate_run_id)
    _write_remote_verification(tmp_path, candidate_run_id)
    round_trip = record_artifact_round_trip(
        source_run_id=candidate_run_id,
        verifier_run_id=_verifier_run_id(candidate_run_id),
        root=tmp_path,
    )
    assert round_trip["schema_version"] == "3.0"
    assert round_trip["source_run_id"] == candidate_run_id
    assert round_trip["verifier_run_id"] == _verifier_run_id(candidate_run_id)
    assert round_trip["remote_raw_manifest_sha256"] == round_trip[
        "local_raw_manifest_sha256"
    ]
    assert round_trip["remote_raw_manifest_size_bytes"] == round_trip[
        "local_raw_manifest_size_bytes"
    ]
    assert round_trip["remote_canonical_manifest_sha256"] == round_trip[
        "local_canonical_manifest_sha256"
    ]
    assert round_trip["remote_verification_completed"] is True
    assert round_trip["local_verification_completed"] is True
    round_trip_gate = "modal_artifact_round_trip_validated"
    round_trip_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[
        round_trip_gate
    ]["receipt_path"]
    assert "remote_and_local_verified" in validate_modal_readiness_receipt(
        round_trip_gate, round_trip_path, root=tmp_path
    )


@pytest.mark.parametrize("exception_type", ("TimeoutError", "OSError"))
def test_cuda_recorder_rejects_unproven_network_denial_receipt(
    tmp_path: Path,
    exception_type: str,
) -> None:
    run_id = "cuda-network-denial-forgery"
    run = _downloaded_cuda_run(tmp_path, run_id)
    probe_path = run / "provider_free_network_denial_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["exception_type"] = exception_type
    _write_json(probe_path, probe)
    _rewrite_artifact_manifest(run, run_id)

    with pytest.raises(
        ValueError,
        match="network denial probe exception classification is unsafe",
    ):
        record_cuda_environment(run_id=run_id, root=tmp_path)


def test_cuda_receipt_rejects_integer_boolean_spoofing(tmp_path: Path) -> None:
    run_id = "cuda-check-1"
    _downloaded_cuda_run(tmp_path, run_id)
    record_cuda_environment(run_id=run_id, root=tmp_path)
    gate = "modal_cuda_environment_validated"
    receipt_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[gate][
        "receipt_path"
    ]
    payload = json.loads(receipt_path.read_text())
    payload["cuda_available"] = 1
    _write_json(receipt_path, payload)

    with pytest.raises(ValueError, match="must be boolean"):
        validate_modal_readiness_receipt(gate, receipt_path, root=tmp_path)


def test_cuda_receipt_rejects_remote_action_hash_rewrite(tmp_path: Path) -> None:
    run_id = "cuda-check-1"
    _downloaded_cuda_run(tmp_path, run_id)
    record_cuda_environment(run_id=run_id, root=tmp_path)
    gate = "modal_cuda_environment_validated"
    receipt_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[gate][
        "receipt_path"
    ]
    payload = json.loads(receipt_path.read_text())
    payload["remote_action_result_sha256"] = "0" * 64
    _write_json(receipt_path, payload)

    with pytest.raises(ValueError, match="differs from artifacts"):
        validate_modal_readiness_receipt(gate, receipt_path, root=tmp_path)


@pytest.mark.parametrize(
    ("action", "message"),
    (
        (
            {"mode": "cuda_environment"},
            "invalid exact schema",
        ),
        (
            {
                "success": False,
                "mode": "cuda_environment",
                "observed_gpu": "NVIDIA T4",
            },
            "must be exactly True",
        ),
        (
            {
                "success": True,
                "mode": "wrong_mode",
                "observed_gpu": "NVIDIA T4",
            },
            "wrong mode",
        ),
        (
            {
                "success": True,
                "mode": "cuda_environment",
                "observed_gpu": "NVIDIA A100",
            },
            "GPU differs from the report",
        ),
    ),
)
def test_cuda_receipt_requires_exact_bound_remote_action_result(
    tmp_path: Path,
    action: dict[str, object],
    message: str,
) -> None:
    run_id = "cuda-check-1"
    run = _downloaded_cuda_run(tmp_path, run_id)
    _write_json(run / "remote_action_result.json", action)
    _rewrite_artifact_manifest(run, run_id)

    with pytest.raises(ValueError, match=message):
        record_cuda_environment(run_id=run_id, root=tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("python", 312, "must be non-empty text"),
        ("python", "3.11.9", "differs from the frozen version"),
        ("platform", "other-platform", "differs from the fingerprint"),
        ("torch", "0.0.0", "differs from the fingerprint"),
        (
            "cuda_compute_capability",
            [7, True],
            "must be two exact integers",
        ),
        (
            "cuda_compute_capability",
            [8, 0],
            "differs from the fingerprint",
        ),
        ("cuda_runtime", "0.0", "differs from the fingerprint"),
        ("cuda_driver", "0.0", "differs from the fingerprint"),
        ("cuda_total_memory_bytes", True, "must be an integer"),
        ("cuda_total_memory_bytes", 0, "must be an integer"),
    ),
)
def test_cuda_receipt_types_and_links_report_scalars_to_fingerprint(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    run_id = "cuda-check-1"
    run = _downloaded_cuda_run(tmp_path, run_id)
    cuda_path = run / "cuda_environment.json"
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    cuda[field] = value
    _write_json(cuda_path, cuda)
    _rewrite_artifact_manifest(run, run_id)

    with pytest.raises(ValueError, match=message):
        record_cuda_environment(run_id=run_id, root=tmp_path)


def test_round_trip_receipt_rejects_non_candidate_execution_context(
    tmp_path: Path,
) -> None:
    run_id = "cuda-check-1"
    _downloaded_cuda_run(tmp_path, run_id)
    _write_remote_verification(tmp_path, run_id)

    with pytest.raises(ValueError, match="must bind a candidate_smoke run"):
        record_artifact_round_trip(
            source_run_id=run_id,
            verifier_run_id=_verifier_run_id(run_id),
            root=tmp_path,
        )


def test_cuda_environment_receipt_requires_git_version(tmp_path: Path) -> None:
    run_id = "cuda-check-1"
    run = _downloaded_cuda_run(tmp_path, run_id)
    cuda_path = run / "cuda_environment.json"
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    del cuda["git_version"]
    _write_json(cuda_path, cuda)
    _rewrite_artifact_manifest(run, run_id)

    with pytest.raises(ValueError, match="invalid exact schema"):
        record_cuda_environment(run_id=run_id, root=tmp_path)


@pytest.mark.parametrize(
    "git_version",
    (
        None,
        2_030_002,
        "",
        "2.30.2",
        "git version ",
        "git version 2",
        "git version 2.30.2\nforged",
    ),
)
def test_cuda_environment_receipt_validates_git_version_type_and_format(
    tmp_path: Path,
    git_version: object,
) -> None:
    run_id = "cuda-check-1"
    run = _downloaded_cuda_run(tmp_path, run_id)
    cuda_path = run / "cuda_environment.json"
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    cuda["git_version"] = git_version
    _write_json(cuda_path, cuda)
    _rewrite_artifact_manifest(run, run_id)

    with pytest.raises(ValueError, match="must be canonical"):
        record_cuda_environment(run_id=run_id, root=tmp_path)


def test_cuda_receipt_rejects_git_version_tampering(tmp_path: Path) -> None:
    run_id = "cuda-check-1"
    run = _downloaded_cuda_run(tmp_path, run_id)
    record_cuda_environment(run_id=run_id, root=tmp_path)
    gate = "modal_cuda_environment_validated"
    receipt_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[gate][
        "receipt_path"
    ]
    cuda_path = run / "cuda_environment.json"
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    cuda["git_version"] = "git version 2.47.3"
    _write_json(cuda_path, cuda)

    with pytest.raises(
        ArtifactIntegrityError,
        match="artifact digest mismatch: cuda_environment.json",
    ):
        validate_modal_readiness_receipt(gate, receipt_path, root=tmp_path)


def _provider_spend_uncertainty_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, tuple[object, ...]],
    list[dict[str, object]],
    dict[str, object],
    Path,
]:
    plan_logical = "outputs/readiness/provider-plan.json"
    plan_path = tmp_path / plan_logical
    plan = {
        "harnesses": [
            {
                "harness": harness,
                "maximum_attempts": 1,
                "generation_settings_sha256": "e" * 64,
                "first_opportunity": {
                    "conservative_input_token_ceiling": 100,
                },
                "request_settings": {"max_completion_tokens": 200},
            }
            for harness in modal_readiness.CANARY_ORDER
        ]
    }
    _write_json(plan_path, plan)
    price_logical = "outputs/readiness/provider-price.json"
    price_path = tmp_path / price_logical
    price = {
        "uncached_input_usd_per_million_tokens": "1",
        "output_usd_per_million_tokens": "2",
        "per_request_fee_usd": "0.1",
    }
    _write_json(price_path, price)
    price_sha256 = hashlib.sha256(price_path.read_bytes()).hexdigest()

    contexts: dict[str, ExecutionContextV1] = {}
    outcomes: list[dict[str, object]] = []
    attempts: list[dict[str, object]] = []
    primary: dict[str, tuple[object, ...]] = {}
    for index, harness in enumerate(modal_readiness.CANARY_ORDER, start=1):
        run_id = f"accepted-provider-{index}"
        attempt_id = f"{index:032x}"
        context = ExecutionContextV1(
            execution_backend="modal",
            run_id=run_id,
            app_name=APP_NAME,
            function_name=f"canary_{harness}",
            modal_app_id=f"ap-provider-{index}",
            modal_function_id=f"fu-provider-{index}",
            modal_call_id=f"fc-provider-{index}",
            modal_image_id="im-provider",
            image_source_sha256="d" * 64,
            artifact_uri=volume_artifact_uri(run_id),
        )
        contexts[run_id] = context
        label = f"canary_{harness}"
        primary[label] = ({}, tmp_path, object(), context)
        ledger_logical = (
            f"outputs/development/modal_downloads/{run_id}/controller/"
            "provider_attempts.jsonl"
        )
        ledger = tmp_path / ledger_logical
        ledger.parent.mkdir(parents=True, exist_ok=True)
        provider_record = {
            "schema_name": "ProviderAttemptRecord",
            "schema_version": "1.0",
            "harness": harness,
            "action": "one_opportunity_engineering_canary",
            "controller_run_id": f"controller-{run_id}",
            "execution_backend": "modal",
            "action_run_id": run_id,
            "modal_call_id": context.modal_call_id,
            "attempt_ordinal": 1,
            "started_at_utc": "2025-01-01T00:00:00Z",
            "ended_at_utc": "2025-01-01T00:00:01Z",
            "status": "success",
            "api_endpoint": modal_readiness.OFFICIAL_OPENAI_API_BASE,
            "model": modal_readiness.TARGET_MODEL,
            "generation_settings_sha256": "e" * 64,
            "provider_response_id": f"response-{index}",
            "provider_request_id": f"request-{index}",
            "usage_known": True,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "error_class": None,
        }
        ledger.write_text(
            json.dumps(provider_record, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outcomes.append(
            {
                "harness": harness,
                "concrete_run_id": run_id,
                "launcher_attempt_id": attempt_id,
                "outcome": "accepted",
                "provider_attempt_ledger_path": ledger_logical,
                "provider_attempt_ledger_sha256": hashlib.sha256(
                    ledger.read_bytes()
                ).hexdigest(),
                "provider_start_uncertain_evidence_path": None,
                "provider_start_uncertain_evidence_sha256": None,
            }
        )
        attempts.append(
            {
                "attempt_id": attempt_id,
                "started_at_utc": "2025-01-01T00:00:00Z",
                "finished_at_utc": "2025-01-01T00:00:02Z",
                "action": "canary",
                "harness": harness,
                "modal_cli_process_started": True,
                "provider_cost_approved": True,
                "provider_price_basis_path": price_logical,
                "provider_price_basis_sha256": price_sha256,
                "provider_approval_plan_path": plan_logical,
                "approval_plan_sha256": "a" * 64,
                "approved_image_source_sha256": "d" * 64,
                "provider_cost_cap_usd": "1",
                "predecessor_receipts": [
                    {
                        "gate": "candidate_resume_preflight_validated",
                        "path": "outputs/readiness/preflight.json",
                        "sha256": "f" * 64,
                    }
                ],
            }
        )

    uncertain_run_id = "provider-start-uncertain"
    uncertain_attempt_id = f"{len(attempts) + 1:032x}"
    uncertain_context = ExecutionContextV1(
        execution_backend="modal",
        run_id=uncertain_run_id,
        app_name=APP_NAME,
        function_name="canary_greedy_autoresearch",
        modal_app_id="ap-provider-uncertain",
        modal_function_id="fu-provider-uncertain",
        modal_call_id="fc-provider-start-uncertain",
        modal_image_id="im-provider",
        image_source_sha256="d" * 64,
        artifact_uri=volume_artifact_uri(uncertain_run_id),
    )
    contexts[uncertain_run_id] = uncertain_context
    uncertain_ledger_logical = (
        f"outputs/development/modal_downloads/{uncertain_run_id}/controller/"
        "provider_attempts.jsonl"
    )
    uncertain_ledger = tmp_path / uncertain_ledger_logical
    uncertain_ledger.parent.mkdir(parents=True, exist_ok=True)
    uncertain_ledger.write_bytes(b"")
    uncertain_evidence_logical = (
        f"outputs/development/modal_downloads/{uncertain_run_id}/controller/"
        "provider_request_start_uncertain.json"
    )
    uncertain_evidence = tmp_path / uncertain_evidence_logical
    _write_json(
        uncertain_evidence,
        _provider_start_uncertain_payload(
            run_id=uncertain_run_id,
            ledger_state="present",
        ),
    )
    uncertain_outcome = {
        "harness": "greedy_autoresearch",
        "concrete_run_id": uncertain_run_id,
        "launcher_attempt_id": uncertain_attempt_id,
        "outcome": "provider_request_start_uncertain",
        "provider_attempt_ledger_path": uncertain_ledger_logical,
        "provider_attempt_ledger_sha256": hashlib.sha256(
            uncertain_ledger.read_bytes()
        ).hexdigest(),
        "provider_start_uncertain_evidence_path": uncertain_evidence_logical,
        "provider_start_uncertain_evidence_sha256": hashlib.sha256(
            uncertain_evidence.read_bytes()
        ).hexdigest(),
    }
    outcomes.append(uncertain_outcome)
    attempts.append(
        {
            "attempt_id": uncertain_attempt_id,
            "started_at_utc": "2025-01-01T00:00:00Z",
            "finished_at_utc": "2025-01-01T00:00:02Z",
            "action": "canary",
            "harness": "greedy_autoresearch",
            "modal_cli_process_started": True,
            "provider_cost_approved": True,
            "provider_price_basis_path": price_logical,
            "provider_price_basis_sha256": price_sha256,
            "provider_approval_plan_path": plan_logical,
            "approval_plan_sha256": "a" * 64,
            "approved_image_source_sha256": "d" * 64,
            "provider_cost_cap_usd": "1",
            "predecessor_receipts": [
                {
                    "gate": "candidate_resume_preflight_validated",
                    "path": "outputs/readiness/preflight.json",
                    "sha256": "f" * 64,
                }
            ],
        }
    )
    roster = {
        "source_tree_sha256": "c" * 64,
        "image_source_sha256": "d" * 64,
        "cohort_id": "provider-spend-fixture",
        "provider_price_basis_path": price_logical,
        "provider_canary_outcomes": outcomes,
    }

    monkeypatch.setattr(
        modal_readiness,
        "_load_price_basis",
        lambda *_args, **_kwargs: (price, price_path, price_sha256),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_provider_approval_plan",
        lambda *_args, **_kwargs: (plan, plan_path),
    )

    def inspect_run(_root, run_id, _downloaded_path):
        return tmp_path, object(), {}, contexts[run_id]

    monkeypatch.setattr(modal_readiness, "_inspect_downloaded_run", inspect_run)
    estimate = modal_readiness._provider_spend_estimate(
        tmp_path,
        roster,
        primary,
        attempts,
    )
    return (
        estimate,
        roster,
        primary,
        attempts,
        uncertain_outcome,
        uncertain_ledger,
    )


def test_provider_start_uncertainty_reserves_one_full_approved_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimate, _roster, _primary, _attempts, uncertain, _ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )

    assert estimate["provider_terminal_attempt_record_count"] == 4
    assert estimate["provider_attempt_count_lower_bound"] == 4
    assert estimate["provider_attempt_count_upper_bound"] == 5
    assert estimate["provider_attempt_count"] == 5
    assert estimate["provider_request_start_uncertain_count"] == 1
    assert estimate["uncertain_request_start_reserve_usd"] == "0.1005"
    assert estimate["known_success_usage_estimate_usd"] == "0.40008"
    assert estimate["conservative_provider_spend_bound_usd"] == "0.50058"
    assert estimate["provider_request_start_uncertainties"] == [
        {
            "harness": "greedy_autoresearch",
            "run_id": "provider-start-uncertain",
            "launcher_attempt_id": uncertain["launcher_attempt_id"],
            "provider_attempt_count_lower_bound": 0,
            "provider_attempt_count_upper_bound": 1,
            "provider_request_started": "unknown",
            "provider_attempt_ledger_state": "present",
            "billing_treatment": "reserve_one_full_approved_request",
            "conservative_input_token_ceiling": 100,
            "requested_completion_token_ceiling": 200,
            "conservative_uncertain_request_reserve_usd": "0.1005",
            "evidence_path": uncertain[
                "provider_start_uncertain_evidence_path"
            ],
            "evidence_sha256": uncertain[
                "provider_start_uncertain_evidence_sha256"
            ],
        }
    ]
    assert modal_readiness._provider_request_state_evidence_paths(estimate) == (
        uncertain["provider_start_uncertain_evidence_path"],
    )
    uncertain_ledger = next(
        item
        for item in estimate["ledgers"]
        if item["run_id"] == "provider-start-uncertain"
    )
    assert uncertain_ledger["sha256"] == hashlib.sha256(b"").hexdigest()
    assert uncertain_ledger["request_state_evidence"] is not None
    bound = next(
        item
        for item in estimate["launcher_approval_bounds"]
        if item["launcher_attempt_id"] == uncertain["launcher_attempt_id"]
    )
    assert bound["uncertain_request_start_reserve_usd"] == "0.1005"
    assert bound["conservative_observed_bound_usd"] == "0.1005"


def test_provider_start_uncertainty_accepts_exactly_missing_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _estimate, roster, primary, attempts, uncertain, ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )
    evidence_path = tmp_path / uncertain[
        "provider_start_uncertain_evidence_path"
    ]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["provider_attempt_ledger_state"] = "missing"
    _write_json(evidence_path, evidence)
    uncertain["provider_start_uncertain_evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    ledger.unlink()
    uncertain["provider_attempt_ledger_sha256"] = None

    estimate = modal_readiness._provider_spend_estimate(
        tmp_path,
        roster,
        primary,
        attempts,
    )

    missing = next(
        item
        for item in estimate["ledgers"]
        if item["run_id"] == "provider-start-uncertain"
    )
    assert missing["sha256"] is None
    assert missing["provider_terminal_attempt_record_count"] == 0
    assert missing["provider_request_start_state"] == "unknown"
    assert estimate["provider_attempt_count_lower_bound"] == 4
    assert estimate["provider_attempt_count_upper_bound"] == 5


def test_provider_start_uncertainty_rejects_ledger_presence_misstatement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _estimate, roster, primary, attempts, uncertain, _ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )
    evidence_path = tmp_path / uncertain[
        "provider_start_uncertain_evidence_path"
    ]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["provider_attempt_ledger_state"] = "missing"
    _write_json(evidence_path, evidence)
    uncertain["provider_start_uncertain_evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="misstates ledger presence"):
        modal_readiness._provider_spend_estimate(
            tmp_path,
            roster,
            primary,
            attempts,
        )


def test_provider_start_uncertainty_rejects_nonempty_terminal_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _estimate, roster, primary, attempts, uncertain, ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )
    source_ledger = tmp_path / roster["provider_canary_outcomes"][0][
        "provider_attempt_ledger_path"
    ]
    record = json.loads(source_ledger.read_text(encoding="utf-8"))
    record.update(
        {
            "harness": "greedy_autoresearch",
            "controller_run_id": "controller-provider-start-uncertain",
            "action_run_id": "provider-start-uncertain",
            "modal_call_id": "fc-provider-start-uncertain",
            "provider_response_id": "response-uncertain",
            "provider_request_id": "request-uncertain",
        }
    )
    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    uncertain["provider_attempt_ledger_sha256"] = hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match="start-uncertain.*contains ledger records"):
        modal_readiness._provider_spend_estimate(
            tmp_path,
            roster,
            primary,
            attempts,
        )


def test_terminal_provider_error_remains_distinct_from_start_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _estimate, roster, primary, attempts, uncertain, ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )
    source_ledger = tmp_path / roster["provider_canary_outcomes"][0][
        "provider_attempt_ledger_path"
    ]
    record = json.loads(source_ledger.read_text(encoding="utf-8"))
    record.update(
        {
            "harness": "greedy_autoresearch",
            "controller_run_id": "controller-provider-terminal-error",
            "action_run_id": "provider-start-uncertain",
            "modal_call_id": "fc-provider-start-uncertain",
            "status": "error",
            "provider_response_id": None,
            "provider_request_id": "request-terminal-error",
            "usage_known": False,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "error_class": "TimeoutError",
        }
    )
    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    uncertain["outcome"] = "failed"
    uncertain["provider_attempt_ledger_sha256"] = hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()
    uncertain["provider_start_uncertain_evidence_path"] = None
    uncertain["provider_start_uncertain_evidence_sha256"] = None

    estimate = modal_readiness._provider_spend_estimate(
        tmp_path,
        roster,
        primary,
        attempts,
    )

    assert estimate["provider_attempt_count"] == 5
    assert estimate["failed_provider_attempt_count"] == 1
    assert estimate["provider_request_start_uncertain_count"] == 0
    assert estimate["failed_attempt_reserve_usd"] == "0.1005"
    assert estimate["uncertain_request_start_reserve_usd"] == "0"


def _replace_uncertain_ledger_with_known_success(
    tmp_path: Path,
    roster: dict[str, object],
    uncertain: dict[str, object],
    ledger: Path,
) -> dict[str, object]:
    source_ledger = tmp_path / roster["provider_canary_outcomes"][0][
        "provider_attempt_ledger_path"
    ]
    record = json.loads(source_ledger.read_text(encoding="utf-8"))
    record.update(
        {
            "harness": "greedy_autoresearch",
            "controller_run_id": "controller-provider-downstream-failure",
            "action_run_id": "provider-start-uncertain",
            "modal_call_id": "fc-provider-start-uncertain",
            "provider_response_id": "response-downstream-failure",
            "provider_request_id": "request-downstream-failure",
        }
    )
    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    uncertain["outcome"] = "failed"
    uncertain["provider_attempt_ledger_sha256"] = hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()
    uncertain["provider_start_uncertain_evidence_path"] = None
    uncertain["provider_start_uncertain_evidence_sha256"] = None
    return record


def test_provider_success_followed_by_downstream_failure_is_retained_and_charged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _estimate, roster, primary, attempts, uncertain, ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )
    _replace_uncertain_ledger_with_known_success(
        tmp_path,
        roster,
        uncertain,
        ledger,
    )

    estimate = modal_readiness._provider_spend_estimate(
        tmp_path,
        roster,
        primary,
        attempts,
    )

    failed_run = next(
        item
        for item in estimate["ledgers"]
        if item["run_id"] == "provider-start-uncertain"
    )
    assert failed_run["outcome"] == "failed"
    assert failed_run["successful_provider_attempt_count"] == 1
    assert failed_run["failed_provider_attempt_count"] == 0
    assert estimate["successful_provider_attempt_count"] == 5
    assert estimate["accepted_successful_provider_attempt_count"] == 4
    assert estimate["failed_provider_attempt_count"] == 0
    assert estimate["provider_request_start_uncertain_count"] == 0
    assert estimate["known_success_usage_estimate_usd"] == "0.50010"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("attempt_ordinal", 2, "not execution-bound"),
        ("generation_settings_sha256", "f" * 64, "not execution-bound"),
        ("input_tokens", 101, "approved token ceilings"),
        ("output_tokens", 201, "approved token ceilings"),
    ),
)
def test_provider_ledger_enforces_one_attempt_settings_and_token_ceilings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    _estimate, roster, primary, attempts, uncertain, ledger = (
        _provider_spend_uncertainty_fixture(tmp_path, monkeypatch)
    )
    record = _replace_uncertain_ledger_with_known_success(
        tmp_path,
        roster,
        uncertain,
        ledger,
    )
    record[field] = value
    if field == "input_tokens":
        record["total_tokens"] = value + record["output_tokens"]
    elif field == "output_tokens":
        record["total_tokens"] = record["input_tokens"] + value
    ledger.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    uncertain["provider_attempt_ledger_sha256"] = hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()

    with pytest.raises(ValueError, match=message):
        modal_readiness._provider_spend_estimate(
            tmp_path,
            roster,
            primary,
            attempts,
        )


def test_cleanup_recorder_binds_all_snapshots_and_zero_resource_counts(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster_logical = roster_path.relative_to(tmp_path).as_posix()

    receipt = record_resource_cleanup(
        cohort_roster_path=roster_logical,
        root=tmp_path,
    )

    assert receipt["active_app_count"] == 0
    assert receipt["volume_present"] is True
    assert receipt["cohort_billing_total_usd"] == "0.16"
    assert receipt["superseded_usage_usd"] == "0.00643852"
    assert receipt["migration_total_usd"] == "0.16643852"
    assert receipt["schema_version"] == "4.0"
    assert len(receipt["accepted_primary_executions"]) == 8
    assert len(receipt["artifact_verifier_executions"]) == 8
    assert len(receipt["action_attempts"]) == 16
    assert receipt["direct_detached_call_inventory"] == (
        "unavailable_in_modal_cli_1_5_3"
    )
    assert receipt["detached_calls_prohibited"] is True
    assert receipt["provider_spend_estimate"]["provider_attempt_count"] == 4
    assert (
        receipt["provider_spend_estimate"][
            "provider_terminal_attempt_record_count"
        ]
        == 4
    )
    assert receipt["provider_spend_estimate"]["provider_attempt_count_lower_bound"] == 4
    assert receipt["provider_spend_estimate"]["provider_attempt_count_upper_bound"] == 4
    assert receipt["provider_spend_estimate"]["failed_provider_attempt_count"] == 0
    assert (
        receipt["provider_spend_estimate"][
            "provider_request_start_uncertain_count"
        ]
        == 0
    )
    assert (
        receipt["provider_spend_estimate"][
            "uncertain_request_start_reserve_usd"
        ]
        == "0"
    )
    assert receipt["migration_provider_spend_estimate"] == {
        "accounting_label": (
            "final_plus_all_prior_conservative_provider_bounds_not_billed_cost"
        ),
        "final_cohort_conservative_provider_spend_bound_usd": receipt[
            "provider_spend_estimate"
        ]["conservative_provider_spend_bound_usd"],
        "prior_quarantined_conservative_provider_spend_bound_usd": "0",
        "migration_conservative_provider_spend_bound_usd": receipt[
            "provider_spend_estimate"
        ]["conservative_provider_spend_bound_usd"],
        "prior_cohorts": [],
    }
    gate = "modal_resource_cleanup_validated"
    receipt_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[gate][
        "receipt_path"
    ]
    assert "resources=0" in validate_modal_readiness_receipt(
        gate, receipt_path, root=tmp_path
    )

    (snapshot_root / "billing_report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="snapshot capture billing_report bytes changed",
    ):
        validate_modal_readiness_receipt(gate, receipt_path, root=tmp_path)


def test_cleanup_accepts_finalized_ordinary_failure_source_recovery(
    tmp_path: Path,
) -> None:
    roster_path, identities = _ordinary_failure_recovery_fixture(tmp_path)

    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_path.relative_to(tmp_path).as_posix(),
        recorded_at_utc="2025-01-02T00:00:00Z",
    )

    assert claims["failed_run_ids"] == [identities["failed_run_id"]]
    assert claims["quarantined_run_ids"] == [identities["failed_run_id"]]
    assert claims["recovery_run_ids"] == ["accepted-offline"]
    assert claims["cohort_billing_total_usd"] == "0.18"
    ordinary = claims["evidence_backed_failed_ordinary_executions"]
    assert ordinary == [
        {
            "failed_attempt_id": identities["failed_attempt_id"],
                "failed_action": "offline-smoke",
            "failed_run_id": identities["failed_run_id"],
            "failed_execution_context": ordinary[0]["failed_execution_context"],
            "source_evidence_verifier_attempt_id": identities[
                "verifier_attempt_id"
            ],
            "source_evidence_verifier_run_id": identities["verifier_run_id"],
            "replacement_attempt_id": identities["accepted_attempt_id"],
                "replacement_run_id": "accepted-offline",
        }
    ]
    assert ordinary[0]["failed_execution_context"]["modal_app_id"] == identities[
        "failed_app_id"
    ]
    additional_paths = modal_readiness._additional_verifier_required_paths(
        claims["additional_artifact_verifier_executions"]
    )
    assert len(additional_paths) == 4
    assert all(
        path is not None and (tmp_path / path).is_file()
        for path in additional_paths
    )
    ordinary_paths = modal_readiness._ordinary_failure_required_paths(
        tmp_path,
        ordinary,
    )
    assert len(ordinary_paths) == 5
    assert all((tmp_path / path).is_file() for path in ordinary_paths)


def test_cleanup_rejects_partial_ordinary_failure_source_capture(
    tmp_path: Path,
) -> None:
    roster_path, identities = _ordinary_failure_recovery_fixture(tmp_path)
    manifest_path = (
        tmp_path
        / "outputs/development/modal_downloads"
        / identities["failed_run_id"]
        / "artifact_manifest.json"
    )
    manifest_path.unlink()

    with pytest.raises(
        ValueError, match="verifier source lacks a bound final execution"
    ):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_swapped_ordinary_failure_app_attribution(
    tmp_path: Path,
) -> None:
    roster_path, identities = _ordinary_failure_recovery_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attribution = next(
        item
        for item in roster["billing_attributions"]
        if item["attempt_id"] == identities["failed_attempt_id"]
    )
    swapped_app_id = "ap-swapped-ordinary-cuda"
    attribution["object_ids"] = [swapped_app_id]
    _write_json(roster_path, roster)
    snapshot_root = (tmp_path / roster["snapshot_capture_manifest_path"]).parent
    app_path = snapshot_root / "app_list.json"
    apps = json.loads(app_path.read_text(encoding="utf-8"))
    next(
        item
        for item in apps
        if item["app_id"] == identities["failed_app_id"]
    )["app_id"] = swapped_app_id
    _write_json(app_path, apps)
    billing_path = snapshot_root / "billing_report.json"
    billing = json.loads(billing_path.read_text(encoding="utf-8"))
    next(
        item
        for item in billing
        if item["object_id"] == identities["failed_app_id"]
    )["object_id"] = swapped_app_id
    _write_json(billing_path, billing)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(
        ValueError, match="snapshot App IDs differ from billing attribution"
    ):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_ordinary_failure_without_exact_replacement_link(
    tmp_path: Path,
) -> None:
    roster_path, identities = _ordinary_failure_recovery_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster["recovery_links"] = []
    roster["declared_recovery_run_ids"] = []
    classification = next(
        item
        for item in roster["attempt_classifications"]
        if item["attempt_id"] == identities["accepted_attempt_id"]
    )
    classification["roles"].remove("recovery")
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="lacks its exact accepted replacement"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_unquarantined_ordinary_failure_recovery(
    tmp_path: Path,
) -> None:
    roster_path, identities = _ordinary_failure_recovery_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    classification = next(
        item
        for item in roster["attempt_classifications"]
        if item["attempt_id"] == identities["failed_attempt_id"]
    )
    classification["roles"].remove("quarantined")
    roster["declared_quarantined_run_ids"] = []
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="must be quarantined"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


@pytest.mark.parametrize("case", ("same_attempt", "recovery_before_failure"))
def test_cleanup_rejects_nondistinct_or_time_reversed_recovery_links(
    tmp_path: Path,
    case: str,
) -> None:
    roster_path, identities = _ordinary_failure_recovery_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if case == "same_attempt":
        roster["recovery_links"][0]["recovery_attempt_id"] = identities[
            "failed_attempt_id"
        ]
    else:
        failed_path = _attempt_root(tmp_path) / (
            f"{identities['failed_attempt_id']}.json"
        )
        failed = json.loads(failed_path.read_text(encoding="utf-8"))
        failed["finished_at_utc"] = "2025-01-01T00:06:00Z"
        _write_json(failed_path, failed)
        verifier_attempt_id = identities["verifier_attempt_id"]
        for suffix in (".intent.json", ".json"):
            verifier_path = _attempt_root(tmp_path) / (
                f"{verifier_attempt_id}{suffix}"
            )
            verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
            source_binding = next(
                item
                for item in verifier["predecessor_receipts"]
                if item["gate"] == "source_action_attempt_terminal"
            )
            source_binding["sha256"] = hashlib.sha256(
                failed_path.read_bytes()
            ).hexdigest()
            _write_json(verifier_path, verifier)
        marker_path = (
            tmp_path
            / modal_readiness.modal_local_process_start_receipt_path(
                verifier_attempt_id
            )
        )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["intent_sha256"] = hashlib.sha256(
            (
                _attempt_root(tmp_path)
                / f"{verifier_attempt_id}.intent.json"
            ).read_bytes()
        ).hexdigest()
        _write_json(marker_path, marker)
        marker_path.chmod(0o600)
        verifier_terminal_path = (
            _attempt_root(tmp_path) / f"{verifier_attempt_id}.json"
        )
        verifier_terminal = json.loads(
            verifier_terminal_path.read_text(encoding="utf-8")
        )
        verifier_terminal["local_process_start_receipt_sha256"] = hashlib.sha256(
            marker_path.read_bytes()
        ).hexdigest()
        _write_json(verifier_terminal_path, verifier_terminal)
    _write_json(roster_path, roster)
    if case == "recovery_before_failure":
        _refresh_migration_lineage(roster_path)

    with pytest.raises(
        ValueError,
        match="recovery link (?:is not bound|does not connect)",
    ):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_preserves_failed_verifier_capture_and_successful_retry(
    tmp_path: Path,
) -> None:
    roster_path, identities = _failed_verifier_retry_fixture(tmp_path)

    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_path.relative_to(tmp_path).as_posix(),
        recorded_at_utc="2025-01-02T00:00:00Z",
    )

    assert claims["failed_run_ids"] == [identities["failed_verifier_run_id"]]
    assert claims["quarantined_run_ids"] == [
        identities["failed_verifier_run_id"]
    ]
    assert claims["cohort_billing_total_usd"] == "0.17"
    evidence = claims["additional_artifact_verifier_executions"]
    assert len(evidence) == 1
    failed = evidence[0]
    assert failed["remote_verifier_outcome"] == "failure"
    assert failed["remote_verification_path"] is None
    assert failed["failure_receipt_path"] is not None
    assert failed["recovery_verifier_attempt_id"] == identities[
        "recovery_attempt_id"
    ]
    assert failed["recovery_verifier_run_id"] == identities[
        "recovery_verifier_run_id"
    ]
    assert failed["recovery_remote_verification_path"] == identities[
        "recovery_remote_path"
    ]
    assert (tmp_path / failed["recovery_remote_verification_path"]).is_file()
    paths = modal_readiness._additional_verifier_required_paths(evidence)
    assert len(paths) == 4
    assert all(path is not None and (tmp_path / path).is_file() for path in paths)


def test_cleanup_rejects_failed_verifier_retry_for_another_source(
    tmp_path: Path,
) -> None:
    roster_path, identities = _failed_verifier_retry_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster["additional_artifact_verifiers"][0][
        "recovery_verifier_attempt_id"
    ] = roster["artifact_verifiers"]["offline_smoke"]["attempt_id"]
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="source-bound successful verifier retry"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_verifier_retry_that_predates_failed_verifier(
    tmp_path: Path,
) -> None:
    roster_path, identities = _failed_verifier_retry_fixture(tmp_path)
    failed_path = _attempt_root(tmp_path) / (
        f"{identities['failed_attempt_id']}.json"
    )
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    failed["finished_at_utc"] = "2025-01-01T00:06:00Z"
    _write_json(failed_path, failed)
    _refresh_migration_lineage(roster_path)

    with pytest.raises(ValueError, match="later successful verifier retry"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_incomplete_failed_verifier_capture(
    tmp_path: Path,
) -> None:
    roster_path, identities = _failed_verifier_retry_fixture(tmp_path)
    capture = tmp_path / (
        modal_readiness.modal_artifact_verifier_capture_directory_path(
            _fixture_identity(tmp_path),
            "cleanup-check-1",
            identities["failed_verifier_run_id"],
            identities["failed_attempt_id"],
        )
    )
    (capture / "image_source_manifest.json").unlink()

    with pytest.raises(ValueError, match="exact file roster"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cohort_rejects_missing_provider_ledger_without_inferring_zero(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    ledger = tmp_path / roster["provider_canary_outcomes"][0][
        "provider_attempt_ledger_path"
    ]
    ledger.unlink()
    _refresh_source_manifest_and_verifier(
        roster_path,
        roster["provider_canary_outcomes"][0]["concrete_run_id"],
    )

    with pytest.raises(ValueError, match="cannot infer zero provider starts"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


def test_cohort_rejects_unclassified_provider_start_uncertainty(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    outcome = roster["provider_canary_outcomes"][0]
    run_id = outcome["concrete_run_id"]
    run_root = tmp_path / f"outputs/development/modal_downloads/{run_id}"
    context = ExecutionContextV1.from_dict(
        json.loads((run_root / "execution_context.json").read_text(encoding="utf-8"))
    )
    _write_json(
        run_root / "controller/provider_request_start_uncertain.json",
        _provider_start_uncertain_payload(
            harness=outcome["harness"],
            run_id=run_id,
            modal_call_id=context.modal_call_id or "",
        ),
    )
    _refresh_source_manifest_and_verifier(roster_path, run_id)
    _refresh_migration_lineage(roster_path)

    with pytest.raises(ValueError, match="differs from its cohort outcome"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


@pytest.mark.parametrize("case", ("missing", "contradictory_not_started"))
def test_uncertain_provider_outcome_requires_exclusive_bound_evidence(
    tmp_path: Path,
    case: str,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    outcome = roster["provider_canary_outcomes"][0]
    run_id = outcome["concrete_run_id"]
    run_root = tmp_path / f"outputs/development/modal_downloads/{run_id}"
    outcome["outcome"] = "provider_request_start_uncertain"
    if case == "missing":
        expected = "path drifted"
    else:
        context = ExecutionContextV1.from_dict(
            json.loads(
                (run_root / "execution_context.json").read_text(encoding="utf-8")
            )
        )
        uncertain_path = (
            run_root / "controller/provider_request_start_uncertain.json"
        )
        _write_json(
            uncertain_path,
            _provider_start_uncertain_payload(
                harness=outcome["harness"],
                run_id=run_id,
                modal_call_id=context.modal_call_id or "",
            ),
        )
        outcome["provider_start_uncertain_evidence_path"] = (
            uncertain_path.relative_to(tmp_path).as_posix()
        )
        outcome["provider_start_uncertain_evidence_sha256"] = hashlib.sha256(
            uncertain_path.read_bytes()
        ).hexdigest()
        _write_json(
            run_root / "controller/provider_request_not_started.json",
            {"contradictory": True},
        )
        expected = "legacy provider zero-attempt evidence is forbidden"
    _write_json(roster_path, roster)
    if case == "contradictory_not_started":
        _refresh_source_manifest_and_verifier(roster_path, run_id)
        _refresh_migration_lineage(roster_path)

    with pytest.raises(ValueError, match=expected):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


def test_final_cohort_rejects_legacy_provider_not_started_outcome(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster["provider_canary_outcomes"][0]["outcome"] = (
        "provider_request_not_started"
    )
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="outcome disposition is unsupported"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


def test_final_cohort_rejects_legacy_zero_attempt_schema_fields(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    outcome = roster["provider_canary_outcomes"][0]
    outcome["provider_zero_attempt_evidence_path"] = None
    outcome["provider_zero_attempt_evidence_sha256"] = None
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="invalid exact schema"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


def test_cleanup_receipt_cannot_omit_uncertainty_accounting_fields(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    record_resource_cleanup(
        cohort_roster_path=roster_path.relative_to(tmp_path).as_posix(),
        root=tmp_path,
    )
    gate = "modal_resource_cleanup_validated"
    receipt_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[gate][
        "receipt_path"
    ]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["provider_spend_estimate"].pop(
        "uncertain_request_start_reserve_usd"
    )
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="provider_spend_estimate differs"):
        validate_modal_readiness_receipt(gate, receipt_path, root=tmp_path)


def test_cleanup_recorder_rejects_boolean_resource_count(tmp_path: Path) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    app_rows = json.loads((snapshot_root / "app_list.json").read_text())
    app_rows[0]["state"] = "ephemeral"
    app_rows[0]["tasks"] = "1"
    app_rows[0]["stopped_at"] = None
    _write_json(snapshot_root / "app_list.json", app_rows)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="did not stop cleanly"):
        record_resource_cleanup(
            cohort_roster_path=roster_path.relative_to(tmp_path).as_posix(),
            root=tmp_path,
        )


def test_cleanup_validator_derives_active_counts_even_if_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster_logical = roster_path.relative_to(tmp_path).as_posix()
    record_resource_cleanup(cohort_roster_path=roster_logical, root=tmp_path)
    gate = "modal_resource_cleanup_validated"
    receipt_path = tmp_path / MODAL_READINESS_RECEIPT_CONTRACTS[gate][
        "receipt_path"
    ]

    app_rows = json.loads((snapshot_root / "app_list.json").read_text())
    app_rows[0]["state"] = "ephemeral"
    app_rows[0]["tasks"] = "1"
    app_rows[0]["stopped_at"] = None
    _write_json(snapshot_root / "app_list.json", app_rows)
    _refresh_snapshot_capture_manifest(snapshot_root)
    receipt = json.loads(receipt_path.read_text())
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / roster["snapshot_capture_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    receipt["cohort_roster_sha256"] = hashlib.sha256(
        roster_path.read_bytes()
    ).hexdigest()
    receipt["snapshot_capture_manifest_sha256"] = roster[
        "snapshot_capture_manifest_sha256"
    ]
    receipt["snapshots"] = manifest["snapshots"]
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="did not stop cleanly"):
        validate_modal_readiness_receipt(gate, receipt_path, root=tmp_path)


def test_cleanup_parser_rejects_unknown_modal_cli_schema(tmp_path: Path) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    rows = json.loads((snapshot_root / "container_list.json").read_text())
    rows.append({"unexpected": "shape"})
    _write_json(snapshot_root / "container_list.json", rows)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="Modal 1.5.3 JSON schema"):
        record_resource_cleanup(
            cohort_roster_path=roster_path.relative_to(tmp_path).as_posix(),
            root=tmp_path,
        )


def test_cleanup_rejects_local_policy_source_not_used_by_bound_run(
    tmp_path: Path,
) -> None:
    roster_path, _ = _aggregate_cleanup_fixture(tmp_path)
    (tmp_path / "modal_app.py").write_text(
        "def invoke_synchronously():\n    return 'newer-source'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from the bound image source"):
        record_resource_cleanup(
            cohort_roster_path=roster_path.relative_to(tmp_path).as_posix(),
            root=tmp_path,
        )


def test_cleanup_rejects_billing_interval_outside_bound_app_lifetime(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    billing = json.loads((snapshot_root / "billing_report.json").read_text())
    billing[0]["interval_start"] = "2025-01-01T02:00:00+00:00"
    _write_json(snapshot_root / "billing_report.json", billing)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="does not overlap its App lifecycle"):
        record_resource_cleanup(
            cohort_roster_path=roster_path.relative_to(tmp_path).as_posix(),
            root=tmp_path,
        )


def test_cleanup_accepts_equal_attributed_app_cap_and_records_overage(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster_logical = roster_path.relative_to(tmp_path).as_posix()
    billing_path = snapshot_root / "billing_report.json"
    billing = json.loads(billing_path.read_text(encoding="utf-8"))
    billing[0]["cost"] = "0.25"
    _write_json(billing_path, billing)
    _refresh_snapshot_capture_manifest(snapshot_root)

    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_logical,
        recorded_at_utc="2025-01-02T00:00:00Z",
    )
    equal = next(
        item
        for item in claims["billing_attributions"]
        if item["billing_total_usd"] == "0.25"
    )
    assert equal["approved_modal_cost_cap_usd"] == "0.25"
    assert equal["attributed_app_billing_within_approved_cap"] is True

    billing[0]["cost"] = "0.25000001"
    _write_json(billing_path, billing)
    _refresh_snapshot_capture_manifest(snapshot_root)
    overage_claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_logical,
        recorded_at_utc="2025-01-02T00:00:00Z",
    )
    overage = next(
        item
        for item in overage_claims["billing_attributions"]
        if item["billing_total_usd"] == "0.25000001"
    )
    assert overage["attributed_app_billing_within_approved_cap"] is False
    exposure = overage_claims["modal_compute_exposure"]["final_cohort"]
    assert overage["attempt_id"] in exposure[
        "local_authorization_cap_breach_attempt_ids"
    ]
    assert exposure["local_authorization_is_platform_hard_bound"] is False


def test_modal_compute_exposure_separates_measurement_reserve_and_cap_breach(
) -> None:
    attempt_ids = [character * 32 for character in "abcde"]
    attempts = [
        {
            "attempt_id": attempt_ids[0],
            "modal_cli_process_started": False,
            "modal_cost_cap_usd": None,
            "status": "preflight_failed",
        },
        {
            "attempt_id": attempt_ids[1],
            "modal_cli_process_started": True,
            "modal_cost_cap_usd": "0.25",
            "status": "succeeded",
        },
        {
            "attempt_id": attempt_ids[2],
            "modal_cli_process_started": True,
            "modal_cost_cap_usd": "0.25",
            "status": "succeeded",
        },
        {
            "attempt_id": attempt_ids[3],
            "modal_cli_process_started": True,
            "modal_cost_cap_usd": "0.25",
            "status": "failed",
        },
        {
            "attempt_id": attempt_ids[4],
            "modal_cli_process_started": True,
            "modal_cost_cap_usd": "0.25",
            "status": "succeeded",
        },
    ]
    exposure = modal_readiness._derive_modal_compute_exposure(
        attempts,
        measured_by_attempt={
            attempt_ids[0]: modal_readiness.Decimal("0"),
            attempt_ids[1]: modal_readiness.Decimal("0.10"),
            attempt_ids[2]: modal_readiness.Decimal("0"),
            attempt_ids[3]: modal_readiness.Decimal("0.02"),
            attempt_ids[4]: modal_readiness.Decimal("0.30"),
        },
        unresolved_attempt_ids={attempt_ids[3]},
        accounting_label="fixture",
    )

    by_attempt = {item["attempt_id"]: item for item in exposure["attempts"]}
    assert by_attempt[attempt_ids[0]]["conservative_compute_exposure_usd"] == "0"
    assert by_attempt[attempt_ids[1]]["unresolved_compute_reserve_usd"] == "0"
    assert by_attempt[attempt_ids[2]]["unresolved_compute_reserve_usd"] == "0.25"
    assert by_attempt[attempt_ids[3]]["unresolved_compute_reserve_usd"] == "0.25"
    assert by_attempt[attempt_ids[4]]["local_authorization_cap_breached"] is True
    assert exposure["measured_app_billing_usd"] == "0.42"
    assert exposure["unresolved_compute_reserve_usd"] == "0.50"
    assert exposure["conservative_compute_exposure_usd"] == "0.92"
    assert exposure["measured_over_local_authorization_cap_usd"] == "0.05"
    assert exposure["local_authorization_cap_breach_attempt_ids"] == [
        attempt_ids[4]
    ]
    assert exposure["local_authorization_is_platform_hard_bound"] is False


def test_modal_compute_exposure_requires_complete_measured_attempt_roster() -> None:
    with pytest.raises(ValueError, match="measured-billing roster"):
        modal_readiness._derive_modal_compute_exposure(
            [
                {
                    "attempt_id": "a" * 32,
                    "modal_cli_process_started": False,
                    "modal_cost_cap_usd": None,
                    "status": "preflight_failed",
                }
            ],
            measured_by_attempt={},
            unresolved_attempt_ids=set(),
            accounting_label="fixture",
        )


def test_modal_billing_decimal_accepts_canonical_exponent_zero_only() -> None:
    assert modal_readiness._decimal_text("0E-8", "billing.cost") == 0
    assert modal_readiness._decimal_text("0E+8", "billing.cost") == 0
    for value in ("-0E-8", "0e-8", "0E-08", "00E-8", "NaN", "Infinity"):
        with pytest.raises(ValueError, match="canonical, finite, and non-negative"):
            modal_readiness._decimal_text(value, "billing.cost")


def test_intent_and_terminal_reject_shared_action_identity_alias_mutations(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    cases = (
        (
            roster["accepted_attempt_ids"]["resume_attempt"],
            "source_run_id",
            "run_id",
            "must differ",
        ),
        (
            roster["artifact_verifiers"]["offline_smoke"]["attempt_id"],
            "verifier_run_id",
            "run_id",
            "must differ",
        ),
        (
            roster["accepted_attempt_ids"]["canary_greedy_autoresearch"],
            "run_id",
            None,
            "harness-specific suffix",
        ),
    )
    for attempt_id, field, alias_field, message in cases:
        for suffix, validator in (
            (".intent.json", modal_readiness._validate_action_intent_receipt),
            (".json", modal_readiness._validate_action_attempt_receipt),
        ):
            path = _attempt_root(tmp_path) / f"{attempt_id}{suffix}"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload[field] = (
                payload[alias_field]
                if alias_field is not None
                else "wrong-semantic-ar"
            )
            with pytest.raises(ValueError, match=message):
                validator(
                    payload,
                    expected_attempt_id=attempt_id,
                    root=tmp_path,
                )


def test_intent_and_terminal_reject_bool_type_and_path_identity_spoofs(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    cases = (
        (roster["accepted_attempt_ids"]["offline_smoke"], "run_id", True),
        (
            roster["accepted_attempt_ids"]["offline_smoke"],
            "run_id",
            "../escape",
        ),
        (
            roster["accepted_attempt_ids"]["resume_attempt"],
            "source_run_id",
            True,
        ),
        (
            roster["accepted_attempt_ids"]["canary_greedy_autoresearch"],
            "harness",
            True,
        ),
        (
            roster["accepted_attempt_ids"]["canary_greedy_autoresearch"],
            "provider_cost_approved",
            1,
        ),
    )
    for attempt_id, field, spoof in cases:
        for suffix, validator in (
            (".intent.json", modal_readiness._validate_action_intent_receipt),
            (".json", modal_readiness._validate_action_attempt_receipt),
        ):
            path = _attempt_root(tmp_path) / f"{attempt_id}{suffix}"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload[field] = spoof
            with pytest.raises((TypeError, ValueError)):
                validator(
                    payload,
                    expected_attempt_id=attempt_id,
                    root=tmp_path,
                )


def test_cleanup_rejects_rewritten_inline_modal_cost_estimate(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    terminal_path = tmp_path / roster["action_attempt_receipts"][0]
    attempt_id = terminal_path.stem
    intent_path = _attempt_root(tmp_path) / f"{attempt_id}.intent.json"
    for path in (intent_path, terminal_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["modal_cost_estimate"]["action_estimate_usd"] = "0.01"
        _write_json(path, payload)
    _refresh_process_start_intent_binding(
        tmp_path,
        attempt_id=attempt_id,
        terminal_path=terminal_path,
    )

    with pytest.raises(ValueError, match="cost estimate changed"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_reconstructs_modal_command_digest_independently(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    terminal_path = tmp_path / roster["action_attempt_receipts"][-1]
    attempt_id = terminal_path.stem
    intent_path = _attempt_root(tmp_path) / f"{attempt_id}.intent.json"
    for path in (intent_path, terminal_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["modal_command_sha256"] = "f" * 64
        _write_json(path, payload)

    with pytest.raises(ValueError, match="command digest changed"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_requires_intent_timestamp_to_equal_attempt_start(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    intent_path = tmp_path / roster["action_intent_receipts"][-1]
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    intent["created_at_utc"] = "2025-01-01T00:04:59Z"
    _write_json(intent_path, intent)
    _refresh_process_start_intent_binding(
        tmp_path,
        attempt_id=intent_path.stem.removesuffix(".intent"),
        terminal_path=intent_path.with_name(
            f"{intent_path.stem.removesuffix('.intent')}.json"
        ),
    )

    with pytest.raises(ValueError, match="reservation owner or identity changed"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_capture_remote_verification_extracts_final_cli_json(
    tmp_path: Path,
) -> None:
    run_id = "candidate-check-1"
    _downloaded_candidate_run(tmp_path, run_id)
    expected = _remote_verification_payload(tmp_path, run_id)
    transcript = tmp_path / "verify-transcript.txt"
    transcript.write_text(
        "Modal verifier completed\n" + json.dumps(expected, indent=2) + "\n",
        encoding="utf-8",
    )

    captured = capture_remote_verification(
        source_run_id=run_id,
        verifier_run_id=_verifier_run_id(run_id),
        identity=_fixture_identity(tmp_path),
        attempt_id=_verifier_attempt_id(run_id),
        transcript_path=transcript,
        root=tmp_path,
    )

    assert captured == expected
    persisted = (
        tmp_path
        / modal_readiness.modal_remote_verification_receipt_path(
            _fixture_identity(tmp_path),
            run_id,
            _verifier_run_id(run_id),
            _verifier_attempt_id(run_id),
        )
    )
    assert json.loads(persisted.read_text()) == expected
    assert record_artifact_round_trip(
        source_run_id=run_id,
        verifier_run_id=_verifier_run_id(run_id),
        root=tmp_path,
    )[
        "remote_verification_completed"
    ] is True


def test_trailing_verification_json_rejects_symlink_and_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backing = tmp_path / "backing.txt"
    backing.write_text('{"verified": true}', encoding="utf-8")
    linked = tmp_path / "linked.txt"
    linked.symlink_to(backing)
    with pytest.raises(ValueError, match="symbolic link"):
        modal_readiness._trailing_json_object(linked)

    hardlinked = tmp_path / "hardlinked.txt"
    hardlinked.hardlink_to(backing)
    with pytest.raises(ValueError, match="one regular file"):
        modal_readiness._trailing_json_object(hardlinked)

    transcript = tmp_path / "transcript.txt"
    transcript.write_text('{"verified": true}', encoding="utf-8")
    replacement = tmp_path / "replacement.txt"
    replacement.write_text('{"verified": false}', encoding="utf-8")
    original_open = modal_readiness._open_regular_file_descriptor
    calls = 0

    def swapping_open(path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement.replace(path)
        return original_open(path)

    monkeypatch.setattr(
        modal_readiness,
        "_open_regular_file_descriptor",
        swapping_open,
    )
    with pytest.raises(ValueError, match="changed while it was read"):
        modal_readiness._trailing_json_object(transcript)


def test_capture_remote_verification_preserves_distinct_attempts_per_source(
    tmp_path: Path,
) -> None:
    run_id = "candidate-multi-verifier"
    run = _downloaded_candidate_run(tmp_path, run_id)
    first = _remote_verification_payload(tmp_path, run_id)
    first_verifier = first["verifier_run_id"]
    first_transcript = tmp_path / "first-verifier.txt"
    first_transcript.write_text(json.dumps(first), encoding="utf-8")
    capture_remote_verification(
        source_run_id=run_id,
        verifier_run_id=first_verifier,
        identity=_fixture_identity(tmp_path),
        attempt_id="1" * 32,
        transcript_path=first_transcript,
        root=tmp_path,
    )
    first_path = tmp_path / modal_readiness._remote_verification_logical(
        _fixture_identity(tmp_path),
        run_id,
        first_verifier,
        "1" * 32,
    )
    first_bytes = first_path.read_bytes()

    second_verifier = f"{first_verifier}-retry"
    second = json.loads(json.dumps(first))
    second["verifier_run_id"] = second_verifier
    second["verifier_execution_context"].update(
        {
            "run_id": second_verifier,
            "modal_app_id": "ap-verifier-retry",
            "modal_function_id": "fu-verifier-retry",
            "modal_call_id": "fc-verifier-retry",
        }
    )
    second_transcript = tmp_path / "second-verifier.txt"
    second_transcript.write_text(json.dumps(second), encoding="utf-8")
    capture_remote_verification(
        source_run_id=run_id,
        verifier_run_id=second_verifier,
        identity=_fixture_identity(tmp_path),
        attempt_id="2" * 32,
        transcript_path=second_transcript,
        root=tmp_path,
    )
    second_path = tmp_path / modal_readiness._remote_verification_logical(
        _fixture_identity(tmp_path),
        run_id,
        second_verifier,
        "2" * 32,
    )

    assert first_path.read_bytes() == first_bytes
    assert second_path.is_file()
    manifest = load_raw_artifact_manifest(
        run / "artifact_manifest.checkpoint.json"
    ).manifest
    assert modal_readiness._remote_download_evidence(
        tmp_path,
        run_id,
        manifest,
        identity=_fixture_identity(tmp_path),
        verifier_run_id=first_verifier,
        verifier_attempt_id="1" * 32,
    )["verifier_run_id"] == first_verifier
    assert modal_readiness._remote_download_evidence(
        tmp_path,
        run_id,
        manifest,
        identity=_fixture_identity(tmp_path),
        verifier_run_id=second_verifier,
        verifier_attempt_id="2" * 32,
    )["verifier_run_id"] == second_verifier


def test_capture_remote_verification_rejects_rewritten_result(
    tmp_path: Path,
) -> None:
    run_id = "cuda-check-1"
    _downloaded_cuda_run(tmp_path, run_id)
    rewritten = _remote_verification_payload(tmp_path, run_id)
    rewritten["file_count"] = 999
    transcript = tmp_path / "verify-transcript.txt"
    transcript.write_text(
        json.dumps(rewritten),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="differs from downloaded artifacts"):
        capture_remote_verification(
            source_run_id=run_id,
            verifier_run_id=_verifier_run_id(run_id),
            identity=_fixture_identity(tmp_path),
            attempt_id=_verifier_attempt_id(run_id),
            transcript_path=transcript,
            root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "image_source_sha256",
            "0" * 64,
            "image source digest differs",
        ),
        (
            "modal_image_id",
            "im-mixed123",
            "image ID differs",
        ),
    ),
)
def test_round_trip_rejects_mixed_verifier_source_or_image(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    run_id = "candidate-check-mixed"
    _downloaded_candidate_run(tmp_path, run_id)
    path = _write_remote_verification(tmp_path, run_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["verifier_execution_context"][field] = value
    _write_json(path, payload)

    with pytest.raises(ValueError, match=message):
        record_artifact_round_trip(
            source_run_id=run_id,
            verifier_run_id=_verifier_run_id(run_id),
            root=tmp_path,
        )


def test_round_trip_rejects_semantically_equal_raw_manifest_rewrite(
    tmp_path: Path,
) -> None:
    run_id = "candidate-check-raw"
    run = _downloaded_candidate_run(tmp_path, run_id)
    _write_remote_verification(tmp_path, run_id)
    manifest_path = run / "artifact_manifest.checkpoint.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(payload, sort_keys=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_manifest_sha256 differs"):
        record_artifact_round_trip(
            source_run_id=run_id,
            verifier_run_id=_verifier_run_id(run_id),
            root=tmp_path,
        )


def test_capture_remote_verification_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    run_id = "candidate-check-dup"
    _downloaded_candidate_run(tmp_path, run_id)
    expected = _remote_verification_payload(tmp_path, run_id)
    encoded = json.dumps(expected)
    transcript = tmp_path / "verify-transcript.txt"
    transcript.write_text(
        encoded[:-1] + ',"verified":true}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        capture_remote_verification(
            source_run_id=run_id,
            verifier_run_id=_verifier_run_id(run_id),
            identity=_fixture_identity(tmp_path),
            attempt_id=_verifier_attempt_id(run_id),
            transcript_path=transcript,
            root=tmp_path,
        )


def test_cleanup_detects_app_name_inside_endpoint_name(tmp_path: Path) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    _write_json(
        snapshot_root / "endpoint_list.json",
        [
            {
                "name": f"service-{APP_NAME}-web",
                "endpoint_id": "we-test123",
                "status": "deployed",
                "created_at": "2025-01-01 00:00:00+00:00",
                "created_by": "test-user",
            }
        ],
    )
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="active_endpoint_count must be zero"):
        record_resource_cleanup(
            cohort_roster_path=roster_path.relative_to(tmp_path).as_posix(),
            root=tmp_path,
        )


def test_bundle_receipt_rejects_boolean_spoof_after_exact_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def record(run_id: str, function_name: str, index: int) -> dict[str, object]:
        return {
            "run_id": run_id,
            "function_name": function_name,
            "modal_app_id": f"ap-{index}",
            "modal_function_id": f"fu-{index}",
            "modal_call_id": f"fc-{index}",
            "modal_image_id": "im-shared",
            "image_source_sha256": "a" * 64,
            "artifact_manifest_sha256": f"{index:x}" * 64,
        }

    canary_ids = {
        "greedy_autoresearch": "canary-greedy-ar",
        "semantic_autoresearch": "canary-semantic-ar",
        "openevolve_generic": "canary-openevolve-generic",
        "openevolve_semantic": "canary-openevolve-semantic",
    }
    executions = {
        "cuda_environment": record("env-run", "cuda_environment", 1),
        "candidate_smoke": record("candidate-run", "candidate_smoke", 2),
        "resume_attempt": record("resume-run", "checkpoint_resume", 3),
        "offline_smoke": record("offline-run", "offline_smoke", 4),
        "canaries": {
            harness: record(run_id, f"canary_{harness}", index)
            for index, (harness, run_id) in enumerate(canary_ids.items(), start=5)
        },
    }
    roster_path = tmp_path / "cohort_roster.json"
    roster_path.write_text("{}\n", encoding="utf-8")
    cohort_roster = {
        "path": "cohort_roster.json",
        "sha256": hashlib.sha256(roster_path.read_bytes()).hexdigest(),
    }
    claims = {
        "source_tree_sha256": "9" * 64,
        "cohort_id": "bundle-boolean-spoof",
        "cohort_roster": cohort_roster,
        "executions": executions,
        "image_source_sha256": "a" * 64,
        "migration_lineage": {
            "path": "lineage.json",
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
        "evidence": {"strict": True},
        "required_artifacts": [
            {"path": "outputs/evidence.json", "sha256": "b" * 64, "size_bytes": 1}
        ],
        "validated": True,
    }
    monkeypatch.setattr(
        modal_readiness,
        "_derive_migration_bundle_claims",
        lambda *args, **kwargs: claims,
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_cohort_roster",
        lambda *args, **kwargs: (
            {
                "source_tree_sha256": "9" * 64,
                "image_source_sha256": "a" * 64,
                "cohort_id": "bundle-boolean-spoof",
                "accepted_primary_runs": {
                    "cuda_environment": "env-run",
                    "offline_smoke": "offline-run",
                    "candidate_smoke": "candidate-run",
                    "resume_attempt": "resume-run",
                    **{
                        f"canary_{harness}": run_id
                        for harness, run_id in canary_ids.items()
                    },
                }
            },
            roster_path,
        ),
    )
    contract = MODAL_READINESS_RECEIPT_CONTRACTS[
        "modal_migration_validation_bundle_validated"
    ]
    payload = {
        **contract["receipt_contract"],
        "recorded_at_utc": "2026-08-09T00:00:00Z",
        **claims,
        "validated": 1,
    }
    receipt = tmp_path / modal_readiness.modal_component_receipt_path(
        ModalLiveCohortIdentity(
            source_tree_sha256=claims["source_tree_sha256"],
            image_source_sha256=claims["image_source_sha256"],
            cohort_id=claims["cohort_id"],
        ),
        "modal_migration_validation_bundle_validated",
    )
    _write_json(receipt, payload)

    with pytest.raises(ValueError, match="validated must be exactly True"):
        validate_modal_readiness_receipt(
            "modal_migration_validation_bundle_validated",
            receipt,
            root=tmp_path,
        )


def test_migration_cleanup_revalidates_all_eight_contexts_and_billing(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster_logical = roster_path.relative_to(tmp_path).as_posix()
    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_logical,
        recorded_at_utc="2025-01-02T00:00:00Z",
    )

    assert len(claims["accepted_primary_executions"]) == 8
    assert len(claims["artifact_verifier_executions"]) == 8
    assert claims["cohort_billing_total_usd"] == "0.16"
    assert claims["active_app_count"] == 0
    assert claims["active_container_count"] == 0
    assert claims["active_endpoint_count"] == 0

    billing_path = snapshot_root / "billing_report.json"
    rows = json.loads(billing_path.read_text(encoding="utf-8"))
    lagged_app_id = rows[-1]["object_id"]
    rows.pop()
    _write_json(billing_path, rows)
    _refresh_snapshot_capture_manifest(snapshot_root)
    lagged_claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_logical,
        recorded_at_utc="2025-01-02T00:00:00Z",
    )
    lagged_attribution = next(
        item
        for item in lagged_claims["billing_attributions"]
        if lagged_app_id in item["object_ids"]
    )
    assert lagged_attribution["billing_total_usd"] == "0"
    lagged_exposure = next(
        item
        for item in lagged_claims["modal_compute_exposure"]["final_cohort"][
            "attempts"
        ]
        if item["attempt_id"] == lagged_attribution["attempt_id"]
    )
    assert lagged_exposure["unresolved_compute_reserve_usd"] == "0.25"


def test_cleanup_ignores_unowned_app_name_billing_in_other_environment(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    billing_path = snapshot_root / "billing_report.json"
    rows = json.loads(billing_path.read_text())
    rows.append(
        {
            "object_id": "ap-staging-unrelated",
            "description": APP_NAME,
            "environment": "staging",
            "interval_start": "2025-01-01T00:00:00+00:00",
            "resource": "CPU",
            "cost": "9.99",
        }
    )
    _write_json(billing_path, rows)
    _refresh_snapshot_capture_manifest(snapshot_root)

    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_path.relative_to(tmp_path).as_posix(),
        recorded_at_utc="2025-01-02T00:00:00Z",
    )

    assert claims["cohort_billing_total_usd"] == "0.16"
    assert all(
        "ap-staging-unrelated" not in attribution["object_ids"]
        for attribution in claims["billing_attributions"]
    )


def test_cleanup_roster_cannot_omit_a_final_cohort_attempt(tmp_path: Path) -> None:
    roster_path, _ = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    omitted = roster["action_attempt_receipts"].pop()
    attempt_id = Path(omitted).stem
    roster["attempt_classifications"] = [
        item
        for item in roster["attempt_classifications"]
        if item["attempt_id"] != attempt_id
    ]
    roster["billing_attributions"] = [
        item
        for item in roster["billing_attributions"]
        if item["attempt_id"] != attempt_id
    ]
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="does not classify every Modal action"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_duplicate_billing_rows_and_app_ids(tmp_path: Path) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster_logical = roster_path.relative_to(tmp_path).as_posix()
    billing_path = snapshot_root / "billing_report.json"
    billing = json.loads(billing_path.read_text(encoding="utf-8"))
    billing.append({**billing[0], "description": "different-description"})
    _write_json(billing_path, billing)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="duplicate accounting row"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_logical,
            recorded_at_utc="2025-01-02T00:00:00Z",
        )

    billing.pop()
    _write_json(billing_path, billing)
    _refresh_snapshot_capture_manifest(snapshot_root)
    app_path = snapshot_root / "app_list.json"
    apps = json.loads(app_path.read_text(encoding="utf-8"))
    apps.append(dict(apps[0]))
    _write_json(app_path, apps)
    _refresh_snapshot_capture_manifest(snapshot_root)
    with pytest.raises(ValueError, match="duplicate app_id"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_logical,
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("created_at", "2025-01-01 00:04:00+00:00"),
        ("stopped_at", "2025-01-01 00:07:00+00:00"),
    ),
    ids=("before-attempt", "after-attempt"),
)
def test_cleanup_requires_app_lifecycle_contained_by_launcher_attempt(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    app_path = snapshot_root / "app_list.json"
    apps = json.loads(app_path.read_text())
    apps[0][field] = value
    _write_json(app_path, apps)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="lifecycle is not contained"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


@pytest.mark.parametrize("run_kind", ("current", "superseded"))
def test_cleanup_rejects_required_run_entry_after_command_capture(
    tmp_path: Path,
    run_kind: str,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text())
    run_id = (
        roster["accepted_primary_runs"]["cuda_environment"]
        if run_kind == "current"
        else roster["superseded_usage"]["run_id"]
    )
    run_path = snapshot_root / "run_directory_list.json"
    rows = json.loads(run_path.read_text())
    matching_row = next(
        row for row in rows if row["filename"] == f"/runs/{run_id}"
    )
    matching_row["created_modified"] = "2025-01-01 01:01 UTC"
    _write_json(run_path, rows)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="owned Volume /runs entry timestamp exceeds"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_artifact_volume_creation_after_command_capture(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    volume_path = snapshot_root / "volume_list.json"
    rows = json.loads(volume_path.read_text())
    rows[0]["created_at"] = "2025-01-01 01:01:00+00:00"
    _write_json(volume_path, rows)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="artifact Volume creation timestamp exceeds"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_allows_prior_app_absent_from_recently_stopped_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text())
    final_identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    lineage, lineage_path, lineage_sha256 = (
        modal_readiness._load_migration_lineage(
            tmp_path,
            roster,
            final_identity,
        )
    )
    prior_identity = ModalLiveCohortIdentity(
        source_tree_sha256="7" * 64,
        image_source_sha256="8" * 64,
        cohort_id="prior-recent-app-history",
    )
    prior_path = modal_readiness.modal_prior_quarantine_accounting_path(
        prior_identity
    ).as_posix()
    prior_app_id = "ap-prior-no-longer-recent"
    modal_exposure = {
        "measured_app_billing_usd": "0",
        "unresolved_compute_reserve_usd": "0",
        "conservative_compute_exposure_usd": "0",
    }
    provider_estimate = {"conservative_provider_spend_bound_usd": "0"}
    prior_payload = {
        "app_lifecycles": [
            {
                "app_id": prior_app_id,
                "created_at_utc": "2024-12-31T22:00:00Z",
                "stopped_at_utc": "2024-12-31T22:01:00Z",
            }
        ],
        "selected_billing_rows": [],
        "modal_compute_exposure": modal_exposure,
        "provider_spend_estimate": provider_estimate,
        "retained_storage_estimate": {
            "conservative_total_bytes": 0,
            "estimated_monthly_usd": "0",
        },
    }
    prior_metadata = {
        "identity": prior_identity,
        "run_ids": set(),
        "app_ids": {prior_app_id},
        "call_ids": set(),
        "provider_request_ids": set(),
        "provider_response_ids": set(),
    }
    lineage["prior_quarantined_cohorts"] = [
        {
            "accounting_receipt": {"path": prior_path},
            "modal_compute_exposure": modal_exposure,
            "provider_spend_estimate": provider_estimate,
        }
    ]
    lineage["prior_modal_measured_app_billing_usd"] = "0"
    lineage["prior_modal_unresolved_compute_reserve_usd"] = "0"
    lineage["prior_modal_conservative_exposure_usd"] = "0"
    lineage["prior_provider_spend_bound_usd"] = "0"
    lineage["migration_provider_spend_bound_usd"] = lineage[
        "final_provider_spend_bound_usd"
    ]
    monkeypatch.setattr(
        modal_readiness,
        "_load_migration_lineage",
        lambda *args, **kwargs: (lineage, lineage_path, lineage_sha256),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_prior_quarantine_accounting",
        lambda *args, **kwargs: (prior_payload, prior_metadata),
    )

    app_rows = json.loads((snapshot_root / "app_list.json").read_text())
    assert prior_app_id not in {row["app_id"] for row in app_rows}
    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_path.relative_to(tmp_path).as_posix(),
        recorded_at_utc="2025-01-02T00:00:00Z",
    )

    assert claims["active_app_count"] == 0
    assert claims["modal_compute_exposure"]["prior_quarantined_cohorts"]


def test_cleanup_preserves_failed_and_recovery_run_ids(tmp_path: Path) -> None:
    roster_path, _ = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    recovery_attempt_id = roster["accepted_attempt_ids"]["cuda_environment"]
    failed_template_id = roster["accepted_attempt_ids"]["offline_smoke"]
    failed_template_path = _attempt_root(tmp_path) / f"{failed_template_id}.json"
    failed_attempt_id = f"{17:032x}"
    failed_receipt = json.loads(
        failed_template_path.read_text(encoding="utf-8")
    )
    failed_receipt.update(
        {
            "attempt_id": failed_attempt_id,
            "status": "preflight_rejected",
            "failure_kind": "preflight",
            "started_at_utc": "2025-01-01T00:01:00Z",
            "finished_at_utc": "2025-01-01T00:02:00Z",
            "run_id": "failed-cuda-run",
            "concrete_remote_run_ids": ["failed-cuda-run"],
            "modal_cli_process_started": False,
            "remote_execution_state": "definitely_not_started",
            "returncode": None,
            "process_group_closed": None,
        }
    )
    _clear_process_start_evidence(failed_receipt)
    _refresh_receipt_reservations(tmp_path, failed_receipt)
    failed_receipt["modal_command_sha256"] = (
        modal_readiness._reconstructed_modal_command_sha256(
            failed_receipt,
            root=tmp_path,
        )
    )
    failed_intent, failed_terminal = _write_attempt_intent_and_terminal(
        tmp_path,
        failed_receipt,
    )
    roster["action_intent_receipts"].append(failed_intent)
    roster["action_intent_receipts"].sort()
    roster["action_attempt_receipts"].append(failed_terminal)
    roster["action_attempt_receipts"].sort()
    for item in roster["attempt_classifications"]:
        if item["attempt_id"] == recovery_attempt_id:
            item["roles"] = sorted([*item["roles"], "recovery"])
    roster["attempt_classifications"].append(
        {"attempt_id": failed_attempt_id, "roles": ["failed"]}
    )
    roster["attempt_classifications"].sort(key=lambda item: item["attempt_id"])
    roster["billing_attributions"].append(
        {
            "attempt_id": failed_attempt_id,
            "disposition": "no_remote_start",
            "object_ids": [],
        }
    )
    roster["billing_attributions"].sort(key=lambda item: item["attempt_id"])
    roster["recovery_links"] = [
        {
            "failed_attempt_id": failed_attempt_id,
            "recovery_attempt_id": recovery_attempt_id,
            "recovered_run_ids": [roster["cleanup_run_id"]],
        }
    ]
    roster["declared_failed_run_ids"] = []
    roster["declared_recovery_run_ids"] = [roster["cleanup_run_id"]]
    _write_json(roster_path, roster)
    _refresh_migration_lineage(roster_path)
    roster_logical = roster_path.relative_to(tmp_path).as_posix()

    claims = modal_readiness._derive_cleanup_claims(
        tmp_path,
        roster_logical,
        recorded_at_utc="2025-01-02T00:00:00Z",
    )
    assert claims["failed_run_ids"] == []
    assert claims["recovery_run_ids"] == ["cleanup-check-1"]

    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster["declared_failed_run_ids"] = ["failed-cuda-run"]
    _write_json(roster_path, roster)
    with pytest.raises(ValueError, match="hides or invents failed run IDs"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_logical,
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_cleanup_rejects_boolean_attempt_returncode(tmp_path: Path) -> None:
    roster_path, _ = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_path = tmp_path / roster["action_attempt_receipts"][0]
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt["returncode"] = True
    _write_json(attempt_path, attempt)

    with pytest.raises(ValueError, match="exact integer or null"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_action_intent_persistence_is_global_only_and_rejected_from_cohort(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["artifact_verifiers"]["offline_smoke"]["attempt_id"]
    terminal_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.update(
        {
            "status": "preflight_failed",
            "failure_kind": "action_intent_persistence",
            "modal_cli_process_started": False,
            "remote_execution_state": "definitely_not_started",
            "returncode": None,
            "process_group_closed": None,
        }
    )
    _clear_process_start_evidence(terminal)
    validated = modal_readiness._validate_action_attempt_receipt(
        terminal,
        expected_attempt_id=attempt_id,
        root=tmp_path,
    )
    assert validated["failure_kind"] == "action_intent_persistence"
    _write_json(terminal_path, terminal)
    intent_path = _attempt_root(tmp_path) / f"{attempt_id}.intent.json"
    intent_raw = intent_path.read_bytes()
    intent_path.unlink()
    with pytest.raises(ValueError, match="attempt ID sets differ"):
        modal_readiness._cohort_action_journal(
            tmp_path,
            _fixture_identity(tmp_path),
        )

    intent_path.write_bytes(intent_raw)
    with pytest.raises(ValueError, match="global-rejection-only"):
        modal_readiness._cohort_action_journal(
            tmp_path,
            _fixture_identity(tmp_path),
        )


def test_action_intent_post_persistence_requires_and_accepts_exact_intent(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["artifact_verifiers"]["offline_smoke"]["attempt_id"]
    terminal_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.update(
        {
            "status": "preflight_failed",
            "failure_kind": "action_intent_post_persistence",
            "modal_cli_process_started": False,
            "remote_execution_state": "definitely_not_started",
            "returncode": None,
            "process_group_closed": None,
        }
    )
    _clear_process_start_evidence(terminal)
    _write_json(terminal_path, terminal)

    _journal, attempts = modal_readiness._cohort_action_journal(
        tmp_path,
        _fixture_identity(tmp_path),
    )
    observed = next(item for item in attempts if item["attempt_id"] == attempt_id)
    assert observed["failure_kind"] == "action_intent_post_persistence"

    intent_path = _attempt_root(tmp_path) / f"{attempt_id}.intent.json"
    intent_path.unlink()
    with pytest.raises(ValueError, match="attempt ID sets differ"):
        modal_readiness._cohort_action_journal(
            tmp_path,
            _fixture_identity(tmp_path),
        )


def test_cohort_journal_rejects_intent_persistence_uncertain_failure_kind(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["artifact_verifiers"]["offline_smoke"]["attempt_id"]
    terminal_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal.update(
        {
            "status": "preflight_failed",
            "failure_kind": "action_intent_persistence_uncertain",
            "modal_cli_process_started": False,
            "remote_execution_state": "definitely_not_started",
            "returncode": None,
            "process_group_closed": None,
        }
    )
    _clear_process_start_evidence(terminal)

    validated = modal_readiness._validate_action_attempt_receipt(
        terminal,
        expected_attempt_id=attempt_id,
        root=tmp_path,
    )
    assert validated["failure_kind"] == "action_intent_persistence_uncertain"
    _write_json(terminal_path, terminal)
    with pytest.raises(ValueError, match="global-rejection-only"):
        modal_readiness._cohort_action_journal(
            tmp_path,
            _fixture_identity(tmp_path),
        )
    (_attempt_root(tmp_path) / f"{attempt_id}.intent.json").unlink()
    with pytest.raises(ValueError, match="attempt ID sets differ"):
        modal_readiness._cohort_action_journal(
            tmp_path,
            _fixture_identity(tmp_path),
        )


def test_cohort_terminal_injection_requires_intent_for_every_prestart_status(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["artifact_verifiers"]["offline_smoke"]["attempt_id"]
    terminal_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    intent_path = _attempt_root(tmp_path) / f"{attempt_id}.intent.json"
    original_terminal = terminal_path.read_bytes()
    original_intent = intent_path.read_bytes()
    terminal_path.unlink()
    with pytest.raises(ValueError, match="attempt ID sets differ"):
        modal_readiness._cohort_action_journal(
            tmp_path,
            _fixture_identity(tmp_path),
        )
    terminal_path.write_bytes(original_terminal)
    cases = (
        ("preflight_failed", "preflight", True),
        ("preflight_failed", "action_intent_persistence", False),
        ("preflight_failed", "action_intent_post_persistence", True),
        ("preflight_failed", "action_intent_persistence_uncertain", False),
        ("preflight_rejected", "preflight", True),
        ("lock_contended", "local_launcher_lock", True),
        ("interrupted", "interrupt", True),
        ("cli_failed", "process_launch", True),
        ("cleanup_failed", "python_execution_cleanup", True),
    )
    for status, failure_kind, paired_allowed in cases:
        terminal = json.loads(original_terminal)
        terminal.update(
            {
                "status": status,
                "failure_kind": failure_kind,
                "modal_cli_process_started": False,
                "remote_execution_state": "definitely_not_started",
                "returncode": None,
                "process_group_closed": None,
            }
        )
        _clear_process_start_evidence(terminal)
        modal_readiness._validate_action_attempt_receipt(
            terminal,
            expected_attempt_id=attempt_id,
            root=tmp_path,
        )
        _write_json(terminal_path, terminal)
        intent_path.unlink()
        with pytest.raises(ValueError, match="attempt ID sets differ"):
            modal_readiness._cohort_action_journal(
                tmp_path,
                _fixture_identity(tmp_path),
            )

        intent_path.write_bytes(original_intent)
        if paired_allowed:
            modal_readiness._cohort_action_journal(
                tmp_path,
                _fixture_identity(tmp_path),
            )
        else:
            with pytest.raises(ValueError, match="global-rejection-only"):
                modal_readiness._cohort_action_journal(
                    tmp_path,
                    _fixture_identity(tmp_path),
                )
        terminal_path.write_bytes(original_terminal)


def test_unstarted_provider_rejection_preserves_unapproved_sanitized_inputs(
    tmp_path: Path,
) -> None:
    roster_path, _ = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["accepted_attempt_ids"]["canary_greedy_autoresearch"]
    path = tmp_path / roster["action_attempt_receipts"][
        [Path(item).stem for item in roster["action_attempt_receipts"]].index(
            attempt_id
        )
    ]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "preflight_rejected",
            "failure_kind": "preflight",
            "provider_cost_approved": False,
            "modal_command_sha256": None,
            "modal_cli_process_started": False,
            "remote_execution_state": "definitely_not_started",
            "returncode": None,
            "process_group_closed": None,
        }
    )
    _clear_process_start_evidence(receipt)

    validated = modal_readiness._validate_action_attempt_receipt(
        receipt,
        expected_attempt_id=attempt_id,
        root=tmp_path,
    )

    assert validated["provider_cost_approved"] is False
    assert validated["provider_approval_plan_path"] is not None


def _aggregate_downstream_failure_cohort(
    root: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, tuple[dict[str, object], Path, object, ExecutionContextV1]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    str,
]:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(root)
    roster, _ = modal_readiness._load_cohort_roster(
        root,
        roster_path.relative_to(root).as_posix(),
    )
    primary = modal_readiness._primary_executions_from_roster(root, roster)
    verifier_evidence = modal_readiness._artifact_verifier_executions(
        root,
        roster,
        primary,
    )
    attempts, _attempt_evidence, _aggregates = (
        modal_readiness._load_action_attempts(
            root,
            _fixture_identity(root),
            roster["action_attempt_receipts"],
            roster["action_intent_receipts"],
            roster["provider_canary_aggregate_outcome_receipts"],
        )
    )
    aggregate_attempt_id = f"{17:032x}"
    template = next(
        item
        for item in attempts
        if item["attempt_id"]
        == roster["accepted_attempt_ids"]["canary_greedy_autoresearch"]
    )
    child_run_ids = {
        harness: f"aggregate-downstream-{index}"
        for index, harness in enumerate(modal_readiness.CANARY_ORDER, start=1)
    }
    aggregate_attempt = {
        **template,
        "attempt_id": aggregate_attempt_id,
        "started_at_utc": "2025-01-01T00:01:00Z",
        "finished_at_utc": "2025-01-01T00:02:00Z",
        "status": "failed",
        "failure_kind": "modal_cli_exit",
        "action": "canaries",
        "run_id": "aggregate-downstream",
        "concrete_remote_run_ids": [
            child_run_ids[harness] for harness in modal_readiness.CANARY_ORDER
        ],
        "harness": None,
        "returncode": 2,
    }
    attempts.append(aggregate_attempt)
    roster["attempt_classifications"].append(
        {
            "attempt_id": aggregate_attempt_id,
            "roles": ["failed", "quarantined"],
        }
    )
    roster["billing_attributions"].append(
        {
            "attempt_id": aggregate_attempt_id,
            "disposition": "billed",
            "object_ids": ["ap-aggregate-downstream"],
        }
    )
    aggregate_children: list[dict[str, object]] = []
    added_outcomes: list[dict[str, object]] = []
    for index, harness in enumerate(modal_readiness.CANARY_ORDER):
        child_failed = index == 0
        child_run_id = child_run_ids[harness]
        aggregate_children.append(
            {
                "harness": harness,
                "run_id": child_run_id,
                "status": "failed" if child_failed else "success",
                "error_type": "RuntimeError" if child_failed else None,
            }
        )
        source_outcome = next(
            item
            for item in roster["provider_canary_outcomes"]
            if item["harness"] == harness
        )
        added_outcomes.append(
            {
                **source_outcome,
                "launcher_attempt_id": aggregate_attempt_id,
                "concrete_run_id": child_run_id,
                "outcome": (
                    "failed" if child_failed else "completed_unaccepted"
                ),
            }
        )
    roster["provider_canary_outcomes"].extend(added_outcomes)
    recovery_attempt_id = roster["accepted_attempt_ids"][
        "canary_greedy_autoresearch"
    ]
    recovery_run_id = roster["accepted_primary_runs"][
        "canary_greedy_autoresearch"
    ]
    recovery_classification = next(
        item
        for item in roster["attempt_classifications"]
        if item["attempt_id"] == recovery_attempt_id
    )
    recovery_classification["roles"] = sorted(
        [*recovery_classification["roles"], "recovery"]
    )
    roster["recovery_links"] = [
        {
            "failed_attempt_id": aggregate_attempt_id,
            "recovery_attempt_id": recovery_attempt_id,
            "recovered_run_ids": [recovery_run_id],
        }
    ]
    roster["declared_failed_run_ids"] = [child_run_ids["greedy_autoresearch"]]
    roster["declared_quarantined_run_ids"] = sorted(child_run_ids.values())
    roster["declared_recovery_run_ids"] = [recovery_run_id]
    aggregates = {
        aggregate_attempt_id: {
            "attempt_id": aggregate_attempt_id,
            "outcomes": aggregate_children,
        }
    }
    return (
        roster,
        attempts,
        primary,
        verifier_evidence,
        aggregates,
        aggregate_attempt_id,
    )


def test_aggregate_downstream_failure_requires_failed_child_outcome(
    tmp_path: Path,
) -> None:
    roster, attempts, primary, verifiers, aggregates, attempt_id = (
        _aggregate_downstream_failure_cohort(tmp_path)
    )

    cohort = modal_readiness._validate_attempt_cohort(
        tmp_path,
        roster,
        attempts,
        primary,
        verifiers,
        [],
        aggregates,
    )
    assert cohort["failed_run_ids"] == ["aggregate-downstream-1"]
    assert cohort["recovery_run_ids"] == ["accepted-canary-greedy-ar"]

    failed_outcome = next(
        item
        for item in roster["provider_canary_outcomes"]
        if item["launcher_attempt_id"] == attempt_id
        and item["harness"] == "greedy_autoresearch"
    )
    failed_outcome["outcome"] = "completed_unaccepted"
    with pytest.raises(ValueError, match="child status differs"):
        modal_readiness._validate_attempt_cohort(
            tmp_path,
            roster,
            attempts,
            primary,
            verifiers,
            [],
            aggregates,
        )


def test_successful_aggregate_provider_child_is_not_recovery_eligible(
    tmp_path: Path,
) -> None:
    roster, attempts, primary, verifiers, aggregates, attempt_id = (
        _aggregate_downstream_failure_cohort(tmp_path)
    )
    semantic_attempt_id = roster["accepted_attempt_ids"][
        "canary_semantic_autoresearch"
    ]
    semantic_run_id = roster["accepted_primary_runs"][
        "canary_semantic_autoresearch"
    ]
    semantic_classification = next(
        item
        for item in roster["attempt_classifications"]
        if item["attempt_id"] == semantic_attempt_id
    )
    semantic_classification["roles"] = sorted(
        [*semantic_classification["roles"], "recovery"]
    )
    roster["recovery_links"].append(
        {
            "failed_attempt_id": attempt_id,
            "recovery_attempt_id": semantic_attempt_id,
            "recovered_run_ids": [semantic_run_id],
        }
    )
    roster["declared_recovery_run_ids"].append(semantic_run_id)
    roster["declared_recovery_run_ids"].sort()

    with pytest.raises(ValueError, match="successful provider child"):
        modal_readiness._validate_attempt_cohort(
            tmp_path,
            roster,
            attempts,
            primary,
            verifiers,
            [],
            aggregates,
        )


@pytest.mark.parametrize(
    ("returncode", "status", "failure_kind", "all_succeeded"),
    (
        (0, "succeeded", None, True),
        (2, "failed", "modal_cli_exit", False),
    ),
)
def test_action_journal_requires_typed_aggregate_for_completed_canaries(
    tmp_path: Path,
    returncode: int,
    status: str,
    failure_kind: str | None,
    all_succeeded: bool,
) -> None:
    attempt_id = "a" * 32
    run_id = "aggregate-journal-run"
    image_sha256 = "b" * 64
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256=image_sha256,
        cohort_id="aggregate-journal",
    )
    journal = tmp_path / modal_readiness.modal_action_attempt_directory(identity)
    journal.mkdir(parents=True)
    preflight_path = tmp_path / (
        modal_readiness.modal_candidate_resume_preflight_receipt_path(
            identity,
            "f" * 64,
        )
    )
    _write_json(preflight_path, {"fixture": "preflight"})
    price_path = (
        tmp_path
        / "outputs/readiness/modal_resource_cleanup/aggregate-journal-run/"
        "provider_price_basis.json"
    )
    _write_json(
        price_path,
        {
            "schema_name": "ProviderPriceBasis",
            "schema_version": "1.0",
            "model": modal_readiness.TARGET_MODEL,
            "official_source_url": "https://openai.com/api/pricing/",
            "retrieved_at_utc": "2025-01-01T00:00:00Z",
            "uncached_input_usd_per_million_tokens": "1",
            "output_usd_per_million_tokens": "1",
            "per_request_fee_usd": "0",
        },
    )
    plan_path = (
        tmp_path
        / "outputs/readiness/provider_canary_approval/aggregate-journal.json"
    )
    plan = _provider_plan_for_fixture(
        identity,
        preflight_path=preflight_path.relative_to(tmp_path).as_posix(),
        preflight_sha256=hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
    )
    _write_json(plan_path, plan)
    (
        modal_price_basis_path,
        modal_price_basis_sha256,
        modal_price_basis,
    ) = _write_modal_price_basis(tmp_path, image_sha256)
    predecessor_receipts = [
        *_local_freeze_predecessors(tmp_path),
        {
            "gate": "candidate_resume_preflight_validated",
            "path": preflight_path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
        }
    ]
    modal_resource_profile = modal_readiness._expected_modal_resource_profile(
        "canaries",
        None,
    )
    modal_cost_estimate = modal_readiness.derive_modal_action_cost_estimate(
        action="canaries",
        harness=None,
        resource_profile=modal_resource_profile,
        price_basis=modal_price_basis,
    )
    shared = {
        "action": "canaries",
        "run_id": run_id,
        "concrete_remote_run_ids": [
            f"{run_id}-{modal_readiness._CANARY_SUFFIXES[harness]}"
            for harness in modal_readiness.CANARY_ORDER
        ],
        **_fixture_local_containment(),
        "source_run_id": None,
        "verifier_run_id": None,
        "harness": None,
        "source_tree_sha256": identity.source_tree_sha256,
        "cohort_id": identity.cohort_id,
        "approved_image_source_sha256": image_sha256,
        "modal_command_sha256": "c" * 64,
        "modal_profile": "scalingintelligence",
        "modal_environment": modal_readiness.MODAL_ENVIRONMENT,
        "outer_cli_timeout_seconds": modal_readiness._expected_attempt_timeout(
            "canaries"
        ),
        "modal_cost_cap_usd": "0.50",
        "modal_resource_profile": modal_resource_profile,
        "modal_price_basis_path": modal_price_basis_path,
        "modal_price_basis_sha256": modal_price_basis_sha256,
        "modal_cost_estimate": modal_cost_estimate,
        "modal_cost_approved": True,
        "provider_cost_approved": True,
        "provider_cost_cap_usd": "2",
        "provider_approval_plan_path": plan_path.relative_to(tmp_path).as_posix(),
        "approval_plan_sha256": plan["approval_plan_sha256"],
        "provider_price_basis_path": price_path.relative_to(tmp_path).as_posix(),
        "provider_price_basis_sha256": hashlib.sha256(
            price_path.read_bytes()
        ).hexdigest(),
        "predecessor_receipts": predecessor_receipts,
        "source_evidence_recovery": False,
    }
    shared["modal_command_sha256"] = (
        modal_readiness._reconstructed_modal_command_sha256(
            shared,
            root=tmp_path,
        )
    )
    reservation_seed = {
        **shared,
        "attempt_id": attempt_id,
        "started_at_utc": "2025-01-01T00:00:00Z",
    }
    _refresh_receipt_reservations(
        tmp_path,
        reservation_seed,
        identity=identity,
    )
    shared["launch_capability_sha256"] = reservation_seed[
        "launch_capability_sha256"
    ]
    shared["remote_run_reservations"] = reservation_seed[
        "remote_run_reservations"
    ]
    terminal = {
        "schema_name": "ModalActionAttemptReceipt",
        "schema_version": "3.6",
        "attempt_id": attempt_id,
        "started_at_utc": "2025-01-01T00:00:00Z",
        "finished_at_utc": "2025-01-01T00:01:00Z",
        "status": status,
        "failure_kind": failure_kind,
        **shared,
        "local_process_start_receipt_path": None,
        "local_process_start_receipt_sha256": None,
        "local_process_id": None,
        "local_process_group_id": None,
        "local_session_id": None,
        "modal_cli_process_started": True,
        "remote_execution_state": "may_have_started",
        "returncode": returncode,
        "process_group_closed": True,
    }
    outcomes = []
    for index, harness in enumerate(modal_readiness.CANARY_ORDER):
        failed = not all_succeeded and index == 0
        outcomes.append(
            {
                "harness": harness,
                "run_id": f"{run_id}-{modal_readiness._CANARY_SUFFIXES[harness]}",
                "status": "failed" if failed else "success",
                "error_type": "RuntimeError" if failed else None,
            }
        )
    aggregate = {
        "schema_name": "ProviderCanaryAggregateOutcomeReceipt",
        "schema_version": "1.1",
        "attempt_id": attempt_id,
        "run_id_prefix": run_id,
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": image_sha256,
        "cohort_id": identity.cohort_id,
        "harness_order": list(modal_readiness.CANARY_ORDER),
        "outcomes": outcomes,
        "all_succeeded": all_succeeded,
    }
    journal_logical = journal.relative_to(tmp_path).as_posix()
    aggregate_logical = f"{journal_logical}/{attempt_id}.aggregate.json"
    intent_logical, terminal_logical = _write_attempt_intent_and_terminal(
        tmp_path,
        terminal,
        identity=identity,
    )
    _write_json(tmp_path / aggregate_logical, aggregate)

    attempts, evidence, aggregates = modal_readiness._load_action_attempts(
        tmp_path,
        identity,
        [terminal_logical],
        [intent_logical],
        [aggregate_logical],
    )

    assert attempts[0]["returncode"] == returncode
    assert evidence[0]["provider_canary_aggregate_outcomes"]["receipt"] == aggregate
    assert aggregates[attempt_id] == aggregate

    (tmp_path / aggregate_logical).unlink()
    with pytest.raises(ValueError, match="do not cover every completed aggregate"):
        modal_readiness._load_action_attempts(
            tmp_path,
            identity,
            [terminal_logical],
            [intent_logical],
            [],
        )


def test_readme_paid_launcher_examples_match_the_approval_contract() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    lines = readme.splitlines()
    commands: list[str] = []
    for index, line in enumerate(lines):
        if line != ".venv/bin/python scripts/launch_modal.py \\":
            continue
        command_lines = [line]
        while command_lines[-1].endswith("\\"):
            index += 1
            command_lines.append(lines[index])
        commands.append("\n".join(command_lines))

    assert commands
    for command in commands:
        assert (
            '--expected-image-source-sha256 "$APPROVED_IMAGE_SOURCE_SHA256"'
            in command
        )
        assert '--modal-cost-cap-usd "$APPROVED_MODAL_ACTION_CAP_USD"' in command
        assert '--modal-price-basis-path "$MODAL_PRICE_BASIS_PATH"' in command
        assert (
            '--modal-price-basis-sha256 "$MODAL_PRICE_BASIS_SHA256"' in command
        )
        assert "--approved" in command
        if "--action canaries" in command:
            assert "--outer-cli-timeout-seconds 2100" in command
        else:
            assert "--outer-cli-timeout-seconds 1200" in command
        if "--action download" in command:
            assert "--source-action-attempt-receipt-path" in command
        if "--action canary" in command:
            for flag in (
                "--provider-approved",
                "--provider-cost-cap-usd",
                "--provider-approval-plan-path",
                "--approval-plan-sha256",
                "--provider-price-basis-path",
                "--provider-price-basis-sha256",
            ):
                assert flag in command

    normalized = " ".join(readme.replace("\\\n", " ").split())
    assert (
        "../.venv/bin/ruff check --isolated --select E4,E7,E9,F --ignore E402 "
        "--target-version py312 --line-length 88 agents analysis architecture_ir "
        "artifacts audits baselines common containment evaluation mechanism "
        "novelty private_eval reconstruction replication reporting "
        "research_ledger review "
        "sealed_eval scripts study tests modal_action_journal.py modal_app.py "
        "modal_boundary.py modal_image_build.py"
    ) in normalized


def test_provider_price_basis_is_create_only(tmp_path: Path) -> None:
    arguments = {
        "source_tree_sha256_value": "9" * 64,
        "image_source_sha256": "a" * 64,
        "cohort_id": "price-basis-run",
        "official_source_url": "https://openai.com/api/pricing/",
        "retrieved_at_utc": "2025-01-01T00:00:00Z",
        "uncached_input_usd_per_million_tokens": "1.25",
        "output_usd_per_million_tokens": "10",
        "per_request_fee_usd": "0",
        "root": tmp_path,
    }
    payload = modal_readiness.create_provider_price_basis(**arguments)
    assert payload["schema_name"] == "ProviderPriceBasis"
    with pytest.raises(FileExistsError):
        modal_readiness.create_provider_price_basis(**arguments)


def test_cleanup_rejects_duplicate_provider_response_ids(tmp_path: Path) -> None:
    roster_path, _ = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    first_run = roster["accepted_primary_runs"]["canary_greedy_autoresearch"]
    second_run = roster["accepted_primary_runs"]["canary_semantic_autoresearch"]
    first_ledger = (
        tmp_path
        / "outputs/development/modal_downloads"
        / first_run
        / "controller/provider_attempts.jsonl"
    )
    second_ledger = (
        tmp_path
        / "outputs/development/modal_downloads"
        / second_run
        / "controller/provider_attempts.jsonl"
    )
    first = json.loads(first_ledger.read_text(encoding="utf-8"))
    second = json.loads(second_ledger.read_text(encoding="utf-8"))
    second["provider_response_id"] = first["provider_response_id"]
    second_ledger.write_text(
        json.dumps(second, sort_keys=True) + "\n", encoding="utf-8"
    )
    for outcome in roster["provider_canary_outcomes"]:
        if outcome["concrete_run_id"] == second_run:
            outcome["provider_attempt_ledger_sha256"] = hashlib.sha256(
                second_ledger.read_bytes()
            ).hexdigest()
    _write_json(roster_path, roster)
    _refresh_source_manifest_and_verifier(roster_path, second_run)

    with pytest.raises(ValueError, match="provider IDs are reused"):
        modal_readiness._derive_cleanup_claims(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
            recorded_at_utc="2025-01-02T00:00:00Z",
        )


def test_candidate_resume_preflight_is_provider_free_and_composes_strict_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execution(run_id: str, function: str, index: int):
        context = ExecutionContextV1(
            execution_backend="modal",
            run_id=run_id,
            app_name=APP_NAME,
            function_name=function,
            modal_app_id=f"ap-{index}",
            modal_function_id=f"fu-{index}",
            modal_call_id=f"fc-{index}",
            modal_image_id="im-shared",
            image_source_sha256="a" * 64,
            artifact_uri=volume_artifact_uri(run_id),
        )
        manifest = SimpleNamespace(
            image_source_sha256="a" * 64,
            manifest_sha256=f"{index}" * 64,
            files=(),
        )
        claim = {
            "run_id": run_id,
            "function_name": function,
            "modal_app_id": context.modal_app_id,
            "modal_function_id": context.modal_function_id,
            "modal_call_id": context.modal_call_id,
            "modal_image_id": context.modal_image_id,
            "image_source_sha256": "a" * 64,
            "artifact_manifest_sha256": manifest.manifest_sha256,
        }
        return claim, tmp_path / run_id, manifest, context

    executions = {
        "environment-run": execution(
            "environment-run", "cuda_environment", 3
        ),
        "offline-run": execution("offline-run", "offline_smoke", 4),
        "candidate-run": execution("candidate-run", "candidate_smoke", 1),
        "resume-run": execution("resume-run", "checkpoint_resume", 2),
    }
    monkeypatch.setattr(
        modal_readiness,
        "_execution_for_run",
        lambda root, run_id, expected_function: executions[run_id],
    )
    monkeypatch.setattr(
        modal_readiness,
        "_validate_candidate_layer_a",
        lambda *args, **kwargs: ({"layer_a": "valid"}, [{"path": "one"}]),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_revalidate_resume_attempt",
        lambda *args, **kwargs: (
            {"resume": "valid"},
            [{"path": "two"}, {"path": "three"}],
        ),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_remote_download_evidence",
        lambda root, run_id, manifest, **_kwargs: {
            "run_id": run_id,
            "verifier_run_id": f"verify-{run_id}",
            "verified": True,
        },
    )
    monkeypatch.setattr(
        modal_readiness,
        "_receipt_evidence",
        lambda root, gate_name, identity: (
            {"path": f"{gate_name}.json", "sha256": "f" * 64},
            (
                {
                    "run_id": "environment-run",
                    "artifact_manifest_sha256": "3" * 64,
                }
                if gate_name == "modal_cuda_environment_validated"
                    else {
                        "source_run_id": "candidate-run",
                        "verifier_run_id": "verify-candidate-run",
                        "verifier_attempt_id": "3" * 32,
                        "local_canonical_manifest_sha256": "1" * 64,
                }
            ),
        ),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_offline_smoke_receipt_evidence",
        lambda root, identity: (
            {
                "path": modal_readiness.MODAL_OFFLINE_SMOKE_VALIDATION_RECEIPT_PATH,
                "sha256": "9" * 64,
            },
                {
                    "run_id": "offline-run",
                    "remote_verifier_run_id": "verify-offline-run",
                    "remote_verifier_attempt_id": "2" * 32,
                    "artifact_manifest_sha256": "4" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        modal_readiness,
        "validate_downloaded_offline_bundle",
        lambda root: {
            "verified": True,
            "network_calls": 0,
            "provider_calls": 0,
            "modal_run_id": "offline-run",
            "artifact_manifest_sha256": "4" * 64,
            "validation_sha256": "e" * 64,
            "study": {"study_id": "offline-study", "run_count": 4},
        },
    )
    monkeypatch.setattr(
        modal_readiness,
        "_preflight_image_binding",
        lambda root, executions: {
            "image_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "modal_image_id": "im-shared",
        },
    )
    selected_identity = ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256="a" * 64,
        cohort_id="preflight-cohort",
    )
    monkeypatch.setattr(
        modal_readiness,
        "_identity_for_recording",
        lambda **_kwargs: selected_identity,
    )
    verifier_bindings = {
        "cuda_environment": {
            "verifier_run_id": "verify-environment-run",
            "verifier_attempt_id": "1" * 32,
        },
        "offline_smoke": {
            "verifier_run_id": "verify-offline-run",
            "verifier_attempt_id": "2" * 32,
        },
        "candidate_smoke": {
            "verifier_run_id": "verify-candidate-run",
            "verifier_attempt_id": "3" * 32,
        },
        "resume_attempt": {
            "verifier_run_id": "verify-resume-run",
            "verifier_attempt_id": "4" * 32,
        },
    }

    report = modal_readiness.validate_candidate_resume_preflight(
        environment_run_id="environment-run",
        offline_run_id="offline-run",
        candidate_run_id="candidate-run",
        resume_run_id="resume-run",
        cohort_id=selected_identity.cohort_id,
        verifier_bindings=verifier_bindings,
        root=tmp_path,
    )

    assert report["valid"] is True


    assert report["validation_mode"] == "local_read_only_provider_free"
    assert report["required_artifact_count"] == 3
    assert report["remote_calls_started"] == 0
    assert report["provider_calls_started"] == 0
    assert report["training_runs_started"] == 0
    assert report["evidence"]["candidate_smoke"] == {"layer_a": "valid"}
    assert report["evidence"]["resume_attempt"] == {"resume": "valid"}

    with pytest.raises(ValueError, match="four distinct early run IDs"):
        modal_readiness.validate_candidate_resume_preflight(
            environment_run_id="environment-run",
            offline_run_id="offline-run",
            candidate_run_id="candidate-run",
            resume_run_id="candidate-run",
            cohort_id=selected_identity.cohort_id,
            verifier_bindings=verifier_bindings,
            root=tmp_path,
        )

    executions["resume-run"] = execution(
        "resume-run",
        "checkpoint_resume",
        1,
    )
    with pytest.raises(ValueError, match="four unique Modal call IDs"):
        modal_readiness.validate_candidate_resume_preflight(
            environment_run_id="environment-run",
            offline_run_id="offline-run",
            candidate_run_id="candidate-run",
            resume_run_id="resume-run",
            cohort_id=selected_identity.cohort_id,
            verifier_bindings=verifier_bindings,
            root=tmp_path,
        )


def test_offline_smoke_receipt_is_create_only_and_reopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_fields = modal_readiness._OFFLINE_SMOKE_VALIDATION_FIELDS - {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
    }
    claims: dict[str, object] = {
        field: f"bound-{field}" for field in claim_fields
    }
    claims.update(
        {
            "run_id": "offline-receipt-run",
            "validation_network_calls": 0,
            "validation_provider_calls": 0,
            "validation_remote_calls_started": 0,
            "validation_training_runs_started": 0,
            "validated": True,
        }
    )
    derivations: list[str] = []

    def derive(
        root: Path,
        *,
        run_id: str,
        identity: ModalLiveCohortIdentity,
        verifier_run_id: str,
        verifier_attempt_id: str,
    ) -> dict[str, object]:
        derivations.append(run_id)
        return dict(claims)

    monkeypatch.setattr(
        modal_readiness,
        "_derive_offline_smoke_validation_claims",
        derive,
    )
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256="a" * 64,
        cohort_id="offline-receipt-cohort",
    )
    raw_manifest = SimpleNamespace(
        manifest=SimpleNamespace(image_source_sha256=identity.image_source_sha256)
    )
    monkeypatch.setattr(
        modal_readiness,
        "_inspect_downloaded_run_raw",
        lambda *args, **kwargs: (tmp_path, raw_manifest, None, None),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_identity_for_recording",
        lambda **_kwargs: identity,
    )
    verifier_run_id = "verify-offline-receipt-run"
    verifier_attempt_id = "5" * 32

    receipt = modal_readiness.record_offline_smoke_validation(
        run_id="offline-receipt-run",
        cohort_id=identity.cohort_id,
        verifier_run_id=verifier_run_id,
        verifier_attempt_id=verifier_attempt_id,
        root=tmp_path,
    )

    assert derivations == ["offline-receipt-run", "offline-receipt-run"]
    assert receipt == modal_readiness.validate_offline_smoke_validation_receipt(
        tmp_path
        / modal_readiness.modal_component_receipt_path(
            identity,
            "modal_offline_smoke_validated",
        ),
        root=tmp_path
    )
    with pytest.raises((FileExistsError, ValueError)):
        modal_readiness.record_offline_smoke_validation(
            run_id="offline-receipt-run",
            cohort_id=identity.cohort_id,
            verifier_run_id=verifier_run_id,
            verifier_attempt_id=verifier_attempt_id,
            root=tmp_path,
        )


def test_offline_smoke_creator_rejects_post_write_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_fields = modal_readiness._OFFLINE_SMOKE_VALIDATION_FIELDS - {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
    }
    claims: dict[str, object] = {
        field: f"bound-{field}" for field in claim_fields
    }
    claims.update(
        {
            "run_id": "offline-tamper-run",
            "validation_network_calls": 0,
            "validation_provider_calls": 0,
            "validation_remote_calls_started": 0,
            "validation_training_runs_started": 0,
            "validated": True,
        }
    )
    monkeypatch.setattr(
        modal_readiness,
        "_derive_offline_smoke_validation_claims",
        lambda root, *, run_id, identity, verifier_run_id, verifier_attempt_id: dict(
            claims
        ),
    )
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256="a" * 64,
        cohort_id="offline-tamper-cohort",
    )
    raw_manifest = SimpleNamespace(
        manifest=SimpleNamespace(image_source_sha256=identity.image_source_sha256)
    )
    monkeypatch.setattr(
        modal_readiness,
        "_inspect_downloaded_run_raw",
        lambda *args, **kwargs: (tmp_path, raw_manifest, None, None),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_identity_for_recording",
        lambda **_kwargs: identity,
    )
    create = modal_readiness.create_json_exclusive

    def create_then_tamper(path: Path, payload: object) -> None:
        create(path, payload)
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["validated"] = False
        _write_json(path, changed)

    monkeypatch.setattr(
        modal_readiness,
        "create_json_exclusive",
        create_then_tamper,
    )

    with pytest.raises(ValueError, match="differs from live artifacts"):
        modal_readiness.record_offline_smoke_validation(
            run_id="offline-tamper-run",
            cohort_id=identity.cohort_id,
            verifier_run_id="verify-offline-tamper-run",
            verifier_attempt_id="6" * 32,
            root=tmp_path,
        )


def test_resume_progression_staging_copies_exact_bound_roster(tmp_path: Path) -> None:
    attempt_root = tmp_path / "resume-run"
    training_relative = "candidate_smoke/seed_1"
    training = attempt_root / training_relative
    for filename in modal_readiness._RESUME_PROGRESSION_ROOT_FILES:
        path = attempt_root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"root:{filename}\n", encoding="utf-8")
    for filename in modal_readiness._RESUME_PROGRESSION_TRAINING_FILES:
        path = training / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"training:{filename}\n".encode())

    staging_root, staged_training = (
        modal_readiness._stage_resume_progression_inputs(
            attempt_root=attempt_root,
            staging_parent=tmp_path / "staging",
            training_relative=training_relative,
        )
    )

    assert staging_root.name == attempt_root.name
    assert {path.name for path in staged_training.iterdir()} == set(
        modal_readiness._RESUME_PROGRESSION_TRAINING_FILES
    )
    assert {
        path.name for path in staging_root.iterdir() if path.is_file()
    } == set(modal_readiness._RESUME_PROGRESSION_ROOT_FILES)
    for filename in modal_readiness._RESUME_PROGRESSION_TRAINING_FILES:
        assert (staged_training / filename).read_bytes() == (
            training / filename
        ).read_bytes()


@pytest.mark.parametrize(
    ("missing_scope", "missing_name"),
    (
        ("root", "image_source_manifest.json"),
        ("training", "training_manifest.json"),
        ("training", "rng_restore_attestation.json"),
    ),
)
def test_resume_progression_staging_requires_context_manifest_and_attestation(
    tmp_path: Path,
    missing_scope: str,
    missing_name: str,
) -> None:
    attempt_root = tmp_path / "resume-run"
    training_relative = "candidate_smoke/seed_1"
    training = attempt_root / training_relative
    for filename in modal_readiness._RESUME_PROGRESSION_ROOT_FILES:
        if missing_scope != "root" or filename != missing_name:
            path = attempt_root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
    for filename in modal_readiness._RESUME_PROGRESSION_TRAINING_FILES:
        if missing_scope != "training" or filename != missing_name:
            path = training / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n", encoding="utf-8")

    with pytest.raises(ValueError, match="resume revalidation input is unsafe"):
        modal_readiness._stage_resume_progression_inputs(
            attempt_root=attempt_root,
            staging_parent=tmp_path / "staging",
            training_relative=training_relative,
        )


def test_candidate_resume_preflight_rejects_one_mismatched_image_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        ("environment-run", "cuda_environment"),
        ("offline-run", "offline_smoke"),
        ("candidate-run", "candidate_smoke"),
        ("resume-run", "checkpoint_resume"),
    )
    executions = {
        run_id: _execution_stub(
            tmp_path,
            run_id=run_id,
            function_name=function_name,
            index=index,
            image_source_sha256=("b" * 64 if run_id == "environment-run" else "a" * 64),
        )
        for index, (run_id, function_name) in enumerate(specs, start=1)
    }
    monkeypatch.setattr(
        modal_readiness,
        "_execution_for_run",
        lambda root, run_id, expected_function: executions[run_id],
    )
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256="b" * 64,
        cohort_id="mismatched-preflight",
    )
    monkeypatch.setattr(
        modal_readiness,
        "_identity_for_recording",
        lambda **_kwargs: identity,
    )
    verifier_bindings = {
        label: {
            "verifier_run_id": f"verify-{run_id}",
            "verifier_attempt_id": f"{index:x}" * 32,
        }
        for index, (label, run_id) in enumerate(
            (
                ("cuda_environment", "environment-run"),
                ("offline_smoke", "offline-run"),
                ("candidate_smoke", "candidate-run"),
                ("resume_attempt", "resume-run"),
            ),
            start=1,
        )
    }

    with pytest.raises(ValueError, match="early runs do not share one image source"):
        modal_readiness.validate_candidate_resume_preflight(
            environment_run_id="environment-run",
            offline_run_id="offline-run",
            candidate_run_id="candidate-run",
            resume_run_id="resume-run",
            cohort_id=identity.cohort_id,
            verifier_bindings=verifier_bindings,
            root=tmp_path,
        )


def test_frozen_bundle_image_binding_recomputes_source_lock_and_shared_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=(
            SourceFileV1("uv.lock", "a" * 64, 1),
            SourceFileV1("worker.py", "b" * 64, 2),
        ),
    )
    executions = _image_binding_inputs(tmp_path, manifest)
    monkeypatch.setattr(
        modal_readiness,
        "build_image_source_manifest",
        lambda root: manifest,
    )

    binding = modal_readiness._frozen_image_source_binding(tmp_path, executions)

    assert binding == {
        "modal_image_id": "im-shared",
        "image_source_sha256": manifest.manifest_sha256,
        "dependency_lock_sha256": "a" * 64,
        "image_source_file_count": 2,
        "image_source_byte_count": 3,
    }


@pytest.mark.parametrize(
    ("image_ids", "message"),
    (
        (("im-other",) + ("im-shared",) * 7, "does not share one Modal image ID"),
        ((None,) + ("im-shared",) * 7, "lacks a nonempty Modal image ID"),
    ),
)
def test_frozen_bundle_image_binding_rejects_mixed_or_missing_image_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_ids: tuple[str | None, ...],
    message: str,
) -> None:
    manifest = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=(SourceFileV1("uv.lock", "a" * 64, 1),),
    )
    executions = _image_binding_inputs(
        tmp_path,
        manifest,
        image_ids=image_ids,
    )
    monkeypatch.setattr(
        modal_readiness,
        "build_image_source_manifest",
        lambda root: manifest,
    )

    with pytest.raises(ValueError, match=message):
        modal_readiness._frozen_image_source_binding(tmp_path, executions)


def test_frozen_bundle_image_binding_rejects_current_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=(SourceFileV1("uv.lock", "a" * 64, 1),),
    )
    current = ImageSourceManifestV1(
        dependency_lock_sha256="b" * 64,
        files=(SourceFileV1("uv.lock", "b" * 64, 2),),
    )
    executions = _image_binding_inputs(tmp_path, downloaded)
    monkeypatch.setattr(
        modal_readiness,
        "build_image_source_manifest",
        lambda root: current,
    )

    with pytest.raises(ValueError, match="differs from the current source tree"):
        modal_readiness._frozen_image_source_binding(tmp_path, executions)


@pytest.mark.parametrize("mismatch", ("dependency", "source-file"))
def test_frozen_bundle_image_binding_rejects_downloaded_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    current = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=(
            SourceFileV1("uv.lock", "a" * 64, 1),
            SourceFileV1("worker.py", "b" * 64, 2),
        ),
    )
    executions = _image_binding_inputs(tmp_path, current)
    if mismatch == "dependency":
        changed = ImageSourceManifestV1(
            dependency_lock_sha256="c" * 64,
            files=(
                SourceFileV1("uv.lock", "c" * 64, 1),
                SourceFileV1("worker.py", "b" * 64, 2),
            ),
        )
        message = "dependency lock differs from current source"
    else:
        changed = ImageSourceManifestV1(
            dependency_lock_sha256="a" * 64,
            files=(
                SourceFileV1("uv.lock", "a" * 64, 1),
                SourceFileV1("worker.py", "c" * 64, 2),
            ),
        )
        message = "image source manifest differs from the current source tree"
    first_root, _, _ = executions["execution_0"]
    _write_json(first_root / "image_source_manifest.json", changed.to_dict())
    monkeypatch.setattr(
        modal_readiness,
        "build_image_source_manifest",
        lambda root: current,
    )

    with pytest.raises(ValueError, match=message):
        modal_readiness._frozen_image_source_binding(tmp_path, executions)


def test_migration_bundle_derivation_rejects_mixed_environment_image_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary_run_ids = {
        "greedy_autoresearch": "canary-bundle-greedy-ar",
        "semantic_autoresearch": "canary-bundle-semantic-ar",
        "openevolve_generic": "canary-bundle-openevolve-generic",
        "openevolve_semantic": "canary-bundle-openevolve-semantic",
    }
    specs = (
        ("environment-run", "cuda_environment"),
        ("offline-run", "offline_smoke"),
        ("candidate-run", "candidate_smoke"),
        ("resume-run", "checkpoint_resume"),
        *(
            (run_id, f"canary_{harness}")
            for harness, run_id in canary_run_ids.items()
        ),
    )
    executions = {
        run_id: _execution_stub(
            tmp_path,
            run_id=run_id,
            function_name=function_name,
            index=index,
            image_source_sha256=("b" * 64 if run_id == "environment-run" else "a" * 64),
        )
        for index, (run_id, function_name) in enumerate(specs, start=1)
    }
    monkeypatch.setattr(
        modal_readiness,
        "_execution_for_run",
        lambda root, run_id, expected_function: executions[run_id],
    )

    def load_environment_artifact(path: Path) -> dict[str, object]:
        if path.name == "remote_action_result.json":
            return {
                "success": True,
                "mode": "cuda_environment",
                "observed_gpu": "Tesla T4",
            }
        if path.name == "cuda_environment.json":
            return {"cuda_device_name": "Tesla T4"}
        raise AssertionError(f"unexpected artifact read before image check: {path}")

    monkeypatch.setattr(modal_readiness, "_load_object", load_environment_artifact)
    roster_path = tmp_path / "cohort_roster.json"
    roster_path.write_text("{}\n", encoding="utf-8")
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="9" * 64,
        image_source_sha256="a" * 64,
        cohort_id="mixed-bundle",
    )
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text("{}\n", encoding="utf-8")
    lineage_sha256 = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        modal_readiness,
        "_load_cohort_roster",
        lambda *args, **kwargs: (
            {
                **modal_readiness.modal_cohort_identity_dict(identity),
                "migration_lineage_path": "lineage.json",
                "migration_lineage_sha256": lineage_sha256,
                "accepted_primary_runs": {
                    "cuda_environment": "environment-run",
                    "offline_smoke": "offline-run",
                    "candidate_smoke": "candidate-run",
                    "resume_attempt": "resume-run",
                    **{
                        f"canary_{harness}": run_id
                        for harness, run_id in canary_run_ids.items()
                    },
                }
            },
            roster_path,
        ),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_migration_lineage",
        lambda *args, **kwargs: ({}, lineage_path, lineage_sha256),
    )

    with pytest.raises(
        ValueError,
        match="migration bundle does not share one image source digest",
    ):
        modal_readiness._derive_migration_bundle_claims(
            tmp_path,
            cohort_roster_path="cohort_roster.json",
        )


def test_invalid_prior_quarantine_accounting_creates_no_canonical_receipt(
    tmp_path: Path,
) -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="invalid-prior-cohort",
    )
    payload = {
        "schema_name": "ModalPriorCohortQuarantineAccounting",
        "schema_version": "1.0",
        **modal_readiness.modal_cohort_identity_dict(identity),
    }
    output = tmp_path / modal_readiness.modal_prior_quarantine_accounting_path(
        identity
    )

    with pytest.raises(ValueError, match="invalid exact schema"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )

    assert not output.exists()


def test_prior_quarantine_accounting_1_1_validates_publishes_and_reloads(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)

    persisted = modal_readiness.create_prior_quarantine_accounting(
        payload=payload,
        root=tmp_path,
    )

    assert persisted == payload
    output = tmp_path / logical
    assert output.stat().st_mode & 0o777 == 0o600
    reloaded, metadata = modal_readiness._load_prior_quarantine_accounting(
        tmp_path,
        logical,
    )
    assert reloaded == payload
    assert metadata["raw_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    output.chmod(0o644)
    with pytest.raises(ValueError, match="exactly 0600"):
        modal_readiness._load_prior_quarantine_accounting(tmp_path, logical)


def test_prior_accounting_publisher_rejects_validated_false(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    payload["validated"] = False

    with pytest.raises(
        ValueError,
        match="prior_quarantine.validated must be exactly True",
    ):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )

    assert not (tmp_path / logical).exists()


def test_prior_accounting_loader_rejects_validated_false(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    payload["validated"] = False
    receipt_path = tmp_path / logical
    _write_json(receipt_path, payload)
    receipt_path.chmod(0o600)

    with pytest.raises(
        ValueError,
        match="prior_quarantine.validated must be exactly True",
    ):
        modal_readiness._load_prior_quarantine_accounting(
            tmp_path,
            logical,
        )


def test_prior_accounting_template_inspect_scaffold_publish_end_to_end(
    tmp_path: Path,
) -> None:
    expected, logical = _prior_quarantine_accounting_fixture(tmp_path)
    request_path = tmp_path / "operator" / "prior-request.json"
    scaffold_path = tmp_path / "operator" / "prior-candidate.json"
    snapshot_path = tmp_path / expected["snapshot_capture_manifest_path"]

    request = modal_readiness.create_prior_quarantine_accounting_template(
        source_tree_sha256_value=expected["source_tree_sha256"],
        image_source_sha256=expected["image_source_sha256"],
        cohort_id=expected["cohort_id"],
        recorded_at_utc=expected["recorded_at_utc"],
        snapshot_capture_manifest_path=snapshot_path,
        output_path=request_path,
        root=tmp_path,
    )

    assert request_path.stat().st_mode & 0o777 == 0o600
    assert request["snapshot_capture_manifest"] == {
        "path": expected["snapshot_capture_manifest_path"],
        "sha256": expected["snapshot_capture_manifest_sha256"],
        "size_bytes": expected["snapshot_capture_manifest_size_bytes"],
    }
    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )
    assert inspection["blockers"] == []
    assert inspection["candidate"] == expected
    assert inspection["canonical_receipt_path"] == (tmp_path / logical).as_posix()
    assert not (tmp_path / logical).exists()

    scaffold = modal_readiness.scaffold_prior_quarantine_accounting(
        request=request,
        output_path=scaffold_path,
        root=tmp_path,
    )
    assert scaffold == expected
    assert scaffold_path.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / logical).exists()

    persisted = modal_readiness.create_prior_quarantine_accounting(
        payload=modal_readiness._load_operator_json_input(scaffold_path),
        root=tmp_path,
    )
    reloaded, _metadata = modal_readiness._load_prior_quarantine_accounting(
        tmp_path,
        logical,
    )
    assert persisted == expected
    assert reloaded == expected


@pytest.mark.parametrize(
    "volume_rows",
    (
        [],
        [
            {
                "name": "different-artifact-volume",
                "created_at": "2025-01-01 00:00:00+00:00",
                "created_by": "test-user",
            }
        ],
    ),
    ids=("missing", "wrong-name"),
)
def test_prior_accounting_missing_artifact_volume_is_incomplete_and_unpublishable(
    tmp_path: Path,
    volume_rows: list[dict[str, str]],
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    _write_json(manifest_path.parent / "volume_list.json", volume_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )

    assert inspection["candidate"] is None
    assert inspection["blockers"] == [
        {
            "code": "incomplete_repository_evidence",
            "message": "selected prior snapshot lacks the artifact Volume",
        }
    ]
    with pytest.raises(ValueError, match="lacks the artifact Volume"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_duplicate_artifact_volume_is_an_integrity_error(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    volume_row = {
        "name": modal_readiness.VOLUME_NAME,
        "created_at": "2025-01-01 00:00:00+00:00",
        "created_by": "test-user",
    }
    _write_json(
        manifest_path.parent / "volume_list.json",
        [volume_row, {**volume_row, "created_by": "second-test-user"}],
    )
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="repeats the artifact Volume"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="repeats the artifact Volume"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_preserves_overlapping_migration_app_and_billing(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    snapshot_root = manifest_path.parent
    extra_app_id = "ap-unattributed-prior-migration"
    app_rows = json.loads((snapshot_root / "app_list.json").read_text())
    app_rows.append(
        {
            "app_id": extra_app_id,
            "description": APP_NAME,
            "state": "stopped",
            "tasks": "0",
            "created_at": "2025-01-01 00:02:00+00:00",
            "stopped_at": "2025-01-01 00:19:00+00:00",
        }
    )
    _write_json(snapshot_root / "app_list.json", app_rows)
    billing_rows = json.loads((snapshot_root / "billing_report.json").read_text())
    billing_rows.append(
        {
            "object_id": extra_app_id,
            "description": APP_NAME,
            "environment": "main",
            "interval_start": "2025-01-01T00:00:00+00:00",
            "resource": "CPU",
            "cost": "0.01",
        }
    )
    _write_json(snapshot_root / "billing_report.json", billing_rows)
    run_rows = json.loads(
        (snapshot_root / "run_directory_list.json").read_text()
    )
    run_rows.append(
        {
            "filename": "/runs/unrelated-unreserved-run",
            "type": "dir",
            "created_modified": "2025-01-01 00:30 UTC",
            "size": "0 B",
        }
    )
    _write_json(snapshot_root / "run_directory_list.json", run_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )

    assert inspection["blockers"] == []
    assert inspection["candidate"] == payload
    modal_readiness.create_prior_quarantine_accounting(
        payload=payload,
        root=tmp_path,
    )
    _persisted, metadata = modal_readiness._load_prior_quarantine_accounting(
        tmp_path,
        logical,
    )
    extra_billing_key = modal_readiness.canonical_sha256(billing_rows[-1])
    assert extra_app_id in metadata["observed_migration_app_ids"]
    assert extra_app_id not in metadata["app_ids"]
    assert metadata["observed_migration_billing_rows"][extra_billing_key] == (
        billing_rows[-1]
    )
    assert extra_billing_key not in metadata["billing_row_keys"]
    assert "unrelated-unreserved-run" not in metadata["observed_volume_run_ids"]
    assert (
        "modal-cuda-env-20260809-02"
        not in metadata["observed_volume_run_ids"]
    )


def test_prior_accounting_preserves_unowned_migration_billing_for_lineage(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    billing_path = manifest_path.parent / "billing_report.json"
    billing_rows = json.loads(billing_path.read_text())
    extra_app_id = "ap-unattributed-prior-billing"
    billing_rows.append(
        {
            "object_id": extra_app_id,
            "description": APP_NAME,
            "environment": "main",
            "interval_start": "2025-01-01T00:00:00+00:00",
            "resource": "CPU",
            "cost": "0.01",
        }
    )
    _write_json(billing_path, billing_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )

    assert inspection["blockers"] == []
    assert inspection["candidate"] == payload
    modal_readiness.create_prior_quarantine_accounting(
        payload=payload,
        root=tmp_path,
    )
    _persisted, metadata = modal_readiness._load_prior_quarantine_accounting(
        tmp_path,
        logical,
    )
    extra_billing_key = modal_readiness.canonical_sha256(billing_rows[-1])
    assert metadata["observed_migration_billing_rows"][extra_billing_key] == (
        billing_rows[-1]
    )
    assert extra_billing_key not in metadata["billing_row_keys"]


def test_prior_accounting_ignores_unowned_app_name_billing_in_other_environment(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    billing_path = manifest_path.parent / "billing_report.json"
    billing_rows = json.loads(billing_path.read_text())
    unrelated_row = {
        "object_id": "ap-staging-unrelated",
        "description": APP_NAME,
        "environment": "staging",
        "interval_start": "2025-01-01T00:00:00+00:00",
        "resource": "CPU",
        "cost": "9.99",
    }
    billing_rows.append(unrelated_row)
    _write_json(billing_path, billing_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )
    assert inspection["blockers"] == []
    assert inspection["candidate"] == payload
    modal_readiness.create_prior_quarantine_accounting(
        payload=payload,
        root=tmp_path,
    )
    _persisted, metadata = modal_readiness._load_prior_quarantine_accounting(
        tmp_path,
        logical,
    )
    assert (
        modal_readiness.canonical_sha256(unrelated_row)
        not in metadata["observed_migration_billing_rows"]
    )


def test_prior_accounting_allows_unrelated_snapshot_descriptions(
    tmp_path: Path,
) -> None:
    payload, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    snapshot_root = manifest_path.parent
    unrelated_app_id = "ap-unrelated-workload"
    app_rows = json.loads((snapshot_root / "app_list.json").read_text())
    app_rows.append(
        {
            "app_id": unrelated_app_id,
            "description": "unrelated-workload",
            "state": "stopped",
            "tasks": "0",
            "created_at": "2025-01-01 00:02:00+00:00",
            "stopped_at": "2025-01-01 00:19:00+00:00",
        }
    )
    _write_json(snapshot_root / "app_list.json", app_rows)
    billing_rows = json.loads((snapshot_root / "billing_report.json").read_text())
    billing_rows.append(
        {
            "object_id": unrelated_app_id,
            "description": "unrelated-workload",
            "environment": "other",
            "interval_start": "2025-01-01T00:00:00+00:00",
            "resource": "CPU",
            "cost": "99.00",
        }
    )
    _write_json(snapshot_root / "billing_report.json", billing_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )

    assert inspection["blockers"] == []
    assert inspection["candidate"] == payload
    assert modal_readiness.create_prior_quarantine_accounting(
        payload=payload,
        root=tmp_path,
    ) == payload


@pytest.mark.parametrize("timestamp_field", ("created_at", "stopped_at"))
def test_prior_accounting_rejects_future_app_lifecycle_timestamps(
    tmp_path: Path,
    timestamp_field: str,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    app_path = manifest_path.parent / "app_list.json"
    app_rows = json.loads(app_path.read_text())
    app_rows[0][timestamp_field] = "2099-01-01 00:00:00+00:00"
    if timestamp_field == "created_at":
        app_rows[0]["stopped_at"] = "2099-01-01 00:01:00+00:00"
    _write_json(app_path, app_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="lifecycle timestamp exceeds"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="lifecycle timestamp exceeds"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_rejects_app_time_after_app_list_capture(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    app_path = manifest_path.parent / "app_list.json"
    app_rows = json.loads(app_path.read_text())
    app_rows[0]["stopped_at"] = "2025-01-01 01:00:45+00:00"
    _write_json(app_path, app_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="lifecycle timestamp exceeds"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="lifecycle timestamp exceeds"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("created_at", "2025-01-01 00:04:00+00:00"),
        ("stopped_at", "2025-01-01 00:07:00+00:00"),
    ),
    ids=("before-attempt", "after-attempt"),
)
def test_prior_accounting_requires_app_lifecycle_contained_by_attempt(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    app_path = manifest_path.parent / "app_list.json"
    app_rows = json.loads(app_path.read_text())
    app_rows[0][field] = value
    _write_json(app_path, app_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="lifecycle is not contained"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="lifecycle is not contained"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_rejects_future_artifact_volume_creation(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    volume_path = manifest_path.parent / "volume_list.json"
    volume_rows = json.loads(volume_path.read_text())
    volume_rows[0]["created_at"] = "2099-01-01 00:00:00+00:00"
    _write_json(volume_path, volume_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="Volume creation timestamp exceeds"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="Volume creation timestamp exceeds"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "created_modified",
            "2099-01-01 00:00 UTC",
            "runs entry timestamp exceeds",
        ),
        (
            "created_modified",
            "not-a-modal-timestamp",
            "timestamp is not Modal CLI output",
        ),
        (
            "created_modified",
            "2025-01-01 08:30 CST",
            "timestamp is not Modal CLI output",
        ),
        ("size", "00 B", "size is not Modal CLI output"),
    ),
)
def test_prior_accounting_rejects_future_or_malformed_volume_run_entry(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    run_path = manifest_path.parent / "run_directory_list.json"
    run_rows = json.loads(run_path.read_text())
    run_rows[0][field] = value
    _write_json(run_path, run_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match=message):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_rejects_volume_run_entry_predating_attempt(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    run_path = manifest_path.parent / "run_directory_list.json"
    run_rows = json.loads(run_path.read_text())
    owned_run_ids = {
        run_id
        for attempt in payload["attempt_dispositions"]
        for run_id in attempt["concrete_remote_run_ids"]
    }
    owned_row = next(
        row
        for row in run_rows
        if PurePosixPath(row["filename"]).name in owned_run_ids
    )
    owned_row["created_modified"] = "2025-01-01 00:03 UTC"
    _write_json(run_path, run_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="predates its launcher attempt"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="predates its launcher attempt"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


@pytest.mark.parametrize(
    "mutation",
    ({"cost": "123.45"}, {"description": "different-description"}),
    ids=("altered-cost", "altered-description"),
)
def test_prior_accounting_rejects_duplicate_billing_accounting_key(
    tmp_path: Path,
    mutation: dict[str, str],
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    billing_path = manifest_path.parent / "billing_report.json"
    billing_rows = json.loads(billing_path.read_text())
    billing_rows.append({**billing_rows[0], **mutation})
    _write_json(billing_path, billing_rows)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="duplicate accounting row"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="duplicate accounting row"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_rejects_empty_export_window_excluding_attempts(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    snapshot_root = manifest_path.parent
    roster_path = snapshot_root.parents[2] / "cohort_roster.v4.0.json"
    roster = json.loads(roster_path.read_text())
    roster["billing_window_start_utc"] = "2024-12-31T23:00:00Z"
    roster["billing_window_end_utc"] = "2025-01-01T00:00:00Z"
    _write_json(roster_path, roster)
    _write_json(snapshot_root / "billing_report.json", [])
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="attempt falls outside the billing window"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="attempt falls outside the billing window"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_rejects_overlong_snapshot_billing_window(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    manifest_path = tmp_path / str(payload["snapshot_capture_manifest_path"])
    roster_path = manifest_path.parent.parents[2] / "cohort_roster.v4.0.json"
    roster = json.loads(roster_path.read_text())
    roster["billing_window_start_utc"] = "2024-11-30T00:00:00Z"
    roster["billing_window_end_utc"] = "2025-01-01T00:00:00Z"
    _write_json(roster_path, roster)
    request = _refresh_prior_snapshot_binding(tmp_path, payload)

    with pytest.raises(ValueError, match="exceeds 31 days"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="exceeds 31 days"):
        modal_readiness.create_prior_quarantine_accounting(
            payload=payload,
            root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


@pytest.mark.parametrize("execution_kind", ("ordinary", "verifier"))
@pytest.mark.parametrize("bad_uri_kind", ("wrong-volume", "wrong-run"))
def test_prior_accounting_rejects_noncanonical_execution_artifact_uri(
    tmp_path: Path,
    execution_kind: str,
    bad_uri_kind: str,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    attempts = {
        item["attempt_id"]: item for item in payload["attempt_dispositions"]
    }
    target = next(
        record
        for record in payload["remote_executions"]
        if (
            attempts[record["attempt_id"]]["action"] in {"download", "verify"}
        )
        == (execution_kind == "verifier")
    )
    context_path = tmp_path / target["execution_context_path"]
    context = json.loads(context_path.read_text())
    if bad_uri_kind == "wrong-volume":
        context["artifact_uri"] = (
            f"volume://wrong-volume/runs/{context['run_id']}"
        )
    else:
        context["artifact_uri"] = volume_artifact_uri("modal-prior-wrong-run")
    _write_json(context_path, context)
    context_raw = context_path.read_bytes()
    target["execution_context_sha256"] = hashlib.sha256(context_raw).hexdigest()
    target["execution_context_size_bytes"] = len(context_raw)
    request = _prior_quarantine_accounting_request(payload)

    with pytest.raises(ValueError, match="artifact URI is not canonical"):
        modal_readiness.inspect_prior_quarantine_accounting(
            request=request,
            root=tmp_path,
        )
        with pytest.raises(
            modal_global_journal.ModalActionJournalIntegrityError,
            match="migration remote execution evidence byte binding changed",
        ):
            modal_readiness.create_prior_quarantine_accounting(
                payload=payload,
                root=tmp_path,
        )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_rejects_swapped_aggregate_child_functions(
    tmp_path: Path,
) -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="aggregate-child-identity",
    )
    run_prefix = "prior-aggregate-identity"
    child_run_ids = [
        f"{run_prefix}-{modal_readiness._CANARY_SUFFIXES[harness]}"
        for harness in modal_readiness.CANARY_ORDER
    ]
    attempt = {
        "attempt_id": "a" * 32,
        "action": "canaries",
        "run_id": run_prefix,
        "concrete_remote_run_ids": child_run_ids,
        "harness": None,
    }
    first_run_id, second_run_id = child_run_ids[:2]
    swapped = ExecutionContextV1(
        execution_backend="modal",
        run_id=first_run_id,
        app_name=APP_NAME,
        function_name=f"canary_{modal_readiness.CANARY_ORDER[1]}",
        modal_app_id="ap-aggregate-identity",
        modal_function_id="fu-aggregate-identity",
        modal_call_id="fc-aggregate-identity",
        modal_image_id="im-aggregate-identity",
        image_source_sha256=identity.image_source_sha256,
        artifact_uri=volume_artifact_uri(first_run_id),
    )
    context_path = (
        tmp_path
        / "outputs/development/modal_downloads"
        / first_run_id
        / "execution_context.json"
    )
    _write_json(context_path, swapped.to_dict())

    with pytest.raises(ValueError, match="function differs from its action"):
        modal_readiness._load_prior_execution_context(
            tmp_path,
            identity,
            attempt,
            first_run_id,
        )
    with pytest.raises(ValueError, match="function differs from its action"):
        modal_readiness._validate_prior_execution_context_identity(
            swapped,
            identity=identity,
            attempt=attempt,
            run_id=first_run_id,
        )
    assert second_run_id != first_run_id


def test_prior_accounting_operator_paths_fail_closed_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_fields = {
        "source_tree_sha256": "1" * 64,
        "image_source_sha256": "2" * 64,
        "cohort_id": "prior-path-safety",
    }
    absolute_snapshot = tmp_path / "capture_manifest.json"
    absolute_output = tmp_path / "request.json"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="lexically absolute"):
        modal_readiness.create_prior_quarantine_accounting_template(
            source_tree_sha256_value=identity_fields["source_tree_sha256"],
            image_source_sha256=identity_fields["image_source_sha256"],
            cohort_id=identity_fields["cohort_id"],
            recorded_at_utc="2025-01-01T02:00:00Z",
            snapshot_capture_manifest_path="capture_manifest.json",
            output_path=absolute_output,
            root=tmp_path,
        )
    assert not absolute_output.exists()

    with pytest.raises(ValueError, match="lexically absolute"):
        modal_readiness.create_prior_quarantine_accounting_template(
            source_tree_sha256_value=identity_fields["source_tree_sha256"],
            image_source_sha256=identity_fields["image_source_sha256"],
            cohort_id=identity_fields["cohort_id"],
            recorded_at_utc="2025-01-01T02:00:00Z",
            snapshot_capture_manifest_path=absolute_snapshot,
            output_path="request.json",
            root=tmp_path,
        )
    assert not absolute_output.exists()

    request = {
        "schema_name": "ModalPriorCohortQuarantineAccountingRequest",
        "schema_version": "1.0",
        **identity_fields,
        "recorded_at_utc": "2025-01-01T02:00:00Z",
        "snapshot_capture_manifest": {
            "path": "outputs/capture_manifest.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
        },
    }
    with pytest.raises(ValueError, match="lexically absolute"):
        modal_readiness.scaffold_prior_quarantine_accounting(
            request=request,
            output_path="candidate.json",
            root=tmp_path,
        )
    assert not (tmp_path / "candidate.json").exists()
    identity = ModalLiveCohortIdentity(**identity_fields)
    canonical = tmp_path / modal_readiness.modal_prior_quarantine_accounting_path(
        identity
    )
    with pytest.raises(ValueError, match="canonical live-cohort receipt namespace"):
        modal_readiness.scaffold_prior_quarantine_accounting(
            request=request,
            output_path=canonical,
            root=tmp_path,
        )
    assert not canonical.exists()

    project = tmp_path / "project"
    project.mkdir()
    outside_snapshot = tmp_path / "outside-snapshot.json"
    _write_json(outside_snapshot, {"snapshot": True})
    outside_snapshot.chmod(0o600)
    outside_output = tmp_path / "outside-request.json"
    with pytest.raises(ValueError, match="authenticated project root"):
        modal_readiness.create_prior_quarantine_accounting_template(
            source_tree_sha256_value=identity_fields["source_tree_sha256"],
            image_source_sha256=identity_fields["image_source_sha256"],
            cohort_id=identity_fields["cohort_id"],
            recorded_at_utc="2025-01-01T02:00:00Z",
            snapshot_capture_manifest_path=outside_snapshot,
            output_path=outside_output,
            root=project,
        )
    assert not outside_output.exists()


def test_prior_accounting_operator_outputs_reject_casefolded_canonical_aliases(
    tmp_path: Path,
) -> None:
    expected, logical = _prior_quarantine_accounting_fixture(tmp_path)
    canonical = tmp_path / logical
    alias = canonical.with_name(canonical.name.upper())
    snapshot = tmp_path / expected["snapshot_capture_manifest_path"]

    with pytest.raises(
        ValueError,
        match="canonical live-cohort receipt namespace",
    ):
        modal_readiness.create_prior_quarantine_accounting_template(
            source_tree_sha256_value=expected["source_tree_sha256"],
            image_source_sha256=expected["image_source_sha256"],
            cohort_id=expected["cohort_id"],
            recorded_at_utc=expected["recorded_at_utc"],
            snapshot_capture_manifest_path=snapshot,
            output_path=alias,
            root=tmp_path,
        )
    assert not canonical.exists()

    with pytest.raises(
        ValueError,
        match="canonical live-cohort receipt namespace",
    ):
        modal_readiness.scaffold_prior_quarantine_accounting(
            request=_prior_quarantine_accounting_request(expected),
            output_path=alias,
            root=tmp_path,
        )
    assert not canonical.exists()


def test_prior_accounting_template_rejects_snapshot_links_and_existing_outputs(
    tmp_path: Path,
) -> None:
    expected, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    snapshot = tmp_path / expected["snapshot_capture_manifest_path"]
    arguments = {
        "source_tree_sha256_value": expected["source_tree_sha256"],
        "image_source_sha256": expected["image_source_sha256"],
        "cohort_id": expected["cohort_id"],
        "recorded_at_utc": expected["recorded_at_utc"],
        "root": tmp_path,
    }

    symlink_snapshot = tmp_path / "snapshot-link.json"
    symlink_snapshot.symlink_to(snapshot)
    symlink_output = tmp_path / "symlink-request.json"
    with pytest.raises(ValueError, match="symbolic link"):
        modal_readiness.create_prior_quarantine_accounting_template(
            **arguments,
            snapshot_capture_manifest_path=symlink_snapshot,
            output_path=symlink_output,
        )
    assert not symlink_output.exists()

    hardlink_snapshot = tmp_path / "snapshot-hardlink.json"
    os.link(snapshot, hardlink_snapshot)
    hardlink_output = tmp_path / "hardlink-request.json"
    with pytest.raises(ValueError, match="one regular file"):
        modal_readiness.create_prior_quarantine_accounting_template(
            **arguments,
            snapshot_capture_manifest_path=hardlink_snapshot,
            output_path=hardlink_output,
        )
    assert not hardlink_output.exists()
    hardlink_snapshot.unlink()

    noncanonical_snapshot = tmp_path / "copied-capture-manifest.json"
    noncanonical_snapshot.write_bytes(snapshot.read_bytes())
    noncanonical_snapshot.chmod(0o600)
    noncanonical_output = tmp_path / "noncanonical-request.json"
    with pytest.raises(ValueError, match="path is not canonical"):
        modal_readiness.create_prior_quarantine_accounting_template(
            **arguments,
            snapshot_capture_manifest_path=noncanonical_snapshot,
            output_path=noncanonical_output,
        )
    assert not noncanonical_output.exists()

    existing_output = tmp_path / "existing-request.json"
    original = b'{"preserved":true}\n'
    existing_output.write_bytes(original)
    existing_output.chmod(0o600)
    with pytest.raises(FileExistsError):
        modal_readiness.create_prior_quarantine_accounting_template(
            **arguments,
            snapshot_capture_manifest_path=snapshot,
            output_path=existing_output,
        )
    assert existing_output.read_bytes() == original

    output_target = tmp_path / "output-target.json"
    _write_json(output_target, {"preserved": True})
    linked_output = tmp_path / "linked-output.json"
    linked_output.symlink_to(output_target)
    with pytest.raises(FileExistsError):
        modal_readiness.create_prior_quarantine_accounting_template(
            **arguments,
            snapshot_capture_manifest_path=snapshot,
            output_path=linked_output,
        )
    assert json.loads(output_target.read_text()) == {"preserved": True}


def test_prior_accounting_snapshot_identity_mismatch_creates_no_scaffold(
    tmp_path: Path,
) -> None:
    expected, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    request = _prior_quarantine_accounting_request(expected)
    request["cohort_id"] = "different-prior-cohort"
    output = tmp_path / "candidate.json"

    with pytest.raises(ValueError, match="selected Modal live cohort"):
        modal_readiness.scaffold_prior_quarantine_accounting(
            request=request,
            output_path=output,
            root=tmp_path,
        )

    assert not output.exists()


def test_prior_accounting_inspect_reports_incomplete_evidence_without_writing(
    tmp_path: Path,
) -> None:
    expected, logical = _prior_quarantine_accounting_fixture(tmp_path)
    request = _prior_quarantine_accounting_request(expected)
    offline_attempt_id = next(
        item["attempt_id"]
        for item in expected["attempt_dispositions"]
        if item["action"] == "offline-smoke"
    )
    context = next(
        item
        for item in expected["remote_executions"]
        if item["attempt_id"] == offline_attempt_id
    )
    (tmp_path / context["execution_context_path"]).unlink()

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )

    assert inspection["candidate"] is None
    assert inspection["blockers"]
    assert all(
        blocker["code"] == "incomplete_repository_evidence"
        for blocker in inspection["blockers"]
    )
    assert not (tmp_path / logical).exists()


def test_prior_accounting_scaffold_detects_source_mutation_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    request = _prior_quarantine_accounting_request(expected)
    output = tmp_path / "candidate.json"
    source_path = tmp_path / expected["action_journal"]["intent_receipts"][0][
        "path"
    ]
    real_derive = modal_readiness._derive_prior_quarantine_accounting_candidate
    calls = 0

    def mutate_after_first_derivation(*args, **kwargs):
        nonlocal calls
        candidate = real_derive(*args, **kwargs)
        calls += 1
        if calls == 1:
            source = json.loads(source_path.read_text(encoding="utf-8"))
            source["cohort_id"] = "tampered-source-cohort"
            _write_json(source_path, source)
        return candidate

    monkeypatch.setattr(
        modal_readiness,
        "_derive_prior_quarantine_accounting_candidate",
        mutate_after_first_derivation,
    )
    with pytest.raises(ValueError):
        modal_readiness.scaffold_prior_quarantine_accounting(
            request=request,
            output_path=output,
            root=tmp_path,
        )

    assert calls == 1
    assert not output.exists()


def test_prior_accounting_operator_workflows_fail_on_global_lock_contention(
    tmp_path: Path,
) -> None:
    from common.modal_action_lock import ModalActionLockContentionError

    expected, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    request = _prior_quarantine_accounting_request(expected)
    output = tmp_path / "candidate.json"
    held = modal_readiness.acquire_modal_action_lock(tmp_path)
    try:
        with pytest.raises(ModalActionLockContentionError):
            modal_readiness.inspect_prior_quarantine_accounting(
                request=request,
                root=tmp_path,
            )
        with pytest.raises(ModalActionLockContentionError):
            modal_readiness.scaffold_prior_quarantine_accounting(
                request=request,
                output_path=output,
                root=tmp_path,
            )
    finally:
        modal_readiness.release_modal_action_lock(held)
    assert not output.exists()


def test_prior_accounting_derivation_does_not_require_cohort_roster(
    tmp_path: Path,
) -> None:
    expected, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    request = _prior_quarantine_accounting_request(expected)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=expected["source_tree_sha256"],
        image_source_sha256=expected["image_source_sha256"],
        cohort_id=expected["cohort_id"],
    )
    (tmp_path / modal_readiness.modal_cohort_roster_path(identity)).unlink()
    (tmp_path / modal_readiness.modal_migration_lineage_path(identity)).unlink()

    inspection = modal_readiness.inspect_prior_quarantine_accounting(
        request=request,
        root=tmp_path,
    )

    assert inspection["blockers"] == []
    assert inspection["candidate"] == expected


def test_prior_accounting_request_time_and_snapshot_binding_fail_before_output(
    tmp_path: Path,
) -> None:
    expected, _logical = _prior_quarantine_accounting_fixture(tmp_path)
    base = _prior_quarantine_accounting_request(expected)
    cases = (
        ("recorded_at_utc", "2025-01-01T01:00:00Z", "predates"),
        ("recorded_at_utc", "2099-01-01T00:00:00Z", "too far in the future"),
        (
            "snapshot_sha256",
            "0" * 64,
            "selected cleanup snapshot binding changed",
        ),
    )
    for index, (field, value, message) in enumerate(cases):
        request = json.loads(json.dumps(base))
        if field == "snapshot_sha256":
            request["snapshot_capture_manifest"]["sha256"] = value
        else:
            request[field] = value
        output = tmp_path / f"candidate-{index}.json"
        with pytest.raises(ValueError, match=message):
            modal_readiness.scaffold_prior_quarantine_accounting(
                request=request,
                output_path=output,
                root=tmp_path,
            )
        assert not output.exists()


def test_prior_accounting_rejects_unjournaled_global_reservation(
    tmp_path: Path,
) -> None:
    expected, logical = _prior_quarantine_accounting_fixture(tmp_path)
    request = _prior_quarantine_accounting_request(expected)
    source = tmp_path / expected["remote_run_reservations"][0]["path"]
    reservation = json.loads(source.read_text(encoding="utf-8"))
    reservation["remote_run_id"] = "unjournaled-prior-reservation"
    injected = tmp_path / modal_readiness.modal_remote_run_reservation_path(
        reservation["remote_run_id"]
    )
    _write_json(injected, reservation)
    output = tmp_path / "candidate.json"

    with pytest.raises(ValueError, match="global reservation namespace differ"):
        modal_readiness.scaffold_prior_quarantine_accounting(
            request=request,
            output_path=output,
            root=tmp_path,
        )

    assert not output.exists()
    assert not (tmp_path / logical).exists()


def test_prior_accounting_scaffold_never_overwrites_any_existing_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
        "schema_name": "ModalPriorCohortQuarantineAccountingRequest",
        "schema_version": "1.0",
        "source_tree_sha256": "1" * 64,
        "image_source_sha256": "2" * 64,
        "cohort_id": "prior-scaffold-create-only",
        "recorded_at_utc": "2025-01-01T02:00:00Z",
        "snapshot_capture_manifest": {
            "path": "outputs/capture_manifest.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
        },
    }
    monkeypatch.setattr(
        modal_readiness,
        "_derive_prior_quarantine_accounting_candidate",
        lambda *_args, **_kwargs: {"validator_clean": True},
    )
    target = tmp_path / "preserved.json"
    original = b'{"preserved":true}\n'
    target.write_bytes(original)
    target.chmod(0o600)
    outputs = [target]
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    outputs.append(symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    outputs.append(hardlink)

    for output in outputs:
        with pytest.raises(FileExistsError):
            modal_readiness.scaffold_prior_quarantine_accounting(
                request=request,
                output_path=output,
                root=tmp_path,
            )
        assert target.read_bytes() == original


def test_prior_accounting_cli_shapes_dispatch_secure_operator_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_path = tmp_path / "request.json"
    _write_json(request_path, {"request": True})
    request_path.chmod(0o600)
    snapshot_path = tmp_path / "capture_manifest.json"
    output_path = tmp_path / "output.json"
    observed: list[tuple[str, dict[str, object]]] = []

    def template(**kwargs):
        observed.append(("template", kwargs))
        return {"result": "template"}

    monkeypatch.setattr(
        modal_readiness,
        "create_prior_quarantine_accounting_template",
        template,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_modal_readiness.py",
            "prior-quarantine-accounting-template",
            "--source-tree-sha256",
            "1" * 64,
            "--image-source-sha256",
            "2" * 64,
            "--cohort-id",
            "prior-cli",
            "--recorded-at-utc",
            "2025-01-01T02:00:00Z",
            "--snapshot-capture-manifest",
            str(snapshot_path),
            "--output",
            str(output_path),
        ],
    )
    assert modal_readiness.main() == 0
    assert json.loads(capsys.readouterr().out) == {"result": "template"}

    def inspect(**kwargs):
        observed.append(("inspect", kwargs))
        return {"result": "inspect"}

    monkeypatch.setattr(
        modal_readiness,
        "inspect_prior_quarantine_accounting",
        inspect,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_modal_readiness.py",
            "prior-quarantine-accounting-inspect",
            "--input",
            str(request_path),
        ],
    )
    assert modal_readiness.main() == 0
    assert json.loads(capsys.readouterr().out) == {"result": "inspect"}

    def scaffold(**kwargs):
        observed.append(("scaffold", kwargs))
        return {"result": "scaffold"}

    monkeypatch.setattr(
        modal_readiness,
        "scaffold_prior_quarantine_accounting",
        scaffold,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_modal_readiness.py",
            "prior-quarantine-accounting-scaffold",
            "--input",
            str(request_path),
            "--output",
            str(output_path),
        ],
    )
    assert modal_readiness.main() == 0
    assert json.loads(capsys.readouterr().out) == {"result": "scaffold"}
    assert [name for name, _kwargs in observed] == [
        "template",
        "inspect",
        "scaffold",
    ]
    assert observed[1][1]["request"] == {"request": True}
    assert observed[2][1] == {
        "request": {"request": True},
        "output_path": str(output_path),
    }


def test_prior_quarantine_accounting_rejects_chronology_and_validation_spoofs(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    cases = (
        ("recorded_at_utc", "2025-01-01T01:00:00Z", "predates"),
        ("recorded_at_utc", "2099-01-01T00:00:00Z", "too far in the future"),
        ("validated", False, "must be exactly True"),
        ("validated", 1, "must be exactly True"),
    )
    for field, value, message in cases:
        candidate = json.loads(json.dumps(payload))
        candidate[field] = value
        frozen, encoded = modal_readiness._exclusive_json_object_bytes(candidate)
        with pytest.raises(ValueError, match=message):
            modal_readiness._validate_prior_quarantine_accounting_payload(
                tmp_path,
                logical,
                frozen,
                encoded,
            )


def test_prior_quarantine_rejects_nonjournal_cheaper_storage_price_basis(
    tmp_path: Path,
) -> None:
    payload, logical = _prior_quarantine_accounting_fixture(tmp_path)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=payload["source_tree_sha256"],
        image_source_sha256=payload["image_source_sha256"],
        cohort_id=payload["cohort_id"],
    )
    alternate_logical, _alternate_sha256, alternate = _write_modal_price_basis(
        tmp_path,
        identity.image_source_sha256,
        retrieved_at_utc="2025-01-01T00:01:00Z",
    )
    alternate["volume_storage_usd_per_gib_month"] = "0.0001"
    alternate_path = tmp_path / alternate_logical
    _write_json(alternate_path, alternate)
    alternate_raw = alternate_path.read_bytes()
    payload["modal_price_basis"] = {
        "path": alternate_logical,
        "sha256": hashlib.sha256(alternate_raw).hexdigest(),
        "size_bytes": len(alternate_raw),
    }
    frozen, encoded = modal_readiness._exclusive_json_object_bytes(payload)

    with pytest.raises(ValueError, match="highest-rate journal binding"):
        modal_readiness._validate_prior_quarantine_accounting_payload(
            tmp_path,
            logical,
            frozen,
            encoded,
        )


def test_operator_input_reader_requires_lexically_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "operator-input.json"
    _write_json(candidate, {"schema_name": "fixture"})
    candidate.chmod(0o600)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="lexically absolute"):
        modal_readiness._load_operator_json_input("operator-input.json")


def test_prior_quarantine_accounting_blocks_concurrent_intent_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.modal_action_lock import ModalActionLockContentionError

    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="locked-prior-cohort",
    )
    payload = {
        "schema_name": "ModalPriorCohortQuarantineAccounting",
        "schema_version": "1.0",
        **modal_readiness.modal_cohort_identity_dict(identity),
    }
    observed_contention: list[bool] = []
    real_create = modal_readiness.create_json_exclusive

    def validate_candidate(
        root: Path,
        logical: str,
        candidate: object,
        raw: bytes,
    ) -> tuple[dict[str, object], dict[str, object]]:
        del root, logical, raw
        assert isinstance(candidate, dict)
        return dict(candidate), {}

    def create_while_old_cohort_writer_contends(
        path: Path,
        value: object,
    ) -> None:
        with pytest.raises(ModalActionLockContentionError):
            modal_readiness.acquire_modal_action_lock(tmp_path)
        observed_contention.append(True)
        real_create(path, value)

    monkeypatch.setattr(
        modal_readiness,
        "_validate_prior_quarantine_accounting_payload",
        validate_candidate,
    )
    monkeypatch.setattr(
        modal_readiness,
        "create_json_exclusive",
        create_while_old_cohort_writer_contends,
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_prior_quarantine_accounting",
        lambda root, logical: (dict(payload), {}),
    )

    persisted = modal_readiness.create_prior_quarantine_accounting(
        payload=payload,
        root=tmp_path,
    )

    assert persisted == payload
    assert observed_contention == [True]


def test_prior_remote_run_dispositions_accept_definite_no_start_and_unresolved(
    tmp_path: Path,
) -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="prior-disposition-cohort",
    )
    no_start = {
        "attempt_id": "1" * 32,
        "action": "canary",
        "run_id": "prior-no-start",
        "harness": "greedy_autoresearch",
        "concrete_remote_run_ids": ["prior-no-start"],
        "modal_cli_process_started": False,
        "status": "preflight_failed",
    }
    unresolved = {
        "attempt_id": "2" * 32,
        "action": "offline-smoke",
        "run_id": "prior-unresolved",
        "harness": None,
        "concrete_remote_run_ids": ["prior-unresolved"],
        "modal_cli_process_started": True,
        "status": "failed",
    }
    dispositions = [
        {
            "attempt_id": no_start["attempt_id"],
            "run_id": "prior-no-start",
            "execution_disposition": "definitely_not_started",
            "provider_disposition": "definitely_not_started",
            "snapshot_disposition": "no_remote_resources_observed",
            "snapshot_app_ids": [],
            "volume_disposition": "absent",
        },
        {
            "attempt_id": unresolved["attempt_id"],
            "run_id": "prior-unresolved",
            "execution_disposition": (
                "may_have_started_unresolved_quarantined"
            ),
            "provider_disposition": "not_applicable",
            "snapshot_disposition": "stopped_resources_bound",
            "snapshot_app_ids": ["ap-prior-unresolved"],
            "volume_disposition": "present_bound",
        },
    ]

    observed = modal_readiness._validate_prior_remote_run_dispositions(
        tmp_path,
        identity,
        [no_start, unresolved],
        dispositions,
    )

    assert observed[("1" * 32, "prior-no-start")][
        "execution_disposition"
    ] == "definitely_not_started"
    assert observed[("2" * 32, "prior-unresolved")][
        "snapshot_disposition"
    ] == "stopped_resources_bound"


def test_prior_remote_run_dispositions_reject_omission_and_hidden_evidence(
    tmp_path: Path,
) -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="prior-hidden-evidence",
    )
    attempt = {
        "attempt_id": "3" * 32,
        "action": "canary",
        "run_id": "prior-hidden-run",
        "harness": "semantic_autoresearch",
        "concrete_remote_run_ids": ["prior-hidden-run"],
        "modal_cli_process_started": True,
        "status": "failed",
    }
    disposition = {
        "attempt_id": attempt["attempt_id"],
        "run_id": attempt["run_id"],
        "execution_disposition": "may_have_started_unresolved_quarantined",
        "provider_disposition": "start_unresolved_conservative",
        "snapshot_disposition": "no_remote_resources_observed",
        "snapshot_app_ids": [],
        "volume_disposition": "absent",
    }

    with pytest.raises(ValueError, match="cover every reserved remote run"):
        modal_readiness._validate_prior_remote_run_dispositions(
            tmp_path,
            identity,
            [attempt],
            [],
        )

    context = (
        tmp_path
        / "outputs/development/modal_downloads/prior-hidden-run/"
        "execution_context.json"
    )
    _write_json(context, {"partial": True})
    with pytest.raises(ValueError, match="execution evidence was omitted"):
        modal_readiness._validate_prior_remote_run_dispositions(
            tmp_path,
            identity,
            [attempt],
            [disposition],
        )
    context.unlink()

    ledger = (
        tmp_path
        / "outputs/development/modal_downloads/prior-hidden-run/controller/"
        "provider_attempts.jsonl"
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")
    # Provider bytes are intentionally reconciled by the full quarantine
    # accounting validator, where they can be preserved as unbound evidence.
    observed = modal_readiness._validate_prior_remote_run_dispositions(
        tmp_path,
        identity,
        [attempt],
        [disposition],
    )
    assert observed[(attempt["attempt_id"], attempt["run_id"])][
        "provider_disposition"
    ] == "start_unresolved_conservative"


def test_terminal_dispositions_require_volume_on_success_but_allow_failed_absence(
    tmp_path: Path,
) -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="terminal-volume-dispositions",
    )
    succeeded = {
        "attempt_id": "4" * 32,
        "action": "offline-smoke",
        "run_id": "successful-run",
        "harness": None,
        "concrete_remote_run_ids": ["successful-run"],
        "modal_cli_process_started": True,
        "status": "succeeded",
    }
    failed = {
        "attempt_id": "5" * 32,
        "action": "offline-smoke",
        "run_id": "failed-run",
        "harness": None,
        "concrete_remote_run_ids": ["failed-run"],
        "modal_cli_process_started": True,
        "status": "failed",
    }
    failed_disposition = {
        "attempt_id": failed["attempt_id"],
        "run_id": failed["run_id"],
        "execution_disposition": "remote_execution_bound",
        "provider_disposition": "not_applicable",
        "snapshot_disposition": "stopped_resources_bound",
        "snapshot_app_ids": ["ap-failed-run"],
        "volume_disposition": "absent_after_failure",
    }
    observed = modal_readiness._validate_prior_remote_run_dispositions(
        tmp_path,
        identity,
        [failed],
        [failed_disposition],
    )
    assert observed[(failed["attempt_id"], failed["run_id"])][
        "volume_disposition"
    ] == "absent_after_failure"

    invalid_success = {
        "attempt_id": succeeded["attempt_id"],
        "run_id": succeeded["run_id"],
        "execution_disposition": "remote_execution_bound",
        "provider_disposition": "not_applicable",
        "snapshot_disposition": "app_volume_and_billing_bound",
        "snapshot_app_ids": ["ap-successful-run"],
        "volume_disposition": "absent",
    }
    with pytest.raises(ValueError, match="terminal start state"):
        modal_readiness._validate_prior_remote_run_dispositions(
            tmp_path,
            identity,
            [succeeded],
            [invalid_success],
        )


def test_provider_unknown_start_reserves_one_and_aggregate_four_requests(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    _journal, attempts = modal_readiness._cohort_action_journal(
        tmp_path,
        identity,
    )
    single = next(
        item
        for item in attempts
        if item["action"] == "canary"
        and item["harness"] == "greedy_autoresearch"
    )
    single_key = (single["attempt_id"], single["run_id"])
    single_estimate = modal_readiness._derive_journal_provider_spend_estimate(
        tmp_path,
        identity=identity,
        attempts=[single],
        remote_run_dispositions={
            single_key: {"provider_disposition": "start_unresolved_conservative"}
        },
        provider_evidence={},
        accounting_label="fixture",
    )
    assert single_estimate["provider_attempt_count_lower_bound"] == 0
    assert single_estimate["provider_attempt_count_upper_bound"] == 1
    assert modal_readiness.Decimal(
        single_estimate["conservative_provider_spend_bound_usd"]
    ) > 0
    assert single_estimate["known_success_usage_estimate_usd"] == "0"

    aggregate = json.loads(json.dumps(single))
    aggregate.update(
        {
            "attempt_id": "e" * 32,
            "action": "canaries",
            "run_id": "prior-aggregate-unknown",
            "harness": None,
            "concrete_remote_run_ids": [
                f"prior-aggregate-unknown-{modal_readiness._CANARY_SUFFIXES[harness]}"
                for harness in modal_readiness.CANARY_ORDER
            ],
        }
    )
    aggregate_dispositions = {
        (aggregate["attempt_id"], run_id): {
            "provider_disposition": "start_unresolved_conservative"
        }
        for run_id in aggregate["concrete_remote_run_ids"]
    }
    aggregate_estimate = (
        modal_readiness._derive_journal_provider_spend_estimate(
            tmp_path,
            identity=identity,
            attempts=[aggregate],
            remote_run_dispositions=aggregate_dispositions,
            provider_evidence={},
            accounting_label="fixture",
        )
    )
    assert aggregate_estimate["provider_attempt_count_upper_bound"] == 4
    assert (
        aggregate_estimate["conservative_provider_spend_bound_usd"]
        == aggregate_estimate["launcher_approval_bounds"][0][
            "source_bound_approval_ceiling_usd"
        ]
    )


def test_provider_definite_no_start_journal_has_zero_exposure(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    _journal, attempts = modal_readiness._cohort_action_journal(
        tmp_path,
        identity,
    )
    attempt = json.loads(
        json.dumps(next(item for item in attempts if item["action"] == "canary"))
    )
    attempt["modal_cli_process_started"] = False
    key = (attempt["attempt_id"], attempt["run_id"])

    estimate = modal_readiness._derive_journal_provider_spend_estimate(
        tmp_path,
        identity=identity,
        attempts=[attempt],
        remote_run_dispositions={
            key: {"provider_disposition": "definitely_not_started"}
        },
        provider_evidence={},
        accounting_label="fixture",
    )

    assert estimate["provider_attempt_count_lower_bound"] == 0
    assert estimate["provider_attempt_count_upper_bound"] == 0
    assert estimate["conservative_provider_spend_bound_usd"] == "0"
    assert estimate["run_cost_dispositions"][0]["cost_disposition"] == (
        "definitely_not_started_zero_exposure"
    )


def test_provider_journal_rejects_non_boolean_process_started(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    _journal, attempts = modal_readiness._cohort_action_journal(
        tmp_path,
        identity,
    )
    template = next(item for item in attempts if item["action"] == "canary")
    for spoof in (0, 1, "false", None):
        attempt = json.loads(json.dumps(template))
        attempt["modal_cli_process_started"] = spoof
        key = (attempt["attempt_id"], attempt["run_id"])
        with pytest.raises(ValueError, match="must be boolean"):
            modal_readiness._derive_journal_provider_spend_estimate(
                tmp_path,
                identity=identity,
                attempts=[attempt],
                remote_run_dispositions={
                    key: {"provider_disposition": "definitely_not_started"}
                },
                provider_evidence={},
                accounting_label="fixture",
            )


def test_unbound_provider_evidence_preserves_terminal_usage_or_full_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="unbound-provider-evidence",
    )
    plan_path = tmp_path / "provider-plan.json"
    price_path = tmp_path / "provider-price.json"
    _write_json(plan_path, {"fixture": "plan"})
    _write_json(price_path, {"fixture": "price"})
    generation_sha256 = "3" * 64
    plan = {
        "harnesses": [
            {
                "harness": "greedy_autoresearch",
                "first_opportunity": {
                    "conservative_input_token_ceiling": 100,
                },
                "request_settings": {"max_completion_tokens": 200},
                "generation_settings_sha256": generation_sha256,
            }
        ]
    }
    price = {
        "uncached_input_usd_per_million_tokens": "1",
        "output_usd_per_million_tokens": "2",
        "per_request_fee_usd": "0",
    }
    price_sha256 = hashlib.sha256(price_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        modal_readiness,
        "_load_provider_approval_plan",
        lambda *_args, **_kwargs: (plan, plan_path),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_price_basis",
        lambda *_args, **_kwargs: (price, price_path, price_sha256),
    )
    attempt_id = "6" * 32
    run_id = "unbound-provider-run"
    attempt = {
        "attempt_id": attempt_id,
        "action": "canary",
        "run_id": run_id,
        "harness": "greedy_autoresearch",
        "modal_cli_process_started": True,
        "provider_cost_approved": True,
        "provider_cost_cap_usd": "1",
        "provider_approval_plan_path": plan_path.name,
        "approval_plan_sha256": "4" * 64,
        "provider_price_basis_path": price_path.name,
        "provider_price_basis_sha256": price_sha256,
        "approved_image_source_sha256": identity.image_source_sha256,
        "predecessor_receipts": [
            {
                "gate": "candidate_resume_preflight_validated",
                "path": "preflight.json",
                "sha256": "5" * 64,
            }
        ],
        "concrete_remote_run_ids": [run_id],
        "started_at_utc": "2025-01-01T00:00:00Z",
        "finished_at_utc": "2025-01-01T00:01:00Z",
        "remote_execution_state": "may_have_started",
    }
    record = ProviderAttemptRecord.from_dict(
        {
            "schema_name": "ProviderAttemptRecord",
            "schema_version": "1.0",
            "harness": "greedy_autoresearch",
            "action": "one_opportunity_engineering_canary",
            "controller_run_id": "controller-unbound-provider-run",
            "execution_backend": "modal",
            "action_run_id": run_id,
            "modal_call_id": "fc-unbound-provider-run",
            "attempt_ordinal": 1,
            "started_at_utc": "2025-01-01T00:00:10Z",
            "ended_at_utc": "2025-01-01T00:00:20Z",
            "status": "success",
            "api_endpoint": modal_readiness.OFFICIAL_OPENAI_API_BASE,
            "model": modal_readiness.TARGET_MODEL,
            "generation_settings_sha256": generation_sha256,
            "provider_response_id": "response-unbound",
            "provider_request_id": "request-unbound",
            "usage_known": True,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "error_class": None,
        }
    )
    key = (attempt_id, run_id)
    disposition = {key: {"provider_disposition": "start_unresolved_conservative"}}
    terminal = modal_readiness._derive_journal_provider_spend_estimate(
        tmp_path,
        identity=identity,
        attempts=[attempt],
        remote_run_dispositions=disposition,
        provider_evidence={
            key: {
                "state": "unbound_observed",
                "records": [record],
                "harness": "greedy_autoresearch",
                "parse_dispositions": ["valid_terminal_records"],
            }
        },
        accounting_label="fixture",
    )
    assert terminal["provider_attempt_count_lower_bound"] == 1
    assert terminal["provider_attempt_count_upper_bound"] == 1
    assert terminal["uncertain_request_start_reserve_usd"] == "0"
    assert terminal["provider_request_ids"] == ["request-unbound"]
    assert terminal["provider_response_ids"] == ["response-unbound"]

    partial = modal_readiness._derive_journal_provider_spend_estimate(
        tmp_path,
        identity=identity,
        attempts=[attempt],
        remote_run_dispositions=disposition,
        provider_evidence={
            key: {
                "state": "unbound_observed",
                "records": [],
                "harness": "greedy_autoresearch",
                "parse_dispositions": ["partial_unparseable"],
            }
        },
        accounting_label="fixture",
    )
    assert partial["provider_attempt_count_lower_bound"] == 0
    assert partial["provider_attempt_count_upper_bound"] == 1
    assert (
        partial["uncertain_request_start_reserve_usd"]
        == partial["launcher_approval_bounds"][0][
            "source_bound_approval_ceiling_usd"
        ]
    )


def test_provider_bound_evidence_cannot_be_omitted_or_inferred_as_zero(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    _journal, attempts = modal_readiness._cohort_action_journal(
        tmp_path,
        identity,
    )
    attempt = next(item for item in attempts if item["action"] == "canary")
    key = (attempt["attempt_id"], attempt["run_id"])
    with pytest.raises(ValueError, match="evidence coverage is incomplete"):
        modal_readiness._derive_journal_provider_spend_estimate(
            tmp_path,
            identity=identity,
            attempts=[attempt],
            remote_run_dispositions={
                key: {"provider_disposition": "evidence_bound"}
            },
            provider_evidence={},
            accounting_label="fixture",
        )


def test_snapshot_terminal_cutoff_accepts_equality_and_rejects_late_terminal(
) -> None:
    manifest = {"started_at_utc": "2025-01-01T00:01:00Z"}
    exact = {
        "attempt_id": "4" * 32,
        "finished_at_utc": "2025-01-01T00:01:00Z",
    }
    modal_readiness._assert_attempts_finished_before_snapshot(
        [exact],
        manifest,
        field="fixture",
    )
    late = {**exact, "finished_at_utc": "2025-01-01T00:01:00.000001Z"}
    with pytest.raises(ValueError, match="after snapshot capture began"):
        modal_readiness._assert_attempts_finished_before_snapshot(
            [late],
            manifest,
            field="fixture",
        )


@pytest.mark.parametrize("closed", (False, None))
def test_terminal_seal_rejects_started_process_without_proven_containment(
    closed: bool | None,
) -> None:
    with pytest.raises(ValueError, match="provisional_unsealable"):
        modal_readiness._assert_attempts_contained_for_seal(
            [
                {
                    "attempt_id": "a" * 32,
                    "modal_cli_process_started": True,
                    "process_group_closed": closed,
                    "local_process_start_receipt_sha256": "c" * 64,
                }
            ],
            field="fixture",
        )


def test_terminal_seal_accepts_closed_and_definitely_unstarted_attempts() -> None:
    modal_readiness._assert_attempts_contained_for_seal(
        [
            {
                "attempt_id": "a" * 32,
                "modal_cli_process_started": True,
                "process_group_closed": True,
                "local_process_start_receipt_sha256": "c" * 64,
            },
            {
                "attempt_id": "b" * 32,
                "modal_cli_process_started": False,
                "process_group_closed": None,
            },
        ],
        field="fixture",
    )


def test_operator_input_reader_rejects_duplicates_mode_symlinks_and_hardlinks(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    _write_json(candidate, {"schema_name": "fixture"})
    candidate.chmod(0o600)
    assert modal_readiness._load_operator_json_input(candidate) == {
        "schema_name": "fixture"
    }

    candidate.write_text('{"key": 1, "key": 2}\n', encoding="utf-8")
    candidate.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate key"):
        modal_readiness._load_operator_json_input(candidate)

    _write_json(candidate, {"schema_name": "fixture"})
    candidate.chmod(0o644)
    with pytest.raises(ValueError, match="mode must be exactly 0600"):
        modal_readiness._load_operator_json_input(candidate)
    candidate.chmod(0o600)

    symlink = tmp_path / "candidate-symlink.json"
    symlink.symlink_to(candidate)
    with pytest.raises(ValueError, match="symbolic link"):
        modal_readiness._load_operator_json_input(symlink)

    hardlink = tmp_path / "candidate-hardlink.json"
    os.link(candidate, hardlink)
    with pytest.raises(ValueError, match="one regular file"):
        modal_readiness._load_operator_json_input(candidate)


@pytest.mark.parametrize(
    ("action", "creator_name"),
    (
        ("prior-quarantine-accounting", "create_prior_quarantine_accounting"),
        ("migration-lineage", "create_modal_migration_lineage_from_input"),
        ("cohort-roster", "create_modal_cohort_roster"),
    ),
)
def test_operator_input_cli_dispatches_secure_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    action: str,
    creator_name: str,
) -> None:
    candidate = tmp_path / f"{action}.json"
    _write_json(candidate, {"candidate": action})
    candidate.chmod(0o600)
    observed: list[dict[str, object]] = []

    def creator(*, payload: dict[str, object]) -> dict[str, object]:
        observed.append(payload)
        return {"published": action}

    monkeypatch.setattr(modal_readiness, creator_name, creator)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record_modal_readiness.py", action, "--input", str(candidate)],
    )

    assert modal_readiness.main() == 0
    assert observed == [{"candidate": action}]
    assert json.loads(capsys.readouterr().out) == {"published": action}


def _lineage_input_from_roster(roster: dict[str, object]) -> dict[str, object]:
    return {
        "schema_name": "ModalMigrationLineageInput",
        "schema_version": "1.0",
        "source_tree_sha256": roster["source_tree_sha256"],
        "image_source_sha256": roster["image_source_sha256"],
        "cohort_id": roster["cohort_id"],
        "accepted_primary_runs": roster["accepted_primary_runs"],
        "accepted_attempt_ids": roster["accepted_attempt_ids"],
        "prior_quarantine_accounting_paths": [],
    }


def test_migration_lineage_input_creates_v11_seal(tmp_path: Path) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    lineage_path = tmp_path / str(roster["migration_lineage_path"])
    lineage_path.unlink()

    persisted = modal_readiness.create_modal_migration_lineage_from_input(
        payload=_lineage_input_from_roster(roster),
        root=tmp_path,
    )

    assert persisted["schema_name"] == "ModalMigrationLineage"
    assert persisted["schema_version"] == "1.1"
    assert lineage_path.is_file()
    assert persisted["selected_final"]["remote_executions"]
    assert persisted["selected_final"]["provider_attempt_evidence"]
    assert persisted["selected_final"]["artifact_manifests"]
    assert modal_readiness.Decimal(
        persisted["final_provider_spend_bound_usd"]
    ) > 0
    assert (
        persisted["migration_provider_spend_bound_usd"]
        == persisted["final_provider_spend_bound_usd"]
    )


def test_lineage_publisher_seals_and_resumes_after_rejection_seal_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text())
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    lineage_path, rejection_path = _remove_global_terminal_seals(
        tmp_path,
        identity,
    )
    real_scan = modal_readiness.scan_modal_global_action_journal
    scan_states: list[tuple[bool, int]] = []

    def observed_scan(**kwargs: object) -> object:
        scan = real_scan(**kwargs)
        scan_states.append(
            (
                scan.global_rejection_seal is not None,
                sum(cohort.sealed for cohort in scan.cohorts),
            )
        )
        return scan

    real_create = modal_readiness.create_json_exclusive
    crashed = False

    def crash_once_at_lineage(path: Path, payload: object) -> None:
        nonlocal crashed
        if Path(path) == lineage_path and not crashed:
            crashed = True
            raise RuntimeError("simulated crash after rejection seal")
        real_create(path, payload)

    monkeypatch.setattr(
        modal_readiness,
        "scan_modal_global_action_journal",
        observed_scan,
    )
    monkeypatch.setattr(
        modal_readiness,
        "create_json_exclusive",
        crash_once_at_lineage,
    )
    arguments = {
        "final_identity": identity,
        "accepted_primary_runs": roster["accepted_primary_runs"],
        "accepted_attempt_ids": roster["accepted_attempt_ids"],
        "root": tmp_path,
    }

    with pytest.raises(RuntimeError, match="simulated crash"):
        modal_readiness.create_modal_migration_lineage(**arguments)
    assert rejection_path.is_file()
    assert not lineage_path.exists()
    rejection_before = rejection_path.read_bytes()
    rejection_inode = rejection_path.stat().st_ino

    persisted = modal_readiness.create_modal_migration_lineage(**arguments)

    assert persisted["validated"] is True
    assert lineage_path.is_file()
    assert rejection_path.read_bytes() == rejection_before
    assert rejection_path.stat().st_ino == rejection_inode
    assert scan_states == [
        (False, 0),
        (True, 0),
        (True, 0),
        (True, 0),
        (True, 1),
    ]


@pytest.mark.parametrize("mutation", ("tamper", "addition"))
def test_lineage_crash_resume_rejects_changed_rejection_seal(
    tmp_path: Path,
    mutation: str,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text())
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    lineage_path = tmp_path / modal_readiness.modal_migration_lineage_path(
        identity
    )
    lineage_path.unlink()
    rejection_path = (
        tmp_path / modal_readiness.modal_global_launch_rejection_seal_path()
    )
    rejection = json.loads(rejection_path.read_text())
    if mutation == "tamper":
        rejection["validated"] = False
    else:
        rejection["unexpected"] = True
    _write_json(rejection_path, rejection)
    rejection_path.chmod(0o600)

    with pytest.raises(modal_global_journal.ModalActionJournalIntegrityError):
        modal_readiness.create_modal_migration_lineage(
            final_identity=identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            root=tmp_path,
        )

    assert not lineage_path.exists()


def test_lineage_publisher_real_scanner_blocks_unresolved_journal_without_seals(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text())
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    lineage_path, rejection_path = _remove_global_terminal_seals(
        tmp_path,
        identity,
    )
    missing_terminal = tmp_path / roster["action_attempt_receipts"][0]
    missing_terminal.unlink()

    with pytest.raises(modal_global_journal.ModalActionJournalBlockedError):
        modal_readiness.create_modal_migration_lineage(
            final_identity=identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            root=tmp_path,
        )

    assert not rejection_path.exists()
    assert not lineage_path.exists()


def test_migration_lineage_terminal_seal_blocks_new_launches(
    tmp_path: Path,
) -> None:
    _roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    lock_descriptor = modal_readiness.acquire_modal_action_lock(tmp_path)
    try:
        scan = modal_global_journal.scan_modal_global_action_journal(
            lock_descriptor=lock_descriptor,
        )
        modal_global_journal.require_modal_global_action_journal_resolved(scan)
        with pytest.raises(
            modal_global_journal.ModalActionJournalBlockedError,
            match="migration_terminal_seal_present",
        ):
            modal_global_journal.require_modal_global_action_gate_clear(
                scan,
                candidate_attempt_id="f" * 32,
            )
    finally:
        modal_readiness.release_modal_action_lock(lock_descriptor)


@pytest.mark.parametrize(
    "publisher",
    ("prior", "roster", "cleanup", "bundle"),
)
def test_terminal_publishers_gate_before_derivation_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher: str,
) -> None:
    from common.modal_action_lock import ModalActionLockContentionError

    identity = ModalLiveCohortIdentity(
        source_tree_sha256="7" * 64,
        image_source_sha256="8" * 64,
        cohort_id="global-journal-publisher-gate",
    )
    observed: list[str] = []

    def block_under_lock(lock_descriptor: int) -> object:
        modal_readiness.assert_modal_action_lock_identity(lock_descriptor)
        with pytest.raises(ModalActionLockContentionError):
            modal_readiness.acquire_modal_action_lock(tmp_path)
        observed.append(publisher)
        raise modal_global_journal.ModalActionJournalBlockedError(
            "global Modal action journal is unresolved: fixture"
        )

    monkeypatch.setattr(
        modal_readiness,
        "_scan_resolved_modal_global_action_journal",
        block_under_lock,
    )
    if publisher == "prior":
        output = (
            tmp_path
            / modal_readiness.modal_prior_quarantine_accounting_path(identity)
        )
        call = partial(
            modal_readiness.create_prior_quarantine_accounting,
            payload=modal_readiness.modal_cohort_identity_dict(identity),
            root=tmp_path,
        )
    elif publisher == "roster":
        output = tmp_path / modal_readiness.modal_cohort_roster_path(identity)
        call = partial(
            modal_readiness.create_modal_cohort_roster,
            payload=modal_readiness.modal_cohort_identity_dict(identity),
            root=tmp_path,
        )
    elif publisher == "cleanup":
        output = tmp_path / modal_readiness.modal_component_receipt_path(
            identity,
            "modal_resource_cleanup_validated",
        )
        call = partial(
            modal_readiness.record_resource_cleanup,
            cohort_roster_path="missing-roster.json",
            root=tmp_path,
        )
    else:
        output = tmp_path / modal_readiness.modal_component_receipt_path(
            identity,
            "modal_migration_validation_bundle_validated",
        )
        call = partial(
            modal_readiness.record_migration_validation_bundle,
            cohort_roster_path="missing-roster.json",
            root=tmp_path,
        )

    with pytest.raises(
        modal_global_journal.ModalActionJournalBlockedError,
        match="unresolved",
    ):
        call()

    assert observed == [publisher]
    assert not output.exists()


def test_migration_lineage_allows_overlapping_prior_snapshots_with_exact_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    final_identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    final_journal = modal_readiness._cohort_action_journal(
        tmp_path,
        final_identity,
    )
    prior_identities = [
        ModalLiveCohortIdentity(
            source_tree_sha256=str(index) * 64,
            image_source_sha256=final_identity.image_source_sha256,
            cohort_id=f"prior-provider-{index}",
        )
        for index in (3, 4)
    ]
    prior_paths = [f"prior-{index}.json" for index in (1, 2)]
    bounds = [modal_readiness.Decimal("0.125"), modal_readiness.Decimal("0.375")]
    prior_by_path: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for identity, logical, bound in zip(
        prior_identities,
        prior_paths,
        bounds,
        strict=True,
    ):
        estimate = {
            "conservative_provider_spend_bound_usd": format(bound, "f")
        }
        modal_exposure = {
            "attempts": [],
            "measured_app_billing_usd": "0",
            "unresolved_compute_reserve_usd": "0",
            "conservative_compute_exposure_usd": "0",
            "approved_cap_usd": "0.25",
            "cap_exceeded": False,
            "amount_above_approved_cap_usd": "0",
            "accounting_basis": (
                "measured_app_billing_plus_unresolved_per_attempt_cap_reserve"
            ),
            "local_authorization_is_platform_hard_bound": False,
        }
        prior_by_path[logical] = (
            {
                "retained_storage_estimate": {},
                "provider_spend_estimate": estimate,
                "modal_compute_exposure": modal_exposure,
                "selected_billing_rows": [],
            },
            {
                "identity": identity,
                "raw_sha256": "5" * 64,
                "size_bytes": 1,
                "subtotal": modal_readiness.Decimal("0"),
                "provider_spend_bound": bound,
                "attempt_ids": set(),
                "run_ids": set(),
                "app_ids": set(),
                "call_ids": set(),
                "function_ids": set(),
                "image_ids": {"im-test123"},
                "provider_request_ids": set(),
                "provider_response_ids": set(),
                "artifact_paths": set(),
                "modal_measured_billing": modal_readiness.Decimal("0"),
                "modal_unresolved_reserve": modal_readiness.Decimal("0"),
                "modal_conservative_exposure": modal_readiness.Decimal("0"),
                "billing_row_keys": set(),
                "snapshot_finished_at": datetime(
                    2024,
                    12,
                    31,
                    23,
                    0,
                    tzinfo=UTC,
                ),
                "observed_migration_app_ids": set(),
                "observed_migration_billing_rows": {},
                "observed_volume_run_ids": set(),
            },
        )
    owned_app_ids = ("ap-owned-by-prior-one", "ap-owned-by-prior-two")
    owned_run_ids = ("run-owned-by-prior-one", "run-owned-by-prior-two")
    owned_billing_rows = {
        f"billing-row-{index}": {
            "object_id": app_id,
            "description": APP_NAME,
            "environment": "main",
            "interval_start": "2025-01-01T00:00:00+00:00",
            "resource": "CPU",
            "cost": "0.01",
        }
        for index, app_id in enumerate(owned_app_ids, start=1)
    }
    for index, logical in enumerate(prior_paths):
        metadata = prior_by_path[logical][1]
        metadata["app_ids"] = {owned_app_ids[index]}
        metadata["run_ids"] = {owned_run_ids[index]}
        metadata["billing_row_keys"] = {f"billing-row-{index + 1}"}
        metadata["observed_migration_app_ids"] = set(owned_app_ids)
        metadata["observed_migration_billing_rows"] = dict(owned_billing_rows)
        metadata["observed_volume_run_ids"] = set(owned_run_ids)
        prior_by_path[logical][0]["selected_billing_rows"] = [
            {
                "app_id": owned_app_ids[index],
                "row_sha256": f"billing-row-{index + 1}",
                "row": owned_billing_rows[f"billing-row-{index + 1}"],
            }
        ]
    later_zero_row = {
        "object_id": owned_app_ids[0],
        "description": APP_NAME,
        "environment": "main",
        "interval_start": "2025-01-02T00:00:00+00:00",
        "resource": "CPU",
        "cost": "0",
    }
    later_zero_key = modal_readiness.canonical_sha256(later_zero_row)
    prior_by_path[prior_paths[1]][1]["observed_migration_billing_rows"][
        later_zero_key
    ] = later_zero_row
    monkeypatch.setattr(
        modal_readiness,
        "_discover_cohort_journal_identities",
        lambda root: [final_identity, *prior_identities],
    )
    monkeypatch.setattr(
        modal_readiness,
        "_cohort_action_journal",
        lambda root, identity: (
            final_journal
            if identity == final_identity
            else (
                {
                    "intent_receipts": [],
                    "terminal_receipts": [],
                    "aggregate_receipts": [],
                },
                [],
            )
        ),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_prior_quarantine_accounting",
        lambda root, logical: prior_by_path[logical],
    )

    claims = modal_readiness._derive_migration_lineage_claims(
        tmp_path,
        final_identity=final_identity,
        accepted_primary_runs=roster["accepted_primary_runs"],
        accepted_attempt_ids=roster["accepted_attempt_ids"],
        prior_quarantine_accounting_paths=prior_paths,
    )

    final_bound = modal_readiness.Decimal(
        claims["final_provider_spend_bound_usd"]
    )
    assert claims["prior_provider_spend_bound_usd"] == "0.500"
    assert modal_readiness.Decimal(
        claims["migration_provider_spend_bound_usd"]
    ) == final_bound + sum(bounds)
    assert [
        item["provider_spend_estimate"]
        for item in claims["prior_quarantined_cohorts"]
    ] == [prior_by_path[path][0]["provider_spend_estimate"] for path in prior_paths]
    assert all(
        later_zero_key
        not in {
            row["row_sha256"]
            for row in prior_by_path[path][0]["selected_billing_rows"]
        }
        for path in prior_paths
    )
    assert all(
        metadata["image_ids"] == {"im-test123"}
        and metadata["identity"].image_source_sha256
        == final_identity.image_source_sha256
        for _payload, metadata in prior_by_path.values()
    )

    first_metadata = prior_by_path[prior_paths[0]][1]
    second_metadata = prior_by_path[prior_paths[1]][1]
    first_metadata["snapshot_finished_at"] = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="before the final cohort starts"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    first_metadata["snapshot_finished_at"] = datetime(
        2024,
        12,
        31,
        23,
        0,
        tzinfo=UTC,
    )

    second_metadata["observed_migration_app_ids"].remove(owned_app_ids[1])
    with pytest.raises(ValueError, match="absent from its owning cohort snapshot"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    second_metadata["observed_migration_app_ids"].add(owned_app_ids[1])

    second_metadata["observed_volume_run_ids"].remove(owned_run_ids[1])
    with pytest.raises(ValueError, match="Volume run is absent from its owning"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    second_metadata["observed_volume_run_ids"].add(owned_run_ids[1])

    original_overlap_row = second_metadata["observed_migration_billing_rows"].pop(
        "billing-row-1"
    )
    second_metadata["observed_migration_billing_rows"][
        "billing-row-1-conflict"
    ] = {**original_overlap_row, "cost": "0.02"}
    with pytest.raises(ValueError, match="conflict on one billing charge"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    second_metadata["observed_migration_billing_rows"].pop(
        "billing-row-1-conflict"
    )
    second_metadata["observed_migration_billing_rows"]["billing-row-1"] = (
        original_overlap_row
    )

    first_metadata["observed_migration_app_ids"].add("ap-unknown-prior")
    with pytest.raises(ValueError, match="unknown stopped App"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    first_metadata["observed_migration_app_ids"].remove("ap-unknown-prior")

    second_metadata["billing_row_keys"].add("billing-row-1")
    with pytest.raises(ValueError, match="owned by multiple prior cohorts"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    second_metadata["billing_row_keys"].remove("billing-row-1")

    original_second_identity = second_metadata["identity"]
    conflicting_second_identity = ModalLiveCohortIdentity(
        source_tree_sha256=original_second_identity.source_tree_sha256,
        image_source_sha256="9" * 64,
        cohort_id=original_second_identity.cohort_id,
    )
    second_metadata["identity"] = conflicting_second_identity
    prior_identities[1] = conflicting_second_identity
    with pytest.raises(ValueError, match="conflicting image source digests"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )
    second_metadata["identity"] = original_second_identity
    prior_identities[1] = original_second_identity

    second_metadata["app_ids"].add(owned_app_ids[0])
    with pytest.raises(ValueError, match="app_ids are reused across cohorts"):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=prior_paths,
        )


def test_final_roster_rejects_overlong_snapshot_billing_window(
    tmp_path: Path,
) -> None:
    roster_path, snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text())
    roster["billing_window_start_utc"] = "2024-11-30T00:00:00Z"
    roster["billing_window_end_utc"] = "2025-01-01T00:00:00Z"
    _write_json(roster_path, roster)
    _refresh_snapshot_capture_manifest(snapshot_root)

    with pytest.raises(ValueError, match="exceeds 31 days"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


def test_final_roster_rejects_terminal_after_snapshot_started(
    tmp_path: Path,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["artifact_verifiers"]["offline_smoke"]["attempt_id"]
    terminal_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["finished_at_utc"] = "2025-01-01T01:00:02Z"
    _write_json(terminal_path, terminal)
    _refresh_migration_lineage(roster_path)

    with pytest.raises(ValueError, match="after snapshot capture began"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra", "invalid exact schema"),
        ("wrong_version", "contract drifted"),
        ("missing_run", "accepted runs are not exact"),
        ("boolean_attempt", "32 lowercase hexadecimal digits"),
        ("duplicate_prior", "sorted and unique"),
    ),
)
def test_migration_lineage_input_rejects_schema_and_type_spoofs(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    payload: dict[str, object] = {
        "schema_name": "ModalMigrationLineageInput",
        "schema_version": "1.0",
        "source_tree_sha256": "1" * 64,
        "image_source_sha256": "2" * 64,
        "cohort_id": "final-cohort",
        "accepted_primary_runs": {
            label: f"run-{index}"
            for index, label in enumerate(modal_readiness._PRIMARY_LABELS)
        },
        "accepted_attempt_ids": {
            label: f"{index + 1:032x}"
            for index, label in enumerate(modal_readiness._PRIMARY_LABELS)
        },
        "prior_quarantine_accounting_paths": [],
    }
    if mutation == "extra":
        payload["extra"] = None
    elif mutation == "wrong_version":
        payload["schema_version"] = "1.1"
    elif mutation == "missing_run":
        cast_runs = payload["accepted_primary_runs"]
        assert isinstance(cast_runs, dict)
        cast_runs.pop("offline_smoke")
    elif mutation == "boolean_attempt":
        cast_attempts = payload["accepted_attempt_ids"]
        assert isinstance(cast_attempts, dict)
        cast_attempts["offline_smoke"] = True
    else:
        payload["prior_quarantine_accounting_paths"] = [
            "prior.json",
            "prior.json",
        ]
    monkeypatch.setattr(
        modal_readiness,
        "create_modal_migration_lineage",
        lambda **kwargs: {"unexpected": kwargs},
    )

    with pytest.raises(ValueError, match=message):
        modal_readiness.create_modal_migration_lineage_from_input(
            payload=payload,
            root=Path.cwd(),
        )


def test_lineage_v10_is_rejected_as_stale(tmp_path: Path) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    lineage_path = tmp_path / roster["migration_lineage_path"]
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["schema_version"] = "1.0"
    _write_json(lineage_path, lineage)
    roster["migration_lineage_sha256"] = hashlib.sha256(
        lineage_path.read_bytes()
    ).hexdigest()
    _write_json(roster_path, roster)

    with pytest.raises(ValueError, match="contract drifted"):
        modal_readiness._load_cohort_roster(
            tmp_path,
            roster_path.relative_to(tmp_path).as_posix(),
        )


def test_cohort_roster_creator_prevalidates_and_holds_shared_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.modal_action_lock import ModalActionLockContentionError

    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster_path.unlink()
    real_create = modal_readiness.create_json_exclusive
    observed: list[bool] = []

    def create_while_launcher_contends(path: Path, value: object) -> None:
        with pytest.raises(ModalActionLockContentionError):
            modal_readiness.acquire_modal_action_lock(tmp_path)
        observed.append(True)
        real_create(path, value)

    monkeypatch.setattr(
        modal_readiness,
        "create_json_exclusive",
        create_while_launcher_contends,
    )
    persisted = modal_readiness.create_modal_cohort_roster(
        payload=roster,
        root=tmp_path,
    )
    assert persisted == roster
    assert observed == [True]


def test_cleanup_and_bundle_recorders_lock_before_derivation_and_do_not_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from common.modal_action_lock import ModalActionLockContentionError

    identity = ModalLiveCohortIdentity(
        source_tree_sha256="7" * 64,
        image_source_sha256="8" * 64,
        cohort_id="publication-lock-fixture",
    )
    roster = modal_readiness.modal_cohort_identity_dict(identity)
    monkeypatch.setattr(
        modal_readiness,
        "_load_cohort_roster",
        lambda *args, **kwargs: (roster, tmp_path / "roster.json"),
    )
    observed: list[str] = []

    def stop_while_lock_is_held(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        with pytest.raises(ModalActionLockContentionError):
            modal_readiness.acquire_modal_action_lock(tmp_path)
        observed.append("locked")
        raise RuntimeError("stop before publication")

    monkeypatch.setattr(
        modal_readiness,
        "_derive_cleanup_claims",
        stop_while_lock_is_held,
    )
    with pytest.raises(RuntimeError, match="stop before publication"):
        modal_readiness.record_resource_cleanup(
            cohort_roster_path="roster.json",
            root=tmp_path,
        )
    cleanup_output = tmp_path / modal_readiness.modal_component_receipt_path(
        identity,
        "modal_resource_cleanup_validated",
    )
    assert not cleanup_output.exists()

    monkeypatch.setattr(
        modal_readiness,
        "_derive_migration_bundle_claims",
        stop_while_lock_is_held,
    )
    with pytest.raises(RuntimeError, match="stop before publication"):
        modal_readiness.record_migration_validation_bundle(
            cohort_roster_path="roster.json",
            root=tmp_path,
        )
    bundle_output = tmp_path / modal_readiness.modal_component_receipt_path(
        identity,
        "modal_migration_validation_bundle_validated",
    )
    assert not bundle_output.exists()
    assert observed == ["locked", "locked"]


def test_invalid_cohort_roster_creates_no_canonical_file(tmp_path: Path) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster_path.unlink()
    roster["unexpected"] = None

    with pytest.raises(ValueError, match="invalid exact schema"):
        modal_readiness.create_modal_cohort_roster(
            payload=roster,
            root=tmp_path,
        )
    assert not roster_path.exists()


@pytest.mark.parametrize(
    "collision_field",
    (
        "app_ids",
        "call_ids",
        "function_ids",
        "provider_request_ids",
        "provider_response_ids",
        "artifact_paths",
    ),
)
def test_lineage_rejects_final_prior_identifier_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_field: str,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    final_identity = ModalLiveCohortIdentity(
        source_tree_sha256=roster["source_tree_sha256"],
        image_source_sha256=roster["image_source_sha256"],
        cohort_id=roster["cohort_id"],
    )
    final_journal, final_attempts = modal_readiness._cohort_action_journal(
        tmp_path,
        final_identity,
    )
    _evidence, final_metadata = (
        modal_readiness._derive_final_lineage_evidence(
            tmp_path,
            identity=final_identity,
            journal=final_journal,
            attempts=final_attempts,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
        )
    )
    collision = next(iter(final_metadata[collision_field]))
    prior_identity = ModalLiveCohortIdentity(
        source_tree_sha256="3" * 64,
        image_source_sha256="4" * 64,
        cohort_id="prior-cohort",
    )
    empty_journal = {
        "intent_receipts": [],
        "terminal_receipts": [],
        "aggregate_receipts": [],
    }
    prior_metadata = {
        "identity": prior_identity,
        "raw_sha256": "5" * 64,
        "size_bytes": 1,
        "subtotal": modal_readiness.Decimal("0"),
        "attempt_ids": set(),
        "run_ids": set(),
        "app_ids": set(),
        "call_ids": set(),
        "function_ids": set(),
        "image_ids": set(),
        "provider_request_ids": set(),
        "provider_response_ids": set(),
        "artifact_paths": set(),
        "snapshot_finished_at": datetime(
            2024,
            12,
            31,
            23,
            0,
            tzinfo=UTC,
        ),
    }
    prior_metadata[collision_field] = {collision}
    prior_payload = {"retained_storage_estimate": {}}
    prior_logical = "prior-accounting.json"

    monkeypatch.setattr(
        modal_readiness,
        "_discover_cohort_journal_identities",
        lambda root: [final_identity, prior_identity],
    )
    monkeypatch.setattr(
        modal_readiness,
        "_cohort_action_journal",
        lambda root, identity: (
            (final_journal, final_attempts)
            if identity == final_identity
            else (empty_journal, [])
        ),
    )
    monkeypatch.setattr(
        modal_readiness,
        "_load_prior_quarantine_accounting",
        lambda root, logical: (prior_payload, prior_metadata),
    )

    with pytest.raises(
        ValueError,
        match=f"final and prior {collision_field} are reused",
    ):
        modal_readiness._derive_migration_lineage_claims(
            tmp_path,
            final_identity=final_identity,
            accepted_primary_runs=roster["accepted_primary_runs"],
            accepted_attempt_ids=roster["accepted_attempt_ids"],
            prior_quarantine_accounting_paths=[prior_logical],
        )


def test_lineage_creator_detects_post_publish_journal_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    lineage_path = tmp_path / roster["migration_lineage_path"]
    lineage_path.unlink()
    attempt_root = _attempt_root(tmp_path)
    source_intent = next(attempt_root.glob("*.intent.json"))
    raced_intent = attempt_root / f"{'f' * 32}.intent.json"
    real_create = modal_readiness.create_json_exclusive

    def create_then_race(path: Path, value: object) -> None:
        real_create(path, value)
        raced_intent.write_bytes(source_intent.read_bytes())
        raced_intent.chmod(0o600)

    monkeypatch.setattr(modal_readiness, "create_json_exclusive", create_then_race)
    with pytest.raises(ValueError, match="intent has the wrong contract"):
        modal_readiness.create_modal_migration_lineage_from_input(
            payload=_lineage_input_from_roster(roster),
            root=tmp_path,
        )
    assert lineage_path.is_file()


@pytest.mark.parametrize(
    ("failure_kind", "process_started", "closed"),
    (
        ("process_group_cleanup", True, False),
        ("python_execution_cleanup", True, True),
        ("python_execution_cleanup", False, None),
        ("process_group_and_python_execution_cleanup", True, False),
    ),
)
def test_attempt_validator_accepts_exact_cleanup_failure_kinds(
    tmp_path: Path,
    failure_kind: str,
    process_started: bool,
    closed: bool | None,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["accepted_attempt_ids"]["offline_smoke"]
    receipt_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "cleanup_failed",
            "failure_kind": failure_kind,
            "modal_cli_process_started": process_started,
            "remote_execution_state": (
                "may_have_started" if process_started else "definitely_not_started"
            ),
            "returncode": None,
            "process_group_closed": closed,
        }
    )
    if not process_started:
        _clear_process_start_evidence(receipt)

    validated = modal_readiness._validate_action_attempt_receipt(
        receipt,
        expected_attempt_id=attempt_id,
        root=tmp_path,
    )
    assert validated["failure_kind"] == failure_kind


@pytest.mark.parametrize(
    ("failure_kind", "process_started", "closed", "returncode"),
    (
        ("python_execution_cleanup", True, False, None),
        ("python_execution_cleanup", False, True, None),
        ("process_group_and_python_execution_cleanup", True, True, None),
        ("process_group_and_python_execution_cleanup", False, None, None),
        ("process_group_cleanup", True, False, 0),
    ),
)
def test_attempt_validator_rejects_spoofed_cleanup_failure_combinations(
    tmp_path: Path,
    failure_kind: str,
    process_started: bool,
    closed: bool | None,
    returncode: int | None,
) -> None:
    roster_path, _snapshot_root = _aggregate_cleanup_fixture(tmp_path)
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    attempt_id = roster["accepted_attempt_ids"]["offline_smoke"]
    receipt_path = _attempt_root(tmp_path) / f"{attempt_id}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "status": "cleanup_failed",
            "failure_kind": failure_kind,
            "modal_cli_process_started": process_started,
            "remote_execution_state": (
                "may_have_started" if process_started else "definitely_not_started"
            ),
            "returncode": returncode,
            "process_group_closed": closed,
        }
    )
    if not process_started:
        _clear_process_start_evidence(receipt)

    with pytest.raises(
        ValueError,
        match="fields do not reconcile|unstarted Modal attempt",
    ):
        modal_readiness._validate_action_attempt_receipt(
            receipt,
            expected_attempt_id=attempt_id,
            root=tmp_path,
        )


def test_lineage_bundle_path_roster_is_exhaustive_for_final_and_prior_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bound_intent(logical: str, predecessor: str) -> dict[str, object]:
        path = tmp_path / logical
        _write_json(
            path,
            {
                "predecessor_receipts": [
                    {
                        "gate": "fixture_gate",
                        "path": predecessor,
                        "sha256": "a" * 64,
                    }
                ]
            },
        )
        raw = path.read_bytes()
        return {
            "path": logical,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    def bound_terminal(logical: str, marker_logical: str) -> dict[str, object]:
        marker_path = tmp_path / marker_logical
        _write_json(marker_path, {"fixture": "process-start"})
        marker_path.chmod(0o600)
        path = tmp_path / logical
        _write_json(
            path,
            {
                "local_process_start_receipt_path": marker_logical,
                "local_process_start_receipt_sha256": hashlib.sha256(
                    marker_path.read_bytes()
                ).hexdigest(),
            },
        )
        raw = path.read_bytes()
        return {
            "path": logical,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    final_intent = bound_intent("final/intent.json", "final/predecessor.json")
    prior_intent = bound_intent("prior/intent.json", "prior/predecessor.json")
    final_terminal = bound_terminal(
        "final/terminal.json",
        "final/process_start.json",
    )
    prior_terminal = bound_terminal(
        "prior/terminal.json",
        "prior/process_start.json",
    )
    snapshot_manifest_path = tmp_path / "prior/snapshot_manifest.json"
    snapshot_paths = {
        name: {"path": f"prior/snapshots/{name}.json"}
        for name in modal_readiness._SNAPSHOT_NAMES
    }
    _write_json(snapshot_manifest_path, {"snapshots": snapshot_paths})
    prior_accounting = {
        "snapshot_capture_manifest_path": "prior/snapshot_manifest.json",
        "remote_executions": [
            {"execution_context_path": "prior/execution_context.json"}
        ],
        "provider_attempt_evidence": [
            {"evidence_path": "prior/provider_evidence.json"}
        ],
        "unbound_provider_evidence": [
            {"evidence_path": "prior/unbound_provider_evidence.json"}
        ],
        "volume_dispositions": [
            {
                "artifact_manifest_disposition": "bound",
                "artifact_manifest_path": "prior/artifact_manifest.json",
            }
        ],
        "modal_price_basis": {"path": "prior/modal_price_basis.json"},
        "provider_spend_estimate": {
            "launcher_approval_bounds": [
                {
                    "approval_plan": {"path": "prior/provider_approval.json"},
                    "price_basis": {"path": "prior/provider_price.json"},
                }
            ]
        },
    }
    monkeypatch.setattr(
        modal_readiness,
        "_load_prior_quarantine_accounting",
        lambda *args, **kwargs: (prior_accounting, {}),
    )
    lineage = {
        "selected_final": {
            "action_journal": {
                "intent_receipts": [final_intent],
                "terminal_receipts": [final_terminal],
                "aggregate_receipts": [{"path": "final/aggregate.json"}],
            },
            "remote_run_reservations": [{"path": "final/reservation.json"}],
            "remote_executions": [
                {"evidence": {"path": "final/execution_context.json"}}
            ],
            "artifact_manifests": [{"path": "final/artifact_manifest.json"}],
            "provider_attempt_evidence": [
                {
                    "ledger": {"path": "final/provider_attempts.jsonl"},
                    "uncertainty": {"path": "final/provider_uncertainty.json"},
                }
            ],
            "provider_spend_estimate": {
                "launcher_approval_bounds": [
                    {
                        "approval_plan": {"path": "final/provider_approval.json"},
                        "price_basis": {"path": "final/provider_price.json"},
                    }
                ]
            },
        },
        "global_remote_run_reservations": [
            {"path": "global/reservation.json"}
        ],
        "prior_quarantined_cohorts": [
            {
                "accounting_receipt": {"path": "prior/accounting.json"},
                "action_journal": {
                    "intent_receipts": [prior_intent],
                    "terminal_receipts": [prior_terminal],
                    "aggregate_receipts": [{"path": "prior/aggregate.json"}],
                },
                "remote_run_reservations": [
                    {"path": "prior/reservation.json"}
                ],
            }
        ],
    }

    observed = modal_readiness._lineage_required_paths(tmp_path, lineage)

    expected = {
        "final/intent.json",
        "final/terminal.json",
        "final/process_start.json",
        "final/aggregate.json",
        "final/reservation.json",
        "final/predecessor.json",
        "final/execution_context.json",
        "final/artifact_manifest.json",
        "final/provider_attempts.jsonl",
        "final/provider_uncertainty.json",
        "final/provider_approval.json",
        "final/provider_price.json",
        "global/reservation.json",
        "prior/accounting.json",
        "prior/intent.json",
        "prior/terminal.json",
        "prior/process_start.json",
        "prior/aggregate.json",
        "prior/reservation.json",
        "prior/predecessor.json",
        "prior/snapshot_manifest.json",
        "prior/execution_context.json",
        "prior/provider_evidence.json",
        "prior/unbound_provider_evidence.json",
        "prior/artifact_manifest.json",
        "prior/modal_price_basis.json",
        "prior/provider_approval.json",
        "prior/provider_price.json",
        *(record["path"] for record in snapshot_paths.values()),
    }
    assert observed == expected


def test_bundle_artifact_hash_rejects_same_content_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    backup = tmp_path / "original.json"
    target.write_bytes(b"same bytes\n")
    replacement.write_bytes(b"same bytes\n")
    real_open = modal_readiness._open_regular_file_descriptor
    calls = 0

    def replace_before_reopen(path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.rename(backup)
            replacement.rename(target)
        return real_open(path)

    monkeypatch.setattr(
        modal_readiness,
        "_open_regular_file_descriptor",
        replace_before_reopen,
    )
    with pytest.raises(ValueError, match="path changed while it was hashed"):
        modal_readiness._required_project_artifact(tmp_path, "evidence.json")


def test_operator_input_rejects_same_content_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "input.json"
    replacement = tmp_path / "replacement.json"
    backup = tmp_path / "original.json"
    target.write_bytes(b'{"value":1}\n')
    replacement.write_bytes(b'{"value":1}\n')
    target.chmod(0o600)
    replacement.chmod(0o600)
    real_open = modal_readiness._open_regular_file_descriptor
    calls = 0

    def replace_before_reopen(path: Path):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.rename(backup)
            replacement.rename(target)
        return real_open(path)

    monkeypatch.setattr(
        modal_readiness,
        "_open_regular_file_descriptor",
        replace_before_reopen,
    )
    with pytest.raises(ValueError, match="path changed while it was read"):
        modal_readiness._load_operator_json_input(target)
