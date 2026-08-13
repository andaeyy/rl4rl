"""Read-only validation of one downloaded Modal offline-study bundle.

The validator starts at the outer Modal artifact manifest, then checks the
credential-free execution context, frozen study assignment, per-run event
chains and frozen indexes.  Finally it sends every assigned run through the
real reconstruction and reporting adapters.  It never follows a path recorded
inside the remote randomization plan and never writes validation output into
the downloaded bundle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from artifacts import FrozenIndexReference, RunArtifactStore
from common.network_denial import (
    validate_provider_free_action_outer_roster,
    validate_provider_free_network_denial_probe,
)
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    APP_NAME,
    ArtifactManifestV1,
    ImageSourceManifestV1,
    SourceFileV1,
    canonical_sha256,
    safe_relative_path,
    sha256_file,
    verify_artifact_manifest,
    volume_artifact_uri,
)
from reporting.adapters import adapt_run_store
from study import BudgetLedger, RandomizationPlan, RunState, StudySpec
from study.scheduling import SequentialAcceleratorScheduler
from study.serialization import content_hash

from reconstruction.rebuild import reconstruct_runs
from reconstruction.tables import build_analysis_table

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_ARTIFACT_FILES = 10_000
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_ARTIFACT_MANIFEST_FIELDS = ArtifactManifestV1.FIELDS
_IMAGE_MANIFEST_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "recipe_version",
        "python_version",
        "uv_version",
        "modal_version",
        "dependency_lock_sha256",
        "files",
    }
)
_ACTION_RESULT_FIELDS = frozenset(
    {
        "success",
        "mode",
        "returncode",
        "stdout_sha256",
        "stdout_size_bytes",
        "stderr_sha256",
        "stderr_size_bytes",
    }
)
_STUDY_INDEX_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "study_id",
        "assignment_hash",
        "run_indexes",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "study_id",
        "assignment_hash",
        "accelerator_kind",
        "run_order",
        "statuses",
        "execution_metadata",
        "active_run_id",
        "revision",
        "created_at",
        "updated_at",
    }
)
_OFFLINE_SMOKE_SUMMARY_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "mode",
        "provider_calls",
        "torch_training_runs",
        "scientific",
        "study_id",
        "assignment_hash",
        "scheduler",
        "runs",
        "artifact_index_manifest",
        "no_search",
    }
)
_OFFLINE_SMOKE_RUN_FIELDS = frozenset(
    {
        "run_id",
        "condition_id",
        "status",
        "seed_evaluations",
        "proposal_opportunities",
    }
)
_NO_SEARCH_FIELDS = frozenset(
    {
        "condition_id",
        "scientific",
        "adaptive_feedback_visible_to_backend",
        "provider_input_constant",
        "request_count",
        "ledger",
    }
)
_SCHEDULER_SUMMARY_FIELDS = frozenset(
    {
        "study_id",
        "assignment_hash",
        "accelerator_kind",
        "counts",
        "active_run_id",
        "revision",
    }
)
_SCHEDULER_COUNT_FIELDS = frozenset(
    {"pending", "running", "completed", "interrupted"}
)
_CREDENTIAL_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "secrets",
        "token",
    }
)


class DownloadedOfflineValidationError(ValueError):
    """The downloaded bundle is incomplete, unsafe, or internally inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DownloadedOfflineValidationError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DownloadedOfflineValidationError(f"required JSON file is unsafe: {path}")
    size = path.stat().st_size
    if size > _MAX_JSON_BYTES:
        raise DownloadedOfflineValidationError(
            f"JSON file exceeds the {_MAX_JSON_BYTES}-byte validation cap: {path}"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DownloadedOfflineValidationError(f"invalid JSON file: {path}") from error
    if not isinstance(payload, dict):
        raise DownloadedOfflineValidationError(f"expected a JSON object: {path}")
    return payload


def _is_path_field(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in {
        "candidate",
        "path",
        "directory",
        "dir",
        "root",
        "ledger",
        "location",
    } or normalized.endswith(
        (
            "_path",
            "_directory",
            "_dir",
            "_root",
            "_ledger",
            "_location",
        )
    )


def _absolute_path_field_paths(value: object, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if (
                _is_path_field(key_text)
                and isinstance(child, str)
                and (
                    Path(child).is_absolute()
                    or PureWindowsPath(child).is_absolute()
                    or bool(PureWindowsPath(child).drive)
                    or child.lower().startswith("file://")
                )
            ):
                found.append(child_path)
            found.extend(_absolute_path_field_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _absolute_path_field_paths(child, path=f"{path}[{index}]")
            )
    return found


def _credential_field_paths(value: object, *, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            normalized = key_text.strip().lower().replace("-", "_")
            child_path = f"{path}.{key_text}"
            if normalized in _CREDENTIAL_FIELD_NAMES or normalized.endswith(
                (
                    "_api_key",
                    "_authorization",
                    "_credential",
                    "_password",
                    "_private_key",
                    "_secret",
                    "_token",
                )
            ):
                found.append(child_path)
            found.extend(_credential_field_paths(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_credential_field_paths(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            "-----begin " in lowered
            and "private key-----" in lowered
            or lowered.startswith("sk-")
            and len(value) >= 20
            or lowered.startswith("hf_")
            and len(value) >= 20
            or lowered.startswith(("ghp_", "gho_", "ghu_", "ghs_", "ghr_"))
            and len(value) >= 20
            or lowered.startswith("github_pat_")
            and len(value) >= 24
            or lowered.startswith(("tinker_", "tml_"))
            and len(value) >= 20
        ):
            found.append(path)
    return found


def _reject_v2_executor_paths(study_root: Path) -> None:
    """Reject machine-specific locations anywhere in a new study bundle."""

    for artifact in sorted(study_root.rglob("*")):
        if not artifact.is_file() or artifact.suffix not in {".json", ".jsonl"}:
            continue
        if artifact.is_symlink():
            raise DownloadedOfflineValidationError(
                f"downloaded study JSON artifact may not be a symlink: {artifact}"
            )
        if artifact.stat().st_size > _MAX_JSON_BYTES:
            raise DownloadedOfflineValidationError(
                f"JSON artifact exceeds validation cap: {artifact}"
            )
        try:
            if artifact.suffix == ".json":
                payloads = [
                    json.loads(
                        artifact.read_text(encoding="utf-8"),
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                ]
            else:
                payloads = [
                    json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                    for line in artifact.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DownloadedOfflineValidationError(
                f"invalid JSON artifact: {artifact}"
            ) from error
        for index, payload in enumerate(payloads):
            absolute = _absolute_path_field_paths(payload)
            if absolute:
                logical = artifact.relative_to(study_root).as_posix()
                raise DownloadedOfflineValidationError(
                    "v2 study artifact contains executor-absolute path fields in "
                    f"{logical} record {index}: {', '.join(absolute[:5])}"
                )
            credentials = _credential_field_paths(payload)
            if credentials:
                logical = artifact.relative_to(study_root).as_posix()
                raise DownloadedOfflineValidationError(
                    "v2 study artifact contains credential-shaped fields in "
                    f"{logical} record {index}: {', '.join(credentials[:5])}"
                )


def _require_fields(
    payload: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise DownloadedOfflineValidationError(
            f"{label} fields differ from the frozen schema; "
            f"missing={missing}, extra={extra}"
        )


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DownloadedOfflineValidationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DownloadedOfflineValidationError(
            f"{label} must be a non-negative integer"
        )
    return value


def _require_exact_bool(value: object, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise DownloadedOfflineValidationError(
            f"{label} must be exactly {expected}"
        )


def _safe_bundle_root(bundle_root: str | Path) -> Path:
    raw = Path(bundle_root).expanduser()
    if raw.is_symlink():
        raise DownloadedOfflineValidationError(
            "downloaded bundle root may not be a symlink"
        )
    try:
        root = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise DownloadedOfflineValidationError(
            "downloaded bundle root is missing"
        ) from error
    if not root.is_dir():
        raise DownloadedOfflineValidationError(
            "downloaded bundle root is not a directory"
        )
    return root


def _snapshot_tree(root: Path) -> dict[str, tuple[int, str]]:
    snapshot: dict[str, tuple[int, str]] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DownloadedOfflineValidationError(
                f"downloaded bundle contains a symlink: {path.relative_to(root)}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise DownloadedOfflineValidationError(
                f"downloaded bundle contains a special file: {path.relative_to(root)}"
            )
        relative = path.relative_to(root).as_posix()
        try:
            safe_relative_path(relative)
        except ValueError as error:
            raise DownloadedOfflineValidationError(
                f"downloaded bundle contains an unsafe path: {relative}"
            ) from error
        snapshot[relative] = (path.stat().st_size, sha256_file(path))
    return dict(sorted(snapshot.items()))


def _load_outer_manifest(root: Path) -> ArtifactManifestV1:
    path = root / "artifact_manifest.json"
    payload = _read_json_object(path)
    _require_fields(payload, _ARTIFACT_MANIFEST_FIELDS, "artifact manifest")
    try:
        manifest = ArtifactManifestV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(str(error)) from error
    if len(manifest.files) > _MAX_ARTIFACT_FILES:
        raise DownloadedOfflineValidationError(
            "artifact manifest file count exceeds cap"
        )
    if sum(item.size_bytes for item in manifest.files) > _MAX_ARTIFACT_BYTES:
        raise DownloadedOfflineValidationError(
            "artifact manifest byte count exceeds cap"
        )
    if root.name != manifest.run_id:
        raise DownloadedOfflineValidationError(
            "download directory name does not match the Modal run ID"
        )
    expected = {item.relative_path for item in manifest.files} | {
        "artifact_manifest.json"
    }
    actual = set(_snapshot_tree(root))
    if actual != expected:
        raise DownloadedOfflineValidationError(
            "downloaded file set differs from the retained Modal artifact manifest"
        )
    try:
        verify_artifact_manifest(root, manifest)
    except Exception as error:
        raise DownloadedOfflineValidationError(str(error)) from error
    return manifest


def _load_image_source_manifest(
    root: Path, *, expected_sha256: str
) -> ImageSourceManifestV1:
    payload = _read_json_object(root / "image_source_manifest.json")
    _require_fields(payload, _IMAGE_MANIFEST_FIELDS, "image-source manifest")
    raw_files = payload["files"]
    if not isinstance(raw_files, list):
        raise DownloadedOfflineValidationError("image-source files must be a list")
    try:
        files = tuple(
            SourceFileV1(
                relative_path=item["relative_path"],
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
            )
            for item in raw_files
            if isinstance(item, dict) and set(item) == SourceFileV1.FIELDS
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(
            "image-source manifest contains an invalid file entry"
        ) from error
    if len(files) != len(raw_files):
        raise DownloadedOfflineValidationError(
            "image-source manifest file entries have unexpected fields"
        )
    try:
        manifest = ImageSourceManifestV1(
            dependency_lock_sha256=payload["dependency_lock_sha256"],
            files=files,
            python_version=payload["python_version"],
            uv_version=payload["uv_version"],
            modal_version=payload["modal_version"],
            recipe_version=payload["recipe_version"],
        )
    except (TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(str(error)) from error
    if payload["schema_name"] != ImageSourceManifestV1.SCHEMA_NAME:
        raise DownloadedOfflineValidationError(
            "expected ModalImageSourceManifest schema"
        )
    if payload["schema_version"] != ImageSourceManifestV1.SCHEMA_VERSION:
        raise DownloadedOfflineValidationError(
            "unsupported image-source manifest version"
        )
    if canonical_sha256(payload) != expected_sha256:
        raise DownloadedOfflineValidationError(
            "image-source manifest does not match the outer source digest"
        )
    lock_entries = tuple(item for item in files if item.relative_path == "uv.lock")
    if (
        len(lock_entries) != 1
        or lock_entries[0].sha256 != manifest.dependency_lock_sha256
    ):
        raise DownloadedOfflineValidationError(
            "dependency lock digest is not bound to the image-source uv.lock entry"
        )
    return manifest


def _validate_execution_receipt(
    root: Path, manifest: ArtifactManifestV1
) -> tuple[ExecutionContextV1, ImageSourceManifestV1]:
    try:
        context = ExecutionContextV1.from_dict(
            _read_json_object(root / "execution_context.json")
        )
    except (TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(str(error)) from error
    if (
        context.execution_backend != "modal"
        or context.run_id != manifest.run_id
        or context.app_name != APP_NAME
        or context.function_name != "offline_smoke"
        or context.artifact_uri != volume_artifact_uri(manifest.run_id)
        or context.image_source_sha256 != manifest.image_source_sha256
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise DownloadedOfflineValidationError(
            "execution context does not identify this Modal offline-smoke artifact"
        )
    image_manifest = _load_image_source_manifest(
        root, expected_sha256=manifest.image_source_sha256
    )

    result = _read_json_object(root / "remote_action_result.json")
    _require_fields(result, _ACTION_RESULT_FIELDS, "remote action result")
    returncode = _require_nonnegative_int(
        result["returncode"], "remote action returncode"
    )
    if (
        result["success"] is not True
        or result["mode"] != "provider_free_offline_smoke"
        or returncode != 0
    ):
        raise DownloadedOfflineValidationError(
            "remote action result is not a successful provider-free offline smoke"
        )
    for stream in ("stdout", "stderr"):
        _require_sha256(result[f"{stream}_sha256"], f"{stream} digest")
        _require_nonnegative_int(result[f"{stream}_size_bytes"], f"{stream} size")
    return context, image_manifest


def _load_study_directories(root: Path) -> tuple[Path, Path]:
    offline_root = root / "offline_study"
    if offline_root.is_symlink() or not offline_root.is_dir():
        raise DownloadedOfflineValidationError("offline_study directory is missing")
    studies = tuple(sorted(path for path in offline_root.iterdir() if path.is_dir()))
    unexpected = tuple(path for path in offline_root.iterdir() if not path.is_dir())
    no_search = tuple(
        path for path in studies if path.name.endswith("-no-search-smoke")
    )
    primary = tuple(path for path in studies if path not in no_search)
    if (
        unexpected
        or len(studies) != 2
        or len(primary) != 1
        or len(no_search) != 1
        or no_search[0].name != f"{primary[0].name}-no-search-smoke"
    ):
        raise DownloadedOfflineValidationError(
            "offline_study must contain exactly one C0-C3 study and its "
            "no-search sibling"
        )
    return primary[0], no_search[0]


def _validate_schedule(study_root: Path, plan: RandomizationPlan) -> dict[str, Any]:
    path = study_root / "schedule_state.json"
    state = _read_json_object(path)
    _require_fields(state, _SCHEDULE_FIELDS, "schedule state")
    if state.get("schema_version") != "2.0" or state.get("accelerator_kind") != "cpu":
        raise DownloadedOfflineValidationError(
            "offline Modal schedule must use the v2 CPU execution condition"
        )
    scheduler = SequentialAcceleratorScheduler(
        plan,
        state_path=path,
        lease_path=study_root / ".validator-never-create-lease",
        accelerator_kind="cpu",
    )
    summary = scheduler.summary()
    if summary["active_run_id"] is not None or summary["counts"] != {
        "pending": 0,
        "running": 0,
        "completed": len(plan.runs),
        "interrupted": 0,
    }:
        raise DownloadedOfflineValidationError(
            "offline schedule is not terminal for every frozen assignment"
        )
    metadata = state["execution_metadata"]
    if any(
        record != {"remote_call_id": None, "artifact_location": None}
        for record in metadata.values()
    ):
        raise DownloadedOfflineValidationError(
            "synthetic inner runs unexpectedly claim remote calls or artifact locations"
        )
    return summary


def _validate_no_search_artifacts(
    no_search_root: Path,
    *,
    primary_spec: StudySpec,
    ledger_payload: dict[str, Any],
) -> dict[str, Any]:
    if no_search_root.is_symlink() or not no_search_root.is_dir():
        raise DownloadedOfflineValidationError(
            "no-search study directory is missing or unsafe"
        )
    top_level = {path.name for path in no_search_root.iterdir()}
    if top_level != {"randomization_plan.json", "runs"}:
        raise DownloadedOfflineValidationError(
            "no-search study contains an unexpected top-level artifact"
        )
    plan_payload = _read_json_object(no_search_root / "randomization_plan.json")
    try:
        plan = RandomizationPlan.from_dict(
            plan_payload,
            execution_root=no_search_root,
        )
    except (TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(
            f"no-search randomization plan is invalid: {error}"
        ) from error
    if (
        plan.schema_version != "2.0"
        or plan.study_id != f"{primary_spec.study_id}-no-search-smoke"
        or no_search_root.name != plan.study_id
        or {run.condition.condition_id.value for run in plan.runs}
        != {"C0", "C1", "C2", "C3"}
    ):
        raise DownloadedOfflineValidationError(
            "no-search randomization plan differs from the bounded C0-C3 carrier"
        )
    _reject_v2_executor_paths(no_search_root)

    runs_root = no_search_root / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise DownloadedOfflineValidationError("no-search runs directory is unsafe")
    local_runs = tuple(sorted(runs_root.iterdir()))
    c0_runs = tuple(
        run for run in plan.runs if run.condition.condition_id.value == "C0"
    )
    if (
        len(c0_runs) != 1
        or len(local_runs) != 1
        or local_runs[0].is_symlink()
        or not local_runs[0].is_dir()
        or local_runs[0].name != c0_runs[0].run_id
    ):
        raise DownloadedOfflineValidationError(
            "no-search execution must contain exactly its frozen C0 carrier run"
        )
    assigned = c0_runs[0]
    run_root = local_runs[0]
    if {path.name for path in run_root.iterdir()} != {
        "artifact_ledger",
        "run_spec.json",
        "run_state.json",
        "study_spec.json",
    }:
        raise DownloadedOfflineValidationError(
            "no-search carrier run contains an unexpected top-level artifact"
        )
    if _read_json_object(run_root / "run_spec.json") != assigned.to_dict():
        raise DownloadedOfflineValidationError(
            "no-search run spec differs from its frozen assignment"
        )
    try:
        spec = StudySpec.from_dict(_read_json_object(run_root / "study_spec.json"))
        state = RunState.from_dict(_read_json_object(run_root / "run_state.json"))
    except (TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(
            f"no-search run state/spec is invalid: {error}"
        ) from error
    shared_spec_fields = (
        "study_seed",
        "budget",
        "portfolio_size",
        "transition_opportunities",
        "common_config_hash",
        "code_hash",
        "environment_hash",
        "scientific",
    )
    primary_payload = primary_spec.to_dict()
    no_search_payload = spec.to_dict()
    if (
        spec.study_id != plan.study_id
        or spec.block_count != 1
        or any(
            no_search_payload[field] != primary_payload[field]
            for field in shared_spec_fields
        )
        or plan.study_spec_hash != spec.spec_hash
    ):
        raise DownloadedOfflineValidationError(
            "no-search study spec differs from the primary bounded infrastructure"
        )
    _require_sha256(spec.initial_candidate_id, "no-search initial candidate ID")
    if (
        state.study_id != spec.study_id
        or state.run_id != assigned.run_id
        or state.block_id != assigned.block_id
        or state.condition_id != "C0"
        or state.assignment_hash != assigned.assignment_hash
        or state.initial_candidate_id != spec.initial_candidate_id
        or state.status != "completed"
        or state.remote_call_id is not None
        or state.artifact_location is not None
        or canonical_sha256(state.ledger) != canonical_sha256(ledger_payload)
    ):
        raise DownloadedOfflineValidationError(
            "no-search summary ledger differs from its retained bounded run"
        )

    ledger_root = run_root / "artifact_ledger"
    if (
        ledger_root.is_symlink()
        or not ledger_root.is_dir()
        or not (ledger_root / "artifact_context.json").is_file()
        or not (ledger_root / "raw_events").is_dir()
        or not (ledger_root / ".artifact-ledger.lock").is_file()
    ):
        raise DownloadedOfflineValidationError(
            "no-search artifact ledger is incomplete or unsafe"
        )
    store = RunArtifactStore.open(ledger_root)
    context = store.context
    if (
        context.study_id != spec.study_id
        or context.block_id != assigned.block_id
        or context.run_id != assigned.run_id
        or context.condition_id != "C0"
        or context.run_seed != assigned.run_seed
        or context.assignment_sha256 != assigned.assignment_hash
        or context.writer_component != "scripts.study_offline_smoke.no_search"
        or context.code_sha256 != spec.code_hash
        or context.config_sha256 != spec.common_config_hash
        or context.environment_sha256 != spec.environment_hash
    ):
        raise DownloadedOfflineValidationError(
            "no-search artifact context differs from its frozen run"
        )
    reference, index = store.load_frozen_index("search_completion")
    store.verify_against_index(index)
    reconstructed = reconstruct_runs((store,))
    if (
        len(reconstructed) != 1
        or reconstructed[0].context.run_id != assigned.run_id
        or reconstructed[0].status != "completed"
    ):
        raise DownloadedOfflineValidationError(
            "no-search retained run does not reconstruct as completed"
        )
    return {
        "study_id": spec.study_id,
        "run_id": assigned.run_id,
        "assignment_sha256": assigned.assignment_hash,
        "artifact_index_sha256": reference.index_sha256,
        "event_count": reference.event_count,
        "last_event_sha256": reference.last_event_sha256,
    }


def _validate_offline_smoke_summary(
    study_root: Path,
    *,
    no_search_root: Path,
    plan: RandomizationPlan,
    schedule_summary: dict[str, Any],
    run_summaries: list[dict[str, Any]],
    study_spec: StudySpec,
) -> dict[str, Any]:
    path = study_root / "offline_smoke_summary.json"
    payload = _read_json_object(path)
    credential_fields = _credential_field_paths(payload)
    if credential_fields:
        raise DownloadedOfflineValidationError(
            "offline smoke summary contains credential-shaped fields: "
            + ", ".join(credential_fields[:5])
        )
    absolute_fields = _absolute_path_field_paths(payload)
    if absolute_fields:
        raise DownloadedOfflineValidationError(
            "offline smoke summary contains executor-absolute path fields: "
            + ", ".join(absolute_fields[:5])
        )
    _require_fields(
        payload,
        _OFFLINE_SMOKE_SUMMARY_FIELDS,
        "offline smoke summary",
    )
    if (
        payload["schema_name"] != "OfflineStudySmokeSummary"
        or payload["schema_version"] != "1.0"
    ):
        raise DownloadedOfflineValidationError(
            "offline smoke summary has the wrong schema contract"
        )
    if payload["mode"] != "offline_synthetic_only":
        raise DownloadedOfflineValidationError(
            "offline smoke summary has the wrong mode"
        )
    if _require_nonnegative_int(payload["provider_calls"], "provider_calls") != 0:
        raise DownloadedOfflineValidationError(
            "offline smoke summary unexpectedly records provider calls"
        )
    if (
        _require_nonnegative_int(
            payload["torch_training_runs"], "torch_training_runs"
        )
        != 0
    ):
        raise DownloadedOfflineValidationError(
            "offline smoke summary unexpectedly records Torch training"
        )
    _require_exact_bool(payload["scientific"], False, "offline smoke scientific")
    if payload["study_id"] != plan.study_id:
        raise DownloadedOfflineValidationError(
            "offline smoke summary study ID differs from the frozen plan"
        )
    _require_sha256(payload["assignment_hash"], "offline smoke assignment hash")
    if payload["assignment_hash"] != plan.assignment_hash:
        raise DownloadedOfflineValidationError(
            "offline smoke summary assignment differs from the frozen plan"
        )
    if payload["artifact_index_manifest"] != "artifact_index_manifest.json":
        raise DownloadedOfflineValidationError(
            "offline smoke summary names an unexpected artifact-index manifest"
        )

    scheduler = payload["scheduler"]
    if not isinstance(scheduler, dict):
        raise DownloadedOfflineValidationError(
            "offline smoke scheduler summary must be an object"
        )
    _require_fields(scheduler, _SCHEDULER_SUMMARY_FIELDS, "offline smoke scheduler")
    counts = scheduler["counts"]
    if not isinstance(counts, dict):
        raise DownloadedOfflineValidationError(
            "offline smoke scheduler counts must be an object"
        )
    _require_fields(counts, _SCHEDULER_COUNT_FIELDS, "offline smoke scheduler counts")
    for status, value in counts.items():
        _require_nonnegative_int(value, f"offline smoke scheduler counts.{status}")
    _require_nonnegative_int(scheduler["revision"], "offline smoke scheduler revision")
    if scheduler != schedule_summary:
        raise DownloadedOfflineValidationError(
            "offline smoke scheduler summary differs from retained schedule state"
        )

    raw_runs = payload["runs"]
    if not isinstance(raw_runs, list):
        raise DownloadedOfflineValidationError("offline smoke runs must be a list")
    for index, record in enumerate(raw_runs):
        if not isinstance(record, dict):
            raise DownloadedOfflineValidationError(
                f"offline smoke run {index} must be an object"
            )
        _require_fields(record, _OFFLINE_SMOKE_RUN_FIELDS, f"offline smoke run {index}")
        for field in ("run_id", "condition_id", "status"):
            if not isinstance(record[field], str) or not record[field]:
                raise DownloadedOfflineValidationError(
                    f"offline smoke run {index} {field} must be nonempty text"
                )
        for field in ("seed_evaluations", "proposal_opportunities"):
            _require_nonnegative_int(
                record[field], f"offline smoke run {index} {field}"
            )
    if raw_runs != run_summaries:
        raise DownloadedOfflineValidationError(
            "offline smoke C0-C3 summaries differ from retained run state"
        )
    conditions = {record["condition_id"] for record in raw_runs}
    if conditions != {"C0", "C1", "C2", "C3"}:
        raise DownloadedOfflineValidationError(
            "offline smoke summary does not contain the complete C0-C3 roster"
        )

    no_search = payload["no_search"]
    if not isinstance(no_search, dict):
        raise DownloadedOfflineValidationError("no-search evidence must be an object")
    _require_fields(no_search, _NO_SEARCH_FIELDS, "no-search evidence")
    if no_search["condition_id"] != "NO_SEARCH":
        raise DownloadedOfflineValidationError(
            "no-search evidence has the wrong condition ID"
        )
    _require_exact_bool(no_search["scientific"], False, "no-search scientific")
    _require_exact_bool(
        no_search["adaptive_feedback_visible_to_backend"],
        False,
        "no-search adaptive feedback visibility",
    )
    _require_exact_bool(
        no_search["provider_input_constant"],
        True,
        "no-search provider input constant",
    )
    request_count = _require_nonnegative_int(
        no_search["request_count"], "no-search request_count"
    )
    if request_count != study_spec.budget.proposal_opportunities:
        raise DownloadedOfflineValidationError(
            "no-search request count differs from frozen proposal opportunities"
        )
    ledger_payload = no_search["ledger"]
    if not isinstance(ledger_payload, dict):
        raise DownloadedOfflineValidationError("no-search ledger must be an object")
    try:
        ledger = BudgetLedger.from_dict(ledger_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(
            f"no-search ledger is invalid: {error}"
        ) from error
    if canonical_sha256(ledger.to_dict()) != canonical_sha256(ledger_payload):
        raise DownloadedOfflineValidationError(
            "no-search ledger does not round-trip without coercion"
        )
    if ledger.spec.to_dict() != study_spec.budget.to_dict():
        raise DownloadedOfflineValidationError(
            "no-search ledger budget differs from the C0-C3 study budget"
        )
    expected_attempts = {
        str(index): 1 for index in range(1, request_count + 1)
    }
    if (
        ledger.schema_version != "2.0"
        or ledger.accelerator_kind != "cpu"
        or ledger.seed_evaluations != study_spec.budget.seed_evaluations
        or ledger.proposal_opportunities != request_count
        or ledger.provider_attempts != request_count
        or ledger.terminal_opportunities != request_count
        or ledger.active_opportunity is not None
        or ledger_payload["provider_attempts_by_opportunity"] != expected_attempts
    ):
        raise DownloadedOfflineValidationError(
            "no-search ledger does not exactly account for its opportunities"
        )
    for field in (
        "unknown_provider_usage",
        "parse_failures",
        "repairs",
        "infrastructure_retries",
        "scientific_failures",
        "infrastructure_failures",
    ):
        if getattr(ledger, field) != 0:
            raise DownloadedOfflineValidationError(
                f"no-search ledger unexpectedly records {field}"
            )
    no_search_artifacts = _validate_no_search_artifacts(
        no_search_root,
        primary_spec=study_spec,
        ledger_payload=ledger_payload,
    )

    return {
        "summary_sha256": sha256_file(path),
        "schema_name": payload["schema_name"],
        "schema_version": payload["schema_version"],
        "c0_c3_run_count": len(raw_runs),
        "condition_ids": sorted(conditions),
        "no_search": {
            "condition_id": no_search["condition_id"],
            "scientific": no_search["scientific"],
            "adaptive_feedback_visible_to_backend": no_search[
                "adaptive_feedback_visible_to_backend"
            ],
            "provider_input_constant": no_search["provider_input_constant"],
            "request_count": request_count,
            "ledger_sha256": canonical_sha256(ledger_payload),
            "artifacts": no_search_artifacts,
        },
    }


def _validate_study(
    study_root: Path,
    no_search_root: Path,
) -> tuple[RandomizationPlan, dict[str, Any], list[dict[str, Any]]]:
    plan_payload = _read_json_object(study_root / "randomization_plan.json")
    try:
        plan = RandomizationPlan.from_dict(
            plan_payload,
            execution_root=(
                study_root if plan_payload.get("schema_version") == "2.0" else None
            ),
        )
    except (TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(str(error)) from error
    if study_root.name != plan.study_id:
        raise DownloadedOfflineValidationError(
            "study directory name differs from the frozen study ID"
        )
    if plan.schema_version == "2.0":
        _reject_v2_executor_paths(study_root)
    schedule_summary = _validate_schedule(study_root, plan)

    runs_root = study_root / "runs"
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise DownloadedOfflineValidationError(
            "offline study runs directory is missing"
        )
    local_run_ids = {
        path.name
        for path in runs_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    planned_run_ids = {run.run_id for run in plan.runs}
    if local_run_ids != planned_run_ids or any(
        not path.is_dir() or path.is_symlink() for path in runs_root.iterdir()
    ):
        raise DownloadedOfflineValidationError(
            "local run directories differ from the frozen assignment roster"
        )

    index_payload = _read_json_object(study_root / "artifact_index_manifest.json")
    _require_fields(index_payload, _STUDY_INDEX_FIELDS, "study artifact-index manifest")
    if (
        index_payload["schema_name"] != "StudyArtifactIndexManifest"
        or index_payload["schema_version"] != "1.0"
        or index_payload["study_id"] != plan.study_id
        or index_payload["assignment_hash"] != plan.assignment_hash
        or not isinstance(index_payload["run_indexes"], list)
    ):
        raise DownloadedOfflineValidationError(
            "study artifact-index manifest does not bind the frozen assignment"
        )
    try:
        expected_index_references = tuple(
            FrozenIndexReference.from_dict(item)
            for item in index_payload["run_indexes"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DownloadedOfflineValidationError(
            "study artifact-index manifest contains an invalid run index"
        ) from error
    if tuple(item.run_id for item in expected_index_references) != tuple(
        run.run_id for run in plan.runs
    ):
        raise DownloadedOfflineValidationError(
            "study run-index order differs from the frozen assignment order"
        )

    stores: list[RunArtifactStore] = []
    references: list[FrozenIndexReference] = []
    specs: list[StudySpec] = []
    run_states: dict[str, RunState] = {}
    plan_by_id = {run.run_id: run for run in plan.runs}
    expected_reference_by_id = {item.run_id: item for item in expected_index_references}
    for run_id in sorted(planned_run_ids):
        assigned = plan_by_id[run_id]
        run_root = runs_root / run_id
        if assigned.schema_version == "2.0":
            if (
                assigned.run_directory != f"runs/{run_id}"
                or assigned.execution_directory != run_root.resolve()
            ):
                raise DownloadedOfflineValidationError(
                    "v2 logical run directory is not portable or bound locally: "
                    f"{run_id}"
                )
        else:
            historical_parts = PurePosixPath(assigned.run_directory).parts
            if tuple(historical_parts[-3:]) != (plan.study_id, "runs", run_id):
                raise DownloadedOfflineValidationError(
                    f"historical run directory has an unexpected suffix: {run_id}"
                )
        if _read_json_object(run_root / "run_spec.json") != assigned.to_dict():
            raise DownloadedOfflineValidationError(
                f"run_spec.json differs from frozen assignment: {run_id}"
            )
        try:
            spec = StudySpec.from_dict(_read_json_object(run_root / "study_spec.json"))
            state = RunState.from_dict(_read_json_object(run_root / "run_state.json"))
        except (TypeError, ValueError) as error:
            raise DownloadedOfflineValidationError(str(error)) from error
        if (
            spec.scientific
            or spec.study_id != plan.study_id
            or spec.spec_hash != plan.study_spec_hash
            or state.study_id != plan.study_id
            or state.block_id != assigned.block_id
            or state.run_id != run_id
            or state.condition_id != assigned.condition.condition_id.value
            or state.assignment_hash != assigned.assignment_hash
            or state.status != "completed"
            or state.remote_call_id is not None
            or state.artifact_location is not None
        ):
            raise DownloadedOfflineValidationError(
                "run state/spec does not match the completed offline assignment: "
                f"{run_id}"
            )
        specs.append(spec)
        run_states[run_id] = state

        ledger_root = run_root / "artifact_ledger"
        if (
            ledger_root.is_symlink()
            or not ledger_root.is_dir()
            or not (ledger_root / "artifact_context.json").is_file()
            or not (ledger_root / "raw_events").is_dir()
            or not (ledger_root / ".artifact-ledger.lock").is_file()
        ):
            raise DownloadedOfflineValidationError(
                f"artifact ledger is incomplete or unsafe: {run_id}"
            )
        store = RunArtifactStore.open(ledger_root)
        context = store.context
        if (
            context.study_id != plan.study_id
            or context.block_id != assigned.block_id
            or context.run_id != run_id
            or context.condition_id != assigned.condition.condition_id.value
            or context.run_seed != assigned.run_seed
            or context.assignment_sha256 != assigned.assignment_hash
            or context.writer_component != "scripts.study_offline_smoke"
            or context.code_sha256 != spec.code_hash
            or context.config_sha256 != spec.common_config_hash
            or context.environment_sha256 != spec.environment_hash
        ):
            raise DownloadedOfflineValidationError(
                f"artifact context differs from the frozen assignment/spec: {run_id}"
            )
        reference, index = store.load_frozen_index("search_completion")
        if reference != expected_reference_by_id[run_id]:
            raise DownloadedOfflineValidationError(
                f"run frozen index differs from study index manifest: {run_id}"
            )
        store.verify_against_index(index)
        references.append(reference)
        stores.append(store)

    if any(spec.to_dict() != specs[0].to_dict() for spec in specs[1:]):
        raise DownloadedOfflineValidationError(
            "offline assignments do not share one immutable study specification"
        )
    retained_run_summaries = [
        {
            "run_id": run.run_id,
            "condition_id": run.condition.condition_id.value,
            "status": run_states[run.run_id].status,
            "seed_evaluations": run_states[run.run_id].ledger["seed_evaluations"],
            "proposal_opportunities": run_states[run.run_id].ledger[
                "proposal_opportunities"
            ],
        }
        for run in plan.runs
    ]
    offline_smoke_evidence = _validate_offline_smoke_summary(
        study_root,
        no_search_root=no_search_root,
        plan=plan,
        schedule_summary=schedule_summary,
        run_summaries=retained_run_summaries,
        study_spec=specs[0],
    )

    reconstructed = reconstruct_runs(stores)
    outcome_table = build_analysis_table(
        reconstructed,
        assigned_run_ids=tuple(run.run_id for run in plan.runs),
    )
    reconstructed_by_id = {run.context.run_id: run for run in reconstructed}

    reporting_runs = []
    reporting_artifacts = []
    run_summaries: list[dict[str, Any]] = []
    for assigned in plan.runs:
        # Stores were opened in lexical run-ID order; reporting follows the frozen
        # assignment order and therefore resolves each immutable identity explicitly.
        selected_store = next(
            item for item in stores if item.context.run_id == assigned.run_id
        )
        selected_reference = next(
            item for item in references if item.run_id == assigned.run_id
        )
        adapted = adapt_run_store(
            selected_store,
            assignment_payload=assigned.assignment_payload(),
        )
        reporting_runs.append(adapted.run_record)
        reporting_artifacts.extend(adapted.artifacts)
        reconstructed_run = reconstructed_by_id[assigned.run_id]
        run_summaries.append(
            {
                "run_id": assigned.run_id,
                "condition_id": assigned.condition.condition_id.value,
                "status": reconstructed_run.status,
                "accelerator_kind": reconstructed_run.accelerator_kind,
                "event_count": selected_reference.event_count,
                "last_event_sha256": reconstructed_run.last_event_sha256,
                "artifact_index_sha256": adapted.artifact_index_sha256,
                "report_record_sha256": content_hash(adapted.run_record.to_dict()),
            }
        )
    artifact_ids = [item.artifact_id for item in reporting_artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise DownloadedOfflineValidationError(
            "reporting adapter produced duplicate artifact IDs across assigned runs"
        )

    reporting_summary = {
        "run_record_count": len(reporting_runs),
        "artifact_count": len(reporting_artifacts),
        "run_records_sha256": content_hash(
            [
                item.to_dict()
                for item in sorted(reporting_runs, key=lambda item: item.run_id)
            ]
        ),
        "artifacts_sha256": content_hash(
            [
                item.to_dict()
                for item in sorted(
                    reporting_artifacts, key=lambda item: item.artifact_id
                )
            ]
        ),
    }
    study_summary = {
        "study_id": plan.study_id,
        "randomization_schema_version": plan.schema_version,
        "assignment_sha256": plan.assignment_hash,
        "study_spec_sha256": plan.study_spec_hash,
        "assigned_run_ids": [run.run_id for run in plan.runs],
        "run_count": len(plan.runs),
        "outcome_table_sha256": outcome_table.table_hash,
        "schedule": schedule_summary,
        "offline_smoke": offline_smoke_evidence,
        "runs": run_summaries,
        "reporting": reporting_summary,
    }
    return plan, study_summary, run_summaries


def validate_downloaded_offline_bundle(bundle_root: str | Path) -> dict[str, Any]:
    """Validate a downloaded Modal run without modifying it or making API calls."""

    root = _safe_bundle_root(bundle_root)
    before = _snapshot_tree(root)
    manifest = _load_outer_manifest(root)
    context, image_manifest = _validate_execution_receipt(root, manifest)
    validate_provider_free_action_outer_roster(
        root,
        function_name="offline_smoke",
    )
    denial_probe_path = root / "provider_free_network_denial_probe.json"
    validate_provider_free_network_denial_probe(
        _read_json_object(denial_probe_path),
        expected_context=context,
    )
    study_root, no_search_root = _load_study_directories(root)
    _, study_summary, _ = _validate_study(study_root, no_search_root)
    after = _snapshot_tree(root)
    if after != before:
        raise DownloadedOfflineValidationError(
            "validation unexpectedly changed the downloaded artifact bundle"
        )

    payload: dict[str, Any] = {
        "schema_name": "DownloadedModalOfflineStudyValidation",
        "schema_version": "1.0",
        "verified": True,
        "validation_mode": "local_read_only_provider_free",
        "network_calls": 0,
        "provider_calls": 0,
        "modal_run_id": manifest.run_id,
        "modal_app_name": context.app_name,
        "modal_function_name": context.function_name,
        "modal_ids": {
            "app_id": context.modal_app_id,
            "function_id": context.modal_function_id,
            "call_id": context.modal_call_id,
            "image_id": context.modal_image_id,
        },
        "artifact_uri": context.artifact_uri,
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "provider_free_network_denial_probe_sha256": sha256_file(
            denial_probe_path
        ),
        "artifact_file_count": len(manifest.files),
        "artifact_bytes": sum(item.size_bytes for item in manifest.files),
        "image_source_sha256": manifest.image_source_sha256,
        "dependency_lock_sha256": image_manifest.dependency_lock_sha256,
        "study": study_summary,
    }
    payload["validation_sha256"] = content_hash(payload)
    return payload


__all__ = [
    "DownloadedOfflineValidationError",
    "validate_downloaded_offline_bundle",
]
