from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    APP_NAME,
    ImageSourceManifestV1,
    SourceFileV1,
    build_artifact_manifest,
    volume_artifact_uri,
    write_artifact_manifest,
)
from reconstruction.downloaded_offline import (
    DownloadedOfflineValidationError,
    validate_downloaded_offline_bundle,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _offline_bundle(tmp_path: Path) -> Path:
    run_id = "modal-offline-test"
    bundle = tmp_path / run_id
    bundle.mkdir(parents=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "study_offline_smoke.py"),
            "--output-dir",
            str(bundle / "offline_study"),
            "--study-id",
            f"modal-offline-{run_id}",
            "--blocks",
            "1",
            "--opportunities",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    source_manifest = ImageSourceManifestV1(
        dependency_lock_sha256="a" * 64,
        files=(SourceFileV1("uv.lock", "a" * 64, 1),),
    )
    _write_json(bundle / "image_source_manifest.json", source_manifest.to_dict())
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=APP_NAME,
        function_name="offline_smoke",
        modal_app_id="ap-offlinetest",
        modal_function_id="fu-offlinetest",
        modal_call_id="fc-offlinetest",
        modal_image_id="im-offlinetest",
        image_source_sha256=source_manifest.manifest_sha256,
        artifact_uri=volume_artifact_uri(run_id),
    )
    _write_json(bundle / "execution_context.json", context.to_dict())
    _write_json(
        bundle / "provider_free_network_denial_probe.json",
        {
            "schema_name": "ProviderFreeNetworkDenialProbe",
            "schema_version": "1.0",
            "attempted_endpoint": {"ip": "1.1.1.1", "port": 443},
            "timeout_seconds": 1.0,
            "denied": True,
            "exception_type": "PermissionError",
            "execution_context": context.to_dict(),
        },
    )
    empty_digest = hashlib.sha256(b"").hexdigest()
    _write_json(
        bundle / "remote_action_result.json",
        {
            "success": True,
            "mode": "provider_free_offline_smoke",
            "returncode": 0,
            "stdout_sha256": empty_digest,
            "stdout_size_bytes": 0,
            "stderr_sha256": empty_digest,
            "stderr_size_bytes": 0,
        },
    )
    manifest = build_artifact_manifest(
        bundle,
        run_id=run_id,
        image_source_sha256=source_manifest.manifest_sha256,
    )
    write_artifact_manifest(bundle, manifest)
    return bundle


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _refresh_outer_manifest(bundle: Path) -> None:
    manifest_path = bundle / "artifact_manifest.json"
    manifest_path.unlink()
    source = json.loads(
        (bundle / "image_source_manifest.json").read_text(encoding="utf-8")
    )
    manifest = build_artifact_manifest(
        bundle,
        run_id=bundle.name,
        image_source_sha256=ImageSourceManifestV1(
            dependency_lock_sha256=source["dependency_lock_sha256"],
            files=tuple(
                SourceFileV1(
                    item["relative_path"], item["sha256"], item["size_bytes"]
                )
                for item in source["files"]
            ),
            python_version=source["python_version"],
            uv_version=source["uv_version"],
            modal_version=source["modal_version"],
            recipe_version=source["recipe_version"],
        ).manifest_sha256,
    )
    write_artifact_manifest(bundle, manifest)


def _offline_summary_path(bundle: Path) -> Path:
    matches = tuple(bundle.rglob("offline_smoke_summary.json"))
    assert len(matches) == 1
    return matches[0]


def test_downloaded_offline_bundle_reconstructs_and_adapts_without_writes(
    tmp_path,
) -> None:
    bundle = _offline_bundle(tmp_path)
    before = _file_bytes(bundle)

    first = validate_downloaded_offline_bundle(bundle)
    second = validate_downloaded_offline_bundle(bundle)

    assert first == second
    assert _file_bytes(bundle) == before
    assert first["verified"] is True
    assert first["network_calls"] == first["provider_calls"] == 0
    assert len(first["provider_free_network_denial_probe_sha256"]) == 64
    assert first["study"]["run_count"] == 4
    assert first["study"]["randomization_schema_version"] == "2.0"
    assert first["study"]["schedule"]["accelerator_kind"] == "cpu"
    assert first["study"]["reporting"]["run_record_count"] == 4
    assert {run["accelerator_kind"] for run in first["study"]["runs"]} == {"cpu"}
    smoke = first["study"]["offline_smoke"]
    assert smoke["schema_name"] == "OfflineStudySmokeSummary"
    assert smoke["schema_version"] == "1.0"
    assert smoke["condition_ids"] == ["C0", "C1", "C2", "C3"]
    assert smoke["c0_c3_run_count"] == 4
    assert smoke["no_search"]["adaptive_feedback_visible_to_backend"] is False
    assert smoke["no_search"]["provider_input_constant"] is True
    assert smoke["no_search"]["request_count"] == 1


def test_downloaded_offline_cli_prints_the_same_deterministic_receipt(tmp_path) -> None:
    bundle = _offline_bundle(tmp_path)
    expected = validate_downloaded_offline_bundle(bundle)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_downloaded_offline_study.py"),
            str(bundle),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == expected


def test_downloaded_offline_bundle_rejects_tamper_and_symlinks(tmp_path) -> None:
    bundle = _offline_bundle(tmp_path)
    state = next(bundle.rglob("run_state.json"))
    state.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        DownloadedOfflineValidationError, match="(size|digest) mismatch"
    ):
        validate_downloaded_offline_bundle(bundle)

    other = _offline_bundle(tmp_path / "other")
    (other / "unsafe-link").symlink_to(other / "execution_context.json")
    with pytest.raises(DownloadedOfflineValidationError, match="symlink"):
        validate_downloaded_offline_bundle(other)


def test_downloaded_offline_bundle_rejects_boolean_returncode(tmp_path) -> None:
    bundle = _offline_bundle(tmp_path)
    result_path = bundle / "remote_action_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["returncode"] = False
    _write_json(result_path, result)
    _refresh_outer_manifest(bundle)

    with pytest.raises(
        DownloadedOfflineValidationError,
        match="remote action returncode must be a non-negative integer",
    ):
        validate_downloaded_offline_bundle(bundle)


@pytest.mark.parametrize("exception_type", ("TimeoutError", "OSError"))
def test_downloaded_offline_bundle_rejects_unproven_network_denial(
    tmp_path: Path,
    exception_type: str,
) -> None:
    bundle = _offline_bundle(tmp_path)
    probe_path = bundle / "provider_free_network_denial_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    probe["exception_type"] = exception_type
    _write_json(probe_path, probe)
    _refresh_outer_manifest(bundle)

    with pytest.raises(
        ValueError,
        match="network denial probe exception classification is unsafe",
    ):
        validate_downloaded_offline_bundle(bundle)


def test_downloaded_offline_summary_rejects_schema_and_boolean_adversaries(
    tmp_path,
) -> None:
    bundle = _offline_bundle(tmp_path)
    summary_path = _offline_summary_path(bundle)
    baseline = json.loads(summary_path.read_text(encoding="utf-8"))

    cases = (
        (
            "missing",
            lambda payload: payload["no_search"].pop("request_count"),
            "no-search evidence fields differ from the frozen schema",
        ),
        (
            "extra",
            lambda payload: payload.__setitem__("unexpected", "extension"),
            "offline smoke summary fields differ from the frozen schema",
        ),
        (
            "mistyped",
            lambda payload: payload["no_search"].__setitem__("request_count", "1"),
            "no-search request_count must be a non-negative integer",
        ),
        (
            "boolean-as-integer",
            lambda payload: payload["no_search"].__setitem__("request_count", True),
            "no-search request_count must be a non-negative integer",
        ),
        (
            "scientific-not-boolean",
            lambda payload: payload.__setitem__("scientific", 0),
            "offline smoke scientific must be exactly False",
        ),
        (
            "feedback-visible",
            lambda payload: payload["no_search"].__setitem__(
                "adaptive_feedback_visible_to_backend", True
            ),
            "no-search adaptive feedback visibility must be exactly False",
        ),
        (
            "provider-input-varies",
            lambda payload: payload["no_search"].__setitem__(
                "provider_input_constant", False
            ),
            "no-search provider input constant must be exactly True",
        ),
        (
            "wrong-request-count",
            lambda payload: payload["no_search"].__setitem__("request_count", 0),
            "no-search request count differs from frozen proposal opportunities",
        ),
        (
            "boolean-run-counter",
            lambda payload: payload["runs"][0].__setitem__(
                "seed_evaluations", True
            ),
            "offline smoke run 0 seed_evaluations must be a non-negative integer",
        ),
    )
    for _label, mutate, message in cases:
        payload = json.loads(json.dumps(baseline))
        mutate(payload)
        _write_json(summary_path, payload)
        _refresh_outer_manifest(bundle)
        with pytest.raises(DownloadedOfflineValidationError, match=message):
            validate_downloaded_offline_bundle(bundle)


def test_downloaded_offline_summary_rejects_invalid_ledger_accounting(tmp_path) -> None:
    bundle = _offline_bundle(tmp_path)
    summary_path = _offline_summary_path(bundle)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ledger = summary["no_search"]["ledger"]
    ledger["provider_attempts"] = 0
    ledger["provider_attempts_by_opportunity"] = {"1": 0}
    _write_json(summary_path, summary)
    _refresh_outer_manifest(bundle)

    with pytest.raises(
        DownloadedOfflineValidationError,
        match="no-search ledger does not exactly account for its opportunities",
    ):
        validate_downloaded_offline_bundle(bundle)


def test_downloaded_offline_summary_rejects_duplicate_keys(tmp_path) -> None:
    bundle = _offline_bundle(tmp_path)
    summary_path = _offline_summary_path(bundle)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    encoded = json.dumps(summary, indent=2, sort_keys=True)
    encoded = encoded.replace(
        '"request_count": 1',
        '"request_count": 1,\n    "request_count": 1',
        1,
    )
    summary_path.write_text(encoded + "\n", encoding="utf-8")
    _refresh_outer_manifest(bundle)

    with pytest.raises(DownloadedOfflineValidationError, match="duplicate key"):
        validate_downloaded_offline_bundle(bundle)


def test_downloaded_offline_summary_rejects_symlink(tmp_path) -> None:
    bundle = _offline_bundle(tmp_path)
    summary_path = _offline_summary_path(bundle)
    target = summary_path.with_name("summary-target.json")
    summary_path.replace(target)
    summary_path.symlink_to(target.name)

    with pytest.raises(DownloadedOfflineValidationError, match="symlink"):
        validate_downloaded_offline_bundle(bundle)


def test_downloaded_offline_summary_rejects_absolute_and_credential_fields(
    tmp_path,
) -> None:
    bundle = _offline_bundle(tmp_path)
    summary_path = _offline_summary_path(bundle)
    baseline = json.loads(summary_path.read_text(encoding="utf-8"))

    absolute = json.loads(json.dumps(baseline))
    absolute["no_search"]["debug_output_path"] = "/mnt/discovery/private"
    _write_json(summary_path, absolute)
    _refresh_outer_manifest(bundle)
    with pytest.raises(
        DownloadedOfflineValidationError,
        match="executor-absolute path fields",
    ):
        validate_downloaded_offline_bundle(bundle)

    credential = json.loads(json.dumps(baseline))
    credential["no_search"]["api_key"] = "sk-example-not-a-real-key"
    _write_json(summary_path, credential)
    _refresh_outer_manifest(bundle)
    with pytest.raises(
        DownloadedOfflineValidationError,
        match="credential-shaped fields",
    ):
        validate_downloaded_offline_bundle(bundle)


@pytest.mark.parametrize(
    ("field_name", "absolute_path"),
    (
        ("debug_output_path", "/mnt/discovery/runs/private"),
        ("debug_output_path", r"C:\modal\runs\private"),
        ("path", "/opt/rl4rl/private"),
        ("path", "file:///mnt/discovery/private"),
    ),
)
def test_downloaded_v2_offline_bundle_rejects_executor_absolute_paths(
    tmp_path, field_name, absolute_path
) -> None:
    bundle = _offline_bundle(tmp_path)
    state_path = next(bundle.rglob("run_state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state[field_name] = absolute_path
    _write_json(state_path, state)
    _refresh_outer_manifest(bundle)

    with pytest.raises(
        DownloadedOfflineValidationError,
        match="executor-absolute path fields",
    ):
        validate_downloaded_offline_bundle(bundle)
