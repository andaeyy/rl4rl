from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import modal_action_journal
import pytest
from modal_boundary import (
    MODAL_LAUNCH_REJECTION_ROOT,
    MODAL_LIVE_COHORT_ROOT,
    MODAL_REMOTE_RUN_RESERVATION_ROOT,
    ArtifactFileV1,
    ArtifactManifestV1,
    RawArtifactManifestV1,
    build_image_source_manifest,
    download_artifacts,
)
from scripts import launch_modal

ROOT = Path(__file__).resolve().parents[1]
_REAL_MODAL_APPROVAL_VALIDATOR = launch_modal._validate_modal_approval_inputs
_REAL_MODAL_LAUNCH_BINDER = launch_modal._open_modal_launch_bindings
_REAL_PYTHON_EXECUTION_MATERIALIZER = (
    launch_modal._materialize_python_execution_copy
)
_REAL_GLOBAL_JOURNAL_SCANNER = launch_modal._scan_modal_global_action_journal
_REAL_GLOBAL_JOURNAL_GATE = launch_modal._require_modal_global_action_gate_clear
_SOURCE_TREE_SHA256 = "9" * 64
_CUDA_RECEIPT_PATH = "outputs/readiness/test_modal_cuda_receipt.json"
_OFFLINE_RECEIPT_PATH = "outputs/readiness/test_modal_offline_receipt.json"
_ROUND_TRIP_RECEIPT_PATH = "outputs/readiness/test_modal_round_trip_receipt.json"
_PREFLIGHT_RECEIPT_PATH = "outputs/readiness/test_modal_preflight_receipt.json"
_TEST_MACHINE_ID = b"rl4rl-test-machine-identity"
_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS = 1_700_000_000_000_000
_TEST_BOOT_IDENTITY = bytes.fromhex("00112233445566778899aabbccddeeff")
_TEST_NEXT_BOOT_IDENTITY = bytes.fromhex("102132435465768798a9babcbddcedfe")
_TEST_PROCESS_BIRTH_IDENTITY = b"rl4rl-test-process-birth-identity"


class _FakeBoundLaunchFile:
    def __init__(
        self,
        canonical_path: Path,
        descriptor: int,
        *,
        fail_on_validation: int | None = None,
        fail_on_cleanup: bool = False,
    ) -> None:
        self.canonical_path = canonical_path
        self.descriptor = descriptor
        self.fail_on_validation = fail_on_validation
        self.fail_on_cleanup = fail_on_cleanup
        self.validation_count = 0
        self.removed = False

    @property
    def binding(self) -> _FakeBoundLaunchFile:
        return self

    @property
    def execution_path(self) -> str:
        return f"/dev/fd/{self.descriptor}"

    def require_current(self) -> None:
        self.validation_count += 1
        if self.validation_count == self.fail_on_validation:
            raise ValueError("synthetic Python execution copy replacement")

    def close(self) -> None:
        self.descriptor = -1

    def close_and_remove(self) -> None:
        self.close()
        if self.fail_on_cleanup:
            raise ValueError("synthetic Python execution copy cleanup failure")
        self.removed = True


class _FakeModalLaunchBindings:
    def __init__(self, *, fail_on_validation: int | None = None) -> None:
        self.python_executable = _FakeBoundLaunchFile(
            Path("/private/modal-python-runtime/python"),
            900,
        )
        self.modal_executable = _FakeBoundLaunchFile(
            Path(launch_modal.sys.executable).with_name("modal"),
            901,
        )
        self.modal_config = _FakeBoundLaunchFile(Path("/passwd/home/.modal.toml"), 902)
        self.fail_on_validation = fail_on_validation
        self.validation_count = 0
        self.closed = False

    @property
    def pass_fds(self) -> tuple[int, int]:
        return (self.modal_executable.descriptor, self.modal_config.descriptor)

    def require_current(self) -> None:
        self.validation_count += 1
        if self.validation_count == self.fail_on_validation:
            raise ValueError("synthetic held launch binding replacement")

    def close(self) -> None:
        self.closed = True


def _fake_local_freeze_bindings() -> tuple[dict[str, str], ...]:
    source_digest = _SOURCE_TREE_SHA256
    directory = (
        "outputs/readiness/modal_only_final/local_engineering_freezes/"
        f"{source_digest}"
    )
    return (
        {
            "gate": "local_unit_tested",
            "path": f"{directory}/unit_test_evidence_receipt.json",
            "sha256": "1" * 64,
        },
        {
            "gate": "local_offline_smoke_tested",
            "path": f"{directory}/offline_smoke_evidence_receipt.json",
            "sha256": "2" * 64,
        },
        {
            "gate": "local_engineering_freeze_validated",
            "path": f"{directory}/local_engineering_freeze_receipt.json",
            "sha256": "3" * 64,
        },
    )


@pytest.fixture(autouse=True)
def _validated_local_freeze(monkeypatch):
    """Unit tests isolate launcher states from the expensive final receipts."""

    monkeypatch.setattr(
        launch_modal,
        "validate_local_freeze_evidence",
        lambda _root, **_kwargs: _fake_local_freeze_bindings(),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_local_engineering_freeze_receipt",
        lambda *args, **kwargs: {
            "source_tree_sha256": _SOURCE_TREE_SHA256,
            "image_source_sha256": kwargs["expected_image_source_sha256"],
        },
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_modal_approval_inputs",
        lambda arguments, **_kwargs: {
            "modal_cost_cap_usd": arguments.modal_cost_cap_usd,
            "modal_resource_profile": launch_modal.modal_resource_profile(
                arguments.action,
                arguments.harness,
            ),
            "modal_price_basis_path": arguments.modal_price_basis_path,
            "modal_price_basis_sha256": arguments.modal_price_basis_sha256,
            "modal_cost_estimate": _fake_modal_cost_estimate(arguments.action),
        },
    )
    monkeypatch.setattr(
        launch_modal,
        "_open_modal_launch_bindings",
        _FakeModalLaunchBindings,
    )
    monkeypatch.setattr(
        launch_modal,
        "_materialize_python_execution_copy",
        lambda *_args, **_kwargs: _FakeBoundLaunchFile(
            Path("/private/modal-python-runtime/copy"),
            903,
        ),
    )
    monkeypatch.setattr(
        launch_modal,
        "_default_machine_identity_provider",
        lambda: _TEST_MACHINE_ID,
    )
    monkeypatch.setattr(
        launch_modal,
        "_default_boot_session_provider",
        lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
    )
    monkeypatch.setattr(
        launch_modal,
        "_default_boot_identity_provider",
        lambda: _TEST_BOOT_IDENTITY,
    )
    monkeypatch.setattr(
        launch_modal,
        "_default_process_birth_identity_provider",
        lambda _pid: _TEST_PROCESS_BIRTH_IDENTITY,
    )
    # Most launcher unit tests use deliberately non-canonical receipt roots
    # and repeated synthetic attempt IDs.  Global scanner behavior has focused
    # integration tests below; isolate unrelated tests from the real checkout.
    monkeypatch.setattr(
        launch_modal,
        "_scan_modal_global_action_journal",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        launch_modal,
        "_require_modal_global_action_gate_clear",
        lambda *_args, **_kwargs: None,
    )


def _fake_modal_cost_estimate(action: str) -> dict[str, object]:
    runtime = "0.249552" if action == "canaries" else (
        "0.013188" if action in {"offline-smoke", "download", "verify"} else "0.062388"
    )
    storage = "0.090703125" if action == "canaries" else "0.02267578125"
    total = {
        "canaries": "0.366631125",
        "offline-smoke": "0.06223978125",
        "download": "0.06223978125",
        "verify": "0.06223978125",
    }.get(action, "0.11143978125")
    return {
        "runtime_request_rate_estimate_usd": runtime,
        "cache_miss_image_build_request_rate_estimate_usd": "0.026376",
        "new_remote_run_count": 4 if action == "canaries" else 1,
        "per_remote_run_storage_bound_gib": "0.251953125",
        "one_month_storage_estimate_usd": storage,
        "download_transfer_bound_gib": (
            "0.50390625" if action == "download" else "0"
        ),
        "download_transfer_pricing": (
            "not_separately_listed_on_official_pricing_page"
        ),
        "download_transfer_estimate_usd": "0",
        "action_estimate_usd": total,
        "scope": (
            "local_pre_popen_request_rate_and_one_gib_month_storage_estimate_"
            "not_platform_billing_cap"
        ),
    }


def _arguments(**overrides) -> argparse.Namespace:
    default_run_id = "modal-cuda-env-20260809-01"
    values = {
        "action": "cuda-environment",
        "attempt_id": "",
        "run_id": default_run_id,
        "cohort_id": default_run_id,
        "source_run_id": "",
        "verifier_run_id": "",
        "harness": "",
        "local_output": "",
        "expected_image_source_sha256": build_image_source_manifest(
            ROOT
        ).manifest_sha256,
        "outer_cli_timeout_seconds": (
            launch_modal.expected_outer_cli_timeout_seconds("cuda-environment")
        ),
        "modal_cost_cap_usd": "0.25",
        "modal_price_basis_path": (
            "outputs/readiness/modal_only_final/modal_price_bases/"
            + "d" * 64
            + "/20260810T000000Z.json"
        ),
        "modal_price_basis_sha256": "8" * 64,
        "approved": True,
        "provider_approved": False,
        "provider_cost_cap_usd": "",
        "provider_approval_plan_path": "",
        "approval_plan_sha256": "",
        "provider_price_basis_path": "",
        "provider_price_basis_sha256": "",
        "source_action_attempt_receipt_path": "",
        "source_action_attempt_receipt_sha256": "",
        "source_evidence_recovery": False,
        "cuda_receipt_path": "",
        "cuda_receipt_sha256": "",
        "offline_smoke_receipt_path": "",
        "offline_smoke_receipt_sha256": "",
        "artifact_round_trip_receipt_path": "",
        "artifact_round_trip_receipt_sha256": "",
        "candidate_resume_preflight_receipt_path": "",
        "candidate_resume_preflight_receipt_sha256": "",
    }
    values.update(overrides)
    if "cohort_id" not in overrides:
        values["cohort_id"] = (
            values["run_id"]
            if values["action"] == "cuda-environment"
            else default_run_id
        )
    predecessor_defaults = {
        "offline-smoke": {
            "cuda_receipt_path": _CUDA_RECEIPT_PATH,
            "cuda_receipt_sha256": "f" * 64,
        },
        "candidate-smoke": {
            "cuda_receipt_path": _CUDA_RECEIPT_PATH,
            "cuda_receipt_sha256": "f" * 64,
            "offline_smoke_receipt_path": _OFFLINE_RECEIPT_PATH,
            "offline_smoke_receipt_sha256": "f" * 64,
        },
        "checkpoint-resume": {
            "artifact_round_trip_receipt_path": _ROUND_TRIP_RECEIPT_PATH,
            "artifact_round_trip_receipt_sha256": "f" * 64,
        },
        "canary": {
            "candidate_resume_preflight_receipt_path": _PREFLIGHT_RECEIPT_PATH,
            "candidate_resume_preflight_receipt_sha256": "f" * 64,
        },
        "canaries": {
            "candidate_resume_preflight_receipt_path": _PREFLIGHT_RECEIPT_PATH,
            "candidate_resume_preflight_receipt_sha256": "f" * 64,
        },
    }.get(values["action"], {})
    for field, value in predecessor_defaults.items():
        if field not in overrides:
            values[field] = value
    if (
        "verifier_run_id" not in overrides
        and values["action"] in {"download", "verify"}
    ):
        values["verifier_run_id"] = "modal-verifier-20260809-01"
    if (
        "source_action_attempt_receipt_path" not in overrides
        and values["action"] in {"download", "verify"}
    ):
        values["source_action_attempt_receipt_path"] = (
            launch_modal.modal_action_terminal_receipt_path(
                launch_modal.ModalLiveCohortIdentity(
                    source_tree_sha256=_SOURCE_TREE_SHA256,
                    image_source_sha256=values[
                        "expected_image_source_sha256"
                    ],
                    cohort_id=values["cohort_id"],
                ),
                "c" * 32,
            ).as_posix()
        )
    if (
        "source_action_attempt_receipt_sha256" not in overrides
        and values["action"] in {"download", "verify"}
    ):
        values["source_action_attempt_receipt_sha256"] = "f" * 64
    if "outer_cli_timeout_seconds" not in overrides and values["action"] == "canaries":
        values["outer_cli_timeout_seconds"] = (
            launch_modal.expected_outer_cli_timeout_seconds("canaries")
        )
    if values["action"] in {"canary", "canaries"}:
        provider_defaults = {
            "provider_cost_cap_usd": "2.00",
            "provider_approval_plan_path": (
                "outputs/readiness/provider_canary_approval/plan.json"
            ),
            "approval_plan_sha256": "a" * 64,
            "provider_price_basis_path": (
                "outputs/readiness/modal_resource_cleanup/price.json"
            ),
            "provider_price_basis_sha256": "b" * 64,
        }
        for field, value in provider_defaults.items():
            if field not in overrides:
                values[field] = value
    return argparse.Namespace(**values)


class _FakeProcess:
    pid = 424242

    def __init__(
        self,
        *,
        returncode: int = 0,
        wait_error: BaseException | None = None,
    ) -> None:
        self.returncode = returncode
        self.wait_error = wait_error
        self.wait_timeout: int | None = None

    def wait(self, *, timeout: int) -> int:
        self.wait_timeout = timeout
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode


def _terminal_receipt_path(directory: Path) -> Path:
    paths = tuple(
        path
        for path in directory.glob("*.json")
        if not path.name.endswith(".intent.json")
    )
    [path] = paths
    return path


def _intent_path(directory: Path) -> Path:
    [path] = directory.glob("*.intent.json")
    return path


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _prepare_minimal_production_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    image_source_sha256: str = "d" * 64,
) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_source_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    return project


def _assert_exact_attempt_receipt_schema(payload: dict[str, object]) -> None:
    assert set(payload) == {
        field.name
        for field in launch_modal.dataclass_fields(
            launch_modal.ModalActionAttemptReceipt
        )
    }


def _modal_price_basis(
    project: Path,
    image_source_sha256: str,
    *,
    retrieved_at_utc: str = "2026-08-10T00:00:00Z",
) -> tuple[str, str, dict[str, object]]:
    payload: dict[str, object] = {
        "schema_name": "ModalPriceBasis",
        "schema_version": "1.0",
        "image_source_sha256": image_source_sha256,
        "official_source_url": "https://modal.com/pricing",
        "retrieved_at_utc": retrieved_at_utc,
        "region": None,
        "cpu_usd_per_core_second": "0.0000131",
        "memory_usd_per_gib_second": "0.00000222",
        "t4_usd_per_gpu_second": "0.000164",
        "volume_storage_usd_per_gib_month": "0.09",
        "included_volume_storage_gib_per_month": "1024",
        "download_transfer_pricing": (
            "not_separately_listed_on_official_pricing_page"
        ),
    }
    logical = launch_modal.modal_readiness.modal_price_basis_logical_path(
        image_source_sha256,
        retrieved_at_utc,
    ).as_posix()
    return logical, _write_json(project / logical, payload), payload


def _test_local_containment_fields(project: Path) -> dict[str, object]:
    anchor_id = "7" * 64
    payload = {
        "schema_name": "ModalLocalHostAnchor",
        "schema_version": "1.0",
        "anchor_id": anchor_id,
        "machine_binding_sha256": launch_modal._local_machine_binding_sha256(
            anchor_id,
            _TEST_MACHINE_ID,
        ),
    }
    logical = launch_modal.modal_local_host_anchor_path().as_posix()
    path = project.joinpath(*Path(logical).parts)
    if path.exists():
        raw = path.read_bytes()
        anchor_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        anchor_sha256 = _write_json(path, payload)
        path.parent.chmod(0o700)
    boot_sha256 = launch_modal._local_boot_session_sha256(
        anchor_sha256,
        _TEST_BOOT_IDENTITY,
    )
    return {
        "local_host_anchor_path": logical,
        "local_host_anchor_sha256": anchor_sha256,
        "local_boot_started_at_unix_microseconds": (
            _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS
        ),
        "local_boot_session_sha256": boot_sha256,
    }


@pytest.mark.parametrize(
    ("action", "harness", "expected_total"),
    (
        ("offline-smoke", None, "0.06223978125"),
        ("verify", None, "0.06223978125"),
        ("download", None, "0.06223978125"),
        ("cuda-environment", None, "0.11143978125"),
        ("canary", "greedy_autoresearch", "0.11143978125"),
        ("canaries", None, "0.366631125"),
    ),
)
def test_modal_cost_estimate_matches_frozen_price_basis(
    tmp_path: Path,
    action: str,
    harness: str | None,
    expected_total: str,
) -> None:
    image_sha256 = "d" * 64
    _logical, _digest, price_basis = _modal_price_basis(
        tmp_path,
        image_sha256,
    )
    estimate = launch_modal.modal_readiness.derive_modal_action_cost_estimate(
        action=action,
        harness=harness,
        resource_profile=launch_modal.modal_resource_profile(action, harness),
        price_basis=price_basis,
    )

    assert estimate["action_estimate_usd"] == expected_total
    assert (
        estimate["cache_miss_image_build_request_rate_estimate_usd"]
        == "0.026376"
    )
    assert estimate["per_remote_run_storage_bound_gib"] == "0.251953125"
    assert estimate["download_transfer_bound_gib"] == (
        "0.50390625" if action == "download" else "0"
    )
    assert estimate["download_transfer_estimate_usd"] == "0"


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("call_count", True),
        ("retries", False),
        ("max_containers", True),
        ("min_containers", 0.0),
        ("cpu_request_cores", 2),
        ("cpu_soft_limit_cores", 2),
    ),
)
def test_modal_cost_estimate_rejects_numeric_type_substitutions(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    _logical, _digest, price_basis = _modal_price_basis(tmp_path, "d" * 64)
    profile = deepcopy(launch_modal.modal_resource_profile("offline-smoke"))
    profile["runtime_function_calls"][0][field] = replacement

    with pytest.raises(ValueError, match="resource profile changed"):
        launch_modal.modal_readiness.derive_modal_action_cost_estimate(
            action="offline-smoke",
            harness=None,
            resource_profile=profile,
            price_basis=price_basis,
        )


@pytest.mark.parametrize(
    ("action", "harness", "approved_cap"),
    (
        ("offline-smoke", "", "0.06223978125"),
        ("offline-smoke", "", "0.07"),
        ("cuda-environment", "", "0.11143978125"),
        ("canaries", "", "0.366631125"),
        ("canaries", "", "0.50"),
    ),
)
def test_modal_approval_accepts_equal_or_higher_source_bound_cap(
    tmp_path: Path,
    action: str,
    harness: str,
    approved_cap: str,
) -> None:
    image_sha256 = "d" * 64
    retrieved = datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )
    logical, digest, _payload = _modal_price_basis(
        tmp_path,
        image_sha256,
        retrieved_at_utc=retrieved,
    )
    arguments = _arguments(
        action=action,
        harness=harness,
        modal_cost_cap_usd=approved_cap,
        modal_price_basis_path=logical,
        modal_price_basis_sha256=digest,
    )

    approval = _REAL_MODAL_APPROVAL_VALIDATOR(
        arguments,
        project_root=tmp_path,
        image_source_sha256=image_sha256,
    )

    assert approval["modal_cost_cap_usd"] == approved_cap
    assert approval["modal_price_basis_sha256"] == digest


@pytest.mark.parametrize(
    ("action", "rejected_cap"),
    (
        ("offline-smoke", "0.06223978124"),
        ("cuda-environment", "0.11143978124"),
        ("canaries", "0.366631124"),
    ),
)
def test_modal_approval_rejects_cap_below_source_bound_estimate(
    tmp_path: Path,
    action: str,
    rejected_cap: str,
) -> None:
    image_sha256 = "d" * 64
    retrieved = datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )
    logical, digest, _payload = _modal_price_basis(
        tmp_path,
        image_sha256,
        retrieved_at_utc=retrieved,
    )
    arguments = _arguments(
        action=action,
        modal_cost_cap_usd=rejected_cap,
        modal_price_basis_path=logical,
        modal_price_basis_sha256=digest,
    )

    with pytest.raises(ValueError, match="below the source-bound action estimate"):
        _REAL_MODAL_APPROVAL_VALIDATOR(
            arguments,
            project_root=tmp_path,
            image_source_sha256=image_sha256,
        )


@pytest.mark.parametrize("mutation", ("raw_hash", "missing_field"))
def test_modal_approval_rejects_tampered_price_basis(
    tmp_path: Path,
    mutation: str,
) -> None:
    image_sha256 = "d" * 64
    retrieved = datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )
    logical, digest, payload = _modal_price_basis(
        tmp_path,
        image_sha256,
        retrieved_at_utc=retrieved,
    )
    if mutation == "raw_hash":
        payload["cpu_usd_per_core_second"] = "0.0000132"
        _write_json(tmp_path / logical, payload)
        expected = "raw SHA-256 changed"
    else:
        payload.pop("download_transfer_pricing")
        digest = _write_json(tmp_path / logical, payload)
        expected = "invalid exact schema"
    arguments = _arguments(
        modal_price_basis_path=logical,
        modal_price_basis_sha256=digest,
    )

    with pytest.raises(ValueError, match=expected):
        _REAL_MODAL_APPROVAL_VALIDATOR(
            arguments,
            project_root=tmp_path,
            image_source_sha256=image_sha256,
        )


def _source_attempt_pair(
    project: Path,
    *,
    action: str = "cuda-environment",
    run_id: str = "source-run-1",
    status: str = "succeeded",
    returncode: int | None = 0,
    failure_kind: str | None = None,
    process_group_closed: bool | None = True,
    image_source_sha256: str = "d" * 64,
) -> tuple[str, dict[str, object], dict[str, object]]:
    attempt_id = "c" * 32
    provider_action = action in {"canary", "canaries"}
    harness = "greedy_autoresearch" if action == "canary" else None
    concrete = (
        tuple(
            f"{run_id}-{launch_modal.canary_run_suffix(item)}"
            for item in launch_modal.CANARY_ORDER
        )
        if action == "canaries"
        else (run_id,)
    )
    predecessor_gates = {
        "cuda-environment": (),
        "offline-smoke": ("modal_cuda_environment_validated",),
        "candidate-smoke": (
            "modal_cuda_environment_validated",
            "modal_offline_smoke_validated",
        ),
        "checkpoint-resume": ("modal_artifact_round_trip_validated",),
        "canary": ("candidate_resume_preflight_validated",),
        "canaries": ("candidate_resume_preflight_validated",),
    }[action]
    predecessor_receipts = (*_fake_local_freeze_bindings(), *tuple(
        {
            "gate": gate,
            "path": f"outputs/readiness/{gate}.json",
            "sha256": "f" * 64,
        }
        for gate in predecessor_gates
    ))
    modal_price_path, modal_price_sha256, modal_price = _modal_price_basis(
        project,
        image_source_sha256,
    )
    resource_profile = launch_modal.modal_resource_profile(action, harness or "")
    modal_cost_estimate = (
        launch_modal.modal_readiness.derive_modal_action_cost_estimate(
            action=action,
            harness=harness,
            resource_profile=resource_profile,
            price_basis=modal_price,
        )
    )
    modal_cap = "0.50" if action == "canaries" else "0.25"
    cohort_id = run_id if action == "cuda-environment" else "source-cohort-1"
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_source_sha256,
        cohort_id=cohort_id,
    )
    source_run_id = (
        "candidate-source-1" if action == "checkpoint-resume" else None
    )
    modal_command_sha256 = launch_modal._modal_command_sha256(
        launch_modal.build_modal_cli_command(
            python_executable=launch_modal.sys.executable,
            project_root=project,
            action=action,
            run_id=run_id,
            source_run_id=source_run_id,
            verifier_run_id=None,
            harness=harness,
            source_tree_sha256=identity.source_tree_sha256,
            cohort_id=identity.cohort_id,
            image_source_sha256=identity.image_source_sha256,
            provider_approved=provider_action,
        )
    )
    launch_capability_sha256 = "e" * 64
    containment = _test_local_containment_fields(project)
    remote_run_reservations = tuple(
        binding
        for binding, _payload in launch_modal._remote_run_reservation_specs(
            concrete_remote_run_ids=concrete,
            attempt_id=attempt_id,
            action=action,
            identity=identity,
            created_at_utc="2026-08-10T00:00:00Z",
            launch_capability_sha256=launch_capability_sha256,
            **containment,
        )
    )
    intent = launch_modal.ModalActionIntent(
        schema_name="ModalActionIntent",
        schema_version="1.6",
        attempt_id=attempt_id,
        created_at_utc="2026-08-10T00:00:00Z",
        action=action,
        run_id=run_id,
        concrete_remote_run_ids=concrete,
        remote_run_reservations=remote_run_reservations,
        **containment,
        source_run_id=source_run_id,
        verifier_run_id=None,
        harness=harness,
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        approved_image_source_sha256=image_source_sha256,
        modal_command_sha256=modal_command_sha256,
        launch_capability_sha256=launch_capability_sha256,
        modal_profile="scalingintelligence",
        modal_environment="main",
        outer_cli_timeout_seconds=launch_modal.expected_outer_cli_timeout_seconds(
            action
        ),
        modal_cost_cap_usd=modal_cap,
        modal_resource_profile=resource_profile,
        modal_price_basis_path=modal_price_path,
        modal_price_basis_sha256=modal_price_sha256,
        modal_cost_estimate=modal_cost_estimate,
        modal_cost_approved=True,
        provider_cost_approved=provider_action,
        provider_cost_cap_usd="2.00" if provider_action else None,
        provider_approval_plan_path=(
            "outputs/readiness/provider_canary_approval/plan.json"
            if provider_action
            else None
        ),
        approval_plan_sha256="a" * 64 if provider_action else None,
        provider_price_basis_path=(
            "outputs/readiness/modal_resource_cleanup/price.json"
            if provider_action
            else None
        ),
        provider_price_basis_sha256="b" * 64 if provider_action else None,
        predecessor_receipts=predecessor_receipts,
        source_evidence_recovery=False,
    )
    directory = project.joinpath(
        *launch_modal.modal_action_attempt_directory(identity).parts
    )
    intent_path = directory / f"{attempt_id}.intent.json"
    intent_sha256 = _write_json(intent_path, asdict(intent))
    process_id = 424242
    process_start = launch_modal.ModalLocalProcessStartReceipt(
        schema_name="ModalLocalProcessStart",
        schema_version="1.1",
        attempt_id=attempt_id,
        created_at_utc="2026-08-10T00:00:00.500000Z",
        action=action,
        run_id=run_id,
        intent_path=launch_modal.modal_action_intent_receipt_path(
            identity,
            attempt_id,
        ).as_posix(),
        intent_sha256=intent_sha256,
        source_tree_sha256=identity.source_tree_sha256,
        image_source_sha256=image_source_sha256,
        cohort_id=identity.cohort_id,
        modal_command_sha256=modal_command_sha256,
        launch_capability_sha256=launch_capability_sha256,
        modal_cost_cap_usd=modal_cap,
        provider_cost_cap_usd="2.00" if provider_action else None,
        **containment,
        process_id=process_id,
        expected_process_group_id=process_id,
        expected_session_id=process_id,
        process_birth_identity_sha256=(
            launch_modal._process_birth_identity_sha256(
                local_boot_session_sha256=containment[
                    "local_boot_session_sha256"
                ],
                process_id=process_id,
                process_birth_identity=_TEST_PROCESS_BIRTH_IDENTITY,
            )
        ),
    )
    process_start_path = project.joinpath(
        *launch_modal.modal_local_process_start_receipt_path(attempt_id).parts
    )
    process_start_sha256 = _write_json(process_start_path, asdict(process_start))
    process_start_path.parent.chmod(0o700)
    terminal = launch_modal.ModalActionAttemptReceipt(
        schema_name="ModalActionAttemptReceipt",
        schema_version="3.6",
        attempt_id=attempt_id,
        started_at_utc="2026-08-10T00:00:00Z",
        finished_at_utc="2026-08-10T00:00:01Z",
        status=status,
        failure_kind=failure_kind,
        action=action,
        run_id=run_id,
        concrete_remote_run_ids=concrete,
        remote_run_reservations=remote_run_reservations,
        **containment,
        source_run_id=intent.source_run_id,
        verifier_run_id=None,
        harness=harness,
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        approved_image_source_sha256=image_source_sha256,
        modal_command_sha256=modal_command_sha256,
        launch_capability_sha256=intent.launch_capability_sha256,
        modal_profile="scalingintelligence",
        modal_environment="main",
        outer_cli_timeout_seconds=intent.outer_cli_timeout_seconds,
        modal_cost_cap_usd=modal_cap,
        modal_resource_profile=intent.modal_resource_profile,
        modal_price_basis_path=intent.modal_price_basis_path,
        modal_price_basis_sha256=intent.modal_price_basis_sha256,
        modal_cost_estimate=intent.modal_cost_estimate,
        modal_cost_approved=True,
        provider_cost_approved=provider_action,
        provider_cost_cap_usd=intent.provider_cost_cap_usd,
        provider_approval_plan_path=intent.provider_approval_plan_path,
        approval_plan_sha256=intent.approval_plan_sha256,
        provider_price_basis_path=intent.provider_price_basis_path,
        provider_price_basis_sha256=intent.provider_price_basis_sha256,
        predecessor_receipts=predecessor_receipts,
        source_evidence_recovery=False,
        local_process_start_receipt_path=(
            launch_modal.modal_local_process_start_receipt_path(
                attempt_id
            ).as_posix()
        ),
        local_process_start_receipt_sha256=process_start_sha256,
        local_process_id=process_id,
        local_process_group_id=process_id,
        local_session_id=process_id,
        modal_cli_process_started=True,
        remote_execution_state="may_have_started",
        returncode=returncode,
        process_group_closed=process_group_closed,
    )
    terminal_path = directory / f"{attempt_id}.json"
    _write_json(terminal_path, asdict(terminal))
    return (
        terminal_path.relative_to(project).as_posix(),
        json.loads((directory / f"{attempt_id}.intent.json").read_text()),
        json.loads(terminal_path.read_text()),
    )


@pytest.mark.parametrize("replacement", (True, 1.0))
def test_source_intent_rejects_estimate_numeric_type_substitution(
    tmp_path: Path,
    replacement: object,
) -> None:
    _terminal_path, intent, _terminal = _source_attempt_pair(tmp_path)
    intent["modal_cost_estimate"]["new_remote_run_count"] = replacement

    with pytest.raises(ValueError, match="Modal cost estimate changed"):
        launch_modal._validate_source_action_intent(
            intent,
            attempt_id="c" * 32,
            identity=launch_modal.ModalLiveCohortIdentity(
                source_tree_sha256=_SOURCE_TREE_SHA256,
                image_source_sha256="d" * 64,
                cohort_id="source-run-1",
            ),
            project_root=tmp_path,
        )


def test_source_intent_rejects_noncanonical_modal_environment(
    tmp_path: Path,
) -> None:
    _terminal_path, intent, _terminal = _source_attempt_pair(tmp_path)
    intent["modal_environment"] = "staging"

    with pytest.raises(ValueError, match="wrong Modal environment"):
        launch_modal._validate_source_action_intent(
            intent,
            attempt_id="c" * 32,
            identity=launch_modal.ModalLiveCohortIdentity(
                source_tree_sha256=_SOURCE_TREE_SHA256,
                image_source_sha256="d" * 64,
                cohort_id="source-run-1",
            ),
            project_root=tmp_path,
        )


def test_source_terminal_rejects_numeric_type_mismatch_in_shared_profile(
    tmp_path: Path,
) -> None:
    _terminal_path, intent, terminal = _source_attempt_pair(tmp_path)
    terminal["modal_resource_profile"] = deepcopy(
        terminal["modal_resource_profile"]
    )
    terminal["modal_resource_profile"]["runtime_function_calls"][0][
        "call_count"
    ] = True

    with pytest.raises(ValueError, match="intent and terminal receipt differ"):
        launch_modal._validate_source_action_terminal(
            terminal,
            intent=intent,
            attempt_id="c" * 32,
        )


def _provider_approval_files(
    project: Path,
    *,
    image_source_sha256: str = "d" * 64,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    harnesses = [
        {
            "harness": harness,
            "maximum_attempts": 1,
            "request_settings": {"max_completion_tokens": 200},
            "first_opportunity": {"conservative_input_token_ceiling": 100},
        }
        for harness in launch_modal.CANARY_ORDER
    ]
    plan: dict[str, object] = {
        "schema_name": "ProviderCanaryApprovalPlan",
        "schema_version": "1.2",
        "approval_plan_sha256_scope": (
            "canonical_json_sha256_of_complete_payload_excluding_"
            "approval_plan_sha256"
        ),
        "image_source_sha256": image_source_sha256,
        "harnesses": harnesses,
    }
    plan["approval_plan_sha256"] = launch_modal.canonical_sha256(plan)
    plan_logical = "outputs/readiness/provider_canary_approval/plan.json"
    _write_json(project / plan_logical, plan)

    cuda_run_id = "modal-cuda-environment-accepted"
    cohort_id = "modal-cuda-env-20260809-01"
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_source_sha256,
        cohort_id=cohort_id,
    )
    binding_sha256 = "7" * 64
    preflight_logical = (
        launch_modal.modal_readiness.modal_candidate_resume_preflight_receipt_path(
            identity,
            binding_sha256,
        ).as_posix()
    )
    preflight = {
        "schema_name": "CandidateResumePreflightReceipt",
        "schema_version": "2.0",
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": image_source_sha256,
        "cohort_id": identity.cohort_id,
        "binding_sha256": binding_sha256,
        "cuda_environment": {"run_id": cuda_run_id},
    }
    _write_json(
        project / preflight_logical,
        preflight,
    )
    preflight_sha256 = hashlib.sha256(
        (project / preflight_logical).read_bytes()
    ).hexdigest()
    plan["source_tree_sha256"] = identity.source_tree_sha256
    plan["cohort_id"] = identity.cohort_id
    plan["candidate_resume_preflight_receipt"] = {
        "path": preflight_logical,
        "sha256": preflight_sha256,
    }
    plan.pop("approval_plan_sha256")
    plan["approval_plan_sha256"] = launch_modal.canonical_sha256(plan)
    _write_json(project / plan_logical, plan)
    price_logical = launch_modal.modal_readiness.modal_provider_price_basis_path(
        identity
    ).as_posix()
    price = {
        "schema_name": "ProviderPriceBasis",
        "schema_version": "1.0",
        "model": launch_modal.TARGET_MODEL,
        "official_source_url": "https://openai.com/api/pricing/",
        "retrieved_at_utc": (
            launch_modal._now_utc().isoformat().replace("+00:00", "Z")
        ),
        "uncached_input_usd_per_million_tokens": "1",
        "output_usd_per_million_tokens": "1",
        "per_request_fee_usd": "0",
    }
    price_sha256 = _write_json(project / price_logical, price)
    overrides: dict[str, object] = {
        "action": "canaries",
        "run_id": "provider-cohort-1",
        "cohort_id": cohort_id,
        "provider_approved": True,
        "expected_image_source_sha256": image_source_sha256,
        "provider_cost_cap_usd": "2.00",
        "provider_approval_plan_path": plan_logical,
        "approval_plan_sha256": plan["approval_plan_sha256"],
        "provider_price_basis_path": price_logical,
        "provider_price_basis_sha256": price_sha256,
        "candidate_resume_preflight_receipt_path": preflight_logical,
        "candidate_resume_preflight_receipt_sha256": preflight_sha256,
    }
    return plan, price, overrides


def test_launcher_imports_no_modal_sdk() -> None:
    tree = ast.parse((ROOT / "scripts" / "launch_modal.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    assert "modal" not in imported
    assert "modal_app" not in imported


def test_valid_launch_uses_an_exact_nonambient_child_environment() -> None:
    source_hash = build_image_source_manifest(ROOT).manifest_sha256
    command, environment = launch_modal.build_launch(
        _arguments(expected_image_source_sha256=source_hash),
        environment={
            "PATH": "/usr/bin:/bin",
            "DISCOVERY_API_KEY": "local-provider-secret",
            "TINKER_API_KEY": "local-tinker-secret",
            "MODAL_TOKEN_ID": "profile-compatible-id",
            "MODAL_TOKEN_SECRET": "must-not-override-profile",
            "MODAL_CONFIG_PATH": "/tmp/ambient-modal-config",
            "MODAL_ENVIRONMENT": "wrong-workspace",
            "MODAL_FORCE_BUILD": "1",
            "MODAL_IGNORE_CACHE": "1",
            "MODAL_SERVER_URL": "https://wrong.invalid",
            "MODAL_PROFILE": "wrong-ambient-profile",
            "HTTP_PROXY": "http://wrong.invalid",
            "https_proxy": "http://wrong.invalid",
            "REQUESTS_CA_BUNDLE": "/tmp/wrong-ca.pem",
            "PIP_INDEX_URL": "https://wrong.invalid/simple",
            "UV_INDEX_URL": "https://wrong.invalid/simple",
            "PYTHONHOME": "/tmp/wrong-python",
            "PYTHONPATH": "/tmp/unrelated-checkout",
            "PYTHONINSPECT": "1",
            "DYLD_INSERT_LIBRARIES": "/tmp/injected.dylib",
            "LD_PRELOAD": "/tmp/injected.so",
            "HOME": "/tmp/alternate-modal-home",
            "OPENAI_API_KEY": "ambient-openai-secret",
            "AWS_SECRET_ACCESS_KEY": "ambient-cloud-secret",
            "UNRELATED_SECRET": "ambient-unrelated-secret",
            launch_modal.MODAL_ACTION_ATTEMPT_ID_ENV: "f" * 32,
        },
        nonce_factory=lambda _size: "a" * 64,
    )

    assert command == [
        str(Path(launch_modal.sys.executable).with_name("modal")),
        "run",
        "--env",
        "main",
        str(ROOT / "modal_app.py"),
        "--action",
        "cuda-environment",
        "--run-id",
        "modal-cuda-env-20260809-01",
        "--source-tree-sha256",
        _SOURCE_TREE_SHA256,
        "--cohort-id",
        "modal-cuda-env-20260809-01",
        "--expected-image-source-sha256",
        source_hash,
        "--approved",
    ]
    assert environment[launch_modal.MODAL_LAUNCH_NONCE_ENV] == "a" * 64
    assert environment[launch_modal.MODAL_LAUNCH_SOURCE_ENV] == source_hash
    assert environment[launch_modal.MODAL_LAUNCH_SOURCE_TREE_ENV] == (
        _SOURCE_TREE_SHA256
    )
    assert environment[launch_modal.MODAL_LAUNCH_COHORT_ENV] == (
        "modal-cuda-env-20260809-01"
    )
    assert environment[launch_modal.MODAL_PROFILE_ENV] == (launch_modal.MODAL_PROFILE)
    assert environment[launch_modal.MODAL_ENVIRONMENT_ENV] == "main"
    assert set(environment) == {
        *launch_modal._PAID_MODAL_BASE_ENVIRONMENT,
        launch_modal.MODAL_LAUNCH_NONCE_ENV,
        launch_modal.MODAL_LAUNCH_SOURCE_ENV,
        launch_modal.MODAL_LAUNCH_SOURCE_TREE_ENV,
        launch_modal.MODAL_LAUNCH_COHORT_ENV,
        launch_modal.MODAL_PROFILE_ENV,
        launch_modal.MODAL_ENVIRONMENT_ENV,
        "PYTHONPATH",
    }
    assert environment["PATH"] == os.defpath
    assert environment["PYTHONPATH"] == os.pathsep.join(
        (str(ROOT), str(launch_modal._canonical_venv_site_packages()))
    )
    assert not {
        "HOME",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "UNRELATED_SECRET",
        "PYTHONINSPECT",
        "DYLD_INSERT_LIBRARIES",
        "LD_PRELOAD",
        "DISCOVERY_API_KEY",
        "TINKER_API_KEY",
    } & set(environment)


def test_modal_config_binding_ignores_ambient_home_and_never_hashes_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_home = tmp_path / "passwd-home"
    canonical_home.mkdir()
    config = canonical_home / ".modal.toml"
    config.write_text(
        "[scalingintelligence]\ntoken_id='sensitive'\ntoken_secret='secret'\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    monkeypatch.setenv("HOME", str(tmp_path / "attacker-home"))

    binding = launch_modal._open_modal_config_binding(
        passwd_lookup=lambda _uid: SimpleNamespace(pw_dir=str(canonical_home)),
    )
    try:
        assert binding.canonical_path == config
        assert binding.execution_path == f"/dev/fd/{binding.descriptor}"
        assert binding.sha256 is None
        launch_modal._require_held_launch_file_binding(binding)
    finally:
        binding.close()


def test_modal_config_binding_rejects_symlink_and_replacement(
    tmp_path: Path,
) -> None:
    canonical_home = tmp_path / "passwd-home"
    canonical_home.mkdir()
    target = tmp_path / "real-modal.toml"
    target.write_text("[scalingintelligence]\n", encoding="utf-8")
    target.chmod(0o600)
    config = canonical_home / ".modal.toml"
    config.symlink_to(target)

    def lookup(_uid: int) -> SimpleNamespace:
        return SimpleNamespace(pw_dir=str(canonical_home))

    with pytest.raises(ValueError, match="metadata is unsafe"):
        launch_modal._open_modal_config_binding(passwd_lookup=lookup)

    config.unlink()
    config.write_text("[scalingintelligence]\n", encoding="utf-8")
    config.chmod(0o600)
    binding = launch_modal._open_modal_config_binding(passwd_lookup=lookup)
    try:
        config.rename(canonical_home / "original-modal.toml")
        config.write_text("[scalingintelligence]\n", encoding="utf-8")
        config.chmod(0o600)
        with pytest.raises(ValueError, match="changed"):
            launch_modal._require_held_launch_file_binding(binding)
    finally:
        binding.close()


def test_modal_executable_binding_rejects_symlink_and_replacement(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real-modal"
    real.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real.chmod(0o700)
    linked = tmp_path / "modal-link"
    linked.symlink_to(real)

    def version_lookup(_name: str) -> str:
        return launch_modal.MODAL_VERSION

    with pytest.raises(ValueError, match="metadata is unsafe"):
        launch_modal._open_modal_executable_binding(
            linked,
            version_lookup=version_lookup,
        )

    binding = launch_modal._open_modal_executable_binding(
        real,
        version_lookup=version_lookup,
    )
    try:
        assert binding.sha256 == hashlib.sha256(real.read_bytes()).hexdigest()
        real.rename(tmp_path / "original-modal")
        real.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        real.chmod(0o700)
        with pytest.raises(ValueError, match="changed"):
            launch_modal._require_held_launch_file_binding(binding)
    finally:
        binding.close()


def test_python_execution_copy_is_create_only_private_and_byte_exact(
    tmp_path: Path,
) -> None:
    readiness = tmp_path / "outputs" / "readiness"
    readiness.mkdir(parents=True)
    source_path = tmp_path / "python-source"
    source_path.write_bytes(b"synthetic-python-executable\n")
    source_path.chmod(0o500)
    source = launch_modal._open_python_executable_binding(source_path)
    execution = _REAL_PYTHON_EXECUTION_MATERIALIZER(
        source,
        project_root=tmp_path,
        attempt_id="e" * 32,
    )
    try:
        assert execution.canonical_path == (
            tmp_path
            / launch_modal._MODAL_PYTHON_RUNTIME_ROOT
            / ("e" * 32)
            / "python"
        )
        assert execution.binding.sha256 == source.sha256
        assert execution.binding.size_bytes == source.size_bytes
        assert execution.binding.mode == 0o500
        assert stat.S_IMODE(execution.canonical_path.stat().st_mode) == 0o500
        source.require_current()
        execution.require_current()
        with pytest.raises(ValueError, match="already exists"):
            _REAL_PYTHON_EXECUTION_MATERIALIZER(
                source,
                project_root=tmp_path,
                attempt_id="e" * 32,
            )
    finally:
        destination = execution.canonical_path
        attempt_directory = destination.parent
        execution.close_and_remove()
        source.close()
    assert execution.removed is True
    assert not destination.exists()
    assert not attempt_directory.exists()


def test_python_execution_copy_cleanup_never_unlinks_a_replacement(
    tmp_path: Path,
) -> None:
    (tmp_path / "outputs" / "readiness").mkdir(parents=True)
    source_path = tmp_path / "python-source"
    source_path.write_bytes(b"synthetic-python-executable\n")
    source_path.chmod(0o500)
    source = launch_modal._open_python_executable_binding(source_path)
    execution = _REAL_PYTHON_EXECUTION_MATERIALIZER(
        source,
        project_root=tmp_path,
        attempt_id="c" * 32,
    )
    destination = execution.canonical_path
    original = destination.with_name("original-python")
    try:
        destination.rename(original)
        destination.write_bytes(b"replacement-must-survive\n")
        destination.chmod(0o500)
        with pytest.raises(ValueError, match="changed"):
            execution.close_and_remove()
        assert destination.read_bytes() == b"replacement-must-survive\n"
        assert execution.removed is False
    finally:
        source.close()


def test_modal_binding_replacement_before_popen_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _FakeModalLaunchBindings(fail_on_validation=3)
    monkeypatch.setattr(
        launch_modal,
        "_open_modal_launch_bindings",
        lambda: bindings,
    )
    materialized = False

    def forbidden_materialization(*_args, **_kwargs):
        nonlocal materialized
        materialized = True
        raise AssertionError("pre-Popen rejection must precede Python copying")

    monkeypatch.setattr(
        launch_modal,
        "_materialize_python_execution_copy",
        forbidden_materialization,
    )
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("replaced binding must fail before Popen")

    with pytest.raises(ValueError, match="held launch binding replacement"):
        launch_modal.run(
            _arguments(),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
            attempt_id_factory=lambda _size: "d" * 32,
        )
    assert called is False
    assert materialized is False
    assert bindings.closed is True
    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert receipt["modal_cli_process_started"] is False
    assert receipt["remote_execution_state"] == "definitely_not_started"
    assert receipt["status"] == "preflight_rejected"


def test_python_execution_copy_replacement_after_popen_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _FakeBoundLaunchFile(
        Path("/private/modal-python-runtime/replaced-copy"),
        904,
        fail_on_validation=2,
    )
    monkeypatch.setattr(
        launch_modal,
        "_materialize_python_execution_copy",
        lambda *_args, **_kwargs: execution,
    )
    process = _FakeProcess(returncode=0)
    cleanup_calls = []

    with pytest.raises(
        launch_modal.ModalProcessStartReceiptError,
        match="local marker",
    ):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda child, **kwargs: cleanup_calls.append(
                (child, kwargs["process_group_id"])
            ),
        )

    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert cleanup_calls == [(process, process.pid)]
    assert receipt["status"] == "cli_failed"
    assert receipt["failure_kind"] == "process_start_receipt_persistence"
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is True
    assert isinstance(receipt["local_process_start_receipt_sha256"], str)
    assert len(receipt["local_process_start_receipt_sha256"]) == 64
    assert receipt["local_process_id"] == process.pid


@pytest.mark.parametrize(
    "scenario",
    ("success", "popen_error", "timeout", "interrupt"),
)
def test_python_execution_copy_is_removed_on_every_launch_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    execution = _FakeBoundLaunchFile(
        Path(f"/private/modal-python-runtime/{scenario}"),
        905,
    )
    monkeypatch.setattr(
        launch_modal,
        "_materialize_python_execution_copy",
        lambda *_args, **_kwargs: execution,
    )
    timeout = launch_modal.expected_outer_cli_timeout_seconds("cuda-environment")
    if scenario == "timeout":
        process = _FakeProcess(
            wait_error=subprocess.TimeoutExpired(["modal", "run"], timeout)
        )
    elif scenario == "interrupt":
        process = _FakeProcess(wait_error=KeyboardInterrupt())
    else:
        process = _FakeProcess(returncode=0)

    def runner(*_args, **_kwargs):
        if scenario == "popen_error":
            raise OSError("synthetic Popen failure")
        return process

    kwargs = {
        "runner": runner,
        "receipt_directory": tmp_path / "attempts",
        "process_group_capture": lambda child: child.pid,
        "process_group_terminator": lambda *_args, **_kwargs: None,
    }
    if scenario == "success":
        assert launch_modal.run(_arguments(), **kwargs) == 0
    elif scenario == "popen_error":
        with pytest.raises(OSError, match="synthetic Popen failure"):
            launch_modal.run(_arguments(), **kwargs)
    elif scenario == "timeout":
        with pytest.raises(launch_modal.ModalCLITimeoutError):
            launch_modal.run(_arguments(), **kwargs)
    else:
        with pytest.raises(KeyboardInterrupt):
            launch_modal.run(_arguments(), **kwargs)

    assert execution.removed is True
    assert execution.descriptor == -1


def test_python_execution_cleanup_failure_is_terminal_and_not_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _FakeBoundLaunchFile(
        Path("/private/modal-python-runtime/cleanup-failure"),
        906,
        fail_on_cleanup=True,
    )
    monkeypatch.setattr(
        launch_modal,
        "_materialize_python_execution_copy",
        lambda *_args, **_kwargs: execution,
    )
    process = _FakeProcess(returncode=0)

    with pytest.raises(ValueError, match="execution copy cleanup failure"):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: None,
        )

    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert execution.removed is False
    assert execution.descriptor == -1
    assert receipt["status"] == "cleanup_failed"
    assert receipt["failure_kind"] == "python_execution_cleanup"
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is True


def test_process_group_and_python_cleanup_failures_are_both_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution = _FakeBoundLaunchFile(
        Path("/private/modal-python-runtime/combined-cleanup-failure"),
        907,
        fail_on_cleanup=True,
    )
    monkeypatch.setattr(
        launch_modal,
        "_materialize_python_execution_copy",
        lambda *_args, **_kwargs: execution,
    )
    process = _FakeProcess(returncode=0)

    with pytest.raises(ValueError, match="execution copy cleanup failure"):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic process group cleanup failure")
            ),
        )

    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "cleanup_failed"
    assert receipt["failure_kind"] == (
        "process_group_and_python_execution_cleanup"
    )
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is False


def test_process_group_cleanup_failure_clears_the_untrusted_returncode(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(returncode=0)

    with pytest.raises(OSError, match="process group cleanup failure"):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic process group cleanup failure")
            ),
        )

    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "cleanup_failed"
    assert receipt["failure_kind"] == "process_group_cleanup"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is False


def test_verifier_actions_require_and_forward_fresh_verifier_id(monkeypatch) -> None:
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda arguments, **_kwargs: (
            (
                {
                    "gate": "candidate_resume_preflight_validated",
                    "path": (
                        _PREFLIGHT_RECEIPT_PATH
                    ),
                    "sha256": "f" * 64,
                },
            )
            if arguments.action in {"canary", "canaries"}
            else ()
        ),
    )
    arguments = _arguments(
        action="verify",
        run_id="candidate-source-1",
        verifier_run_id="candidate-verifier-1",
    )
    command, _ = launch_modal.build_launch(
        arguments,
        nonce_factory=lambda _size: "a" * 64,
    )
    verifier_index = command.index("--verifier-run-id")
    assert command[verifier_index + 1] == "candidate-verifier-1"

    with pytest.raises(ValueError, match="fresh --verifier-run-id"):
        launch_modal.build_launch(
            _arguments(
                action="verify",
                run_id="candidate-source-1",
                verifier_run_id="",
            )
        )
    with pytest.raises(ValueError, match="must differ"):
        launch_modal.build_launch(
            _arguments(
                action="download",
                run_id="candidate-source-1",
                verifier_run_id="candidate-source-1",
                local_output="outputs/development/modal_downloads",
            )
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"action": "plan"},
        {"approved": False},
        {"modal_cost_cap_usd": ""},
        {"modal_cost_cap_usd": "00.25"},
        {"modal_cost_cap_usd": "0"},
        {"provider_approved": True},
        {"provider_cost_cap_usd": "1.00"},
        {"action": "canaries", "provider_approved": False},
        {
            "action": "canary",
            "provider_approved": True,
            "harness": "not-frozen",
        },
        {"action": "checkpoint-resume", "source_run_id": ""},
        {"action": "download", "local_output": ""},
        {"action": "cuda-environment", "local_output": "outputs/out"},
    ),
)
def test_invalid_or_unapproved_launch_never_starts_modal(
    overrides,
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError):
        launch_modal.run(
            _arguments(**overrides),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False
    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_name"] == "ModalActionAttemptReceipt"
    assert receipt["schema_version"] == "3.6"
    assert receipt["status"] == "preflight_rejected"
    assert receipt["modal_command_sha256"] is None
    assert receipt["concrete_remote_run_ids"] == list(
        launch_modal._receipt_fields(_arguments(**overrides))[
            "concrete_remote_run_ids"
        ]
    )
    assert receipt["modal_cli_process_started"] is False
    assert receipt["remote_execution_state"] == "definitely_not_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is None
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is None
    assert not tuple((tmp_path / "attempts").glob("*.intent.json"))


@pytest.mark.parametrize(
    "invalid_value",
    ("-0", "-0.00", "1e-1", "NaN", "Infinity", "+0.25"),
)
def test_modal_cost_cap_rejects_noncanonical_money_before_popen(
    invalid_value: str,
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid cost cap must not start Modal")

    with pytest.raises(ValueError, match="modal_cost_cap_usd"):
        launch_modal.run(
            _arguments(modal_cost_cap_usd=invalid_value),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


@pytest.mark.parametrize(
    "failure_kind",
    (
        "missing_plan",
        "plan_symlink",
        "approval_digest",
        "stale_plan",
        "price_path",
        "price_symlink",
        "price_digest",
        "price_model",
        "provider_cap",
    ),
)
def test_provider_plan_price_and_cap_fail_closed_before_popen(
    failure_kind: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    expected_plan, price, overrides = _provider_approval_files(
        project,
        image_source_sha256=image_sha256,
    )
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "build_provider_canary_approval_plan",
        lambda *_args, **_kwargs: expected_plan,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: (
            {
                "gate": "candidate_resume_preflight_validated",
                "path": _PREFLIGHT_RECEIPT_PATH,
                "sha256": "f" * 64,
            },
        ),
    )

    if failure_kind == "missing_plan":
        overrides["provider_approval_plan_path"] = (
            "outputs/readiness/provider_canary_approval/missing.json"
        )
    elif failure_kind == "plan_symlink":
        original = project / str(overrides["provider_approval_plan_path"])
        alias = original.with_name("plan-link.json")
        alias.symlink_to(original)
        overrides["provider_approval_plan_path"] = alias.relative_to(
            project
        ).as_posix()
    elif failure_kind == "approval_digest":
        overrides["approval_plan_sha256"] = "0" * 64
    elif failure_kind == "stale_plan":
        stale_plan = dict(expected_plan)
        stale_plan["image_source_sha256"] = "0" * 64
        stale_plan.pop("approval_plan_sha256")
        stale_plan["approval_plan_sha256"] = launch_modal.canonical_sha256(
            stale_plan
        )
        _write_json(
            project / str(overrides["provider_approval_plan_path"]),
            stale_plan,
        )
        overrides["approval_plan_sha256"] = stale_plan["approval_plan_sha256"]
    elif failure_kind == "price_path":
        overrides["provider_price_basis_path"] = (
            "outputs/readiness/provider_price_basis.json"
        )
    elif failure_kind == "price_symlink":
        original = project / str(overrides["provider_price_basis_path"])
        target = original.with_name("provider_price_basis_target.json")
        original.rename(target)
        original.symlink_to(target)
    elif failure_kind == "price_digest":
        overrides["provider_price_basis_sha256"] = "0" * 64
    elif failure_kind == "price_model":
        invalid_price = dict(price)
        invalid_price["model"] = "not-the-approved-model"
        overrides["provider_price_basis_sha256"] = _write_json(
            project / str(overrides["provider_price_basis_path"]),
            invalid_price,
        )
    else:
        overrides["provider_cost_cap_usd"] = "0.0001"

    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid provider approval must not start Modal")

    with pytest.raises((FileNotFoundError, ValueError)):
        launch_modal.run(
            _arguments(**overrides),
            runner=forbidden_runner,
            project_root=project,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False
    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert receipt["modal_cli_process_started"] is False
    assert receipt["remote_execution_state"] == "definitely_not_started"
    assert not tuple((tmp_path / "attempts").glob("*.intent.json"))


@pytest.mark.parametrize(
    ("timestamp", "input_rate", "output_rate", "fee", "message"),
    (
        ("2026-08-07T23:59:59Z", "1", "1", "0", "older than 48 hours"),
        ("2026-08-10T00:05:01Z", "1", "1", "0", "too far in the future"),
        ("2026-08-10T00:00:00+00:00", "1", "1", "0", "Z-form"),
        ("2026-08-10T00:00:00Z", "0", "1", "0", "positive"),
        ("2026-08-10T00:00:00Z", "1", "0", "0", "positive"),
        ("2026-08-10T00:00:00Z", "1", "1", "-0", "canonical"),
    ),
)
def test_provider_price_basis_freshness_and_money_are_deterministic(
    timestamp: str,
    input_rate: str,
    output_rate: str,
    fee: str,
    message: str,
) -> None:
    payload = {
        "schema_name": "ProviderPriceBasis",
        "schema_version": "1.0",
        "model": launch_modal.TARGET_MODEL,
        "official_source_url": "https://openai.com/api/pricing/",
        "retrieved_at_utc": timestamp,
        "uncached_input_usd_per_million_tokens": input_rate,
        "output_usd_per_million_tokens": output_rate,
        "per_request_fee_usd": fee,
    }
    with pytest.raises(ValueError, match=message):
        launch_modal._validate_price_basis(
            payload,
            now_utc=launch_modal.datetime.fromisoformat(
                "2026-08-10T00:00:00+00:00"
            ),
        )


@pytest.mark.parametrize("preflight_state", ("missing", "stale"))
def test_provider_preflight_missing_or_stale_never_starts_modal(
    preflight_state: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    _plan, _price, overrides = _provider_approval_files(
        project,
        image_source_sha256=image_sha256,
    )
    preflight_path = project / str(
        overrides["candidate_resume_preflight_receipt_path"]
    )
    if preflight_state == "missing":
        preflight_path.unlink()
    else:
        stale = json.loads(preflight_path.read_text(encoding="utf-8"))
        stale["image_source_sha256"] = "0" * 64
        _write_json(preflight_path, stale)
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_provider_approval_inputs",
        lambda *_args, **_kwargs: {
            "provider_cost_cap_usd": overrides["provider_cost_cap_usd"],
            "provider_approval_plan_path": overrides[
                "provider_approval_plan_path"
            ],
            "approval_plan_sha256": overrides["approval_plan_sha256"],
            "provider_price_basis_path": overrides["provider_price_basis_path"],
            "provider_price_basis_sha256": overrides[
                "provider_price_basis_sha256"
            ],
        },
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_candidate_resume_preflight_receipt",
        lambda *_args, **_kwargs: None,
    )
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid preflight must not start Modal")

    with pytest.raises((FileNotFoundError, ValueError)):
        launch_modal.run(
            _arguments(**overrides),
            runner=forbidden_runner,
            project_root=project,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


def test_provider_intent_binds_exact_approval_chain_before_popen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    _plan, _price, overrides = _provider_approval_files(
        project,
        image_source_sha256=image_sha256,
    )
    provider_fields = {
        field: overrides[field]
        for field in (
            "provider_cost_cap_usd",
            "provider_approval_plan_path",
            "approval_plan_sha256",
            "provider_price_basis_path",
            "provider_price_basis_sha256",
        )
    }
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_provider_approval_inputs",
        lambda *_args, **_kwargs: provider_fields,
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_candidate_resume_preflight_receipt",
        lambda *_args, **_kwargs: None,
    )

    def runner(_command, **kwargs):
        intent = json.loads(
            _intent_path(tmp_path / "attempts").read_text(encoding="utf-8")
        )
        assert intent["modal_cost_cap_usd"] == "0.25"
        assert intent["modal_resource_profile"] == (
            launch_modal.modal_resource_profile("canaries")
        )
        for field, value in provider_fields.items():
            assert intent[field] == value
        assert intent["predecessor_receipts"] == [
            *_fake_local_freeze_bindings(),
                {
                    "gate": "candidate_resume_preflight_validated",
                    "path": overrides[
                        "candidate_resume_preflight_receipt_path"
                    ],
                    "sha256": hashlib.sha256(
                        (
                            project
                            / str(
                                overrides[
                                    "candidate_resume_preflight_receipt_path"
                                ]
                            )
                        ).read_bytes()
                ).hexdigest(),
            }
        ]
        assert kwargs["env"][launch_modal.MODAL_ACTION_ATTEMPT_ID_ENV] == intent[
            "attempt_id"
        ]
        assert kwargs["stdin"] == subprocess.DEVNULL
        return _FakeProcess()

    assert (
        launch_modal.run(
            _arguments(**overrides),
            runner=runner,
            project_root=project,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: None,
        )
        == 0
    )
    intent = json.loads(_intent_path(tmp_path / "attempts").read_text())
    terminal = json.loads(_terminal_receipt_path(tmp_path / "attempts").read_text())
    for field in (
        "modal_cost_cap_usd",
        "modal_resource_profile",
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
        "predecessor_receipts",
        "source_evidence_recovery",
    ):
        assert terminal[field] == intent[field]


@pytest.mark.parametrize(
    "predecessor_case",
    (
        "offline_missing_cuda",
        "offline_stale_cuda",
        "candidate_missing_offline",
        "resume_source_mismatch",
    ),
)
def test_paid_stage_predecessor_receipts_fail_closed_before_popen(
    predecessor_case: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_modal_readiness_receipt",
        lambda *_args, **_kwargs: None,
    )
    if hasattr(
        launch_modal.modal_readiness,
        "validate_offline_smoke_validation_receipt",
    ):
        monkeypatch.setattr(
            launch_modal.modal_readiness,
            "validate_offline_smoke_validation_receipt",
            lambda *_args, **_kwargs: None,
        )

    if predecessor_case == "offline_missing_cuda":
        arguments = _arguments(
            action="offline-smoke",
            run_id="offline-run-1",
            expected_image_source_sha256=image_sha256,
        )
    elif predecessor_case == "offline_stale_cuda":
        cuda_sha256 = _write_json(
            project / _CUDA_RECEIPT_PATH,
            {"image_source_sha256": "0" * 64},
        )
        arguments = _arguments(
            action="offline-smoke",
            run_id="offline-run-1",
            expected_image_source_sha256=image_sha256,
            cuda_receipt_sha256=cuda_sha256,
        )
    elif predecessor_case == "candidate_missing_offline":
        cuda_sha256 = _write_json(
            project / _CUDA_RECEIPT_PATH,
            {"image_source_sha256": image_sha256},
        )
        arguments = _arguments(
            action="candidate-smoke",
            run_id="candidate-run-1",
            expected_image_source_sha256=image_sha256,
            cuda_receipt_sha256=cuda_sha256,
        )
    else:
        round_trip_sha256 = _write_json(
            project / _ROUND_TRIP_RECEIPT_PATH,
            {
                "source_run_id": "different-candidate-source",
                "downloaded_run_path": "outputs/development/modal_downloads/source",
            },
        )
        arguments = _arguments(
            action="checkpoint-resume",
            source_run_id="candidate-source-1",
            run_id="resume-run-1",
            expected_image_source_sha256=image_sha256,
            artifact_round_trip_receipt_sha256=round_trip_sha256,
        )

    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("missing or stale predecessor must not start Modal")

    with pytest.raises((FileNotFoundError, ValueError)):
        launch_modal.run(
            arguments,
            runner=forbidden_runner,
            project_root=project,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False
    receipt = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert receipt["modal_cli_process_started"] is False
    assert receipt["predecessor_receipts"] == []


def test_stale_approved_source_never_starts_modal(tmp_path: Path) -> None:
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="current local plan"):
        launch_modal.run(
            _arguments(expected_image_source_sha256="0" * 64),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


def test_missing_source_attempt_attribution_never_starts_download(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unattributed Volume run must not start a verifier")

    with pytest.raises(FileNotFoundError):
        launch_modal.run(
            _arguments(
                action="download",
                run_id="source-run-1",
                local_output=launch_modal.MODAL_DOWNLOAD_OUTPUT_ROOT,
                expected_image_source_sha256=image_sha256,
            ),
            runner=forbidden_runner,
            project_root=project,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


def test_download_binds_exact_source_intent_and_terminal_pair(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    terminal_path, _intent, _terminal = _source_attempt_pair(project)
    arguments = _arguments(
        action="download",
        run_id="source-run-1",
        local_output="downloads",
        source_action_attempt_receipt_path=terminal_path,
        source_action_attempt_receipt_sha256=hashlib.sha256(
            (project / terminal_path).read_bytes()
        ).hexdigest(),
        expected_image_source_sha256="d" * 64,
    )
    bindings = launch_modal._validate_source_action_attribution(
        arguments,
        project_root=project,
        identity=launch_modal.ModalLiveCohortIdentity(
            source_tree_sha256=_SOURCE_TREE_SHA256,
            image_source_sha256="d" * 64,
            cohort_id="source-run-1",
        ),
    )
    assert [item["gate"] for item in bindings] == [
        "source_action_intent",
        "source_action_attempt_terminal",
        "source_local_process_start",
    ]
    assert all(len(item["sha256"]) == 64 for item in bindings)

    with pytest.raises(ValueError, match="outside|differs"):
        launch_modal._validate_source_action_attribution(
            _arguments(
                action="download",
                run_id="arbitrary-run-1",
                local_output="downloads",
                source_action_attempt_receipt_path=terminal_path,
                source_action_attempt_receipt_sha256=hashlib.sha256(
                    (project / terminal_path).read_bytes()
                ).hexdigest(),
                expected_image_source_sha256="d" * 64,
            ),
            project_root=project,
            identity=launch_modal.ModalLiveCohortIdentity(
                source_tree_sha256=_SOURCE_TREE_SHA256,
                image_source_sha256="d" * 64,
                cohort_id="source-run-1",
            ),
        )


def test_uncertain_source_download_requires_explicit_evidence_recovery(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    terminal_path, _intent, _terminal = _source_attempt_pair(
        project,
        status="timed_out",
        returncode=None,
        failure_kind="outer_cli_timeout",
        process_group_closed=True,
    )
    base = {
        "action": "download",
        "run_id": "source-run-1",
        "local_output": "downloads",
        "source_action_attempt_receipt_path": terminal_path,
        "source_action_attempt_receipt_sha256": hashlib.sha256(
            (project / terminal_path).read_bytes()
        ).hexdigest(),
        "expected_image_source_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="evidence recovery"):
        launch_modal._validate_source_action_attribution(
            _arguments(**base),
            project_root=project,
            identity=launch_modal.ModalLiveCohortIdentity(
                source_tree_sha256=_SOURCE_TREE_SHA256,
                image_source_sha256="d" * 64,
                cohort_id="source-run-1",
            ),
        )
    bindings = launch_modal._validate_source_action_attribution(
        _arguments(**base, source_evidence_recovery=True),
        project_root=project,
        identity=launch_modal.ModalLiveCohortIdentity(
            source_tree_sha256=_SOURCE_TREE_SHA256,
            image_source_sha256="d" * 64,
            cohort_id="source-run-1",
        ),
    )
    assert [item["gate"] for item in bindings] == [
        "source_action_intent",
        "source_action_attempt_terminal",
        "source_local_process_start",
    ]


def test_verifier_rejects_terminal_bytes_changed_after_raw_sha_approval(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    terminal_logical, _intent, _terminal = _source_attempt_pair(project)
    terminal_path = project / terminal_logical
    approved_sha256 = hashlib.sha256(terminal_path.read_bytes()).hexdigest()
    terminal_path.write_bytes(terminal_path.read_bytes() + b"\n")
    arguments = _arguments(
        action="verify",
        run_id="source-run-1",
        source_action_attempt_receipt_path=terminal_logical,
        source_action_attempt_receipt_sha256=approved_sha256,
        expected_image_source_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="approved raw SHA-256"):
        launch_modal._validate_source_action_attribution(
            arguments,
            project_root=project,
            identity=launch_modal.ModalLiveCohortIdentity(
                source_tree_sha256=_SOURCE_TREE_SHA256,
                image_source_sha256="d" * 64,
                cohort_id="source-run-1",
            ),
        )


def test_aggregate_exit_two_binds_all_four_exact_child_outcomes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    prefix = "provider-cohort-1"
    terminal_path, _intent, _terminal = _source_attempt_pair(
        project,
        action="canaries",
        run_id=prefix,
        status="failed",
        returncode=2,
        failure_kind="modal_cli_exit",
        process_group_closed=True,
    )
    attempt_id = "c" * 32
    outcomes = [
        {
            "harness": harness,
            "run_id": f"{prefix}-{launch_modal.canary_run_suffix(harness)}",
            "status": "failed" if index == 1 else "success",
            "error_type": "RuntimeError" if index == 1 else None,
        }
        for index, harness in enumerate(launch_modal.CANARY_ORDER)
    ]
    aggregate = {
        "schema_name": "ProviderCanaryAggregateOutcomeReceipt",
        "schema_version": "1.1",
        "attempt_id": attempt_id,
        "run_id_prefix": prefix,
        "source_tree_sha256": _SOURCE_TREE_SHA256,
        "image_source_sha256": "d" * 64,
        "cohort_id": "source-cohort-1",
        "harness_order": list(launch_modal.CANARY_ORDER),
        "outcomes": outcomes,
        "all_succeeded": False,
    }
    _write_json(
        project.joinpath(
            *launch_modal.provider_canary_aggregate_outcome_receipt_path(
                launch_modal.ModalLiveCohortIdentity(
                    source_tree_sha256=_SOURCE_TREE_SHA256,
                    image_source_sha256="d" * 64,
                    cohort_id="source-cohort-1",
                ),
                attempt_id,
            ).parts
        ),
        aggregate,
    )
    selected = outcomes[0]["run_id"]
    bindings = launch_modal._validate_source_action_attribution(
        _arguments(
            action="download",
            run_id=selected,
            local_output="downloads",
            source_action_attempt_receipt_path=terminal_path,
            source_action_attempt_receipt_sha256=hashlib.sha256(
                (project / terminal_path).read_bytes()
            ).hexdigest(),
            expected_image_source_sha256="d" * 64,
        ),
        project_root=project,
        identity=launch_modal.ModalLiveCohortIdentity(
            source_tree_sha256=_SOURCE_TREE_SHA256,
            image_source_sha256="d" * 64,
            cohort_id="source-cohort-1",
        ),
    )
    assert [item["gate"] for item in bindings] == [
        "source_action_intent",
        "source_action_attempt_terminal",
        "source_local_process_start",
        "provider_canary_aggregate_outcomes",
    ]

    (project / bindings[-1]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        launch_modal._validate_source_action_attribution(
            _arguments(
                action="download",
                run_id=selected,
                local_output="downloads",
                source_action_attempt_receipt_path=terminal_path,
                source_action_attempt_receipt_sha256=hashlib.sha256(
                    (project / terminal_path).read_bytes()
                ).hexdigest(),
                expected_image_source_sha256="d" * 64,
            ),
            project_root=project,
            identity=launch_modal.ModalLiveCohortIdentity(
                source_tree_sha256=_SOURCE_TREE_SHA256,
                image_source_sha256="d" * 64,
                cohort_id="source-cohort-1",
            ),
        )


def test_patch_bundle_failure_never_starts_modal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    called = False

    def reject_patch_bundle(_root):
        raise ValueError("OpenEvolve patch bundle is not applied")

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        reject_patch_bundle,
    )
    with pytest.raises(ValueError, match="patch bundle"):
        launch_modal.run(
            _arguments(),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


def test_missing_local_freeze_receipts_never_start_modal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    called = False

    def missing_receipts(_root, **_kwargs):
        raise ValueError(
            "paid Modal launch requires a fresh current-source local freeze"
        )

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        launch_modal,
        "validate_local_freeze_evidence",
        missing_receipts,
    )
    with pytest.raises(ValueError, match="fresh current-source"):
        launch_modal.run(
            _arguments(),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )

    assert called is False
    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "preflight_rejected"
    assert receipt["modal_cli_process_started"] is False


def test_local_freeze_components_and_aggregate_bind_intent_and_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str | None] = []

    def validate_freeze(_root, *, expected_image_source_sha256=None):
        calls.append(expected_image_source_sha256)
        return _fake_local_freeze_bindings()

    monkeypatch.setattr(
        launch_modal,
        "validate_local_freeze_evidence",
        validate_freeze,
    )

    assert (
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: _FakeProcess(),
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: None,
        )
        == 0
    )
    intent = json.loads(_intent_path(tmp_path / "attempts").read_text())
    terminal = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text()
    )
    expected = list(_fake_local_freeze_bindings())
    assert intent["predecessor_receipts"] == expected
    assert terminal["predecessor_receipts"] == expected
    assert calls == [
        _arguments().expected_image_source_sha256,
        _arguments().expected_image_source_sha256,
    ]


def test_local_freeze_change_after_intent_fails_before_popen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = 0
    first = _fake_local_freeze_bindings()
    changed = tuple(dict(item) for item in first)
    changed[0]["sha256"] = "4" * 64

    def validate_freeze(_root, **_kwargs):
        nonlocal calls
        calls += 1
        return first if calls == 1 else changed

    monkeypatch.setattr(
        launch_modal,
        "validate_local_freeze_evidence",
        validate_freeze,
    )
    started = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal started
        started = True
        raise AssertionError("changed freeze must not start Modal")

    with pytest.raises(ValueError, match="approval chain changed"):
        launch_modal.run(
            _arguments(),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )

    assert calls == 2
    assert started is False
    assert _intent_path(tmp_path / "attempts").is_file()
    terminal = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text()
    )
    assert terminal["modal_cli_process_started"] is False


def test_approval_chain_revalidation_rejects_numeric_type_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def changing_modal_approval(arguments, **_kwargs):
        nonlocal calls
        calls += 1
        profile = launch_modal.modal_resource_profile(
            arguments.action,
            arguments.harness,
        )
        if calls == 2:
            profile["runtime_function_calls"][0]["call_count"] = True
        return {
            "modal_cost_cap_usd": arguments.modal_cost_cap_usd,
            "modal_resource_profile": profile,
            "modal_price_basis_path": arguments.modal_price_basis_path,
            "modal_price_basis_sha256": arguments.modal_price_basis_sha256,
            "modal_cost_estimate": _fake_modal_cost_estimate(arguments.action),
        }

    monkeypatch.setattr(
        launch_modal,
        "_validate_modal_approval_inputs",
        changing_modal_approval,
    )
    started = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal started
        started = True
        raise AssertionError("type-substituted approval must not start Modal")

    with pytest.raises(ValueError, match="approval chain changed"):
        launch_modal.run(
            _arguments(),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )

    assert calls == 2
    assert started is False


def test_modal_price_binding_is_revalidated_after_intent_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def changing_modal_approval(arguments, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Modal price-basis raw SHA-256 changed")
        return {
            "modal_cost_cap_usd": arguments.modal_cost_cap_usd,
            "modal_resource_profile": launch_modal.modal_resource_profile(
                arguments.action,
                arguments.harness,
            ),
            "modal_price_basis_path": arguments.modal_price_basis_path,
            "modal_price_basis_sha256": arguments.modal_price_basis_sha256,
            "modal_cost_estimate": _fake_modal_cost_estimate(arguments.action),
        }

    monkeypatch.setattr(
        launch_modal,
        "_validate_modal_approval_inputs",
        changing_modal_approval,
    )
    started = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal started
        started = True
        raise AssertionError("changed Modal price basis must not start Modal")

    with pytest.raises(ValueError, match="price-basis raw SHA-256 changed"):
        launch_modal.run(
            _arguments(),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )

    assert calls == 2
    assert started is False
    assert _intent_path(tmp_path / "attempts").is_file()
    terminal = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text()
    )
    assert terminal["modal_cli_process_started"] is False


def test_overlong_aggregate_canary_ids_never_start_modal(tmp_path: Path) -> None:
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="run_id"):
        launch_modal.run(
            _arguments(
                action="canaries",
                run_id="a" * 63,
                provider_approved=True,
            ),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


def test_single_canary_mismatched_recovery_suffix_never_starts_modal(
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="frozen harness suffix"):
        launch_modal.run(
            _arguments(
                action="canary",
                run_id="modal-recovery-greedy-ar",
                harness="openevolve_generic",
                provider_approved=True,
            ),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


@pytest.mark.parametrize(
    "unsafe_kind",
    ("output_symlink", "parent_symlink", "target", "target_symlink"),
)
def test_unsafe_or_nonfresh_download_destination_never_starts_modal(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    local_output = "downloads"
    if unsafe_kind == "output_symlink":
        (project / "downloads").symlink_to(outside, target_is_directory=True)
    elif unsafe_kind == "parent_symlink":
        (project / "outputs").symlink_to(outside, target_is_directory=True)
        local_output = "outputs/downloads"
    elif unsafe_kind == "target":
        destination = project / "downloads" / "modal-download-1"
        destination.mkdir(parents=True)
    else:
        downloads = project / "downloads"
        downloads.mkdir()
        (downloads / "modal-download-1").symlink_to(
            outside / "missing-target",
            target_is_directory=True,
        )
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="download|symbolic link|symlink"):
        launch_modal.run(
            _arguments(
                action="download",
                run_id="modal-download-1",
                local_output=local_output,
            ),
            runner=forbidden_runner,
            project_root=project,
            receipt_directory=tmp_path / "attempts",
        )
    assert called is False


def test_launcher_calls_modal_only_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []
    cleanup_calls = []
    process = _FakeProcess(returncode=17)
    local_secret = "provider-secret-must-not-reach-attempt-receipt"
    monkeypatch.setenv("DISCOVERY_API_KEY", local_secret)

    def runner(command, **kwargs):
        intent = json.loads(
            _intent_path(tmp_path / "attempts").read_text(encoding="utf-8")
        )
        assert intent["schema_name"] == "ModalActionIntent"
        assert intent["schema_version"] == "1.6"
        assert intent["attempt_id"]
        assert intent["action"] == "cuda-environment"
        assert intent["run_id"] == "modal-cuda-env-20260809-01"
        assert intent["concrete_remote_run_ids"] == [
            "modal-cuda-env-20260809-01"
        ]
        assert intent["modal_cost_cap_usd"] == "0.25"
        assert intent["provider_cost_cap_usd"] is None
        assert intent["provider_approval_plan_path"] is None
        assert intent["approval_plan_sha256"] is None
        assert intent["provider_price_basis_path"] is None
        assert intent["provider_price_basis_sha256"] is None
        assert intent["predecessor_receipts"] == list(
            _fake_local_freeze_bindings()
        )
        assert intent["source_evidence_recovery"] is False
        calls.append((command, kwargs))
        return process

    def terminate(child, *, process_group_id):
        cleanup_calls.append((child, process_group_id))

    assert (
        launch_modal.run(
            _arguments(),
            runner=runner,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=terminate,
        )
        == 17
    )
    assert len(calls) == 1
    assert calls[0][0][:2] == [
        "/private/modal-python-runtime/copy",
        "/dev/fd/901",
    ]
    assert calls[0][0][2:6] == [
        "run",
        "--env",
        "main",
        str(ROOT / "modal_app.py"),
    ]
    assert calls[0][1]["cwd"] == ROOT
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["executable"] == "/private/modal-python-runtime/copy"
    assert calls[0][1]["pass_fds"] == (901, 902)
    assert calls[0][1]["env"]["MODAL_CONFIG_PATH"] == "/dev/fd/902"
    assert set(calls[0][1]["env"]) == {
        *launch_modal._PAID_MODAL_BASE_ENVIRONMENT,
        launch_modal.MODAL_LAUNCH_NONCE_ENV,
        launch_modal.MODAL_LAUNCH_SOURCE_ENV,
        launch_modal.MODAL_LAUNCH_SOURCE_TREE_ENV,
        launch_modal.MODAL_LAUNCH_COHORT_ENV,
        launch_modal.MODAL_ACTION_ATTEMPT_ID_ENV,
        launch_modal.MODAL_ACTION_INTENT_SHA256_ENV,
        launch_modal.MODAL_PROFILE_ENV,
        launch_modal.MODAL_ENVIRONMENT_ENV,
        "MODAL_CONFIG_PATH",
        "PYTHONPATH",
    }
    assert "HOME" not in calls[0][1]["env"]
    assert local_secret not in calls[0][1]["env"].values()
    assert process.wait_timeout == (
        launch_modal.expected_outer_cli_timeout_seconds("cuda-environment")
    )
    assert cleanup_calls == [(process, process.pid)]

    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    serialized = receipt_path.read_text(encoding="utf-8")
    receipt = json.loads(serialized)
    intent = json.loads(
        _intent_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    canonical_command = [
        str(Path(launch_modal.sys.executable).with_name("modal")),
        *calls[0][0][2:],
    ]
    assert intent["modal_command_sha256"] == launch_modal._modal_command_sha256(
        canonical_command
    )
    assert intent["attempt_id"] == receipt["attempt_id"]
    assert intent["modal_command_sha256"] == receipt["modal_command_sha256"]
    assert intent["approved_image_source_sha256"] == (
        receipt["approved_image_source_sha256"]
    )
    assert receipt["schema_name"] == "ModalActionAttemptReceipt"
    assert receipt["schema_version"] == "3.6"
    assert receipt["status"] == "failed"
    assert receipt["failure_kind"] == "modal_cli_exit"
    assert receipt["concrete_remote_run_ids"] == [
        "modal-cuda-env-20260809-01"
    ]
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] == 17
    assert receipt["process_group_closed"] is True
    assert receipt["modal_profile"] == "scalingintelligence"
    assert receipt["modal_cost_cap_usd"] == "0.25"
    assert receipt["modal_resource_profile"] == intent[
        "modal_resource_profile"
    ]
    assert receipt["predecessor_receipts"] == list(
        _fake_local_freeze_bindings()
    )
    assert receipt["source_evidence_recovery"] is False
    assert receipt["modal_command_sha256"]
    assert local_secret not in serialized


def test_outer_cli_deadline_is_frozen_per_action() -> None:
    assert launch_modal.expected_outer_cli_timeout_seconds("cuda-environment") == 1200
    assert launch_modal.expected_outer_cli_timeout_seconds("canaries") == 2100
    with pytest.raises(ValueError, match="exactly 1200 seconds"):
        launch_modal.build_launch(
            _arguments(outer_cli_timeout_seconds=1199),
        )


def test_reviewed_attempt_id_is_used_without_calling_random_factory(
    tmp_path: Path,
) -> None:
    reviewed_attempt_id = "a" * 32

    assert (
        launch_modal.run(
            _arguments(attempt_id=reviewed_attempt_id),
            runner=lambda *_args, **_kwargs: _FakeProcess(),
            receipt_directory=tmp_path / "attempts",
            attempt_id_factory=lambda _size: pytest.fail(
                "reviewed attempt ID unexpectedly used the random factory"
            ),
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: None,
        )
        == 0
    )

    intent = json.loads(
        (
            tmp_path
            / "attempts"
            / f"{reviewed_attempt_id}.intent.json"
        ).read_text(encoding="utf-8")
    )
    terminal = json.loads(
        (tmp_path / "attempts" / f"{reviewed_attempt_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert intent["attempt_id"] == reviewed_attempt_id
    assert terminal["attempt_id"] == reviewed_attempt_id


@pytest.mark.parametrize(
    "invalid_attempt_id",
    ("A" * 32, "a" * 31, "g" * 32, True),
)
def test_reviewed_attempt_id_must_be_exact_lowercase_hex_before_any_write(
    tmp_path: Path,
    invalid_attempt_id: object,
) -> None:
    called = False

    def forbidden_runner(*_args: object, **_kwargs: object) -> _FakeProcess:
        nonlocal called
        called = True
        raise AssertionError("invalid reviewed attempt ID reached Modal Popen")

    with pytest.raises(ValueError, match="attempt ID"):
        launch_modal.run(
            _arguments(attempt_id=invalid_attempt_id),
            runner=forbidden_runner,
            receipt_directory=tmp_path / "attempts",
        )

    assert called is False
    assert not (tmp_path / "attempts").exists()


def test_modal_resource_profile_reports_runtime_limits_and_build_limitation() -> None:
    aggregate = launch_modal.modal_resource_profile("canaries")
    assert aggregate["runtime_cpu_request_equals_soft_limit"] is True
    assert aggregate["runtime_memory_request_equals_hard_limit"] is True
    assert aggregate["runtime_platform_compute_cost_ceiling_enforced"] is False
    assert aggregate["runtime_functions_preemptible"] is True
    assert aggregate["platform_preemption_restart_possible"] is True
    assert aggregate["logical_call_count_is_not_container_attempt_ceiling"] is True
    assert [item["function_name"] for item in aggregate["runtime_function_calls"]] == [
        f"canary_{harness}" for harness in launch_modal.CANARY_ORDER
    ]
    for runtime in aggregate["runtime_function_calls"]:
        assert (
            runtime["cpu_request_cores"]
            == runtime["cpu_soft_limit_cores"]
            == 2.0
        )
        assert runtime["memory_request_mib"] == runtime["memory_limit_mib"] == 8192
        assert runtime["region"] is None
        assert runtime["call_count"] == 1
        assert runtime["retries"] == 0
        assert runtime["provider_secret_attached"] is True
        assert runtime["network_mode"] == "provider_egress_enabled"
    image_build = aggregate["image_build"]
    assert image_build == {
        "invocation_condition": "backend_cache_miss",
        "cpu_request_cores": 2.0,
        "cpu_soft_limit_cores": None,
        "memory_request_mib": 8192,
        "memory_limit_mib": None,
        "gpu": None,
        "region": None,
        "timeout_seconds": 600,
        "subprocess_thread_limit": 2,
        "resource_limits_exposed": False,
        "platform_compute_cost_ceiling_enforced": False,
        "provider_secret_attached": False,
        "network_mode": "dependency_install_egress_required",
    }

    provider_free = launch_modal.modal_resource_profile("candidate-smoke")
    assert provider_free["runtime_function_calls"][0][
        "provider_secret_attached"
    ] is False
    assert provider_free["runtime_function_calls"][0]["network_mode"] == "blocked"


def test_local_lock_contention_is_zero_write_and_zero_remote_call_attempt(
    tmp_path: Path,
) -> None:
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Modal CLI must not start during lock contention")

    holder = launch_modal._acquire_launcher_lock(ROOT)
    try:
        with pytest.raises(
            launch_modal.ModalLaunchLockContentionError,
            match="holds",
        ):
            launch_modal.run(
                _arguments(),
                runner=forbidden_runner,
                receipt_directory=tmp_path / "attempts",
            )
    finally:
        launch_modal._release_launcher_lock(holder)

    assert called is False
    assert not (tmp_path / "attempts").exists()


def test_global_journal_gate_runs_before_containment_approval_or_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    attempt_id = "a" * 32
    events: list[str] = []

    def fake_probe(
        locked_root: Path,
        *,
        process_start_receipt_path: str,
        process_start_receipt_sha256: str,
        **providers,
    ) -> str:
        assert locked_root == project
        assert process_start_receipt_path.endswith(f"/{'f' * 32}.json")
        assert process_start_receipt_sha256 == "e" * 64
        assert all(provider is not None for provider in providers.values())
        events.append("probe")
        return "different_boot_session"

    sentinel = object()

    def fake_scan(*, lock_descriptor: int, process_probe):
        launch_modal._assert_launcher_lock_identity(lock_descriptor)
        events.append("scan")
        assert (
            process_probe(
                project,
                f"outputs/private/modal_local_process_starts/{'f' * 32}.json",
                "e" * 64,
            )
            == "different_boot_session"
        )
        return sentinel

    def blocking_gate(scan, *, candidate_attempt_id: str) -> None:
        assert scan is sentinel
        assert candidate_attempt_id == attempt_id
        events.append("gate")
        raise modal_action_journal.ModalActionJournalBlockedError(
            "synthetic unresolved journal"
        )

    monkeypatch.setattr(
        launch_modal,
        "probe_same_boot_modal_process_group",
        fake_probe,
    )
    monkeypatch.setattr(
        launch_modal,
        "_scan_modal_global_action_journal",
        fake_scan,
    )
    monkeypatch.setattr(
        launch_modal,
        "_require_modal_global_action_gate_clear",
        blocking_gate,
    )
    monkeypatch.setattr(
        launch_modal,
        "_open_or_create_local_containment_binding",
        lambda **_kwargs: pytest.fail("containment must follow the global gate"),
    )
    monkeypatch.setattr(
        launch_modal,
        "_build_validated_launch",
        lambda *_args, **_kwargs: pytest.fail("approval must follow the gate"),
    )

    with pytest.raises(
        modal_action_journal.ModalActionJournalBlockedError,
        match="unresolved journal",
    ):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            project_root=project,
            attempt_id_factory=lambda _size: attempt_id,
        )

    assert events == ["scan", "probe", "gate"]
    assert not project.joinpath(*MODAL_LAUNCH_REJECTION_ROOT.parts).exists()
    assert not project.joinpath(*MODAL_LIVE_COHORT_ROOT.parts).exists()
    assert not project.joinpath(*MODAL_REMOTE_RUN_RESERVATION_ROOT.parts).exists()
    assert not project.joinpath(
        *launch_modal.MODAL_LOCAL_CONTAINMENT_ROOT.parts
    ).exists()
    descriptor = launch_modal._acquire_launcher_lock(project)
    launch_modal._release_launcher_lock(descriptor)


def test_real_global_journal_blocker_is_zero_write_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="blocked-prior-cohort",
    )
    prior_attempt_id = "b" * 32
    [reservation] = modal_action_journal.build_modal_remote_run_reservation_specs(
        concrete_remote_run_ids=("blocked-prior-run",),
        attempt_id=prior_attempt_id,
        action="cuda-environment",
        identity=identity,
        created_at_utc="2025-01-01T00:00:00Z",
        launch_capability_sha256="3" * 64,
        local_host_anchor_path=(
            launch_modal.modal_local_host_anchor_path().as_posix()
        ),
        local_host_anchor_sha256="4" * 64,
        local_boot_started_at_unix_microseconds=(
            _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS
        ),
        local_boot_session_sha256="5" * 64,
    )
    reservation_path = project.joinpath(*Path(reservation.binding["path"]).parts)
    _write_json(reservation_path, dict(reservation.payload))

    monkeypatch.setattr(
        launch_modal,
        "_scan_modal_global_action_journal",
        _REAL_GLOBAL_JOURNAL_SCANNER,
    )
    monkeypatch.setattr(
        launch_modal,
        "_require_modal_global_action_gate_clear",
        _REAL_GLOBAL_JOURNAL_GATE,
    )

    with pytest.raises(
        modal_action_journal.ModalActionJournalBlockedError,
        match="unresolved",
    ):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            project_root=project,
            attempt_id_factory=lambda _size: "c" * 32,
        )

    assert reservation_path.is_file()
    assert not project.joinpath(*MODAL_LAUNCH_REJECTION_ROOT.parts).exists()
    assert not project.joinpath(*MODAL_LIVE_COHORT_ROOT.parts).exists()
    assert not project.joinpath(
        *launch_modal.MODAL_LOCAL_CONTAINMENT_ROOT.parts
    ).exists()
    descriptor = launch_modal._acquire_launcher_lock(project)
    launch_modal._release_launcher_lock(descriptor)


@pytest.mark.parametrize(
    ("action", "path_field", "sha_field", "gate"),
    (
        (
            "offline-smoke",
            "cuda_receipt_path",
            "cuda_receipt_sha256",
            "modal_cuda_environment_validated",
        ),
        (
            "candidate-smoke",
            "offline_smoke_receipt_path",
            "offline_smoke_receipt_sha256",
            "modal_offline_smoke_validated",
        ),
        (
            "checkpoint-resume",
            "artifact_round_trip_receipt_path",
            "artifact_round_trip_receipt_sha256",
            "modal_artifact_round_trip_validated",
        ),
    ),
)
def test_paid_component_predecessor_paths_cannot_cross_cohorts(
    action: str,
    path_field: str,
    sha_field: str,
    gate: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    current = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id="current-cohort",
    )
    other = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id="other-cohort",
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_modal_readiness_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_offline_smoke_validation_receipt",
        lambda *_args, **_kwargs: None,
    )
    overrides: dict[str, object] = {
        "action": action,
        "run_id": f"{action}-destination",
        "cohort_id": current.cohort_id,
        "expected_image_source_sha256": image_sha256,
        path_field: launch_modal.modal_readiness.modal_component_receipt_path(
            other,
            gate,
        ).as_posix(),
        sha_field: "f" * 64,
    }
    if action == "checkpoint-resume":
        overrides["source_run_id"] = "candidate-source-1"
    if action == "candidate-smoke":
        cuda_logical = launch_modal.modal_readiness.modal_component_receipt_path(
            current,
            "modal_cuda_environment_validated",
        ).as_posix()
        overrides["cuda_receipt_path"] = cuda_logical
        overrides["cuda_receipt_sha256"] = _write_json(
            project / cuda_logical,
            {
                "source_tree_sha256": current.source_tree_sha256,
                "image_source_sha256": current.image_source_sha256,
                "cohort_id": current.cohort_id,
            },
        )
    with pytest.raises(ValueError, match="outside the current live cohort"):
        launch_modal._validate_predecessor_receipts(
            _arguments(**overrides),
            project_root=project,
            identity=current,
        )


def test_paid_predecessor_payload_identity_cannot_cross_cohorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    current = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256="d" * 64,
        cohort_id="current-cohort",
    )
    cuda_logical = launch_modal.modal_readiness.modal_component_receipt_path(
        current,
        "modal_cuda_environment_validated",
    ).as_posix()
    cuda_sha256 = _write_json(
        project / cuda_logical,
        {
            "source_tree_sha256": current.source_tree_sha256,
            "image_source_sha256": current.image_source_sha256,
            "cohort_id": "different-cohort",
        },
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_modal_readiness_receipt",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="different live cohort"):
        launch_modal._validate_predecessor_receipts(
            _arguments(
                action="offline-smoke",
                run_id="offline-destination",
                cohort_id=current.cohort_id,
                expected_image_source_sha256=current.image_source_sha256,
                cuda_receipt_path=cuda_logical,
                cuda_receipt_sha256=cuda_sha256,
            ),
            project_root=project,
            identity=current,
        )


def test_candidate_preflight_path_and_payload_cannot_cross_cohorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    current = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256="d" * 64,
        cohort_id="current-cohort",
    )
    other = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256="d" * 64,
        cohort_id="other-cohort",
    )
    binding_sha256 = "7" * 64
    preflight_logical = (
        launch_modal.modal_readiness.modal_candidate_resume_preflight_receipt_path(
            other,
            binding_sha256,
        ).as_posix()
    )
    preflight_sha256 = _write_json(
        project / preflight_logical,
        {
            "source_tree_sha256": other.source_tree_sha256,
            "image_source_sha256": other.image_source_sha256,
            "cohort_id": other.cohort_id,
            "binding_sha256": binding_sha256,
        },
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_candidate_resume_preflight_receipt",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="different live cohort"):
        launch_modal._validate_predecessor_receipts(
            _arguments(
                action="canary",
                run_id="canary-greedy-ar",
                harness="greedy_autoresearch",
                cohort_id=current.cohort_id,
                expected_image_source_sha256=current.image_source_sha256,
                candidate_resume_preflight_receipt_path=preflight_logical,
                candidate_resume_preflight_receipt_sha256=preflight_sha256,
            ),
            project_root=project,
            identity=current,
        )


def test_private_download_can_authorize_checkpoint_resume_without_chmod_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    source_run_id = "downloaded-candidate-source"
    source_manifest = {
        "schema_name": "ModalImageSourceManifest",
        "schema_version": "1.0",
        "files": [],
    }
    source_manifest_bytes = (
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    image_sha256 = launch_modal.canonical_sha256(source_manifest)
    artifact_manifest = ArtifactManifestV1(
        run_id=source_run_id,
        created_at_utc="2026-08-10T00:00:00Z",
        image_source_sha256=image_sha256,
        files=(
            ArtifactFileV1(
                relative_path="image_source_manifest.json",
                sha256=hashlib.sha256(source_manifest_bytes).hexdigest(),
                size_bytes=len(source_manifest_bytes),
            ),
        ),
    )
    raw_manifest = RawArtifactManifestV1.from_bytes(
        filename="artifact_manifest.json",
        raw_bytes=(
            json.dumps(artifact_manifest.to_dict(), indent=2, sort_keys=True)
            + "\n"
        ).encode(),
    )
    downloaded = download_artifacts(
        raw_manifest,
        local_root=project / "outputs/development/modal_downloads",
        reader=lambda _path: source_manifest_bytes,
    )
    assert stat.S_IMODE(
        (downloaded / "image_source_manifest.json").stat().st_mode
    ) == 0o600

    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id="checkpoint-cohort",
    )
    round_trip_logical = launch_modal.modal_readiness.modal_component_receipt_path(
        identity,
        "modal_artifact_round_trip_validated",
    ).as_posix()
    round_trip_sha256 = _write_json(
        project / round_trip_logical,
        {
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "source_run_id": source_run_id,
            "downloaded_run_path": downloaded.relative_to(project).as_posix(),
        },
    )
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "validate_modal_readiness_receipt",
        lambda *_args, **_kwargs: None,
    )
    assert launch_modal.run(
        _arguments(
            action="checkpoint-resume",
            run_id="checkpoint-resume-destination",
            source_run_id=source_run_id,
            cohort_id=identity.cohort_id,
            expected_image_source_sha256=image_sha256,
            artifact_round_trip_receipt_path=round_trip_logical,
            artifact_round_trip_receipt_sha256=round_trip_sha256,
        ),
        runner=lambda *_args, **_kwargs: _FakeProcess(),
        project_root=project,
        receipt_directory=tmp_path / "attempts",
        attempt_id_factory=lambda _size: "7" * 32,
        process_group_capture=lambda child: child.pid,
        process_group_terminator=lambda *_args, **_kwargs: None,
    ) == 0


def test_launcher_timeout_closes_group_and_records_attempt(tmp_path: Path) -> None:
    timeout = launch_modal.expected_outer_cli_timeout_seconds("cuda-environment")
    process = _FakeProcess(
        wait_error=subprocess.TimeoutExpired(["modal", "run"], timeout),
    )
    cleanup_calls = []

    with pytest.raises(launch_modal.ModalCLITimeoutError, match="1200 seconds"):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda child, **kwargs: cleanup_calls.append(
                (child, kwargs["process_group_id"])
            ),
        )

    assert cleanup_calls == [(process, process.pid)]
    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "timed_out"
    assert receipt["failure_kind"] == "outer_cli_timeout"
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is True


def test_launcher_interrupt_closes_group_and_records_attempt(tmp_path: Path) -> None:
    process = _FakeProcess(wait_error=KeyboardInterrupt())
    cleanup_calls = []

    with pytest.raises(KeyboardInterrupt):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda child, **kwargs: cleanup_calls.append(
                (child, kwargs["process_group_id"])
            ),
        )

    assert cleanup_calls == [(process, process.pid)]
    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "interrupted"
    assert receipt["failure_kind"] == "interrupt"
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is True


def test_process_launch_failure_records_definite_zero_remote_process(
    tmp_path: Path,
) -> None:
    def failed_runner(*_args, **_kwargs):
        raise OSError("synthetic Popen failure")

    with pytest.raises(OSError, match="synthetic Popen failure"):
        launch_modal.run(
            _arguments(),
            runner=failed_runner,
            receipt_directory=tmp_path / "attempts",
        )

    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "cli_failed"
    assert receipt["failure_kind"] == "process_launch"
    assert receipt["modal_cli_process_started"] is False
    assert receipt["remote_execution_state"] == "definitely_not_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is None
    assert _intent_path(tmp_path / "attempts").is_file()


def test_terminal_receipt_failure_preserves_pre_popen_intent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    process = _FakeProcess()
    cleanup_calls = []

    def failed_terminal_write(*_args, **_kwargs):
        raise OSError("synthetic terminal receipt failure")

    monkeypatch.setattr(
        launch_modal,
        "_write_attempt_receipt",
        failed_terminal_write,
    )
    with pytest.raises(
        launch_modal.ModalAttemptReceiptError,
        match="attempt receipt could not be created",
    ):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda child, **kwargs: cleanup_calls.append(
                (child, kwargs["process_group_id"])
            ),
        )

    assert cleanup_calls == [(process, process.pid)]
    intent = json.loads(
        _intent_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert intent["schema_name"] == "ModalActionIntent"
    assert intent["attempt_id"]
    assert tuple((tmp_path / "attempts").glob("*.intent.json"))
    assert not tuple(
        path
        for path in (tmp_path / "attempts").glob("*.json")
        if not path.name.endswith(".intent.json")
    )


def test_wait_failure_records_uncertain_remote_execution_and_closed_group(
    tmp_path: Path,
) -> None:
    process = _FakeProcess(wait_error=RuntimeError("synthetic wait failure"))

    with pytest.raises(RuntimeError, match="synthetic wait failure"):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: None,
        )

    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "cli_failed"
    assert receipt["failure_kind"] == "modal_cli"
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is True


def test_process_group_capture_failure_still_closes_expected_child_group(
    tmp_path: Path,
) -> None:
    process = _FakeProcess()
    cleanup_calls = []

    def failed_capture(_process):
        raise OSError("synthetic PGID capture failure")

    with pytest.raises(OSError, match="synthetic PGID capture failure"):
        launch_modal.run(
            _arguments(),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=failed_capture,
            process_group_terminator=lambda child, **kwargs: cleanup_calls.append(
                (child, kwargs["process_group_id"])
            ),
        )

    assert cleanup_calls == [(process, process.pid)]
    receipt_path = _terminal_receipt_path(tmp_path / "attempts")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "cli_failed"
    assert receipt["failure_kind"] == "process_launch"
    assert receipt["modal_cli_process_started"] is True
    assert receipt["remote_execution_state"] == "may_have_started"
    assert receipt["returncode"] is None
    assert receipt["process_group_closed"] is True


def test_attempt_receipt_concrete_remote_ids_are_action_exact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider_chain = {
        "provider_cost_cap_usd": "2.00",
        "provider_approval_plan_path": (
            "outputs/readiness/provider_canary_approval/plan.json"
        ),
        "approval_plan_sha256": "a" * 64,
        "provider_price_basis_path": (
            "outputs/readiness/modal_resource_cleanup/price.json"
        ),
        "provider_price_basis_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        launch_modal,
        "_validate_provider_approval_inputs",
        lambda *_args, **_kwargs: provider_chain,
    )
    monkeypatch.setattr(
        launch_modal,
        "_accepted_cuda_run_id_from_preflight",
        lambda _root, _path: "modal-cuda-environment-accepted",
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda arguments, **_kwargs: (
            (
                {
                    "gate": "candidate_resume_preflight_validated",
                    "path": (
                        _PREFLIGHT_RECEIPT_PATH
                    ),
                    "sha256": "f" * 64,
                },
            )
            if arguments.action in {"canary", "canaries"}
            else ()
        ),
    )
    cases = (
        (
            _arguments(
                action="download",
                run_id="candidate-source-1",
                verifier_run_id="candidate-verifier-1",
                local_output="outputs/development/modal_downloads",
            ),
            ["candidate-verifier-1"],
        ),
        (
            _arguments(
                action="canaries",
                run_id="provider-cohort-1",
                provider_approved=True,
            ),
            [
                f"provider-cohort-1-{launch_modal.canary_run_suffix(harness)}"
                for harness in launch_modal.CANARY_ORDER
            ],
        ),
    )

    for index, (arguments, expected_ids) in enumerate(cases):
        attempt_directory = tmp_path / f"attempts-{index}"
        assert (
            launch_modal.run(
                arguments,
                runner=lambda *_args, **_kwargs: _FakeProcess(),
                receipt_directory=attempt_directory,
                process_group_capture=lambda child: child.pid,
                process_group_terminator=lambda *_args, **_kwargs: None,
            )
            == 0
        )
        receipt_path = _terminal_receipt_path(attempt_directory)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "succeeded"
        assert receipt["failure_kind"] is None
        assert receipt["concrete_remote_run_ids"] == expected_ids
        assert receipt["modal_cli_process_started"] is True
        assert receipt["remote_execution_state"] == "may_have_started"
        assert receipt["returncode"] == 0
        assert receipt["process_group_closed"] is True


def test_attempt_receipt_refuses_overwrite(tmp_path: Path) -> None:
    receipt = launch_modal.ModalActionAttemptReceipt(
        schema_name="ModalActionAttemptReceipt",
        schema_version="3.6",
        attempt_id="a" * 32,
        started_at_utc="2026-08-09T00:00:00Z",
        finished_at_utc="2026-08-09T00:00:01Z",
        status="preflight_rejected",
        failure_kind="preflight",
        action="cuda-environment",
        run_id="modal-cuda-env-20260809-01",
        concrete_remote_run_ids=("modal-cuda-env-20260809-01",),
        remote_run_reservations=(),
        local_host_anchor_path=None,
        local_host_anchor_sha256=None,
        local_boot_started_at_unix_microseconds=None,
        local_boot_session_sha256=None,
        source_run_id=None,
        verifier_run_id=None,
        harness=None,
        source_tree_sha256=None,
        cohort_id="modal-cuda-env-20260809-01",
        approved_image_source_sha256="b" * 64,
        modal_command_sha256=None,
        launch_capability_sha256=None,
        modal_profile="scalingintelligence",
        modal_environment="main",
        outer_cli_timeout_seconds=1200,
        modal_cost_cap_usd="0.25",
        modal_resource_profile=launch_modal.modal_resource_profile(
            "cuda-environment"
        ),
        modal_price_basis_path=None,
        modal_price_basis_sha256=None,
        modal_cost_estimate=None,
        modal_cost_approved=True,
        provider_cost_approved=False,
        provider_cost_cap_usd=None,
        provider_approval_plan_path=None,
        approval_plan_sha256=None,
        provider_price_basis_path=None,
        provider_price_basis_sha256=None,
        predecessor_receipts=(),
        source_evidence_recovery=False,
        local_process_start_receipt_path=None,
        local_process_start_receipt_sha256=None,
        local_process_id=None,
        local_process_group_id=None,
        local_session_id=None,
        modal_cli_process_started=False,
        remote_execution_state="definitely_not_started",
        returncode=None,
        process_group_closed=None,
    )
    first = launch_modal._write_attempt_receipt(
        receipt,
        project_root=tmp_path,
        receipt_directory=tmp_path / "attempts",
    )
    original = first.read_bytes()
    metadata = first.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1

    with pytest.raises(FileExistsError):
        launch_modal._write_attempt_receipt(
            receipt,
            project_root=tmp_path,
            receipt_directory=tmp_path / "attempts",
        )
    assert first.read_bytes() == original


def test_production_journals_are_cohort_scoped_and_rejections_are_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    run_id = "fresh-cuda-cohort-1"
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id=run_id,
    )
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )

    captured_environment: dict[str, str] = {}

    def validating_runner(_command, **kwargs):
        captured_environment.update(kwargs["env"])
        assert launch_modal.local_launch_authorized(
            kwargs["env"],
            image_source_sha256=image_sha256,
            project_root=project,
        )
        canonical_python = launch_modal.sys.executable
        with monkeypatch.context() as private_runtime:
            private_runtime.setattr(
                launch_modal.sys,
                "executable",
                str(project / "outputs" / "readiness" / "private-python-copy"),
            )
            assert launch_modal.local_launch_authorized(
                kwargs["env"],
                image_source_sha256=image_sha256,
                project_root=project,
                modal_command_python_executable=canonical_python,
            )
            private_validated = (
                launch_modal.validate_local_action_intent_for_entrypoint(
                    project_root=project,
                    identity=identity,
                    attempt_id="a" * 32,
                    expected_intent_sha256=kwargs["env"][
                        launch_modal.MODAL_ACTION_INTENT_SHA256_ENV
                    ],
                    launch_nonce=kwargs["env"][
                        launch_modal.MODAL_LAUNCH_NONCE_ENV
                    ],
                    action="cuda-environment",
                    run_id=run_id,
                    source_run_id=None,
                    verifier_run_id=None,
                    harness=None,
                    modal_command_python_executable=canonical_python,
                )
            )
            assert private_validated["attempt_id"] == "a" * 32
        wrong_environment = dict(kwargs["env"])
        wrong_environment[launch_modal.MODAL_ENVIRONMENT_ENV] = "staging"
        assert not launch_modal.local_launch_authorized(
            wrong_environment,
            image_source_sha256=image_sha256,
            project_root=project,
        )
        wrong_profile = dict(kwargs["env"])
        wrong_profile[launch_modal.MODAL_PROFILE_ENV] = "default"
        assert not launch_modal.local_launch_authorized(
            wrong_profile,
            image_source_sha256=image_sha256,
            project_root=project,
        )
        validated = launch_modal.validate_local_action_intent_for_entrypoint(
            project_root=project,
            identity=identity,
            attempt_id="a" * 32,
            expected_intent_sha256=kwargs["env"][
                launch_modal.MODAL_ACTION_INTENT_SHA256_ENV
            ],
            launch_nonce=kwargs["env"][launch_modal.MODAL_LAUNCH_NONCE_ENV],
            action="cuda-environment",
            run_id=run_id,
            source_run_id=None,
            verifier_run_id=None,
            harness=None,
        )
        assert validated["attempt_id"] == "a" * 32
        with pytest.raises(ValueError, match="differs from the invocation"):
            launch_modal.validate_local_action_intent_for_entrypoint(
                project_root=project,
                identity=identity,
                attempt_id="a" * 32,
                expected_intent_sha256=kwargs["env"][
                    launch_modal.MODAL_ACTION_INTENT_SHA256_ENV
                ],
                launch_nonce=kwargs["env"][launch_modal.MODAL_LAUNCH_NONCE_ENV],
                action="cuda-environment",
                run_id="different-cuda-run",
                source_run_id=None,
                verifier_run_id=None,
                harness=None,
            )
        return _FakeProcess()

    assert (
        launch_modal.run(
            _arguments(
                run_id=run_id,
                cohort_id=run_id,
                expected_image_source_sha256=image_sha256,
            ),
            runner=validating_runner,
            project_root=project,
            attempt_id_factory=lambda _size: "a" * 32,
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: None,
        )
        == 0
    )
    intent_path = project.joinpath(
        *launch_modal.modal_action_intent_receipt_path(
            identity,
            "a" * 32,
        ).parts
    )
    terminal_path = project.joinpath(
        *launch_modal.modal_action_terminal_receipt_path(
            identity,
            "a" * 32,
        ).parts
    )
    assert intent_path.is_file()
    assert terminal_path.is_file()
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    for payload in (intent, terminal):
        assert payload["source_tree_sha256"] == _SOURCE_TREE_SHA256
        assert payload["approved_image_source_sha256"] == image_sha256
        assert payload["cohort_id"] == run_id
        assert payload["modal_environment"] == "main"
        assert payload["modal_resource_profile"]["modal_environment"] == "main"
        assert payload["remote_run_reservations"] == [
            {
                "run_id": run_id,
                "path": launch_modal.modal_remote_run_reservation_path(
                    run_id
                ).as_posix(),
                "sha256": intent["remote_run_reservations"][0]["sha256"],
            }
        ]
    with pytest.raises(ValueError, match="terminal receipt already exists"):
        launch_modal.validate_local_action_intent_for_entrypoint(
            project_root=project,
            identity=identity,
            attempt_id="a" * 32,
            expected_intent_sha256=captured_environment[
                launch_modal.MODAL_ACTION_INTENT_SHA256_ENV
            ],
            launch_nonce=captured_environment[launch_modal.MODAL_LAUNCH_NONCE_ENV],
            action="cuda-environment",
            run_id=run_id,
            source_run_id=None,
            verifier_run_id=None,
            harness=None,
        )
    assert not launch_modal.local_launch_authorized(
        captured_environment,
        image_source_sha256=image_sha256,
        project_root=project,
    )

    with pytest.raises(ValueError, match="approval"):
        launch_modal.run(
            _arguments(approved=False),
            project_root=project,
            attempt_id_factory=lambda _size: "b" * 32,
        )
    rejection_path = project.joinpath(
        *launch_modal.modal_launch_rejection_receipt_path("b" * 32).parts
    )
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    assert rejection["source_tree_sha256"] is None
    assert rejection["modal_cli_process_started"] is False
    cohort_root = project.joinpath(
        *launch_modal.modal_action_attempt_directory(identity).parent.parts
    )
    assert not list(cohort_root.rglob(f"{'b' * 32}*"))


def test_global_run_reservation_rejects_cross_cohort_reuse_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: _fake_local_freeze_bindings(),
    )
    shared_run_id = "globally-shared-offline-run"
    assert launch_modal.run(
        _arguments(
            action="offline-smoke",
            run_id=shared_run_id,
            cohort_id="cuda-cohort-a",
            expected_image_source_sha256=image_sha256,
        ),
        runner=lambda *_args, **_kwargs: _FakeProcess(),
        project_root=project,
        attempt_id_factory=lambda _size: "1" * 32,
        process_group_capture=lambda child: child.pid,
        process_group_terminator=lambda *_args, **_kwargs: None,
    ) == 0

    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("globally reused run ID must not start Modal")

    with pytest.raises(ValueError, match="already globally reserved"):
        launch_modal.run(
            _arguments(
                action="offline-smoke",
                run_id=shared_run_id,
                cohort_id="cuda-cohort-b",
                expected_image_source_sha256=image_sha256,
            ),
            runner=forbidden_runner,
            project_root=project,
            attempt_id_factory=lambda _size: "2" * 32,
        )
    assert called is False
    second_identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id="cuda-cohort-b",
    )
    assert not project.joinpath(
        *launch_modal.modal_action_intent_receipt_path(
            second_identity,
            "2" * 32,
        ).parts
    ).exists()
    assert not project.joinpath(
        *launch_modal.modal_action_terminal_receipt_path(
            second_identity,
            "2" * 32,
        ).parts
    ).exists()
    rejection_path = project.joinpath(
        *launch_modal.modal_launch_rejection_receipt_path("2" * 32).parts
    )
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    _assert_exact_attempt_receipt_schema(rejection)
    assert rejection["failure_kind"] == "preflight"
    assert rejection["source_tree_sha256"] == _SOURCE_TREE_SHA256
    assert rejection["cohort_id"] == second_identity.cohort_id
    assert rejection["remote_run_reservations"][0]["run_id"] == shared_run_id
    assert rejection["modal_cli_process_started"] is False


def test_global_reservation_allows_verifier_source_reference_but_not_destination_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: _fake_local_freeze_bindings(),
    )
    cohort_id = "cuda-cohort-verifier"
    source_run_id = "candidate-source-for-verifier"
    verifier_run_id = "fresh-verifier-destination"
    assert launch_modal.run(
        _arguments(
            action="candidate-smoke",
            run_id=source_run_id,
            cohort_id=cohort_id,
            expected_image_source_sha256=image_sha256,
        ),
        runner=lambda *_args, **_kwargs: _FakeProcess(),
        project_root=project,
        attempt_id_factory=lambda _size: "3" * 32,
        process_group_capture=lambda child: child.pid,
        process_group_terminator=lambda *_args, **_kwargs: None,
    ) == 0
    assert launch_modal.run(
        _arguments(
            action="verify",
            run_id=source_run_id,
            verifier_run_id=verifier_run_id,
            cohort_id=cohort_id,
            expected_image_source_sha256=image_sha256,
        ),
        runner=lambda *_args, **_kwargs: _FakeProcess(),
        project_root=project,
        attempt_id_factory=lambda _size: "4" * 32,
        process_group_capture=lambda child: child.pid,
        process_group_terminator=lambda *_args, **_kwargs: None,
    ) == 0
    with pytest.raises(ValueError, match="already globally reserved"):
        launch_modal.run(
            _arguments(
                action="verify",
                run_id=source_run_id,
                verifier_run_id=verifier_run_id,
                cohort_id="another-verifier-cohort",
                expected_image_source_sha256=image_sha256,
            ),
            runner=lambda *_args, **_kwargs: pytest.fail("Modal must not start"),
            project_root=project,
            attempt_id_factory=lambda _size: "5" * 32,
        )


def test_preserved_legacy_remote_run_id_is_never_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: _fake_local_freeze_bindings(),
    )
    with pytest.raises(ValueError, match="preserved legacy ID"):
        launch_modal.run(
            _arguments(
                action="offline-smoke",
                run_id="modal-cuda-env-20260809-02",
                cohort_id="new-cohort",
                expected_image_source_sha256=image_sha256,
            ),
            runner=lambda *_args, **_kwargs: pytest.fail("Modal must not start"),
            project_root=project,
            attempt_id_factory=lambda _size: "6" * 32,
        )


def test_final_migration_lineage_seal_rejects_new_intent_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    image_sha256 = "d" * 64
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id="sealed-cohort",
    )
    seal = project.joinpath(
        *launch_modal.modal_migration_lineage_path(identity).parts
    )
    _write_json(seal, {"schema_name": "ModalMigrationLineage"})
    monkeypatch.setattr(
        launch_modal,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_sha256),
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_applied_patch_bundle",
        lambda _root: None,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: _fake_local_freeze_bindings(),
    )
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("sealed cohort must not start Modal")

    with pytest.raises(ValueError, match="lineage seal already exists"):
        launch_modal.run(
            _arguments(
                action="offline-smoke",
                run_id="sealed-cohort-new-run",
                cohort_id=identity.cohort_id,
                expected_image_source_sha256=image_sha256,
            ),
            runner=forbidden_runner,
            project_root=project,
            attempt_id_factory=lambda _size: "8" * 32,
        )
    assert called is False
    assert not project.joinpath(
        *launch_modal.modal_action_intent_receipt_path(
            identity,
            "8" * 32,
        ).parts
    ).exists()
    assert not project.joinpath(
        *launch_modal.modal_remote_run_reservation_path(
            "sealed-cohort-new-run"
        ).parts
    ).exists()
    assert not project.joinpath(
        *launch_modal.modal_action_terminal_receipt_path(
            identity,
            "8" * 32,
        ).parts
    ).exists()
    rejection_path = project.joinpath(
        *launch_modal.modal_launch_rejection_receipt_path("8" * 32).parts
    )
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    _assert_exact_attempt_receipt_schema(rejection)
    assert rejection["failure_kind"] == "preflight"
    assert rejection["source_tree_sha256"] == _SOURCE_TREE_SHA256
    assert rejection["cohort_id"] == identity.cohort_id
    assert rejection["remote_run_reservations"] == []
    assert rejection["modal_cli_process_started"] is False


def test_predecessor_path_and_raw_sha_pairs_are_action_exact() -> None:
    with pytest.raises(ValueError, match="cuda-receipt-sha256"):
        launch_modal._validate_arguments(
            _arguments(action="offline-smoke", cuda_receipt_sha256="")
        )
    with pytest.raises(ValueError, match="unrelated"):
        launch_modal._validate_arguments(
            _arguments(cuda_receipt_path=_CUDA_RECEIPT_PATH)
        )
    with pytest.raises(ValueError, match="source terminal receipt"):
        launch_modal._validate_arguments(
            _arguments(action="verify", source_action_attempt_receipt_sha256="")
        )


def test_attempt_journal_rejects_parent_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from study import serialization

    attempt_id = "b" * 32
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    original_attempts = tmp_path / "attempts-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = serialization._open_exclusive_json_parent
    swapped = False

    def swapping_open(destination: Path, *, create: bool):
        nonlocal swapped
        descriptor, absolute = original_open(destination, create=create)
        if not swapped:
            swapped = True
            attempts.rename(original_attempts)
            attempts.symlink_to(outside, target_is_directory=True)
        return descriptor, absolute

    monkeypatch.setattr(
        serialization,
        "_open_exclusive_json_parent",
        swapping_open,
    )

    with pytest.raises(ValueError, match="parent changed|symlink"):
        launch_modal._write_attempt_payload(
            {"attempt_id": attempt_id},
            attempt_id=attempt_id,
            filename=f"{attempt_id}.intent.json",
            project_root=tmp_path,
            receipt_directory=attempts,
        )

    assert not (outside / f"{attempt_id}.intent.json").exists()
    assert not (original_attempts / f"{attempt_id}.intent.json").exists()


def test_secure_approval_reader_rejects_public_mode_and_hardlinks(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    logical = "approvals/input.json"
    approval = project / logical
    _write_json(approval, {"approved": True})
    approval.chmod(0o644)
    with pytest.raises(ValueError, match="private owned single-link"):
        launch_modal._read_project_json_file(project, logical, "approval")

    approval.chmod(0o600)
    alias = approval.with_name("alias.json")
    alias.hardlink_to(approval)
    with pytest.raises(ValueError, match="private owned single-link"):
        launch_modal._read_project_json_file(
            project,
            alias.relative_to(project).as_posix(),
            "approval",
        )


def test_secure_approval_reader_rejects_leaf_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    logical = "approvals/input.json"
    approval = project / logical
    _write_json(approval, {"approved": True})
    original_read = launch_modal.os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            approval.rename(approval.with_name("input-original.json"))
            _write_json(approval, {"approved": False})
        return chunk

    monkeypatch.setattr(launch_modal.os, "read", swapping_read)
    with pytest.raises(ValueError, match="changed after it was read"):
        launch_modal._read_project_json_file(project, logical, "approval")


def test_secure_approval_reader_rejects_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    approval_parent = project / "approvals"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    logical = "approvals/input.json"
    _write_json(project / logical, {"approved": True})
    original_parent = project / "approvals-original"
    original_reader = launch_modal._read_json_leaf_from_directory
    swapped = False

    def swapping_reader(*args, **kwargs):
        nonlocal swapped
        result = original_reader(*args, **kwargs)
        if not swapped:
            swapped = True
            approval_parent.rename(original_parent)
            approval_parent.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(
        launch_modal,
        "_read_json_leaf_from_directory",
        swapping_reader,
    )
    with pytest.raises(ValueError, match="unsafe component|parent changed"):
        launch_modal._read_project_json_file(project, logical, "approval")


def test_direct_local_modal_app_import_exposes_no_remote_objects(monkeypatch) -> None:
    monkeypatch.delenv(launch_modal.MODAL_LAUNCH_NONCE_ENV, raising=False)
    monkeypatch.delenv(launch_modal.MODAL_LAUNCH_SOURCE_ENV, raising=False)
    spec = importlib.util.spec_from_file_location(
        "modal_app_without_launch_authorization",
        ROOT / "modal_app.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.modal is not None
    assert module._MODAL_OBJECTS_ENABLED is False
    assert module.app is None
    assert module.IMAGE is None
    assert module.ARTIFACT_VOLUME is None
    assert module.PROVIDER_SECRET is None
    assert not hasattr(module, "candidate_smoke")


def test_launch_authorization_rejects_format_only_environment() -> None:
    source_hash = "b" * 64
    environment = {
        launch_modal.MODAL_LAUNCH_NONCE_ENV: "a" * 64,
        launch_modal.MODAL_LAUNCH_SOURCE_ENV: source_hash,
        launch_modal.MODAL_LAUNCH_SOURCE_TREE_ENV: "e" * 64,
        launch_modal.MODAL_LAUNCH_COHORT_ENV: "cohort-1",
        launch_modal.MODAL_ACTION_ATTEMPT_ID_ENV: "c" * 32,
        launch_modal.MODAL_ACTION_INTENT_SHA256_ENV: "d" * 64,
    }
    assert not launch_modal.local_launch_authorized(
        environment,
        image_source_sha256=source_hash,
    )
    assert not launch_modal.local_launch_authorized(
        environment,
        image_source_sha256="c" * 64,
    )
    environment[launch_modal.MODAL_LAUNCH_NONCE_ENV] = "not-a-capability"
    assert not launch_modal.local_launch_authorized(
        environment,
        image_source_sha256=source_hash,
    )


def test_local_host_anchor_is_private_machine_bound_and_reboot_stable(
    tmp_path: Path,
) -> None:
    receipt_directory = tmp_path / "attempts"
    first = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=receipt_directory,
        host_anchor_id_factory=lambda _size: "1" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
    )
    anchor_path = first.anchor.canonical_path
    original = anchor_path.read_bytes()
    first_sha256 = first.host_anchor_sha256
    first_session_sha256 = first.boot_session_sha256
    metadata = anchor_path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert _TEST_MACHINE_ID not in original
    assert _TEST_BOOT_IDENTITY not in original
    assert b"hostname" not in original.lower()
    first.anchor.ctime_ns -= 1
    first.require_current()
    first.close()

    rebooted = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=receipt_directory,
        host_anchor_id_factory=lambda _size: "2" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: (
            _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_000_000
        ),
        boot_identity_provider=lambda: _TEST_NEXT_BOOT_IDENTITY,
    )
    try:
        assert rebooted.host_anchor_sha256 == first_sha256
        assert anchor_path.read_bytes() == original
        assert rebooted.boot_session_sha256 != first_session_sha256
        assert launch_modal.modal_local_boot_session_relation(
            local_host_anchor_sha256=first_sha256,
            local_boot_started_at_unix_microseconds=(
                _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS
            ),
            local_boot_session_sha256=first_session_sha256,
            boot_session_provider=lambda: (
                _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_000_000
            ),
            boot_identity_provider=lambda: _TEST_NEXT_BOOT_IDENTITY,
        ) == "different_boot_session"
    finally:
        rebooted.close()

    with pytest.raises(ValueError, match="another machine"):
        launch_modal._open_or_create_local_containment_binding(
            project_root=tmp_path,
            receipt_directory=receipt_directory,
            host_anchor_id_factory=lambda _size: "3" * 64,
            machine_identity_provider=lambda: b"different-test-machine-id",
            boot_session_provider=lambda: (
                _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_000_000
            ),
            boot_identity_provider=lambda: _TEST_NEXT_BOOT_IDENTITY,
        )
    assert anchor_path.read_bytes() == original


@pytest.mark.parametrize("mutation", ("mode", "hardlink", "symlink"))
def test_local_host_anchor_rejects_unsafe_leaf_metadata(
    tmp_path: Path,
    mutation: str,
) -> None:
    receipt_directory = tmp_path / "attempts"
    binding = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=receipt_directory,
        host_anchor_id_factory=lambda _size: "4" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
    )
    anchor_path = binding.anchor.canonical_path
    original = anchor_path.read_bytes()
    binding.close()
    if mutation == "mode":
        anchor_path.chmod(0o644)
    elif mutation == "hardlink":
        anchor_path.with_name("host-anchor-alias.json").hardlink_to(anchor_path)
    else:
        preserved = anchor_path.with_name("host-anchor-original.json")
        anchor_path.rename(preserved)
        anchor_path.symlink_to(preserved)

    with pytest.raises(ValueError, match="unsafe|changed"):
        launch_modal._open_or_create_local_containment_binding(
            project_root=tmp_path,
            receipt_directory=receipt_directory,
            host_anchor_id_factory=lambda _size: "5" * 64,
            machine_identity_provider=lambda: _TEST_MACHINE_ID,
            boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
            boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
        )
    if mutation != "symlink":
        assert anchor_path.read_bytes() == original


def test_local_host_anchor_rejects_owner_and_namespace_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_directory = tmp_path / "attempts"
    binding = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=receipt_directory,
        host_anchor_id_factory=lambda _size: "6" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
    )
    anchor_path = binding.anchor.canonical_path
    containment = anchor_path.parent
    preserved = containment.with_name(".modal_local_containment-original")
    original = anchor_path.read_bytes()
    containment.rename(preserved)
    containment.mkdir(mode=0o700)
    _write_json(containment / anchor_path.name, json.loads(original))
    with pytest.raises(ValueError, match="path changed"):
        binding.require_current()
    assert (preserved / anchor_path.name).read_bytes() == original
    binding.close()

    replacement = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=receipt_directory,
        host_anchor_id_factory=lambda _size: "7" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
    )
    replacement.close()
    real_uid = os.getuid()
    monkeypatch.setattr(launch_modal.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(ValueError, match="unsafe"):
        launch_modal._open_or_create_local_containment_binding(
            project_root=tmp_path,
            receipt_directory=receipt_directory,
            host_anchor_id_factory=lambda _size: "8" * 64,
            machine_identity_provider=lambda: _TEST_MACHINE_ID,
            boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
            boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
        )


def test_programmatic_receipt_directory_never_creates_through_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = tmp_path / "redirect"
    redirect.symlink_to(outside, target_is_directory=True)
    outside_created = outside / "must-not-be-created"

    with pytest.raises(ValueError, match="unsafe component"):
        launch_modal._prepare_local_containment_directory(
            project_root=tmp_path,
            receipt_directory=redirect / outside_created.name,
            include_process_starts=True,
        )

    assert outside_created.exists() is False


@pytest.mark.parametrize(
    "invalid",
    (True, 1.0, "1700000000000000", -1, 10**30),
)
def test_boot_session_provider_rejects_type_and_time_spoof(invalid: object) -> None:
    with pytest.raises(ValueError, match="boot-session"):
        launch_modal._validated_boot_started_at_unix_microseconds(invalid)


@pytest.mark.parametrize(
    "invalid",
    (
        "00112233-4455-6677-8899-aabbccddeeff",
        bytearray(_TEST_BOOT_IDENTITY),
        b"",
        b"\x00" * 16,
        b"x" * 15,
        b"x" * 17,
    ),
)
def test_boot_identity_provider_rejects_raw_type_and_size_spoof(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="OS boot identity"):
        launch_modal._validated_boot_identity(invalid)


@pytest.mark.parametrize(
    ("current_start", "accepted"),
    (
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 409_185, True),
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 641_717, True),
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS - 1, False),
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_000_000, False),
    ),
)
def test_same_boot_uuid_allows_only_same_second_start_time(
    current_start: int,
    accepted: bool,
) -> None:
    host_sha256 = "a" * 64
    recorded_session = launch_modal._local_boot_session_sha256(
        host_sha256,
        _TEST_BOOT_IDENTITY,
    )
    kwargs = {
        "local_host_anchor_sha256": host_sha256,
        "local_boot_started_at_unix_microseconds": (
            _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 641_717
        ),
        "local_boot_session_sha256": recorded_session,
        "boot_session_provider": lambda: current_start,
        "boot_identity_provider": lambda: _TEST_BOOT_IDENTITY,
    }
    if accepted:
        assert launch_modal.modal_local_boot_session_relation(
            **kwargs
        ) == "same_boot_session"
    else:
        with pytest.raises(ValueError, match="same OS boot identity"):
            launch_modal.modal_local_boot_session_relation(**kwargs)


@pytest.mark.parametrize(
    ("current_start", "accepted"),
    (
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 409_185, False),
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 641_717, False),
        (_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_409_185, True),
    ),
)
def test_changed_boot_uuid_requires_strictly_later_start_second(
    current_start: int,
    accepted: bool,
) -> None:
    host_sha256 = "b" * 64
    recorded_session = launch_modal._local_boot_session_sha256(
        host_sha256,
        _TEST_BOOT_IDENTITY,
    )
    kwargs = {
        "local_host_anchor_sha256": host_sha256,
        "local_boot_started_at_unix_microseconds": (
            _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 641_717
        ),
        "local_boot_session_sha256": recorded_session,
        "boot_session_provider": lambda: current_start,
        "boot_identity_provider": lambda: _TEST_NEXT_BOOT_IDENTITY,
    }
    if accepted:
        assert launch_modal.modal_local_boot_session_relation(
            **kwargs
        ) == "different_boot_session"
    else:
        with pytest.raises(ValueError, match="does not have a later start"):
            launch_modal.modal_local_boot_session_relation(**kwargs)


def test_held_local_boot_session_rejects_provider_drift(tmp_path: Path) -> None:
    observed = [_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS]
    binding = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=tmp_path / "attempts",
        host_anchor_id_factory=lambda _size: "9" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: observed[0],
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
    )
    observed[0] += 1_000_000
    try:
        with pytest.raises(ValueError, match="boot-start time changed"):
            binding.require_current()
    finally:
        binding.close()


def test_held_local_boot_session_allows_same_second_subsecond_drift(
    tmp_path: Path,
) -> None:
    observed = [_TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 641_717]
    binding = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=tmp_path / "attempts",
        host_anchor_id_factory=lambda _size: "9" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: observed[0],
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
    )
    observed[0] = _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 409_185
    try:
        binding.require_current()
    finally:
        binding.close()


def test_held_local_boot_identity_rejects_provider_drift(tmp_path: Path) -> None:
    observed = [_TEST_BOOT_IDENTITY]
    binding = launch_modal._open_or_create_local_containment_binding(
        project_root=tmp_path,
        receipt_directory=tmp_path / "attempts",
        host_anchor_id_factory=lambda _size: "a" * 64,
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: observed[0],
    )
    observed[0] = _TEST_NEXT_BOOT_IDENTITY
    try:
        with pytest.raises(ValueError, match="OS boot identity changed"):
            binding.require_current()
    finally:
        binding.close()


@pytest.mark.parametrize(
    "invalid",
    (
        "raw-machine-id",
        bytearray(b"raw-machine-id-value"),
        b"",
        b"\x00" * 16,
        b"x" * 257,
    ),
)
def test_machine_identity_provider_rejects_raw_type_and_size_spoof(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="machine identity"):
        launch_modal._validated_machine_identity(invalid)


@pytest.mark.parametrize(
    "invalid",
    (
        "process-birth",
        bytearray(b"process-birth"),
        b"",
        b"\x00" * 8,
        b"x" * 7,
        b"x" * 257,
    ),
)
def test_process_birth_identity_provider_rejects_raw_type_and_size_spoof(
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="process birth identity"):
        launch_modal._validated_process_birth_identity(invalid)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("process_id", True),
        ("process_id", "424242"),
        ("process_id", 424242.0),
        ("expected_process_group_id", False),
        ("expected_session_id", 0),
    ),
)
def test_local_process_start_rejects_pid_type_spoof(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    _terminal_path, _intent, terminal = _source_attempt_pair(tmp_path)
    marker_path = tmp_path / terminal["local_process_start_receipt_path"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = replacement
    with pytest.raises(ValueError, match="process identifier"):
        launch_modal._validate_modal_local_process_start_receipt(marker)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("local_host_anchor_sha256", True, "host-anchor SHA-256"),
        ("local_boot_session_sha256", 1, "boot-session SHA-256"),
        (
            "local_boot_started_at_unix_microseconds",
            1.0,
            "boot-session start",
        ),
        ("process_birth_identity_sha256", True, "birth-identity SHA-256"),
    ),
)
def test_local_process_start_rejects_containment_type_spoof(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    _terminal_path, _intent, terminal = _source_attempt_pair(tmp_path)
    marker_path = tmp_path / terminal["local_process_start_receipt_path"]
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = replacement
    with pytest.raises(ValueError, match=message):
        launch_modal._validate_modal_local_process_start_receipt(marker)


def test_local_process_start_marker_is_create_only_and_attempt_unique(
    tmp_path: Path,
) -> None:
    _terminal_path, _intent, terminal = _source_attempt_pair(tmp_path)
    marker_path = tmp_path / terminal["local_process_start_receipt_path"]
    original = marker_path.read_bytes()
    receipt = launch_modal.ModalLocalProcessStartReceipt(
        **json.loads(original)
    )

    with pytest.raises(
        launch_modal.ModalProcessStartReceiptError,
        match="attempt ID is already used",
    ):
        launch_modal._publish_modal_local_process_start(
            receipt,
            project_root=tmp_path,
            receipt_directory=None,
        )

    assert marker_path.read_bytes() == original


def test_same_boot_process_probe_is_signal_zero_only_and_reuse_safe(
    tmp_path: Path,
) -> None:
    _terminal_path, _intent, terminal = _source_attempt_pair(tmp_path)
    calls: list[tuple[int, int]] = []
    process_id = terminal["local_process_id"]
    result = launch_modal.probe_same_boot_modal_process_group(
        tmp_path,
        process_start_receipt_path=terminal[
            "local_process_start_receipt_path"
        ],
        process_start_receipt_sha256=terminal[
            "local_process_start_receipt_sha256"
        ],
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
        process_group_lookup=lambda _pid: process_id,
        session_lookup=lambda _pid: process_id,
        signal_zero=lambda pgid, signal_number: calls.append(
            (pgid, signal_number)
        ),
    )
    assert result == "same_boot_process_group_exists"
    assert calls == [(process_id, 0)]

    calls.clear()
    changed = launch_modal.probe_same_boot_modal_process_group(
        tmp_path,
        process_start_receipt_path=terminal[
            "local_process_start_receipt_path"
        ],
        process_start_receipt_sha256=terminal[
            "local_process_start_receipt_sha256"
        ],
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
        process_group_lookup=lambda _pid: process_id + 1,
        session_lookup=lambda _pid: process_id,
        signal_zero=lambda pgid, signal_number: calls.append(
            (pgid, signal_number)
        ),
    )
    assert changed == "same_boot_process_identity_changed"
    assert calls == []

    exact_numeric_reuse = launch_modal.probe_same_boot_modal_process_group(
        tmp_path,
        process_start_receipt_path=terminal[
            "local_process_start_receipt_path"
        ],
        process_start_receipt_sha256=terminal[
            "local_process_start_receipt_sha256"
        ],
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: _TEST_BOOT_IDENTITY,
        process_birth_identity_provider=lambda _pid: (
            b"different-exact-process-birth-identity"
        ),
        process_group_lookup=lambda _pid: process_id,
        session_lookup=lambda _pid: process_id,
        signal_zero=lambda *_args: pytest.fail(
            "a reused numeric PID must not authorize a group probe"
        ),
    )
    assert exact_numeric_reuse == "same_boot_process_identity_changed"

    rebooted = launch_modal.probe_same_boot_modal_process_group(
        tmp_path,
        process_start_receipt_path=terminal[
            "local_process_start_receipt_path"
        ],
        process_start_receipt_sha256=terminal[
            "local_process_start_receipt_sha256"
        ],
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: (
            _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_000_000
        ),
        boot_identity_provider=lambda: _TEST_NEXT_BOOT_IDENTITY,
        process_group_lookup=lambda _pid: pytest.fail(
            "different boot must not probe a PID"
        ),
        signal_zero=lambda *_args: pytest.fail(
            "different boot must not signal a group"
        ),
    )
    assert rebooted == "different_boot_session"


@pytest.mark.parametrize(
    ("probe_error", "expected"),
    (
        (None, "same_boot_process_group_exists"),
        (ProcessLookupError(), "same_boot_process_group_absent"),
        (PermissionError(), "same_boot_process_group_exists"),
    ),
)
def test_same_boot_probe_checks_group_when_session_leader_is_absent(
    tmp_path: Path,
    probe_error: BaseException | None,
    expected: str,
) -> None:
    _terminal_path, _intent, terminal = _source_attempt_pair(tmp_path)
    process_id = terminal["local_process_id"]
    calls: list[tuple[int, int]] = []

    def signal_zero(process_group_id: int, signal_number: int) -> None:
        calls.append((process_group_id, signal_number))
        if probe_error is not None:
            raise probe_error

    result = launch_modal.probe_same_boot_modal_process_group(
        tmp_path,
        process_start_receipt_path=terminal[
            "local_process_start_receipt_path"
        ],
        process_start_receipt_sha256=terminal[
            "local_process_start_receipt_sha256"
        ],
        machine_identity_provider=lambda: _TEST_MACHINE_ID,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        process_group_lookup=lambda _pid: (_ for _ in ()).throw(
            ProcessLookupError()
        ),
        session_lookup=lambda _pid: pytest.fail(
            "session lookup must stop after the leader disappears"
        ),
        signal_zero=signal_zero,
    )

    assert result == expected
    assert calls == [(process_id, 0)]


@pytest.mark.parametrize("phase", ("before", "after"))
@pytest.mark.parametrize("reservation_index", range(len(launch_modal.CANARY_ORDER)))
def test_host_anchor_precedes_every_crash_interrupted_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    reservation_index: int,
) -> None:
    attempt_directory = tmp_path / "attempts"
    original_create = launch_modal.create_json_exclusive
    reservation_calls = 0
    preserved: dict[Path, bytes] = {}
    monkeypatch.setattr(
        launch_modal,
        "_validate_provider_approval_inputs",
        lambda *_args, **_kwargs: {
            "provider_cost_cap_usd": "2.00",
            "provider_approval_plan_path": (
                "outputs/readiness/provider_canary_approval/plan.json"
            ),
            "approval_plan_sha256": "a" * 64,
            "provider_price_basis_path": (
                "outputs/readiness/modal_resource_cleanup/price.json"
            ),
            "provider_price_basis_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        launch_modal,
        "_accepted_cuda_run_id_from_preflight",
        lambda *_args, **_kwargs: "modal-cuda-environment-accepted",
    )

    def crashing_create(path: str | Path, payload: object) -> None:
        nonlocal reservation_calls
        destination = Path(path)
        is_reservation = destination.parent.name == "remote_run_reservations"
        if is_reservation and reservation_calls == reservation_index:
            if phase == "before":
                raise KeyboardInterrupt("synthetic crash before reservation")
            original_create(destination, payload)
            preserved[destination] = destination.read_bytes()
            raise KeyboardInterrupt("synthetic crash after reservation")
        original_create(destination, payload)
        if is_reservation:
            preserved[destination] = destination.read_bytes()
            reservation_calls += 1

    monkeypatch.setattr(launch_modal, "create_json_exclusive", crashing_create)
    with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
        launch_modal.run(
            _arguments(
                action="canaries",
                run_id=f"crash-reservation-{reservation_index}-{phase}",
                provider_approved=True,
            ),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            receipt_directory=attempt_directory,
        )
    anchor = launch_modal._local_containment_destination(
        launch_modal.modal_local_host_anchor_path().as_posix(),
        project_root=ROOT,
        receipt_directory=attempt_directory,
    )
    assert anchor.is_file()
    assert stat.S_IMODE(anchor.stat().st_mode) == 0o600
    expected_count = reservation_index + (phase == "after")
    reservations = sorted(
        (attempt_directory / "remote_run_reservations").glob("*.json")
    )
    assert len(reservations) == expected_count
    assert all(
        json.loads(path.read_text())["schema_version"] == "1.2"
        for path in reservations
    )
    assert all(path.read_bytes() == preserved[path] for path in reservations)
    terminal = json.loads(
        _terminal_receipt_path(attempt_directory).read_text(encoding="utf-8")
    )
    assert terminal["modal_cli_process_started"] is False
    assert terminal["local_host_anchor_sha256"] == hashlib.sha256(
        anchor.read_bytes()
    ).hexdigest()


def test_partial_reservation_publication_routes_terminal_to_global_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_sha256 = "d" * 64
    project = _prepare_minimal_production_project(
        tmp_path,
        monkeypatch,
        image_source_sha256=image_sha256,
    )
    attempt_id = "d" * 32
    cohort_id = "partial-reservation-cohort"
    run_id = "partial-reservation-canaries"
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id=cohort_id,
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_provider_approval_inputs",
        lambda *_args, **_kwargs: {
            "provider_cost_cap_usd": "2.00",
            "provider_approval_plan_path": (
                "outputs/readiness/provider_canary_approval/plan.json"
            ),
            "approval_plan_sha256": "a" * 64,
            "provider_price_basis_path": (
                "outputs/readiness/modal_resource_cleanup/price.json"
            ),
            "provider_price_basis_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        launch_modal,
        "_validate_predecessor_receipts",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        launch_modal,
        "_accepted_cuda_run_id_from_preflight",
        lambda *_args, **_kwargs: "modal-cuda-environment-accepted",
    )
    original_create = launch_modal.create_json_exclusive
    reservation_calls = 0

    def partially_failing_create(path: str | Path, payload: object) -> None:
        nonlocal reservation_calls
        destination = Path(path)
        if destination.parent.name == "modal_remote_run_reservations":
            if reservation_calls == 1:
                raise OSError("synthetic partial reservation publication")
            reservation_calls += 1
        original_create(destination, payload)

    monkeypatch.setattr(
        launch_modal,
        "create_json_exclusive",
        partially_failing_create,
    )
    with pytest.raises(OSError, match="partial reservation publication"):
        launch_modal.run(
            _arguments(
                action="canaries",
                run_id=run_id,
                cohort_id=cohort_id,
                expected_image_source_sha256=image_sha256,
                provider_approved=True,
            ),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            project_root=project,
            attempt_id_factory=lambda _size: attempt_id,
        )

    reservations = tuple(
        project.joinpath(
            *launch_modal.modal_remote_run_reservation_path(
                f"{run_id}-{launch_modal.canary_run_suffix(harness)}"
            ).parts
        )
        for harness in launch_modal.CANARY_ORDER
    )
    assert [path.exists() for path in reservations] == [True, False, False, False]
    assert not project.joinpath(
        *launch_modal.modal_action_intent_receipt_path(
            identity,
            attempt_id,
        ).parts
    ).exists()
    assert not project.joinpath(
        *launch_modal.modal_action_terminal_receipt_path(
            identity,
            attempt_id,
        ).parts
    ).exists()
    rejection_path = project.joinpath(
        *launch_modal.modal_launch_rejection_receipt_path(attempt_id).parts
    )
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    _assert_exact_attempt_receipt_schema(rejection)
    assert rejection["status"] == "preflight_failed"
    assert rejection["failure_kind"] == "preflight"
    assert rejection["source_tree_sha256"] == _SOURCE_TREE_SHA256
    assert rejection["cohort_id"] == cohort_id
    assert len(rejection["remote_run_reservations"]) == len(
        launch_modal.CANARY_ORDER
    )
    assert rejection["modal_cli_process_started"] is False
    assert launch_modal._scan_validated_cohort_intents(
        project_root=project,
        receipt_directory=None,
        identity=identity,
    ) == {}


def test_crash_after_intent_preserves_anchor_reservations_and_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_directory = tmp_path / "attempts"
    original_write = launch_modal._write_action_intent
    frozen: list[tuple[Path, bytes]] = []

    def crashing_write(*args, **kwargs):
        path = original_write(*args, **kwargs)
        frozen.append((path, path.read_bytes()))
        raise KeyboardInterrupt("synthetic crash after intent")

    monkeypatch.setattr(launch_modal, "_write_action_intent", crashing_write)
    with pytest.raises(launch_modal.ModalAttemptReceiptError):
        launch_modal.run(
            _arguments(run_id="crash-after-intent"),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            receipt_directory=attempt_directory,
        )
    [(intent_path, original)] = frozen
    assert intent_path.read_bytes() == original
    intent = json.loads(original)
    assert intent["schema_version"] == "1.6"
    reservation_path = attempt_directory / "remote_run_reservations" / (
        "crash-after-intent.json"
    )
    assert reservation_path.is_file()
    assert json.loads(reservation_path.read_text())["schema_version"] == "1.2"
    terminal = json.loads(
        _terminal_receipt_path(attempt_directory).read_text(encoding="utf-8")
    )
    assert terminal["status"] == "preflight_failed"
    assert terminal["failure_kind"] == "action_intent_post_persistence"
    assert terminal["modal_cli_process_started"] is False
    assert [
        terminal[field]
        for field in (
            "local_process_start_receipt_path",
            "local_process_start_receipt_sha256",
            "local_process_id",
            "local_process_group_id",
            "local_session_id",
        )
    ] == [None] * 5


@pytest.mark.parametrize(
    ("phase", "expected_failure_kind", "cohort_owned"),
    (
        ("before", "action_intent_persistence", False),
        ("after", "action_intent_post_persistence", True),
        ("tampered", "action_intent_persistence_uncertain", False),
    ),
)
def test_intent_write_error_routes_only_exact_owned_intent_to_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_failure_kind: str,
    cohort_owned: bool,
) -> None:
    image_sha256 = "d" * 64
    project = _prepare_minimal_production_project(
        tmp_path,
        monkeypatch,
        image_source_sha256=image_sha256,
    )
    run_id = f"intent-write-{phase}"
    attempt_id = {"before": "a", "after": "b", "tampered": "c"}[phase] * 32
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id=run_id,
    )
    original_write = launch_modal._write_action_intent
    preserved_intent: list[tuple[Path, bytes]] = []

    def failing_write(*args, **kwargs):
        if phase == "before":
            raise OSError("synthetic failure before intent publication")
        path = original_write(*args, **kwargs)
        if phase == "tampered":
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["modal_cost_estimate"]
            _write_json(path, payload)
        preserved_intent.append((path, path.read_bytes()))
        raise OSError("synthetic failure after intent publication")

    monkeypatch.setattr(launch_modal, "_write_action_intent", failing_write)
    with pytest.raises(launch_modal.ModalAttemptReceiptError):
        launch_modal.run(
            _arguments(
                run_id=run_id,
                cohort_id=run_id,
                expected_image_source_sha256=image_sha256,
            ),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            project_root=project,
            attempt_id_factory=lambda _size: attempt_id,
        )

    intent_path = project.joinpath(
        *launch_modal.modal_action_intent_receipt_path(
            identity,
            attempt_id,
        ).parts
    )
    cohort_terminal = project.joinpath(
        *launch_modal.modal_action_terminal_receipt_path(
            identity,
            attempt_id,
        ).parts
    )
    global_terminal = project.joinpath(
        *launch_modal.modal_launch_rejection_receipt_path(attempt_id).parts
    )
    terminal_path = cohort_terminal if cohort_owned else global_terminal
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    _assert_exact_attempt_receipt_schema(terminal)
    assert terminal["status"] == "preflight_failed"
    assert terminal["failure_kind"] == expected_failure_kind
    assert terminal["source_tree_sha256"] == _SOURCE_TREE_SHA256
    assert terminal["cohort_id"] == run_id
    assert terminal["remote_run_reservations"]
    assert terminal["modal_cli_process_started"] is False
    assert cohort_terminal.exists() is cohort_owned
    assert global_terminal.exists() is (not cohort_owned)

    if phase == "before":
        assert not intent_path.exists()
        assert launch_modal._scan_validated_cohort_intents(
            project_root=project,
            receipt_directory=None,
            identity=identity,
        ) == {}
    elif phase == "after":
        [(preserved_path, original)] = preserved_intent
        assert preserved_path == intent_path
        assert intent_path.read_bytes() == original
        records = launch_modal._scan_validated_cohort_intents(
            project_root=project,
            receipt_directory=None,
            identity=identity,
        )
        assert records[attempt_id][1] == hashlib.sha256(original).hexdigest()
    else:
        [(preserved_path, tampered)] = preserved_intent
        assert preserved_path == intent_path
        assert intent_path.read_bytes() == tampered
        with pytest.raises(ValueError, match="invalid exact schema"):
            launch_modal._scan_validated_cohort_intents(
                project_root=project,
                receipt_directory=None,
                identity=identity,
            )


def test_post_intent_preflight_rejection_is_readiness_valid_cohort_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_sha256 = "d" * 64
    project = _prepare_minimal_production_project(
        tmp_path,
        monkeypatch,
        image_source_sha256=image_sha256,
    )
    run_id = "post-intent-preflight-rejection"
    attempt_id = "d" * 32
    identity = launch_modal.ModalLiveCohortIdentity(
        source_tree_sha256=_SOURCE_TREE_SHA256,
        image_source_sha256=image_sha256,
        cohort_id=run_id,
    )
    price_basis_path, price_basis_sha256, _payload = _modal_price_basis(
        project,
        image_sha256,
    )
    freeze_directory = project / (
        "outputs/readiness/modal_only_final/local_engineering_freezes/"
        f"{_SOURCE_TREE_SHA256}"
    )
    local_freeze_bindings = tuple(
        {
            "gate": gate,
            "path": path.relative_to(project).as_posix(),
            "sha256": _write_json(path, {"gate": gate}),
        }
        for gate, path in (
            (
                "local_unit_tested",
                freeze_directory / "unit_test_evidence_receipt.json",
            ),
            (
                "local_offline_smoke_tested",
                freeze_directory / "offline_smoke_evidence_receipt.json",
            ),
            (
                "local_engineering_freeze_validated",
                freeze_directory / "local_engineering_freeze_receipt.json",
            ),
        )
    )
    monkeypatch.setattr(
        launch_modal,
        "validate_local_freeze_evidence",
        lambda _root, **_kwargs: local_freeze_bindings,
    )
    monkeypatch.setattr(
        launch_modal.modal_readiness,
        "historical_local_engineering_freeze_predecessor_bindings",
        lambda bindings, **_kwargs: tuple(bindings),
    )
    original_validate = launch_modal._validate_approval_chain
    validation_calls = 0

    def reject_after_intent(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise ValueError("synthetic post-intent approval rejection")
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        launch_modal,
        "_validate_approval_chain",
        reject_after_intent,
    )
    with pytest.raises(
        ValueError,
        match="synthetic post-intent approval rejection",
    ):
        launch_modal.run(
            _arguments(
                run_id=run_id,
                cohort_id=run_id,
                expected_image_source_sha256=image_sha256,
                modal_price_basis_path=price_basis_path,
                modal_price_basis_sha256=price_basis_sha256,
            ),
            runner=lambda *_args, **_kwargs: pytest.fail("Popen must not run"),
            project_root=project,
            attempt_id_factory=lambda _size: attempt_id,
        )

    assert validation_calls == 2
    intent_path = project.joinpath(
        *launch_modal.modal_action_intent_receipt_path(
            identity,
            attempt_id,
        ).parts
    )
    terminal_path = project.joinpath(
        *launch_modal.modal_action_terminal_receipt_path(
            identity,
            attempt_id,
        ).parts
    )
    global_terminal = project.joinpath(
        *launch_modal.modal_launch_rejection_receipt_path(attempt_id).parts
    )
    assert intent_path.is_file()
    assert terminal_path.is_file()
    assert not global_terminal.exists()
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "preflight_rejected"
    assert terminal["failure_kind"] == "preflight"
    assert terminal["modal_cli_process_started"] is False

    journal, attempts = launch_modal.modal_readiness._cohort_action_journal(
        project,
        identity,
    )
    assert [attempt["attempt_id"] for attempt in attempts] == [attempt_id]
    assert [entry["path"] for entry in journal["intent_receipts"]] == [
        launch_modal.modal_action_intent_receipt_path(
            identity,
            attempt_id,
        ).as_posix()
    ]
    assert [entry["path"] for entry in journal["terminal_receipts"]] == [
        launch_modal.modal_action_terminal_receipt_path(
            identity,
            attempt_id,
        ).as_posix()
    ]


@pytest.mark.parametrize("phase", ("before", "after"))
def test_process_marker_crash_closes_group_and_preserves_any_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    attempt_directory = tmp_path / "attempts"
    process = _FakeProcess(returncode=0)
    cleanup_calls: list[int] = []
    original_publish = launch_modal._publish_modal_local_process_start
    marker_bytes: list[tuple[Path, bytes]] = []

    def crashing_publish(*args, **kwargs):
        if phase == "before":
            raise launch_modal.ModalProcessStartReceiptError(
                "synthetic crash before marker"
            )
        held = original_publish(*args, **kwargs)
        held.binding.ctime_ns -= 1
        held.require_current()
        marker_bytes.append(
            (held.binding.canonical_path, held.binding.canonical_path.read_bytes())
        )
        held.close()
        raise launch_modal.ModalProcessStartReceiptError(
            "synthetic crash after marker"
        )

    monkeypatch.setattr(
        launch_modal,
        "_publish_modal_local_process_start",
        crashing_publish,
    )
    with pytest.raises(launch_modal.ModalProcessStartReceiptError):
        launch_modal.run(
            _arguments(run_id=f"process-marker-{phase}"),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=attempt_directory,
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda _child, **kwargs: cleanup_calls.append(
                kwargs["process_group_id"]
            ),
        )
    assert cleanup_calls == [process.pid]
    terminal = json.loads(
        _terminal_receipt_path(attempt_directory).read_text(encoding="utf-8")
    )
    assert terminal["failure_kind"] == "process_start_receipt_persistence"
    assert terminal["modal_cli_process_started"] is True
    assert terminal["process_group_closed"] is True
    assert terminal["local_process_id"] == process.pid
    assert terminal["local_process_group_id"] == process.pid
    assert terminal["local_session_id"] == process.pid
    assert terminal["local_process_start_receipt_sha256"] is None
    if phase == "after":
        [(marker_path, original)] = marker_bytes
        assert marker_path.read_bytes() == original
    else:
        assert marker_bytes == []


def test_process_marker_failure_and_cleanup_failure_are_both_terminalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=0)
    monkeypatch.setattr(
        launch_modal,
        "_publish_modal_local_process_start",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            launch_modal.ModalProcessStartReceiptError("marker write failed")
        ),
    )
    with pytest.raises(launch_modal.ProcessGroupClosureError):
        launch_modal.run(
            _arguments(run_id="marker-and-cleanup-failure"),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                launch_modal.ProcessGroupClosureError("group remained")
            ),
        )
    terminal = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "cleanup_failed"
    assert terminal["failure_kind"] == (
        "process_start_receipt_and_process_group_cleanup"
    )
    assert terminal["process_group_closed"] is False
    assert terminal["local_process_start_receipt_sha256"] is None


def test_process_marker_namespace_replacement_quarantines_started_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(returncode=0)
    cleanup_calls: list[int] = []
    original_publish = launch_modal._publish_modal_local_process_start
    preserved: list[tuple[Path, bytes]] = []

    def replacing_publish(*args, **kwargs):
        held = original_publish(*args, **kwargs)
        marker_path = held.binding.canonical_path
        original_path = marker_path.with_name(f"{marker_path.stem}.original.json")
        original = marker_path.read_bytes()
        marker_path.rename(original_path)
        marker_path.write_bytes(original)
        marker_path.chmod(0o600)
        preserved.append((original_path, original))
        return held

    monkeypatch.setattr(
        launch_modal,
        "_publish_modal_local_process_start",
        replacing_publish,
    )
    with pytest.raises(launch_modal.ModalAttemptReceiptError):
        launch_modal.run(
            _arguments(run_id="marker-replacement"),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda _child, **kwargs: cleanup_calls.append(
                kwargs["process_group_id"]
            ),
        )
    [(original_path, original)] = preserved
    assert original_path.read_bytes() == original
    assert cleanup_calls == [process.pid]
    terminal = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert terminal["failure_kind"] == "process_start_receipt_persistence"
    assert terminal["process_group_closed"] is True


def test_invalid_popen_pid_is_never_persisted_or_used_as_a_group(
    tmp_path: Path,
) -> None:
    class InvalidPidProcess(_FakeProcess):
        pid = True

        def __init__(self) -> None:
            super().__init__()
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    process = InvalidPidProcess()
    with pytest.raises(launch_modal.ModalProcessStartReceiptError):
        launch_modal.run(
            _arguments(run_id="invalid-pid"),
            runner=lambda *_args, **_kwargs: process,
            receipt_directory=tmp_path / "attempts",
            process_group_terminator=lambda *_args, **_kwargs: pytest.fail(
                "unverified PID must not be used as a process group"
            ),
        )
    assert process.killed is True
    terminal = json.loads(
        _terminal_receipt_path(tmp_path / "attempts").read_text(encoding="utf-8")
    )
    assert terminal["modal_cli_process_started"] is True
    assert terminal["local_process_id"] is None
    assert terminal["local_process_group_id"] is None
    assert terminal["local_session_id"] is None
    assert terminal["process_group_closed"] is False


def test_containment_evidence_never_serializes_machine_config_or_token_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine_secret = b"MACHINE-IDENTITY-MUST-NOT-LEAK"
    boot_secret = bytes.fromhex("ffeeddccbbaa99887766554433221100")
    process_birth_secret = b"PROCESS-BIRTH-IDENTITY-MUST-NOT-LEAK"
    token_secret = "MODAL-TOKEN-MUST-NOT-LEAK"
    monkeypatch.setenv("MODAL_TOKEN_SECRET", token_secret)
    captured_environment: dict[str, str] = {}

    def runner(_command, **kwargs):
        captured_environment.update(kwargs["env"])
        return _FakeProcess()

    attempt_directory = tmp_path / "attempts"
    assert launch_modal.run(
        _arguments(run_id="no-containment-secret-leak"),
        runner=runner,
        receipt_directory=attempt_directory,
        machine_identity_provider=lambda: machine_secret,
        boot_session_provider=lambda: _TEST_BOOT_STARTED_AT_UNIX_MICROSECONDS,
        boot_identity_provider=lambda: boot_secret,
        process_birth_identity_provider=lambda _pid: process_birth_secret,
        attempt_id_factory=lambda _size: "d" * 32,
        host_anchor_id_factory=lambda _size: "e" * 64,
        process_group_capture=lambda child: child.pid,
        process_group_terminator=lambda *_args, **_kwargs: None,
    ) == 0
    assert token_secret not in captured_environment.values()
    assert not any(
        "host_anchor" in key or "boot" in key
        for key in captured_environment
    )
    serialized = b"".join(
        path.read_bytes()
        for path in attempt_directory.rglob("*")
        if path.is_file()
    )
    assert machine_secret not in serialized
    assert boot_secret not in serialized
    assert process_birth_secret not in serialized
    assert token_secret.encode() not in serialized
    terminal = json.loads(
        _terminal_receipt_path(attempt_directory).read_text(encoding="utf-8")
    )
    intent = json.loads(_intent_path(attempt_directory).read_text(encoding="utf-8"))
    reservation = json.loads(
        next((attempt_directory / "remote_run_reservations").glob("*.json")).read_text()
    )
    marker_path = launch_modal._local_containment_destination(
        terminal["local_process_start_receipt_path"],
        project_root=ROOT,
        receipt_directory=attempt_directory,
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert terminal["schema_version"] == "3.6"
    assert intent["schema_version"] == "1.6"
    assert reservation["schema_version"] == "1.2"
    assert marker["schema_name"] == "ModalLocalProcessStart"
    assert marker["schema_version"] == "1.1"
    assert len(marker["process_birth_identity_sha256"]) == 64
    for field in (
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
    ):
        assert terminal[field] == intent[field] == reservation[field] == marker[field]
