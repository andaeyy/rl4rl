from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from types import SimpleNamespace

import modal_boundary
import pytest
import scripts.launch_modal as launch_modal
import scripts.record_local_engineering_evidence as local_evidence
from study.serialization import content_hash


def _write(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_bytes(payload)
    path.chmod(0o600)


def _write_json(path: Path, payload: object) -> None:
    _write(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _configure_small_identity_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    project_root = tmp_path / "workspace" / "architecture_discovery"
    monkeypatch.setattr(
        local_evidence,
        "_WORKSPACE_SOURCE_ROOT_FILES",
        ("Makefile", "pyproject.toml", "uv.lock"),
    )
    monkeypatch.setattr(
        local_evidence,
        "_WORKSPACE_SOURCE_DIRECTORIES",
        ("src", "tests", "configs", "schemas", "data"),
    )
    monkeypatch.setattr(
        local_evidence,
        "_SOURCE_ROOT_FILES",
        (
            "experiment_manifest.yaml",
            "pyproject.toml",
            "uv.lock",
            "vendor/starting_model/checkpoints/best.pt",
        ),
    )
    monkeypatch.setattr(
        local_evidence,
        "_SOURCE_DIRECTORIES",
        ("common", "private_eval", "scripts", "tests"),
    )
    monkeypatch.setattr(
        local_evidence,
        "_VALIDATION_INPUT_FILES",
        ("README.md", "MODAL_MIGRATION_NOTES.md", "readiness_evidence.yaml"),
    )
    workspace_files = {
        "Makefile": "check:\n\t@true\n",
        "pyproject.toml": "[project]\nname='fixture'\nversion='0'\n",
        "uv.lock": "version = 1\n",
        "src/pkg.py": "VALUE = 1\n",
        "tests/test_pkg.py": "def test_value():\n    assert True\n",
        "configs/taxonomy.toml": "[taxonomy]\n",
        "schemas/event.json": "{}\n",
        "data/example.jsonl": "{}\n",
    }
    architecture_files = {
        "experiment_manifest.yaml": "schema_version: '3'\n",
        "pyproject.toml": "[project]\nname='architecture-fixture'\n",
        "uv.lock": "version = 1\n",
        "vendor/starting_model/checkpoints/best.pt": b"checkpoint-v1\n",
        "common/runtime.py": "VALUE = 1\n",
        "private_eval/regression.py": "PRIVATE = True\n",
        "scripts/study_offline_smoke.py": "# bound script\n",
        "tests/test_runtime.py": "def test_runtime():\n    assert True\n",
        "README.md": "readiness v1\n",
        "MODAL_MIGRATION_NOTES.md": "migration v1\n",
        "readiness_evidence.yaml": "status: pending\n",
    }
    for relative, payload in workspace_files.items():
        _write(project_root.parent / relative, payload)
    for relative, payload in architecture_files.items():
        _write(project_root / relative, payload)
    monkeypatch.setattr(local_evidence, "_git_revision", lambda _root: "a" * 40)
    monkeypatch.setattr(
        local_evidence,
        "execution_environment_manifest",
        lambda _root: {
            **local_evidence.EXECUTION_ENVIRONMENT_MANIFEST_CONTRACT,
            "environment": {"PATH": "/usr/bin:/bin"},
            "python": {"sha256": "e" * 64, "version": "fixture"},
            "tools": {},
            "dependencies": {},
            "modal_cli": {
                "distribution_version": modal_boundary.MODAL_VERSION,
                "sha256": "f" * 64,
            },
            "openevolve_installation": {"installed_matches_vendor": True},
            "workspace_python": None,
        },
    )
    return project_root


def _phase2_stdout(spec: object) -> str:
    kind = spec.result_kind
    if kind == "pytest":
        return "1 passed in 0.01s\n"
    if kind == "ruff":
        return "All checks passed!\n"
    if kind == "configuration":
        return "configuration invariants: PASS\n"
    if kind == "environment":
        return json.dumps(
            {
                "scientific_cpu_fallback": False,
                "credentials": {
                    "DISCOVERY_API_KEY": False,
                    "DISCOVERY_API_BASE": False,
                    "DISCOVERY_MODEL": False,
                },
            }
        )
    if kind == "four_controller_static":
        return json.dumps(
            {
                "static_controller_surfaces_passed": True,
                "provider_calls": 0,
                "training_runs": 0,
                "entrypoint_execution_runs": 0,
                "static_controller_surfaces": {"harnesses": [{}, {}, {}, {}]},
            }
        )
    if kind == "modal_plan":
        return json.dumps(
            {
                "schema_name": "ModalExecutionPlan",
                "remote_calls_started": 0,
                "functions": {
                    "fixture": {
                        "max_containers": 1,
                        "min_containers": 0,
                        "retries": 0,
                    }
                },
            }
        )
    return ""


def _phase2_record(spec: object, project_root: Path) -> dict[str, object]:
    stdout = _phase2_stdout(spec)
    stderr = ""
    environment = local_evidence._phase2_execution_environment(
        spec,
        project_root=project_root,
    )
    resolved_command = local_evidence._phase2_execution_command(
        spec,
        project_root=project_root,
        environment=environment,
    )
    record: dict[str, object] = {
        "component_id": spec.component_id,
        "command": list(spec.command),
        "resolved_command": resolved_command,
        "cwd": spec.cwd,
        "environment_overrides": dict(spec.environment_overrides),
        "execution_environment": environment,
        "timeout_seconds": spec.timeout_seconds,
        "returncode": 0,
        "stdout": stdout,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stdout_bytes": len(stdout.encode()),
        "stderr": stderr,
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "stderr_bytes": 0,
        "checks_completed": 0,
        "passed": True,
    }
    record["checks_completed"] = local_evidence._phase2_component_checks(spec, record)
    return record


def _run_real_offline_fixture(
    project_root: Path,
    validation_identity_sha256: str,
) -> bytes:
    artifact_root = local_evidence._offline_artifact_root(
        project_root,
        validation_identity_sha256,
    )
    study_id = f"readiness-offline-smoke-{validation_identity_sha256[:16]}"
    completed = subprocess.run(
        [
            sys.executable,
            str(local_evidence.ROOT / "scripts" / "study_offline_smoke.py"),
            "--output-dir",
            str(artifact_root),
            "--study-id",
            study_id,
            "--study-seed",
            "7",
            "--blocks",
            "1",
            "--opportunities",
            "3",
        ],
        cwd=local_evidence.ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert completed.stderr == b""
    return completed.stdout


def _create_complete_freeze_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, local_evidence._ValidationIdentity, str]:
    project_root = _configure_small_identity_fixture(tmp_path, monkeypatch)
    image_digest = "c" * 64
    monkeypatch.setattr(
        local_evidence,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256=image_digest),
    )
    monkeypatch.setattr(
        local_evidence,
        "_phase2_image_metrics",
        lambda _root: {
            "dependency_lock_sha256": "d" * 64,
            "image_source_file_count": 3,
            "image_source_total_bytes": 100,
            "image_source_two_copy_upper_bound_bytes": 200,
        },
    )
    monkeypatch.setattr(
        local_evidence,
        "_phase2_execution_environment",
        lambda spec, *, project_root: {
            "PATH": "/usr/bin:/bin",
            **dict(spec.environment_overrides),
        },
    )
    monkeypatch.setattr(
        local_evidence,
        "_phase2_execution_command",
        lambda spec, *, project_root, environment=None: [
            f"/fixture-tools/{spec.component_id}",
            *spec.command[1:],
        ],
    )
    identity = local_evidence._current_validation_identity(project_root)
    local_evidence._ensure_identity_manifests(
        project_root,
        identity,
        create_missing=True,
    )
    validation_digest = identity.validation_identity_sha256
    environment = local_evidence._minimal_execution_environment(project_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment = dict(sorted(environment.items()))

    unit_result_logical = local_evidence.local_engineering_result_path(
        "unit_tested",
        validation_identity_digest=validation_digest,
    )
    unit_result = b"10 passed in 0.10s\n"
    _write(project_root / unit_result_logical, unit_result)
    unit_command = local_evidence.frozen_local_engineering_command(
        "unit_tested",
        validation_identity_digest=validation_digest,
    )
    unit_resolved = [os.path.abspath(sys.executable), *unit_command[1:]]
    unit_payload = {
        **local_evidence.LOCAL_ENGINEERING_RECEIPT_CONTRACTS["unit_tested"][
            "receipt_contract"
        ],
        **local_evidence._identity_receipt_fields(identity),
        "recorded_at_utc": "2026-08-09T00:00:00Z",
        "source_revision": identity.source_revision,
        "source_tree_sha256": identity.source_manifest_sha256,
        "source_tree_sha256_before_command": identity.source_manifest_sha256,
        "source_tree_sha256_after_command": identity.source_manifest_sha256,
        "image_source_sha256": image_digest,
        "command": unit_command,
        "command_sha256": content_hash(unit_command),
        "resolved_command": unit_resolved,
        "resolved_command_sha256": content_hash(unit_resolved),
        "execution_environment": environment,
        "execution_environment_sha256": content_hash(environment),
        "result_path": unit_result_logical.as_posix(),
        "result_sha256": hashlib.sha256(unit_result).hexdigest(),
        "artifact_manifest_path": None,
        "artifact_manifest_sha256": None,
        "checks_completed": 10,
        **local_evidence._action_accounting(),
    }
    unit_receipt_logical = local_evidence.local_engineering_receipt_path(
        "unit_tested",
        validation_identity_digest=validation_digest,
    )
    _write_json(project_root / unit_receipt_logical, unit_payload)

    offline_stdout = _run_real_offline_fixture(project_root, validation_digest)
    offline_result_logical = local_evidence.local_engineering_result_path(
        "offline_smoke_tested",
        validation_identity_digest=validation_digest,
    )
    _write(project_root / offline_result_logical, offline_stdout)
    artifact_manifest, checks = local_evidence._validate_offline_artifact_tree(
        project_root,
        validation_digest,
        offline_stdout,
    )
    artifact_manifest_logical = (
        local_evidence.local_engineering_freeze_directory(validation_digest)
        / local_evidence.OFFLINE_ARTIFACT_MANIFEST_FILENAME
    )
    _write_json(project_root / artifact_manifest_logical, artifact_manifest)
    offline_command = local_evidence.frozen_local_engineering_command(
        "offline_smoke_tested",
        validation_identity_digest=validation_digest,
    )
    offline_resolved = [os.path.abspath(sys.executable), *offline_command[1:]]
    offline_payload = {
        **local_evidence.LOCAL_ENGINEERING_RECEIPT_CONTRACTS[
            "offline_smoke_tested"
        ]["receipt_contract"],
        **local_evidence._identity_receipt_fields(identity),
        "recorded_at_utc": "2026-08-09T00:00:01Z",
        "source_revision": identity.source_revision,
        "source_tree_sha256": identity.source_manifest_sha256,
        "source_tree_sha256_before_command": identity.source_manifest_sha256,
        "source_tree_sha256_after_command": identity.source_manifest_sha256,
        "image_source_sha256": image_digest,
        "command": offline_command,
        "command_sha256": content_hash(offline_command),
        "resolved_command": offline_resolved,
        "resolved_command_sha256": content_hash(offline_resolved),
        "execution_environment": environment,
        "execution_environment_sha256": content_hash(environment),
        "result_path": offline_result_logical.as_posix(),
        "result_sha256": hashlib.sha256(offline_stdout).hexdigest(),
        "artifact_manifest_path": artifact_manifest_logical.as_posix(),
        "artifact_manifest_sha256": content_hash(artifact_manifest),
        "checks_completed": checks,
        **local_evidence._action_accounting(),
    }
    offline_receipt_logical = local_evidence.local_engineering_receipt_path(
        "offline_smoke_tested",
        validation_identity_digest=validation_digest,
    )
    _write_json(project_root / offline_receipt_logical, offline_payload)

    specs = local_evidence._phase2_command_specs()
    phase2_payload = {
        **local_evidence.LOCAL_PHASE2_VALIDATION_RECEIPT_CONTRACT,
        **local_evidence._identity_receipt_fields(identity),
        "recorded_at_utc": "2026-08-09T00:00:02Z",
        "source_revision": identity.source_revision,
        "source_tree_sha256": identity.source_manifest_sha256,
        "source_tree_sha256_before_commands": identity.source_manifest_sha256,
        "source_tree_sha256_after_commands": identity.source_manifest_sha256,
        "image_source_sha256": image_digest,
        "dependency_lock_sha256": "d" * 64,
        "image_source_file_count": 3,
        "image_source_total_bytes": 100,
        "image_source_two_copy_upper_bound_bytes": 200,
        "mandatory_component_ids": list(
            local_evidence.MANDATORY_PHASE2_VALIDATION_COMPONENTS
        ),
        "component_receipt_coverage": dict(
            local_evidence._PHASE2_COMPONENT_RECEIPT_COVERAGE
        ),
        "executed_components": [
            _phase2_record(spec, project_root) for spec in specs
        ],
        **local_evidence._action_accounting(),
    }
    phase2_logical = local_evidence.local_phase2_validation_receipt_path(
        validation_digest
    )
    _write_json(project_root / phase2_logical, phase2_payload)

    component_paths = (
        ("unit_tested", unit_receipt_logical),
        ("offline_smoke_tested", offline_receipt_logical),
        ("phase2_validated", phase2_logical),
    )
    component_receipts = [
        {
            "level_name": level,
            "path": logical.as_posix(),
            "sha256": hashlib.sha256((project_root / logical).read_bytes()).hexdigest(),
        }
        for level, logical in component_paths
    ]
    aggregate_payload = {
        **local_evidence.LOCAL_ENGINEERING_FREEZE_RECEIPT_CONTRACT,
        **local_evidence._identity_receipt_fields(identity),
        "recorded_at_utc": "2026-08-09T00:00:03Z",
        "source_revision": identity.source_revision,
        "source_tree_sha256": identity.source_manifest_sha256,
        "image_source_sha256": image_digest,
        "component_receipts": component_receipts,
        **local_evidence._action_accounting(),
    }
    aggregate_logical = local_evidence.local_engineering_freeze_receipt_path(
        validation_digest
    )
    _write_json(project_root / aggregate_logical, aggregate_payload)
    return project_root, identity, image_digest


def test_historical_freeze_allows_only_validation_document_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, _identity, image_digest = _create_complete_freeze_fixture(
        tmp_path,
        monkeypatch,
    )
    bindings = local_evidence.local_engineering_freeze_predecessor_bindings(
        root=project_root,
        expected_image_source_sha256=image_digest,
    )
    for relative in (
        "README.md",
        "MODAL_MIGRATION_NOTES.md",
        "readiness_evidence.yaml",
    ):
        path = project_root / relative
        _write(path, path.read_bytes() + b"historical accounting update\n")

    assert (
        local_evidence.historical_local_engineering_freeze_predecessor_bindings(
            bindings,
            root=project_root,
            expected_image_source_sha256=image_digest,
        )
        == bindings
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        local_evidence.local_engineering_freeze_predecessor_bindings(
            root=project_root,
            expected_image_source_sha256=image_digest,
        )


def test_historical_freeze_rejects_execution_git_environment_and_image_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, _identity, image_digest = _create_complete_freeze_fixture(
        tmp_path,
        monkeypatch,
    )
    bindings = local_evidence.local_engineering_freeze_predecessor_bindings(
        root=project_root,
        expected_image_source_sha256=image_digest,
    )

    source_path = project_root / "common/runtime.py"
    source_bytes = source_path.read_bytes()
    _write(source_path, source_bytes + b"SOURCE_DRIFT = True\n")
    with pytest.raises(ValueError, match="execution source"):
        local_evidence.historical_local_engineering_freeze_predecessor_bindings(
            bindings,
            root=project_root,
            expected_image_source_sha256=image_digest,
        )
    _write(source_path, source_bytes)

    monkeypatch.setattr(local_evidence, "_git_revision", lambda _root: "b" * 40)
    with pytest.raises(ValueError, match="Git revision"):
        local_evidence.historical_local_engineering_freeze_predecessor_bindings(
            bindings,
            root=project_root,
            expected_image_source_sha256=image_digest,
        )
    monkeypatch.setattr(local_evidence, "_git_revision", lambda _root: "a" * 40)

    original_environment = local_evidence.execution_environment_manifest(
        project_root
    )
    monkeypatch.setattr(
        local_evidence,
        "execution_environment_manifest",
        lambda _root: {**original_environment, "drift": True},
    )
    with pytest.raises(ValueError, match="environment changed"):
        local_evidence.historical_local_engineering_freeze_predecessor_bindings(
            bindings,
            root=project_root,
            expected_image_source_sha256=image_digest,
        )
    monkeypatch.setattr(
        local_evidence,
        "execution_environment_manifest",
        lambda _root: original_environment,
    )

    monkeypatch.setattr(
        local_evidence,
        "build_image_source_manifest",
        lambda _root: SimpleNamespace(manifest_sha256="e" * 64),
    )
    with pytest.raises(ValueError, match="image source"):
        local_evidence.historical_local_engineering_freeze_predecessor_bindings(
            bindings,
            root=project_root,
            expected_image_source_sha256=image_digest,
        )


def test_real_execution_source_manifest_covers_all_consumed_roots() -> None:
    manifest = local_evidence.execution_source_manifest(local_evidence.ROOT)
    paths = {record["relative_path"] for record in manifest["files"]}
    assert {
        "workspace/Makefile",
        "workspace/pyproject.toml",
        "workspace/uv.lock",
        "workspace/src/rl4rl/cli.py",
        "workspace/tests/test_metrics.py",
        "workspace/configs/taxonomy.toml",
        "workspace/schemas/trajectory-event.schema.json",
        "workspace/data/examples/synthetic_trajectory.jsonl",
        "architecture_discovery/private_eval/regression.py",
        "architecture_discovery/tests/test_local_engineering_evidence.py",
        "architecture_discovery/vendor/starting_model/checkpoints/best.pt",
        "architecture_discovery/vendor/starting_model/src/model.py",
        "architecture_discovery/vendor/openevolve/openevolve/api.py",
    } <= paths
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["total_bytes"] == sum(
        record["size_bytes"] for record in manifest["files"]
    )
    assert [record["relative_path"] for record in manifest["files"]] == sorted(paths)


@pytest.mark.parametrize(
    "relative",
    (
        "../Makefile",
        "../pyproject.toml",
        "../uv.lock",
        "vendor/starting_model/checkpoints/best.pt",
        "private_eval/regression.py",
    ),
)
def test_root_checkpoint_and_private_inputs_rotate_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = _configure_small_identity_fixture(tmp_path, monkeypatch)
    before = local_evidence.source_tree_sha256(root)
    target = root / relative
    _write(target, target.read_bytes() + b"changed\n")
    assert local_evidence.source_tree_sha256(root) != before


@pytest.mark.parametrize(
    "relative",
    ("README.md", "MODAL_MIGRATION_NOTES.md", "readiness_evidence.yaml"),
)
def test_dynamic_documents_rotate_validation_identity_not_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    root = _configure_small_identity_fixture(tmp_path, monkeypatch)
    source_before = local_evidence.source_tree_sha256(root)
    identity_before = local_evidence._current_validation_identity(root)
    _write(root / relative, (root / relative).read_bytes() + b"live evidence\n")
    identity_after = local_evidence._current_validation_identity(root)
    assert local_evidence.source_tree_sha256(root) == source_before
    assert identity_after.validation_input_manifest_sha256 != (
        identity_before.validation_input_manifest_sha256
    )
    assert identity_after.validation_identity_sha256 != (
        identity_before.validation_identity_sha256
    )


def test_namespace_rotates_for_head_and_execution_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _configure_small_identity_fixture(tmp_path, monkeypatch)
    revision = ["a" * 40]
    environment_tag = ["first"]
    monkeypatch.setattr(local_evidence, "_git_revision", lambda _root: revision[0])
    monkeypatch.setattr(
        local_evidence,
        "execution_environment_manifest",
        lambda _root: {
            **local_evidence.EXECUTION_ENVIRONMENT_MANIFEST_CONTRACT,
            "tag": environment_tag[0],
        },
    )
    first = local_evidence._current_validation_identity(root)
    revision[0] = "b" * 40
    second = local_evidence._current_validation_identity(root)
    environment_tag[0] = "second"
    third = local_evidence._current_validation_identity(root)
    assert len({first.validation_identity_sha256, second.validation_identity_sha256, third.validation_identity_sha256}) == 3
    assert local_evidence.local_engineering_freeze_directory(
        first.validation_identity_sha256
    ) != local_evidence.local_engineering_freeze_directory(
        second.validation_identity_sha256
    )


def test_complete_manifests_are_persisted_and_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _configure_small_identity_fixture(tmp_path, monkeypatch)
    identity = local_evidence._current_validation_identity(root)
    local_evidence._ensure_identity_manifests(root, identity, create_missing=True)
    directory = root / local_evidence.local_engineering_freeze_directory(
        identity.validation_identity_sha256
    )
    assert json.loads(
        (directory / local_evidence.EXECUTION_SOURCE_MANIFEST_FILENAME).read_text()
    ) == identity.source_manifest
    assert json.loads(
        (directory / local_evidence.VALIDATION_INPUT_MANIFEST_FILENAME).read_text()
    ) == identity.validation_input_manifest
    assert json.loads(
        (directory / local_evidence.EXECUTION_ENVIRONMENT_MANIFEST_FILENAME).read_text()
    ) == identity.execution_environment_manifest
    for filename, _, _ in local_evidence._identity_manifest_specs(identity):
        assert stat.S_IMODE((directory / filename).stat().st_mode) == 0o600
    local_evidence._ensure_identity_manifests(root, identity, create_missing=False)


def test_manifest_scanner_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    _write(root / "source.py", "VALUE = 1\n")
    (root / "linked.py").symlink_to(root / "source.py")
    with pytest.raises(ValueError, match="symbolic link"):
        local_evidence._scan_manifest_scope(
            root,
            namespace="fixture",
            include_complete_root=True,
            max_file_bytes=1024,
            max_files=10,
            max_total_bytes=4096,
        )
    (root / "linked.py").unlink()
    os.link(root / "source.py", root / "hardlink.py")
    with pytest.raises(ValueError, match="exactly one hard link"):
        local_evidence._scan_manifest_scope(
            root,
            namespace="fixture",
            include_complete_root=True,
            max_file_bytes=1024,
            max_files=10,
            max_total_bytes=4096,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"max_file_bytes": 3, "max_files": 10, "max_total_bytes": 100}, "per-file"),
        ({"max_file_bytes": 100, "max_files": 1, "max_total_bytes": 100}, "file-count"),
        ({"max_file_bytes": 100, "max_files": 10, "max_total_bytes": 5}, "total-byte"),
    ),
)
def test_manifest_scanner_enforces_all_caps(
    tmp_path: Path,
    kwargs: dict[str, int],
    match: str,
) -> None:
    root = tmp_path / "tree"
    _write(root / "a.py", "1234")
    _write(root / "b.py", "5678")
    with pytest.raises(ValueError, match=match):
        local_evidence._scan_manifest_scope(
            root,
            namespace="fixture",
            include_complete_root=True,
            **kwargs,
        )


def test_manifest_scanner_detects_leaf_swap_between_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    target = root / "source.py"
    _write(target, "VALUE = 1\n")
    original = local_evidence._scan_manifest_scope_once
    calls = 0

    def swap_after_first(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            _write(target, "VALUE = 2\n")
        return result

    monkeypatch.setattr(local_evidence, "_scan_manifest_scope_once", swap_after_first)
    with pytest.raises(ValueError, match="changed while"):
        local_evidence._scan_manifest_scope(
            root,
            namespace="fixture",
            include_complete_root=True,
            max_file_bytes=1024,
            max_files=10,
            max_total_bytes=4096,
        )


def test_manifest_scanner_detects_same_byte_ancestor_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    source_dir = root / "source"
    _write(source_dir / "module.py", "VALUE = 1\n")
    original = local_evidence._scan_manifest_scope_once
    calls = 0

    def swap_directory_after_first(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            moved = root / "moved"
            source_dir.rename(moved)
            _write(source_dir / "module.py", "VALUE = 1\n")
        return result

    monkeypatch.setattr(
        local_evidence,
        "_scan_manifest_scope_once",
        swap_directory_after_first,
    )
    with pytest.raises(ValueError, match="changed while"):
        local_evidence._scan_manifest_scope(
            root,
            namespace="fixture",
            directories=("source",),
            max_file_bytes=1024,
            max_files=10,
            max_total_bytes=4096,
        )


def test_minimal_environment_strips_provider_and_python_pytest_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DISCOVERY_API_KEY",
        "MODAL_TOKEN_SECRET",
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONINSPECT",
    ):
        monkeypatch.setenv(name, "must-not-pass")
    environment = local_evidence._minimal_execution_environment(local_evidence.ROOT)
    assert not set(environment) & local_evidence._STRIPPED_EXECUTION_ENVIRONMENT
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert set(environment) <= {
        "PATH",
        "LANG",
        "LC_ALL",
        "NO_COLOR",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONUTF8",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "TMPDIR",
    }


def test_real_environment_manifest_binds_python_tools_dependencies_and_vendor() -> None:
    manifest = local_evidence.execution_environment_manifest(local_evidence.ROOT)
    assert manifest["python"]["invoked_path"] == os.path.abspath(sys.executable)
    assert len(manifest["python"]["sha256"]) == 64
    assert set(manifest["tools"]) == {"git", "make", "ruff", "uv"}
    assert {"modal", "openevolve", "pytest", "torch"} <= set(
        manifest["dependencies"]
    )
    modal_cli = manifest["modal_cli"]
    assert modal_cli["distribution_version"] == modal_boundary.MODAL_VERSION
    assert modal_cli["invoked_path"] == str(Path(sys.executable).with_name("modal"))
    assert modal_cli["symlink_target"] is None
    assert len(modal_cli["sha256"]) == 64
    assert modal_cli["inode"] > 0
    proof = manifest["openevolve_installation"]
    assert proof["installed_matches_vendor"] is True
    assert proof["file_count"] > 0
    assert len(proof["files_sha256"]) == 64


def test_openevolve_installation_proof_is_independent_of_import_search_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = local_evidence._openevolve_installation_proof(local_evidence.ROOT)
    monkeypatch.syspath_prepend(
        str((local_evidence.ROOT / "vendor" / "openevolve").resolve())
    )
    after = local_evidence._openevolve_installation_proof(local_evidence.ROOT)

    assert after == before
    assert after["installed_root"] == str(
        (Path(sysconfig.get_path("purelib")) / "openevolve").resolve()
    )


def test_streaming_executor_rejects_output_flood_before_accumulation() -> None:
    environment = local_evidence._minimal_execution_environment(local_evidence.ROOT)
    with pytest.raises(local_evidence._ProcessOutputLimitError, match="exceeded"):
        local_evidence._run_streaming_bounded_process(
            [sys.executable, "-c", "import sys; sys.stdout.write('x' * 10000)"],
            cwd=local_evidence.ROOT,
            environment=environment,
            timeout_seconds=5,
            max_output_bytes=128,
        )


def test_streaming_executor_times_out_and_closes_group() -> None:
    environment = local_evidence._minimal_execution_environment(local_evidence.ROOT)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        local_evidence._run_streaming_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=local_evidence.ROOT,
            environment=environment,
            timeout_seconds=0.1,
            max_output_bytes=1024,
        )
    assert time.monotonic() - started < 3


def test_streaming_executor_closes_background_descendant_after_success() -> None:
    environment = local_evidence._minimal_execution_environment(local_evidence.ROOT)
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(p.pid, flush=True)"
    )
    completed = local_evidence._run_streaming_bounded_process(
        [sys.executable, "-c", code],
        cwd=local_evidence.ROOT,
        environment=environment,
        timeout_seconds=5,
        max_output_bytes=1024,
    )
    descendant = int(completed.stdout.strip())
    assert completed.returncode == 0
    with pytest.raises(ProcessLookupError):
        os.kill(descendant, 0)


def test_phase2_component_records_resolved_argv_and_exact_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = next(
        item
        for item in local_evidence._phase2_command_specs()
        if item.component_id == "configuration_validation"
    )
    captured: dict[str, object] = {}

    def fake_run(command, *, cwd, environment, timeout_seconds, max_output_bytes):
        captured.update(
            command=command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        return subprocess.CompletedProcess(command, 0, b"configuration invariants: PASS\n", b"")

    monkeypatch.setattr(local_evidence, "_run_streaming_bounded_process", fake_run)
    record = local_evidence._execute_phase2_component(
        spec,
        project_root=local_evidence.ROOT,
    )
    assert record["resolved_command"] == captured["command"]
    assert record["execution_environment"] == captured["environment"]
    assert record["resolved_command"][0] == os.path.abspath(sys.executable)
    assert "PYTHONPATH" not in record["execution_environment"]
    local_evidence._validate_phase2_component_record(
        spec,
        record,
        project_root=local_evidence.ROOT,
    )


def test_real_offline_tree_validates_exact_stdout_run_ids_indexes_and_ledger(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    validation_digest = "1" * 64
    stdout = _run_real_offline_fixture(project_root, validation_digest)
    assert local_evidence._validate_offline_result_bytes(stdout) == 4
    manifest, checks = local_evidence._validate_offline_artifact_tree(
        project_root,
        validation_digest,
        stdout,
    )
    assert checks == 4
    assert manifest["file_count"] > 100
    assert manifest["total_bytes"] > 0


def test_offline_stdout_rejects_unknown_schema_field(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    validation_digest = "2" * 64
    stdout = _run_real_offline_fixture(project_root, validation_digest)
    payload = json.loads(stdout)
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="invalid exact schema"):
        local_evidence._validate_offline_result_bytes(json.dumps(payload).encode())


@pytest.mark.parametrize("mutation", ("summary", "index"))
def test_offline_tree_rejects_retained_summary_or_index_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    project_root = tmp_path / "project"
    validation_digest = "3" * 64
    stdout = _run_real_offline_fixture(project_root, validation_digest)
    study_id = f"readiness-offline-smoke-{validation_digest[:16]}"
    artifact_root = local_evidence._offline_artifact_root(
        project_root,
        validation_digest,
    )
    filename = (
        "offline_smoke_summary.json"
        if mutation == "summary"
        else "artifact_index_manifest.json"
    )
    target = artifact_root / study_id / filename
    payload = json.loads(target.read_text())
    payload["unexpected"] = True
    _write_json(target, payload)
    with pytest.raises(ValueError, match="offline artifact"):
        local_evidence._validate_offline_artifact_tree(
            project_root,
            validation_digest,
            stdout,
        )


def test_launch_validation_transitively_revalidates_dynamic_input_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, identity, image_digest = _create_complete_freeze_fixture(
        tmp_path,
        monkeypatch,
    )
    bindings = launch_modal.validate_local_freeze_evidence(
        project_root,
        expected_image_source_sha256=image_digest,
    )
    assert len(bindings) == 3
    assert all(identity.validation_identity_sha256 in item["path"] for item in bindings)
    _write(project_root / "readiness_evidence.yaml", "status: live-updated\n")
    with pytest.raises(ValueError, match="fresh current-source local engineering freeze"):
        launch_modal.validate_local_freeze_evidence(
            project_root,
            expected_image_source_sha256=image_digest,
        )


def test_local_counters_distinguish_remote_runs_from_fixture_training() -> None:
    accounting = local_evidence._action_accounting()
    assert accounting == {
        "provider_calls": 0,
        "remote_actions": 0,
        "remote_training_runs": 0,
        "scientific_runs": 0,
        "local_fixture_training_permitted": True,
        "scientific": False,
        "externally_attested": False,
        "passed": True,
    }


def test_readme_ruff_scope_includes_private_eval_and_modal_action_journal() -> None:
    readme = (local_evidence.ROOT / "README.md").read_text(encoding="utf-8")
    ruff_line = next(line for line in readme.splitlines() if "ruff check --isolated" in line)
    assert " private_eval " in f" {ruff_line} "
    assert " modal_action_journal.py " in f" {ruff_line} "
    assert "private_eval" in local_evidence._MIGRATION_RUFF_TARGETS
    assert "modal_action_journal.py" in local_evidence._SOURCE_ROOT_FILES
    assert "modal_action_journal.py" in local_evidence._MIGRATION_RUFF_TARGETS
    assert "modal_action_journal.py" in local_evidence._COMPILE_VALIDATION_TARGETS
    assert "tests/test_modal_action_journal.py" in local_evidence._MODAL_FOCUSED_TESTS
    assert "private_eval" not in modal_boundary.IMAGE_SOURCE_DIRECTORIES
