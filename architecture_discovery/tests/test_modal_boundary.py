from __future__ import annotations

import ast
import json
import os
import stat
from pathlib import Path

import pytest

import modal_boundary
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    APP_NAME,
    CANARY_ORDER,
    FUNCTION_CPU_REQUEST_CORES,
    FUNCTION_CPU_SOFT_LIMIT_CORES,
    FUNCTION_MEMORY_LIMIT_MIB,
    FUNCTION_MEMORY_REQUEST_MIB,
    FUNCTION_SPECS,
    GPU_TYPE,
    IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    IMAGE_SOURCE_DIRECTORIES,
    MAX_ARTIFACT_DOWNLOAD_FILE_BYTES,
    MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES,
    MODAL_VERSION,
    PYTHON_VERSION,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    ArtifactFileV1,
    ArtifactIntegrityError,
    ArtifactManifestV1,
    ArtifactVerificationV1,
    FunctionSpec,
    ModalBoundaryError,
    ModalLiveCohortIdentity,
    RawArtifactManifestV1,
    build_artifact_manifest,
    build_image_source_manifest,
    build_image_source_snapshot,
    build_provider_canary_aggregate_outcome_receipt,
    create_fresh_run_directory,
    download_artifacts,
    load_artifact_manifest,
    provider_canary_aggregate_outcome_receipt_path,
    resolve_existing_volume_run_directory,
    run_canaries_synchronously,
    safe_relative_path,
    stage_image_source,
    validate_artifact_download_bounds,
    validate_provider_canary_aggregate_outcome_receipt,
    verify_artifact_manifest,
    volume_artifact_uri,
    volume_object_path,
    volume_run_path,
    write_artifact_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def _modal_command_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "python_executable": tmp_path / "venv/bin/python",
        "project_root": tmp_path,
        "action": "offline-smoke",
        "run_id": "offline-run-1",
        "source_tree_sha256": "1" * 64,
        "cohort_id": "cohort-1",
        "image_source_sha256": "2" * 64,
        "provider_approved": False,
    }


@pytest.mark.parametrize(
    "override",
    (
        {
            "action": "checkpoint-resume",
            "run_id": "resume-run-1",
            "source_run_id": "resume-run-1",
        },
        {
            "action": "download",
            "run_id": "source-run-1",
            "verifier_run_id": "source-run-1",
        },
        {
            "action": "verify",
            "run_id": "source-run-1",
            "verifier_run_id": "source-run-1",
        },
        {
            "action": "canary",
            "run_id": "study-semantic-ar",
            "harness": "greedy_autoresearch",
            "provider_approved": True,
        },
    ),
)
def test_modal_command_rejects_cross_action_identity_aliases(
    tmp_path: Path,
    override: dict[str, object],
) -> None:
    kwargs = {**_modal_command_kwargs(tmp_path), **override}
    with pytest.raises(ValueError, match="must differ|harness-specific suffix"):
        modal_boundary.build_modal_cli_command(**kwargs)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"run_id": True}, "run_id"),
        ({"run_id": "../escape"}, "run_id"),
        (
            {
                "action": "checkpoint-resume",
                "source_run_id": True,
            },
            "run_id",
        ),
        (
            {
                "action": "canary",
                "run_id": "study-greedy-ar",
                "harness": True,
                "provider_approved": True,
            },
            "harness",
        ),
        ({"provider_approved": 1}, "boolean"),
    ),
)
def test_modal_command_rejects_bool_type_and_path_identity_spoofs(
    tmp_path: Path,
    override: dict[str, object],
    message: str,
) -> None:
    kwargs = {**_modal_command_kwargs(tmp_path), **override}
    with pytest.raises((TypeError, ValueError), match=message):
        modal_boundary.build_modal_cli_command(**kwargs)


def test_modal_command_accepts_distinct_and_suffix_bound_action_identities(
    tmp_path: Path,
) -> None:
    base = _modal_command_kwargs(tmp_path)
    commands = [
        modal_boundary.build_modal_cli_command(
            **{
                **base,
                "action": "checkpoint-resume",
                "run_id": "resume-destination-1",
                "source_run_id": "resume-source-1",
            }
        ),
        modal_boundary.build_modal_cli_command(
            **{
                **base,
                "action": "download",
                "run_id": "download-source-1",
                "verifier_run_id": "download-verifier-1",
            }
        ),
        modal_boundary.build_modal_cli_command(
            **{
                **base,
                "action": "canary",
                "run_id": "study-greedy-ar",
                "harness": "greedy_autoresearch",
                "provider_approved": True,
            }
        ),
    ]
    expected_prefix = (str(tmp_path / "venv/bin/modal"), "run")
    assert all(command[:2] == expected_prefix for command in commands)


def _raw_manifest(
    manifest: ArtifactManifestV1,
    *,
    filename: str = "artifact_manifest.json",
) -> RawArtifactManifestV1:
    raw_bytes = (
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=True,
            indent=3,
            sort_keys=False,
        )
        + "\n"
    ).encode("utf-8")
    return RawArtifactManifestV1.from_bytes(
        filename=filename,
        raw_bytes=raw_bytes,
    )


def test_modal_live_cohort_paths_bind_every_identity_component() -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="cuda-cohort-1",
    )
    attempt_id = "3" * 32
    root = modal_boundary.modal_live_cohort_root(identity)
    assert root.as_posix() == (
        "outputs/readiness/modal_only_final/modal_live_cohorts/"
        + "1" * 64
        + "/"
        + "2" * 64
        + "/cuda-cohort-1"
    )
    assert (
        modal_boundary.modal_action_intent_receipt_path(identity, attempt_id)
        == root / "action_attempts" / f"{attempt_id}.intent.json"
    )
    assert (
        modal_boundary.modal_action_terminal_receipt_path(identity, attempt_id)
        == root / "action_attempts" / f"{attempt_id}.json"
    )
    recovery_root = root / "action_recoveries"
    assert modal_boundary.modal_action_recovery_directory(identity) == recovery_root
    assert (
        modal_boundary.modal_action_recovery_intent_path(identity, attempt_id)
        == recovery_root / f"{attempt_id}.intent.v1.0.json"
    )
    assert (
        modal_boundary.modal_action_host_containment_path(identity, attempt_id)
        == recovery_root / f"{attempt_id}.host-containment.v1.0.json"
    )
    assert (
        modal_boundary.modal_action_recovery_resolution_path(identity, attempt_id)
        == recovery_root / f"{attempt_id}.resolution.v1.0.json"
    )
    assert modal_boundary.modal_migration_lineage_path(identity) == (
        root / "migration_lineage.v1.1.json"
    )
    assert modal_boundary.modal_remote_run_reservation_path(
        "destination-run-1"
    ).as_posix() == (
        "outputs/readiness/modal_only_final/modal_remote_run_reservations/"
        "destination-run-1.json"
    )
    verification_root = (
        root / "artifact_verifications" / "source-run-1" / "verifier-run-1" / attempt_id
    )
    assert (
        modal_boundary.modal_remote_verification_receipt_path(
            identity,
            "source-run-1",
            "verifier-run-1",
            attempt_id,
        )
        == verification_root / "remote_verification.json"
    )
    assert (
        modal_boundary.modal_artifact_verifier_capture_directory_path(
            identity,
            "source-run-1",
            "verifier-run-1",
            attempt_id,
        )
        == verification_root / "volume_capture" / "verifier-run-1"
    )
    assert modal_boundary.modal_launch_rejection_receipt_path(
        attempt_id
    ).as_posix() == (
        f"outputs/readiness/modal_only_final/modal_launch_rejections/{attempt_id}.json"
    )
    assert modal_boundary.modal_global_launch_rejection_seal_path().as_posix() == (
        "outputs/readiness/modal_only_final/modal_launch_rejections/seal.v1.json"
    )

    with pytest.raises(ValueError, match="source-tree"):
        ModalLiveCohortIdentity(
            source_tree_sha256="not-a-digest",
            image_source_sha256="2" * 64,
            cohort_id="cuda-cohort-1",
        )
    with pytest.raises(ValueError, match="attempt ID"):
        modal_boundary.modal_action_terminal_receipt_path(identity, "unsafe")
    with pytest.raises(ValueError, match="attempt ID"):
        modal_boundary.modal_action_recovery_intent_path(identity, "unsafe")
    with pytest.raises(ValueError, match="attempt ID"):
        modal_boundary.modal_action_host_containment_path(identity, "unsafe")
    with pytest.raises(ValueError, match="attempt ID"):
        modal_boundary.modal_action_recovery_resolution_path(identity, "unsafe")


def test_modal_local_containment_paths_are_private_and_attempt_unique() -> None:
    assert modal_boundary.modal_local_host_anchor_path().as_posix() == (
        "outputs/readiness/.modal_local_containment/host_anchor.json"
    )
    assert modal_boundary.modal_local_process_start_receipt_path(
        "a" * 32
    ).as_posix() == (
        f"outputs/readiness/.modal_local_containment/process_starts/{'a' * 32}.json"
    )
    with pytest.raises(ValueError, match="attempt ID"):
        modal_boundary.modal_local_process_start_receipt_path("not-an-attempt")


def test_function_specs_freeze_resource_deadline_and_secret_boundaries() -> None:
    assert PYTHON_VERSION == "3.12"
    assert MODAL_VERSION == "1.5.3"
    assert IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS == 600
    assert VOLUME_NAME == "rl4rl-architecture-artifacts"
    assert str(VOLUME_MOUNT_PATH) == "/mnt/discovery"
    provider_names = {
        name for name, spec in FUNCTION_SPECS.items() if spec.provider_secret
    }
    assert provider_names == {
        "canary_greedy_autoresearch",
        "canary_semantic_autoresearch",
        "canary_openevolve_generic",
        "canary_openevolve_semantic",
    }
    for spec in FUNCTION_SPECS.values():
        assert spec.max_containers == 1
        assert spec.min_containers == 0
        assert spec.retries == 0
        assert spec.timeout_seconds == 300
        assert spec.cpu_request_cores == FUNCTION_CPU_REQUEST_CORES == 2.0
        assert spec.cpu_soft_limit_cores == FUNCTION_CPU_SOFT_LIMIT_CORES == 2.0
        assert spec.memory_request_mib == FUNCTION_MEMORY_REQUEST_MIB == 8192
        assert spec.memory_limit_mib == FUNCTION_MEMORY_LIMIT_MIB == 8192
        assert spec.region is None
        assert spec.cpu_request_cores == spec.cpu_soft_limit_cores
        assert spec.memory_request_mib == spec.memory_limit_mib
        assert spec.volume_mount_path == "/mnt/discovery"
        if spec.name.startswith(("cuda_", "candidate_", "checkpoint_", "canary_")):
            assert spec.gpu == GPU_TYPE


@pytest.mark.parametrize(
    "override",
    (
        {"cpu_request_cores": 1.0},
        {"cpu_soft_limit_cores": 3.0},
        {"memory_request_mib": 4096},
        {"memory_limit_mib": 16384},
        {"region": "us-east"},
    ),
)
def test_function_spec_rejects_resource_request_or_limit_drift(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="frozen|base-rate"):
        FunctionSpec("resource-drift", None, False, **override)


def test_image_source_manifest_is_allowlisted_and_content_bound() -> None:
    first = build_image_source_manifest(ROOT)
    second = build_image_source_manifest(ROOT)
    assert first == second
    assert first.manifest_sha256 == second.manifest_sha256
    paths = {item.relative_path for item in first.files}
    assert {
        "pyproject.toml",
        "uv.lock",
        "modal_action_journal.py",
        "modal_app.py",
        "modal_boundary.py",
        "modal_image_build.py",
        "common/runtime_context.py",
        "analysis/power.py",
        "mechanism/plans.py",
        "novelty/corpus.py",
        "reconstruction/rebuild.py",
        "replication/clean_room.py",
        "reporting/report.py",
        "research_ledger/ledger.py",
        "review/adjudication.py",
        "sealed_eval/orchestration.py",
        "sealed_eval/qualification.py",
        "sealed_eval/snapshot.py",
        "scripts/modal_plan.py",
        "vendor/openevolve/pyproject.toml",
        "vendor_patches/openevolve_process_isolation.patch",
        "vendor_patches/openevolve_provider_attempt_ledger.patch",
    } <= paths
    forbidden = {
        ".git",
        ".venv",
        ".env",
        "outputs",
        "logs",
        "checkpoints",
        "private_eval",
        "tests",
        "custody",
    }
    for path in paths:
        lowered = set(Path(path).parts)
        assert lowered.isdisjoint(forbidden)
        assert "custody" not in path.lower()
        assert not Path(path).name.startswith(".env")
    assert first.dependency_lock_sha256 == next(
        item.sha256 for item in first.files if item.relative_path == "uv.lock"
    )
    assert {
        "analysis",
        "mechanism",
        "novelty",
        "reconstruction",
        "replication",
        "reporting",
        "research_ledger",
        "review",
        "sealed_eval",
        "vendor_patches",
    } <= set(IMAGE_SOURCE_DIRECTORIES)


def test_sealed_eval_image_source_is_code_only(tmp_path: Path) -> None:
    package = tmp_path / "sealed_eval"
    package.mkdir()
    module = package / "orchestration.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    synthetic_cases = package / "synthetic_cases.json"
    synthetic_cases.write_text('{"case": 1}\n', encoding="utf-8")

    assert modal_boundary._include_source_file(tmp_path, module) is True
    assert modal_boundary._include_source_file(tmp_path, synthetic_cases) is False


def test_image_manifest_derives_lock_identity_from_the_hashed_file(
    tmp_path, monkeypatch
) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(modal_boundary, "_ALLOWED_ROOT_FILES", ("uv.lock",))
    monkeypatch.setattr(modal_boundary, "IMAGE_SOURCE_DIRECTORIES", ())
    monkeypatch.setattr(
        modal_boundary,
        "sha256_file",
        lambda _path: pytest.fail("path-based second hash must not be used"),
    )

    manifest = modal_boundary.build_image_source_manifest(tmp_path)

    expected = __import__("hashlib").sha256(lock.read_bytes()).hexdigest()
    assert manifest.dependency_lock_sha256 == expected
    assert manifest.files[0].sha256 == manifest.dependency_lock_sha256


def test_image_source_rejects_credential_names_content_and_size(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        modal_boundary,
        "_ALLOWED_ROOT_FILES",
        ("pyproject.toml", "uv.lock"),
    )
    monkeypatch.setattr(modal_boundary, "IMAGE_SOURCE_DIRECTORIES", ("scripts",))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    safe = scripts / "safe.py"
    safe.write_text("VALUE = 1\n")
    assert safe in modal_boundary.selected_image_source_paths(tmp_path)

    credential_name = scripts / "token.txt"
    credential_name.write_text("placeholder")
    with pytest.raises(ModalBoundaryError, match="credential-like filename"):
        modal_boundary.selected_image_source_paths(tmp_path)
    credential_name.unlink()

    leaked_key = scripts / "settings.txt"
    leaked_key.write_text("sk-" + "x" * 30)
    with pytest.raises(ModalBoundaryError, match="credential material"):
        modal_boundary.selected_image_source_paths(tmp_path)
    leaked_key.unlink()

    monkeypatch.setattr(modal_boundary, "MAX_IMAGE_SOURCE_FILE_BYTES", 4)
    with pytest.raises(ModalBoundaryError, match="per-file byte cap"):
        modal_boundary.selected_image_source_paths(tmp_path)

    monkeypatch.setattr(modal_boundary, "MAX_IMAGE_SOURCE_FILE_BYTES", 1024)
    monkeypatch.setattr(modal_boundary, "MAX_IMAGE_SOURCE_TOTAL_BYTES", 10)
    with pytest.raises(ModalBoundaryError, match="total byte cap"):
        modal_boundary.selected_image_source_paths(tmp_path)


@pytest.mark.parametrize(
    ("prefix", "alphabet"),
    (
        ("ghp_", "g"),
        ("github_pat_", "p"),
        ("hf_", "h"),
        ("ak-", "a"),
        ("as-", "s"),
        ("tinker_", "t"),
        ("tml-", "m"),
    ),
)
def test_image_source_rejects_non_openai_service_token_shapes(
    tmp_path,
    monkeypatch,
    prefix: str,
    alphabet: str,
) -> None:
    monkeypatch.setattr(
        modal_boundary,
        "_ALLOWED_ROOT_FILES",
        ("pyproject.toml", "uv.lock"),
    )
    monkeypatch.setattr(modal_boundary, "IMAGE_SOURCE_DIRECTORIES", ("scripts",))
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "settings.py").write_text(prefix + alphabet * 30)

    with pytest.raises(ModalBoundaryError, match="credential material"):
        modal_boundary.selected_image_source_paths(tmp_path)


def test_image_upload_snapshot_is_bound_before_checkout_can_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"VALUE = 1\n"
    (source / "module.py").write_bytes(payload)
    (source / "uv.lock").write_bytes(b"version = 1\n")
    monkeypatch.setattr(
        modal_boundary,
        "_ALLOWED_ROOT_FILES",
        ("module.py", "uv.lock"),
    )
    monkeypatch.setattr(modal_boundary, "IMAGE_SOURCE_DIRECTORIES", ())
    snapshot = build_image_source_snapshot(source)
    manifest = snapshot.manifest

    staged = stage_image_source(
        source,
        tmp_path / "staged",
        manifest,
        snapshot=snapshot,
    )
    (source / "module.py").write_bytes(b"VALUE = 2\n")

    assert (staged / "module.py").read_bytes() == payload
    staged_again = stage_image_source(
        source,
        tmp_path / "staged-from-snapshot",
        manifest,
        snapshot=snapshot,
    )
    assert (staged_again / "module.py").read_bytes() == payload
    with pytest.raises(ModalBoundaryError, match="changed after approval"):
        stage_image_source(source, tmp_path / "rejected", manifest)


def _minimal_image_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: bytes = b"VALUE = 1\n",
) -> tuple[Path, Path]:
    root = tmp_path / "image-source"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "uv.lock").write_bytes(b"version = 1\n")
    source = scripts / "safe.py"
    source.write_bytes(payload)
    monkeypatch.setattr(modal_boundary, "_ALLOWED_ROOT_FILES", ("uv.lock",))
    monkeypatch.setattr(modal_boundary, "IMAGE_SOURCE_DIRECTORIES", ("scripts",))
    return root, source


def test_image_snapshot_stages_scanned_bytes_after_same_size_credential_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe = b"A" * 23
    credential = b"sk-" + b"x" * 20
    root, source = _minimal_image_source(
        tmp_path,
        monkeypatch,
        payload=safe,
    )
    snapshot = build_image_source_snapshot(root)

    source.write_bytes(credential)
    staged = stage_image_source(
        root,
        tmp_path / "staged-safe-snapshot",
        snapshot.manifest,
        snapshot=snapshot,
    )

    assert (staged / "scripts" / "safe.py").read_bytes() == safe
    with pytest.raises(ModalBoundaryError, match="credential material"):
        build_image_source_snapshot(root)


def test_image_snapshot_object_cannot_be_forged_with_unscanned_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _source = _minimal_image_source(tmp_path, monkeypatch)
    snapshot = build_image_source_snapshot(root)
    files = tuple(
        modal_boundary.ImageSourceSnapshotFile(
            item.relative_path,
            b"sk-" + b"x" * 20
            if item.relative_path == "scripts/safe.py"
            else item.payload,
        )
        for item in snapshot.files
    )
    forged_manifest = modal_boundary.ImageSourceManifestV1(
        dependency_lock_sha256=snapshot.manifest.dependency_lock_sha256,
        files=tuple(item.source_file for item in files),
    )

    with pytest.raises(ModalBoundaryError, match="credential material"):
        modal_boundary.ImageSourceSnapshot(
            manifest=forged_manifest,
            files=files,
        )


def test_image_snapshot_rejects_leaf_swap_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source = _minimal_image_source(tmp_path, monkeypatch)
    source_inode = source.stat().st_ino
    original_read = modal_boundary.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, count)
        if not swapped and os.fstat(descriptor).st_ino == source_inode:
            swapped = True
            source.rename(source.with_suffix(".original"))
            source.write_bytes(b"VALUE = 1\n")
        return chunk

    monkeypatch.setattr(modal_boundary.os, "read", swapping_read)

    with pytest.raises(ModalBoundaryError, match="changed|identity"):
        build_image_source_snapshot(root)


def test_image_snapshot_rejects_ancestor_symlink_swap_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source = _minimal_image_source(tmp_path, monkeypatch)
    source_inode = source.stat().st_ino
    scripts = root / "scripts"
    original_scripts = root / "scripts-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "safe.py").write_bytes(b"VALUE = 1\n")
    original_read = modal_boundary.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, count)
        if not swapped and os.fstat(descriptor).st_ino == source_inode:
            swapped = True
            scripts.rename(original_scripts)
            scripts.symlink_to(outside, target_is_directory=True)
        return chunk

    monkeypatch.setattr(modal_boundary.os, "read", swapping_read)

    with pytest.raises(ModalBoundaryError, match="symlink|changed|identity"):
        build_image_source_snapshot(root)


def test_image_snapshot_rejects_metadata_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, source = _minimal_image_source(tmp_path, monkeypatch)
    source_inode = source.stat().st_ino
    original_read = modal_boundary.os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, count)
        if not mutated and os.fstat(descriptor).st_ino == source_inode:
            mutated = True
            source.write_bytes(b"VALUE = 2\n")
        return chunk

    monkeypatch.setattr(modal_boundary.os, "read", mutating_read)

    with pytest.raises(ModalBoundaryError, match="changed while reading"):
        build_image_source_snapshot(root)


@pytest.mark.parametrize("kind", ("fifo", "hardlink"))
def test_image_snapshot_rejects_nonregular_and_hardlinked_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    root, source = _minimal_image_source(tmp_path, monkeypatch)
    source.unlink()
    if kind == "fifo":
        os.mkfifo(source)
        expected = "not a regular"
    else:
        original = source.with_suffix(".original")
        original.write_bytes(b"VALUE = 1\n")
        os.link(original, source)
        expected = "link count"

    with pytest.raises(ModalBoundaryError, match=expected):
        build_image_source_snapshot(root)


def test_safe_run_and_volume_paths_refuse_reuse_or_traversal(tmp_path) -> None:
    created = create_fresh_run_directory(tmp_path, "modal-run-1")
    assert created == tmp_path.resolve() / "runs" / "modal-run-1"
    assert volume_run_path("modal-run-1").as_posix() == (
        "/mnt/discovery/runs/modal-run-1"
    )
    assert volume_object_path("modal-run-1", "nested/result.json") == (
        "/runs/modal-run-1/nested/result.json"
    )
    with pytest.raises(ModalBoundaryError, match="already exists"):
        create_fresh_run_directory(tmp_path, "modal-run-1")
    with pytest.raises(ValueError, match="run_id"):
        create_fresh_run_directory(tmp_path, "../escape")
    for invalid in ("../escape", "/absolute", "a//b", "a/./b", "a\\b"):
        with pytest.raises(ValueError):
            safe_relative_path(invalid)


def test_fresh_run_rejects_mount_root_symlink_without_explicit_trust(tmp_path) -> None:
    volume = tmp_path / "volume"
    volume.mkdir()
    mount = tmp_path / "mount-link"
    mount.symlink_to(volume, target_is_directory=True)

    with pytest.raises(ModalBoundaryError, match="without explicit trust"):
        create_fresh_run_directory(mount, "modal-run-2")
    assert not (volume / "runs").exists()


def test_fresh_run_accepts_and_pins_explicitly_trusted_mount_symlink(
    tmp_path,
) -> None:
    first_target = tmp_path / "first-volume"
    second_target = tmp_path / "second-volume"
    first_target.mkdir()
    second_target.mkdir()
    mount = tmp_path / "mount-link"
    mount.symlink_to(first_target, target_is_directory=True)

    created = create_fresh_run_directory(
        mount,
        "modal-run-2",
        allow_mount_root_symlink=True,
    )

    assert created == first_target.resolve() / "runs" / "modal-run-2"
    assert created.is_dir()
    mount.unlink()
    mount.symlink_to(second_target, target_is_directory=True)
    assert created.is_relative_to(first_target.resolve())
    assert not (second_target / "runs").exists()


def test_fresh_run_rejects_symlinks_below_canonical_mount_root(tmp_path) -> None:
    volume = tmp_path / "volume"
    outside = tmp_path / "outside"
    volume.mkdir()
    outside.mkdir()
    mount = tmp_path / "mount-link"
    mount.symlink_to(volume, target_is_directory=True)

    (volume / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ModalBoundaryError, match="runs directory may not be a symlink"):
        create_fresh_run_directory(
            mount,
            "modal-run-3",
            allow_mount_root_symlink=True,
        )
    assert not (outside / "modal-run-3").exists()

    (volume / "runs").unlink()
    (volume / "runs").mkdir()
    (volume / "runs" / "modal-run-3").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(ModalBoundaryError, match="already exists"):
        create_fresh_run_directory(
            mount,
            "modal-run-3",
            allow_mount_root_symlink=True,
        )


def test_fresh_run_no_follow_open_rejects_swapped_runs_symlink(
    tmp_path,
    monkeypatch,
) -> None:
    volume = tmp_path / "volume"
    outside = tmp_path / "outside"
    volume.mkdir()
    outside.mkdir()
    mount = tmp_path / "mount-link"
    mount.symlink_to(volume, target_is_directory=True)
    real_mkdir = modal_boundary.os.mkdir

    def swap_runs(path, mode=0o777, *, dir_fd=None):
        if path == "runs":
            (volume / "runs").symlink_to(outside, target_is_directory=True)
            raise FileExistsError
        return real_mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(modal_boundary.os, "mkdir", swap_runs)

    with pytest.raises(ModalBoundaryError, match="runs directory is unsafe"):
        create_fresh_run_directory(
            mount,
            "modal-run-race",
            allow_mount_root_symlink=True,
        )
    assert not (outside / "modal-run-race").exists()


def test_fresh_run_rejects_unsafe_mount_roots(tmp_path) -> None:
    missing = tmp_path / "missing"
    broken_mount = tmp_path / "broken-mount"
    broken_mount.symlink_to(missing, target_is_directory=True)
    with pytest.raises(ModalBoundaryError, match="missing or cannot be resolved"):
        create_fresh_run_directory(
            broken_mount,
            "modal-run-4",
            allow_mount_root_symlink=True,
        )

    traversing_mount = tmp_path / "child" / ".."
    with pytest.raises(ModalBoundaryError, match="traversal"):
        create_fresh_run_directory(traversing_mount, "modal-run-4")

    with pytest.raises(ModalBoundaryError, match="must be absolute"):
        create_fresh_run_directory(Path("relative-volume"), "modal-run-4")


def test_existing_run_accepts_and_pins_explicitly_trusted_mount_symlink(
    tmp_path: Path,
) -> None:
    first_target = tmp_path / "first-volume"
    second_target = tmp_path / "second-volume"
    source_run = first_target / "runs" / "modal-run-existing-1"
    source_run.mkdir(parents=True)
    second_target.mkdir()
    mount = tmp_path / "mount-link"
    mount.symlink_to(first_target, target_is_directory=True)

    with pytest.raises(ModalBoundaryError, match="without explicit trust"):
        resolve_existing_volume_run_directory(mount, "modal-run-existing-1")

    resolved = resolve_existing_volume_run_directory(
        mount,
        "modal-run-existing-1",
        allow_mount_root_symlink=True,
    )
    mount.unlink()
    mount.symlink_to(second_target, target_is_directory=True)

    assert resolved == source_run.resolve()
    assert resolved.is_relative_to(first_target.resolve())


@pytest.mark.parametrize("symlink_level", ("runs", "run"))
def test_existing_run_rejects_symlinks_below_trusted_mount(
    tmp_path: Path,
    symlink_level: str,
) -> None:
    volume = tmp_path / "volume"
    outside = tmp_path / "outside"
    volume.mkdir()
    outside.mkdir()
    mount = tmp_path / "mount-link"
    mount.symlink_to(volume, target_is_directory=True)
    if symlink_level == "runs":
        (volume / "runs").symlink_to(outside, target_is_directory=True)
        expected = "runs directory"
    else:
        (volume / "runs").mkdir()
        (volume / "runs" / "modal-run-existing-2").symlink_to(
            outside,
            target_is_directory=True,
        )
        expected = "run directory"

    with pytest.raises(ModalBoundaryError, match=expected):
        resolve_existing_volume_run_directory(
            mount,
            "modal-run-existing-2",
            allow_mount_root_symlink=True,
        )


def test_artifact_manifest_detects_tamper_and_symlinks(tmp_path) -> None:
    run = tmp_path / "run-1"
    (run / "nested").mkdir(parents=True)
    (run / "nested" / "result.json").write_text('{"ok": true}\n')
    manifest = build_artifact_manifest(
        run,
        run_id="run-1",
        image_source_sha256="a" * 64,
    )
    write_artifact_manifest(run, manifest)
    assert verify_artifact_manifest(run, manifest)["verified"] is True

    (run / "nested" / "result.json").write_text("tampered\n")
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        verify_artifact_manifest(run, manifest)

    other = tmp_path / "run-2"
    other.mkdir()
    (other / "link").symlink_to(run / "nested" / "result.json")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        build_artifact_manifest(
            other,
            run_id="run-2",
            image_source_sha256="b" * 64,
        )


def test_artifact_manifest_signs_tmp_and_manifest_prefixed_payloads(
    tmp_path,
) -> None:
    run = tmp_path / "run-exact-file-set"
    (run / "candidate").mkdir(parents=True)
    (run / "candidate" / "hidden.tmp").write_bytes(b"temporary-looking payload")
    (run / "artifact_manifest_payload.bin").write_bytes(b"manifest-looking payload")

    manifest = build_artifact_manifest(
        run,
        run_id=run.name,
        image_source_sha256="c" * 64,
    )
    paths = {item.relative_path for item in manifest.files}

    assert paths == {
        "artifact_manifest_payload.bin",
        "candidate/hidden.tmp",
    }
    write_artifact_manifest(run, manifest)
    assert verify_artifact_manifest(run, manifest)["verified"] is True


def test_artifact_build_rejects_descendant_directory_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "artifact-descendant-swap"
    nested = run / "nested"
    nested.mkdir(parents=True)
    (nested / "signed.bin").write_bytes(b"signed")
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (outside / "unsigned.bin").write_bytes(b"unsigned")
    displaced = run / "nested-displaced"
    real_open = modal_boundary.os.open
    swapped = False

    def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            path == "nested"
            and dir_fd is not None
            and flags & getattr(os, "O_DIRECTORY", 0)
            and not swapped
        ):
            swapped = True
            nested.rename(displaced)
            nested.symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(modal_boundary.os, "open", swap_before_open)

    with pytest.raises(ArtifactIntegrityError, match="changed or became unsafe"):
        build_artifact_manifest(
            run,
            run_id=run.name,
            image_source_sha256="d" * 64,
        )


def test_artifact_verify_rejects_leaf_target_swap_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "artifact-target-swap"
    run.mkdir()
    target = run / "result.bin"
    target.write_bytes(b"trusted" * (256 * 1024))
    manifest = build_artifact_manifest(
        run,
        run_id=run.name,
        image_source_sha256="e" * 64,
    )
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"untrusted")
    real_read = modal_boundary.os.read
    swapped = False

    def read_then_swap(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, size)
        if chunk and not swapped:
            swapped = True
            replacement.replace(target)
        return chunk

    monkeypatch.setattr(modal_boundary.os, "read", read_then_swap)

    with pytest.raises(ArtifactIntegrityError, match="changed while hashing"):
        verify_artifact_manifest(run, manifest)


def test_artifact_scan_rejects_hard_links_special_files_and_oversize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hard_link_run = tmp_path / "artifact-hard-link"
    hard_link_run.mkdir()
    source = hard_link_run / "source.bin"
    source.write_bytes(b"payload")
    os.link(source, hard_link_run / "alias.bin")
    with pytest.raises(ArtifactIntegrityError, match="link count is not one"):
        build_artifact_manifest(
            hard_link_run,
            run_id=hard_link_run.name,
            image_source_sha256="f" * 64,
        )

    special_run = tmp_path / "artifact-special-file"
    special_run.mkdir()
    os.mkfifo(special_run / "pipe")
    with pytest.raises(ArtifactIntegrityError, match="not a regular file or directory"):
        build_artifact_manifest(
            special_run,
            run_id=special_run.name,
            image_source_sha256="a" * 64,
        )

    oversized_run = tmp_path / "artifact-oversized-file"
    oversized_run.mkdir()
    (oversized_run / "large.bin").write_bytes(b"12345")
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_FILE_BYTES", 4)
    with pytest.raises(ArtifactIntegrityError, match="per-file byte cap"):
        build_artifact_manifest(
            oversized_run,
            run_id=oversized_run.name,
            image_source_sha256="b" * 64,
        )


def test_artifact_manifest_publication_rejects_competing_creator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "artifact-competing-creator"
    run.mkdir()
    (run / "payload.bin").write_bytes(b"payload")
    manifest = build_artifact_manifest(
        run,
        run_id=run.name,
        image_source_sha256="c" * 64,
    )
    competitor = b"competitor-owned\n"
    real_open = modal_boundary.os.open
    real_write = modal_boundary.os.write
    created = False

    def create_competitor(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal created
        if path == "artifact_manifest.json" and flags & os.O_CREAT and not created:
            created = True
            competitor_descriptor = real_open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
            try:
                assert real_write(competitor_descriptor, competitor) == len(competitor)
            finally:
                os.close(competitor_descriptor)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(modal_boundary.os, "open", create_competitor)

    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        write_artifact_manifest(run, manifest)
    assert (run / "artifact_manifest.json").read_bytes() == competitor


def test_artifact_manifest_publication_never_follows_existing_symlink(
    tmp_path: Path,
) -> None:
    run = tmp_path / "artifact-manifest-symlink"
    run.mkdir()
    (run / "payload.bin").write_bytes(b"payload")
    manifest = build_artifact_manifest(
        run,
        run_id=run.name,
        image_source_sha256="d" * 64,
    )
    outside = tmp_path / "outside-manifest.json"
    outside.write_bytes(b"outside-owned\n")
    destination = run / "artifact_manifest.json"
    destination.symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        write_artifact_manifest(run, manifest)
    assert destination.is_symlink()
    assert outside.read_bytes() == b"outside-owned\n"


def test_artifact_manifest_publication_handles_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "artifact-manifest-short-write"
    run.mkdir()
    (run / "payload.bin").write_bytes(b"payload")
    manifest = build_artifact_manifest(
        run,
        run_id=run.name,
        image_source_sha256="e" * 64,
    )
    real_write = modal_boundary.os.write
    write_count = 0

    def short_write(descriptor: int, payload) -> int:
        nonlocal write_count
        write_count += 1
        limit = max(1, len(payload) // 3)
        return real_write(descriptor, payload[:limit])

    monkeypatch.setattr(modal_boundary.os, "write", short_write)

    destination = write_artifact_manifest(run, manifest)
    assert write_count > 1
    assert load_artifact_manifest(destination) == manifest
    assert not list(run.glob(".artifact_manifest.*.tmp"))


def test_artifact_manifest_zero_progress_leaves_quarantined_final_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "artifact-manifest-zero-progress"
    run.mkdir()
    (run / "payload.bin").write_bytes(b"payload")
    manifest = build_artifact_manifest(
        run,
        run_id=run.name,
        image_source_sha256="f" * 64,
    )
    real_write = modal_boundary.os.write
    monkeypatch.setattr(modal_boundary.os, "write", lambda _fd, _payload: 0)

    with pytest.raises(ArtifactIntegrityError, match="no progress.*quarantined"):
        write_artifact_manifest(run, manifest)

    destination = run / "artifact_manifest.json"
    assert destination.is_file()
    assert not destination.is_symlink()
    assert destination.read_bytes() == b""

    monkeypatch.setattr(modal_boundary.os, "write", real_write)
    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        write_artifact_manifest(run, manifest)


def test_download_is_exclusive_and_hash_verified(tmp_path) -> None:
    payloads = {
        "/runs/download-1/a.txt": b"alpha",
        "/runs/download-1/nested/b.json": b'{"b": 2}\n',
    }
    files = tuple(
        ArtifactFileV1(
            relative_path=relative,
            sha256=__import__("hashlib").sha256(data).hexdigest(),
            size_bytes=len(data),
        )
        for relative, data in (
            ("a.txt", payloads["/runs/download-1/a.txt"]),
            ("nested/b.json", payloads["/runs/download-1/nested/b.json"]),
        )
    )
    manifest = ArtifactManifestV1(
        run_id="download-1",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="c" * 64,
        files=files,
    )
    raw_manifest = _raw_manifest(manifest)
    destination = download_artifacts(
        raw_manifest,
        local_root=tmp_path / "downloads",
        reader=lambda path: (payloads[path],),
    )
    assert (destination / "a.txt").read_bytes() == b"alpha"
    assert (destination / raw_manifest.filename).read_bytes() == (
        raw_manifest.raw_bytes
    )
    assert stat.S_IMODE((destination / "a.txt").stat().st_mode) == 0o600
    assert stat.S_IMODE((destination / "nested" / "b.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((destination / raw_manifest.filename).stat().st_mode) == 0o600
    assert not list((tmp_path / "downloads").glob(".download-1.download-*"))
    assert load_artifact_manifest(destination / "artifact_manifest.json") == manifest
    assert verify_artifact_manifest(destination, manifest)["verified"] is True
    with pytest.raises(ArtifactIntegrityError, match="already exists"):
        download_artifacts(
            raw_manifest,
            local_root=tmp_path / "downloads",
            reader=lambda path: (payloads[path],),
        )

    corrupted = ArtifactManifestV1(
        run_id="download-2",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="d" * 64,
        files=(
            ArtifactFileV1(
                relative_path="bad.bin",
                sha256="e" * 64,
                size_bytes=4,
            ),
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="failed verification"):
        download_artifacts(
            _raw_manifest(corrupted),
            local_root=tmp_path / "corrupt",
            reader=lambda _path: (b"nope",),
        )
    assert not (tmp_path / "corrupt" / "download-2").exists()
    assert not list((tmp_path / "corrupt").glob(".download-2.download-*"))


def test_download_preserves_checkpoint_manifest_name_and_exact_bytes(
    tmp_path: Path,
) -> None:
    manifest = ArtifactManifestV1(
        run_id="checkpoint-download-1",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="a" * 64,
        files=(),
    )
    raw_manifest = _raw_manifest(
        manifest,
        filename="artifact_manifest.checkpoint.json",
    )

    destination = download_artifacts(
        raw_manifest,
        local_root=tmp_path / "downloads",
        reader=lambda _path: b"",
    )

    assert not (destination / "artifact_manifest.json").exists()
    persisted = destination / "artifact_manifest.checkpoint.json"
    assert persisted.read_bytes() == raw_manifest.raw_bytes
    assert load_artifact_manifest(persisted) == manifest


def test_download_manifest_byte_caps_are_checked_before_transfer(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_FILE_BYTES", 4)
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES", 6)
    reader_calls: list[str] = []

    oversized_file = ArtifactManifestV1(
        run_id="download-file-cap",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="a" * 64,
        files=(
            ArtifactFileV1(
                relative_path="large.bin",
                sha256="b" * 64,
                size_bytes=5,
            ),
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="per-file download byte cap"):
        download_artifacts(
            _raw_manifest(oversized_file),
            local_root=tmp_path / "file-cap",
            reader=lambda path: reader_calls.append(path) or b"",
        )

    oversized_total = ArtifactManifestV1(
        run_id="download-total-cap",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="c" * 64,
        files=(
            ArtifactFileV1(
                relative_path="a.bin",
                sha256="d" * 64,
                size_bytes=4,
            ),
            ArtifactFileV1(
                relative_path="b.bin",
                sha256="e" * 64,
                size_bytes=3,
            ),
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="aggregate download byte cap"):
        download_artifacts(
            _raw_manifest(oversized_total),
            local_root=tmp_path / "total-cap",
            reader=lambda path: reader_calls.append(path) or b"",
        )

    assert reader_calls == []
    assert not (tmp_path / "file-cap").exists()
    assert not (tmp_path / "total-cap").exists()


def test_download_stream_cannot_exceed_manifest_declared_size(tmp_path) -> None:
    payload = b"safe"
    manifest = ArtifactManifestV1(
        run_id="download-stream-cap",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="f" * 64,
        files=(
            ArtifactFileV1(
                relative_path="result.bin",
                sha256=__import__("hashlib").sha256(payload).hexdigest(),
                size_bytes=len(payload),
            ),
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="manifest-declared artifact size"):
        download_artifacts(
            _raw_manifest(manifest),
            local_root=tmp_path / "stream-cap",
            reader=lambda _path: (payload, b"unexpected"),
        )

    assert not (tmp_path / "stream-cap" / manifest.run_id).exists()


def test_artifact_download_caps_are_fixed_and_accept_the_exact_boundary(
    monkeypatch,
) -> None:
    assert MAX_ARTIFACT_DOWNLOAD_FILE_BYTES == 64 * 1024 * 1024
    assert MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES == 256 * 1024 * 1024
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_FILE_BYTES", 4)
    monkeypatch.setattr(modal_boundary, "MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES", 6)
    manifest = ArtifactManifestV1(
        run_id="download-exact-cap",
        created_at_utc="2026-08-08T00:00:00+00:00",
        image_source_sha256="1" * 64,
        files=(
            ArtifactFileV1("a.bin", "2" * 64, 4),
            ArtifactFileV1("b.bin", "3" * 64, 2),
        ),
    )

    assert validate_artifact_download_bounds(manifest) == 6


def test_canaries_use_only_synchronous_remote_in_fixed_order() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeRemote:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, **kwargs):
            calls.append((self.name, kwargs))
            return {"harness": self.name}

    functions = {name: FakeRemote(name) for name in CANARY_ORDER}
    results = run_canaries_synchronously(functions, run_id_prefix="canary")

    assert [name for name, _ in calls] == list(CANARY_ORDER)
    assert results["all_succeeded"] is True
    assert [item["harness"] for item in results["outcomes"]] == list(CANARY_ORDER)
    assert all(item["status"] == "success" for item in results["outcomes"])
    assert all(kwargs["opportunities"] == 1 for _, kwargs in calls)
    assert [kwargs["run_id"] for _, kwargs in calls] == [
        "canary-greedy-ar",
        "canary-semantic-ar",
        "canary-openevolve-generic",
        "canary-openevolve-semantic",
    ]


@pytest.mark.parametrize("failed_position", range(len(CANARY_ORDER)))
def test_canary_aggregate_attempts_every_harness_once_after_failure(
    failed_position: int,
) -> None:
    calls: list[str] = []

    class FakeRemote:
        def __init__(self, harness: str, position: int) -> None:
            self.harness = harness
            self.position = position

        def remote(self, **_kwargs):
            calls.append(self.harness)
            if self.position == failed_position:
                raise RuntimeError("provider-bearing detail must not be retained")
            return {"harness": self.harness}

    functions = {
        harness: FakeRemote(harness, position)
        for position, harness in enumerate(CANARY_ORDER)
    }

    result = run_canaries_synchronously(functions, run_id_prefix="canary")

    assert calls == list(CANARY_ORDER)
    assert result["all_succeeded"] is False
    assert [item["status"] for item in result["outcomes"]] == [
        "failed" if position == failed_position else "success"
        for position in range(len(CANARY_ORDER))
    ]
    failed = result["outcomes"][failed_position]
    assert failed["error_type"] == "RuntimeError"
    assert "provider-bearing" not in json.dumps(result)


def test_canary_aggregate_does_not_catch_process_control_exceptions() -> None:
    calls: list[str] = []

    class FakeRemote:
        def __init__(self, harness: str) -> None:
            self.harness = harness

        def remote(self, **_kwargs):
            calls.append(self.harness)
            raise KeyboardInterrupt

    functions = {harness: FakeRemote(harness) for harness in CANARY_ORDER}

    with pytest.raises(KeyboardInterrupt):
        run_canaries_synchronously(functions, run_id_prefix="canary")

    assert calls == [CANARY_ORDER[0]]


def test_canary_aggregate_outcome_receipt_is_exact_and_sanitized() -> None:
    identity = ModalLiveCohortIdentity(
        source_tree_sha256="c" * 64,
        image_source_sha256="b" * 64,
        cohort_id="cuda-cohort-1",
    )
    outcomes = [
        {
            "harness": harness,
            "run_id": f"canary-{modal_boundary.canary_run_suffix(harness)}",
            "status": "failed" if index == 1 else "success",
            "result": (
                None if index == 1 else {"provider_content": "must-not-persist"}
            ),
            "error_type": "RuntimeError" if index == 1 else None,
        }
        for index, harness in enumerate(CANARY_ORDER)
    ]
    aggregate = {
        "schema_name": "ModalProviderCanaryAggregateResult",
        "schema_version": "1.0",
        "run_id_prefix": "canary",
        "harness_order": list(CANARY_ORDER),
        "outcomes": outcomes,
        "all_succeeded": False,
    }

    receipt = build_provider_canary_aggregate_outcome_receipt(
        aggregate,
        identity=identity,
        attempt_id="a" * 32,
    )

    assert receipt["schema_name"] == "ProviderCanaryAggregateOutcomeReceipt"
    assert receipt["schema_version"] == "1.1"
    assert receipt["source_tree_sha256"] == "c" * 64
    assert receipt["cohort_id"] == "cuda-cohort-1"
    assert receipt["attempt_id"] == "a" * 32
    assert receipt["all_succeeded"] is False
    assert all(
        set(outcome) == {"harness", "run_id", "status", "error_type"}
        for outcome in receipt["outcomes"]
    )
    assert "provider_content" not in json.dumps(receipt)
    assert (
        provider_canary_aggregate_outcome_receipt_path(
            identity,
            "a" * 32,
        ).as_posix()
        == "outputs/readiness/modal_only_final/modal_live_cohorts/"
        + ("c" * 64)
        + "/"
        + ("b" * 64)
        + "/cuda-cohort-1/action_attempts/"
        + ("a" * 32)
        + ".aggregate.json"
    )
    assert (
        validate_provider_canary_aggregate_outcome_receipt(
            receipt,
            expected_attempt_id="a" * 32,
            expected_run_id_prefix="canary",
            expected_source_tree_sha256="c" * 64,
            expected_image_source_sha256="b" * 64,
            expected_cohort_id="cuda-cohort-1",
        )
        == receipt
    )

    drifted = json.loads(json.dumps(receipt))
    drifted["outcomes"][1]["status"] = "success"
    drifted["outcomes"][1]["error_type"] = None
    with pytest.raises(ValueError, match="status is inconsistent"):
        validate_provider_canary_aggregate_outcome_receipt(drifted)

    provider_bearing_failure = json.loads(json.dumps(aggregate))
    provider_bearing_failure["outcomes"][1]["result"] = {
        "provider_content": "must-never-be-treated-as-a-failed-outcome"
    }
    with pytest.raises(ValueError, match="result outcome changed"):
        build_provider_canary_aggregate_outcome_receipt(
            provider_bearing_failure,
            identity=identity,
            attempt_id="a" * 32,
        )


def test_modal_app_source_contains_no_async_or_unbounded_invocation() -> None:
    source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert ".spawn(" not in source
    assert ".map(" not in source
    assert "modal.deploy" not in source
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"web_endpoint", "asgi_app", "wsgi_app"}
        for node in ast.walk(tree)
    )


def test_local_modal_plan_imports_no_modal_and_starts_no_calls(capsys) -> None:
    from scripts import modal_plan

    tree = ast.parse((ROOT / "scripts" / "modal_plan.py").read_text(encoding="utf-8"))
    imported = {
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.Import)
            else (
                [node.module.split(".", 1)[0]]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
        )
    }
    assert "modal" not in imported
    assert "modal_app" not in imported

    modal_plan.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["remote_calls_started"] == 0
    assert plan["schema_version"] == "1.2"
    source = build_image_source_manifest(ROOT)
    assert plan["image_source_sha256"] == source.manifest_sha256
    source_total_bytes = sum(item.size_bytes for item in source.files)
    assert plan["image_source"] == {
        "file_count": len(source.files),
        "total_bytes": source_total_bytes,
        "copy_source_bytes_upper_bound": 2 * source_total_bytes,
    }
    assert set(plan["functions"]) == set(FUNCTION_SPECS)
    assert plan["runtime_functions_preemptible"] is True
    assert plan["platform_preemption_restart_possible"] is True
    assert plan["logical_call_count_is_not_container_attempt_ceiling"] is True
    assert plan["modal_cost_gate"] == {
        "price_basis_schema": "ModalPriceBasis/1.0",
        "price_basis_max_age_hours": 48,
        "scope": (
            "local_pre_popen_request_rate_and_one_gib_month_storage_estimate_"
            "not_platform_billing_cap"
        ),
        "platform_billing_cap_enforced": False,
    }
    assert plan["image_build"] == {
        "cpu_request_cores": IMAGE_BUILD_CPU_REQUEST_CORES,
        "cpu_soft_limit_cores": None,
        "memory_request_mib": IMAGE_BUILD_MEMORY_REQUEST_MIB,
        "memory_limit_mib": None,
        "gpu": None,
        "region": None,
        "timeout_seconds": IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
        "subprocess_thread_limit": IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
        "resource_limits_exposed": False,
        "platform_compute_cost_ceiling_enforced": False,
        "network_required": True,
        "source_copy_layers": 2,
        "source_copy_backend_resource_limits_exposed": False,
    }
    for name, function in plan["functions"].items():
        spec = FUNCTION_SPECS[name]
        assert function["cpu_request_cores"] == spec.cpu_request_cores
        assert function["cpu_soft_limit_cores"] == spec.cpu_soft_limit_cores
        assert function["cpu_limit_kind"] == "soft_throttle_threshold"
        assert function["memory_request_mib"] == spec.memory_request_mib
        assert function["memory_limit_mib"] == spec.memory_limit_mib
        assert function["region"] is None
        assert function["cpu_request_cores"] == function["cpu_soft_limit_cores"]
        assert function["memory_request_mib"] == function["memory_limit_mib"]
        assert function["memory_limit_kind"] == "hard"
        assert function["platform_compute_cost_ceiling_enforced"] is False
        assert function["runtime_network_blocked"] is (
            not FUNCTION_SPECS[name].provider_secret
        )


def test_artifact_manifest_parser_rejects_extra_fields() -> None:
    payload = {
        "schema_name": "ModalRunArtifactManifest",
        "schema_version": "1.0",
        "run_id": "run-1",
        "created_at_utc": "2026-08-08T00:00:00+00:00",
        "image_source_sha256": "f" * 64,
        "files": [],
        "credentials": {"token": "secret"},
    }
    with pytest.raises(ValueError, match="unexpected or missing"):
        ArtifactManifestV1.from_dict(json.loads(json.dumps(payload)))


def test_raw_artifact_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "artifact_manifest.json"
    path.write_bytes(
        b'{"schema_name":"ModalRunArtifactManifest",'
        b'"schema_name":"ModalRunArtifactManifest",'
        b'"schema_version":"1.0","run_id":"run-1",'
        b'"created_at_utc":"2026-08-08T00:00:00+00:00",'
        b'"image_source_sha256":"ffffffffffffffffffffffffffffffff'
        b'ffffffffffffffffffffffffffffffff","files":[]}\n'
    )

    with pytest.raises(ArtifactIntegrityError, match="duplicate JSON key"):
        load_artifact_manifest(path)


def test_artifact_verification_contract_binds_verifier_provenance() -> None:
    source_run_id = "source-run-1"
    verifier_run_id = "verifier-run-1"
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id=verifier_run_id,
        app_name=APP_NAME,
        function_name="artifact_verify",
        modal_app_id="ap-verifier123",
        modal_function_id="fu-verifier123",
        modal_call_id="fc-verifier123",
        modal_image_id="im-verifier123",
        image_source_sha256="a" * 64,
        artifact_uri=volume_artifact_uri(source_run_id),
    )
    verification = ArtifactVerificationV1(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        manifest_filename="artifact_manifest.checkpoint.json",
        raw_manifest_sha256="b" * 64,
        raw_manifest_size_bytes=123,
        canonical_manifest_sha256="c" * 64,
        file_count=4,
        verifier_execution_context=context,
    )

    payload = verification.to_dict()
    assert set(payload) == ArtifactVerificationV1.FIELDS
    assert payload["schema_name"] == "ModalArtifactVerificationResult"
    assert payload["schema_version"] == "1.0"
    assert ArtifactVerificationV1.from_dict(payload) == verification

    missing_image = dict(payload)
    missing_image["verifier_execution_context"] = dict(
        payload["verifier_execution_context"]
    )
    missing_image["verifier_execution_context"]["modal_image_id"] = None
    with pytest.raises(ValueError, match="not source-run bound"):
        ArtifactVerificationV1.from_dict(missing_image)

    payload["verifier_run_id"] = source_run_id
    with pytest.raises(ValueError, match="must differ"):
        ArtifactVerificationV1.from_dict(payload)
