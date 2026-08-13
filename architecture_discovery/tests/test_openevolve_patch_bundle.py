from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from scripts import openevolve_patch_bundle as patch_bundle

ROOT = Path(__file__).resolve().parents[1]


def _copy_unpatched_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    vendor_parent = project / "vendor"
    patch_parent = project / "vendor_patches"
    vendor_parent.mkdir(parents=True)
    patch_parent.mkdir()
    shutil.copy2(
        ROOT / patch_bundle.OPENEVOLVE_PATCH_RELATIVE_PATH,
        project / patch_bundle.OPENEVOLVE_PATCH_RELATIVE_PATH,
    )
    shutil.copy2(
        ROOT / patch_bundle.OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH,
        project / patch_bundle.OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH,
    )
    completed = subprocess.run(
        [
            "git",
            "clone",
            "--local",
            "--no-hardlinks",
            str(ROOT / patch_bundle.OPENEVOLVE_VENDOR_RELATIVE_PATH),
            str(project / patch_bundle.OPENEVOLVE_VENDOR_RELATIVE_PATH),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    return project


def test_checked_in_patch_bundle_and_materialized_files_are_exact() -> None:
    status = patch_bundle.validate_applied_patch_bundle(ROOT)

    assert status.base_commit == patch_bundle.OPENEVOLVE_BASE_COMMIT
    assert status.patch_sha256 == patch_bundle.OPENEVOLVE_PATCH_SHA256
    assert (
        status.provider_patch_sha256
        == patch_bundle.OPENEVOLVE_PROVIDER_PATCH_SHA256
    )
    assert status.patched_file_sha256 == (
        patch_bundle.OPENEVOLVE_PATCHED_FILE_SHA256
    )
    assert status.applied is True


def test_apply_is_explicit_and_idempotent_in_temporary_clone(tmp_path: Path) -> None:
    project = _copy_unpatched_project(tmp_path)
    vendor = project / patch_bundle.OPENEVOLVE_VENDOR_RELATIVE_PATH

    with pytest.raises(
        patch_bundle.OpenEvolvePatchBundleError,
        match="not applied",
    ):
        patch_bundle.ensure_openevolve_patch_bundle(project)

    first = patch_bundle.ensure_openevolve_patch_bundle(project, apply=True)
    before = {
        relative: (vendor / relative).read_bytes()
        for relative in patch_bundle.OPENEVOLVE_PATCHED_FILE_SHA256
    }
    second = patch_bundle.ensure_openevolve_patch_bundle(project, apply=True)
    after = {
        relative: (vendor / relative).read_bytes()
        for relative in patch_bundle.OPENEVOLVE_PATCHED_FILE_SHA256
    }

    assert first == second
    assert before == after
    assert all(
        patch_bundle._sha256_bytes(after[relative]) == expected
        for relative, expected in patch_bundle.OPENEVOLVE_PATCHED_FILE_SHA256.items()
    )


def test_apply_rejects_partial_or_unreviewed_worktree_without_mutating_it(
    tmp_path: Path,
) -> None:
    project = _copy_unpatched_project(tmp_path)
    target = (
        project
        / patch_bundle.OPENEVOLVE_VENDOR_RELATIVE_PATH
        / "openevolve/process_parallel.py"
    )
    target.write_bytes(target.read_bytes() + b"\n# unreviewed drift\n")
    before = target.read_bytes()

    with pytest.raises(
        patch_bundle.OpenEvolvePatchBundleError,
        match="neither exact base nor exact reviewed patched state",
    ):
        patch_bundle.ensure_openevolve_patch_bundle(project, apply=True)

    assert target.read_bytes() == before


def test_apply_rejects_patch_digest_drift_before_vendor_mutation(
    tmp_path: Path,
) -> None:
    project = _copy_unpatched_project(tmp_path)
    patch = project / patch_bundle.OPENEVOLVE_PATCH_RELATIVE_PATH
    patch.write_bytes(patch.read_bytes() + b"\n")
    vendor = project / patch_bundle.OPENEVOLVE_VENDOR_RELATIVE_PATH
    before = {
        relative: (vendor / relative).read_bytes()
        for relative in patch_bundle.OPENEVOLVE_PATCHED_FILE_SHA256
    }

    with pytest.raises(
        patch_bundle.OpenEvolvePatchBundleError,
        match="patch SHA-256 differs",
    ):
        patch_bundle.ensure_openevolve_patch_bundle(project, apply=True)

    assert before == {
        relative: (vendor / relative).read_bytes()
        for relative in patch_bundle.OPENEVOLVE_PATCHED_FILE_SHA256
    }


def test_validation_rejects_symlinked_vendor_ancestor(tmp_path: Path) -> None:
    project = _copy_unpatched_project(tmp_path)
    vendor_parent = project / "vendor"
    outside_vendor_parent = tmp_path / "outside-vendor"
    vendor_parent.rename(outside_vendor_parent)
    vendor_parent.symlink_to(outside_vendor_parent, target_is_directory=True)

    with pytest.raises(
        patch_bundle.OpenEvolvePatchBundleError,
        match="may not traverse symbolic links",
    ):
        patch_bundle.ensure_openevolve_patch_bundle(project, apply=True)
