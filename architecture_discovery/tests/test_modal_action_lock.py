from __future__ import annotations

import os
from pathlib import Path

import pytest

import common.modal_action_lock as modal_lock


def test_modal_action_lock_is_create_only_owned_0600_and_releasable(
    tmp_path: Path,
) -> None:
    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    lock_path = tmp_path / modal_lock.MODAL_ACTION_LOCK_PATH

    modal_lock.assert_modal_action_lock_identity(descriptor)
    assert modal_lock.held_modal_action_lock_project_root(descriptor) == tmp_path
    metadata = lock_path.stat()
    assert metadata.st_uid == os.getuid()
    assert metadata.st_nlink == 1
    assert metadata.st_mode & 0o777 == 0o600

    modal_lock.release_modal_action_lock(descriptor)

    with pytest.raises(ValueError, match="not held"):
        modal_lock.held_modal_action_lock_project_root(descriptor)


def test_modal_action_lock_rejects_symlink_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "outputs").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlinks"):
        modal_lock.acquire_modal_action_lock(tmp_path)


def test_modal_action_lock_rejects_unsafe_existing_leaf(tmp_path: Path) -> None:
    lock_path = tmp_path / modal_lock.MODAL_ACTION_LOCK_PATH
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(0o644)

    with pytest.raises(ValueError, match="single-link 0600"):
        modal_lock.acquire_modal_action_lock(tmp_path)


def test_modal_action_lock_reports_nonblocking_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_flock = modal_lock.fcntl.flock

    def contend(file_descriptor: int, operation: int) -> None:
        if operation == modal_lock.fcntl.LOCK_EX | modal_lock.fcntl.LOCK_NB:
            raise BlockingIOError
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(modal_lock.fcntl, "flock", contend)
    with pytest.raises(
        modal_lock.ModalActionLockContentionError,
        match="holds the lock",
    ):
        modal_lock.acquire_modal_action_lock(tmp_path)


def test_modal_action_lock_detects_ancestor_swap_after_acquisition(
    tmp_path: Path,
) -> None:
    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    readiness = tmp_path / "outputs/readiness"
    displaced = tmp_path / "outputs/readiness-displaced"
    readiness.rename(displaced)
    readiness.mkdir(mode=0o700)
    replacement = readiness / modal_lock.MODAL_ACTION_LOCK_PATH.name
    replacement.touch(mode=0o600)

    try:
        with pytest.raises(ValueError, match="ancestor was replaced"):
            modal_lock.assert_modal_action_lock_identity(descriptor)
    finally:
        modal_lock.release_modal_action_lock(descriptor)


@pytest.mark.parametrize("relative", (Path("outputs"), Path("outputs/readiness")))
def test_modal_action_lock_revalidation_never_recreates_missing_ancestor(
    tmp_path: Path,
    relative: Path,
) -> None:
    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    original = tmp_path / relative
    displaced = original.with_name(f"{original.name}-displaced")
    original.rename(displaced)

    try:
        with pytest.raises(ValueError, match="parent is missing"):
            modal_lock.held_modal_action_lock_project_root(descriptor)
        assert not original.exists()
    finally:
        modal_lock.release_modal_action_lock(descriptor)
