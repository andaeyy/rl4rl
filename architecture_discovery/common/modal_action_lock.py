"""Shared local serialization gate for Modal launches and terminal sealing."""

from __future__ import annotations

import errno
import fcntl
import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

MODAL_ACTION_LOCK_PATH = Path("outputs/readiness/.modal_action.lock")


class ModalActionLockContentionError(RuntimeError):
    """Raised when another local Modal launch or seal holds the gate."""


@dataclass(frozen=True, slots=True)
class _HeldModalActionLock:
    project_root: Path
    directory_descriptors: tuple[int, ...]
    directory_identities: tuple[tuple[int, int], ...]
    leaf_identity: tuple[int, int]


_HELD_LOCKS: dict[int, _HeldModalActionLock] = {}


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _validate_directory(metadata: os.stat_result, *, field: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise ValueError(f"{field} must be an owned, non-writable directory")


def _open_project_root(project_root: Path) -> tuple[Path, int, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(project_root)))
    try:
        before = os.lstat(absolute)
    except FileNotFoundError:
        raise ValueError("Modal action lock project root is missing") from None
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("Modal action lock project root may not be a symlink")
    _validate_directory(before, field="Modal action lock project root")
    try:
        descriptor = os.open(absolute, _directory_flags())
    except OSError as error:
        raise ValueError("Modal action lock project root changed") from error
    opened = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise ValueError("Modal action lock project root changed")
    _validate_directory(opened, field="Modal action lock project root")
    return absolute, descriptor, opened


def _open_or_create_private_directory(
    parent_descriptor: int,
    component: str,
) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(
            component,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        with suppress(FileExistsError):
            os.mkdir(component, mode=0o700, dir_fd=parent_descriptor)
        before = os.stat(
            component,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("Modal action lock path may not traverse symlinks")
    _validate_directory(before, field="Modal action lock parent")
    try:
        descriptor = os.open(
            component,
            _directory_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ValueError("Modal action lock parent changed") from error
    opened = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise ValueError("Modal action lock parent changed")
    _validate_directory(opened, field="Modal action lock parent")
    return descriptor, opened


def _open_existing_private_directory(
    parent_descriptor: int,
    component: str,
) -> tuple[int, os.stat_result]:
    """Open one existing lock ancestor without repairing its namespace."""

    try:
        before = os.stat(
            component,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise ValueError("Modal action lock parent is missing") from None
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("Modal action lock path may not traverse symlinks")
    _validate_directory(before, field="Modal action lock parent")
    try:
        descriptor = os.open(
            component,
            _directory_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ValueError("Modal action lock parent changed") from error
    opened = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise ValueError("Modal action lock parent changed")
    _validate_directory(opened, field="Modal action lock parent")
    return descriptor, opened


def _open_lock_leaf(parent_descriptor: int) -> tuple[int, os.stat_result]:
    leaf = MODAL_ACTION_LOCK_PATH.name
    base_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    base_flags |= getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(
            leaf,
            base_flags | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        created = True
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise ValueError("Modal action lock leaf could not be created") from error
        try:
            descriptor = os.open(leaf, base_flags, dir_fd=parent_descriptor)
        except OSError as reopen_error:
            raise ValueError("Modal action lock leaf changed") from reopen_error
    try:
        if created:
            os.fchmod(descriptor, 0o600)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ValueError(
                "Modal action lock must be an owned single-link 0600 regular file"
            )
        rebound = os.stat(
            leaf,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (rebound.st_dev, rebound.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Modal action lock path differs from its descriptor")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _open_existing_lock_leaf(
    parent_descriptor: int,
) -> tuple[int, os.stat_result]:
    """Open and validate the lock leaf without ever creating it."""

    leaf = MODAL_ACTION_LOCK_PATH.name
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(leaf, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        raise ValueError("Modal action lock leaf is missing") from None
    except OSError as error:
        raise ValueError("Modal action lock leaf changed") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ValueError(
                "Modal action lock must be an owned single-link 0600 regular file"
            )
        rebound = os.stat(
            leaf,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (rebound.st_dev, rebound.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Modal action lock path differs from its descriptor")
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def acquire_modal_action_lock(project_root: str | Path) -> int:
    """Acquire the nonblocking gate shared by paid launches and final sealing."""

    absolute, root_descriptor, root_metadata = _open_project_root(Path(project_root))
    directory_descriptors = [root_descriptor]
    directory_identities = [(root_metadata.st_dev, root_metadata.st_ino)]
    lock_descriptor: int | None = None
    try:
        parent_descriptor = root_descriptor
        for component in MODAL_ACTION_LOCK_PATH.parent.parts:
            descriptor, metadata = _open_or_create_private_directory(
                parent_descriptor,
                component,
            )
            directory_descriptors.append(descriptor)
            directory_identities.append((metadata.st_dev, metadata.st_ino))
            parent_descriptor = descriptor
        lock_descriptor, lock_metadata = _open_lock_leaf(parent_descriptor)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModalActionLockContentionError(
                "another local Modal action or lineage seal holds the lock"
            ) from error
        _HELD_LOCKS[lock_descriptor] = _HeldModalActionLock(
            project_root=absolute,
            directory_descriptors=tuple(directory_descriptors),
            directory_identities=tuple(directory_identities),
            leaf_identity=(lock_metadata.st_dev, lock_metadata.st_ino),
        )
        assert_modal_action_lock_identity(lock_descriptor)
        return lock_descriptor
    except BaseException:
        if lock_descriptor is not None:
            with suppress(OSError):
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
            _HELD_LOCKS.pop(lock_descriptor, None)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)
        raise


def acquire_existing_modal_action_lock(project_root: str | Path) -> int:
    """Acquire the gate without creating or repairing any filesystem object.

    This variant is for genuinely read-only inspection commands.  It fails if
    the private lock namespace has not already been initialized by a mutating
    Modal action instead of turning an inspection into a persistent write.
    """

    absolute, root_descriptor, root_metadata = _open_project_root(Path(project_root))
    directory_descriptors = [root_descriptor]
    directory_identities = [(root_metadata.st_dev, root_metadata.st_ino)]
    lock_descriptor: int | None = None
    try:
        parent_descriptor = root_descriptor
        for component in MODAL_ACTION_LOCK_PATH.parent.parts:
            descriptor, metadata = _open_existing_private_directory(
                parent_descriptor,
                component,
            )
            directory_descriptors.append(descriptor)
            directory_identities.append((metadata.st_dev, metadata.st_ino))
            parent_descriptor = descriptor
        lock_descriptor, lock_metadata = _open_existing_lock_leaf(parent_descriptor)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModalActionLockContentionError(
                "another local Modal action or lineage seal holds the lock"
            ) from error
        _HELD_LOCKS[lock_descriptor] = _HeldModalActionLock(
            project_root=absolute,
            directory_descriptors=tuple(directory_descriptors),
            directory_identities=tuple(directory_identities),
            leaf_identity=(lock_metadata.st_dev, lock_metadata.st_ino),
        )
        assert_modal_action_lock_identity(lock_descriptor)
        return lock_descriptor
    except BaseException:
        if lock_descriptor is not None:
            with suppress(OSError):
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
            _HELD_LOCKS.pop(lock_descriptor, None)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)
        raise


def assert_modal_action_lock_identity(file_descriptor: int) -> None:
    """Fail if a held lock's namespace was replaced after acquisition."""

    held = _HELD_LOCKS.get(file_descriptor)
    if held is None:
        raise ValueError("Modal action lock descriptor is not held by this process")
    current = os.fstat(file_descriptor)
    if (
        (current.st_dev, current.st_ino) != held.leaf_identity
        or not stat.S_ISREG(current.st_mode)
        or current.st_uid != os.getuid()
        or current.st_nlink != 1
        or stat.S_IMODE(current.st_mode) != 0o600
    ):
        raise ValueError("held Modal action lock descriptor changed")
    parent_descriptor = held.directory_descriptors[-1]
    rebound = os.stat(
        MODAL_ACTION_LOCK_PATH.name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (rebound.st_dev, rebound.st_ino) != held.leaf_identity:
        raise ValueError("held Modal action lock path was replaced")

    descriptor, metadata = _open_project_root(held.project_root)[1:]
    reopened = [descriptor]
    observed = [(metadata.st_dev, metadata.st_ino)]
    try:
        parent = descriptor
        for component in MODAL_ACTION_LOCK_PATH.parent.parts:
            child, child_metadata = _open_existing_private_directory(
                parent,
                component,
            )
            reopened.append(child)
            observed.append((child_metadata.st_dev, child_metadata.st_ino))
            parent = child
        if tuple(observed) != held.directory_identities:
            raise ValueError("held Modal action lock ancestor was replaced")
        rebound_from_path = os.stat(
            MODAL_ACTION_LOCK_PATH.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (rebound_from_path.st_dev, rebound_from_path.st_ino) != (held.leaf_identity):
            raise ValueError("held Modal action lock namespace was replaced")
    finally:
        for reopened_descriptor in reversed(reopened):
            os.close(reopened_descriptor)


def held_modal_action_lock_project_root(file_descriptor: int) -> Path:
    """Return the exact project root protected by one currently held lock.

    Callers that make security decisions about project-relative state must not
    accept an independently supplied root: a valid lock for one checkout must
    never authorize reads or writes in another checkout.  Revalidating the
    complete lock namespace before returning the captured root keeps that
    binding explicit.
    """

    assert_modal_action_lock_identity(file_descriptor)
    held = _HELD_LOCKS.get(file_descriptor)
    if held is None:  # pragma: no cover - guarded by the assertion above
        raise ValueError("Modal action lock descriptor is not held by this process")
    return held.project_root


def release_modal_action_lock(file_descriptor: int) -> None:
    """Release a gate acquired by :func:`acquire_modal_action_lock`."""

    held = _HELD_LOCKS.pop(file_descriptor, None)
    if held is None:
        raise ValueError("Modal action lock descriptor is not held by this process")
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(file_descriptor)
        for descriptor in reversed(held.directory_descriptors):
            os.close(descriptor)
