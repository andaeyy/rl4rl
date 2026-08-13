from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from study.serialization import create_json_exclusive


def test_create_json_exclusive_is_private_create_only_and_durable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "new" / "nested" / "frozen.json"
    fsync_types: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsync_types.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    create_json_exclusive(destination, {"value": 1})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"value": 1}
    metadata = destination.stat(follow_symlinks=False)
    assert stat.S_ISREG(metadata.st_mode)
    assert metadata.st_nlink == 1
    assert stat.S_IMODE(metadata.st_mode) & 0o077 == 0
    assert stat.S_IFREG in fsync_types
    assert stat.S_IFDIR in fsync_types

    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        create_json_exclusive(destination, {"value": 2})
    assert destination.read_bytes() == original


def test_create_json_exclusive_rejects_parent_and_destination_symlinks(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="parent may not contain a symlink"):
        create_json_exclusive(linked_parent / "escaped.json", {"safe": True})
    assert not list(outside.iterdir())

    safe_parent = tmp_path / "safe-parent"
    safe_parent.mkdir()
    target = outside / "target.json"
    target.write_text("unchanged\n", encoding="utf-8")
    destination = safe_parent / "frozen.json"
    destination.symlink_to(target)
    with pytest.raises(FileExistsError):
        create_json_exclusive(destination, {"safe": True})
    assert target.read_text(encoding="utf-8") == "unchanged\n"
    assert destination.is_symlink()


def test_create_json_exclusive_parent_swap_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent = tmp_path / "stable-parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "frozen.json"
    real_open = os.open
    swapped = False

    def swap_then_open(
        path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == destination.name and flags & os.O_CREAT and not swapped:
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_then_open)
    with pytest.raises(ValueError, match="parent may not contain a symlink"):
        create_json_exclusive(destination, {"safe": True})

    assert not list(outside.iterdir())
    assert json.loads((moved / "frozen.json").read_text(encoding="utf-8")) == {
        "safe": True
    }


def test_create_json_exclusive_keeps_partial_file_as_quarantine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "safe" / "frozen.json"
    real_write = os.write
    writes = 0

    def partial_then_fail(descriptor: int, payload) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return real_write(descriptor, payload[:5])
        raise OSError("injected direct-write failure")

    monkeypatch.setattr(os, "write", partial_then_fail)
    with pytest.raises(OSError, match="injected direct-write failure"):
        create_json_exclusive(destination, {"safe": True})

    partial = destination.read_bytes()
    assert partial
    assert len(partial) == 5
    with pytest.raises(FileExistsError):
        create_json_exclusive(destination, {"safe": False})
    assert destination.read_bytes() == partial


def test_create_json_exclusive_uses_full_direct_write_without_hard_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "safe" / "frozen.json"
    real_write = os.write
    writes = 0

    def short_write(descriptor: int, payload) -> int:
        nonlocal writes
        writes += 1
        return real_write(descriptor, payload[:3])

    def forbid_link(*_args, **_kwargs):
        raise AssertionError("exclusive JSON publication may not use hard links")

    monkeypatch.setattr(os, "write", short_write)
    monkeypatch.setattr(os, "link", forbid_link)
    create_json_exclusive(destination, {"safe": True})

    assert writes > 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {"safe": True}
