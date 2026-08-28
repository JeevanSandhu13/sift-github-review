from __future__ import annotations

import os
import hashlib
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from sift.file_lock import exclusive_file_lock
from sift.secure_file import (
    _windows_creation_disposition,
    append_bytes_no_follow,
    copy_regular_no_follow,
    open_regular_no_follow,
    read_bytes_no_follow,
    write_bytes_no_follow,
)


def test_secure_copy_hashes_bytes_and_refuses_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable source")
    destination = tmp_path / "destination.bin"
    size, digest = copy_regular_no_follow(source, destination, max_bytes=64)
    assert size == len(b"stable source")
    assert digest == hashlib.sha256(b"stable source").hexdigest()
    assert destination.read_bytes() == b"stable source"
    with pytest.raises(FileExistsError):
        copy_regular_no_follow(source, destination)


def test_secure_copy_accepts_ctime_only_metadata_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud hydration/provenance updates must not look like data mutation."""
    source = tmp_path / "source.bin"
    payload = b"stable cloud-backed source"
    source.write_bytes(payload)
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    real_fstat = os.fstat
    source_fstat_calls = 0

    def metadata_change(descriptor: int):
        nonlocal source_fstat_calls
        metadata = real_fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != source_identity:
            return metadata
        source_fstat_calls += 1
        if source_fstat_calls < 3:
            return metadata
        return SimpleNamespace(
            **{
                name: getattr(metadata, name)
                for name in dir(metadata)
                if name.startswith("st_") and name != "st_ctime_ns"
            },
            st_ctime_ns=metadata.st_ctime_ns + 1,
        )

    monkeypatch.setattr(os, "fstat", metadata_change)
    destination = tmp_path / "destination.bin"
    size, digest = copy_regular_no_follow(source, destination)

    assert source_fstat_calls >= 4  # includes the content recheck
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload


def test_secure_copy_rejects_content_change_even_when_only_ctime_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ctime fallback must remain a content-integrity check."""
    source = tmp_path / "source.bin"
    original = b"original content"
    replacement = b"modified content"
    assert len(original) == len(replacement)
    source.write_bytes(original)
    original_stat = source.stat()
    source_identity = (original_stat.st_dev, original_stat.st_ino)
    real_fstat = os.fstat
    real_lseek = os.lseek
    source_fstat_calls = 0
    replaced = False

    def metadata_change(descriptor: int):
        nonlocal source_fstat_calls
        metadata = real_fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != source_identity:
            return metadata
        source_fstat_calls += 1
        if source_fstat_calls < 3:
            return metadata
        return SimpleNamespace(
            **{
                name: getattr(metadata, name)
                for name in dir(metadata)
                if name.startswith("st_")
                and name not in {"st_ctime_ns", "st_mtime_ns"}
            },
            st_ctime_ns=metadata.st_ctime_ns + 1,
            st_mtime_ns=original_stat.st_mtime_ns,
        )

    def replace_before_verification(
        descriptor: int, position: int, whence: int,
    ) -> int:
        nonlocal replaced
        metadata = real_fstat(descriptor)
        if (
            not replaced
            and (metadata.st_dev, metadata.st_ino) == source_identity
            and position == 0
            and whence == os.SEEK_SET
        ):
            source.write_bytes(replacement)
            os.utime(
                source,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            replaced = True
        return real_lseek(descriptor, position, whence)

    monkeypatch.setattr(os, "fstat", metadata_change)
    monkeypatch.setattr(os, "lseek", replace_before_verification)
    destination = tmp_path / "destination.bin"

    with pytest.raises(OSError, match="source file changed while copying"):
        copy_regular_no_follow(source, destination)
    assert replaced
    assert not destination.exists()


def test_secure_copy_rejects_source_symlink_and_size_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"too large")
    link = tmp_path / "source-link.bin"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(OSError):
        copy_regular_no_follow(link, tmp_path / "from-link.bin")
    with pytest.raises(OSError, match="size limit"):
        copy_regular_no_follow(source, tmp_path / "oversized.bin", max_bytes=2)
    assert not (tmp_path / "from-link.bin").exists()
    assert not (tmp_path / "oversized.bin").exists()


def test_windows_creation_never_truncates_before_validation() -> None:
    assert _windows_creation_disposition(os.O_WRONLY | os.O_CREAT | os.O_TRUNC) == 4
    assert _windows_creation_disposition(os.O_WRONLY | os.O_CREAT | os.O_EXCL) == 1
    assert _windows_creation_disposition(os.O_RDONLY) == 3


def test_secure_open_refuses_symlink_without_modifying_target(tmp_path: Path) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    link = tmp_path / "log.txt"
    link.symlink_to(target)
    with pytest.raises(OSError):
        open_regular_no_follow(link, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    assert target.read_text(encoding="utf-8") == "preserve"


@pytest.mark.skipif(os.name == "nt", reason="exercises the POSIX open path")
def test_secure_open_defers_truncation_until_regular_file_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.txt"
    target.write_text("preserve", encoding="utf-8")
    real_open = os.open
    real_fstat = os.fstat
    observed_flags: list[int] = []

    def tracked_open(path, flags, mode=0o777):
        observed_flags.append(flags)
        return real_open(path, flags, mode)

    def special_file(_descriptor):
        metadata = real_fstat(_descriptor)
        return SimpleNamespace(**{
            name: getattr(metadata, name)
            for name in dir(metadata)
            if name.startswith("st_") and name != "st_mode"
        }, st_mode=stat.S_IFIFO)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "fstat", special_file)
    with pytest.raises(OSError, match="not a regular file"):
        open_regular_no_follow(target, os.O_WRONLY | os.O_TRUNC)

    assert observed_flags and not observed_flags[0] & os.O_TRUNC
    assert target.read_text(encoding="utf-8") == "preserve"


def test_secure_read_write_and_append_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.bin"
    write_bytes_no_follow(path, b"first", sync=True)
    append_bytes_no_follow(path, b"-second", sync=True)
    assert read_bytes_no_follow(path, max_bytes=32) == b"first-second"
    with pytest.raises(OSError, match="size limit"):
        read_bytes_no_follow(path, max_bytes=3)


def test_secure_helpers_refuse_symlink_for_every_access_mode(tmp_path: Path) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    link = tmp_path / "state.txt"
    link.symlink_to(target)
    with pytest.raises(OSError):
        append_bytes_no_follow(link, b"bad")
    with pytest.raises(OSError):
        write_bytes_no_follow(link, b"bad")
    with pytest.raises(OSError):
        read_bytes_no_follow(link)
    assert target.read_text(encoding="utf-8") == "preserve"


def test_file_lock_refuses_symlink_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    link = tmp_path / "state.lock"
    link.symlink_to(target)
    with pytest.raises(OSError):
        with exclusive_file_lock(link):
            raise AssertionError("lock body must not run")
    assert target.read_text(encoding="utf-8") == "preserve"
