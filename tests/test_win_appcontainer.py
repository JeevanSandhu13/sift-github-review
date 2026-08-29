"""Tests for the Windows AppContainer and Job Object backend.

This module is split deliberately into pure planning functions (no
OS calls -- ``plan_acl_grants``, ``plan_capability_sids``,
``plan_job_limits``) and an OS-calling application layer
(``create_appcontainer_profile``, ``grant_acl``, ``spawn_in_appcontainer``,
``AppContainerRun``, ``probe_appcontainer_health``). Only the pure
layer is exercised on every platform and is the single source of truth for
what gets granted. Native Windows qualification separately exercises the API
bindings and live confinement behavior.

The OS-calling layer is guarded by ``_require_windows()`` and raises
immediately when called off Windows; that guard itself is pinned
below (``test_os_calling_functions_refuse_off_windows``) since a
future refactor that accidentally let one of those functions run
partway before failing would be exactly the kind of "looks like it
worked" false confidence the module's docstring warns about.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sift import win_appcontainer as wa


def test_profile_deletion_retries_the_documented_undetermined_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_hr = 0x80070020
    delete_results = [failed_hr, failed_hr, failed_hr, 0]
    sleeps: list[float] = []

    class _FakeUserEnv:
        @staticmethod
        def CreateAppContainerProfile(*_args) -> int:
            return 0

        @staticmethod
        def DeleteAppContainerProfile(_name: str) -> int:
            return delete_results.pop(0)

    monkeypatch.setattr(wa, "_require_windows", lambda: None)
    monkeypatch.setattr(wa, "_userenv", _FakeUserEnv(), raising=False)
    monkeypatch.setattr(
        wa,
        "_advapi32",
        SimpleNamespace(FreeSid=lambda _sid: 0),
        raising=False,
    )
    monkeypatch.setattr(wa.time, "sleep", sleeps.append)

    _sid, cleanup = wa.create_appcontainer_profile("Sift.Test.Retry")
    cleanup()

    assert delete_results == []
    assert sleeps == list(wa._PROFILE_DELETE_RETRY_DELAYS_SECONDS[:3])

# ---------------------------------------------------------------------------
# plan_capability_sids
# ---------------------------------------------------------------------------


def test_capability_sids_is_empty() -> None:
    """The central design decision behind "no network":
    zero capabilities granted, ever. A future contributor adding a
    capability here (e.g. to fix some feature request) would silently
    reopen network/device access for every future script -- this test
    exists specifically to make that change require deliberately
    editing this assertion, not just adding a line elsewhere."""
    assert wa.plan_capability_sids() == ()


# ---------------------------------------------------------------------------
# plan_acl_grants
# ---------------------------------------------------------------------------


def test_acl_grants_basic_shape(tmp_path: Path) -> None:
    run_dir = tmp_path / ".sift" / "runs" / "abc123"
    run_dir.mkdir(parents=True)
    cwd = tmp_path

    grants = wa.plan_acl_grants(run_dir, cwd)

    # Exactly 4 operations: protect/traverse .sift and .sift/runs before
    # granting the workspace, then expose only this run subtree.
    assert len(grants) == 4
    assert grants[0] == wa.AclGrant(
        path=str(cwd / ".sift"),
        mask=wa.GENERIC_EXECUTE,
        allow=True,
        inherit=False,
        protect=True,
    )
    assert grants[1] == wa.AclGrant(
        path=str(cwd / ".sift" / "runs"),
        mask=wa.GENERIC_EXECUTE,
        allow=True,
        inherit=False,
        protect=True,
    )
    assert grants[2] == wa.AclGrant(
        path=str(cwd),
        mask=wa._READ_WRITE_MASK,
        allow=True,
    )
    assert grants[3] == wa.AclGrant(
        path=str(run_dir),
        mask=wa._READ_WRITE_MASK,
        allow=True,
    )


def test_acl_grants_protects_sift_before_cwd_allow(tmp_path: Path) -> None:
    """The boundary must stop inheritance before the broad cwd ACE lands."""
    run_dir = tmp_path / ".sift" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    grants = wa.plan_acl_grants(run_dir, tmp_path)

    cwd_allow_idx = next(
        i for i, g in enumerate(grants) if g.path == str(tmp_path) and g.allow
    )
    sift_protect_idx = next(
        i
        for i, g in enumerate(grants)
        if g.path == str(tmp_path / ".sift") and g.protect
    )
    assert sift_protect_idx < cwd_allow_idx
    assert grants[sift_protect_idx].inherit is False
    assert grants[sift_protect_idx].mask == wa.GENERIC_EXECUTE


def test_acl_grants_run_dir_exposed_after_protected_boundaries(tmp_path: Path) -> None:
    """Only the current run gets a recursive read/write package ACE."""
    run_dir = tmp_path / ".sift" / "runs" / "r1"
    run_dir.mkdir(parents=True)
    grants = wa.plan_acl_grants(run_dir, tmp_path)

    boundaries = [i for i, g in enumerate(grants) if g.protect]
    run_idx = next(i for i, g in enumerate(grants) if g.path == str(run_dir))
    assert boundaries and run_idx > max(boundaries)
    assert grants[run_idx].mask == wa._READ_WRITE_MASK
    assert grants[run_idx].inherit is True
    assert all(g.allow for g in grants)


def test_acl_grants_same_cwd_and_run_does_not_duplicate_write_grant(
    tmp_path: Path,
) -> None:
    grants = wa.plan_acl_grants(tmp_path, tmp_path)
    assert grants == (
        wa.AclGrant(path=str(tmp_path), mask=wa._READ_WRITE_MASK, allow=True),
    )


def test_acl_application_preserves_dacl_protection_state() -> None:
    source = Path(wa.__file__).read_text(encoding="utf-8")
    assert "GetSecurityDescriptorControl" in source
    assert "PROTECTED_DACL_SECURITY_INFORMATION" in source
    assert "UNPROTECTED_DACL_SECURITY_INFORMATION" in source
    assert "was_protected" in source


def test_acl_grants_extra_read_paths_included_when_they_exist(
    tmp_path: Path,
) -> None:
    extra = tmp_path / "python_install"
    extra.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    grants = wa.plan_acl_grants(run_dir, tmp_path, extra_read_paths=(str(extra),))

    read_grant = next(g for g in grants if g.path == str(extra))
    assert read_grant.allow is True
    assert read_grant.mask == wa._READ_EXECUTE_MASK
    # Read+execute must NOT include write -- a script must never be
    # able to modify its own interpreter install.
    assert not (read_grant.mask & wa.GENERIC_WRITE)


def test_acl_grants_extra_read_paths_skips_nonexistent(tmp_path: Path) -> None:
    """Mirrors ``_bwrap_argv``'s ``_ro_bind_if_exists`` pattern: a
    caller-supplied extra path that doesn't exist on this machine is
    silently skipped rather than granted (which would just fail at
    apply time anyway) or raising (which would take down the whole
    run over an optional path)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    missing = tmp_path / "does-not-exist"

    grants = wa.plan_acl_grants(run_dir, tmp_path, extra_read_paths=(str(missing),))
    assert all(g.path != str(missing) for g in grants)


def test_acl_grants_empty_extra_path_string_skipped(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    grants = wa.plan_acl_grants(run_dir, tmp_path, extra_read_paths=("",))
    assert all(g.path != "" for g in grants)


def test_acl_grants_write_mask_excludes_delete_and_dac(tmp_path: Path) -> None:
    """No grant anywhere in the plan includes DELETE, WRITE_DAC, or
    WRITE_OWNER -- a script must never be able to widen its own
    confinement by taking ownership of or re-ACLing a path it can
    write to. GENERIC_READ/GENERIC_WRITE are themselves composite
    masks the kernel expands, but none of the *raw* high bits used
    here alias into the owner/DACL-modification rights."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    grants = wa.plan_acl_grants(run_dir, tmp_path, extra_read_paths=("/usr",))
    DELETE = 0x00010000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    for g in grants:
        assert not (g.mask & DELETE)
        assert not (g.mask & WRITE_DAC)
        assert not (g.mask & WRITE_OWNER)
    writable = [g for g in grants if g.mask & wa.GENERIC_WRITE]
    assert writable
    assert all(g.mask & wa.GENERIC_EXECUTE for g in writable)


# ---------------------------------------------------------------------------
# plan_job_limits
# ---------------------------------------------------------------------------


def test_job_limits_basic() -> None:
    limits = wa.plan_job_limits(600, 2 * 1024**3, 64)
    assert limits.cpu_seconds == 600
    assert limits.memory_bytes == 2 * 1024**3
    assert limits.max_processes == 64
    assert limits.kill_on_job_close is True


def test_job_limits_clamps_negative_to_zero() -> None:
    """Zero means "disabled" (see the module's ``create_job_object``
    docstring); a negative input is nonsensical and must not silently
    become "unlimited" or crash the struct-building code -- clamp to
    the disabled value instead."""
    limits = wa.plan_job_limits(-5, -100, -1)
    assert limits.cpu_seconds == 0
    assert limits.memory_bytes == 0
    assert limits.max_processes == 0


def test_job_limits_zero_disables() -> None:
    limits = wa.plan_job_limits(0, 0, 0)
    assert limits.cpu_seconds == 0
    assert limits.memory_bytes == 0
    assert limits.max_processes == 0


def test_windows_environment_block_is_case_insensitively_sorted() -> None:
    block = wa._windows_environment_block({"zeta": "3", "Alpha": "1", "beta": "2"})
    assert block == "Alpha=1\0beta=2\0zeta=3\0\0"


@pytest.mark.parametrize(
    "environment",
    [
        {"": "value"},
        {"BAD=KEY": "value"},
        {"BAD\0KEY": "value"},
        {"KEY": "bad\0value"},
        {"Path": "one", "PATH": "two"},
    ],
)
def test_windows_environment_block_rejects_ambiguous_entries(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        wa._windows_environment_block(environment)


# ---------------------------------------------------------------------------
# OS-calling layer must refuse cleanly off Windows
# ---------------------------------------------------------------------------


def test_localfree_is_bound_from_its_documented_kernel32_dll() -> None:
    source = Path(wa.__file__).read_text(encoding="utf-8")
    assert "_kernel32.LocalFree.argtypes" in source
    assert "_kernel32.LocalFree.restype" in source
    assert "_advapi32.LocalFree" not in source


def test_health_probe_uses_the_same_python_runtime_as_real_analysis() -> None:
    source = Path(wa.__file__).read_text(encoding="utf-8")
    probe = source.split("def probe_appcontainer_health", 1)[1]
    assert '"vendor_python" / "python.exe"' in probe
    assert "sys.executable" in probe
    assert '"LOCALAPPDATA"' in probe
    assert "WindowsPowerShell" not in probe


def test_os_calling_functions_refuse_off_windows() -> None:
    """Every function past the pure-planning layer must raise
    immediately via ``_require_windows()`` rather than attempting a
    ctypes call that would fail in some less predictable way (or,
    worse, silently no-op) on a platform with no
    ``kernel32``/``advapi32``/``userenv`` DLLs to bind against."""
    if wa._IS_WINDOWS:
        pytest.skip("this test asserts the off-Windows refusal path")

    with pytest.raises(RuntimeError, match="win32"):
        wa.create_appcontainer_profile("Sift.Test")
    with pytest.raises(RuntimeError, match="win32"):
        wa.grant_acl(wa.AclGrant(path="/tmp", mask=1, allow=True), sid=None)
    with pytest.raises(RuntimeError, match="win32"):
        wa.create_job_object(wa.plan_job_limits(1, 1, 1))
    with pytest.raises(RuntimeError, match="win32"):
        wa.spawn_in_appcontainer(["cmd"], Path("/tmp"), {}, sid=None, job_handle=None)


def test_appcontainer_run_context_manager_refuses_off_windows(
    tmp_path: Path,
) -> None:
    if wa._IS_WINDOWS:
        pytest.skip("this test asserts the off-Windows refusal path")
    ctx = wa.AppContainerRun(
        ["cmd"],
        tmp_path,
        tmp_path / "run",
        {},
        extra_read_paths=(),
        cpu_seconds=10,
        memory_bytes=0,
        max_processes=8,
    )
    with pytest.raises(RuntimeError, match="win32"):
        ctx.__enter__()


def test_probe_appcontainer_health_returns_false_off_windows() -> None:
    """The mandatory health probe itself must degrade to a clean,
    honest ``(False, ...)`` off Windows rather than raising -- this
    is what lets ``env_detect.appcontainer_probe_result`` short-
    circuit on non-Windows platforms without every caller needing its
    own try/except around the probe."""
    if wa._IS_WINDOWS:
        pytest.skip("this test asserts the off-Windows short-circuit")
    ok, detail = wa.probe_appcontainer_health()
    assert ok is False
    assert "windows" in detail.lower()


@pytest.mark.skipif(not wa._IS_WINDOWS, reason="requires a real Windows kernel")
def test_live_appcontainer_security_probe() -> None:
    """Release-lane proof: the native backend may ship only when the
    exact production-shaped probe passes on an actual Windows host."""
    ok, detail = wa.probe_appcontainer_health()
    assert ok, detail


# ---------------------------------------------------------------------------
# AclGrant / JobLimits are plain, comparable, immutable data
# ---------------------------------------------------------------------------


def test_acl_grant_is_frozen_and_comparable() -> None:
    g1 = wa.AclGrant(path="/x", mask=1, allow=True)
    g2 = wa.AclGrant(path="/x", mask=1, allow=True)
    assert g1 == g2
    with pytest.raises(Exception):  # noqa: B017 — dataclass FrozenInstanceError
        g1.path = "/y"  # type: ignore[misc]


def test_job_limits_is_frozen_and_comparable() -> None:
    j1 = wa.plan_job_limits(1, 2, 3)
    j2 = wa.plan_job_limits(1, 2, 3)
    assert j1 == j2
    with pytest.raises(Exception):  # noqa: B017 — dataclass FrozenInstanceError
        j1.cpu_seconds = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Windows single-file ceiling planning/monitoring (portable Python layer)
# ---------------------------------------------------------------------------


def test_file_size_monitor_covers_nested_workspace_files(tmp_path: Path) -> None:
    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=32,
    )
    target = tmp_path / "exports" / "nested" / "runaway.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 33)

    violation = monitor.check()

    assert violation == wa.FileSizeViolation(target, 33, 32)


def test_file_size_monitor_matches_sift_deny_then_run_reallow(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".sift" / "runs" / "current"
    private_dir = tmp_path / ".sift" / "private"
    run_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    monitor = wa.WritableFileSizeMonitor(
        (
            wa.WritableScope(
                root=tmp_path,
                excluded_subtrees=(tmp_path / ".sift",),
            ),
            wa.WritableScope(root=run_dir),
        ),
        limit_bytes=16,
    )

    # Unrelated Sift state is denied to the AppContainer and therefore is
    # not part of the script-writable surface the parent must attribute.
    (private_dir / "session.db").write_bytes(b"x" * 100)
    assert monitor.check() is None

    # The current run is explicitly exposed inside that protected tree.
    result = run_dir / "oversized.jsonl"
    result.write_bytes(b"y" * 17)
    violation = monitor.check()
    assert violation is not None
    assert violation.path == result


def test_file_size_monitor_grandfathers_only_untouched_oversized_files(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "research-data.bin"
    existing.write_bytes(b"x" * 64)
    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=32,
    )

    assert monitor.check() is None
    with existing.open("ab") as handle:
        handle.write(b"y")

    violation = monitor.check()
    assert violation is not None
    assert violation.path == existing
    assert violation.observed_bytes == 65


def test_disabled_file_size_monitor_does_not_touch_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        wa,
        "_scan_writable_scopes",
        lambda _scopes, **_kwargs: pytest.fail(
            "disabled monitor scanned the filesystem"
        ),
    )
    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=0,
    )
    assert monitor.check() is None


def test_disk_reserve_detects_aggregate_many_small_file_exhaustion_without_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    free_values = iter((1_000, 899))
    disk_usage_calls: list[Path] = []

    def _disk_usage(path):
        disk_usage_calls.append(Path(path))
        return SimpleNamespace(free=next(free_values))

    monkeypatch.setattr(wa.shutil, "disk_usage", _disk_usage)
    monkeypatch.setattr(
        wa,
        "_scan_writable_scopes",
        lambda _scopes, **_kwargs: pytest.fail(
            "aggregate capacity guard performed a directory walk"
        ),
    )
    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=0,
        min_free_disk_bytes=900,
    )

    violation = monitor.check()

    assert violation == wa.DiskReserveViolation(tmp_path, 899, 900)
    assert disk_usage_calls == [tmp_path, tmp_path]


def test_disk_reserve_refuses_to_start_when_margin_is_already_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        wa.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=511),
    )

    with pytest.raises(wa.DiskReserveExceeded) as raised:
        wa.WritableFileSizeMonitor(
            (wa.WritableScope(root=tmp_path),),
            limit_bytes=0,
            min_free_disk_bytes=512,
        )

    assert raised.value.errno == wa.errno.ENOSPC
    assert raised.value.violation == wa.DiskReserveViolation(tmp_path, 511, 512)


def test_disk_reserve_checks_overlapping_scopes_once_per_poll(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    calls: list[Path] = []

    def _disk_usage(path):
        calls.append(Path(path))
        return SimpleNamespace(free=10_000)

    monkeypatch.setattr(wa.shutil, "disk_usage", _disk_usage)
    monitor = wa.WritableFileSizeMonitor(
        (
            wa.WritableScope(root=tmp_path),
            wa.WritableScope(root=nested),
        ),
        limit_bytes=0,
        min_free_disk_bytes=100,
    )
    assert monitor.check() is None

    # Both paths are on one st_dev, so initialization and the live check each
    # issue one capacity probe rather than one per overlapping ACL scope.
    assert calls == [tmp_path, tmp_path]


def test_disk_reserve_probe_failure_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def _disk_usage(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(free=10_000)
        raise OSError("capacity unavailable")

    monkeypatch.setattr(wa.shutil, "disk_usage", _disk_usage)
    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=0,
        min_free_disk_bytes=100,
    )

    with pytest.raises(OSError, match="capacity unavailable"):
        monitor.check()


def test_file_size_monitor_scan_failure_is_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=32,
    )

    def _fail(_scopes, **_kwargs):
        raise PermissionError("workspace traversal denied")

    monkeypatch.setattr(wa, "_scan_writable_scopes", _fail)
    with pytest.raises(PermissionError, match="traversal denied"):
        monitor.check()


def test_file_size_monitor_adapts_poll_interval_to_scan_cost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    moments = iter((0.0, 0.01, 1.0, 1.1))
    monkeypatch.setattr(wa.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(
        wa, "_scan_writable_scopes", lambda _scopes, **_kwargs: {},
    )

    monitor = wa.WritableFileSizeMonitor(
        (wa.WritableScope(root=tmp_path),),
        limit_bytes=32,
    )
    assert monitor.poll_seconds == wa._FILE_SIZE_MIN_POLL_SECONDS

    assert monitor.check() is None
    assert monitor.poll_seconds == pytest.approx(0.4)


def test_file_size_monitor_refuses_a_tree_too_slow_to_police(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    moments = iter((0.0, wa._FILE_SIZE_MAX_SCAN_SECONDS + 0.001))
    monkeypatch.setattr(wa.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(
        wa, "_scan_writable_scopes", lambda _scopes, **_kwargs: {},
    )

    with pytest.raises(wa.FileSizeMonitorUnavailable, match="scan took"):
        wa.WritableFileSizeMonitor(
            (wa.WritableScope(root=tmp_path),),
            limit_bytes=32,
        )


def test_large_overlapping_tree_is_scanned_once_without_busy_looping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    for directory_index in range(40):
        directory = data / f"partition-{directory_index:03d}"
        directory.mkdir(parents=True)
        for file_index in range(25):
            payload = b"x" * (33 if directory_index == file_index == 0 else 1)
            (directory / f"part-{file_index:03d}.bin").write_bytes(payload)

    real_scandir = wa.os.scandir
    calls: dict[str, int] = {}

    def _counting_scandir(path):
        key = wa._normalized_path(Path(path))
        calls[key] = calls.get(key, 0) + 1
        return real_scandir(path)

    monkeypatch.setattr(wa.os, "scandir", _counting_scandir)
    started = time.perf_counter()
    monitor = wa.WritableFileSizeMonitor(
        (
            wa.WritableScope(root=tmp_path),
            wa.WritableScope(root=data),  # intentionally overlapping
        ),
        limit_bytes=32,
    )
    elapsed = time.perf_counter() - started

    # Traversal covers the full tree, while retained baseline memory scales
    # only with grandfathered oversized files rather than all 1,000 entries.
    assert len(monitor._baseline) == 1
    assert calls
    assert max(calls.values()) == 1
    assert elapsed < 1.0
    assert monitor.poll_seconds >= wa._FILE_SIZE_MIN_POLL_SECONDS


def test_appcontainer_run_binds_monitor_to_every_effective_writable_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / ".sift" / "runs" / "current"
    run_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _FakeMonitor:
        def __init__(self, scopes, limit_bytes, min_free_disk_bytes=0):
            self.scopes = scopes
            captured["scopes"] = scopes
            captured["limit"] = limit_bytes
            captured["reserve"] = min_free_disk_bytes

    class _FakeProcess:
        def __init__(self) -> None:
            self._cleanup = lambda: None

        def close(self) -> None:
            self._cleanup()

    fake_process = _FakeProcess()
    monkeypatch.setattr(wa, "_require_windows", lambda: None)
    monkeypatch.setattr(wa, "_acquire_acl_mutex", lambda: (lambda: None))
    monkeypatch.setattr(
        wa,
        "create_appcontainer_profile",
        lambda _name: (object(), lambda: None),
    )
    monkeypatch.setattr(wa, "plan_acl_grants", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(wa, "create_job_object", lambda _limits: 123)
    monkeypatch.setattr(wa, "WritableFileSizeMonitor", _FakeMonitor)
    monkeypatch.setattr(
        wa,
        "spawn_in_appcontainer",
        lambda *_args, **kwargs: (
            captured.update({"spawn_monitor": kwargs["file_size_monitor"]})
            or fake_process
        ),
    )
    monkeypatch.setattr(
        wa,
        "_kernel32",
        SimpleNamespace(CloseHandle=lambda _handle: True),
        raising=False,
    )

    with wa.AppContainerRun(
        ["python.exe", "script.py"],
        tmp_path,
        run_dir,
        {},
        extra_read_paths=(),
        cpu_seconds=1,
        memory_bytes=2,
        max_processes=3,
        max_file_size_bytes=4096,
        min_free_disk_bytes=512 * 1024 * 1024,
    ):
        pass

    scopes = captured["scopes"]
    assert scopes == (
        wa.WritableScope(tmp_path, (tmp_path / ".sift",)),
        wa.WritableScope(run_dir),
    )
    assert captured["limit"] == 4096
    assert captured["reserve"] == 512 * 1024 * 1024
    assert captured["spawn_monitor"].scopes == scopes
