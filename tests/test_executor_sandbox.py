"""Integration tests for the executor's sandbox profile.

These exercise the real subprocess path — sandbox-exec spawning Rscript
— to confirm the ``(deny default)`` profile enforces what the design
requires: scripts can read the researcher's cwd and their R package
library, and they cannot read the rest of the home directory or write
outside the run scratch dir.

Portability notes (learned from the 2026-04-20 review):

- Some environments run pytest inside a harness that blocks nested
  ``sandbox-exec`` with ``sandbox_apply: Operation not permitted``.
  ``requires_sandbox_apply`` preflights this by trying a trivial
  ``sandbox-exec`` invocation before each test and skipping if the
  outer sandbox prevents it.

- Some environments mount HOME read-only, so we cannot ``write_text``
  into ``~/`` during test setup. The probes below therefore target
  files that already exist on every Mac and live OUTSIDE the narrowed
  sandbox allowlist (e.g. ``/Library/Keychains/System.keychain``), and
  write-block tests target paths the test process does not need to
  create first.

The pure-SBPL invariants (profile shape, no /private broadly, HOME
narrowly, etc.) live in ``test_executor_profile.py`` and run
unconditionally. This file is the belt-and-suspenders run-it-for-real
layer.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sift.env_detect import find_sandbox_exec
from sift.executor import run_script


_RSCRIPT = shutil.which("Rscript")


def _sandbox_apply_works() -> bool:
    """Return True iff ``sandbox-exec`` can actually apply a profile in
    the current environment. Nested sandbox harnesses return False.
    """
    exe = find_sandbox_exec()
    if exe is None:
        return False
    try:
        r = subprocess.run(
            [exe, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


requires_rscript = pytest.mark.skipif(
    _RSCRIPT is None,
    reason="Rscript not on PATH; sandbox integration tests need R.",
)

requires_sandbox_apply = pytest.mark.skipif(
    sys.platform != "darwin" or not _sandbox_apply_works(),
    reason=(
        "sandbox-exec cannot apply a profile in this environment "
        "(non-macOS or nested-sandbox harness)."
    ),
)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    """A minimal dataset so scripts have something to load from cwd."""
    path = tmp_path / "data.csv"
    path.write_text(
        "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n6,60\n"
        "7,70\n8,80\n9,90\n10,100\n11,110\n12,120\n"
    )
    return path


# ---------------------------------------------------------------------------
# Happy path — sandbox lets R do its job
# ---------------------------------------------------------------------------

@requires_sandbox_apply
@requires_rscript
def test_sandbox_allows_cwd_read_and_runtime_write(tmp_path: Path, tiny_csv: Path):
    """The profile must let R read data from cwd and emit a result payload."""
    code = r'''
df <- read.csv("data.csv")
sift$from_lm(lm(y ~ x, data = df), label = "ok")
'''
    r = run_script("R", code, tmp_path)
    assert r.ok, f"script failed: error={r.error}\nstderr={r.raw_stderr}"
    assert r.result_payloads
    # Helper emits the canonical descriptive name; legacy alias
    # ``linear_regression`` is accepted by the sanitizer for back-compat
    # with stored payloads but is no longer the freshly-emitted form.
    assert r.result_payloads[0]["type"] == "coefficient_table_with_fit_stats"
    assert r.result_payloads[0]["n"] == 12


@requires_sandbox_apply
@requires_rscript
def test_sandbox_canonicalizes_symlinked_workspace(tmp_path: Path):
    """SBPL grants must match the kernel-canonical workspace path."""
    real = tmp_path / "real-workspace"
    real.mkdir()
    (real / "data.csv").write_text(
        "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(1, 13)) + "\n",
        encoding="utf-8",
    )
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(real, target_is_directory=True)

    result = run_script(
        "R",
        'df <- read.csv("data.csv")\n'
        'sift$from_lm(lm(y ~ x, data=df), label="canonical-workspace")',
        alias,
    )
    assert result.ok, result.raw_stderr or result.error
    assert result.run_dir.is_relative_to(real.resolve())


# ---------------------------------------------------------------------------
# Security invariants — sandbox blocks what matters
# ---------------------------------------------------------------------------

# System files guaranteed to exist on macOS and OUTSIDE the narrowed
# sandbox allowlist (keychains were part of a broad ``/Library``, now
# excluded by the narrowed profile; system.log lives under
# ``/private/var/log`` which is no longer allowed).
_OUTSIDE_ALLOWLIST_PROBES = [
    "/Library/Keychains/System.keychain",
    "/private/var/log/system.log",
]


@requires_sandbox_apply
@requires_rscript
def test_sandbox_blocks_read_outside_allowlist(tmp_path: Path, tiny_csv: Path):
    """A read to a system path OUTSIDE the narrowed allowlist must fail.

    We probe a pre-existing file (no test-setup writes needed — the
    earlier HOME-setup version broke in environments where HOME is
    mounted read-only) and verify the script can't recover its
    contents.
    """
    target = next(
        (p for p in _OUTSIDE_ALLOWLIST_PROBES if Path(p).exists()),
        None,
    )
    if target is None:
        pytest.skip("no out-of-allowlist probe file present on this machine")

    code = (
        f'probe <- tryCatch(readLines("{target}", n = 1, warn = FALSE),\n'
        '  error = function(e) paste("DENIED:", conditionMessage(e)),\n'
        '  warning = function(w) paste("DENIED:", conditionMessage(w)))\n'
        'df <- data.frame(x = 1:12, y = (1:12) * 2)\n'
        'sift$from_lm(lm(y ~ x, data = df), '
        'label = paste0("probe=", substr(paste(probe, collapse="|"), 1, 80)))\n'
    )
    r = run_script("R", code, tmp_path)
    assert r.ok, f"executor failure: {r.error}"
    label = r.result_payloads[0]["label"]
    assert "DENIED" in label, (
        f"sandbox failed — out-of-allowlist read was permitted: {label!r}"
    )


@requires_sandbox_apply
@requires_rscript
def test_sandbox_blocks_home_dotfile_reads(tmp_path: Path, tiny_csv: Path):
    """Reads from ``~/.zshrc`` etc must fail — the canary for 'malicious
    script exfils user config files via HOME'.

    No setup: we probe existing dotfiles, no test writes to HOME.
    """
    home = Path.home()
    candidates = [home / ".zshrc", home / ".bashrc", home / ".profile"]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        pytest.skip("no standard dotfile in HOME to probe")

    code = (
        f'probe <- tryCatch(readLines("{target}", n = 1, warn = FALSE),\n'
        '  error = function(e) paste("DENIED:", conditionMessage(e)),\n'
        '  warning = function(w) paste("DENIED:", conditionMessage(w)))\n'
        'df <- data.frame(x = 1:12, y = (1:12) * 3)\n'
        'sift$from_lm(lm(y ~ x, data = df), '
        'label = paste0("home-probe=", substr(paste(probe, collapse="|"), 1, 80)))\n'
    )
    r = run_script("R", code, tmp_path)
    assert r.ok
    label = r.result_payloads[0]["label"]
    assert "DENIED" in label, f"home dotfile read was NOT denied: label={label!r}"


@requires_sandbox_apply
@requires_rscript
def test_sandbox_blocks_write_outside_run_dir(tmp_path: Path, tiny_csv: Path):
    """Writes to a path outside the run scratch dir and allowed temp
    trees must fail.

    We target ``/Library/Caches``: present on every Mac, world-writable
    by the user OUTSIDE the sandbox (so a non-sandbox baseline would
    succeed), but NOT in the sandbox write allowlist. If the file
    appears, the sandbox didn't enforce the write boundary.

    If ``/Library/Caches`` isn't user-writable in this environment
    (unlikely but possible), we skip — there's no portable victim
    that's both out-of-allowlist and guaranteed-writable-outside-
    sandbox across every macOS configuration.
    """
    import uuid

    caches = Path("/Library/Caches")
    if not caches.is_dir():
        pytest.skip("/Library/Caches not present")
    # Probe whether the test process itself can write there; if not,
    # the test can't distinguish sandbox-denied from permission-denied.
    probe = caches / f".sift_test_permcheck_{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("x")
        probe.unlink()
    except OSError:
        pytest.skip("/Library/Caches not user-writable here")

    victim = caches / f".sift_test_victim_{uuid.uuid4().hex[:8]}.txt"
    if victim.exists():
        victim.unlink()
    try:
        code = (
            f'tryCatch(writeLines("pwned", "{victim}"), '
            'error = function(e) e, warning = function(w) w)\n'
            'df <- data.frame(x = 1:12, y = (1:12) * 4)\n'
            'sift$from_lm(lm(y ~ x, data = df), label = "write-probe")\n'
        )
        r = run_script("R", code, tmp_path)
        assert not victim.exists(), (
            f"sandbox failed — script wrote {victim} outside its "
            f"scratch dir (r.error={r.error})"
        )
    finally:
        try:
            victim.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Missing-sandbox preflight — pure Python, no sandbox-exec needed
# ---------------------------------------------------------------------------

def test_run_script_refuses_without_sandbox(tmp_path: Path):
    """If sandbox-exec is unavailable (e.g. Linux/Windows), run_script
    must refuse rather than fall through to an unsandboxed subprocess.
    """
    from sift import env_detect, executor

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec=None,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "sandbox" in (r.error or "").lower()


def test_run_script_refuses_when_sandbox_baseline_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Second gate on darwin: ``sandbox_exec`` binary PRESENT is not
    the same as sandbox-exec being able to apply a minimal profile.
    ``sift --doctor`` (``_sandbox_exec_report``) has always
    distinguished these two failure shapes via
    ``sandbox_baseline_result``; this preflight used to only check
    the first (binary presence), so a researcher could see "sandbox:
    blocked, baseline check fails" from ``--doctor`` and then have a
    script submission still attempt a real (doomed) sandboxed run
    instead of refusing cleanly with the same diagnosis. Forcing the
    baseline result to fail here, with a real (fake-path) binary
    present, must produce the SAME kind of refusal the doctor would
    show — not a confusing subprocess failure further down."""
    from sift import env_detect, executor

    monkeypatch.setattr(executor.sys, "platform", "darwin")
    monkeypatch.setattr(
        env_detect, "sandbox_baseline_result",
        lambda: (False, "sandbox-exec rejected a minimal allow-default profile (exit 1)."),
    )

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None, python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "cannot apply a minimal profile" in (r.error or "")
    assert "rejected a minimal allow-default profile" in (r.error or "")
    assert "unsandboxed" in (r.error or "").lower()


def test_run_script_refuses_when_bwrap_baseline_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Same second gate as the darwin test above, for the Linux/bwrap
    backend: ``bwrap`` binary present but ``bwrap_baseline_result``
    reports it can't actually apply a minimal sandbox (nested-
    namespace harness, kernel without unprivileged user namespaces,
    AppArmor policy blocking it, etc.). Must refuse with that
    diagnosis rather than attempt a real (doomed) bwrap invocation."""
    from sift import env_detect, executor

    monkeypatch.setattr(executor.sys, "platform", "linux")
    monkeypatch.setattr(
        env_detect, "bwrap_baseline_result",
        lambda: (False, "bwrap rejected a minimal read-only-root profile (exit 1)."),
    )

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None, python=None,
        sandbox_exec=None,
        bwrap="/usr/bin/bwrap",
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "cannot apply a minimal sandbox" in (r.error or "")
    assert "rejected a minimal read-only-root profile" in (r.error or "")
    assert "unsandboxed" in (r.error or "").lower()


def test_run_script_proceeds_past_darwin_baseline_gate_when_healthy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Negative control: when the baseline check reports healthy, the
    new second gate must not itself block anything — the preflight
    should proceed past the backend checks entirely (reaching the
    interpreter-missing check next, since ``env.r`` is None here),
    not get stuck reporting a false baseline failure."""
    from sift import env_detect, executor

    monkeypatch.setattr(executor.sys, "platform", "darwin")
    monkeypatch.setattr(
        env_detect, "sandbox_baseline_result", lambda: (True, ""),
    )

    fake_env = env_detect.Environment(
        r=None, stata=None, python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    # Must have progressed to the "R not found" check, not stalled on
    # a (non-existent, since baseline is healthy) backend refusal.
    assert "cannot apply a minimal profile" not in (r.error or "")
    assert "rscript not found" in (r.error or "").lower()


def test_run_script_refuses_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Exercise the "API surface not present" AppContainer gate.

    Native security behavior requires a Windows kernel, so the executor also
    requires the live health probe. With ``appcontainer_support``
    left at its default False, as it would be on pre-Windows-8 or an
    otherwise broken install, ``run_script`` must still refuse to
    run scripts unsandboxed rather than falling through to a plain
    subprocess. See ``test_run_script_uses_appcontainer_when_probe_passes``
    below for the "backend present and verified" counterpart.

    This test locks that refusal in directly: force
    ``executor.sys.platform`` to a Windows-shaped value regardless of
    what platform CI/dev actually runs on, and confirm ``run_script``
    still refuses with a message that names Windows specifically
    (not a generic, unhelpful "no backend" string) -- a future
    refactor of the platform-dispatch block that silently drops this
    branch would otherwise ship an unsandboxed Windows execution path
    with no test catching it.
    """
    from sift import env_detect, executor

    monkeypatch.setattr(executor.sys, "platform", "win32")

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec="/usr/bin/sandbox-exec",  # present but irrelevant on win32
        bwrap="/usr/bin/bwrap",  # present but irrelevant on win32
        appcontainer_support=False,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "sandbox" in (r.error or "").lower()
    assert "windows" in (r.error or "").lower()
    # Confirms the refusal is genuinely platform-driven, not an
    # accidental fallthrough of the darwin/linux missing-backend
    # branches (both of which would find their respective backend
    # present in this fake_env and pass preflight if the platform
    # dispatch were broken).
    assert "win32" in (r.error or "")


def test_run_script_refuses_when_appcontainer_probe_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """The second gate: ``appcontainer_support=True`` (API surface
    present) but the live empirical health probe reports failure.
    ``run_script`` must STILL refuse -- "the API exists" is
    deliberately never sufficient on its own for a backend whose
    ctypes bindings have never been exercised against a real Windows
    kernel while being written (see ``win_appcontainer``'s module
    docstring). Without this second gate, a subtly wrong
    ``SECURITY_CAPABILITIES`` struct could produce a process that
    looks sandboxed while actually running with full access -- this
    test pins that the executor refuses rather than trusts an
    unverified confinement boundary."""
    from sift import env_detect, executor

    monkeypatch.setattr(executor.sys, "platform", "win32")
    monkeypatch.setattr(
        env_detect, "_APPCONTAINER_PROBE_CACHE",
        (False, "CRITICAL: did NOT deny a file read outside its granted paths"),
    )

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None, python=None,
        sandbox_exec=None, bwrap=None,
        appcontainer_support=True,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "did NOT deny" in (r.error or "")
    assert "unsandboxed" in (r.error or "").lower()


def test_run_script_uses_appcontainer_when_probe_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Both gates pass -- confirms ``run_script`` actually reaches
    and calls into ``win_appcontainer.AppContainerRun`` (rather than
    silently falling through to the darwin/linux ``subprocess.Popen``
    + argv-wrapper path) when it should.

    This test is deliberately scoped to WIRING, not confinement: it
    stubs out ``AppContainerRun`` itself with a fake that records how
    it was called and returns a fake Popen-shaped process, because
    the real ``win_appcontainer`` module's OS-calling functions raise
    immediately off Windows (see ``_require_windows``) and this
    sandbox has no Windows kernel to run the real thing against. What
    IS verified, and IS meaningfully testable from here: that
    ``run_script``'s platform dispatch actually constructs
    ``AppContainerRun`` with the right arguments and a BARE command
    (no ``sandbox-exec``/``bwrap`` argv prefix -- Windows confinement
    is applied via ``CreateProcess`` flags, not an external wrapper
    binary), that it calls ``communicate()`` on the object
    ``AppContainerRun.__enter__()`` returns, and that it tears the
    context down via ``__exit__`` afterward. The confinement itself
    (does the AppContainer actually deny what it should) is the live
    probe's job, pinned by the doctor/executor "probe fails" tests
    above -- this test would stay green even if the real ctypes layer
    were subtly wrong, which is exactly why it's not a substitute for
    that probe, only a check that the executor calls it.
    """
    from sift import env_detect, executor

    calls: list[dict] = []

    class _FakeAppContainerProcess:
        def __init__(self) -> None:
            self.pid = 4242
            self.returncode: int | None = None

        def communicate(self, timeout=None):
            self.returncode = 0
            return "", ""

        def kill(self) -> None:  # pragma: no cover — not exercised here
            pass

    class _FakeAppContainerRun:
        def __init__(
            self, cmd, cwd, run_dir, env, extra_read_paths,
            cpu_seconds, memory_bytes, max_processes, max_file_size_bytes,
            min_free_disk_bytes,
        ) -> None:
            calls.append({
                "cmd": list(cmd), "cwd": cwd, "run_dir": run_dir,
                "max_file_size_bytes": max_file_size_bytes,
                "min_free_disk_bytes": min_free_disk_bytes,
            })
            self._proc = _FakeAppContainerProcess()

        def __enter__(self):
            return self._proc

        def __exit__(self, exc_type, exc, tb) -> bool:
            calls.append({"exited": True})
            return False

    import sift.win_appcontainer as win_appcontainer_module
    monkeypatch.setattr(win_appcontainer_module, "AppContainerRun", _FakeAppContainerRun)
    monkeypatch.setattr(executor.sys, "platform", "win32")
    monkeypatch.setattr(env_detect, "_APPCONTAINER_PROBE_CACHE", (True, ""))

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None, python=None,
        sandbox_exec=None, bwrap=None,
        appcontainer_support=True,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)

    assert len(calls) == 2  # one construction + one __exit__
    launched = calls[0]
    assert launched["cmd"][0] == "/bin/true"
    # No sandbox-exec/bwrap wrapper prepended -- Windows confinement
    # is a CreateProcess-flag thing, not an argv-prefix thing.
    assert "sandbox-exec" not in launched["cmd"]
    assert "bwrap" not in launched["cmd"]
    assert launched["max_file_size_bytes"] == 2 * 1024 * 1024 * 1024
    assert launched["min_free_disk_bytes"] == 512 * 1024 * 1024
    assert calls[1] == {"exited": True}
    # The fake process never wrote a result.json, so this run reports
    # failure -- that's expected and fine; this test's assertions are
    # entirely about the dispatch/wiring above, not the outcome.
    assert not r.ok


def test_run_script_kills_process_on_unexpected_communicate_exception() -> None:
    """POSIX-specific regression: an unexpected (non-timeout)
    exception out of ``proc.communicate()`` must still kill the
    subprocess before propagating, exactly like the TimeoutExpired
    path does two branches above it.

    Before this fix, the ``except Exception:`` branch only tore down
    ``_appcontainer_ctx`` (always ``None`` on darwin/linux) and
    re-raised -- it never killed ``proc``. On Windows that's masked
    because closing the Job Object handle (inside that same
    teardown) kills every process in the job automatically; POSIX has
    no equivalent, so the child interpreter (and anything it forked
    via ``parallel::makeCluster`` / ``multiprocessing.Pool``) would be
    orphaned, still running against the researcher's data, until it
    exited on its own.

    Simulated by monkeypatching ``subprocess.Popen`` to return a fake
    process whose ``.communicate()`` raises ``OSError`` (never
    ``TimeoutExpired``) and whose ``.pid`` doesn't correspond to a
    real process -- so ``os.getpgid`` genuinely raises
    ``ProcessLookupError`` and the code must fall back to
    ``proc.kill()``, exactly as the real fallback path is written to
    do. Recording that ``.kill()`` was actually called is the
    regression check.
    """
    import subprocess as _subprocess

    from sift import env_detect, executor

    class _Boom(OSError):
        pass

    kill_calls: list[int] = []

    class _FakeProc:
        def __init__(self) -> None:
            # A pid astronomically unlikely to be a real process in
            # this test run, so the real ``os.getpgid`` call in the
            # fallback path raises ProcessLookupError for real
            # rather than needing its own mock.
            self.pid = 2_000_000_000
            self.returncode = None

        def communicate(self, timeout=None):
            raise _Boom("simulated pipe failure")

        def kill(self) -> None:
            kill_calls.append(self.pid)

    def _fake_popen(*args, **kwargs):
        return _FakeProc()

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(executor.sys, "platform", "darwin")
        monkeypatch.setattr(
            env_detect, "sandbox_baseline_result", lambda: (True, ""),
        )
        monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

        fake_env = env_detect.Environment(
            r=env_detect.Tool(name="R", binary="/bin/true"),
            stata=None, python=None,
            sandbox_exec="/usr/bin/sandbox-exec",
        )
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(_Boom):
                executor.run_script(
                    "R", "cat('hi')", Path(td), env=fake_env,
                )
        assert kill_calls == [2_000_000_000], (
            "proc.kill() must be called on the unexpected-exception "
            "path, the same as on the timeout path"
        )
    finally:
        monkeypatch.undo()


def test_appcontainer_error_is_missing_interpreter_classification():
    """Pure-Python unit test of ``AppContainerError.is_missing_interpreter``
    -- no ctypes/Windows involved, so this runs on any platform. Pins
    the exact classification the executor's exception handling
    finding: "Windows FileNotFoundError misreporting") relies on:
    only a ``CreateProcessW`` failure with a file/path-not-found code
    counts as "the interpreter binary is missing"; every other
    AppContainer failure (ACL setup, job object, profile creation, or
    ANY OTHER CreateProcessW error code such as access-denied) is a
    genuine sandbox-plumbing bug, not a missing interpreter.
    """
    from sift.win_appcontainer import (
        ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND, AppContainerError,
    )

    assert AppContainerError("CreateProcessW", ERROR_FILE_NOT_FOUND).is_missing_interpreter()
    assert AppContainerError("CreateProcessW", ERROR_PATH_NOT_FOUND).is_missing_interpreter()
    # Same op, different code (e.g. ERROR_ACCESS_DENIED = 5) -- NOT
    # a missing interpreter.
    assert not AppContainerError("CreateProcessW", 5).is_missing_interpreter()
    # File-not-found code, but from a DIFFERENT operation -- also not
    # classified as a missing interpreter (the code only means what
    # this predicate says when it comes from CreateProcessW itself;
    # e.g. AssignProcessToJobObject failing with code 2 would be a
    # bizarre, genuinely-unexpected plumbing failure worth a bug
    # report, not "go install Python").
    assert not AppContainerError("AssignProcessToJobObject", ERROR_FILE_NOT_FOUND).is_missing_interpreter()
    assert not AppContainerError("CreateAppContainerProfile", 0x80070005).is_missing_interpreter()


def test_run_script_reports_missing_interpreter_plainly_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """The actual fix: when ``AppContainerRun.__enter__()`` raises
    ``AppContainerError("CreateProcessW", ERROR_FILE_NOT_FOUND)`` --
    the Windows-side equivalent of ``subprocess.Popen`` raising a
    bare ``FileNotFoundError`` on macOS/Linux for the exact same
    underlying condition (interpreter binary doesn't exist) --
    ``run_script`` must report the same friendly, actionable
    "interpreter not found" message the POSIX branch gives, NOT the
    generic "could not launch... please report this as a bug"
    message. Before this fix, EVERY ``AppContainerError`` (this one
    included) got the generic message, which sent a Windows
    researcher whose Python was simply uninstalled down the wrong
    troubleshooting path (filing a bug report) instead of the right
    one (reinstalling Python).
    """
    from sift import env_detect, executor
    from sift.win_appcontainer import AppContainerError

    class _FailingAppContainerRun:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise AppContainerError("CreateProcessW", 2)  # ERROR_FILE_NOT_FOUND

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    import sift.win_appcontainer as win_appcontainer_module
    monkeypatch.setattr(win_appcontainer_module, "AppContainerRun", _FailingAppContainerRun)
    monkeypatch.setattr(executor.sys, "platform", "win32")
    monkeypatch.setattr(env_detect, "_APPCONTAINER_PROBE_CACHE", (True, ""))

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/nonexistent/R.exe"),
        stata=None, python=None,
        sandbox_exec=None, bwrap=None,
        appcontainer_support=True,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)

    assert not r.ok
    assert r.error is not None
    assert "interpreter not found" in r.error.lower(), (
        f"expected the friendly missing-interpreter message, got: {r.error!r}"
    )
    assert "report this as a bug" not in r.error.lower(), (
        "a missing interpreter is a researcher-fixable condition, not "
        "a Sift bug -- must not tell them to file one"
    )


def test_run_script_still_reports_genuine_appcontainer_bug_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    """Negative control for the fix above: a REAL AppContainer/Job-
    Object plumbing failure (e.g. ``AssignProcessToJobObject``
    failing) must still get the "please report this as a bug"
    message -- the new classification must not accidentally swallow
    genuine Sift-side bugs into a falsely-reassuring "just install
    Python" message.
    """
    from sift import env_detect, executor
    from sift.win_appcontainer import AppContainerError

    class _FailingAppContainerRun:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise AppContainerError("AssignProcessToJobObject", 5)  # ERROR_ACCESS_DENIED

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    import sift.win_appcontainer as win_appcontainer_module
    monkeypatch.setattr(win_appcontainer_module, "AppContainerRun", _FailingAppContainerRun)
    monkeypatch.setattr(executor.sys, "platform", "win32")
    monkeypatch.setattr(env_detect, "_APPCONTAINER_PROBE_CACHE", (True, ""))

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None, python=None,
        sandbox_exec=None, bwrap=None,
        appcontainer_support=True,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)

    assert not r.ok
    assert r.error is not None
    assert "report this as a bug" in r.error.lower()
    assert "interpreter not found" not in r.error.lower()
