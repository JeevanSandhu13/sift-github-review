"""Tests for the ``sift --doctor`` health check.

The doctor exists to close the researcher-side diagnostic gap: when
Sift can't run a script, the researcher needs to see *what's broken*
and *how to fix it* without depending on the chat model to speculate.

The invariants here:

  * Each runtime gets one report; status is the worst applicable
    level (``ok`` / ``warning`` / ``unavailable`` / ``blocked``).
  * Apple's xcrun-stub failure shape — interpreter present-but-
    sandbox-rejected — is identified specifically and the advice
    names the fix (install a real Python via Homebrew or
    python.org), not just "python3 broken".
  * "blocked" on the report maps to a non-zero CLI exit code so a
    shell-init wrapper can gate the .app launch on a clean check.
  * The report serialises cleanly to a JSON-safe dict for the
    bridge method ``SiftBridge.doctor_report``.
"""

from __future__ import annotations

import sys

import pytest

from sift import env_detect as _env_detect_mod
from sift.env_detect import (
    Environment,
    Tool,
    _SANDBOX_PROBE_CACHE,
)
from sift.doctor import (
    DoctorReport,
    RuntimeReport,
    main_cli,
    render_report_text,
    run_doctor,
)


@pytest.fixture(autouse=True)
def _clear_probe_caches():
    """Reset all three backend baseline/probe caches (macOS
    sandbox-exec, Linux bwrap, Windows AppContainer) before each
    test. They survive across tests by default; null them explicitly
    so test ordering can't masquerade as a cached value."""
    _SANDBOX_PROBE_CACHE.clear()
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None
    _env_detect_mod._BWRAP_BASELINE_CACHE = None
    _env_detect_mod._APPCONTAINER_PROBE_CACHE = None
    yield
    _SANDBOX_PROBE_CACHE.clear()
    _env_detect_mod._SANDBOX_BASELINE_CACHE = None
    _env_detect_mod._BWRAP_BASELINE_CACHE = None
    _env_detect_mod._APPCONTAINER_PROBE_CACHE = None


# The doctor's sandbox row is named after whichever backend the
# CURRENT platform actually uses ("sandbox-exec" on macOS, "bwrap" on
# Linux) — see ``_sandbox_report``'s platform branch in doctor.py.
# Tests that need to find "the sandbox row" regardless of platform
# use this tuple; tests that exercise backend-specific failure shapes
# (missing / baseline-fails) branch on ``sys.platform`` directly so
# the same test file proves the real behaviour on whichever platform
# actually runs it, instead of hardcoding a macOS assumption that
# would silently stop testing anything real once CI runs on Linux.
_SANDBOX_ROW_NAMES = ("sandbox-exec", "bwrap", "appcontainer")


def _env(
    python=None, r=None, stata=None,
    sandbox="/usr/bin/sandbox-exec", bwrap="/usr/bin/bwrap",
    appcontainer_support=False,
):
    """Both backend fields are independently settable and default to
    "healthy" so tests that aren't specifically about sandbox-backend
    detection (Stata-optional, CLI exit code, etc.) get a sandbox row
    that reports healthy on WHICHEVER platform actually runs this
    file, without needing every caller to know which field the
    current platform's ``_sandbox_report`` branch actually reads.

    ``appcontainer_support`` defaults to False (unlike ``sandbox``/
    ``bwrap``, which default "present") because a True value alone
    isn't enough to make the win32 row report healthy anyway — the
    live probe (``env_detect.appcontainer_probe_result``) still has
    to pass, and that probe can't run for real off Windows. Tests
    that specifically exercise the win32 branch monkeypatch the probe
    cache directly (see
    ``test_doctor_marks_appcontainer_ok_when_probe_passes``).
    """
    return Environment(
        python=python, r=r, stata=stata,
        sandbox_exec=sandbox, bwrap=bwrap,
        appcontainer_support=appcontainer_support,
    )


# ---------------------------------------------------------------------------
# Per-runtime classification
# ---------------------------------------------------------------------------

def test_doctor_marks_sandbox_blocked_when_missing():
    """No sandbox backend for THIS platform → no scripts can run,
    period. The whole report is blocked even if every interpreter is
    present. Branches on the real ``sys.platform`` so this test
    exercises the actual backend the doctor would pick in production
    on whichever machine runs the suite."""
    if sys.platform == "darwin":
        env = _env(
            python=Tool(name="Python", binary="/p", version="Python 3.12"),
            sandbox=None,
        )
    else:
        env = _env(
            python=Tool(name="Python", binary="/p", version="Python 3.12"),
            bwrap=None,
        )
    report = run_doctor(env)
    sandbox = next(
        r for r in report.runtimes if r.runtime in _SANDBOX_ROW_NAMES
    )
    assert sandbox.status == "blocked"
    assert report.blocked is True


def test_doctor_blocks_when_posix_resource_launcher_is_missing(monkeypatch):
    import sift.doctor as doctor_module

    monkeypatch.setattr(doctor_module.sys, "platform", "linux")
    monkeypatch.setattr(doctor_module.Path, "is_file", lambda _path: False)
    monkeypatch.setattr(
        doctor_module, "_sandbox_report",
        lambda _env: RuntimeReport("bwrap", "ok", "healthy"),
    )
    report = run_doctor(
        _env(python=Tool(name="Python", binary="/p", version="Python 3.12"))
    )

    limits = next(r for r in report.runtimes if r.runtime == "resource-limits")
    assert limits.status == "blocked"
    assert "Bash" in limits.detail
    assert report.blocked is True


def test_doctor_marks_sandbox_blocked_on_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise the "API surface not present" branch of the two-gate
    Windows design. ``appcontainer_support=False`` (matching a
    pre-Windows-8 or otherwise broken install) must produce a
    ``blocked`` row naming Windows/win32/sandbox explicitly, mirroring
    ``test_run_script_refuses_on_unsupported_platform`` in
    ``test_executor_sandbox.py``, which pins the same contract on the
    executor side. Both must independently keep refusing whenever the
    backend cannot be confirmed present and healthy. A doctor that
    disagreed with the executor (reported healthy right before the
    executor refuses the researcher's first script) would be worse
    than no doctor at all. See
    ``test_doctor_marks_appcontainer_ok_when_probe_passes`` below for
    the "backend present and verified" counterpart."""
    import sift.doctor as doctor_module
    monkeypatch.setattr(doctor_module.sys, "platform", "win32")

    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        sandbox="/usr/bin/sandbox-exec",
        bwrap="/usr/bin/bwrap",
        appcontainer_support=False,
    )
    report = run_doctor(env)
    sandbox = next(r for r in report.runtimes if r.runtime == "appcontainer")
    assert sandbox.status == "blocked"
    assert report.blocked is True
    assert "windows" in sandbox.detail.lower() or any(
        "windows" in tip.lower() for tip in sandbox.advice
    )


def test_doctor_marks_appcontainer_blocked_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed live probe blocks Windows even when the API is present.

    The probe verifies denied file and network access; API availability alone
    cannot establish that the isolation boundary is effective.
    """
    import sift.doctor as doctor_module
    monkeypatch.setattr(doctor_module.sys, "platform", "win32")
    monkeypatch.setattr(
        _env_detect_mod, "_APPCONTAINER_PROBE_CACHE",
        (False, "CRITICAL: did NOT deny a file read outside its granted paths"),
    )

    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        appcontainer_support=True,
    )
    report = run_doctor(env)
    sandbox = next(r for r in report.runtimes if r.runtime == "appcontainer")
    assert sandbox.status == "blocked"
    assert report.blocked is True
    assert "did NOT deny" in sandbox.detail


def test_doctor_marks_appcontainer_ok_when_probe_passes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Both gates pass: the API surface exists AND the live probe
    positively confirmed confinement holds on this machine. Only then
    does the win32 sandbox row report ``ok`` -- pins the "happy path"
    counterpart to the two blocked-branch tests above so a future
    refactor that accidentally makes this branch unreachable (or
    silently marks it ``ok`` without the probe having actually run)
    gets caught."""
    import sift.doctor as doctor_module
    monkeypatch.setattr(doctor_module.sys, "platform", "win32")
    monkeypatch.setattr(_env_detect_mod, "_APPCONTAINER_PROBE_CACHE", (True, ""))

    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        appcontainer_support=True,
    )
    report = run_doctor(env)
    sandbox = next(r for r in report.runtimes if r.runtime == "appcontainer")
    assert sandbox.status == "ok"


@pytest.mark.parametrize(
    ("platform_name", "cache_name", "cache_value", "runtime_name"),
    (
        (
            "darwin",
            "_SANDBOX_BASELINE_CACHE",
            (False, "sandbox-exec rejected a minimal allow-default profile."),
            "sandbox-exec",
        ),
        (
            "linux",
            "_BWRAP_BASELINE_CACHE",
            (False, "bwrap rejected a minimal read-only-root profile."),
            "bwrap",
        ),
    ),
)
def test_doctor_marks_sandbox_blocked_when_baseline_fails(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    cache_name: str,
    cache_value: tuple[bool, str],
    runtime_name: str,
) -> None:
    """Sandbox backend present but baseline profile won't apply.
    Distinct row from "not present" — the advice diverges (this is
    not an install problem). Without this distinction a researcher
    inside a nested sandbox harness would get pointed at Homebrew (or
    ``apt install bubblewrap``, which would be equally useless)."""
    import sift.doctor as doctor_module

    monkeypatch.setattr(doctor_module.sys, "platform", platform_name)
    monkeypatch.setattr(_env_detect_mod, cache_name, cache_value)
    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
    )
    report = run_doctor(env)
    sandbox = next(r for r in report.runtimes if r.runtime == runtime_name)
    assert sandbox.status == "blocked"
    assert "cannot apply a minimal" in sandbox.detail
    # The advice must name the actual cause (nested sandbox or
    # customised OS), not "install something".
    advice = " ".join(sandbox.advice)
    assert "sandbox" in advice.lower() or "namespace" in advice.lower()
    assert "install" not in advice.lower()


def test_doctor_marks_python_ok_when_healthy():
    py = Tool(
        name="Python", binary="/opt/homebrew/bin/python3",
        version="Python 3.12.0",
        missing_packages=(),
        optional_missing_packages=(),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "ok"
    assert "healthy" in python.detail.lower()


def test_doctor_marks_python_blocked_when_hard_packages_missing():
    """``pandas`` and ``numpy`` are hard-required by the runtime
    library itself. Without them every script crashes at runtime
    import. The doctor must mark this as ``blocked`` and the
    advice must name the install command."""
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("pandas", "numpy"),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "blocked"
    assert "pandas" in python.detail
    assert any("pip install" in tip for tip in python.advice)


def test_doctor_marks_bundled_python_ok_with_honest_note():
    """A ``Tool`` sourced from Sift's own vendored runtime
    (``bundled=True``) is reported exactly like any other healthy
    Python EXCEPT the detail text says so honestly -- a researcher
    reading doctor output shouldn't think they installed something
    they didn't."""
    py = Tool(
        name="Python", binary="/Applications/Sift.app/.../vendor_python/bin/python3",
        version="Python 3.12.0",
        missing_packages=(),
        optional_missing_packages=(),
        bundled=True,
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "ok"
    assert "bundled" in python.detail.lower()


def test_doctor_marks_non_bundled_python_ok_without_bundled_note():
    """The bundled note must NOT appear for a normal PATH-discovered
    interpreter -- ``bundled`` defaults to False and stays False for
    every non-bundled ``Tool(...)`` construction site."""
    py = Tool(
        name="Python", binary="/opt/homebrew/bin/python3",
        version="Python 3.12.0",
        missing_packages=(),
        optional_missing_packages=(),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "ok"
    assert "bundled" not in python.detail.lower()


def test_doctor_marks_bundled_python_warning_with_honest_note():
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("statsmodels", "scipy"),
        optional_missing_packages=("matplotlib",),
        bundled=True,
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "warning"
    assert "bundled" in python.detail.lower()


def test_doctor_hard_required_set_is_the_same_object_executor_enforces():
    """The doctor derives its hard-required packages from the executor.

    It previously classified "hard-required"
    Python packages by slicing ``env_detect._PYTHON_REQUIRED_PACKAGES
    [:2]`` -- a positional assumption with no mechanism keeping it in
    sync with ``executor._PYTHON_HARD_REQUIRED``, the set that
    ACTUALLY gates whether the executor refuses to run a Python
    script. The two happened to agree (both were {"pandas", "numpy"})
    purely because of list order in env_detect, not because anything
    enforced it -- reordering or extending the env_detect list would
    silently desync the doctor's "blocked" banner from what the
    executor actually blocks, which is precisely the "diagnostic
    lies about the real behavior" failure class this module exists
    to prevent.

    The fix: doctor.py now imports and uses
    ``executor._PYTHON_HARD_REQUIRED`` directly instead of re-
    deriving it. This test pins that there is exactly ONE set doing
    this job, not two that happen to match today.
    """
    from sift.doctor import _PYTHON_HARD_REQUIRED as doctor_hard_required
    from sift.executor import _PYTHON_HARD_REQUIRED as executor_hard_required

    assert doctor_hard_required is executor_hard_required, (
        "doctor.py must import executor.py's hard-required set "
        "directly, not maintain an independently-derived copy"
    )


def test_doctor_hard_soft_split_survives_reordering_the_env_detect_list():
    """Direct proof the desync bug is closed: reorder (and extend)
    ``env_detect._PYTHON_REQUIRED_PACKAGES`` so a positional [:2]
    slice would now pick the WRONG two packages as "hard" --
    ``numpy``/``statsmodels`` instead of ``pandas``/``numpy`` -- and
    confirm the doctor's classification is completely unaffected,
    because it no longer reads that list's order at all.
    """
    import sift.env_detect as env_detect_mod

    original = env_detect_mod._PYTHON_REQUIRED_PACKAGES
    env_detect_mod._PYTHON_REQUIRED_PACKAGES = (
        "scipy", "numpy", "statsmodels", "pandas",
    )
    try:
        py = Tool(
            name="Python", binary="/p", version="Python 3.12",
            missing_packages=("pandas", "statsmodels"),
        )
        report = run_doctor(_env(python=py))
        python = next(r for r in report.runtimes if r.runtime == "Python")
        # pandas is still genuinely hard-required regardless of
        # where env_detect's list now puts it -- must still block.
        assert python.status == "blocked", (
            "pandas missing must still block even after the "
            "env_detect list was reordered so a positional slice "
            "would have missed it"
        )
        assert "pandas" in python.detail
    finally:
        env_detect_mod._PYTHON_REQUIRED_PACKAGES = original


def test_doctor_marks_python_warning_for_soft_missing_packages():
    """``statsmodels`` / ``scipy`` missing doesn't block all runs —
    descriptive helpers still work. Status is ``warning``, with
    advice on how to enable the missing helpers."""
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("statsmodels", "scipy"),
        optional_missing_packages=("matplotlib",),
    )
    report = run_doctor(_env(python=py))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "warning"
    assert "statsmodels" in python.detail
    assert "matplotlib" in python.detail


# ---------------------------------------------------------------------------
# Apple-stub failure shape — present-but-sandbox-rejected
# ---------------------------------------------------------------------------

def test_doctor_identifies_apple_xcrun_stub_rejection():
    """The exact bug class this branch is fixing: ``find_python``
    returned None (so ``env.python`` is None) NOT because no python3
    exists but because the sandbox probe rejected the candidates.
    The doctor must distinguish this from "no python3" and surface
    the fix (install a real Python)."""
    _SANDBOX_PROBE_CACHE["/usr/bin/python3"] = (
        False,
        "xcrun: error: unable to load libxcrun ... blocked open()",
    )
    report = run_doctor(_env(python=None))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert python.status == "blocked"
    # Detail names the actual problem — not "not found on PATH".
    assert "/usr/bin/python3" in python.detail
    assert "sandbox" in python.detail.lower()
    # Advice points at the fix: Homebrew or python.org, NOT
    # "reinstall python3" or "fix your PATH".
    advice_text = " ".join(python.advice)
    assert "Homebrew" in advice_text or "brew install" in advice_text
    assert "python.org" in advice_text
    # The rejected-candidates list carries the binary + stderr tail
    # separately so the UI banner can show it expandably.
    assert any(path == "/usr/bin/python3"
               for path, _ in report.rejected_python_candidates)


def test_doctor_distinguishes_no_python3_from_rejected_python3():
    """When the cache is empty and python is None, the message is
    "not found on PATH" — different fix path from "rejected by
    sandbox". Confusing the two is exactly what the model did in
    the original incident."""
    report = run_doctor(_env(python=None))
    python = next(r for r in report.runtimes if r.runtime == "Python")
    assert "not found on PATH" in python.detail


# ---------------------------------------------------------------------------
# Optional runtimes — absence is unavailable, not a fault
# ---------------------------------------------------------------------------

def test_doctor_missing_stata_does_not_block_overall_report(monkeypatch):
    """Stata is optional. A researcher who only writes Python or R
    must not have ``--doctor`` exit non-zero just because Stata
    isn't installed. Per-language status reads ``unavailable`` (Stata
    scripts will fail), but the top-level ``blocked`` flag is False
    so long as some other language is usable."""
    from sift import doctor as _doctor
    monkeypatch.setattr(
        _doctor, "_sandbox_report",
        lambda _env: RuntimeReport("sandbox", "ok", "healthy"),
    )
    env = _env(python=Tool(name="Python", binary="/p", version="Python 3.12"))
    report = run_doctor(env)
    stata = next(r for r in report.runtimes if r.runtime == "Stata")
    # Per-language: Stata scripts are unavailable, but this is not a broken
    # Sift installation and must not produce a startup warning.
    assert stata.status == "unavailable"
    assert ".dta" in stata.detail
    assert "still open" in stata.detail
    # Overall: Python is usable, so Sift is not blocked.
    assert report.blocked is False


def test_doctor_blocks_only_when_every_language_is_blocked(monkeypatch):
    """``blocked`` on the top-level report means script execution
    will fail. One working language + one missing one is NOT
    blocked — the working language is still usable."""
    from sift import doctor as _doctor
    monkeypatch.setattr(
        _doctor, "_sandbox_report",
        lambda _env: RuntimeReport("sandbox", "ok", "healthy"),
    )
    env = _env(
        python=Tool(name="Python", binary="/p", version="Python 3.12"),
        # R and Stata absent.
    )
    report = run_doctor(env)
    assert report.blocked is False


def test_doctor_blocked_when_no_language_is_usable():
    """No interpreters at all → blocked. Researcher cannot run
    anything regardless of which language they pick."""
    report = run_doctor(_env())
    assert report.blocked is True


# ---------------------------------------------------------------------------
# Rendering — terminal output and CLI exit code
# ---------------------------------------------------------------------------

def test_render_text_includes_status_glyphs_and_advice():
    py = Tool(
        name="Python", binary="/p", version="Python 3.12",
        missing_packages=("pandas",),
    )
    text = render_report_text(run_doctor(_env(python=py)))
    assert "[fail]" in text  # blocked status uses [fail]
    # Advice line is indented under the runtime line.
    assert "    ->" in text


def test_main_cli_returns_nonzero_when_blocked(capsys):
    """``sift --doctor`` must exit non-zero on blocked so shell-init
    wrappers can gate launch. Cleaning the cache + clearing the
    detected env lets us force ``blocked=True`` deterministically."""
    from sift import doctor as _doctor
    # Force the report to a blocked state by stubbing detect_environment.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(_doctor, "detect_environment",
                       lambda: Environment(python=None, r=None, stata=None,
                                           sandbox_exec=None))
        rc = main_cli()
    finally:
        monkey.undo()
    assert rc != 0
    out = capsys.readouterr().out
    assert "BLOCKED" in out


def test_main_cli_returns_zero_when_healthy(capsys):
    from sift import doctor as _doctor
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(
            _doctor, "detect_environment",
            lambda: Environment(
                python=Tool(name="Python", binary="/p",
                            version="Python 3.12.0"),
                r=None, stata=None,
                sandbox_exec="/usr/bin/sandbox-exec",
                bwrap="/usr/bin/bwrap",
            ),
        )
        monkey.setattr(
            _doctor, "_sandbox_report",
            lambda _env: RuntimeReport("sandbox", "ok", "healthy"),
        )
        rc = main_cli()
    finally:
        monkey.undo()
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


# ---------------------------------------------------------------------------
# Bridge wiring — UI-facing JSON shape
# ---------------------------------------------------------------------------

def test_bridge_doctor_report_returns_json_safe_shape():
    """``SiftBridge.doctor_report`` must return a primitive-only dict
    so pywebview's JSON serialiser doesn't choke on the
    dataclasses. Asserts the exact keys the UI banner will consume."""
    from sift.ui import SiftBridge
    bridge = SiftBridge(cwd=None)
    payload = bridge.doctor_report()
    assert set(payload.keys()) >= {
        "blocked", "runtimes", "rejected_python_candidates",
    }
    assert isinstance(payload["blocked"], bool)
    assert isinstance(payload["runtimes"], list)
    for r in payload["runtimes"]:
        assert set(r.keys()) >= {"runtime", "status", "detail", "advice"}
        assert isinstance(r["advice"], list)
