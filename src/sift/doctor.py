"""Researcher-facing health check for the Sift runtime environment.

Why this module exists: the bug class this whole branch is addressing
(Apple xcrun stub fails inside the sandbox; every script silently
dies; the model speculates) is fundamentally a diagnostic-gap
problem. With the sandbox probe in ``env_detect``, the gap is closed
at detection time — Sift knows which interpreters work — but the
researcher still has no way to see that information without
submitting a script and watching it fail.

The doctor closes the researcher-side gap. It runs the same probes
the executor relies on, prints a short status per runtime, and
when something is wrong it names the *fix* (install Homebrew Python,
allow openssl.cnf, etc.) rather than the failure mode.

Two surfaces consume it:

  * ``sift --doctor`` (CLI) — for terminal launches and incident
    triage. Returns a non-zero exit code if any blocking issue
    exists, so a shell-init wrapper can refuse to launch the UI
    until the environment is healthy.
  * The UI banner — same data structure, rendered as a chat-side
    alert when ``Python`` (or whichever runtime the researcher's
    script needs) is unhealthy. Without this, the chat UI accepts
    the script, the executor refuses it, and the model paraphrases
    the refusal back to the researcher with no actionable next
    step.

Phase-safe by construction: every field this module produces is an
interpreter path, version string, package name, or launcher-stderr
tail. None of them touch researcher data, so the report can be
rendered anywhere (terminal stdout, UI banner, log file) without
the redaction posture the executor's runtime stderr needs.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sift.env_detect import (
    Environment,
    detect_environment,
    python_sandbox_probe_results,
)

# Single source of truth for "which Python packages actually gate
# script execution" -- imported from ``executor`` rather than
# re-derived here. This report used to classify hard-vs-soft by
# slicing ``env_detect._PYTHON_REQUIRED_PACKAGES[:2]``, a positional
# assumption that happened to match ``executor._PYTHON_HARD_REQUIRED``
# today but had no mechanism keeping the two in sync: reordering or
# extending the env_detect list (e.g. to alphabetize it, or add a
# fifth required package) would silently change what this banner
# calls "blocked" without touching what the executor actually
# refuses to run -- the exact "diagnostic lies about the real
# behavior" failure mode this whole module exists to prevent.
from sift.executor import _PYTHON_HARD_REQUIRED


Status = Literal["ok", "warning", "unavailable", "blocked"]


@dataclass
class RuntimeReport:
    """Per-runtime block within ``DoctorReport``.

    ``status`` is the worst applicable level:
      * ``ok`` — usable as-is.
      * ``warning`` — usable, but a feature is degraded
        (e.g., Python without matplotlib means plots won't render).
      * ``unavailable`` — an optional language runtime is not installed.
        This is not a product fault and must not nag a researcher who can
        use another runtime. A request to execute that language will still
        fail with its language-specific preflight error.
      * ``blocked`` — a required platform facility is unusable, or a
        present runtime is broken (probe failed / hard packages missing).

    ``advice`` is the concrete next step. Phrased imperatively so the
    UI can render it as a button label or a copyable command.
    """
    runtime: str            # "Python" / "R" / "Stata" / "sandbox-exec" / "bwrap"
    status: Status
    detail: str             # one-line human summary
    advice: list[str] = field(default_factory=list)


@dataclass
class DoctorReport:
    """Top-level report assembled by ``run_doctor``.

    ``blocked`` is True iff any RuntimeReport is at ``blocked``
    status AND that runtime is on Sift's required path (today:
    the platform sandbox backend — ``sandbox-exec`` on macOS,
    ``bwrap`` on Linux — plus at least one of R/Python/Stata
    being usable). The CLI's exit code derives from this.

    ``rejected_python_candidates`` is the list of (path, stderr_tail)
    entries from the sandbox-health probe cache — surfaced
    separately so a researcher who has a *working* Python alongside
    the rejected one still sees that Sift deliberately skipped the
    other binary (and why). The Apple-xcrun-stub case is the
    canonical example.
    """
    runtimes: list[RuntimeReport]
    rejected_python_candidates: list[tuple[str, str]]
    blocked: bool


def _python_report(env: Environment) -> RuntimeReport:
    py = env.python
    if py is None:
        # ``find_python`` returned None. Two distinct shapes: no
        # python3 on PATH at all, or every candidate failed the
        # sandbox probe. The probe cache disambiguates.
        rejected = [
            (path, stderr) for path, (ok, stderr)
            in python_sandbox_probe_results().items() if not ok
        ]
        if rejected:
            tail = (rejected[0][1] or "").strip().splitlines()
            stderr_summary = tail[-1] if tail else "(no stderr captured)"
            return RuntimeReport(
                runtime="Python",
                status="blocked",
                detail=(
                    f"python3 was found at {rejected[0][0]} but failed "
                    f"the sandbox-health probe: {stderr_summary}"
                ),
                advice=[
                    "Install a real Python via Homebrew "
                    "(``brew install python``) or python.org. "
                    "Apple's bundled /usr/bin/python3 is an xcselect "
                    "stub that cannot start under the Sift sandbox.",
                    "After installing, re-launch Sift.",
                ],
            )
        return RuntimeReport(
            runtime="Python",
            status="unavailable",
            detail=(
                "python3 not found on PATH and no usable bundled Python "
                "analysis runtime was found."
            ),
            advice=[
                "Released Sift builds include Python. Reinstall Sift if its "
                "bundled runtime is missing; source builds may install "
                "Python 3 separately.",
            ],
        )

    hard_missing = sorted(
        set(py.missing_packages or ()) & _PYTHON_HARD_REQUIRED
    )
    if hard_missing:
        return RuntimeReport(
            runtime="Python",
            status="blocked",
            detail=(
                f"Python at {py.binary} is missing required packages: "
                f"{', '.join(hard_missing)}"
            ),
            advice=[
                f"``{py.binary} -m pip install {' '.join(hard_missing)}``",
                "Then re-launch Sift.",
            ],
        )
    soft_missing = sorted(
        set(py.missing_packages or ()) - _PYTHON_HARD_REQUIRED
    )
    optional_missing = sorted(py.optional_missing_packages or ())
    if soft_missing or optional_missing:
        notes = []
        if soft_missing:
            notes.append(
                "method packages missing: "
                f"{', '.join(soft_missing)} (the workflows backed by these "
                "packages are unavailable in this interpreter)"
            )
        if optional_missing:
            notes.append(
                "optional packages missing: "
                f"{', '.join(optional_missing)} (plot helpers won't work "
                "without matplotlib)"
            )
        return RuntimeReport(
            runtime="Python",
            status="warning",
            detail=(
                f"Python at {py.binary} ({py.version}) is usable. "
                + "; ".join(notes)
                + (
                    " This is Sift's own bundled interpreter, not a "
                    "system install." if py.bundled else ""
                )
            ),
            advice=[
                f"``{py.binary} -m pip install "
                f"{' '.join(soft_missing + optional_missing)}`` "
                "to enable the missing helpers."
            ] if (soft_missing or optional_missing) else [],
        )
    bundled_note = (
        " This is Sift's own bundled interpreter -- no system Python "
        "install was needed." if py.bundled else ""
    )
    return RuntimeReport(
        runtime="Python",
        status="ok",
        detail=f"Python at {py.binary} ({py.version}) is healthy.{bundled_note}",
    )


def _r_report(env: Environment) -> RuntimeReport:
    r = env.r
    if r is None:
        return RuntimeReport(
            runtime="R",
            status="unavailable",
            detail="Rscript not found on PATH.",
            advice=[
                "Install R from https://cran.r-project.org and "
                "re-launch Sift only if you need R-language execution; "
                "otherwise use an available Python or Stata runtime."
            ],
        )
    optional_missing = sorted(r.optional_missing_packages or ())
    if optional_missing:
        return RuntimeReport(
            runtime="R",
            status="warning",
            detail=(
                f"R at {r.binary} ({r.version or 'version unknown'}) is "
                f"usable. Optional packages missing: "
                f"{', '.join(optional_missing)}"
            ),
            advice=[
                f"In R: ``install.packages(c({', '.join(repr(p) for p in optional_missing)}))``"
            ],
        )
    return RuntimeReport(
        runtime="R",
        status="ok",
        detail=f"R at {r.binary} ({r.version or 'version unknown'}) is healthy.",
    )


def _stata_report(env: Environment) -> RuntimeReport:
    st = env.stata
    if st is None:
        return RuntimeReport(
            runtime="Stata",
            status="unavailable",
            detail=(
                "Stata is not installed. Sift can still open and analyze "
                ".dta files with its bundled reader; only Stata-language "
                "script execution is unavailable."
            ),
            advice=[
                "No action is needed to use .dta data. Install a licensed "
                "copy of Stata only if you specifically want Sift to execute "
                "Stata-language scripts.",
            ],
        )
    return RuntimeReport(
        runtime="Stata",
        status="ok",
        detail=f"Stata at {st.binary} is healthy.",
    )


def _sandbox_report(env: Environment) -> RuntimeReport:
    """Platform-aware sandbox health row.

    Branches the same way ``executor.run_script``'s preflight does:
    darwin checks sandbox-exec, linux checks bwrap, anything else has
    no supported backend at all. Keeping this a straight mirror of
    the executor's own branch (rather than inventing separate
    detection logic here) is deliberate — a doctor that disagrees
    with the executor about which backend is authoritative would be
    worse than no doctor at all, since it would report "healthy"
    right before the executor refuses the researcher's first script.
    """
    if sys.platform == "darwin":
        return _sandbox_exec_report(env)
    if sys.platform.startswith("linux"):
        return _bwrap_report(env)
    if sys.platform.startswith("win"):
        return _appcontainer_report(env)
    return RuntimeReport(
        runtime="sandbox",
        status="blocked",
        detail=(
            f"no supported sandbox backend for platform {sys.platform!r} "
            "— Sift refuses to run scripts unsandboxed on any platform "
            "without one."
        ),
        advice=[
            "Sift's script execution requires macOS (sandbox-exec), "
            "Linux (bubblewrap), or Windows (AppContainer) in this "
            "version.",
        ],
    )


def _resource_limit_report() -> RuntimeReport:
    """Report whether the executor can apply its configured kernel limits."""
    if sys.platform.startswith("win"):
        return RuntimeReport(
            runtime="resource-limits",
            status="ok",
            detail="Windows Job Object resource limits are available through AppContainer.",
        )
    from sift.executor import (
        script_cpu_limit_seconds,
        script_file_size_limit_bytes,
        script_memory_limit_bytes,
    )

    needs_launcher = any((
        script_cpu_limit_seconds() > 0,
        script_file_size_limit_bytes() > 0,
        sys.platform.startswith("linux") and script_memory_limit_bytes() > 0,
    ))
    if not needs_launcher:
        return RuntimeReport(
            runtime="resource-limits",
            status="ok",
            detail="POSIX kernel resource limits are explicitly disabled.",
        )
    shell = next(
        (path for path in (Path("/bin/bash"), Path("/usr/bin/bash")) if path.is_file()),
        None,
    )
    if shell is None:
        return RuntimeReport(
            runtime="resource-limits",
            status="blocked",
            detail=(
                "Bash is unavailable, so Sift cannot safely apply its configured "
                "POSIX CPU and file resource limits. Script execution is blocked."
            ),
            advice=[
                "Install Bash with the operating system's package manager and re-run "
                "``sift --doctor``.",
            ],
        )
    return RuntimeReport(
        runtime="resource-limits",
        status="ok",
        detail=f"POSIX kernel resource-limit launcher verified at {shell}.",
    )


def _sandbox_exec_report(env: Environment) -> RuntimeReport:
    if env.sandbox_exec is None:
        return RuntimeReport(
            runtime="sandbox-exec",
            status="blocked",
            detail=(
                "sandbox-exec not available on this machine — Sift "
                "refuses to run scripts unsandboxed."
            ),
            advice=[
                "This binary lives at /usr/bin/sandbox-exec and is "
                "always present on macOS; if it's missing, something "
                "unusual has happened to this machine's base install.",
            ],
        )
    # Distinct failure shape: sandbox-exec is present but won't apply
    # a minimal profile (SBPL compiler broken, OS doesn't honour
    # sandbox_apply, nested-sandbox harness). Surfaced as its own row
    # because the advice differs from "install something" — there's
    # nothing for the researcher to install. ``_check_sandbox_baseline``
    # is cached so this doesn't re-probe per ``--doctor`` call.
    from sift.env_detect import sandbox_baseline_result
    baseline_ok, baseline_err = sandbox_baseline_result()
    if not baseline_ok:
        return RuntimeReport(
            runtime="sandbox-exec",
            status="blocked",
            detail=(
                f"sandbox-exec at {env.sandbox_exec} cannot apply a "
                f"minimal profile: {baseline_err}"
            ),
            advice=[
                "If you're running Sift inside another sandbox or "
                "container, exit it and re-launch on the host.",
                "If you're on a heavily customised macOS variant, "
                "verify that ``sandbox-exec -p '(version 1)(allow "
                "default)' /usr/bin/true`` exits zero from a normal "
                "terminal — Sift needs at least that much to function.",
            ],
        )
    return RuntimeReport(
        runtime="sandbox-exec",
        status="ok",
        detail=f"sandbox-exec at {env.sandbox_exec}.",
    )


def _bwrap_report(env: Environment) -> RuntimeReport:
    if env.bwrap is None:
        return RuntimeReport(
            runtime="bwrap",
            status="blocked",
            detail=(
                "bubblewrap (bwrap) not available on this machine — "
                "Sift refuses to run scripts unsandboxed."
            ),
            advice=[
                "Install it with your distribution's package manager: "
                "``apt install bubblewrap``, ``dnf install bubblewrap``, "
                "or ``pacman -S bubblewrap``.",
            ],
        )
    # Distinct failure shape: bwrap is present but can't actually
    # apply a minimal sandbox (nested-namespace harness, kernel
    # without unprivileged user namespaces, AppArmor/SELinux policy
    # blocking it). Same "there's nothing to install" advice split as
    # the macOS baseline check, and cached the same way.
    from sift.env_detect import bwrap_baseline_result
    baseline_ok, baseline_err = bwrap_baseline_result()
    if not baseline_ok:
        return RuntimeReport(
            runtime="bwrap",
            status="blocked",
            detail=(
                f"bwrap at {env.bwrap} cannot apply a minimal sandbox: "
                f"{baseline_err}"
            ),
            advice=[
                "If you're running Sift inside a container or another "
                "sandbox, unprivileged user namespaces may be blocked "
                "— exit it and re-launch on the host.",
                "On some hardened distributions, unprivileged user "
                "namespaces are disabled by default; check "
                "``sysctl kernel.unprivileged_userns_clone`` (Debian/"
                "Ubuntu family) or your distro's AppArmor policy for "
                "bwrap.",
            ],
        )
    return RuntimeReport(
        runtime="bwrap",
        status="ok",
        detail=f"bwrap at {env.bwrap}.",
    )


def _appcontainer_report(env: Environment) -> RuntimeReport:
    """Report Windows AppContainer and Job Object readiness.

    Three-way split, one level deeper than the macOS/Linux reports:
    "API surface missing" (old Windows, or something broken about
    this install), "API surface present but the live health probe
    failed" (confinement doesn't actually hold, or can't be verified,
    on this specific machine — see
    ``win_appcontainer.probe_appcontainer_health``), and "ok" (the
    probe positively confirmed a denied file read AND a denied
    network connect from inside a real throwaway AppContainer on this
    machine, moments ago). The middle case is the one this backend
    has that sandbox-exec/bwrap don't need as sharply: those two are
    mature OS components with their own probes. The API surface alone is
    deliberately never treated as sufficient — see ``run_script``'s
    win32 preflight branch for the same two-gate logic applied at
    execution time, not just at doctor-report time.
    """
    if not env.appcontainer_support:
        return RuntimeReport(
            runtime="appcontainer",
            status="blocked",
            detail=(
                "Windows AppContainer sandbox support is not available "
                "on this machine — Sift refuses to run scripts "
                "unsandboxed."
            ),
            advice=[
                "AppContainer requires Windows 8 or later. If you're "
                "on a supported version and seeing this, something "
                "unusual has happened to this machine's base install "
                "— userenv.dll should always expose "
                "CreateAppContainerProfile.",
            ],
        )
    from sift.env_detect import appcontainer_probe_result
    probe_ok, probe_detail = appcontainer_probe_result()
    if not probe_ok:
        return RuntimeReport(
            runtime="appcontainer",
            status="blocked",
            detail=(
                "AppContainer support is present but failed its "
                f"startup health check: {probe_detail}"
            ),
            advice=[
                "This is not necessarily something to install — the "
                "probe actually launched a throwaway sandboxed "
                "process and confirmed it could deny a file read and "
                "a network connection; if either check failed "
                "unexpectedly, this may be a permissions issue with "
                "the current Windows account (CreateAppContainerProfile "
                "needs standard user rights, not admin, but some "
                "locked-down enterprise policies restrict it) or a "
                "genuine bug — please report it.",
            ],
        )
    return RuntimeReport(
        runtime="appcontainer",
        status="ok",
        detail=(
            "AppContainer + Job Objects sandbox verified on this "
            "machine (denied file read and denied network connect "
            "both confirmed by the startup health probe)."
        ),
    )


def run_doctor(env: Environment | None = None) -> DoctorReport:
    """Build a health report for the current Sift environment.

    Pass ``env`` to skip detection (testing / pre-computed state); by
    default this calls ``detect_environment()`` directly so probe
    caches inside ``env_detect`` populate as a side effect — meaning
    a subsequent ``submit_script`` in the same Sift process reuses
    them without re-probing.

    ``blocked`` on the returned report means script execution will
    fail. The CLI maps this to exit code 1; the UI banner maps it
    to disabling the chat input.
    """
    env = env or detect_environment()
    runtimes = [
        _sandbox_report(env),
        _resource_limit_report(),
        _python_report(env),
        _r_report(env),
        _stata_report(env),
    ]
    # A "blocked" report blocks the corresponding language. Sift is
    # blocked overall when sandbox is blocked (no scripts can run at
    # all) or when EVERY language is blocked (no script in any
    # language can run). One language blocked + another working is a
    # warning the UI surfaces but doesn't gate on — the researcher
    # can still proceed in the working language.
    # Both sandbox backends (and the no-supported-backend fallback
    # row) count as "the sandbox row" for gating purposes — the set
    # membership check (rather than a single string) is what keeps
    # this gate correct across platforms without needing a second
    # per-platform branch here too.
    _SANDBOX_ROW_NAMES = (
        "sandbox-exec", "bwrap", "appcontainer", "sandbox", "resource-limits",
    )
    sandbox_blocked = any(
        r.runtime in _SANDBOX_ROW_NAMES and r.status == "blocked"
        for r in runtimes
    )
    language_runtimes = [
        r for r in runtimes if r.runtime not in _SANDBOX_ROW_NAMES
    ]
    all_languages_blocked = (
        all(r.status not in {"ok", "warning"} for r in language_runtimes)
        if language_runtimes else True
    )
    blocked = sandbox_blocked or all_languages_blocked

    rejected = [
        (path, stderr)
        for path, (ok, stderr) in python_sandbox_probe_results().items()
        if not ok
    ]
    return DoctorReport(
        runtimes=runtimes,
        rejected_python_candidates=rejected,
        blocked=blocked,
    )


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------

_STATUS_GLYPHS = {
    "ok": "[ok]",
    "warning": "[warn]",
    "unavailable": "[n/a]",
    "blocked": "[fail]",
}


def render_report_text(report: DoctorReport) -> str:
    """Render the report as plain text suitable for terminal output.

    Used by ``sift --doctor``. The UI banner consumes the
    DoctorReport directly and renders its own DOM, so this function
    is the terminal-only path.
    """
    lines: list[str] = []
    lines.append("Sift environment check")
    lines.append("=" * 30)
    for r in report.runtimes:
        glyph = _STATUS_GLYPHS.get(r.status, r.status)
        lines.append(f"{glyph} {r.runtime}: {r.detail}")
        for tip in r.advice:
            lines.append(f"    -> {tip}")
    if report.rejected_python_candidates:
        lines.append("")
        lines.append("Python candidates rejected by the sandbox probe:")
        for path, stderr in report.rejected_python_candidates:
            tail = (stderr or "").strip().splitlines()
            summary = tail[-1] if tail else "(no stderr captured)"
            lines.append(f"  {path}")
            lines.append(f"    {summary}")
    lines.append("")
    if report.blocked:
        lines.append(
            "Status: BLOCKED — script execution will fail until the "
            "issues above are fixed."
        )
    else:
        lines.append(
            "Status: OK — Sift is ready to run scripts."
        )
    return "\n".join(lines) + "\n"


def main_cli() -> int:
    """Entry point for ``sift --doctor``. Returns an exit code so the
    caller (typically ``ui.main``) can ``sys.exit`` on it; tests can
    call this directly and assert on the return value without
    intercepting ``sys.exit``."""
    report = run_doctor()
    sys.stdout.write(render_report_text(report))
    sys.stdout.flush()
    return 1 if report.blocked else 0
