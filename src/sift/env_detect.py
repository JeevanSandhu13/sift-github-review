"""Sift — runtime environment detection.

Finds the researcher's installed `Rscript` / `stata` / `python3`
binaries so the executor knows what to invoke. Checks `PATH` and
common macOS install locations. No fuzziness: either we find an
executable or we don't.

The result is consulted at app startup so the banner can honestly tell
the researcher what Sift will and won't be able to run for them.

Python detection also probes the maintained scientific engines and data
readers used by model-authored scripts. A bare ``python3`` can still be
discovered — ``find_python`` records what's missing so the UI can explain
which workflows are unavailable instead of allowing a cryptic
``ModuleNotFoundError`` on the first analysis.
"""

from __future__ import annotations

import os
import shutil
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sift.subprocess_safety import run_bounded_capture


# Common macOS install locations for Stata. `stata` / `stata-mp` /
# `stata-se` on PATH is preferred because users configure that themselves;
# falling back to /Applications paths means we find it even when PATH
# isn't set up.
_STATA_APP_LOCATIONS: tuple[str, ...] = (
    "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp",
    "/Applications/Stata/StataSE.app/Contents/MacOS/stata-se",
    "/Applications/Stata/Stata.app/Contents/MacOS/stata",
    "/Applications/StataMP.app/Contents/MacOS/stata-mp",
    "/Applications/StataSE.app/Contents/MacOS/stata-se",
    "/Applications/Stata.app/Contents/MacOS/stata",
)

# Windows: Stata's installer bakes the major version into the install
# directory name ("C:\Program Files\Stata18\", ...) -- unlike
# macOS's fixed, versionless .app path, there's no single canonical
# location, so a licensed researcher's copy could be under any of the
# last several major releases. Enumerate a reasonable span of recent
# versions, each in MP/SE/plain flavor, both 64-bit and legacy 32-bit
# executable names, under both the normal and (x86) Program Files
# roots a 32-bit installer would use even on a 64-bit OS.
_STATA_WINDOWS_LOCATIONS: tuple[str, ...] = tuple(
    rf"{root}\Stata{version}\{exe}"
    for root in (r"C:\Program Files", r"C:\Program Files (x86)")
    for version in ("19", "18", "17", "16", "15", "14")
    for exe in (
        "StataMP-64.exe", "StataSE-64.exe", "Stata-64.exe",
        "StataMP.exe", "StataSE.exe", "Stata.exe",
    )
)

# Linux: StataCorp distributes Stata for Linux as a tarball that
# unpacks to "/usr/local/stataNN/" by convention and does NOT add
# itself to PATH -- the researcher has to run the bundled `stinit`
# script or manually symlink the binaries into PATH, a step many
# never take. `stata-mp` / `stata-se` / `stata` are the command-line
# (non-GUI) executables this integration needs; the `xstata-*` GUI
# launchers aren't scriptable the same way and aren't probed here.
_STATA_LINUX_LOCATIONS: tuple[str, ...] = tuple(
    f"/usr/local/stata{version}/{exe}"
    for version in ("19", "18", "17", "16", "15", "14")
    for exe in ("stata-mp", "stata-se", "stata")
)


@dataclass(frozen=True)
class Tool:
    """A discovered statistical runtime."""
    name: str        # Human-readable, e.g. "R" or "Stata"
    binary: str      # Absolute path to the executable
    version: str | None = None
    # Optional: which packages the discovered interpreter is missing,
    # for runtimes (Python today) where having the binary isn't enough.
    # Empty tuple means "ready to go." None means "not checked yet."
    missing_packages: tuple[str, ...] = ()
    # Optional packages whose absence DOES NOT block runs but DOES
    # disable specific features. ``matplotlib`` is the canonical
    # case: scripts that don't plot run fine without it, but
    # ``sift.plot_*`` helpers fail silently on import. The executor
    # surfaces these in the missing-deps hint so a researcher who
    # wants plot vision knows exactly what to install.
    optional_missing_packages: tuple[str, ...] = ()
    # Extra filesystem subpaths the executor's sandbox profile should
    # allow reads from when this interpreter runs. For Python this is
    # ``sys.prefix`` (and ``sys.exec_prefix`` if different) — the
    # interpreter needs to read its own stdlib and site-packages,
    # which can live outside the system trees the default profile
    # already covers (venvs, pyenv installs, conda envs, …).
    extra_read_paths: tuple[str, ...] = ()
    # True iff this Tool is Sift's own vendored runtime (returned by
    # find_bundled_python()) rather than something discovered on the
    # researcher's PATH. Deliberately does NOT change ``name`` — every
    # existing ``tool.name == "Python"`` call site (executor.py's
    # package-summary view, doctor.py's report) keeps working
    # unchanged for a bundled Tool exactly as it does for a
    # PATH-discovered one. This field exists purely so UI/doctor
    # copy CAN say "this is Sift's bundled interpreter, not your
    # system one" when it's useful context, without that being load-
    # bearing for any executor behavior.
    bundled: bool = False


def _binary_read_roots(binary: str) -> tuple[str, ...]:
    """Narrow install roots a sandboxed non-Python runtime must read."""
    path = Path(binary).resolve()
    for parent in (path, *path.parents):
        if parent.suffix.lower() in {".app", ".framework"}:
            return (str(parent),)

    anchors = [Path("/opt"), Path("/usr/local")]
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        if os.environ.get(variable):
            anchors.append(Path(os.environ[variable]))
    for anchor in anchors:
        try:
            relative = path.relative_to(anchor)
        except ValueError:
            continue
        if relative.parts:
            return (str(anchor / relative.parts[0]),)
    return (str(path.parent),)


def _existing_roots(*paths: Path | str) -> tuple[str, ...]:
    """Return existing roots as canonical absolute paths.

    Sandbox backends require absolute bind/profile paths. Environment-provided
    values such as a relative ``R_LIBS=packages`` previously survived here and
    produced a relative bubblewrap bind (which fails at launch) or an ambiguous
    macOS profile grant. Resolve only existing entries and skip filesystem
    errors so optional runtime discovery remains fail-soft.
    """
    roots: list[str] = []
    for raw in paths:
        try:
            candidate = Path(raw).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        value = str(candidate)
        if value not in roots:
            roots.append(value)
    return tuple(roots)


# R packages we probe at startup. ``haven`` is needed to read .dta
# files (Stata's native format) — without it, R can't open any of
# the user's Stata datasets and the model's first attempt to
# ``library(haven)`` fails. ``ggplot2`` is the most common plotting
# library; helpers fall back to base graphics, but raw model
# scripts often reach for it. None of these are HARD requirements
# — base R can still run analyses without them — but advertising
# their availability in the system prompt lets the model pick the
# right path on the first try instead of discovering missing-
# package errors by failing.
_R_OPTIONAL_PACKAGES: tuple[str, ...] = (
    "haven",
    "ggplot2",
)


def find_r() -> Tool | None:
    """Return the discovered R runtime, or None."""
    path = shutil.which("Rscript")
    if path is None:
        return None
    optional_missing = _r_missing_packages(path, _R_OPTIONAL_PACKAGES)
    configured_r_libs = tuple(
        entry
        for variable in ("R_LIBS", "R_LIBS_USER", "R_LIBS_SITE")
        for entry in os.environ.get(variable, "").split(os.pathsep)
        if entry
    )
    user_home = Path.home()
    return Tool(
        name="R", binary=path, version=_r_version(path),
        optional_missing_packages=optional_missing,
        extra_read_paths=tuple(dict.fromkeys((
            *_binary_read_roots(path),
            *_r_library_paths(path),
            *_existing_roots(
                *configured_r_libs,
                user_home / "R",
                user_home / "Library" / "R",
                user_home / "Documents" / "R" / "win-library",
            ),
        ))),
    )


def _r_missing_packages(
    rscript: str, packages: tuple[str, ...],
) -> tuple[str, ...]:
    """Probe an R installation for ``packages``. Returns the names
    that aren't installed. Done in a single ``Rscript`` invocation
    (one subprocess per package would inflate startup time on
    machines with slow R). The probe writes a single boolean per
    package to stdout, separated by spaces.

    Uses ``system.file(package = pkg)`` rather than
    ``requireNamespace(pkg)``. ``requireNamespace`` LOADS the
    package namespace, which fires the package's ``.onLoad`` hook
    and runs arbitrary R code OUTSIDE the analysis sandbox (the
    probe is a vanilla ``Rscript`` invocation at app startup and
    after package installs). A malicious or compromised package
    named ``haven`` or ``ggplot2`` would execute code during the
    probe. The subprocess receives the same credential-stripping environment
    allowlist as a real analysis. ``system.file`` only checks the
    installed-package directory on disk and does NOT load
    namespaces — answers "is this package installed?" without
    executing package code.

    Failures (Rscript missing, weird R version, OS error) return
    "all missing" rather than the conservative "none missing" so
    the system prompt is honest about uncertainty.
    """
    if not packages:
        return ()
    expr = "; ".join(
        f"cat(nzchar(system.file(package=\"{pkg}\")), \" \")"
        for pkg in packages
    )
    from sift.executor import _filter_env

    try:
        out = run_bounded_capture(
            [rscript, "--vanilla", "-e", expr],
            timeout=10,
            check=False,
            env=_filter_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError):
        return tuple(packages)
    flags = out.stdout.strip().split()
    if len(flags) != len(packages):
        return tuple(packages)
    return tuple(
        pkg for pkg, present in zip(packages, flags)
        if present.upper() != "TRUE"
    )


def _r_library_paths(rscript: str) -> tuple[str, ...]:
    """Discover R's effective library roots without loading any package.

    R expands version placeholders and platform defaults inside ``R_LIBS*``;
    parsing those environment strings in Python can therefore grant the wrong
    directory and make a valid user library unreadable inside the sandbox.
    ``.libPaths()`` is base-R metadata and does not load package code.
    """
    from sift.executor import _filter_env

    try:
        out = run_bounded_capture(
            [
                rscript,
                "--vanilla",
                "-e",
                'cat(normalizePath(.libPaths(), winslash="/", '
                'mustWork=TRUE), sep="\\n")',
            ],
            timeout=10,
            check=False,
            env=_filter_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if out.returncode != 0:
        return ()
    return _existing_roots(
        *(line.strip() for line in out.stdout.splitlines() if line.strip())
    )


def _stata_fallback_locations(platform: str | None = None) -> tuple[str, ...]:
    """Return only Stata binaries executable by the current platform.

    Filesystem presence alone is not enough: a native Windows process cannot
    execute a Linux or macOS binary, and POSIX mode bits have no useful
    executable meaning on Windows. PATH remains the supported escape hatch
    for WSL, Wine, containers, and administrator-provided launchers.
    """
    current = sys.platform if platform is None else platform
    if current == "win32":
        return _STATA_WINDOWS_LOCATIONS
    if current == "darwin":
        return _STATA_APP_LOCATIONS
    if current.startswith("linux"):
        return _STATA_LINUX_LOCATIONS
    return ()


def find_stata() -> Tool | None:
    """Return the discovered Stata runtime, or None.

    Tries `stata-mp`, `stata-se`, `stata` on PATH first, then falls back to
    common install locations for the current platform. Other platforms'
    binaries are deliberately excluded because their presence does not make
    them executable by this process.
    """
    for cmd in ("stata-mp", "stata-se", "stata"):
        path = shutil.which(cmd)
        if path:
            return Tool(
                name="Stata", binary=path,
                extra_read_paths=tuple(dict.fromkeys((
                    *_binary_read_roots(path),
                    *_existing_roots(Path.home() / "ado", Path("C:/ado")),
                ))),
            )
    for p in _stata_fallback_locations():
        if Path(p).is_file() and os.access(p, os.X_OK):
            return Tool(
                name="Stata", binary=p,
                extra_read_paths=tuple(dict.fromkeys((
                    *_binary_read_roots(p),
                    *_existing_roots(Path.home() / "ado", Path("C:/ado")),
                ))),
            )
    return None


# Packages the Python runtime helpers (``sift.runtime.sift`` Python
# module) need to actually do work. Pandas is the lingua franca for
# data; statsmodels covers OLS / GLM / t-tests with the SE/CI fields
# the sanitizer expects. SciPy is pulled in transitively by
# statsmodels but checked explicitly so the missing-dep message is
# unambiguous.
_PYTHON_REQUIRED_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "statsmodels",
    "scipy",
    "sklearn",
    "factor_analyzer",
    "pyfixest",
    "rdrobust",
    "differences",
    "geopandas",
    "arviz",
    "pymc",
    "polars",
    # Readers used by model-authored scripts. The host can inspect these
    # formats itself, but selecting a PATH Python without the same reader
    # would make local analysis fail despite a complete bundled runtime.
    "pyarrow",
    "openpyxl",
    "xlrd",
    "odf",
    "pyreadstat",
    "pyreadr",
)

# Optional but feature-gating. ``matplotlib`` powers every plot
# helper; missing it means ``sift.plot_*`` calls fail silently and
# the model thinks it produced an image while the researcher sees
# nothing. Probed but NOT required so non-plotting scripts still
# run; the executor surfaces missing optionals in its hint text so
# a researcher who wanted plots knows what to install.
_PYTHON_OPTIONAL_PACKAGES: tuple[str, ...] = (
    "matplotlib",
    # ``duckdb`` unlocks the out-of-core path for large files: a
    # script can run SQL directly against a CSV/Parquet file without
    # loading it into pandas, which is the difference between "fits
    # in RAM" and "fits on disk". Optional — everything works without
    # it — but the system prompt advertises its presence so the model
    # reaches for it on big datasets instead of a doomed full load.
    "duckdb",
)

# Complete import surface promised by the self-contained analysis runtime.
# Release packaging verifies this exact set after copying the runtime into the
# frozen artifact. Keep it explicit: data readers are app dependencies but
# must also exist inside the separate sandboxed interpreter used by scripts.
_BUNDLED_ANALYSIS_PACKAGES: tuple[str, ...] = (
    *_PYTHON_REQUIRED_PACKAGES,
    *_PYTHON_OPTIONAL_PACKAGES,
)


def find_python() -> Tool | None:
    """Return the discovered Python 3 interpreter, or None.

    Tries ``python3`` then ``python`` on PATH. For each candidate that
    exists and reports a Python 3 version, runs a sandbox-health probe
    (``_probe_sandbox_health``, cached in-memory) before accepting it.
    An interpreter that runs fine outside the sandbox but can't start
    under it — canonically Apple's ``/usr/bin/python3``, an
    ``xcselect`` stub that dlopens ``/Library/Developer/CommandLine
    Tools/usr/lib/libxcrun.dylib`` at startup, a path the executor's
    profile doesn't allow — is rejected so the executor doesn't
    accept a Python it will then fail to run.

    Missing required / optional packages are recorded on
    ``Tool.missing_packages`` so the executor can refuse with a
    clear message rather than letting the script crash with
    ``ModuleNotFoundError`` after the sandbox is up.

    Refuses to consider Python 2 (still installed on some macOS
    setups via Homebrew) — the runtime library uses dataclasses and
    f-strings.
    """
    incomplete_path_candidates: list[Tool] = []
    for cmd in ("python3", "python"):
        path = shutil.which(cmd)
        if not path:
            continue
        version = _python_version(path)
        if version is None or not version.startswith("Python 3"):
            continue
        # Sandbox-health probe before accepting. The probe lazily
        # populates ``_SANDBOX_PROBE_CACHE`` so the first ``find_python``
        # call pays ~200ms once, and downstream callers (system_prompt,
        # ui banner, executor's preamble lookup) hit the cache. Failed
        # candidates stay in the cache so ``python_sandbox_probe_results``
        # can explain the rejection to the doctor / UI layer.
        if path not in _SANDBOX_PROBE_CACHE:
            _SANDBOX_PROBE_CACHE[path] = _probe_sandbox_health(path)
        if not _SANDBOX_PROBE_CACHE[path][0]:
            continue
        missing = _python_missing_packages(path, _PYTHON_REQUIRED_PACKAGES)
        optional_missing = _python_missing_packages(path, _PYTHON_OPTIONAL_PACKAGES)
        prefixes = tuple(dict.fromkeys((
            *_binary_read_roots(path),
            *_python_prefixes(path),
        )))
        tool = Tool(
            name="Python",
            binary=path,
            version=version,
            missing_packages=missing,
            optional_missing_packages=optional_missing,
            extra_read_paths=prefixes,
        )
        # A complete researcher-managed environment wins and retains access
        # to their custom packages. A partial installation must not shadow
        # Sift's complete bundled runtime and make built-in methods fail.
        if not missing:
            return tool
        incomplete_path_candidates.append(tool)

    bundled = find_bundled_python()
    if bundled is not None and not bundled.missing_packages:
        return bundled

    # Development builds may not contain the bundled runtime. Preserve the
    # best actionable partial candidate so the doctor can name its missing
    # packages rather than collapsing the diagnosis to "Python not found".
    candidates = [*incomplete_path_candidates]
    if bundled is not None:
        candidates.append(bundled)
    if not candidates:
        return None
    return min(candidates, key=lambda item: len(item.missing_packages))


# ---------------------------------------------------------------------------
# Bundled Python runtime
# ---------------------------------------------------------------------------
#
# Sift's own zero-setup Python floor. Unlike R and Stata (licensed /
# heavyweight, never vendored), Python's ecosystem can be fully
# bundled with no licensing complication, so a researcher who has
# never installed Python at all still gets a working analysis
# runtime out of the box in a released .app build.
#
# Native release jobs build this runtime on each target OS, import every
# promised engine, freeze it, and repeat the import gate from the artifact.
# Source/dev runs may intentionally omit it and fall through to PATH Python.
# Production build scripts refuse to skip it.

_BUNDLED_PYTHON_ENV_VAR = "SIFT_BUNDLED_PYTHON_DIR"


def _bundled_python_root() -> "Path | None":
    """Locate the vendored Python distribution's install root, if any.

    Resolution order:

    1. ``SIFT_BUNDLED_PYTHON_DIR`` — override for dev/test, and an
       escape hatch a researcher or packager could use to point at a
       hand-vendored distribution without a full PyInstaller rebuild.
    2. ``<frozen app>/vendor_python`` next to the frozen executable's
       data root, when running as a PyInstaller .app (detected via
       ``sys.frozen`` / ``sys._MEIPASS``) — populated by
       ``packaging/sift.spec``'s ``VENDOR_PYTHON_DATAS`` from
       whatever ``packaging/vendor_python.sh`` produced at
       ``packaging/vendor/python`` at build time.

    Returns ``None`` — not a guess — in every other case: a dev
    checkout (``python -m sift`` / ``uv run sift``), or a frozen
    build where the vendor step was never run before packaging.
    Does not verify the directory actually contains a working
    interpreter; that's ``find_bundled_python``'s job, via the same
    probes every other Python candidate goes through.
    """
    override = os.environ.get(_BUNDLED_PYTHON_ENV_VAR)
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidate = Path(meipass) / "sift" / "vendor_python"
            if candidate.is_dir():
                return candidate
    return None


def _bundled_python_binary(root: "Path") -> str | None:
    """Return the vendored interpreter's executable path under
    ``root``, or ``None`` if the expected layout isn't there.

    ``packaging/vendor_python.sh`` lays a relocated
    python-build-standalone distribution out as
    ``<root>/bin/python3`` (mirroring a normal install prefix's
    ``bin/`` layout). Matches that exact path rather than globbing
    for "anything executable named python*" — a differently-shaped
    ``root`` (partial vendor run, wrong tool, stray file) is rejected
    instead of silently picked up and probed as if it were trustworthy.
    """
    candidates = (
        (root / "python.exe", root / "bin" / "python.exe")
        if sys.platform.startswith("win")
        else (root / "bin" / "python3",)
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def find_bundled_python() -> "Tool | None":
    """Return the vendored Python runtime Sift ships inside its own
    .app bundle, or ``None`` if this build wasn't vendored.

    Reuses every probe ``find_python`` runs on a PATH-discovered
    interpreter — sandbox-health, required/optional package check,
    prefix resolution — so a vendored interpreter gets exactly the
    same scrutiny as a system one. "We bundled it" is not treated as
    an automatic pass: in steady state (a correctly-run vendor step)
    every check comes back clean, but a partially-failed vendoring
    run surfaces here as "missing packages" via the normal
    ``Tool.missing_packages`` path, not a silent crash on the
    researcher's first script.
    """
    root = _bundled_python_root()
    if root is None:
        return None
    path = _bundled_python_binary(root)
    if path is None:
        return None
    version = _python_version(path)
    if version is None or not version.startswith("Python 3"):
        return None
    if path not in _SANDBOX_PROBE_CACHE:
        _SANDBOX_PROBE_CACHE[path] = _probe_sandbox_health(path)
    if not _SANDBOX_PROBE_CACHE[path][0]:
        return None
    missing = _python_missing_packages(path, _PYTHON_REQUIRED_PACKAGES)
    optional_missing = _python_missing_packages(path, _PYTHON_OPTIONAL_PACKAGES)
    prefixes = tuple(dict.fromkeys((
        *_binary_read_roots(path),
        *_python_prefixes(path),
    )))
    return Tool(
        name="Python",
        binary=path,
        version=version,
        missing_packages=missing,
        optional_missing_packages=optional_missing,
        extra_read_paths=prefixes,
        bundled=True,
    )


# ---------------------------------------------------------------------------
# Sandbox-health probe for candidate Python interpreters
# ---------------------------------------------------------------------------
#
# Why this exists: outside the sandbox, ``binary -c "print(1)"`` answers
# "does this interpreter respond to ``-c``" — and that's exactly the
# question Apple's ``/usr/bin/python3`` (an ``xcselect`` stub that
# dispatches via libxcrun) passes cleanly. Inside the sandbox, the
# stub dies before main() because libxcrun's dylib lives outside the
# read allowlist. The two checks have different threat models, and the
# gap is where every script-startup failure of this class hides.
#
# The probe closes that gap by running the candidate under the SAME
# profile builder the executor uses for real script runs
# (``executor._sandbox_profile_string``). Drift between probe and run
# would defeat the purpose; sharing the builder is the invariant.
#
# In-memory cache only, lazy on first call. Disk persistence doesn't
# earn its keep — a stale "good" cache pointing at an uninstalled
# interpreter is a worse failure than a 200ms cold-start probe, and
# probe cost is small enough that within-process caching covers the
# realistic cases (system_prompt, ui banner, the executor's preamble
# lookup all hit the same path within one Sift session).
_SANDBOX_PROBE_CACHE: dict[str, tuple[bool, str]] = {}

# Separate cache for the SANDBOX layer's own health (phase A of the
# probe). The interpreter probe runs ``sandbox-exec -f <profile>
# <binary> ...`` — if sandbox-exec or the profile compiler is itself
# broken, every interpreter candidate fails identically and the doctor
# would mis-attribute the failure to the interpreter. Caching the
# baseline check separately lets us report "sandbox is broken" without
# blaming the (possibly fine) Python install.
_SANDBOX_BASELINE_CACHE: "tuple[bool, str] | None" = None


def _check_sandbox_baseline() -> tuple[bool, str]:
    """Verify the exact Sift macOS policy and its security invariants.

    The interpreter probe builds a real Sift profile and runs the
    candidate binary under it. That sequence can fail for two
    distinct reasons:

      * **Sandbox layer broken** — sandbox-exec is missing or
        misconfigured, the SBPL compiler rejects the profile, the
        OS doesn't honour ``sandbox_apply`` (nested-sandbox
        harnesses, exotic macOS variants). Every interpreter
        probed under such a sandbox fails identically.
      * **Interpreter rejected by a working sandbox** — the Apple
        xcrun stub class of failure, what the doctor was originally
        built to catch.

    Without distinguishing these, a researcher whose sandbox is
    broken would see "install Homebrew Python" advice that won't
    help. This function runs a minimal ``(allow default)`` profile
    against ``/usr/bin/true`` — the simplest possible sandbox
    invocation. If that fails, sandbox-exec itself is the problem,
    and ``_probe_sandbox_health`` short-circuits with the baseline
    error instead of running per-candidate probes.

    Cached in-memory for the lifetime of the process (the sandbox
    layer's health doesn't change mid-session). Non-macOS hosts
    return ``(True, "")`` — the executor doesn't sandbox there, so
    the baseline question is moot and the "sandbox not present"
    case is reported separately by the doctor's sandbox-runtime
    row.
    """
    global _SANDBOX_BASELINE_CACHE
    if _SANDBOX_BASELINE_CACHE is not None:
        return _SANDBOX_BASELINE_CACHE

    sandbox_exec = find_sandbox_exec()
    if sandbox_exec is None:
        _SANDBOX_BASELINE_CACHE = (True, "")
        return _SANDBOX_BASELINE_CACHE

    import shlex
    import tempfile

    from sift.executor import _filter_env, _sandbox_profile_string

    try:
        with tempfile.TemporaryDirectory(prefix="sift-macos-probe-") as scratch_str:
            scratch = Path(scratch_str).resolve()
            run_dir = scratch / ".sift" / "runs" / "health-probe"
            run_dir.mkdir(parents=True)
            hidden = scratch / ".sift" / "private-canary.txt"
            hidden.write_text("SIFT_PRIVATE_CANARY", encoding="utf-8")
            with tempfile.TemporaryDirectory(
                prefix="sift-macos-outside-"
            ) as outside_str:
                outside = Path(outside_str).resolve() / "outside-canary.txt"
                outside.write_text("OUTSIDE_PRIVATE_CANARY", encoding="utf-8")
                profile_path = run_dir / "sandbox.sb"
                profile_path.write_text(
                    _sandbox_profile_string(run_dir, scratch), encoding="utf-8"
                )
                q = shlex.quote
                script = "\n".join(
                    [
                        "set -eu",
                        f"printf CWD_OK > {q(str(scratch / 'allowed-workspace.txt'))}",
                        f"printf RUN_OK > {q(str(run_dir / 'allowed-run.txt'))}",
                        f"if cat {q(str(hidden))} >/dev/null 2>&1; then exit 41; fi",
                        f"if cat {q(str(outside))} >/dev/null 2>&1; then exit 42; fi",
                        # Clipboard access is an OS-broker route outside the
                        # file/network rules and must be denied independently.
                        "if /usr/bin/pbpaste >/dev/null 2>&1; then exit 43; fi",
                        "printf SECURITY_OK",
                    ]
                )
                out = run_bounded_capture(
                    [sandbox_exec, "-f", str(profile_path), "/bin/sh", "-c", script],
                    timeout=10,
                    cwd=str(scratch),
                    env=_filter_env(dict(os.environ)),
                )
                workspace_write = scratch / "allowed-workspace.txt"
                run_write = run_dir / "allowed-run.txt"
                writes_exist = (
                    workspace_write.is_file()
                    and run_write.is_file()
                    and workspace_write.read_text(encoding="utf-8") == "CWD_OK"
                    and run_write.read_text(encoding="utf-8") == "RUN_OK"
                )
    except (OSError, subprocess.SubprocessError, UnicodeError) as e:
        _SANDBOX_BASELINE_CACHE = (
            False, f"sandbox-exec failed to launch: {e}",
        )
        return _SANDBOX_BASELINE_CACHE
    if out.returncode != 0 or out.stdout != "SECURITY_OK" or not writes_exist:
        _SANDBOX_BASELINE_CACHE = (
            False,
            (
                "sandbox-exec did not positively prove Sift's production policy "
                f"(exit {out.returncode}). stderr: "
                f"{(out.stderr or '').strip() or '(empty)'}"
            ),
        )
        return _SANDBOX_BASELINE_CACHE
    _SANDBOX_BASELINE_CACHE = (True, "")
    return _SANDBOX_BASELINE_CACHE


def sandbox_baseline_result() -> tuple[bool, str]:
    """Public accessor for the sandbox-baseline cache.

    Used by the doctor's ``_sandbox_report`` to surface "sandbox
    layer broken" as a distinct failure from "sandbox-exec not
    present", and by tests that need to pre-seed the cache to
    simulate either branch.
    """
    return _check_sandbox_baseline()


# Same role as ``_SANDBOX_BASELINE_CACHE`` above, for the Linux
# backend. Kept as a fully separate cache (not unified with the macOS
# one) because the two probes run different commands and mean
# different things — conflating them would make a Linux-only bwrap
# failure show up as a "sandbox-exec" row and vice versa, which is
# exactly the mis-attribution class of bug this whole module exists
# to prevent.
_BWRAP_BASELINE_CACHE: "tuple[bool, str] | None" = None


def _check_bwrap_baseline() -> tuple[bool, str]:
    """Verify the exact Sift bwrap policy and its security invariants.

    Mirrors ``_check_sandbox_baseline`` for the Linux backend. The
    naive probe — ``bwrap --unshare-all --die-with-parent
    /usr/bin/true`` with no binds at all — is NOT a valid baseline
    check: bwrap builds its container filesystem from nothing, so a
    bind-free invocation fails with "No such file or directory"
    (there is no ``/usr/bin/true`` to exec — nothing was bound in)
    even when bwrap itself is completely healthy. That failure mode
    was hit and fixed during development of this module's sibling
    test suite (``tests/test_bwrap_sandbox.py``). The correct minimal
    probe binds the real root read-only first, which exercises the
    exact unprivileged-user-namespace capability Sift depends on
    without needing any of the executor's own bind logic.

    Distinguishes "bwrap missing" (reported separately, as a
    please-install-bubblewrap row) from "bwrap present but can't
    actually sandbox anything" (nested-namespace harness, kernel
    without unprivileged user namespaces enabled, AppArmor profile
    blocking it, etc.) — the same two-failure-mode split the macOS
    baseline check makes, and for the same reason: the advice is
    completely different ("install bubblewrap" vs "you're inside
    another container that blocks nested sandboxing").

    Cached for the process lifetime. Non-Linux hosts return
    ``(True, "")`` — bwrap is irrelevant there and "bwrap not
    present" is reported separately by the doctor's sandbox-runtime
    row.
    """
    global _BWRAP_BASELINE_CACHE
    if _BWRAP_BASELINE_CACHE is not None:
        return _BWRAP_BASELINE_CACHE

    if not sys.platform.startswith("linux"):
        _BWRAP_BASELINE_CACHE = (True, "")
        return _BWRAP_BASELINE_CACHE

    bwrap = find_bwrap()
    if bwrap is None:
        _BWRAP_BASELINE_CACHE = (True, "")
        return _BWRAP_BASELINE_CACHE

    import shlex
    import tempfile

    from sift.executor import _bwrap_argv

    try:
        with tempfile.TemporaryDirectory(prefix="sift-bwrap-probe-") as root_str:
            root = Path(root_str).resolve()
            cwd = root / "workspace"
            run_dir = cwd / ".sift" / "runs" / "health-probe"
            run_dir.mkdir(parents=True)
            hidden = cwd / ".sift" / "private-canary.txt"
            hidden.write_text("SIFT_PRIVATE_CANARY", encoding="utf-8")
            outside = root / "outside-canary.txt"
            outside.write_text("OUTSIDE_PRIVATE_CANARY", encoding="utf-8")
            parent_pid = os.getpid()

            q = shlex.quote
            script = "\n".join(
                [
                    "set -eu",
                    f"printf CWD_OK > {q(str(cwd / 'allowed-workspace.txt'))}",
                    f"printf RUN_OK > {q(str(run_dir / 'allowed-run.txt'))}",
                    f"if cat {q(str(hidden))} >/dev/null 2>&1; then exit 41; fi",
                    f"if cat {q(str(outside))} >/dev/null 2>&1; then exit 42; fi",
                    f"if test -e /proc/{parent_pid}; then exit 43; fi",
                    # A fresh network namespace must have no IPv4 routes.
                    "test \"$(wc -l < /proc/net/route)\" -eq 1",
                    "printf SECURITY_OK",
                ]
            )
            out = run_bounded_capture(
                [bwrap, *_bwrap_argv(run_dir, cwd, Path.home()), "/bin/sh", "-c", script],
                timeout=10,
                cwd=str(cwd),
            )
            workspace_write = cwd / "allowed-workspace.txt"
            run_write = run_dir / "allowed-run.txt"
            writes_exist = (
                workspace_write.is_file()
                and run_write.is_file()
                and workspace_write.read_text(encoding="utf-8") == "CWD_OK"
                and run_write.read_text(encoding="utf-8") == "RUN_OK"
            )
    except (OSError, subprocess.SubprocessError, UnicodeError) as e:
        _BWRAP_BASELINE_CACHE = (False, f"bwrap security probe failed: {e}")
        return _BWRAP_BASELINE_CACHE
    if out.returncode != 0 or out.stdout != "SECURITY_OK" or not writes_exist:
        _BWRAP_BASELINE_CACHE = (
            False,
            (
                "bwrap did not positively prove Sift's production isolation "
                f"policy (exit {out.returncode}). stdout: {(out.stdout or '')!r}; "
                f"stderr: {(out.stderr or '').strip() or '(empty)'}"
            ),
        )
        return _BWRAP_BASELINE_CACHE
    _BWRAP_BASELINE_CACHE = (True, "")
    return _BWRAP_BASELINE_CACHE


def bwrap_baseline_result() -> tuple[bool, str]:
    """Public accessor for the bwrap-baseline cache.

    Used by the doctor's ``_sandbox_report`` on Linux to surface
    "bwrap is installed but can't actually sandbox" as a distinct
    failure from "bwrap not installed", and by tests that need to
    pre-seed the cache to simulate either branch.
    """
    return _check_bwrap_baseline()


def _probe_sandbox_health(binary: str) -> tuple[bool, str]:
    """Run a trivial ``binary -I -c "print(1)"`` under a representative
    Sift sandbox profile and report whether it succeeded.

    Returns ``(ok, stderr_excerpt)``:
      * ``ok=True`` and empty stderr when the sandboxed subprocess
        exited 0 with ``"1"`` on stdout.
      * ``ok=False`` with up to ~4 KB of the failing stderr (the tail,
        which is what carries the actual error). Callers — the doctor
        command and the executor's error path — pass this back to the
        researcher unredacted: at this phase no researcher data has
        been touched, so launcher / dlopen / sandbox-denial output is
        safe to surface.

    On non-macOS systems (no ``sandbox-exec``) the probe is a no-op
    and returns ``(True, "")``. The executor doesn't sandbox there, so
    probing for sandbox compatibility is moot.

    Profile shape: built by ``executor._sandbox_profile_string`` so
    the probe and the executor's per-run profile share one source of
    truth. The probe's ephemeral cwd / run_dir live under
    ``tempfile.mkdtemp()``; both are added to the read allowlist by
    the same code the real run uses, so a candidate that passes here
    starts cleanly under the real run too (modulo cwd-specific
    paths, which don't affect interpreter startup).
    """
    sandbox_exec = find_sandbox_exec()
    if sandbox_exec is None:
        return True, ""

    # Phase A: is the sandbox layer itself usable? If a minimal
    # ``(allow default)`` profile against ``/usr/bin/true`` already
    # fails, every interpreter probe would fail identically and the
    # diagnostic (and the doctor's downstream rendering) would
    # mis-attribute the failure to the interpreter rather than the
    # sandbox. Short-circuit here with the baseline error so the
    # rejection cache carries an explicit "sandbox layer broken"
    # signal instead of N copies of the same downstream symptom.
    baseline_ok, baseline_err = _check_sandbox_baseline()
    if not baseline_ok:
        return False, (
            "sandbox-exec itself is unusable; this rejection is not "
            "specific to the interpreter at "
            f"{binary}. {baseline_err}"
        )

    # Lazy local imports to avoid an import-cycle with executor /
    # package_installer at module load. Same pattern as
    # ``_python_missing_packages`` / ``_python_prefixes`` below.
    import tempfile
    from sift.executor import _filter_env, _sandbox_profile_string

    prefixes = _python_prefixes(binary)
    with tempfile.TemporaryDirectory(prefix="sift-probe-") as scratch_str:
        # ``.resolve()`` is required on macOS: /var/folders/... is
        # reached as /private/var/folders/... at the kernel level,
        # and SBPL matches resolved paths. Without resolve(), the
        # profile's ``(subpath "/var/folders/...")`` allow doesn't
        # cover the kernel-side path the probe actually accesses.
        scratch = Path(scratch_str).resolve()
        run_dir = scratch / ".sift" / "runs" / "probe"
        run_dir.mkdir(parents=True)
        profile_text = _sandbox_profile_string(
            run_dir=run_dir, cwd=scratch, extra_read_paths=prefixes,
        )
        profile_path = run_dir / "sandbox.sb"
        profile_path.write_text(profile_text)

        try:
            out = run_bounded_capture(
                [
                    sandbox_exec, "-f", str(profile_path),
                    binary, "-I", "-B", "-c", "print(1)",
                ],
                timeout=10,
                env=_filter_env(dict(os.environ)),
                cwd=str(scratch),
            )
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"probe could not launch: {e}"

    if out.returncode == 0 and (out.stdout or "").strip() == "1":
        return True, ""
    # Tail of stderr — most launcher / dlopen errors emit a single
    # short line, but Python's traceback formatter and SBPL's deny
    # messages can run long. Cap at ~4 KB so the UI / doctor output
    # doesn't balloon, but keep the tail (the proximate cause is
    # almost always the last line).
    return False, (out.stderr or "")[-4000:]


def python_sandbox_probe_results() -> dict[str, tuple[bool, str]]:
    """Return a snapshot of the in-memory sandbox-probe cache.

    Each key is a probed interpreter path; value is
    ``(ok, stderr_excerpt)``. Used by the ``sift doctor`` command and
    by the executor's "no python3 found" error path to explain why a
    candidate Python was rejected — without this, every script
    silently dies and the researcher has no clue why.
    """
    return dict(_SANDBOX_PROBE_CACHE)


def find_sandbox_exec() -> str | None:
    """Return the path to macOS sandbox-exec, or None on non-macOS or if missing.

    sandbox-exec is deprecated by Apple but still functional through
    current macOS. On Linux, the equivalent confinement comes from
    ``find_bwrap`` instead. On either platform, if the confinement
    binary for that platform is missing, the executor REFUSES to run
    scripts rather than falling back to unsandboxed execution — see
    ``run_script``'s preflight. (Earlier phrasing of this docstring
    said "falls back to running unsandboxed with a prominent warning"
    — that was never actually true of the shipped behaviour; refusing
    outright is the deliberate, harder posture.)
    """
    # Stable path on macOS. shutil.which may miss it if /usr/bin isn't
    # first on PATH in some shells.
    if Path("/usr/bin/sandbox-exec").is_file():
        return "/usr/bin/sandbox-exec"
    return shutil.which("sandbox-exec")


def find_bwrap() -> str | None:
    """Return the path to Linux's bubblewrap (``bwrap``), or None if
    missing.

    bwrap is the Linux confinement backend, playing the same role
    ``sandbox-exec`` plays on macOS: unprivileged (no setuid/root
    needed — it works entirely through user + mount + pid namespaces,
    available on any kernel with unprivileged user namespaces enabled,
    which is the default on mainstream distributions since well
    before this was written), builds a from-scratch filesystem view
    for the child rather than punching holes in the real one, and
    unshares the network namespace so the child has no network
    interface at all — not merely a firewalled one. See
    ``executor._bwrap_argv`` for how the actual confinement policy is
    built, and ``executor.run_script`` for the platform branch that
    picks this over ``sandbox_exec``.
    """
    if Path("/usr/bin/bwrap").is_file():
        return "/usr/bin/bwrap"
    return shutil.which("bwrap")


# ---------------------------------------------------------------------------
# Windows confinement backend: AppContainer + Job Objects, via
# ``sift.win_appcontainer``. Two-tier health check, same shape as the
# macOS/Linux backends above:
#
#   1. ``find_appcontainer_support`` — cheap, "does this Windows version
#      even have the API surface" check (Windows 8+; ``CreateAppContainerProfile``
#      is unavailable on anything older). Mirrors ``find_sandbox_exec`` /
#      ``find_bwrap``'s "binary present" role.
#   2. ``appcontainer_probe_result`` — expensive, cached, EMPIRICAL check
#      that confinement actually holds on this specific machine (see
#      ``win_appcontainer.probe_appcontainer_health``'s docstring for why
#      this exists and isn't optional). Mirrors
#      ``sandbox_baseline_result`` / ``bwrap_baseline_result``'s role, but
#      goes further than either: those two confirm "a minimal profile
#      applies without erroring," which is a much weaker claim than
#      "we positively observed a denied read and a denied network
#      connect," because sandbox-exec/bwrap are mature, independently-
#      audited OS components this codebase has actually run against,
#      while the AppContainer code path has not.
_APPCONTAINER_PROBE_CACHE: "tuple[bool, str] | None" = None


def find_appcontainer_support() -> bool:
    """True if this machine is Windows 8+ with the AppContainer API
    surface available at all. Non-Windows always returns False —
    irrelevant there, mirrors ``find_sandbox_exec``/``find_bwrap``
    returning ``None`` off their own platform.

    Deliberately does NOT run the live health probe — that's a
    separate, expensive, cached call (``appcontainer_probe_result``)
    so this cheap check can be called freely (e.g. for a UI badge)
    without triggering a real sandboxed subprocess launch every time.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        from sift.win_appcontainer import _userenv  # noqa: F401 — presence check only
    except Exception:  # noqa: BLE001 — any import/DLL-load failure means "not available"
        return False
    return True


def appcontainer_probe_result() -> tuple[bool, str]:
    """Public, cached accessor for the AppContainer live health probe.

    Cached for the process lifetime, same as
    ``sandbox_baseline_result``/``bwrap_baseline_result`` — the probe
    actually launches a handful of real sandboxed subprocesses (see
    ``win_appcontainer.probe_appcontainer_health``), so it runs once
    per Sift session, not once per script submission.
    """
    global _APPCONTAINER_PROBE_CACHE
    if _APPCONTAINER_PROBE_CACHE is not None:
        return _APPCONTAINER_PROBE_CACHE
    if not sys.platform.startswith("win"):
        _APPCONTAINER_PROBE_CACHE = (True, "")
        return _APPCONTAINER_PROBE_CACHE
    if not find_appcontainer_support():
        _APPCONTAINER_PROBE_CACHE = (
            False, "AppContainer API surface not available on this machine",
        )
        return _APPCONTAINER_PROBE_CACHE
    from sift.win_appcontainer import probe_appcontainer_health
    _APPCONTAINER_PROBE_CACHE = probe_appcontainer_health()
    return _APPCONTAINER_PROBE_CACHE


@dataclass(frozen=True)
class Environment:
    r: Tool | None
    stata: Tool | None
    python: Tool | None
    sandbox_exec: str | None
    # Linux confinement backend. ``None`` on macOS (irrelevant there)
    # and on any Linux system without bubblewrap installed.
    bwrap: str | None = None
    # Windows confinement backend presence. True only means
    # "the API surface exists" — see ``has_sandbox_backend`` and
    # ``executor.run_script``'s win32 branch for why this alone is
    # NOT sufficient to run a script unsandboxed; the live probe
    # (``appcontainer_probe_result``) is the actual gate.
    appcontainer_support: bool = False

    def has_any_runtime(self) -> bool:
        # "Has the binary" — a Python install with missing packages
        # still counts here so the banner says "Python detected" and
        # the executor can refuse with a specific
        # please-install-pandas message. has_runnable_runtime() is
        # the strict variant.
        return any(t is not None for t in (self.r, self.stata, self.python))

    def has_sandbox_backend(self) -> bool:
        """True if THIS platform's confinement backend is present AND
        (Windows only) has passed its live empirical health probe.

        Mirrors the platform branch in ``executor.run_script``'s
        preflight exactly: darwin checks ``sandbox_exec``, linux
        checks ``bwrap``, win32 checks ``appcontainer_support`` AND
        the cached probe result, anything else has no supported
        backend and is always False. Centralised here (rather than
        re-implemented per test file) so test skip-gates and the real
        executor preflight can never silently drift apart — a test
        file hand-rolling ``sandbox_exec is None`` would keep
        skipping forever on Linux even after bwrap support landed,
        which is exactly the bug this method exists to prevent.

        The win32 branch intentionally checks the probe cache rather
        than just ``self.appcontainer_support`` — a machine can have
        the AppContainer API surface present (Windows 8+) while the
        live probe still fails (confinement doesn't actually hold,
        or the researcher's account lacks the rights
        ``CreateAppContainerProfile`` needs) and this method must
        report False in that case. The darwin/linux branches now
        apply the SAME two-gate discipline via
        ``sandbox_baseline_result``/``bwrap_baseline_result``: binary
        presence alone used to be treated as sufficient there, which
        let this method (and anything gating on it) disagree with
        ``sift --doctor`` — the doctor already distinguished "backend
        missing" from "backend present but can't apply a minimal
        profile" via those same two functions, but this method and
        the executor's own preflight (see ``run_script``) didn't
        check the second gate, so a researcher could see "sandbox:
        blocked, baseline check fails" from ``--doctor`` while a
        script submission still attempted (and failed more
        confusingly) a real sandboxed run.
        """
        platform = sys.platform
        if platform == "darwin":
            if self.sandbox_exec is None:
                return False
            baseline_ok, _ = sandbox_baseline_result()
            return baseline_ok
        if platform.startswith("linux"):
            if self.bwrap is None:
                return False
            baseline_ok, _ = bwrap_baseline_result()
            return baseline_ok
        if platform.startswith("win"):
            if not self.appcontainer_support:
                return False
            probe_ok, _ = appcontainer_probe_result()
            return probe_ok
        return False


def detect_environment() -> Environment:
    return Environment(
        r=find_r(),
        stata=find_stata(),
        python=find_python(),
        sandbox_exec=find_sandbox_exec(),
        bwrap=find_bwrap(),
        appcontainer_support=find_appcontainer_support(),
    )


# ---------------------------------------------------------------------------
# Version probing
# ---------------------------------------------------------------------------

def _r_version(binary: str) -> str | None:
    """Run `Rscript --version` and extract a short version string."""
    from sift.executor import _filter_env

    try:
        out = run_bounded_capture(
            [binary, "--version"],
            timeout=5,
            env=_filter_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # R writes the version banner to stderr on some versions, stdout on
    # others. Check both.
    text = (out.stdout + out.stderr).strip()
    first_line = text.split("\n", 1)[0] if text else ""
    return first_line or None


def _python_version(binary: str) -> str | None:
    """Run ``python --version`` and return the first banner line.

    ``--version`` short-circuits CPython startup before ``site.py``
    runs, so ``sitecustomize`` / ``usercustomize`` and the inherited
    ``PYTHONPATH`` can't execute code via this probe. The filtered
    env + no ``-I`` here is intentional: ``-I`` doesn't compose
    with bare ``--version`` on all Python versions and the version
    flag doesn't load anything off sys.path anyway. The filtered
    env still strips parent-process secrets (``ANTHROPIC_API_KEY``,
    AWS creds) before the subprocess inherits them, matching the
    other probes.
    """
    from sift.executor import _filter_env
    try:
        out = run_bounded_capture(
            [binary, "--version"],
            timeout=5,
            env=_filter_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout + out.stderr).strip()
    first_line = text.split("\n", 1)[0] if text else ""
    return first_line or None


def _python_missing_packages(
    binary: str, required: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the subset of ``required`` packages the interpreter at
    ``binary`` cannot import. An empty tuple means "all present."

    Runs a single ``python -c`` import probe rather than one
    subprocess per package (faster, and survives an interpreter
    that's slow to start). Any non-import-error failure
    (interpreter crash, timeout) is conservatively reported as
    "all packages missing" so the executor's missing-packages
    branch trips and surfaces a coherent error to the researcher.

    Uses ``importlib.util.find_spec(pkg)`` rather than
    ``__import__(pkg)``. ``__import__`` executes the package's
    ``__init__.py`` (and any imports it triggers) OUTSIDE the
    analysis sandbox: this probe runs at app startup and after
    ``install_packages``, with full filesystem and network. A
    package masquerading as ``pandas`` / ``statsmodels`` would
    get code execution during detection. ``find_spec`` only
    consults sys.path finders — for top-level package names it
    returns metadata without importing the package, so no
    ``__init__.py`` runs.

    The probe is launched with ``-I`` (isolated mode) so the
    inherited ``PYTHONPATH`` and the user-site ``usercustomize.py``
    can't inject startup code. The Sift package dir is added to
    ``sys.path`` explicitly inside the probe rather than via
    ``PYTHONPATH`` env (which ``-I`` ignores). Without ``-I`` an
    attacker-controlled ``sitecustomize.py`` / ``usercustomize.py``
    on the inherited path would execute at every detection run.
    """
    if not required:
        return ()
    # Lazy import: ``package_installer`` and ``executor`` are sibling
    # modules and cheap to import, but keeping them lazy avoids any
    # import-cycle surprise if env_detect ever gets pulled in earlier
    # in startup.
    from sift.package_installer import sift_python_pkg_dir
    from sift.executor import _filter_env
    pkg_dir = str(sift_python_pkg_dir(binary))
    # ``find_spec`` returns ``None`` when the package isn't on
    # sys.path. Wrap in try/except so a finder that raises (rare,
    # but possible with broken namespace packages) doesn't mark
    # ALL packages missing — only the offending one.
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {pkg_dir!r})\n"
        "import importlib.util\n"
        f"required = {list(required)!r}\n"
        "missing = []\n"
        "for pkg in required:\n"
        "    try:\n"
        "        spec = importlib.util.find_spec(pkg)\n"
        "    except Exception:\n"
        "        spec = None\n"
        "    if spec is None:\n"
        "        missing.append(pkg)\n"
        "sys.stdout.write(json.dumps(missing))\n"
    )
    # Filter the probe's env through the same allowlist the executor
    # uses for analysis scripts. The probe runs OUTSIDE the script
    # sandbox and with no network deny. Without the filter, the
    # interpreter inherits secrets like ``ANTHROPIC_API_KEY`` / AWS
    # credentials from the parent process env. The script executor's
    # ``_filter_env`` is the canonical allowlist; using it here keeps
    # the two surfaces aligned.
    probe_env = _filter_env(dict(os.environ))
    try:
        out = run_bounded_capture(
            [binary, "-I", "-B", "-c", probe],
            timeout=10,
            env=probe_env,
        )
    except (OSError, subprocess.SubprocessError):
        return tuple(required)
    if out.returncode != 0:
        return tuple(required)
    try:
        import json as _json
        result = _json.loads(out.stdout.strip() or "[]")
    except (ValueError, TypeError):
        return tuple(required)
    if not isinstance(result, list):
        return tuple(required)
    return tuple(str(x) for x in result)


def _python_prefixes(binary: str) -> tuple[str, ...]:
    """Return the deduped ``sys.prefix`` / ``sys.exec_prefix`` /
    ``sys.base_prefix`` / ``sys.base_exec_prefix`` paths for the
    given Python interpreter.

    These paths feed the executor's sandbox profile so the
    interpreter can read its own stdlib + site-packages — without
    them, a venv-based Python (which lives outside the system trees
    the default sandbox already covers) would fail to load even the
    standard library inside the sandbox.

    The base prefixes matter for venvs specifically: a venv's
    ``sys.prefix`` holds only ``site-packages`` and bin stubs, while
    the stdlib stays under ``sys.base_prefix`` (the parent install).
    When that parent is itself outside the system trees — canonically
    a uv-managed CPython under ``~/.local/share/uv/python/`` — a
    profile granting only ``sys.prefix`` lets the interpreter launch
    and then die with ``Failed to import encodings module`` before
    main(). Homebrew / python.org installs are unaffected (base ==
    prefix, deduped below).

    ``-I`` (isolated mode) + filtered env: the probe runs OUTSIDE
    the analysis sandbox at startup. Without ``-I``, an inherited
    ``PYTHONPATH`` pointing at attacker-controlled ``sitecustomize.py``
    would execute code during the probe. The probe only reads
    ``sys.prefix`` / ``sys.exec_prefix``, which are populated
    independently of the user/parent PYTHONPATH, so ``-I`` is safe
    here.

    Best-effort: a probe failure returns an empty tuple. The
    sandbox falls back to the default system trees, which works for
    Apple's bundled python3 and Homebrew installs but breaks venvs.
    """
    from sift.executor import _filter_env
    try:
        out = run_bounded_capture(
            [
                binary, "-I", "-B", "-c",
                "import sys\n"
                "for p in (sys.prefix, sys.exec_prefix, "
                "sys.base_prefix, sys.base_exec_prefix):\n"
                "    print(p)\n",
            ],
            timeout=5,
            env=_filter_env(dict(os.environ)),
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if out.returncode != 0:
        return ()
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    # Canonicalization is load-bearing for bubblewrap/SBPL, both of which
    # require unambiguous absolute roots. Dedupe preserves prefix order.
    return _existing_roots(*lines)
