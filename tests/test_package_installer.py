"""Security and runtime-path regressions for ``sift.package_installer``.

The protected behaviors are:

1. Python install path mismatch: ``pip install --user`` writes to a
   directory the executor (``python -I``) and the sandbox both refuse
   to read from, so ``submit_script`` fails with
   ``ModuleNotFoundError`` after a successful install. Installs route
   through ``--target <sift_python_pkg_dir>`` and wire that directory into
   the executor preamble, sandbox, and environment probe. These tests pin the
   command shape and its integration with the executor.

2. Package validator allowed pip option injection: the previous
   ``^[A-Za-z0-9._-]+$`` matched ``-r``, ``-e``, ``--no-index``,
   ``.``, ``..`` — pip parses any of those as flags / local-path
   installs rather than registry names. The validator now requires
   a leading alphanumeric char and the pip argv has a ``--``
   end-of-options separator before the package list as defense-in-
   depth.

3. Stata command joiner used ``;``: Stata's default delimiter inside
   a do-file is a newline, so the joined string was a syntax error
   even for a single-package reinstall. The joiner now emits one
   line per command.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Package-name validator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hostile", [
    "-r",
    "-e",
    "--no-index",
    "--target",
    ".",
    "..",
    "-rrequirements.txt",
    "-",
])
def test_validate_names_rejects_pip_option_shapes(hostile: str) -> None:
    """Each value would be parsed by pip as an option or local path.

    The leading-alphanumeric anchor on the name regex closes the bypass.
    """
    from sift.package_installer import _validate_names

    valid, rejected = _validate_names([hostile])
    assert valid == []
    assert len(rejected) == 1
    assert rejected[0].status == "failed"
    assert "rejected" in rejected[0].detail


@pytest.mark.parametrize("legit", [
    "matplotlib",
    "scipy",
    "scikit-learn",
    "numpy",
    "statsmodels",
    "pandas",
    # Names with internal dots / dashes / underscores stay valid;
    # the anchor only forbids LEADING punctuation.
    "ggplot2",
    "data.table",
    "py4j",
    "google-cloud-storage",
])
def test_validate_names_accepts_canonical_registry_names(legit: str) -> None:
    """The tightened regex must not regress on real PyPI / CRAN /
    SSC names — those still start with an alphanumeric char."""
    from sift.package_installer import _validate_names

    valid, rejected = _validate_names([legit])
    assert valid == [legit]
    assert rejected == []


def test_validate_names_rejects_overlong() -> None:
    from sift.package_installer import _validate_names, _NAME_MAX_LEN

    valid, rejected = _validate_names(["a" * (_NAME_MAX_LEN + 1)])
    assert valid == []
    assert len(rejected) == 1


def test_python_remove_refuses_package_not_in_sift_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDC + safety closure: ``pip uninstall`` has no ``--target`` —
    it resolves via ``sys.path`` and removes the first copy it
    finds. If the requested package is NOT in Sift's managed Python
    dir but IS in the researcher's system / user / venv
    site-packages, pip would happily remove it from there. A model
    that calls ``install_packages(action='remove', packages=['pandas'])``
    when Sift never installed pandas to its own dir could yank
    pandas from the researcher's broader Python environment —
    breaking Sift itself.

    The fix: before launching ``pip uninstall``, scan the Sift target
    dir's ``*.dist-info`` and ``*.egg-info`` and refuse any package
    name that isn't present there. The refusal surfaces as a
    ``skipped`` per-package status; the subprocess never starts if
    no eligible names remain.
    """
    import asyncio
    monkeypatch.setenv(
        "SIFT_PYTHON_PKG_BASE", str(tmp_path / "sift-pkgs"),
    )
    from sift.package_installer import (
        InstallResult,
        install_packages,
        sift_python_pkg_dir,
    )
    # Stub the env-detect to a Python that resolves to a real
    # binary path (it never runs, since we'll intercept the
    # subprocess).
    import sift.env_detect as _env_detect
    from sift.env_detect import Environment, Tool
    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    # The Sift target dir EXISTS and contains ONE installed package
    # (``managed``) but NOT the package the model is asking to
    # remove (``site_only``).
    target = sift_python_pkg_dir("/usr/bin/python3")
    target.mkdir(parents=True, exist_ok=True)
    (target / "managed-1.2.3.dist-info").mkdir()
    (target / "managed-1.2.3.dist-info" / "METADATA").write_text(
        "Name: managed\n", encoding="utf-8",
    )

    # Intercept subprocess.Popen for pip invocations only. The
    # installer now uses Popen + communicate (so it can killpg the
    # process tree on timeout / cancel). ``sift_python_pkg_dir``
    # still uses subprocess.run for its version probe, which goes
    # through Popen internally — let those passthrough to the real
    # Popen by checking ``start_new_session`` (only the install path
    # sets it).
    pip_launched: list[list[str]] = []
    import subprocess as _subprocess
    real_popen = _subprocess.Popen

    class _FakePopen:
        def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
            self._cmd = list(cmd) if isinstance(cmd, list) else cmd
            self.pid = -1
            self.returncode = 0

        def communicate(self, input=None, timeout=None):  # type: ignore[no-untyped-def]
            return ("", "")

        def kill(self) -> None:  # pragma: no cover — never reached
            pass

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        is_install_call = kwargs.get("start_new_session")
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-m"
            and cmd[2] == "pip"
        )
        if is_install_call and is_pip:
            pip_launched.append(list(cmd))
            return _FakePopen(cmd, **kwargs)
        return real_popen(cmd, **kwargs)
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

    # Case 1: removing a package that ISN'T in Sift's target. The
    # request must fail cleanly without pip ever being launched.
    result = asyncio.run(install_packages(
        language="Python", packages=["site_only"], action="remove",
    ))
    assert isinstance(result, InstallResult)
    assert result.error is not None
    assert pip_launched == [], (
        "pip uninstall must NOT be launched for a package that "
        "isn't in Sift's target dir — otherwise pip would remove "
        "it from the researcher's broader Python environment"
    )
    statuses_by_name = {s.name: s for s in result.statuses}
    assert "site_only" in statuses_by_name
    assert statuses_by_name["site_only"].status == "skipped"

    # Case 2: a MIX of eligible + non-eligible names. The eligible
    # one goes through; the non-eligible one is skipped, never
    # reaches pip.
    pip_launched.clear()
    result2 = asyncio.run(install_packages(
        language="Python", packages=["managed", "site_only"],
        action="remove",
    ))
    assert len(pip_launched) == 1
    pip_argv = pip_launched[0]
    assert "managed" in pip_argv
    assert "site_only" not in pip_argv, (
        "the non-eligible package leaked into the pip argv — that "
        "would let pip uninstall it from site-packages"
    )
    statuses2_by_name = {s.name: s for s in result2.statuses}
    assert statuses2_by_name["site_only"].status == "skipped"


@pytest.mark.parametrize("name", [
    "sift",
    "Sift",
    "SIFT",
])
def test_python_install_refuses_sift_distribution_name(
    name: str, tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python ``sift`` helper module (``sift.from_lm``,
    ``sift.result``, ``sift.plot_*``) is staged onto every script's
    sys.path by the executor preamble. A model that calls
    ``install_packages(language='Python', packages=['sift'])`` is
    confused: there's nothing to fetch (the helpers are already
    importable), and the literal ``sift`` distribution on PyPI is an
    unrelated empty placeholder owned by another author (~1 KB,
    metadata only, no module code). Installing it does not provide
    the helpers and leaves a useless ``.dist-info`` in the Sift pkg
    dir.

    The guard rejects the name (PEP 503 normalised: case-folded,
    ``-``/``_``/``.`` collapsed) at the installer boundary BEFORE pip
    ever launches, and surfaces a ``skipped`` per-package status with
    the instruction to just ``import sift``.
    """
    import asyncio
    monkeypatch.setenv(
        "SIFT_PYTHON_PKG_BASE", str(tmp_path / "sift-pkgs"),
    )
    from sift.package_installer import InstallResult, install_packages
    import sift.env_detect as _env_detect
    from sift.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    # Track every pip subprocess that gets launched — must remain
    # empty for the sift-only case.
    pip_launched: list[list[str]] = []
    import subprocess as _subprocess
    real_popen = _subprocess.Popen

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        is_install_call = kwargs.get("start_new_session")
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-m"
            and cmd[2] == "pip"
        )
        if is_install_call and is_pip:
            pip_launched.append(list(cmd))

            class _FakePopen:
                pid = -1
                returncode = 0

                def communicate(self, input=None, timeout=None):  # noqa: ARG002
                    return ("", "")

                def kill(self) -> None:  # pragma: no cover
                    pass

            return _FakePopen()
        return real_popen(cmd, **kwargs)
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

    result = asyncio.run(install_packages(
        language="Python", packages=[name], action="install",
    ))

    assert isinstance(result, InstallResult)
    assert pip_launched == [], (
        f"pip must NOT launch for the {name!r} distribution name — "
        "the runtime helpers are preloaded by the executor preamble"
    )
    assert result.error is not None
    statuses_by_name = {s.name: s for s in result.statuses}
    assert name in statuses_by_name
    blocked = statuses_by_name[name]
    assert blocked.status == "skipped"
    assert "import sift" in blocked.detail
    assert "placeholder" in blocked.detail


def test_python_install_blocks_sift_alongside_legitimate_packages(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed-list case: a model request like ``['pandas', 'sift']``
    must let ``pandas`` through to pip while ``sift`` is filtered
    out as ``skipped``. Don't tank the entire call because of one
    bad name."""
    import asyncio
    monkeypatch.setenv(
        "SIFT_PYTHON_PKG_BASE", str(tmp_path / "sift-pkgs"),
    )
    from sift.package_installer import install_packages
    import sift.env_detect as _env_detect
    from sift.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    pip_launched: list[list[str]] = []
    import subprocess as _subprocess
    real_popen = _subprocess.Popen

    def _fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        is_install_call = kwargs.get("start_new_session")
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-m"
            and cmd[2] == "pip"
        )
        if is_install_call and is_pip:
            pip_launched.append(list(cmd))

            class _FakePopen:
                pid = -1
                returncode = 0

                def communicate(self, input=None, timeout=None):  # noqa: ARG002
                    return ("", "")

                def kill(self) -> None:  # pragma: no cover
                    pass

            return _FakePopen()
        return real_popen(cmd, **kwargs)
    monkeypatch.setattr(_subprocess, "Popen", _fake_popen)

    result = asyncio.run(install_packages(
        language="Python", packages=["pandas", "sift"], action="install",
    ))

    assert len(pip_launched) == 1, (
        "exactly one pip invocation expected — for pandas only"
    )
    pip_argv = pip_launched[0]
    assert "pandas" in pip_argv
    assert "sift" not in pip_argv, (
        "the sift name leaked into the pip argv — guard must run "
        "before pip dispatch"
    )
    statuses_by_name = {s.name: s for s in result.statuses}
    assert statuses_by_name["sift"].status == "skipped"
    assert statuses_by_name["pandas"].status == "ok"


def test_python_remove_handles_pep503_name_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip normalises distribution names per PEP 503 — ``scikit_learn``
    on disk is ``scikit-learn`` to the consumer, and vice versa. The
    Sift-target presence check must match how pip wrote the dir, so
    a remove request with the underscore form still matches the
    dash form on disk (and vice versa).
    """
    monkeypatch.setenv(
        "SIFT_PYTHON_PKG_BASE", str(tmp_path / "sift-pkgs"),
    )
    from sift.package_installer import (
        _python_packages_installed_in_sift_target,
        sift_python_pkg_dir,
    )
    target = sift_python_pkg_dir("/usr/bin/python3")
    target.mkdir(parents=True, exist_ok=True)
    # On-disk: ``scikit_learn-1.0.dist-info`` (underscore form).
    (target / "scikit_learn-1.0.dist-info").mkdir()
    # Request uses the dash form.
    found = _python_packages_installed_in_sift_target(
        target, ["scikit-learn"],
    )
    assert found == {"scikit-learn"}
    # And vice versa: dash on disk, underscore in request.
    (target / "another-pkg-2.0.dist-info").mkdir()
    found2 = _python_packages_installed_in_sift_target(
        target, ["another_pkg"],
    )
    assert found2 == {"another_pkg"}


def test_install_packages_tool_scrubs_credentials_from_raw_excerpts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDC closure: the ``install_packages`` MCP tool returns
    ``raw_stdout_excerpt`` and ``raw_stderr_excerpt`` on failure so
    the model can diagnose. The script-sandbox path runs everything
    through ``error_summary.extract_debug_excerpt`` (language-
    anchored extraction + credential/path scrub) — the install path
    skips the language anchor but MUST NOT skip the scrub.

    The headline leak we lock against is the private pip index URL
    that pip echoes on every run from ``~/.pip/pip.conf`` (or
    ``PIP_INDEX_URL``):

        Looking in indexes: https://USER:TOKEN@private-pypi.acme.com/simple

    Without the scrub, the embedded user:token rides the failure
    response straight into the model's context. The fix pipes both
    raw excerpts through ``error_summary.scrub_raw_output`` before
    they reach the response payload.
    """
    import asyncio
    import json

    from sift.package_installer import InstallResult
    from sift.tools import HANDLERS
    import sift.tools as tools_mod

    fake_result = InstallResult(
        language="Python",
        action="install",
        statuses=(),
        raw_stdout=(
            "Looking in indexes: https://leaked_user:leaked_token@"
            "private-pypi.acme.com/simple\n"
            "Collecting pandas\n"
        ),
        raw_stderr=(
            "ERROR: HTTPSConnectionPool(host='private-pypi.acme.com', "
            "port=443): Max retries exceeded\n"
            "OPENAI_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567890\n"
            "Failed at /Users/jdoe/.cache/pip/wheels/build.log\n"
        ),
        error="installer exited 1",
        duration_seconds=0.5,
    )

    async def _fake_install(language, packages, action, proc_register=None):  # type: ignore[no-untyped-def]
        return fake_result

    monkeypatch.setattr(
        tools_mod,
        "install_packages",
        tools_mod.install_packages,
    )
    # Patch the module-level import target that the tool handler
    # reaches for inside its body.
    import sift.package_installer as pkg_mod
    monkeypatch.setattr(pkg_mod, "install_packages", _fake_install)
    # The tool now gates installs behind a researcher-side
    # confirmation modal; stub the approval so the install runs and
    # we can assert on the failure-path scrub behavior.
    import sift.install_confirmation as ic_mod

    async def _approve(**_kwargs):
        return True
    monkeypatch.setattr(ic_mod, "request_confirmation", _approve)

    payload = asyncio.run(HANDLERS["install_packages"]({
        "language": "Python",
        "packages": ["pandas"],
        "action": "install",
    }))
    body = json.loads(next(
        b for b in payload["content"] if b.get("type") == "text"
    )["text"])
    assert body["status"] == "error"
    stdout_excerpt = body["raw_stdout_excerpt"]
    stderr_excerpt = body["raw_stderr_excerpt"]

    # Embedded URL credentials are gone; scheme + host are preserved
    # so a reader can still tell what was happening.
    assert "leaked_user" not in stdout_excerpt
    assert "leaked_token" not in stdout_excerpt
    assert "[redacted-credential]" in stdout_excerpt
    assert "https://" in stdout_excerpt

    # OpenAI-style key in stderr is redacted.
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in stderr_excerpt
    assert "[redacted-credential]" in stderr_excerpt

    # Absolute path is reduced to its basename.
    assert "/Users/jdoe" not in stderr_excerpt
    assert "build.log" in stderr_excerpt


def test_install_packages_caps_list_length() -> None:
    """A massive package list shouldn't launch a single multi-hour
    invocation the researcher can't easily interrupt. The per-call
    cap rejects oversized batches before any subprocess fires.
    """
    import asyncio

    from sift.package_installer import install_packages as _do_install

    # 60 names — over the 50-package cap, all individually well-formed.
    names = [f"pkg{i:03d}" for i in range(60)]
    result = asyncio.run(_do_install("Python", names, "install"))

    assert result.error is not None
    assert "too many" in result.error.lower() or "cap" in result.error.lower()
    # No subprocess fired, so no statuses, no stdout, no stderr.
    assert result.statuses == ()
    assert result.raw_stdout == ""
    assert result.raw_stderr == ""


# ---------------------------------------------------------------------------
# 1. Python install command shape
# ---------------------------------------------------------------------------

def test_python_command_install_uses_target_not_user(tmp_path, monkeypatch) -> None:
    """``--user`` was the bug — it writes to a path the executor's
    isolated-mode interpreter and the sandbox both ignore. The fix
    is ``--target <sift_python_pkg_dir>``. Pin the argv shape."""
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path))
    from sift.package_installer import _python_command, sift_python_pkg_dir

    cmd = _python_command("/usr/bin/python3", ["matplotlib", "scipy"], "install")
    expected_target = str(sift_python_pkg_dir("/usr/bin/python3"))

    assert "--user" not in cmd, "must not use --user; see docstring"
    assert "--target" in cmd
    assert cmd[cmd.index("--target") + 1] == expected_target
    assert "--" in cmd
    # Package names appear AFTER the ``--`` end-of-options separator.
    sep = cmd.index("--")
    assert cmd[sep + 1:] == ["matplotlib", "scipy"]


def test_python_command_reinstall_uses_force_reinstall_no_deps(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path))
    from sift.package_installer import _python_command

    cmd = _python_command("/usr/bin/python3", ["matplotlib"], "reinstall")
    assert "--force-reinstall" in cmd
    assert "--no-deps" in cmd
    assert "--target" in cmd
    assert "--user" not in cmd
    assert "--" in cmd


def test_python_command_remove_uses_uninstall(tmp_path, monkeypatch) -> None:
    """``pip uninstall`` has no ``--target``; the runner separately
    sets PYTHONPATH so pip locates the package via sys.path. The
    argv shape here is just ``uninstall -y -- pkg…``."""
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path))
    from sift.package_installer import _python_command

    cmd = _python_command("/usr/bin/python3", ["matplotlib"], "remove")
    assert "uninstall" in cmd
    assert "-y" in cmd
    # End-of-options separator is present even on uninstall — keeps
    # the package-name boundary identical across actions.
    assert "--" in cmd


def test_python_command_creates_target_dir(tmp_path, monkeypatch) -> None:
    """The ``--target`` dir must exist before pip writes; if it
    doesn't, pip fails with a confusing path error. Verify the
    command builder eagerly creates it."""
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path))
    from sift.package_installer import _python_command, sift_python_pkg_dir

    expected = sift_python_pkg_dir("/usr/bin/python3")
    assert not expected.exists()
    _python_command("/usr/bin/python3", ["matplotlib"], "install")
    assert expected.is_dir()


# ---------------------------------------------------------------------------
# 4. Stata joiner — newlines, not semicolons
# ---------------------------------------------------------------------------

def test_stata_command_install_uses_newlines() -> None:
    from sift.package_installer import _stata_command

    cmd, stdin = _stata_command(
        "/Applications/Stata/StataMP.app/Contents/MacOS/StataMP",
        ["estout", "ftools"],
        "install",
    )
    # No semicolon separators (Stata's default delimiter is \n).
    # Internal commas are fine — they're option separators inside a
    # single ``ssc install pkg, replace`` command.
    assert ";" not in stdin
    assert stdin.count("\n") >= 2  # one per package, plus trailing
    assert "ssc install estout, replace" in stdin
    assert "ssc install ftools, replace" in stdin
    assert cmd[-1] == "install.do"
    assert "/dev/stdin" not in cmd


def test_stata_command_single_package_reinstall_no_semicolons() -> None:
    """Even a single-package reinstall used to embed an internal
    ``;`` (``capture ado uninstall p; ssc install p, replace``),
    failing on its own — not just on multi-package joins. Pin the
    single-package behavior as well as multi-package joins."""
    from sift.package_installer import _stata_command

    _, stdin = _stata_command("/path/to/stata", ["estout"], "reinstall")
    assert ";" not in stdin
    assert "capture ado uninstall estout" in stdin
    assert "ssc install estout, replace" in stdin
    # Each on its own line.
    lines = [ln for ln in stdin.splitlines() if ln.strip()]
    assert "capture ado uninstall estout" in lines
    assert "ssc install estout, replace" in lines


def test_stata_command_remove_uses_newlines() -> None:
    from sift.package_installer import _stata_command

    _, stdin = _stata_command("/path/to/stata", ["a", "b"], "remove")
    assert ";" not in stdin
    lines = [ln for ln in stdin.splitlines() if ln.strip()]
    assert "ado uninstall a" in lines
    assert "ado uninstall b" in lines


def test_stata_package_install_without_stata_explains_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import sift.env_detect as env_detect
    from sift.env_detect import Environment
    from sift.package_installer import install_packages

    monkeypatch.setattr(
        env_detect,
        "detect_environment",
        lambda: Environment(
            r=None, stata=None, python=None, sandbox_exec=None, bwrap=None,
        ),
    )
    result = asyncio.run(install_packages("Stata", ["estout"]))
    assert result.error is not None
    assert "existing licensed Stata runtime" in (result.error or "")
    assert "not required to open or analyze .dta" in (result.error or "")


# ---------------------------------------------------------------------------
# Executor integration — the install dir reaches both surfaces
# ---------------------------------------------------------------------------

def test_python_preamble_adds_sift_pkg_dir_to_sys_path(
    tmp_path, monkeypatch,
) -> None:
    """The preamble must add the Sift pkg dir to ``sys.path`` —
    without this, ``-I`` mode never sees Sift-installed packages even
    if pip wrote them to the right place. This is half of the
    install/import alignment fix."""
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    import sift.env_detect as env_detect
    from sift.env_detect import Tool
    from sift.executor import _write_script
    from sift.package_installer import sift_python_pkg_dir

    monkeypatch.setattr(
        env_detect,
        "find_python",
        lambda: Tool(name="Python", binary=sys.executable),
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "lib").mkdir()
    script_path = _write_script(run_dir, "Python", "print('hi')\n")

    text = script_path.read_text(encoding="utf-8")
    expected_pkg_dir = str(sift_python_pkg_dir(sys.executable))
    assert repr(expected_pkg_dir) in text, (
        "preamble must insert sift_python_pkg_dir on sys.path so "
        "Sift-installed packages resolve under -I mode"
    )


def test_python_preamble_lib_dir_wins_over_pkg_dir(
    tmp_path, monkeypatch,
) -> None:
    """The staged Sift runtime (``<run_dir>/lib/sift.py``) must take
    precedence over anything in ``sift_python_pkg_dir`` on ``sys.path``.

    Concrete threat: a model-authored ``install_packages(["sift"])``
    call (or any third-party wheel that ships a top-level ``sift``
    module) writes to ``sift_python_pkg_dir``. If pkg_dir is on
    ``sys.path`` ahead of ``lib_dir``, the user script's
    ``import sift`` resolves to the installed (potentially malicious)
    module instead of the staged runtime — bypassing the
    ``SIFT_RUN_TOKEN`` env-pop and the authenticity-token machinery
    the runtime owns. The preamble must therefore order the inserts
    so ``lib_dir`` ends up at ``sys.path[0]``.

    The same ordering also ensures ``pkg_dir`` sits after stdlib (via
    ``append`` rather than ``insert(0, ...)``), so an installed
    package can't shadow ``os`` / ``json`` / other stdlib modules.
    """
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    import sift.env_detect as env_detect
    from sift.env_detect import Tool
    from sift.executor import _write_script
    from sift.package_installer import sift_python_pkg_dir

    monkeypatch.setattr(
        env_detect,
        "find_python",
        lambda: Tool(name="Python", binary=sys.executable),
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lib_dir = run_dir / "lib"
    lib_dir.mkdir()
    script_path = _write_script(run_dir, "Python", "print('hi')\n")

    text = script_path.read_text(encoding="utf-8")
    pkg_dir = str(sift_python_pkg_dir(sys.executable))
    # Both paths must be in the preamble.
    assert repr(str(lib_dir)) in text
    assert repr(pkg_dir) in text
    # ``lib_dir`` must be inserted at the front of sys.path (so it
    # wins over everything else). The literal we check for is the
    # ``insert(0, ...)`` form on lib_dir, paired with anything other
    # than ``insert(0, ...)`` on pkg_dir.
    assert f"insert(0, {str(lib_dir)!r})" in text, (
        "lib_dir must use insert(0, ...) so the staged runtime wins"
    )
    assert f"insert(0, {pkg_dir!r})" not in text, (
        "pkg_dir must NOT use insert(0, ...) — that would shadow "
        "lib_dir's sift.py with any package masquerading as 'sift', "
        "and shadow stdlib modules too"
    )


def test_sandbox_profile_grants_read_on_sift_pkg_dir(
    tmp_path, monkeypatch,
) -> None:
    """The other half of the alignment: even with the preamble's
    sys.path entry, the sandbox-exec read allowlist would still deny
    a script's import of ``sift_python_pkg_dir`` content. The
    executor must include it in ``extra_read_paths`` for Python
    runs."""
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    from sift.executor import _sandbox_profile_string
    from sift.package_installer import sift_python_pkg_dir

    pkg_dir = str(sift_python_pkg_dir(sys.executable))
    profile = _sandbox_profile_string(
        run_dir=tmp_path / "run",
        cwd=tmp_path / "cwd",
        home=tmp_path,
        extra_read_paths=(pkg_dir,),
    )
    # Path appears as an SBPL quoted string. Validate the same escaping
    # the profile writer applies so this remains meaningful on Windows,
    # where every native path contains backslashes.
    expected_literal = (
        '"' + pkg_dir.replace("\\", "\\\\").replace('"', '\\"') + '"'
    )
    assert expected_literal in profile


# ---------------------------------------------------------------------------
# sift_python_pkg_dir — the new single source of truth
# ---------------------------------------------------------------------------

def test_sift_python_pkg_dir_namespaces_by_version(tmp_path, monkeypatch) -> None:
    """C-extension wheels (numpy, scipy, …) are tagged for a specific
    CPython ABI. Mixing 3.11 and 3.12 wheels in one ``--target`` dir
    crashes at import. Per-version subdir avoids this."""
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path))
    import sys
    from sift.package_installer import sift_python_pkg_dir

    d = sift_python_pkg_dir(sys.executable)
    expected_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert d.parent == tmp_path
    assert d.name == expected_tag


# ---------------------------------------------------------------------------
# Subprocess env scrub — installer must not leak parent secrets
# ---------------------------------------------------------------------------

def test_install_subprocess_env_drops_parent_secrets(
    tmp_path, monkeypatch,
) -> None:
    """``install_packages`` runs pip / R / Stata installers with
    network access AND OUTSIDE the analysis sandbox, so any secret
    in the parent process env (API keys, AWS creds) is reachable
    by the installer's post-install hooks. The fix mirrors the
    executor's allowlist: the subprocess inherits only the env
    vars on ``_SUBPROCESS_ENV_ALLOWLIST``. Verify the symmetric
    contract by intercepting ``subprocess.run`` and inspecting
    the ``env`` argument it would have received.
    """
    import asyncio
    import sys as _sys

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-leaktest-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leaktest-openai")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "leaktest-aws")
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))

    captured: dict[str, dict[str, str] | None] = {"env": None}

    # Capture the install Popen call ONLY. ``_python_version_tag``
    # (called from ``sift_python_pkg_dir``) also uses the bounded Popen
    # helper; distinguish it from the actual pip command by argv.
    import subprocess as _sp
    real_popen = _sp.Popen

    class _FakePopen:
        def __init__(self, cmd, **kwargs):  # type: ignore[no-untyped-def]
            self._cmd = list(cmd) if isinstance(cmd, list) else cmd
            self.pid = -1
            self.returncode = 0
            # Only mark the install env — let probe calls go through
            # real Popen by raising NotImplementedError up through
            # the dispatch wrapper.
            captured["env"] = dict(kwargs.get("env") or {})

        def communicate(self, input=None, timeout=None):  # type: ignore[no-untyped-def]
            return ("", "")

        def kill(self) -> None:  # pragma: no cover
            pass

    def _dispatch_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        # Identify the install command (pip / Rscript / Stata
        # install) — those have ``start_new_session=True`` set,
        # which the executor's run_script / package_installer use
        # but the version probe does not.
        is_pip = (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1:3] == ["-m", "pip"]
        )
        if kwargs.get("start_new_session") and is_pip:
            return _FakePopen(cmd, **kwargs)
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(_sp, "Popen", _dispatch_popen)
    # Run the install branch synchronously so the test doesn't
    # depend on the asyncio thread-pool executor seeing our patched
    # Popen (which it does, but reasoning about the capture order
    # is simpler this way).
    async def _sync_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", _sync_to_thread)

    # Patch ``detect_environment`` to a fixed fake Environment.
    # Without this, the real env_detect calls ``subprocess.run`` to
    # probe interpreters — but ``run`` is patched to a no-op
    # returning empty output, so the probe sees no python and
    # ``install_packages`` bails before reaching the audited path.
    # ``install_packages`` imports ``detect_environment`` lazily, so
    # the patch lives on ``sift.env_detect`` (the source module),
    # not on ``sift.package_installer``.
    import sift.env_detect as _env_detect
    from sift.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary="/usr/bin/python3",
            version="Python 3.12.0",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setattr(_env_detect, "detect_environment", lambda: fake_env)

    from sift.package_installer import install_packages

    asyncio.run(install_packages(
        language="Python", packages=["pandas"], action="install",
    ))

    env = captured["env"]
    assert env is not None, "subprocess.Popen was not called with an env dict"
    # The three secrets we set MUST NOT cross to the installer.
    for leak in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        assert leak not in env, (
            f"{leak} leaked into installer env — package post-install "
            f"hooks run outside the analysis sandbox with network "
            f"access, so this is a credential-exfil channel"
        )
    # PATH (allowlisted) survives so pip can find python tooling.
    assert "PATH" in env
    # PYTHONPATH is the install-target injection — must point at our
    # sift pkg dir so ``pip uninstall`` resolves the right copy.
    assert "PYTHONPATH" in env
    assert str(tmp_path / "pkgs") in env["PYTHONPATH"]


def _patch_python_installer_command(monkeypatch, tmp_path, command: list[str]) -> None:
    """Route one installer call to a local, network-free test command."""
    import sift.env_detect as env_detect
    import sift.package_installer as package_installer
    from sift.env_detect import Environment, Tool

    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    monkeypatch.setattr(
        env_detect,
        "detect_environment",
        lambda: Environment(
            python=Tool(
                name="Python", binary=os.sys.executable,
                version="Python test", missing_packages=(),
                optional_missing_packages=(), extra_read_paths=(),
            ),
            r=None, stata=None, sandbox_exec=None,
        ),
    )
    monkeypatch.setattr(
        package_installer,
        "_python_command",
        lambda _binary, _packages, _action: command,
    )


def test_installer_output_is_bounded_while_both_pipes_are_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A noisy build hook cannot move its memory DoS into Sift's parent."""
    import asyncio
    import sys
    from sift.package_installer import (
        _INSTALLER_TRUNCATION_MARKER,
        _MAX_INSTALLER_STREAM_BYTES,
        install_packages,
    )

    payload_bytes = _MAX_INSTALLER_STREAM_BYTES + 512 * 1024
    code = (
        "import os\n"
        "chunk=b'x'*65536\n"
        f"remaining={payload_bytes}\n"
        "while remaining:\n"
        " part=chunk[:min(len(chunk),remaining)]\n"
        " os.write(1,part); os.write(2,part); remaining-=len(part)\n"
    )
    _patch_python_installer_command(
        monkeypatch, tmp_path, [sys.executable, "-c", code],
    )

    spawned: list[subprocess.Popen[str]] = []
    result = asyncio.run(
        install_packages(
            "Python", ["example-package"], proc_register=spawned.append,
        ),
    )

    assert result.error is None
    marker = _INSTALLER_TRUNCATION_MARKER.format(
        limit=_MAX_INSTALLER_STREAM_BYTES,
    )
    for captured in (result.raw_stdout, result.raw_stderr):
        assert captured.startswith("x" * 1024)
        assert captured.endswith(marker)
        assert captured.count(marker) == 1
        assert len(captured.encode("utf-8")) == (
            _MAX_INSTALLER_STREAM_BYTES + len(marker.encode("utf-8"))
        )
    assert len(spawned) == 1
    assert spawned[0].stdout is not None and spawned[0].stdout.closed
    assert spawned[0].stderr is not None and spawned[0].stderr.closed


@pytest.mark.skipif(os.name != "posix", reason="setsid regression is POSIX-specific")
def test_installer_normal_exit_kills_detached_marker_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A package hook cannot survive by daemonizing before pip exits."""
    import asyncio
    import signal
    import sys
    import time
    from sift.package_installer import install_packages

    pid_path = tmp_path / "detached.pid"
    child_code = (
        "import os,time,pathlib; os.setsid(); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time,pathlib; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
        f"path=pathlib.Path({str(pid_path)!r}); "
        "deadline=time.monotonic()+5; "
        "exec('while not path.exists() and time.monotonic()<deadline:\\n time.sleep(.01)')"
    )
    _patch_python_installer_command(
        monkeypatch, tmp_path, [sys.executable, "-c", parent_code],
    )

    detached_pid: int | None = None
    try:
        started = time.monotonic()
        result = asyncio.run(install_packages("Python", ["example-package"]))
        assert time.monotonic() - started < 8
        assert result.error is None
        assert pid_path.exists(), "the detached regression child never started"
        detached_pid = int(pid_path.read_text(encoding="utf-8"))

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(detached_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("detached installer child survived normal completion cleanup")
    finally:
        if detached_pid is not None:
            try:
                os.kill(detached_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="process lifecycle probe is POSIX-specific")
def test_installer_timeout_preserves_prefix_and_kills_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import signal
    import sys
    import time
    import sift.package_installer as package_installer

    pid_path = tmp_path / "timeout.pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "print('useful-early-diagnostic',flush=True); time.sleep(30)"
    )
    _patch_python_installer_command(
        monkeypatch, tmp_path, [sys.executable, "-c", code],
    )
    monkeypatch.setattr(package_installer, "_INSTALL_TIMEOUT_SECONDS", 0.25)

    process_pid: int | None = None
    try:
        result = asyncio.run(
            package_installer.install_packages("Python", ["example-package"]),
        )
        assert result.error is not None and "timed out" in result.error
        assert "useful-early-diagnostic" in result.raw_stdout
        process_pid = int(pid_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(process_pid, 0)
    finally:
        if process_pid is not None:
            try:
                os.kill(process_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="process lifecycle probe is POSIX-specific")
def test_installer_cancellation_kills_process_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import signal
    import sys
    import time
    from sift.package_installer import install_packages

    pid_path = tmp_path / "cancel.pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    _patch_python_installer_command(
        monkeypatch, tmp_path, [sys.executable, "-c", code],
    )

    async def _cancel() -> None:
        task = asyncio.create_task(
            install_packages("Python", ["example-package"]),
        )
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert pid_path.exists(), "installer did not start before cancellation"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    process_pid: int | None = None
    try:
        asyncio.run(_cancel())
        process_pid = int(pid_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(process_pid, 0)
    finally:
        if process_pid is not None:
            try:
                os.kill(process_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="process lifecycle probe is POSIX-specific")
def test_installer_unexpected_wait_failure_still_kills_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import signal
    import sys
    import time
    import sift.package_installer as package_installer

    pid_path = tmp_path / "unexpected.pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    _patch_python_installer_command(
        monkeypatch, tmp_path, [sys.executable, "-c", code],
    )
    real_wait = package_installer._BoundedInstallerCapture.wait

    def _fail_after_start(self, proc_stdin, timeout):  # type: ignore[no-untyped-def]
        self._start_readers()
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise RuntimeError("synthetic post-spawn failure")

    monkeypatch.setattr(
        package_installer._BoundedInstallerCapture, "wait", _fail_after_start,
    )
    process_pid: int | None = None
    try:
        with pytest.raises(RuntimeError, match="synthetic post-spawn failure"):
            asyncio.run(
                package_installer.install_packages(
                    "Python", ["example-package"],
                ),
            )
        process_pid = int(pid_path.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(process_pid, 0)
    finally:
        monkeypatch.setattr(
            package_installer._BoundedInstallerCapture, "wait", real_wait,
        )
        if process_pid is not None:
            try:
                os.kill(process_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_windows_shaped_installer_uses_tree_kill_fallback_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows tree cleanup still falls back to the direct Popen and reaps it."""
    import asyncio
    import subprocess as sp
    import sift.env_detect as env_detect
    import sift.package_installer as package_installer
    from sift.env_detect import Environment, Tool

    fake_env = Environment(
        python=Tool(
            name="Python", binary=os.sys.executable, version="Python test",
            missing_packages=(), optional_missing_packages=(),
            extra_read_paths=(),
        ),
        r=None, stata=None, sandbox_exec=None,
    )
    monkeypatch.setenv("SIFT_PYTHON_PKG_BASE", str(tmp_path / "pkgs"))
    monkeypatch.setattr(env_detect, "detect_environment", lambda: fake_env)
    monkeypatch.setattr(package_installer, "_IS_POSIX", False)

    real_popen = sp.Popen
    launched: dict[str, object] = {}

    class _WindowsShapedPopen:
        pid = 43210
        returncode = 0

        def __init__(self) -> None:
            self.kill_calls = 0
            self.wait_calls = 0

        def communicate(self, input=None, timeout=None):  # noqa: ARG002
            return "installed", ""

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, timeout=None):  # noqa: ARG002
            self.wait_calls += 1
            return 0

    fake_proc = _WindowsShapedPopen()

    def _dispatch(command, **kwargs):  # type: ignore[no-untyped-def]
        is_pip = (
            isinstance(command, list)
            and len(command) >= 3
            and command[1:3] == ["-m", "pip"]
        )
        if kwargs.get("start_new_session") and is_pip:
            launched["kwargs"] = kwargs
            return fake_proc
        return real_popen(command, **kwargs)

    monkeypatch.setattr(sp, "Popen", _dispatch)
    result = asyncio.run(
        package_installer.install_packages(
            "Python", ["example-package"], action="install",
        ),
    )

    assert result.error is None
    assert launched["kwargs"]["start_new_session"] is True  # type: ignore[index]
    assert fake_proc.kill_calls == 1
    assert fake_proc.wait_calls == 1
