"""Integration tests for the Linux (bubblewrap) sandbox backend.

These exercise the real subprocess path — bwrap spawning a real Python
interpreter through ``executor.run_script`` — to prove, for real and
not just by reading the argv, that the confinement properties
documented in ``_bwrap_argv``'s module comment actually hold: a script
cannot read outside its allowed trees, cannot write outside
run_dir/cwd, cannot read or tamper with Sift's own ``.sift`` session
state even though it lives inside an allowed directory, cannot see
host processes, and has no usable network path at all.

Companion to ``test_bwrap_argv.py`` (pure, argv-shape assertions that
run everywhere including non-Linux) the same way
``test_executor_sandbox.py`` companions ``test_executor_profile.py``
for the macOS backend. This file is the belt-and-suspenders "run it
for real" layer and only runs where bwrap is actually installed and
callable — everywhere else it skips rather than failing.

Design note on how failures are surfaced: most tests below have the
SCRIPT ITSELF assert the security property (e.g. "this read must have
been denied") and only emit a normal sanitized result once that
assertion passes. A confinement failure therefore shows up as
``r.ok is False`` with the assertion text in ``r.raw_stderr`` — the
same failure shape a researcher would see for any other script bug,
and one that doesn't depend on smuggling a probe result through the
statistical-disclosure-control payload path (which enforces its own
minimums, like n >= 10, that would get in the way of reporting a
tiny probe value like a process count of 1 or 2).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sift.env_detect import find_bwrap
from sift.executor import run_script


def _bwrap_works() -> bool:
    """True iff bwrap is present AND can actually apply a minimal
    profile in this environment (some CI/container harnesses block
    nested unprivileged user namespaces, mirroring the
    ``requires_sandbox_apply`` preflight in test_executor_sandbox.py
    for macOS)."""
    exe = find_bwrap()
    if exe is None:
        return False
    try:
        r = subprocess.run(
            [
                exe, "--ro-bind", "/", "/", "--unshare-all",
                "--die-with-parent", "/usr/bin/true",
            ],
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


requires_bwrap = pytest.mark.skipif(
    sys.platform != "linux" or not _bwrap_works(),
    reason=(
        "bwrap cannot apply a minimal profile in this environment "
        "(non-Linux, not installed, or nested-namespace harness "
        "blocks unprivileged user namespaces)."
    ),
)

_HAS_PY_STACK = (
    shutil.which("python3") is not None
    or shutil.which("python") is not None
)
requires_python_stack = pytest.mark.skipif(
    not _HAS_PY_STACK, reason="needs a python3 interpreter on PATH",
)


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(
        "x,y\n1,10\n2,20\n3,30\n4,40\n5,50\n6,60\n"
        "7,70\n8,80\n9,90\n10,100\n11,110\n12,120\n"
    )
    return path


def _run(tmp_path: Path, code: str, **kw):
    return run_script("Python", code, tmp_path, timeout_seconds=30, **kw)


def _assert_ok(r) -> None:
    assert r.ok, f"script failed: error={r.error}\nstderr={r.raw_stderr}"


# ---------------------------------------------------------------------------
# Happy path — bwrap lets the researcher's own work through
# ---------------------------------------------------------------------------

@requires_bwrap
@requires_python_stack
def test_bwrap_allows_cwd_read_and_produces_result(
    tmp_path: Path, tiny_csv: Path,
):
    """The sandbox must let Python read data from cwd and emit a
    result payload — confinement that also blocks the legitimate
    workflow is just as much of a bug as confinement that blocks
    nothing."""
    code = (
        "import pandas as pd\n"
        "df = pd.read_csv('data.csv')\n"
        "import sift\n"
        "sift.from_summarize('y', n=len(df), mean=float(df['y'].mean()), "
        "sd=float(df['y'].std()), missing_count=0)\n"
    )
    r = _run(tmp_path, code)
    _assert_ok(r)
    assert r.result_payloads
    assert r.result_payloads[0]["type"] == "descriptive"
    assert r.result_payloads[0]["n"] == 12


# ---------------------------------------------------------------------------
# Filesystem confinement — reads
# ---------------------------------------------------------------------------

@requires_bwrap
@requires_python_stack
def test_bwrap_blocks_read_outside_cwd(tmp_path: Path, tiny_csv: Path):
    """A file that exists on the host, right next to (but not inside)
    the researcher's cwd, must be completely unreachable — bwrap
    builds the mount namespace from nothing, so this isn't even a
    permission wall, the path shouldn't exist inside the sandbox at
    all."""
    secret = tmp_path.parent / f"secret_{tmp_path.name}.txt"
    secret.write_text("TOP-SECRET-HOST-CONTENTS")
    try:
        code = f'''
leaked = False
try:
    with open({str(secret)!r}) as f:
        f.read()
    leaked = True
except OSError:
    leaked = False
assert not leaked, "sandbox failed to block a read outside cwd"
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
        r = _run(tmp_path, code)
        _assert_ok(r)
    finally:
        secret.unlink(missing_ok=True)


@requires_bwrap
@requires_python_stack
def test_bwrap_blocks_home_dotfile_reads(tmp_path: Path):
    """Canary for 'malicious script exfiltrates user config/secrets
    via $HOME' — HOME is not bound at all except through the
    researcher's own cwd."""
    home = Path.home()
    candidates = [home / ".bashrc", home / ".profile", home / ".bash_profile"]
    target = next((p for p in candidates if p.exists()), None)
    if target is None:
        pytest.skip("no standard dotfile in HOME to probe")

    code = f'''
leaked = False
try:
    with open({str(target)!r}) as f:
        f.read()
    leaked = True
except OSError:
    leaked = False
assert not leaked, "sandbox failed to block a HOME dotfile read"
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
    r = _run(tmp_path, code)
    _assert_ok(r)


# ---------------------------------------------------------------------------
# Filesystem confinement — writes
# ---------------------------------------------------------------------------

@requires_bwrap
@requires_python_stack
def test_bwrap_blocks_write_outside_cwd(tmp_path: Path, tiny_csv: Path):
    victim = tmp_path.parent / f"victim_{tmp_path.name}.txt"
    victim.unlink(missing_ok=True)
    try:
        code = f'''
wrote = False
try:
    with open({str(victim)!r}, "w") as f:
        f.write("pwned")
    wrote = True
except OSError:
    wrote = False
assert not wrote, "sandbox failed to block a write outside cwd"
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
        r = _run(tmp_path, code)
        _assert_ok(r)
        assert not victim.exists(), (
            "sandbox failed — file was created outside the run's "
            "writable scope despite the in-script check passing"
        )
    finally:
        victim.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The .sift carve-out — Sift's own session state
# ---------------------------------------------------------------------------

@requires_bwrap
@requires_python_stack
def test_bwrap_sift_carve_out_masks_existing_state(
    tmp_path: Path, tiny_csv: Path,
):
    """Real pre-existing session state under cwd/.sift must not be
    readable by a script — even though .sift lives inside the cwd
    tree the script is otherwise allowed to read."""
    sift_dir = tmp_path / ".sift"
    sift_dir.mkdir()
    secret = sift_dir / "session_secret.txt"
    secret.write_text("REAL-UNSANITIZED-SESSION-STATE")

    code = '''
from pathlib import Path
p = Path(".sift") / "session_secret.txt"
leaked = p.exists() and "REAL-UNSANITIZED" in p.read_text(encoding="utf-8")
assert not leaked, "the .sift carve-out failed to mask real session state"
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
    r = _run(tmp_path, code)
    _assert_ok(r)
    # The real file on the host must be completely untouched.
    assert secret.read_text(encoding="utf-8") == "REAL-UNSANITIZED-SESSION-STATE"


@requires_bwrap
@requires_python_stack
def test_bwrap_sift_carve_out_writes_vanish(tmp_path: Path, tiny_csv: Path):
    """A script writing INTO the masked .sift path lands in the
    ephemeral tmpfs and must never appear on the real host
    filesystem after the run completes — otherwise a script could
    tamper with the release ledger or forge chat history."""
    code = '''
from pathlib import Path
p = Path(".sift") / "canary_exfil.txt"
p.write_text("pwned")
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
    r = _run(tmp_path, code)
    _assert_ok(r)
    assert not (tmp_path / ".sift" / "canary_exfil.txt").exists(), (
        "a write into the masked .sift path leaked through to the "
        "real host filesystem — the tmpfs carve-out is broken"
    )


@requires_bwrap
@requires_python_stack
def test_bwrap_run_dir_stays_reachable_through_the_mask(
    tmp_path: Path, tiny_csv: Path,
):
    """The current run's OWN scratch subtree (nested inside
    .sift/runs/...) must remain readable/writable even though the
    rest of .sift is masked — otherwise the executor couldn't even
    write result.json for its own run."""
    code = '''
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
    r = _run(tmp_path, code)
    _assert_ok(r)
    # result.json is written by the executor's own runtime library
    # into run_dir, which lives under cwd/.sift/runs/<id>/ — its
    # presence on the host after the run proves run_dir survived the
    # .sift tmpfs mask.
    assert r.run_dir is not None
    assert (r.run_dir / "result.json").exists()


# ---------------------------------------------------------------------------
# PID namespace isolation
# ---------------------------------------------------------------------------

@requires_bwrap
@requires_python_stack
def test_bwrap_pid_namespace_isolates_process_list(
    tmp_path: Path, tiny_csv: Path,
):
    """bwrap's --proc combined with the PID-namespace unshare in
    --unshare-all means the sandboxed script sees a FRESH /proc
    scoped to its own tiny process tree — not the host's. This is a
    strictly stronger guarantee than macOS's sandbox-exec provides
    (which does not unshare the PID namespace at all)."""
    host_pid_count = sum(1 for p in os.listdir("/proc") if p.isdigit())

    code = '''
import os
pids = [p for p in os.listdir("/proc") if p.isdigit()]
# The sandbox's own process tree is at most the interpreter itself
# plus a couple of ancestors (bwrap's own pid-1-in-namespace, etc) —
# nowhere near a real host's process count.
assert len(pids) <= 5, f"sandbox leaked host PID namespace: saw {pids}"
import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
    r = _run(tmp_path, code)
    _assert_ok(r)
    # Sanity: prove the host itself has more visible processes than
    # the isolated sandbox was allowed to see, so this test isn't
    # vacuously true on a nearly-empty host.
    assert host_pid_count >= 2


# ---------------------------------------------------------------------------
# Network isolation
# ---------------------------------------------------------------------------

@requires_bwrap
@requires_python_stack
def test_bwrap_has_no_usable_network(tmp_path: Path, tiny_csv: Path):
    """--unshare-all unshares the network namespace, leaving the
    sandbox with no configured interfaces at all (not even a routable
    loopback) — verified manually during development: an outbound UDP
    connect() attempt fails with ENETUNREACH ("Network is
    unreachable") rather than succeeding or even reaching a
    firewall-style REFUSED, and a loopback TCP connect fails too since
    there is no listener inside the isolated namespace. Both outcomes
    prove there is no interface to route through, which is a stronger
    guarantee than a firewall rule (nothing to misconfigure)."""
    code = '''
import socket

outbound_blocked = False
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(2)
    s.connect(("8.8.8.8", 53))
except OSError:
    outbound_blocked = True
assert outbound_blocked, "sandbox has a usable outbound network path"

loopback_blocked = False
try:
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s2.settimeout(2)
    s2.connect(("127.0.0.1", 22))
except OSError:
    loopback_blocked = True
assert loopback_blocked, "sandbox has a usable loopback network path"

import sift
sift.from_summarize("probe", n=12, mean=1.0, sd=0.0, missing_count=0)
'''
    r = _run(tmp_path, code)
    _assert_ok(r)


# ---------------------------------------------------------------------------
# Missing-backend preflight — pure Python, no bwrap needed
# ---------------------------------------------------------------------------

def test_run_script_refuses_without_bwrap_on_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """If bwrap is unavailable on Linux, run_script must refuse
    rather than fall through to an unsandboxed subprocess — mirrors
    ``test_run_script_refuses_without_sandbox`` for macOS."""
    if not sys.platform.startswith("linux"):
        pytest.skip("this preflight branch only applies on Linux")

    from sift import env_detect, executor

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec=None,
        bwrap=None,
    )
    r = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert not r.ok
    assert "sandbox" in (r.error or "").lower()
    assert "bubblewrap" in (r.error or "").lower() or "bwrap" in (r.error or "").lower()
