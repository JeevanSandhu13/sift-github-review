"""Pure unit tests on the generated bubblewrap argv list.

Mirrors ``test_executor_profile.py``'s role for the macOS SBPL string:
these assert the invariants ``_bwrap_argv`` MUST satisfy without
needing ``bwrap`` to actually be installed or callable, so they run
unconditionally on every platform (including macOS dev boxes with no
bwrap on PATH at all). The real, "run it for real against the actual
binary" layer lives in ``test_bwrap_sandbox.py``.

If a reviewer wants to tighten or loosen what a Linux-sandboxed script
can reach, the argv built here is the whole contract — same role
``_sandbox_profile_string`` plays for the macOS profile text.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from sift.executor import _bwrap_argv


def _example_argv(**overrides) -> list[str]:
    kwargs = dict(
        # These model paths inside a Linux mount namespace even when this
        # pure policy test runs from Windows. ``Path`` would reinterpret them
        # with the host's separators and invalidate the policy assertion.
        run_dir=PurePosixPath("/home/researcher/project/.sift/runs/run-1234"),
        cwd=PurePosixPath("/home/researcher/project"),
        home=PurePosixPath("/home/researcher"),
    )
    kwargs.update(overrides)
    return _bwrap_argv(**kwargs)


# ---------------------------------------------------------------------------
# Namespace isolation flags — the core confinement posture
# ---------------------------------------------------------------------------

def test_argv_unshares_all_namespaces():
    argv = _example_argv()
    assert "--unshare-all" in argv


def test_argv_dies_with_parent():
    argv = _example_argv()
    assert "--die-with-parent" in argv


def test_argv_uses_new_session():
    """TIOCSTI terminal-injection protection."""
    argv = _example_argv()
    assert "--new-session" in argv


def test_argv_mounts_fresh_proc():
    argv = _example_argv()
    assert "--proc" in argv
    idx = argv.index("--proc")
    assert argv[idx + 1] == "/proc"


def test_argv_mounts_synthetic_dev():
    argv = _example_argv()
    assert "--dev" in argv
    idx = argv.index("--dev")
    assert argv[idx + 1] == "/dev"


def test_argv_starts_with_namespace_flags():
    """The isolation flags should be established before anything is
    bound in — order doesn't strictly matter to bwrap itself for most
    of these, but keeping them first documents the "start from
    nothing, then add" posture the module comment describes."""
    argv = _example_argv()
    assert argv[0] == "--unshare-all"


# ---------------------------------------------------------------------------
# System trees — existence-checked, whole-directory read-only binds
# ---------------------------------------------------------------------------

def test_argv_binds_real_system_dirs_readonly():
    """Binary/library roots are bound read-only as directories."""
    argv = _example_argv()
    for path in ("/usr", "/bin"):
        if Path(path).exists():
            assert "--ro-bind" in argv
            # find the specific pair for this path
            pairs = list(zip(argv, argv[1:]))
            assert (path, path) in [
                (argv[i + 1], argv[i + 2])
                for i, a in enumerate(argv)
                if a == "--ro-bind" and i + 2 < len(argv)
            ], f"{path} not bound read-only as a whole directory"


def test_argv_does_not_expose_all_system_configuration_or_opt():
    argv = _example_argv()
    pairs = [
        (argv[i + 1], argv[i + 2])
        for i, value in enumerate(argv)
        if value == "--ro-bind" and i + 2 < len(argv)
    ]
    assert ("/etc", "/etc") not in pairs
    assert ("/opt", "/opt") not in pairs
    if Path("/etc/ld.so.cache").exists():
        assert ("/etc/ld.so.cache", "/etc/ld.so.cache") in pairs


def test_argv_never_exposes_hidden_r_configuration_directory():
    argv = _example_argv()
    assert "/home/researcher/.R" not in argv


def test_argv_skips_nonexistent_system_dirs():
    """/lib32 does not exist on most 64-bit distros without multilib;
    the function must not emit a bind for a source path that isn't
    there (bwrap would hard-fail with 'Can't find source path')."""
    argv = _example_argv()
    if not Path("/lib32").exists():
        # No occurrence of /lib32 anywhere in the argv at all.
        assert "/lib32" not in argv


def test_argv_never_binds_a_path_that_does_not_exist():
    """General form of the above: every --ro-bind source this
    function emits must exist on disk, or bwrap refuses to start."""
    argv = _example_argv()
    for i, tok in enumerate(argv):
        if tok == "--ro-bind" and i + 1 < len(argv):
            src = argv[i + 1]
            assert Path(src).exists(), (
                f"_bwrap_argv emitted a --ro-bind for {src!r}, which "
                f"does not exist on this machine — bwrap would fail "
                f"to start"
            )


# ---------------------------------------------------------------------------
# extra_read_paths — interpreter-specific extras (venv, pkg dir)
# ---------------------------------------------------------------------------

def test_argv_binds_existing_extra_read_paths(tmp_path: Path):
    extra_dir = tmp_path / "venv"
    extra_dir.mkdir()
    argv = _example_argv(extra_read_paths=(str(extra_dir),))
    idx = argv.index("--ro-bind")
    binds = [
        (argv[i + 1], argv[i + 2])
        for i, a in enumerate(argv)
        if a == "--ro-bind" and i + 2 < len(argv)
    ]
    assert (str(extra_dir), str(extra_dir)) in binds


def test_argv_skips_nonexistent_extra_read_paths(tmp_path: Path):
    missing = tmp_path / "does-not-exist"
    argv = _example_argv(extra_read_paths=(str(missing),))
    assert str(missing) not in argv


def test_argv_skips_empty_string_extra_read_paths():
    """A caller passing an empty string (e.g. an unset optional path)
    must not crash the existence check or bind cwd's parent by
    accident."""
    argv = _example_argv(extra_read_paths=("",))
    # Should behave identically to no extra paths at all.
    assert argv == _example_argv(extra_read_paths=())


def test_argv_skips_relative_extra_read_paths(tmp_path: Path, monkeypatch):
    relative = Path("relative-runtime")
    (tmp_path / relative).mkdir()
    monkeypatch.chdir(tmp_path)
    argv = _example_argv(extra_read_paths=(str(relative),))
    assert str(relative) not in argv


# ---------------------------------------------------------------------------
# Writable workspace + the .sift carve-out — the security-critical part
# ---------------------------------------------------------------------------

def test_argv_binds_cwd_writable():
    argv = _example_argv()
    binds = [
        (argv[i + 1], argv[i + 2])
        for i, a in enumerate(argv)
        if a == "--bind" and i + 2 < len(argv)
    ]
    assert ("/home/researcher/project", "/home/researcher/project") in binds


def test_argv_binds_run_dir_writable():
    argv = _example_argv()
    binds = [
        (argv[i + 1], argv[i + 2])
        for i, a in enumerate(argv)
        if a == "--bind" and i + 2 < len(argv)
    ]
    assert (
        "/home/researcher/project/.sift/runs/run-1234",
        "/home/researcher/project/.sift/runs/run-1234",
    ) in binds


def test_argv_masks_sift_dir_with_tmpfs():
    argv = _example_argv()
    assert "--tmpfs" in argv
    idx = argv.index("--tmpfs")
    assert argv[idx + 1] == "/home/researcher/project/.sift"


def test_argv_rebinds_run_dir_after_the_sift_mask():
    """The critical ordering property: cwd is bound, THEN .sift is
    masked with a tmpfs (shadowing Sift's own session state), THEN
    run_dir is bound again on top of that mask so the current run's
    own scratch subtree (which lives inside .sift/runs/...) is still
    reachable. bwrap applies binds in argv order and a later bind at
    an equal-or-nested path wins — so this exact ordering is what
    makes the carve-out work at all. Get this backwards and either
    the script can't see its own run_dir, or the .sift mask doesn't
    actually shadow anything."""
    argv = _example_argv()

    tmpfs_idx = argv.index("--tmpfs")
    sift_path = argv[tmpfs_idx + 1]
    assert sift_path == "/home/researcher/project/.sift"

    # First cwd bind must happen before the tmpfs mask.
    first_cwd_bind_idx = None
    for i, tok in enumerate(argv):
        if tok == "--bind" and argv[i + 1] == "/home/researcher/project":
            first_cwd_bind_idx = i
            break
    assert first_cwd_bind_idx is not None
    assert first_cwd_bind_idx < tmpfs_idx, (
        "cwd must be bound before the .sift tmpfs mask is applied"
    )

    # run_dir must be bound again AFTER the tmpfs mask (a second
    # --bind run_dir run_dir occurrence past tmpfs_idx).
    run_dir_binds_after_mask = [
        i for i, tok in enumerate(argv)
        if tok == "--bind"
        and i > tmpfs_idx
        and argv[i + 1] == "/home/researcher/project/.sift/runs/run-1234"
    ]
    assert run_dir_binds_after_mask, (
        "run_dir must be re-bound AFTER the .sift tmpfs mask, or the "
        "current run's own scratch directory would be swallowed by "
        "the mask along with the rest of .sift"
    )


def test_argv_binds_run_dir_at_least_twice():
    """Once as part of the general writable-workspace bind, once again
    after the .sift mask to re-expose it. If this collapses to a
    single bind (e.g. a future refactor de-dupes it "for cleanliness")
    the carve-out silently breaks."""
    argv = _example_argv()
    count = sum(
        1 for i, tok in enumerate(argv)
        if tok == "--bind"
        and i + 1 < len(argv)
        and argv[i + 1] == "/home/researcher/project/.sift/runs/run-1234"
    )
    assert count >= 2


# ---------------------------------------------------------------------------
# R user-library paths — "built, not proven" per the module comment
# ---------------------------------------------------------------------------

def test_argv_binds_home_r_dirs_only_if_present(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    argv_without = _example_argv(home=home)
    assert str(home / "R") not in argv_without
    assert str(home / ".R") not in argv_without

    (home / "R").mkdir()
    argv_with = _example_argv(home=home)
    binds = [
        (argv_with[i + 1], argv_with[i + 2])
        for i, a in enumerate(argv_with)
        if a == "--ro-bind" and i + 2 < len(argv_with)
    ]
    assert (str(home / "R"), str(home / "R")) in binds


# ---------------------------------------------------------------------------
# Purity — no subprocess spawned, no bwrap binary required
# ---------------------------------------------------------------------------

def test_argv_remounts_root_read_only_as_the_last_step():
    """Locks the synthetic root so unbound paths can't be silently
    auto-vivified as writable (discovered empirically during
    development — see the long comment above the flag in
    executor.py). Must be the LAST setup argument so every real
    writable bind (cwd, run_dir, the .sift tmpfs) is already in place
    before the remount is applied; remount-ro doesn't recurse into
    mounts nested beneath the path it targets, so ordering here is
    what keeps those nested mounts writable."""
    argv = _example_argv()
    assert argv[-1] == "/"
    assert argv[-2] == "--remount-ro"


def test_argv_is_pure_no_bwrap_binary_needed():
    """This whole file must be able to run on a machine with no bwrap
    installed at all (e.g. a macOS dev box) — _bwrap_argv only does
    Path.exists() checks, never invokes bwrap itself."""
    argv = _example_argv()
    assert isinstance(argv, list)
    assert all(isinstance(a, str) for a in argv)
