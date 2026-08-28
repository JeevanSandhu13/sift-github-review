# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds the self-contained Sift binary.

The bundle entry is ``__main__.py`` — a thin shim that calls
``sift.ui.main`` to bring up the pywebview-based UI directly.
Researchers double-click Sift.app and get the chat window with no
Terminal popup.

Produces ``dist/sift/`` with the ``sift`` executable plus every
dependency (Python runtime, pandas, pyreadstat, claude-agent-sdk,
pywebview, etc.) so researchers don't need Python/uv/pip installed.

Run from the repo root:
    uv run pyinstaller packaging/sift.spec --clean --noconfirm

What needs explicit handling here and cannot be inferred by
PyInstaller:

- Runtime libraries (`sift.R`, `sift_result_*.ado`) are data
  files, not Python modules — PyInstaller won't pick them up
  without an explicit `datas` entry.
- Web UI assets (HTML, JavaScript, CSS, icons, and fonts) live under
  ``src/sift/web/``; the same
  story, listed in `datas`.
- ``pyreadstat`` is a C extension with a subdivided module layout;
  the default import scan sometimes misses submodules. Listed
  explicitly in ``hiddenimports``.
- ``claude_agent_sdk`` has dynamic imports for transport plugins and ships a
  platform-native Claude executable under ``_bundled``. Both the modules and
  executable must survive the tree-shake. An API-key user must never need a
  separate npm installation after installing Sift.
- ``webview`` (pywebview) loads platform backends dynamically;
  ``webview.platforms.cocoa`` is the macOS one and won't be
  picked up by static analysis.
"""

import sys
from pathlib import Path

from importlib.metadata import PackageNotFoundError

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


# Repo root — PyInstaller runs with the spec file as its working ref,
# so we resolve relative to the spec's directory.
REPO_ROOT = Path(SPECPATH).parent  # type: ignore[name-defined]  # SPECPATH from PyInstaller

# UI entry. The shim re-exports ``sift.ui.main`` so ``uv run sift``
# (console-script) and ``python -m sift`` (package entry) both end up
# at the same place; PyInstaller targets the file rather than a
# console-script so it doesn't need entry-point metadata at build time.
ENTRY = str(REPO_ROOT / "src" / "sift" / "__main__.py")

# Runtime libraries (R + Stata + Python) are loaded via
# ``importlib.resources`` inside ``executor._stage_runtime``.
# PyInstaller preserves the package layout when we list them
# explicitly as data files.
#
# Glob, don't enumerate. The previous hand-maintained list missed
# 8 of the 13 .ado helpers (correlation, every plot helper,
# safe_export, plot_export, the standalone ttest) plus the Python
# runtime ``sift.py`` — every Stata script that used a plot helper
# or correlation, and every Python script entirely, crashed in
# .app builds with FileNotFoundError because
# ``importlib.resources.files("sift.runtime").joinpath(name)
# .read_text()`` returned nothing for the un-bundled files. The
# dev install (pip / uv from source) worked because the package
# dir on disk had every file; only the PyInstaller bundle was
# missing them. Globbing closes the door on this regression class:
# any new helper dropped into ``runtime/`` ships with the build
# without a spec edit. We include .py too because:
#   - ``__init__.py`` is required for ``importlib.resources.files
#     ("sift.runtime")`` to resolve as a package
#   - ``sift.py`` is the Python user-runtime that the executor
#     stages into every Python script's ``lib_dir`` and the user
#     script then imports — it has to be readable as a *file
#     resource*, not just importable as a module (PyInstaller's
#     bytecode-only archive doesn't satisfy resources.files's
#     ``.read_text()`` call)
#   - any other Python helper module in ``runtime/`` is also
#     picked up by PyInstaller's normal tree-shake; listing it as
#     data is harmless redundancy.
# Hidden / cache files (``__pycache__``, ``.DS_Store``) are
# skipped by ``is_file()`` + dot-prefix filter.
RUNTIME_DIR = REPO_ROOT / "src" / "sift" / "runtime"
RUNTIME_DATAS = [
    (str(p), "sift/runtime")
    for p in sorted(RUNTIME_DIR.iterdir())
    if p.is_file() and not p.name.startswith(".")
]

# Web UI assets (HTML + JS + CSS + icons + the locally-hosted font files
# under web/fonts/). The UI shell loads these from
# ``Path(__file__).parent / "web"`` at runtime — see ui.py near
# `webview.create_window`. PyInstaller's static import scan can't see
# static asset files; without an explicit datas entry the bundle
# ships without the UI.
#
# rglob walks subdirectories so anything dropped into web/ (a future
# asset, an additional font weight, a different Lottie animation,
# etc.) gets bundled without spec edits — the rule is "everything
# under web/ goes into web/ in the bundle, preserving subpaths".
# Hidden files (``.DS_Store`` etc.) and dot-prefixed dirs are skipped.
# The destination keeps the asset's relative path under web/ so
# CSS references like ``url('fonts/Lexend-VF-latin.woff2')`` resolve
# the same in the bundle as they do from source.
WEB_DIR = REPO_ROOT / "src" / "sift" / "web"
WEB_DATAS = [
    (str(p), str(Path("sift/web") / p.relative_to(WEB_DIR).parent))
    for p in sorted(WEB_DIR.rglob("*"))
    if p.is_file()
    and not p.name.startswith(".")
    and not any(part.startswith(".") for part in p.relative_to(WEB_DIR).parts)
]

# A production build embeds the public update trust store and its pinned
# policy. The private signing key is never present here. Development builds
# intentionally omit this directory, making the runtime report updates as
# unavailable instead of trusting a placeholder key.
UPDATE_POLICY_DIR = REPO_ROOT / "packaging" / "generated" / "update"
UPDATE_POLICY_DATAS = (
    [
        (str(UPDATE_POLICY_DIR / "update-policy.json"), "sift/update"),
        (str(UPDATE_POLICY_DIR / "release-trust-store.json"), "sift/update"),
    ]
    if all((UPDATE_POLICY_DIR / name).is_file() for name in (
        "update-policy.json", "release-trust-store.json",
    ))
    else []
)


def optional_metadata(distribution):
    """Collect SQLAlchemy dialect entry points when an extra is installed."""
    try:
        return copy_metadata(distribution)
    except PackageNotFoundError:
        return []


def runtime_submodules(package):
    """Collect executable package code without shipping upstream test suites."""
    return collect_submodules(
        package,
        filter=lambda name: not any(
            part in {
                "test", "tests", "testing", "conftest",
                "benchmark", "benchmarks",
            }
            for part in name.split(".")
        ),
    )


def is_vendored_runtime_file(path, root):
    """Keep executable runtime files while excluding upstream QA artifacts.

    The private interpreter must remain complete enough to import, execute,
    and install supported analysis packages. Distribution test suites,
    benchmarks, and bytecode caches for those excluded modules serve no
    product function and materially increase the signed application surface.
    """
    relative = path.relative_to(root)
    excluded_directories = {
        "test", "tests", "testing", "benchmark", "benchmarks", "__pycache__",
    }
    if any(part.casefold() in excluded_directories for part in relative.parts[:-1]):
        return False
    # A filename prefix is not a safe QA boundary: PyMC, for example, ships
    # ``variational/test_functions.py`` as executable runtime code.  Directory
    # boundaries and the unambiguous pytest configuration filename are safe.
    return relative.name.casefold() != "conftest.py"


def is_runtime_bundle_destination(destination):
    """Apply the QA-tree boundary to data injected by third-party hooks.

    PyInstaller hooks run after our explicit data selection and some collect
    whole upstream packages, including their tests.  Filtering the final data
    table is the only reliable cross-platform boundary.  Filename prefixes are
    intentionally preserved because packages such as PyMC use
    ``test_functions.py`` as real runtime code.
    """
    parts = Path(str(destination)).parts
    excluded_directories = {
        "test", "tests", "testing", "benchmark", "benchmarks", "__pycache__",
    }
    if any(part.casefold() in excluded_directories for part in parts[:-1]):
        return False
    return not parts or parts[-1].casefold() != "conftest.py"


# SQLAlchemy resolves third-party dialects through package entry points. The
# Python modules alone are insufficient in a frozen app unless the matching
# dist-info metadata is present as well.
DATABASE_DIALECT_DATAS = sum((
    optional_metadata("snowflake-sqlalchemy"),
    optional_metadata("sqlalchemy-bigquery"),
    optional_metadata("sqlalchemy-redshift"),
    optional_metadata("databricks-sqlalchemy"),
), [])

# Pyogrio and Rasterio wheels carry the GDAL and PROJ registries beside their
# extension modules. Their loaders discover those exact package-relative
# directories; bundling only the extensions produces an app that imports but
# warns that GDAL_DATA is missing and can mis-handle coordinate systems or
# structured geospatial formats. PyInstaller has no standard hooks for these
# two packages, so preserve only their runtime registries explicitly.
GEOSPATIAL_RUNTIME_DATAS = (
    collect_data_files(
        "pyogrio", includes=["gdal_data/**", "proj_data/**"],
    )
    + collect_data_files(
        "rasterio", includes=["gdal_data/**", "proj_data/**"],
    )
)

# The Anthropic Agent SDK drives its API session through the native Claude
# executable included in the SDK wheel. PyInstaller follows Python imports but
# does not automatically collect package executables, which previously left a
# fully configured Anthropic user with a misleading "Claude Code not found"
# error on the first message. Treat the executable as a required binary on
# every target platform and fail the build immediately if the installed SDK
# wheel does not provide it.
CLAUDE_AGENT_SDK_DIRS = collect_data_files(
    "claude_agent_sdk", includes=["_bundled/.gitignore"],
)
if not CLAUDE_AGENT_SDK_DIRS:
    raise RuntimeError("claude-agent-sdk package directory could not be resolved")
CLAUDE_AGENT_SDK_ROOT = Path(CLAUDE_AGENT_SDK_DIRS[0][0]).parent.parent
CLAUDE_AGENT_CLI_NAME = "claude.exe" if sys.platform == "win32" else "claude"
CLAUDE_AGENT_CLI = (
    CLAUDE_AGENT_SDK_ROOT / "_bundled" / CLAUDE_AGENT_CLI_NAME
)
if not CLAUDE_AGENT_CLI.is_file():
    raise RuntimeError(
        f"claude-agent-sdk is missing required bundled CLI: {CLAUDE_AGENT_CLI}"
    )
CLAUDE_AGENT_CLI_BINARIES = [
    (str(CLAUDE_AGENT_CLI), "claude_agent_sdk/_bundled"),
]

# Sift Skills library — ``sift.skills.list_builtin_skills``
# reads these ``*.md`` files via ``Path(__file__).parent /
# "skills_library"`` at runtime, same "package data, not importable
# code" story as the runtime .ado/.R helpers above. PyInstaller's
# module analysis only follows ``.py`` files; without this explicit
# datas entry a bundled Sift would ship with zero skills and
# ``get_skill`` would return "no skill with slug ..." for every
# builtin slug the system prompt advertises. Same glob-don't-
# enumerate rationale as RUNTIME_DATAS: any skill dropped into
# skills_library/ ships without a spec edit.
SKILLS_DIR = REPO_ROOT / "src" / "sift" / "skills_library"
SKILLS_DATAS = [
    (str(p), "sift/skills_library")
    for p in sorted(SKILLS_DIR.iterdir())
    if p.is_file() and not p.name.startswith(".")
]

# Bundled Python analysis runtime — OPTIONAL. A maintainer
# runs ``packaging/vendor_python.sh`` before a release build to
# populate ``packaging/vendor/python/`` with a portable, relocatable
# CPython (astral-sh/python-build-standalone) plus Sift's analysis
# stack (pandas, numpy, scipy, statsmodels, matplotlib, duckdb,
# pyarrow, openpyxl, pyreadstat, pyreadr) pip-installed into it. If
# that directory exists at build time, it's bundled into the .app at
# ``sift/vendor_python`` — the exact path
# ``env_detect._bundled_python_root()`` looks under
# ``sys._MEIPASS`` for. If the vendor step was never run (every dev
# build, and any release build where a maintainer skipped it), this
# list is simply empty and Python detection falls through to PATH-only.
# Never a hard requirement, unlike RUNTIME_DATAS /
# WEB_DATAS / SKILLS_DATAS above, all of which the app cannot
# function without.
#
# rglob (not iterdir) because a real CPython distribution is deeply
# nested (bin/, lib/pythonX.Y/, lib/pythonX.Y/site-packages/, …) —
# unlike the flat runtime/skills dirs above, this needs every file at
# every depth preserved at its relative path so the interpreter can
# still find its own stdlib once bundled.
VENDOR_PYTHON_DIR = REPO_ROOT / "packaging" / "vendor" / "python"
if VENDOR_PYTHON_DIR.is_dir():
    VENDOR_PYTHON_DATAS = [
        (str(p), str(Path("sift/vendor_python") / p.relative_to(VENDOR_PYTHON_DIR).parent))
        for p in sorted(VENDOR_PYTHON_DIR.rglob("*"))
        if p.is_file() and is_vendored_runtime_file(p, VENDOR_PYTHON_DIR)
    ]
    print(
        f"[sift.spec] bundling vendored Python runtime "
        f"({len(VENDOR_PYTHON_DATAS)} files) from {VENDOR_PYTHON_DIR}"
    )
else:
    VENDOR_PYTHON_DATAS = []
    print(
        f"[sift.spec] no vendored Python runtime at {VENDOR_PYTHON_DIR} "
        f"-- building without one (run packaging/vendor_python.sh "
        f"first for a release build). Python detection will fall "
        f"through to PATH-only."
    )

# Hidden imports — things PyInstaller's static scan may miss because
# they're loaded dynamically.
HIDDEN_IMPORTS = [
    *runtime_submodules("claude_agent_sdk"),
    *runtime_submodules("pyreadstat"),
    *runtime_submodules("pyreadr"),
    *runtime_submodules("openpyxl"),
    *runtime_submodules("xlrd"),
    *runtime_submodules("odf"),
    *runtime_submodules("duckdb"),
    # pywebview loads its window backend at runtime via
    # ``webview.platforms.<platform>``; the cocoa one is the macOS
    # backend. Without these the bundle starts up and immediately
    # complains that no GUI toolkit is available.
    *runtime_submodules("webview"),
    # Provider packages and SDKs are selected dynamically from the model
    # picker.  The frozen ``--integration-check`` imports each exact surface,
    # but these explicit collections ensure the tree-shaker cannot omit them.
    *runtime_submodules("openai"),
    *runtime_submodules("google.genai"),
    *runtime_submodules("anthropic"),
    # PyObjC framework packages expose most symbols through lazy metadata,
    # which PyInstaller cannot infer from ``webview.platforms.cocoa`` alone.
    # Keep this platform-guarded so Windows/Linux builds never try to resolve
    # macOS-only modules.
    *(
        collect_submodules("objc")
        + collect_submodules("Cocoa")
        + collect_submodules("Foundation")
        + collect_submodules("AppKit")
        + collect_submodules("Quartz")
        + collect_submodules("WebKit")
        if sys.platform == "darwin" else []
    ),
    *runtime_submodules("keyring.backends"),
    # Linux keyring's secure Freedesktop Secret Service backend is loaded
    # through entry points and D-Bus helpers; preserve it in frozen bundles.
    *runtime_submodules("secretstorage"),
    *runtime_submodules("jeepney"),
    *runtime_submodules("sqlalchemy.dialects"),
    # Cloud/database drivers are selected dynamically from the researcher's
    # URI, so static import analysis cannot see them. Release builds install
    # every optional database extra before this spec runs.
    *runtime_submodules("psycopg"),
    *runtime_submodules("pymysql"),
    "pyodbc",  # extension module, not a package with discoverable submodules
    *runtime_submodules("oracledb"),
    *runtime_submodules("snowflake"),
    *runtime_submodules("sqlalchemy_bigquery"),
    *runtime_submodules("sqlalchemy_redshift"),
    *runtime_submodules("redshift_connector"),
    *runtime_submodules("databricks"),
    *runtime_submodules("boto3"),
    *runtime_submodules("botocore"),
    *runtime_submodules("google.cloud.storage"),
    *runtime_submodules("azure.identity"),
    *runtime_submodules("azure.storage.blob"),
    *runtime_submodules("paramiko"),
    # Complex data adapters run in an offline child process and import their
    # parser lazily after the researcher makes an explicit selection. Static
    # analysis cannot see those branches, so release bundles include the
    # installed cross-platform parser stacks explicitly.
    *runtime_submodules("fastavro"),
    *runtime_submodules("defusedxml"),
    *runtime_submodules("dbfread"),
    *runtime_submodules("h5py"),
    *runtime_submodules("xarray"),
    *runtime_submodules("netCDF4"),
    *runtime_submodules("astropy"),
    *runtime_submodules("geopandas"),
    *runtime_submodules("pyogrio"),
    *runtime_submodules("shapely"),
    *runtime_submodules("rasterio"),
    *(runtime_submodules("pysam") if sys.platform != "win32" else []),
    *runtime_submodules("nibabel"),
    *runtime_submodules("pydicom"),
    # pandas and numpy are covered by PyInstaller's bundled hooks.
]


block_cipher = None
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
EXECUTABLE_NAME = "Sift" if IS_WINDOWS else "sift"
APP_ICON = (
    str(REPO_ROOT / "packaging" / "Sift.icns")
    if IS_MACOS
    else str(REPO_ROOT / "packaging" / "windows" / "Sift.ico")
    if IS_WINDOWS
    else None
)
WINDOWS_VERSION_INFO = (
    str(REPO_ROOT / "packaging" / "generated" / "windows-version-info.txt")
    if IS_WINDOWS else None
)

a = Analysis(
    [ENTRY],
    pathex=[str(REPO_ROOT / "src")],
    binaries=CLAUDE_AGENT_CLI_BINARIES,
    # Keep the complete Analysis data expression on this line.  Besides making
    # the frozen inputs easy to audit, the packaging contract test deliberately
    # verifies that the vendored analysis runtime cannot disappear from this
    # exact boundary during future edits.
    datas=RUNTIME_DATAS + WEB_DATAS + SKILLS_DATAS + UPDATE_POLICY_DATAS + VENDOR_PYTHON_DATAS + DATABASE_DIALECT_DATAS + GEOSPATIAL_RUNTIME_DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Shave weight — we don't ship notebooks, plotting, or tests.
        "tkinter",
        "matplotlib",
        "IPython",
        "notebook",
        "jupyter",
        "pytest",
        "hypothesis",
    ],
    noarchive=False,
    cipher=block_cipher,
)

# Third-party PyInstaller hooks may add package data after Analysis is
# constructed.  Enforce the same bounded runtime surface on those hook-added
# entries before COLLECT assembles the frozen directory.
a.datas = [
    entry for entry in a.datas
    if is_runtime_bundle_destination(entry[0])
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXECUTABLE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # macOS uses the windowed bootloader so the .app opens without a
    # Terminal window.  Windows deliberately uses the console bootloader:
    # PyInstaller's windowed bootloader replaces stdin/stdout/stderr with
    # None, which breaks Sift's machine-readable support checks when a
    # researcher runs them from PowerShell. ``hide-early`` only hides a
    # console that Sift itself owns (Start-menu/double-click launch); an
    # existing PowerShell console remains attached and receives diagnostics.
    # Linux ignores this option and always uses its console bootloader.
    console=IS_WINDOWS,
    hide_console="hide-early" if IS_WINDOWS else None,
    disable_windowed_traceback=False,
    # Pin to arm64 so the bundle can never silently go Intel just
    # because someone ran the build under a Rosetta-installed Python.
    # ``None`` defers to the host interpreter's arch, which is fine on
    # an arm64 uv/Python but means an Intel build sneaks through with
    # no warning if the toolchain is mixed. PyInstaller will refuse to
    # produce an arm64 binary from an x86_64 Python, surfacing the
    # mismatch loudly at build time. release.sh's verify step also
    # asserts the resulting Mach-O is arm64 as a belt-and-suspenders
    # guard against future spec edits.
    target_arch="arm64" if IS_MACOS else None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON,
    version=WINDOWS_VERSION_INFO,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Sift" if IS_WINDOWS else "sift",
)
