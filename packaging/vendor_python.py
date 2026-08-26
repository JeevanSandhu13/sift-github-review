"""Build Sift's relocatable analysis runtime for the current platform."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from sift.platform_support import windows_x64_emulation

PYTHON_VERSION = "3.12.11"
ANALYSIS_PACKAGES = (
    # Direct engines are exact pins from uv.lock. Scientific release
    # qualification is against these versions; silently resolving a newer
    # numerical stack would invalidate that evidence.
    "pandas==2.3.3",
    "numpy==2.3.5",
    "scipy==1.17.1",
    "statsmodels==0.14.6",
    "matplotlib==3.11.1",
    "duckdb==1.5.5",
    "pyarrow==25.0.1",
    "openpyxl==3.1.5",
    "xlrd==2.0.2",
    "odfpy==1.4.1",
    "pyreadstat==1.3.6",
    "pyreadr==0.5.6",
    # Maintained typed-method engines. These are runtime product
    # dependencies, not merely test conveniences: method_runtime.py exposes
    # helpers that import them when the corresponding supported workflow runs.
    "scikit-learn==1.6.1",
    "factor-analyzer==0.5.1",
    "pyfixest==0.60.0",
    "rdrobust==2.0.0",
    "differences==0.3.0",
    "geopandas==1.1.4",
    "arviz==1.3.0",
    "pymc==6.3.1",
    # ``differences`` imports Polars. The compatibility runtime is required on
    # Windows-on-ARM because Polars cannot issue its x86 CPUID probe from an
    # emulated process; native x64 Windows also remains fully supported.
    "polars==1.43.2",
    "polars-runtime-compat==1.43.2",
)

ANALYSIS_IMPORTS = (
    "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "duckdb",
    "pyarrow", "openpyxl", "xlrd", "odf", "pyreadstat", "pyreadr",
    "sklearn", "factor_analyzer", "pyfixest", "rdrobust", "differences",
    "geopandas", "arviz", "pymc", "polars",
)


def _uv_python_id(
    *,
    system_platform: str | None = None,
    machine: str | None = None,
    python_platform: str | None = None,
) -> str:
    """Return the standalone-Python target matching this interpreter.

    Windows 11 ARM can run an x64 Python process.  In that configuration
    ``platform.machine()`` can describe the ARM host even though every wheel
    and executable produced by the selected interpreter must be x64.  The
    interpreter's sysconfig platform is the authoritative target on Windows.
    Optional arguments keep this decision directly testable on every host.
    """
    current = (system_platform or sys.platform).lower()
    current_machine = (machine or platform.machine()).lower()
    target = (python_platform or sysconfig.get_platform()).lower()

    if current == "darwin" and current_machine in {"arm64", "aarch64"}:
        return f"cpython-{PYTHON_VERSION}-macos-aarch64-none"
    if current == "darwin" and current_machine in {"x86_64", "amd64"}:
        return f"cpython-{PYTHON_VERSION}-macos-x86_64-none"
    if current == "win32":
        if target in {"win-amd64", "win-x86_64"}:
            return f"cpython-{PYTHON_VERSION}-windows-x86_64-none"
        raise SystemExit(
            "Windows releases require an x64 Python target "
            f"(found interpreter platform {target!r})"
        )
    if current.startswith("linux") and current_machine in {"x86_64", "amd64"}:
        return f"cpython-{PYTHON_VERSION}-linux-x86_64-gnu"
    if current.startswith("linux") and current_machine in {"arm64", "aarch64"}:
        return f"cpython-{PYTHON_VERSION}-linux-aarch64-gnu"
    raise SystemExit(f"unsupported release platform: {current} {current_machine}")


def _find_interpreter(
    root: Path,
    *,
    system_platform: str | None = None,
) -> Path:
    current = system_platform or sys.platform
    names = ("python.exe",) if current == "win32" else ("python3",)
    candidates = sorted(
        path
        for name in names
        for path in root.rglob(name)
        if path.is_file() and (current == "win32" or os.access(path, os.X_OK))
    )
    if not candidates:
        raise SystemExit(f"uv installed no usable interpreter under {root}")
    # A Windows standalone distribution also contains
    # ``Scripts/python.exe``. Lexicographic sorting places that launcher ahead
    # of the real top-level interpreter (capital ``Scripts`` sorts before
    # lower-case ``python.exe``), which previously caused the release builder
    # to copy only the four files in Scripts and produce a broken runtime.
    # Prefer the shallowest interpreter, then use the path solely as a stable
    # tie-breaker. On Unix the real interpreter lives one level deeper in
    # ``bin`` and remains the shallowest match.
    return min(
        candidates,
        key=lambda path: (len(path.relative_to(root).parts), str(path).lower()),
    )


def _distribution_root(interpreter: Path) -> Path:
    """Return the relocatable distribution root for an interpreter path."""
    if interpreter.parent.name.lower() in {"bin", "scripts"}:
        return interpreter.parent.parent
    return interpreter.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "vendor" / "python",
    )
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required to build the analysis runtime")

    with tempfile.TemporaryDirectory(prefix="sift-vendor-python-") as temp_str:
        temp = Path(temp_str)
        python_id = _uv_python_id()
        subprocess.run(
            [uv, "python", "install", "--install-dir", str(temp), python_id],
            check=True,
        )
        interpreter = _find_interpreter(temp)
        distribution = _distribution_root(interpreter)
        destination = args.destination.resolve()
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(distribution, destination, symlinks=True)

    bundled = (
        destination / "python.exe"
        if sys.platform == "win32"
        else destination / "bin" / "python3"
    )
    if not bundled.is_file():
        raise SystemExit(f"relocated interpreter missing at {bundled}")
    subprocess.run([str(bundled), "--version"], check=True)
    subprocess.run(
        [
            uv, "pip", "install", "--python", str(bundled),
            # uv-managed standalone interpreters carry PEP 668's
            # EXTERNALLY-MANAGED marker. This copied interpreter is the
            # application runtime we intentionally populate, not the host's
            # system Python, so the explicit override is both required and
            # safely scoped to ``bundled``.
            "--break-system-packages", *ANALYSIS_PACKAGES,
        ],
        check=True,
    )
    verification = (
        "import importlib,sys\n"
        "packages=" + repr(ANALYSIS_IMPORTS) + "\n"
        "failed=[]\n"
        "for package in packages:\n"
        "    try: importlib.import_module(package)\n"
        "    except Exception as exc: failed.append((package,type(exc).__name__,str(exc)))\n"
        "print('failed imports:',failed) if failed else print('analysis runtime verified')\n"
        "sys.exit(bool(failed))\n"
    )
    # The verification imports are read-only release checks. ``-B`` prevents
    # them from leaving machine-generated bytecode caches in the runtime that
    # is about to be signed and shipped.
    verification_env = os.environ.copy()
    if windows_x64_emulation():
        verification_env["POLARS_SKIP_CPU_CHECK"] = "1"
    subprocess.run(
        [str(bundled), "-I", "-B", "-c", verification],
        check=True,
        env=verification_env,
    )
    print(f"Vendored analysis runtime ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
