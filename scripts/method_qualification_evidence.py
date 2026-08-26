#!/usr/bin/env python3
"""Execute and content-bind the exact Stage 10 method qualification nodes."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sift.evaluation import (  # noqa: E402
    METHOD_EXECUTION_NODES,
    METHOD_TEST_EVIDENCE_RELATIVE,
    _qualification_source_binding,
)
from sift.reliability import atomic_write_json  # noqa: E402
from sift.subprocess_safety import run_bounded_capture  # noqa: E402


def _case_status(case: Any) -> str:
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    if case.find("skipped") is not None:
        return "skipped"
    return "pass"


def _module_name(filename: str) -> str:
    return filename.removesuffix(".py").replace("/", ".")


def _runner_manifest() -> dict[str, object]:
    python_packages: dict[str, str] = {}
    for name in (
        "numpy", "scipy", "statsmodels", "scikit-learn", "pandas", "lifelines",
        "factor-analyzer", "pymc", "arviz", "differences", "rdrobust", "pytest",
    ):
        try:
            python_packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    r_manifest: dict[str, object] = {"version": "unavailable", "platform": "unavailable", "packages": {}}
    rscript = shutil.which("Rscript")
    if rscript:
        packages = (
            "MASS", "survival", "nnet", "did", "rdrobust", "lme4", "psych",
            "lavaan", "poLCA", "fixest", "marginaleffects", "survey", "pscl",
        )
        expression = (
            "cat('version=',R.version.string,'\\n',sep='');"
            "cat('platform=',R.version$platform,'\\n',sep='');"
            f"for(p in c({','.join(repr(value) for value in packages)}))"
            "if(requireNamespace(p,quietly=TRUE))cat('package=',p,'=',as.character(packageVersion(p)),'\\n',sep='')"
        )
        try:
            completed = run_bounded_capture(
                [rscript, "--vanilla", "-e", expression], timeout=60, check=False,
            )
            found: dict[str, str] = {}
            for line in completed.stdout.splitlines():
                if line.startswith("version="):
                    r_manifest["version"] = line.partition("=")[2]
                elif line.startswith("platform="):
                    r_manifest["platform"] = line.partition("=")[2]
                elif line.startswith("package="):
                    name, _, version = line.removeprefix("package=").partition("=")
                    if name and version:
                        found[name] = version
            r_manifest["packages"] = found
        except Exception as exc:
            # Runner provenance must make a failed reference-runtime probe
            # distinguishable from a machine where R was never present.
            r_manifest["probe_error"] = type(exc).__name__
    return {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "pytest": python_packages.get("pytest", "unavailable"),
        "python_packages": python_packages,
        "r": r_manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / METHOD_TEST_EVIDENCE_RELATIVE,
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    source_ok, _, source_digest = _qualification_source_binding(PROJECT_ROOT)
    required_nodes = sorted({
        node for nodes in METHOD_EXECUTION_NODES.values() for node in nodes
    })
    node_lookup = {
        (_module_name(node.partition("::")[0]), node.partition("::")[2]): node
        for node in required_nodes
    }
    with tempfile.TemporaryDirectory(prefix="sift-method-evidence-") as temp:
        temp_root = Path(temp)
        junit = temp_root / "pytest.xml"
        basetemp = temp_root / "pytest"
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env["SIFT_QUALIFICATION_EXACT_NODES"] = "1"
        inherited_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(SOURCE_ROOT), inherited_pythonpath) if value
        )
        command = [
            sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "-o", "addopts=", "--basetemp", str(basetemp),
            "--junitxml", str(junit), *required_nodes,
        ]
        try:
            completed = run_bounded_capture(
                command, cwd=PROJECT_ROOT, env=env, check=False,
                timeout=args.timeout, stdout_limit=2 * 1024 * 1024,
                stderr_limit=2 * 1024 * 1024,
            )
            exit_code = int(completed.returncode)
        except Exception:  # a runner failure must still materialize failing evidence
            exit_code = 125

        node_results: dict[str, dict[str, object]] = {
            node: {"status": "missing", "cases": []} for node in required_nodes
        }
        unmatched = 0
        try:
            root = ET.parse(junit).getroot()
            for case in root.iter("testcase"):
                classname = str(case.attrib.get("classname", ""))
                concrete_name = str(case.attrib.get("name", ""))
                base_name = concrete_name.split("[", 1)[0]
                node = node_lookup.get((classname, base_name))
                if node is None:
                    unmatched += 1
                    continue
                status = _case_status(case)
                cases = node_results[node]["cases"]
                if not isinstance(cases, list):
                    exit_code = exit_code or 126
                    continue
                cases.append({"node_id": f"{node}[{concrete_name}]", "status": status})
            for result in node_results.values():
                cases = result["cases"]
                if not isinstance(cases, list):
                    exit_code = exit_code or 126
                    result["status"] = "malformed"
                    continue
                statuses = {case["status"] for case in cases}
                result["status"] = "pass" if cases and statuses == {"pass"} else (
                    "missing" if not cases else sorted(statuses - {"pass"})[0]
                )
        except (OSError, ET.ParseError, KeyError, TypeError):
            exit_code = exit_code or 126

    passed = bool(
        source_ok and exit_code == 0 and unmatched == 0
        and all(result["status"] == "pass" for result in node_results.values())
    )
    artifact = {
        "format": "sift-method-test-evidence",
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "source_binding_sha256": source_digest,
        "pytest_exit_code": exit_code,
        "unmatched_cases": unmatched,
        "runner": _runner_manifest(),
        "nodes": node_results,
    }
    atomic_write_json(args.output, artifact)
    print(
        f"Method qualification evidence: {artifact['status']} "
        f"({sum(row['status'] == 'pass' for row in node_results.values())}/{len(node_results)} nodes)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
