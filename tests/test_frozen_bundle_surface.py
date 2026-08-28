from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "packaging" / "verify_frozen_bundle.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_frozen_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_surface_rejects_qa_trees(tmp_path: Path) -> None:
    module = _module()
    bundle = tmp_path / "bundle"
    (bundle / "pkg" / "tests").mkdir(parents=True)
    (bundle / "pkg" / "tests" / "test_private.py").write_text("secret")
    (bundle / "pkg" / "test_helper.py").write_text("secret")
    (bundle / "pkg" / "runtime.py").write_text("safe")

    assert module.prohibited_bundle_entries(bundle) == ["pkg/tests/"]


def test_bundle_surface_accepts_runtime_and_refuses_ambiguous_root(
    tmp_path: Path,
) -> None:
    module = _module()
    bundle = tmp_path / "bundle"
    (bundle / "pymc" / "variational").mkdir(parents=True)
    (bundle / "pkg").mkdir()
    (bundle / "pkg" / "runtime.py").write_text("safe")
    (bundle / "pymc" / "variational" / "test_functions.py").write_text("safe")
    assert module.prohibited_bundle_entries(bundle) == []
    with pytest.raises(ValueError, match="real directory"):
        module.prohibited_bundle_entries(tmp_path / "missing")


def test_bundle_surface_requires_bundled_anthropic_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    expected = (
        bundle
        / "_internal"
        / "claude_agent_sdk"
        / "_bundled"
        / ("claude.exe" if module.os.name == "nt" else "claude")
    )

    failures = module.required_runtime_failures(bundle)
    assert failures == [
        f"missing Anthropic agent runtime: {expected.relative_to(bundle)}"
    ]

    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"runtime")
    if module.os.name != "nt":
        expected.chmod(0o755)
    assert module.required_runtime_failures(bundle) == []


def test_provider_runtime_probe_starts_bundled_anthropic_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    bundle = tmp_path / "bundle"
    runtime = (
        bundle
        / "_internal"
        / "claude_agent_sdk"
        / "_bundled"
        / ("claude.exe" if module.os.name == "nt" else "claude")
    )
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return module.subprocess.CompletedProcess(command, 0, "2.1.235\n", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.provider_runtime_probe_failures(bundle) == []
    assert observed["command"] == [str(runtime), "--version"]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 60,
    }


@pytest.mark.parametrize(
    "builder",
    [
        ROOT / "packaging" / "build_app.sh",
        ROOT / "packaging" / "build_linux.sh",
        ROOT / "packaging" / "build_windows.ps1",
    ],
)
def test_packaged_analysis_check_cannot_dirty_accepted_bundle(builder: Path) -> None:
    text = builder.read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE" in text
    assert text.index("--analysis-check") < text.index("verify_frozen_bundle.py")
