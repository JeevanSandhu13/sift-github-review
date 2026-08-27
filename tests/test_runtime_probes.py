"""Regression coverage for optional scientific-runtime probes."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from tests import runtime_probes


def setup_function() -> None:
    runtime_probes.r_package_loadable.cache_clear()


def test_r_package_probe_reports_loadable_package(monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []

    def completed(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime_probes.subprocess, "run", completed)

    assert runtime_probes.r_package_loadable("Rscript", "survival")
    assert runtime_probes.r_package_loadable("Rscript", "survival")
    assert len(calls) == 1
    assert calls[0][0][:2] == ["Rscript", "--vanilla"]
    assert calls[0][1]["timeout"] == 60.0


def test_r_package_probe_fails_closed_on_timeout(monkeypatch) -> None:
    def timed_out(command: list[str], **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runtime_probes.subprocess, "run", timed_out)

    assert not runtime_probes.r_package_loadable("Rscript", "survival")


def test_r_package_probe_fails_closed_on_launch_error(monkeypatch) -> None:
    def unavailable(command: list[str], **kwargs):
        raise OSError("runtime unavailable")

    monkeypatch.setattr(runtime_probes.subprocess, "run", unavailable)

    assert not runtime_probes.r_package_loadable("Rscript", "survival")


def test_r_package_probe_rejects_untrusted_package_name(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid package names must not reach Rscript")

    monkeypatch.setattr(runtime_probes.subprocess, "run", unexpected)

    assert not runtime_probes.r_package_loadable("Rscript", "pkg);system('bad')")
    assert not runtime_probes.r_package_loadable(None, "survival")
