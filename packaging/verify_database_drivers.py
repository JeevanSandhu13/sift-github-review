"""Fail a release build if an advertised database connector is absent."""

from __future__ import annotations

import importlib

from sift.integrations import database_adapter_status


def main() -> int:
    failures: list[str] = []
    for row in database_adapter_status():
        if not row["installed"]:
            failures.append(f'{row["label"]}: Python module is absent')
            continue
        try:
            # Finding a module is insufficient for binary drivers.  For
            # example, pyodbc can be installed while its libodbc dependency is
            # absent, leaving the advertised SQL Server connector unusable.
            importlib.import_module(str(row["driver_module"]))
        except Exception as exc:  # pragma: no cover - exact loader errors vary by OS
            failures.append(f'{row["label"]}: {type(exc).__name__}: {exc}')
    if failures:
        raise SystemExit(
            "release environment has unusable advertised database drivers: "
            + "; ".join(failures)
        )
    print("All advertised database drivers import successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
