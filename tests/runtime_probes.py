"""Safe, cached probes for optional scientific runtimes used by tests."""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache


_R_PACKAGE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._]*\Z")


@lru_cache(maxsize=None)
def r_package_loadable(
    rscript: str | None,
    package: str,
    timeout_seconds: float = 60.0,
) -> bool:
    """Return whether an R package can be loaded without failing collection.

    CI hosts can take substantially longer to start R for the first time,
    particularly on Windows while antivirus scanning is active. Optional
    scientific dependencies must skip cleanly when their probe cannot finish;
    they must never abort pytest collection.
    """
    if rscript is None or not _R_PACKAGE_NAME.fullmatch(package):
        return False

    expression = (
        f'suppressMessages(library("{package}", character.only=TRUE))'
    )
    try:
        result = subprocess.run(
            [rscript, "--vanilla", "-e", expression],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0
