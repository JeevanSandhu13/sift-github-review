"""Package entry point: ``python -m sift`` brings up the pywebview window.

The .app launcher and the ``sift`` console-script (declared in
pyproject.toml) both ultimately call ``sift.ui.main``; this module
exists so ``python -m sift`` from a source checkout takes the same
path without going through the script shim.
"""

from sift.ui import main


if __name__ == "__main__":
    main()
