"""CLI entry points for mjswan scripts."""

import subprocess
import sys
from pathlib import Path


def _run_module(module_path: str) -> None:
    """Run a module with ``python -m``."""
    project_root = Path(__file__).parent.parent.parent

    result = subprocess.run(
        [sys.executable, "-m", module_path],
        check=False,
        cwd=project_root,
    )
    sys.exit(result.returncode)


def main() -> None:
    """Run examples/demo/main.py"""
    _run_module("examples.demo.main")


def simple() -> None:
    """Run examples/demo/simple.py"""
    _run_module("examples.demo.simple")


def mjlab() -> None:
    """Run examples/mjlab/defaults.py"""
    _run_module("examples.mjlab.defaults")


def serve() -> None:
    """Launch a pre-built mjswan app from a dist directory.

    Usage: serve <dist-dir>
    """
    if len(sys.argv) < 2:
        print("Usage: serve <dist-dir>", file=sys.stderr)
        sys.exit(1)

    from mjswan.app import mjswanApp

    app = mjswanApp(Path(sys.argv[1]).resolve())
    app.launch()
