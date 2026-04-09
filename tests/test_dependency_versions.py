"""Tests for dependency version consistency across Python and npm manifests.

Layer: L1 (manifest parsing only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
TEMPLATE_PACKAGE_JSON = PROJECT_ROOT / "src" / "mjswan" / "template" / "package.json"


def _read_python_mujoco_version() -> str:
    pyproject = PYPROJECT_TOML.read_text()
    match = re.search(
        r'^\s*"mujoco==(?P<version>[^"]+)",\s*$',
        pyproject,
        re.MULTILINE,
    )
    if match is None:
        raise AssertionError(
            f"Could not find Python mujoco dependency in {PYPROJECT_TOML}"
        )
    return match.group("version")


def _read_npm_mujoco_version() -> str:
    package_json = json.loads(TEMPLATE_PACKAGE_JSON.read_text())
    version = package_json.get("dependencies", {}).get("@mujoco/mujoco")
    if version is None:
        raise AssertionError(
            f"Could not find npm @mujoco/mujoco dependency in {TEMPLATE_PACKAGE_JSON}"
        )
    return version


class TestMuJoCoDependencyVersions:
    def test_python_and_npm_mujoco_versions_match(self):
        python_version = _read_python_mujoco_version()
        npm_version = _read_npm_mujoco_version()

        assert python_version == npm_version, (
            "Python package 'mujoco' and npm package '@mujoco/mujoco' must use the "
            "same version. Passing MjModel between Python and the frontend is only "
            "guaranteed to work when both sides use the same MuJoCo version, so "
            f"keeping these versions aligned is strongly recommended "
            f"(Python: {python_version}, npm: {npm_version})."
        )
