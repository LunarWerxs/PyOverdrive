"""Packaging invariants: what a user installs must describe itself correctly.

These are cheap, but each one guards a defect that actually shipped or
would have been invisible until a user hit it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pyoverdrive

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)


def test_version_matches_pyproject():
    # A hardcoded __version__ reported 0.0.1.dev0 out of a 0.1.0 wheel;
    # __version__ now reads the installed distribution's metadata, and
    # this pins the two together.
    declared = _pyproject()["project"]["version"]
    assert pyoverdrive.__version__ == declared


def test_version_is_pep440_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]?(a|b|rc|dev)\d+)?", pyoverdrive.__version__)


def test_typed_marker_ships_with_the_package():
    # the "Typing :: Typed" classifier is a promise to type checkers;
    # without py.typed inside the package it is a false claim
    marker = Path(pyoverdrive.__file__).parent / "py.typed"
    assert marker.is_file()
    classifiers = _pyproject()["project"]["classifiers"]
    assert "Typing :: Typed" in classifiers
    assert "pyoverdrive" in _pyproject()["tool"]["setuptools"]["package-data"]


def test_readme_and_license_are_declared():
    project = _pyproject()["project"]
    assert project["readme"] == "README.md"
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / project["license"]["file"]).is_file()


def test_project_urls_point_at_the_public_repo():
    urls = _pyproject()["project"]["urls"]
    for key in ("Homepage", "Repository", "Issues", "Changelog"):
        assert urls[key].startswith("https://github.com/LunarWerxs/PyOverdrive")


def test_public_api_is_exported():
    for name in (
        "enable", "disable", "enabled", "explain", "selfcheck",
        "status", "report", "calibrate", "disable_path", "enable_path",
        "supported_operations", "FastPath", "__version__",
    ):
        assert name in pyoverdrive.__all__, name
        assert hasattr(pyoverdrive, name), name
