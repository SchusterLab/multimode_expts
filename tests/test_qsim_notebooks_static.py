# -*- coding: utf-8 -*-
"""The qsim migration notebooks' names and imports, without running them.

The notebooks are a suite for hardware runs (docs/qsim/mock_suite_plan.md),
so a broken name in one would otherwise first show up on the measurement PC,
mid-session. These two checks find most of those breaks in seconds, with no
station and no mock:

- Every import in a notebook resolves: the module imports, and each name
  taken from it exists. A helper renamed or moved during the refactor fails
  here.
- No undefined names (ruff F821). A jupytext ``.py`` notebook is one module,
  so a name used in one cell and bound in none fails here.

What this does not check: that a call's arguments still match a changed
signature, or anything at run time. The pytest mock acquisition tests and a
hardware run cover those.

Run:  pixi run python -m pytest tests/test_qsim_notebooks_static.py -v
"""
import ast
import importlib
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIRS = [
    REPO_ROOT / "measurement_notebooks" / "202609_qsim_migration",
    REPO_ROOT / "analysis_notebooks" / "202609_qsim_migration",
]
NOTEBOOKS = sorted(p for d in NOTEBOOK_DIRS for p in d.glob("*.py"))


def _imports(path):
    """-> [(line, module, [names])] for every import in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [alias.name for alias in node.names if alias.name != "*"]
            found.append((node.lineno, node.module, names))
    return found


def _rel(path):
    return path.relative_to(REPO_ROOT).as_posix()


@pytest.mark.parametrize("path", NOTEBOOKS, ids=_rel)
def test_notebook_imports_resolve(path):
    problems = []
    for line, module_name, names in _imports(path):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            problems.append(f"line {line}: import {module_name}: "
                            f"{type(exc).__name__}: {exc}")
            continue
        for name in names:
            if hasattr(module, name):
                continue
            try:  # a submodule not yet imported by its package
                importlib.import_module(f"{module_name}.{name}")
            except ImportError:
                problems.append(f"line {line}: {module_name} has no {name!r}")
    assert not problems, f"{_rel(path)}:\n  " + "\n  ".join(problems)


def test_notebooks_have_no_undefined_names():
    """ruff F821, run through `pixi exec` so it needs no project dependency."""
    pixi = shutil.which("pixi")
    if pixi is None:
        pytest.skip("pixi is not on PATH")
    result = subprocess.run(
        [pixi, "exec", "ruff", "check", "--select", "F821",
         "--output-format", "concise", *map(str, NOTEBOOKS)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    if result.returncode not in (0, 1):  # 1 = violations found
        pytest.skip(f"ruff did not run: {result.stderr.strip()[-300:]}")
    assert result.returncode == 0, result.stdout
