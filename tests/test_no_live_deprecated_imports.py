# -*- coding: utf-8 -*-
"""Live code does not import ``experiments.qsim.deprecated`` (MBR redesign step 7e).

docs/qsim/mbr_step7_plan.md, section 2: code that left the canonical path
moved to ``experiments/qsim/deprecated/`` and the ``dormant/`` notebook
folders. Moved code may import live code, never the other way. Tests are not
live code: they may import deprecated code, for example to compare an old
program's pulses with the new one's.

Live code here: every module under ``experiments/`` (but ``deprecated/``),
``fitting/``, ``job_server/`` and ``tools/``, and the canonical migration
notebooks (not ``dormant/``). Only real import statements count, not strings.
"""
import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPRECATED = "experiments.qsim.deprecated"

LIVE_DIRS = ["experiments", "fitting", "job_server", "tools"]
NOTEBOOK_DIRS = [
    REPO_ROOT / "measurement_notebooks" / "202609_qsim_migration",
    REPO_ROOT / "analysis_notebooks" / "202609_qsim_migration",
]


def live_files():
    files = []
    for name in LIVE_DIRS:
        for path in (REPO_ROOT / name).rglob("*.py"):
            parts = path.relative_to(REPO_ROOT).parts
            if "deprecated" in parts or ".ipynb_checkpoints" in parts:
                continue
            files.append(path)
    for directory in NOTEBOOK_DIRS:
        files.extend(sorted(directory.glob("*.py")))  # not dormant/
    return sorted(files)


def deprecated_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == DEPRECATED or node.module.startswith(DEPRECATED + "."):
                found.append(f"{node.lineno}: from {node.module}")
            elif node.module == "experiments.qsim" and any(
                    alias.name == "deprecated" for alias in node.names):
                found.append(f"{node.lineno}: from experiments.qsim import deprecated")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == DEPRECATED or alias.name.startswith(DEPRECATED + "."):
                    found.append(f"{node.lineno}: import {alias.name}")
    return found


def test_there_is_live_code_to_check():
    assert len(live_files()) > 100


@pytest.mark.parametrize("path", live_files(),
                         ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_live_code_does_not_import_deprecated(path):
    found = deprecated_imports(path)
    assert not found, (f"{path.relative_to(REPO_ROOT)} imports deprecated code: {found}. "
                       f"See docs/qsim/mbr_step7_plan.md, section 2.")
