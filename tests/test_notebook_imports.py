# -*- coding: utf-8 -*-
"""Every import a guan notebook does can actually be satisfied.

The dead `multimode_expts.` prefix
----------------------------------
An old package layout put this repo under a `multimode_expts` package. It
does not exist any more, so `from multimode_expts.experiments... import X`
raises `ModuleNotFoundError` -- but only when that cell is run, which for a
calibration cell can be months after the line was written. Fifteen of them
were still in guan's notebooks and the shared qsim analysis notebook.

What this test does
-------------------
Collects every top-level `from experiments... import` / `from fitting...
import` in guan's Jupytext notebooks and the shared analysis notebook, and
resolves each name for real. That covers both the dead prefix (the module
would not import) and the other slow failure -- a class the refactor moved or
renamed out from under a notebook.

Limited to guan's own files and the shared notebook on purpose: other users'
sandboxes are theirs, and several still carry the old prefix. Not this
test's business to fail on their behalf.
"""
import ast
import importlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "measurement_notebooks" / "guan").glob("*.py"))
NOTEBOOKS += sorted((REPO_ROOT / "analysis_notebooks" / "guan").glob("*.py"))
SHARED_IPYNB = REPO_ROOT / "analysis_notebooks" / "qsim analysis.ipynb"

# Prefixes this repo actually provides. An import of anything else (numpy,
# qutip, slab) is not this test's concern.
OURS = ("experiments", "fitting", "job_server", "slab", "multimode_expts")


def _source(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", []))
                         for cell in notebook["cells"]
                         if cell["cell_type"] == "code")
    return path.read_text(encoding="utf-8")


def _imports(path):
    """-> [(module, name)] for every in-repo `from x import y` in the file.

    Notebook sources are not always parseable as a whole (`%magic` lines), so
    this walks cell by cell and skips what will not parse -- a cell that does
    not parse cannot be run either.
    """
    found = []
    text = _source(path)
    chunks = text.split("\n# %%") if path.suffix == ".py" else text.split("\n\n")
    for chunk in chunks:
        try:
            tree = ast.parse(chunk)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in OURS:
                    for alias in node.names:
                        found.append((node.module, alias.name))
    return found


CASES = [(path.name, module, name)
         for path in NOTEBOOKS + [SHARED_IPYNB]
         for module, name in _imports(path)]


def test_the_scan_found_something():
    """Guards against the collection above silently matching nothing."""
    assert len(CASES) > 20, f"only {len(CASES)} in-repo imports found"


@pytest.mark.parametrize("notebook,module,name", sorted(set(CASES)),
                         ids=[f"{n}:{m}.{x}" for n, m, x in sorted(set(CASES))])
def test_the_import_resolves(notebook, module, name):
    try:
        imported = importlib.import_module(module)
    except ImportError as error:
        pytest.fail(f"{notebook} imports {module!r}, which does not import: "
                    f"{error}")
    assert hasattr(imported, name), (
        f"{notebook} imports {name!r} from {module!r}, which has no such "
        f"attribute -- it was probably moved or renamed")
