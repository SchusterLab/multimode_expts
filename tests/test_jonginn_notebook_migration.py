# -*- coding: utf-8 -*-
"""Jonginn's two notebooks still address the MBR stage classes correctly.

Why these two notebooks get their own file
------------------------------------------
`test_notebook_imports.py` is scoped to guan's sandbox on purpose. These two
are the notebooks the refactored code was accumulated from -- 23,103 code
lines between them, and the only in-repo callers of several MBR workflows --
so a refactor that moves a class out from under them should fail here rather
than on the measurement PC.

They are `.ipynb`, not Jupytext, and `data_postprocess.ipynb` carries 27
stored figures, so they are read as JSON rather than imported.

What is checked
---------------
1. `analyze(stage=...)` is gone. The god Experiment raises on it by design, so
   any surviving call is a cell that cannot run.
2. Every `<StageClass>.<attr>` the notebooks name actually exists. This is the
   check that catches the next move: `MBRSpectrumExperiment.spectroscopy_batch`
   going somewhere else breaks a test instead of a campaign.
3. Every `analyze`/`display` call on an aggregate whose class can be traced
   accepts the keywords it passes. The stage signatures name their knobs now,
   so a keyword that used to vanish into `**kwargs` raises -- which is the
   point, but it means the notebooks have to be right.
4. The acquisition provenance is untouched: `ExptClass=` and the HDF5 filename
   construction still name the class the jobs were acquired under.
5. `tools/migrate_jonginn_notebooks.py --check` reports nothing left to do, so
   the migration in the tree matches the migration the script describes.

Run:  pixi run python -m pytest tests/test_jonginn_notebook_migration.py -v
"""
import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "measurement_notebooks" / "jonginn"

# `qsim_experiments.ipynb` and `data_postprocess.ipynb` were retired by the
# stage-2 notebook split; their successors are the themed Jupytext entry
# points checked by SUCCESSOR_DIRS below, which get the same checks. The
# high-Kerr sibling was explicitly outside that pass and is still an .ipynb,
# so it keeps the JSON path: a near-duplicate of the retired first notebook
# (223 of 263 code cells were byte-identical) that carries 179 stored figures
# of its own.
NAMES = ["qsim_experiments_highkerr_untracked_refactored.ipynb"]

# The stage-2 successors. Jupytext `py:percent` files are valid Python, so
# these are read as text rather than as notebook JSON.
SUCCESSOR_DIRS = [
    REPO_ROOT / "measurement_notebooks" / "202609_qsim_migration",
    REPO_ROOT / "analysis_notebooks" / "202609_qsim_migration",
    REPO_ROOT / "experiments" / "qsim" / "notebook_helpers",
]

STAGE_MODULES = {
    "MBRPhaseCorrectionExperiment": "experiments.qsim.legacy_mbr",
    "MBRSpectrumExperiment": "experiments.qsim.legacy_mbr",
    "MBROrthogonalityExperiment": "experiments.qsim.legacy_mbr",
    "MBRPropagatorExperiment": "experiments.qsim.legacy_mbr",
    "EncodingHamiltonianSpectroscopyExperiment":
        "experiments.qsim.floquet_dark_mode_readout",
}
LOADERS = ("from_job_files", "from_job_ids", "_from_expts", "from_batch")
STAGE_ARGUMENT = re.compile(
    r"\bstage\s*=\s*['\"](?:calibration|spectrum|orthogonality|propagator)['\"]")


@pytest.fixture(scope="module")
def stage_classes():
    return {name: getattr(importlib.import_module(module), name)
            for name, module in STAGE_MODULES.items()}


def _cells(name):
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    return ["".join(cell.get("source", []))
            for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _successor_sources():
    """(label, [source chunks]) for each stage-2 successor file."""
    out = []
    for directory in SUCCESSOR_DIRS:
        for path in sorted(directory.rglob("*.py")):
            out.append((
                str(path.relative_to(REPO_ROOT)),
                [path.read_text(encoding="utf-8", errors="replace")],
            ))
    return out


SUCCESSORS = _successor_sources()


@pytest.fixture(scope="module", params=NAMES)
def notebook(request):
    return request.param, _cells(request.param)


@pytest.fixture(scope="module", params=[label for label, _ in SUCCESSORS])
def successor(request):
    """One stage-2 successor file, as a single source chunk."""
    return request.param, dict(SUCCESSORS)[request.param]


def test_the_scan_found_the_notebooks(notebook):
    """Guards against every test below passing on an empty list."""
    name, cells = notebook
    assert len(cells) > 200, f"{name}: only {len(cells)} code cells"


# Only two of the four notebook checks carry over to the successors. The other
# two -- the analyze/display keyword check and the acquisition-provenance
# check -- read per-cell notebook conventions: a traceable receiver for
# `analyze(...)`, and a literal `ExptClass=` beside an HDF5 filename. In the
# helper modules the receiver is a function parameter and the class arrives via
# `campaign.EncSpec`, so those checks would be asserting a shape this code
# deliberately does not have. The mock-acquisition suite covers that ground
# instead, by building the programs.


def test_no_successor_still_passes_stage(successor):
    """`EncSpec.analyze(stage=...)` raises, so a survivor is dead code."""
    name, chunks = successor
    offenders = [i for i, source in enumerate(chunks, 1)
                 if STAGE_ARGUMENT.search(source)]
    assert not offenders, f"{name}: stage= still present"


def test_every_successor_stage_attribute_exists(successor, stage_classes):
    """The check that catches the next move out from under the new entry points."""
    name, chunks = successor
    missing = []
    for source in chunks:
        for cls_name, cls in stage_classes.items():
            for attr in re.findall(rf"\b{cls_name}\.(\w+)", source):
                if not hasattr(cls, attr):
                    missing.append(f"{cls_name}.{attr}")
    assert not missing, f"{name}: missing {sorted(set(missing))}"


def test_the_retired_notebooks_are_gone():
    """The stage-2 split retired these two; nothing should resurrect them.

    If one comes back, the checks above stop covering it and the split's
    accounting no longer holds.
    """
    for retired in ("qsim_experiments.ipynb", "data_postprocess.ipynb"):
        assert not (NOTEBOOKS / retired).exists(), (
            f"{retired} is back; either re-add it to NAMES or remove it again"
        )


def test_the_successors_exist():
    """Guards against SUCCESSOR_DIRS silently going empty."""
    found = _successor_sources()
    assert len(found) >= 30, (
        f"expected the stage-2 tree, found {len(found)} files"
    )


def test_no_cell_still_passes_stage(notebook):
    """`EncSpec.analyze(stage=...)` raises, so a survivor is a dead cell."""
    name, cells = notebook
    offenders = [index for index, source in enumerate(cells, 1)
                 if STAGE_ARGUMENT.search(source)]
    assert not offenders, f"{name}: stage= still in cells {offenders}"


def test_every_stage_class_attribute_exists(notebook, stage_classes):
    """The check that catches the *next* move out from under these notebooks."""
    name, cells = notebook
    missing = []
    for source in cells:
        for owner, attribute in re.findall(
                r"\b(MBR\w+Experiment|EncodingHamiltonianSpectroscopyExperiment)"
                r"\.(\w+)", source):
            if attribute == "__name__":
                continue
            if not hasattr(stage_classes[owner], attribute):
                missing.append(f"{owner}.{attribute}")
    assert not missing, f"{name}: {sorted(set(missing))}"


def _aggregate_owners(cells, stage_classes):
    """-> {variable: stage class name} for aggregates built by a stage class."""
    owners = {}
    for source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.value, ast.Call)):
                continue
            call = node.value.func
            if (isinstance(call, ast.Attribute) and call.attr in LOADERS
                    and isinstance(call.value, ast.Name)
                    and call.value.id in stage_classes):
                owners[ast.unparse(node.targets[0])] = call.value.id
    return owners


def test_analyze_and_display_accept_what_they_are_passed(notebook, stage_classes):
    """The stage signatures have no `**kwargs`, so a wrong keyword raises.

    Only calls whose receiver can be traced back to a stage constructor are
    checked; the rest are notebook-local objects this test knows nothing
    about. `test_every_analysed_aggregate_is_a_stage_object` below is what
    keeps that set from quietly shrinking to nothing.
    """
    name, cells = notebook
    owners = _aggregate_owners(cells, stage_classes)
    problems, checked = [], 0
    for index, source in enumerate(cells, 1):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("analyze", "display")):
                continue
            owner = owners.get(ast.unparse(node.func.value))
            if owner is None:
                continue
            checked += 1
            keywords = [k.arg for k in node.keywords if k.arg]
            signature = inspect.signature(
                getattr(stage_classes[owner], node.func.attr))
            try:
                signature.bind_partial(None, **{k: None for k in keywords})
            except TypeError as error:
                problems.append(
                    f"cell {index}: {owner}.{node.func.attr}"
                    f"({', '.join(keywords)}) -> {error}")
    assert checked > 20, f"{name}: only {checked} calls traced; scan is broken"
    assert not problems, f"{name}:\n" + "\n".join(problems)


def test_acquisition_provenance_is_untouched(notebook):
    """`ExptClass=` and the HDF5 filename still name the acquired class.

    The queue records `experiment_class`/`experiment_module` per job and the
    saved file is `JOB-<id>_<ClassName>.h5`, so pointing acquisition at a
    stage class would orphan every dataset jonginn has. `from_batch` exists so
    the aggregate can be re-wrapped for analysis without touching this.
    """
    name, cells = notebook
    text = "\n".join(cells)
    for stage_class in ("MBRPhaseCorrectionExperiment", "MBRSpectrumExperiment",
                        "MBROrthogonalityExperiment", "MBRPropagatorExperiment"):
        assert f"ExptClass={stage_class}" not in text, (
            f"{name}: acquisition must keep recording the class the data was "
            f"acquired under, not {stage_class}")
        assert f"{{{stage_class}.__name__}}" not in text, (
            f"{name}: saved filenames must keep naming the acquired class")
    # And nothing derives a saved filename or a queue record from a name the
    # migration rebound. This is the failure that does not raise: the loader
    # goes looking for `JOB-..._MBRSpectrumExperiment.h5`, finds nothing, and
    # reports missing data for jobs sitting right there on disk.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_migrate", REPO_ROOT / "tools" / "migrate_jonginn_notebooks.py")
    migrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migrate)
    plan = next(p for p in migrate.PLANS if p["notebook"] == name)
    notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
    leaked = migrate.rebound_alias_in_provenance(notebook, plan["aliases"])
    assert not leaked, f"{name}: {leaked}"


def test_the_migration_script_has_nothing_left_to_do():
    """The tree matches what the script says the migration is.

    Without this the script could drift from the notebooks and its `--check`
    would stop being the thing that finds the next gap.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "migrate_jonginn_notebooks.py"),
         "--check"],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("nothing to do") == len(NAMES), result.stdout
