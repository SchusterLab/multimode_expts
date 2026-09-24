# -*- coding: utf-8 -*-
"""No notebook in the repo still calls `analyze(stage=...)`.

`EncodingHamiltonianSpectroscopyExperiment.analyze` raises `TypeError` on a
`stage` argument on purpose, with a message naming the replacement class. That
makes a surviving call a cell that cannot run -- and a cell that cannot run is
only discovered when someone runs it, which for a calibration cell is months
later and on the measurement PC.

So the repo-wide sweep is the test. Scoped to notebooks rather than all of
`experiments/`, because the library side has its own coverage
(`test_mbr_stage_split.py`) and the deliberate raise itself lives there.

This is the check that would have caught `measurement_notebooks/guan/mbramsey.py`
straight after the stage split: the notebook-side migration of 2026-09-12
repaired imports, and a `stage=` argument is not an import.

Run:  pixi run python -m pytest tests/test_no_stage_dispatch_remains.py -v
"""
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRS = ["measurement_notebooks", "analysis_notebooks"]
STAGE_ARGUMENT = re.compile(
    r"\bstage\s*=\s*['\"](?:calibration|spectrum|orthogonality|propagator)['\"]")


def _notebooks():
    found = []
    for directory in NOTEBOOK_DIRS:
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix in (".py", ".ipynb") and ".ipynb_checkpoints" not in path.parts:
                found.append(path)
    return found


NOTEBOOKS = _notebooks()


def _code(path):
    if path.suffix == ".ipynb":
        notebook = json.loads(path.read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", []))
                         for cell in notebook["cells"]
                         if cell["cell_type"] == "code")
    return path.read_text(encoding="utf-8")


def test_the_sweep_found_the_notebooks():
    """Guards against the parametrization below silently covering nothing."""
    assert len(NOTEBOOKS) > 20, f"only found {len(NOTEBOOKS)} notebooks"


@pytest.mark.parametrize(
    "path", NOTEBOOKS,
    ids=[str(p.relative_to(REPO_ROOT)) for p in NOTEBOOKS])
def test_no_notebook_calls_the_retired_stage_dispatch(path):
    """The four stages are four classes; `stage=` names none of them.

    Replacement, per `analysis_notebooks/guan/MBR_analysis.py`:

        expt = <StageClass>.from_job_ids(ids, station=station)
        expt.analyze()

    and for a freshly acquired batch,
    `<StageClass>._from_expts(runner.execute(configs=...), job_ids=...)`, which
    keeps the runner recording the class the data was acquired under.
    """
    text = _code(path)
    hits = STAGE_ARGUMENT.findall(text)
    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} still passes stage={set(hits)}; "
        f"this raises. See the migration table in "
        f"analysis_notebooks/guan/MBR_analysis.py")
