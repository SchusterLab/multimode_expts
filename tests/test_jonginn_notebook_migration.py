"""The retired Jonginn notebooks are replaced by maintained entry points."""
import ast
import importlib
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "measurement_notebooks" / "jonginn"
SUCCESSOR_DIRS = [REPO_ROOT / path for path in (
    "measurement_notebooks/202609_qsim_migration",
    "analysis_notebooks/202609_qsim_migration",
    "experiments/qsim/notebook_helpers")]
STAGE_MODULES = {
    "MBRSpectrumExperiment": "experiments.qsim.mbr_spectrum",
    "MBROrthogonalityExperiment": "experiments.qsim.mbr_orthogonality",
    "MBRDisorderEnsembleExperiment": "experiments.qsim.mbr_disorder_ensemble",
    "MBRCalibrationSetExperiment": "experiments.qsim.mbr_calibration_set",
}
STAGE_ARGUMENT = re.compile(r"\bstage\s*=\s*['\"](?:calibration|spectrum|orthogonality|propagator)['\"]")

@pytest.fixture(scope="module")
def stage_classes():
    return {name: getattr(importlib.import_module(module), name)
            for name, module in STAGE_MODULES.items()}

def _successor_sources():
    """(label, [source chunks]) for each stage-2 successor file."""
    out = []
    for directory in SUCCESSOR_DIRS:
        for path in sorted(directory.rglob("*.py")):
            if ".ipynb_checkpoints" in path.parts:
                continue              # Jupyter autosaves, not tracked
            out.append((
                str(path.relative_to(REPO_ROOT)),
                [path.read_text(encoding="utf-8", errors="replace")],
            ))
    return out


SUCCESSORS = _successor_sources()


@pytest.fixture(scope="module", params=[label for label, _ in SUCCESSORS])
def successor(request):
    """One stage-2 successor file, as a single source chunk."""
    return request.param, dict(SUCCESSORS)[request.param]


def test_no_successor_still_passes_stage(successor):
    """`EncSpec.analyze(stage=...)` raises, so a survivor is dead code."""
    name, chunks = successor
    offenders = [i for i, source in enumerate(chunks, 1)
                 if STAGE_ARGUMENT.search(source)]
    assert not offenders, f"{name}: stage= still present"


def _imported_names(source):
    """-> {local name: object} for the top-level `from x import y` lines of a file."""
    names = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                local = alias.asname or alias.name
                if local in STAGE_MODULES:
                    module = importlib.import_module(node.module)
                    names[local] = getattr(module, alias.name)
    return names


def test_every_successor_stage_attribute_exists(successor, stage_classes):
    """The check that catches the next move out from under the new entry points."""
    name, chunks = successor
    missing = []
    for source in chunks:
        imported = _imported_names(source)
        for cls_name, cls in stage_classes.items():
            # Since MBR redesign step 6b a successor may import the new class
            # under the old name (experiments.qsim.mbr_spectrum); check the
            # class the file actually imports.
            cls = imported.get(cls_name, cls)
            for attr in re.findall(rf"\b{cls_name}\.(\w+)", source):
                if not hasattr(cls, attr):
                    missing.append(f"{cls_name}.{attr}")
    assert not missing, f"{name}: missing {sorted(set(missing))}"


def test_the_retired_notebooks_are_gone():
    for name in (
        "qsim_experiments.ipynb", "data_postprocess.ipynb", "data_recollecting.ipynb",
        "qsim_experiments_highkerr_untracked.ipynb",
        "qsim_experiments_highkerr_untracked_refactored.ipynb"):
        assert not (NOTEBOOKS / name).exists(), f"{name} has been resurrected"


def test_the_successors_exist():
    assert len(_successor_sources()) >= 20
