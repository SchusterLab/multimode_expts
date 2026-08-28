# -*- coding: utf-8 -*-
"""Prove each ``analyze(stage=...)`` branch became its own Experiment intact.

The god Experiment answered five unrelated questions behind one string
argument. Each stage moves out to the class that owns it (spec 7.3/7.4). That
is a pure move of the stage's methods, so it has one failure mode: the code
changed on the way out.

AST comparison against the commit before each stage's split, so a later
re-indentation or comment reflow cannot make it cry wolf, while any change to a
statement fails it. Each stage carries its own pin because they land in
separate commits.

Once a stage's class starts evolving on purpose, this test has done its job for
that stage: delete its row, do not re-bless it.
"""
import ast
import importlib
import subprocess
from pathlib import Path

import pytest

GOD = "experiments/qsim/floquet_dark_mode_readout.py"
GODCLASS = "EncodingHamiltonianSpectroscopyExperiment"

STAGES = [
    dict(stage="orthogonality",
         module="mbr_orthogonality",
         cls="MBROrthogonalityExperiment",
         pin="ac03ea1",
         methods=["reconstruct_orthogonality", "display_orthogonality",
                  "orthogonality_batch"]),
    dict(stage="propagator",
         module="mbr_propagator",
         cls="MBRPropagatorExperiment",
         pin="77473d3",
         methods=["reconstruct_propagator", "propagator_batch"]),
]

CASES = [(s["stage"], m) for s in STAGES for m in s["methods"]]
BY_STAGE = {s["stage"]: s for s in STAGES}


def _repo_root():
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True).stdout.strip()


def _methods(source, class_name):
    """Method definitions of one class, keyed by name."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    return {n.name: n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


@pytest.fixture(scope="module")
def before():
    """The god class's methods at each stage's pre-split commit."""
    root = _repo_root()
    out = {}
    for pin in {s["pin"] for s in STAGES}:
        shown = subprocess.run(
            ["git", "-C", root, "show", f"{pin}:{GOD}"],
            capture_output=True, text=True, encoding="utf-8")
        assert shown.returncode == 0, (
            f"cannot read {GOD} at {pin}: {shown.stderr}")
        out[pin] = _methods(shown.stdout, GODCLASS)
    return out


@pytest.fixture(scope="module")
def after():
    root = Path(_repo_root())
    return {
        s["stage"]: _methods(
            (root / "experiments" / "qsim" / f"{s['module']}.py").read_text(
                encoding="utf-8"),
            s["cls"])
        for s in STAGES
    }


@pytest.mark.parametrize("stage,method", CASES)
def test_moved_method_is_unchanged(stage, method, before, after):
    spec = BY_STAGE[stage]
    old = before[spec["pin"]]
    assert method in old, f"{method} was not on {GODCLASS} at {spec['pin']}"
    assert method in after[stage], f"{method} is missing from {spec['cls']}"
    assert ast.unparse(after[stage][method]) == ast.unparse(old[method])


@pytest.mark.parametrize("stage,method", CASES)
def test_method_left_the_god_class(stage, method):
    """Defined in two places, the subclass silently shadows -- and drifts."""
    root = Path(_repo_root())
    god = _methods((root / GOD).read_text(encoding="utf-8"), GODCLASS)
    assert method not in god


@pytest.mark.parametrize("stage", sorted(BY_STAGE))
def test_the_stage_facade_routes_to_the_owner(stage):
    """jonginn's notebooks still call analyze(stage=...); it must still land."""
    spec = BY_STAGE[stage]
    legacy = importlib.import_module(
        "experiments.qsim.floquet_dark_mode_readout")
    owner = importlib.import_module(f"experiments.qsim.{spec['module']}")
    assert legacy._stage_owner(stage) is getattr(owner, spec["cls"])


@pytest.mark.parametrize("stage", sorted(BY_STAGE))
def test_the_new_class_can_load_its_own_data(stage):
    """The loading layer is still inherited, so the new class is usable alone."""
    spec = BY_STAGE[stage]
    cls = getattr(
        importlib.import_module(f"experiments.qsim.{spec['module']}"),
        spec["cls"])
    for name in ("from_job_files", "from_job_ids", "_quadrature",
                 "_saved_parameters"):
        assert hasattr(cls, name), f"{spec['cls']} lost {name}"
    assert "analyze" in vars(cls), (
        f"{spec['cls']} must define its own analyze, not inherit the "
        "stage dispatch")


# ---------------------------------------------------------------------------
# Runtime coverage for the stages the golden baseline never reaches.
#
# tests/test_mbr_analysis_golden.py only exercises stage='spectrum' and
# stage='calibration'. Orthogonality and propagator have no recorded fixture,
# so the AST pin above is their only safety net -- and an AST pin cannot catch
# a broken *delegation*: the method can be byte-identical in its new home while
# the facade no longer reaches it, or reaches it with the wrong arguments.
#
# These build the smallest data that each display and each facade branch will
# accept. Synthetic, not physical: they assert plumbing, not numbers.

def _orthogonality_data(size=3):
    import numpy as np
    from slab import AttrDict

    matrix = np.eye(size, dtype=complex) + 0.1j * np.tri(size, k=-1)
    magnitude = np.abs(matrix)
    diagonal = np.diag(magnitude)
    return AttrDict(dict(
        matrix=matrix,
        occupations=[(size - i, i, 0) for i in range(size)],
        diagonal_amplitude=diagonal,
        offdiagonal_normalized_power=(
            magnitude ** 2 / np.outer(diagonal, diagonal)),
    ))


@pytest.fixture(autouse=True, scope="module")
def _headless():
    import matplotlib
    previous = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    yield
    matplotlib.use(previous, force=True)


def test_orthogonality_display_runs_on_its_own_class():
    import matplotlib.pyplot as plt
    from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment

    expt = MBROrthogonalityExperiment.__new__(MBROrthogonalityExperiment)
    expt.data = _orthogonality_data()
    figure = expt.display()
    assert figure is not None
    plt.close(figure)


def test_the_god_display_still_reaches_the_moved_orthogonality_plot():
    """The facade branch keys off `"matrix" in self.data`, not off `stage`."""
    import matplotlib.pyplot as plt
    from experiments.qsim.floquet_dark_mode_readout import (
        EncodingHamiltonianSpectroscopyExperiment as God,
    )

    expt = God.__new__(God)
    expt.data = _orthogonality_data()
    figure = expt.display()
    assert figure is not None
    plt.close(figure)


def test_orthogonality_display_rejects_foreign_data():
    """The guard has to survive the move, or a spectrum plots as a matrix."""
    from slab import AttrDict
    from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment

    expt = MBROrthogonalityExperiment.__new__(MBROrthogonalityExperiment)
    expt.data = AttrDict(dict(spectrum={}))
    with pytest.raises(ValueError, match="orthogonality display requires"):
        expt.display()


def test_the_god_analyze_delegates_propagator_with_its_arguments(monkeypatch):
    """Propagator has no display and no fixture, so pin the call itself.

    A wrong-arity or wrong-order delegation is exactly the failure the AST pin
    is blind to, and no recorded data would catch it either.
    """
    from experiments.qsim.floquet_dark_mode_readout import (
        EncodingHamiltonianSpectroscopyExperiment as God,
    )
    from experiments.qsim.mbr_propagator import MBRPropagatorExperiment

    seen = {}

    def spy(cls, expts, occupations=None):
        seen["expts"] = expts
        seen["occupations"] = occupations
        return "reconstructed"

    monkeypatch.setattr(MBRPropagatorExperiment, "reconstruct_propagator",
                        classmethod(spy))
    expt = God.__new__(God)
    expt.data = {}
    expt.batch_expts = ["job-a", "job-b"]
    order = [(2, 0, 0), (1, 1, 0)]

    assert expt.analyze(stage="propagator", occupations=order) == "reconstructed"
    assert seen == dict(expts=["job-a", "job-b"], occupations=order)
