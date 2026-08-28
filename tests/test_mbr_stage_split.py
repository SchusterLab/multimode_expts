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
