# -*- coding: utf-8 -*-
"""Prove the qsim measurement-family split moved code without changing it.

``floquet_dark_mode_readout.py`` held the acquire/analyze/display triples of a
dozen unrelated measurements. Splitting them into one module per measurement is
a pure move, so it has exactly one failure mode: the code changed on the way
out. This compares every moved definition against the pre-split commit.

It is an AST comparison, not a text diff, so re-indentation or comment reflow
in a later cleanup will not make it cry wolf -- but any change to a statement
will fail it. Unlike a runtime check it covers every branch, including the
hardware-only pulse bodies no offline test can reach.

The pin is the parent of the split commit. Once these modules start evolving on
purpose, this test has done its job: delete the row, do not re-bless it.
"""
import ast
import subprocess

import pytest

PRE_SPLIT = "c7578de"
GOD = "experiments/qsim/floquet_dark_mode_readout.py"

# name -> module it moved to, under experiments/qsim/.
MOVED = {
    "BroadbandGeValidationProgram": "dark_mode_broadband_ge_validation",
    "CentralBosonLocalReturnExperiment": "central_boson_local_return",
    "CentralBosonLocalReturnProgram": "central_boson_local_return",
    "configure_central_return_metadata": "central_boson_local_return",
    "validate_central_return_occupations": "central_boson_local_return",
    "DarkBaseRProgram": "dark_mode_multiparity_chevron",
    "ManStorMultiparityChevronRExperiment": "dark_mode_multiparity_chevron",
    "ManStorMultiparityChevronRProgram": "dark_mode_multiparity_chevron",
    "DarkT1Experiment": "dark_mode_t1",
    "DarkT1Program": "dark_mode_t1",
    "FloquetDisplacementKerrExperiment": "floquet_displacement_kerr",
    "FloquetDisplacementKerrProgram": "floquet_displacement_kerr",
    "SidebandStarkAmplificationModifiedProgram": "sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_newold": "sideband_stark_shift_cal",
    "SidebandStarkAmplificationModifiedProgram_old": "sideband_stark_shift_cal",
    "StorageSwapPhaseAccumulationProgram": "storage_swap_phase_cal",
}


def _repo_root():
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True).stdout.strip()


def _definitions(source):
    """Top-level class and function definitions, keyed by name."""
    return {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }


@pytest.fixture(scope="module")
def before():
    root = _repo_root()
    shown = subprocess.run(
        ["git", "-C", root, "show", f"{PRE_SPLIT}:{GOD}"],
        capture_output=True, text=True, encoding="utf-8")
    assert shown.returncode == 0, (
        f"cannot read {GOD} at {PRE_SPLIT}: {shown.stderr}")
    return _definitions(shown.stdout)


@pytest.fixture(scope="module")
def after():
    from pathlib import Path
    root = Path(_repo_root())
    definitions = {}
    for module in set(MOVED.values()):
        path = root / "experiments" / "qsim" / f"{module}.py"
        definitions[module] = _definitions(
            path.read_text(encoding="utf-8"))
    return definitions


@pytest.mark.parametrize("name", sorted(MOVED))
def test_moved_definition_is_unchanged(name, before, after):
    module = MOVED[name]
    assert name in before, f"{name} was not in the god file at {PRE_SPLIT}"
    assert name in after[module], f"{name} is missing from {module}.py"
    assert ast.unparse(after[module][name]) == ast.unparse(before[name])


def test_nothing_moved_twice():
    """Two modules defining the same name would let filesystem order pick.

    ``experiments/__init__.py`` flattens every discovered class into one
    namespace and silently keeps whichever it imported last.
    """
    from pathlib import Path
    root = Path(_repo_root())
    owners = {}
    modules = set(MOVED.values()) | {"floquet_dark_mode_readout"}
    for module in sorted(modules):
        source = (root / "experiments" / "qsim" / f"{module}.py").read_text(
            encoding="utf-8")
        for name in _definitions(source):
            owners.setdefault(name, []).append(module)
    duplicated = {n: m for n, m in owners.items() if len(m) > 1}
    assert not duplicated


@pytest.mark.parametrize("name", sorted(MOVED))
def test_the_old_address_still_resolves(name):
    """Acquisition notebooks say ``meas.qsim.floquet_dark_mode_readout.X``."""
    import importlib

    legacy = importlib.import_module(
        "experiments.qsim.floquet_dark_mode_readout")
    new = importlib.import_module(
        f"experiments.qsim.{MOVED[name]}")
    assert getattr(legacy, name) is getattr(new, name)
    assert name not in vars(legacy), (
        f"{name} is bound in the legacy module, so the flattening exporter "
        "would export it twice")
