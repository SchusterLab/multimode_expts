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
import inspect
import subprocess
from pathlib import Path

import pytest

GOD = "experiments/qsim/floquet_dark_mode_readout.py"
GODCLASS = "EncodingHamiltonianSpectroscopyExperiment"

STAGES = [
    dict(stage="spectrum",
         module="mbr_spectrum",
         cls="MBRSpectrumExperiment",
         pin="be90ca8",
         # ``subsample_spectroscopy_shots`` used to be pinned here. Its
         # numerics moved on to fitting/qsim/mbr_reconstruction.py and the
         # method is now a lane-resolving wrapper, so the pin has done its
         # job for it: the row is deleted rather than re-blessed, per the
         # note at the top of this file.
         methods=["_postprocess_reconstruction",
                  "reconstruct_pair_spectroscopy", "reconstruct_spectroscopy",
                  "analyze_matrix_pencil_occupation", "analyze_level_statistics",
                  "analyze_sff", "display_local_density_of_states",
                  "display_occupation", "display_occupations",
                  "display_level_statistics", "display_sff",
                  "display_matrix_pencil", "display_matrix_pencil_occupation",
                  "display_result", "spectroscopy_batch"],
         edits=[
             ("EncodingHamiltonianSpectroscopyExperiment"
              ".display_local_density_of_states",
              "MBRSpectrumExperiment.display_local_density_of_states"),
         ]),
    dict(stage="calibration",
         module="mbr_phase_correction",
         cls="MBRPhaseCorrectionExperiment",
         pin="c8ce067",
         methods=["_calibration_data", "phase_correction_from_calibration",
                  "analyze_cycle_phase", "display_cycle_phase",
                  "display_calibration_results", "display_calibration_summary",
                  "analyze_calibration", "calibration_batch"],
         # Declared edits: names the move invalidated, re-addressed. Applied to
         # the pre-split source before comparing, so every other statement
         # stays pinned. Same idea as tools/verify_moved_code.py's normalize().
         edits=[
             ("cls.analyze(calibration, stage='calibration')",
              "cls.analyze(calibration)"),
             ("EncodingHamiltonianSpectroscopyExperiment.display_cycle_phase",
              "MBRPhaseCorrectionExperiment.display_cycle_phase"),
         ]),
    dict(stage="orthogonality",
         module="mbr_orthogonality",
         cls="MBROrthogonalityExperiment",
         pin="ac03ea1",
         # ``display_orthogonality`` and ``orthogonality_batch`` were pinned
         # here until their `**kwargs` became named parameters. That was an
         # intentional edit, so their rows are deleted rather than
         # re-blessed; `test_the_stage_signatures_name_their_knobs` below
         # covers what replaced them.
         methods=["reconstruct_orthogonality"]),
    dict(stage="propagator",
         module="mbr_propagator",
         cls="MBRPropagatorExperiment",
         pin="77473d3",
         # ``reconstruct_propagator`` and ``propagator_batch`` were pinned
         # here until abe358f added ``phase_correction_location``. That was
         # an intentional edit, so their rows are deleted rather than
         # re-blessed.
         methods=[]),
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

    was = ast.unparse(old[method])
    for target, replacement in spec.get("edits", ()):
        was = was.replace(target, replacement)
    assert ast.unparse(after[stage][method]) == was


@pytest.mark.parametrize("stage", sorted(BY_STAGE))
def test_every_declared_edit_was_needed(stage, before):
    """An edit that no longer matches is a hole in the pin, not a no-op."""
    spec = BY_STAGE[stage]
    source = "\n".join(
        ast.unparse(before[spec["pin"]][m]) for m in spec["methods"])
    for target, _ in spec.get("edits", ()):
        assert target in source, (
            f"{stage}: declared edit {target!r} matches nothing at "
            f"{spec['pin']}; delete the row instead of leaving it")


@pytest.mark.parametrize("stage,method", CASES)
def test_method_left_the_god_class(stage, method):
    """Defined in two places, the subclass silently shadows -- and drifts."""
    root = Path(_repo_root())
    god = _methods((root / GOD).read_text(encoding="utf-8"), GODCLASS)
    assert method not in god


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
# These build the smallest data each display will accept. Synthetic, not
# physical: they assert plumbing, not numbers.

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


# Each stage class is both a single job (what the worker runs) and the
# aggregate over jobs. The ``hasattr(self, "batch_expts")`` guards that route
# between the two send aggregate data without child jobs down the per-job path.
# Splitting the two roles into separate classes fixes this; these tests are its
# acceptance criteria. strict: remove the marker once they pass.
_TWO_ROLES = pytest.mark.xfail(
    strict=True,
    reason="stage class doubles as per-job class; batch_expts guard misroutes")


@_TWO_ROLES
def test_orthogonality_display_runs_on_its_own_class():
    import matplotlib.pyplot as plt
    from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment

    expt = MBROrthogonalityExperiment.__new__(MBROrthogonalityExperiment)
    expt.data = _orthogonality_data()
    figure = expt.display()
    assert figure is not None
    plt.close(figure)


@_TWO_ROLES
def test_orthogonality_display_rejects_foreign_data():
    """The guard has to survive the move, or a spectrum plots as a matrix."""
    from slab import AttrDict
    from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment

    expt = MBROrthogonalityExperiment.__new__(MBROrthogonalityExperiment)
    expt.data = AttrDict(dict(spectrum={}))
    with pytest.raises(ValueError, match="orthogonality display requires"):
        expt.display()


# ---------------------------------------------------------------------------
# The break is deliberate, so it has to be a good break.
#
# No shim forwards the old addresses: `stage=` is gone and so are both
# __getattr__ hooks. What replaces them is a message that names the class to
# use. Consumers migrate once, against the exemplar, instead of running on
# forwarding layers indefinitely.

def test_no_forwarding_shim_survives():
    """A shim left behind is a consumer that never migrates."""
    root = Path(_repo_root())
    source = (root / GOD).read_text(encoding="utf-8")
    for shim in ("_stage_owner", "_MOVED_METHODS", "_StageForwarding"):
        assert shim not in source, f"{shim} is still forwarding"


def test_the_god_analyze_has_no_stage_branches():
    root = Path(_repo_root())
    god = _methods((root / GOD).read_text(encoding="utf-8"), GODCLASS)
    for method in ("analyze", "display"):
        source = ast.unparse(god[method])
        for stage in ("calibration", "orthogonality", "propagator", "spectrum"):
            assert f"'{stage}'" not in source or "_stage_migration_message" in source


@pytest.mark.parametrize("stage", sorted(BY_STAGE))
def test_the_stage_error_names_its_replacement(stage):
    """The traceback is the migration note, so it must be actionable."""
    from experiments.qsim.floquet_dark_mode_readout import (
        STAGE_CLASSES,
        EncodingHamiltonianSpectroscopyExperiment as God,
    )
    spec = BY_STAGE[stage]
    assert STAGE_CLASSES[stage].endswith("." + spec["cls"])
    assert STAGE_CLASSES[stage] == (
        f"experiments.qsim.{spec['module']}.{spec['cls']}")

    expt = God.__new__(God)
    expt.data = {}
    with pytest.raises(TypeError) as raised:
        expt.analyze(stage=stage)
    message = str(raised.value)
    assert spec["cls"] in message
    assert f"experiments.qsim.{spec['module']}" in message
    assert "MBR_analysis.py" in message


def test_the_named_replacement_actually_imports():
    """A message naming a class that does not exist is worse than no message."""
    from experiments.qsim.floquet_dark_mode_readout import STAGE_CLASSES

    for stage, target in STAGE_CLASSES.items():
        module, _, name = target.rpartition(".")
        assert hasattr(importlib.import_module(module), name), (
            f"{stage} points at {target}, which does not resolve")


def test_an_unknown_stage_lists_the_real_ones():
    from experiments.qsim.floquet_dark_mode_readout import (
        EncodingHamiltonianSpectroscopyExperiment as God,
    )
    expt = God.__new__(God)
    expt.data = {}
    with pytest.raises(TypeError, match="unknown stage"):
        expt.analyze(stage="nonsense")


# ---------------------------------------------------------------------------
# The explicit analyze signature.
#
# Two things changed that nothing else covers. The 19 Matrix-Pencil defaults
# the old code spelled out were each identical to analyze_matrix_pencil's own,
# so forwarding only what the caller passed is equivalent -- but only if the
# forwarding actually works. And the old kwargs.get chain silently ignored a
# typo, which is the failure these lock out.

def _spectrum_expt():
    from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment
    return MBRSpectrumExperiment.__new__(MBRSpectrumExperiment)


def test_the_analyze_signature_names_its_knobs():
    """The point of the change: discoverable from the signature, not the body."""
    import inspect
    from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

    parameters = inspect.signature(MBRSpectrumExperiment.analyze).parameters
    for name in ("occupations", "calibration", "cycle_branches",
                 "second_branch", "phase_frame", "manual_kerr_MHz", "legacy",
                 "spectrum_method", "fft_window", "zero_padding",
                 "shots_per_point", "shot_seed"):
        assert name in parameters, f"analyze lost the {name} parameter"
    assert "kwargs" not in parameters


def test_matrix_pencil_options_lose_their_prefix():
    from experiments.qsim.mbr_spectrum import _matrix_pencil_options

    assert _matrix_pencil_options({"mpm_pencil_length": 7}) == {
        "pencil_length": 7}
    assert _matrix_pencil_options({}) == {}
    # Accepted unprefixed too, since that is what the callee actually names.
    assert _matrix_pencil_options({"pencil_length": 7}) == {"pencil_length": 7}


def test_a_misspelled_matrix_pencil_option_raises():
    """The old kwargs.get chain ignored this, so the knob silently did nothing."""
    from experiments.qsim.mbr_spectrum import _matrix_pencil_options

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _matrix_pencil_options({"mpm_pencil_lenght": 7})


@_TWO_ROLES
def test_an_unknown_analyze_argument_raises():
    expt = _spectrum_expt()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        expt.analyze(mpm_not_a_real_knob=1)
    # And a plain misspelling of a real parameter, which **kwargs would swallow.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        expt.analyze(zero_paddding=2)


def test_the_forwarded_options_reach_analyze_matrix_pencil():
    """Equivalence to the old 19-line block rests on this actually forwarding."""
    from experiments.qsim.mbr_spectrum import _matrix_pencil_options
    from fitting.qsim import matrix_pencil

    forwarded = _matrix_pencil_options({
        "mpm_pencil_length": 5,
        "mpm_minimum_consecutive_ranks": 4,
        "mpm_store_rank_sweeps": True,
    })
    parameters = inspect.signature(matrix_pencil.analyze_matrix_pencil).parameters
    for name, value in forwarded.items():
        assert name in parameters
        assert parameters[name].default != value, (
            f"{name} test value equals the default, so this proves nothing")


def test_every_old_mpm_default_matched_the_callee():
    """The collapse is only sound because the two sets of defaults agreed.

    Pins that agreement: if analyze_matrix_pencil changes a default, this is
    the record of what the pre-collapse call site used to pass.
    """
    from fitting.qsim import matrix_pencil

    was = {
        "requested_max_modes": None, "pencil_length": None,
        "minimum_consecutive_ranks": 3, "minimum_supporting_rows": 1,
        "track_frequency_tolerance_bins": 1.5,
        "merge_frequency_tolerance_bins": None,
        "dedup_frequency_tolerance_bins": None,
        "track_decay_tolerance_per_us": None,
        "dedup_decay_tolerance_per_us": None, "match_decay": True,
        "numerical_floor": 1e-10, "noise_singular_value_factor": 2.858,
        "minimum_pole_radius": 0.2, "maximum_pole_radius": 1.05,
        "require_early_start": True, "rank_sweep_extra": None,
        "clip_growth": True, "least_squares_rcond": None,
        "store_rank_sweeps": False,
    }
    parameters = inspect.signature(matrix_pencil.analyze_matrix_pencil).parameters
    # Subset, not equality: new options may be added (e.g. the
    # calibration-derived merge tolerance). What must not drift is the
    # default of anything the pre-collapse call site passed.
    assert set(was) <= set(parameters)
    for name, value in was.items():
        assert parameters[name].default == value, (
            f"{name}: analyze_matrix_pencil now defaults to "
            f"{parameters[name].default!r}, but the old call site passed "
            f"{value!r}. Collapsing the block changed behaviour.")


# --------------------------------------------------------------------------
# Explicit signatures. The spectrum stage's `analyze` got this treatment
# first (see the tests above); these are the other three.
# --------------------------------------------------------------------------

EXPLICIT = [
    ("mbr_phase_correction", "MBRPhaseCorrectionExperiment", "analyze",
     ["data", "occupations", "cycle_pairs", "repeats"]),
    ("mbr_phase_correction", "MBRPhaseCorrectionExperiment", "display",
     ["data", "ncols"]),
    ("mbr_orthogonality", "MBROrthogonalityExperiment", "analyze",
     ["data", "occupations"]),
    ("mbr_orthogonality", "MBROrthogonalityExperiment", "display",
     ["data", "figsize"]),
    ("mbr_orthogonality", "MBROrthogonalityExperiment",
     "display_orthogonality", ["data", "figsize"]),
    ("mbr_orthogonality", "MBROrthogonalityExperiment",
     "orthogonality_batch",
     ["default_expt_cfg", "swap_stors", "occupations", "sync_cycles", "reps",
      "correction_mode"]),
    # The last four came back with the tomography branch. They were read out
    # of `**kwargs` by the old string dispatch; naming them is the same
    # treatment the other three stages got.
    ("mbr_propagator", "MBRPropagatorExperiment", "analyze",
     ["data", "occupations", "calibration", "floquet_cycle_us",
      "finite_difference_cycles", "eigenphase_cycle"]),
]


@pytest.mark.parametrize("module,cls,method,expected",
                         EXPLICIT,
                         ids=[f"{c}.{m}" for _, c, m, _ in EXPLICIT])
def test_the_stage_signatures_name_their_knobs(module, cls, method, expected):
    """No `**kwargs`, so a misspelled argument raises instead of vanishing.

    Every one of these used to end in `**kwargs`, and in most of them the
    kwargs were read by nothing at all: `expt.analyze(occupation=[...])` --
    singular by accident -- ran to completion and analysed the default order
    instead. The failure was a plot that looked plausible.
    """
    owner = getattr(importlib.import_module(f"experiments.qsim.{module}"), cls)
    signature = inspect.signature(getattr(owner, method))
    names = [name for name, parameter in signature.parameters.items()
             if name != "self"
             and parameter.kind is not parameter.VAR_KEYWORD]

    assert names == expected
    assert not any(p.kind is p.VAR_KEYWORD
                   for p in signature.parameters.values()), (
        f"{cls}.{method} still takes **kwargs; an unknown argument there is "
        f"silently ignored")


@pytest.mark.parametrize("module,cls,method",
                         [(m, c, f) for m, c, f, _ in EXPLICIT],
                         ids=[f"{c}.{m}" for _, c, m, _ in EXPLICIT])
def test_an_unknown_stage_argument_raises(module, cls, method):
    """The behaviour the signature change buys, checked rather than assumed."""
    owner = getattr(importlib.import_module(f"experiments.qsim.{module}"), cls)
    bound = getattr(owner, method)
    # bind_partial, so a required argument left out does not mask the point:
    # what must fail is the unknown name, not the missing one.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        inspect.signature(bound).bind_partial(definitely_not_a_knob=1)


# --------------------------------------------------------------------------
# Ported, not split
# --------------------------------------------------------------------------
# A method that was never on *this* branch's god class, so `STAGES` above
# cannot pin it: its pin is a commit from the other side of a merge.
#
# The case that needed this: jonginn added `analyze_propagator_dynamics` on
# 2026-08-28, after the propagator stage had already been split out here. The
# `jonginn` -> `guan` merge (6ca1a6e) resolved `floquet_dark_mode_readout.py`
# by taking guan's 3,925-line refactored side over jonginn's 8,772-line side,
# and the addition went out with it -- no conflict, no commit that removes it,
# nothing in the file's simplified history that ever had it. `git log -S` on
# the path finds only the commit that *added* it.
#
# 22a1e7c is that merge's other parent: the last commit reachable from `guan`
# where the method still existed, and its copy of the file is byte-identical
# to main's.

PORTED = [
    dict(method="analyze_propagator_dynamics",
         pin="22a1e7c",
         module="mbr_propagator",
         cls="MBRPropagatorExperiment",
         edits=[
             # `_calibration_data` moved to the calibration stage, and
             # resolving a calibration given as paths needs a station, so the
             # method takes one. Same re-addressing `mbr_spectrum.py` does.
             ("cls._calibration_data(calibration)",
              "MBRPhaseCorrectionExperiment._calibration_data(calibration, station)"),
             ("eigenphase_cycle=None):", "eigenphase_cycle=None, station=None):"),
         ]),
]


@pytest.mark.parametrize("spec", PORTED, ids=[s["method"] for s in PORTED])
def test_ported_method_is_unchanged(spec):
    """A port is a move too, so it gets the same verbatim check.

    Without this the 219 lines could have been paraphrased on the way in and
    nothing would have noticed.
    """
    root = Path(_repo_root())
    shown = subprocess.run(
        ["git", "-C", str(root), "show", f"{spec['pin']}:{GOD}"],
        capture_output=True, text=True, encoding="utf-8")
    assert shown.returncode == 0, (
        f"cannot read {GOD} at {spec['pin']}: {shown.stderr}")

    old = _methods(shown.stdout, GODCLASS)
    assert spec["method"] in old, (
        f"{spec['method']} was not on {GODCLASS} at {spec['pin']}")
    new = _methods(
        (root / "experiments" / "qsim" / f"{spec['module']}.py").read_text(
            encoding="utf-8"),
        spec["cls"])
    assert spec["method"] in new, f"{spec['method']} is missing from {spec['cls']}"

    was = ast.unparse(old[spec["method"]])
    for target, replacement in spec["edits"]:
        assert target in was, (
            f"declared edit {target!r} matches nothing at {spec['pin']}; "
            f"delete the row instead of leaving it")
        was = was.replace(target, replacement)
    assert ast.unparse(new[spec["method"]]) == was


@pytest.mark.parametrize("spec", PORTED, ids=[s["method"] for s in PORTED])
def test_ported_method_is_not_also_on_the_god_class(spec):
    """Defined in two places, the subclass silently shadows -- and drifts."""
    root = Path(_repo_root())
    god = _methods((root / GOD).read_text(encoding="utf-8"), GODCLASS)
    assert spec["method"] not in god
