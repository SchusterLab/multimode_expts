"""Smoke tests for the stage-2 qsim notebook helper modules.

The stage-2 split moved ~23k lines of notebook code into
`experiments/qsim/notebook_helpers/`. These tests are the cheap, always-true
checks that the move did not break: every module imports, every generated
notebook parses and round-trips through jupytext, and the two HDF5 loading
paths the split found running in parallel agree on a real file.

They deliberately do not test physics. Verifying numerical agreement against
the original notebooks is the later, theme-specific work the stage-2
instructions defer.
"""
import ast
import importlib
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
HELPERS = REPO / "experiments" / "qsim" / "notebook_helpers"
NOTEBOOK_DIRS = [
    REPO / "measurement_notebooks" / "202609_qsim_migration",
    REPO / "analysis_notebooks" / "202609_qsim_migration",
]

# Cell 57 of the source data_postprocess.ipynb lost the indentation of a whole
# function body before this refactor ever ran, and the dormant relocation
# preserves it byte-for-byte rather than guessing where the dedent should
# stop. So this one file is expected not to parse.
KNOWN_UNPARSEABLE = {"flux_excursion.py"}


def helper_modules():
    return sorted(
        p.stem for p in HELPERS.glob("*.py") if not p.name.startswith("_")
    )


def notebook_files():
    out = []
    for directory in NOTEBOOK_DIRS:
        out.extend(sorted(directory.rglob("*.py")))
    return out


@pytest.mark.parametrize("name", helper_modules())
def test_helper_module_imports(name):
    importlib.import_module(f"experiments.qsim.notebook_helpers.{name}")


@pytest.mark.parametrize(
    "path", notebook_files(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_notebook_parses(path):
    if path.name in KNOWN_UNPARSEABLE:
        pytest.xfail(
            "cell 57 of the source notebook lost a function body's "
            "indentation; preserved as found"
        )
    ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "path", notebook_files(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_notebook_round_trips_through_jupytext(path):
    jupytext = pytest.importorskip("jupytext")
    notebook = jupytext.read(path)
    assert notebook.cells, f"{path.name} produced no cells"


def test_no_active_notebook_reaches_for_globals():
    """The stage-2 instructions forbid passing globals() or using exec/%run.

    Dormant files are excluded: they are byte-faithful relocations, and the
    source did whatever it did.
    """
    offenders = []
    for path in notebook_files():
        if "dormant" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in ("globals()", "exec(", "%run"):
            if pattern in text:
                offenders.append(f"{path.name}: {pattern}")
    assert not offenders, offenders


# --------------------------------------------------------------------------
# The parallel HDF5 loaders.
# --------------------------------------------------------------------------

DATA_DIR = pathlib.Path(r"C:\experiments\260818_qsim_spectroscopy\data")


def one_encspec_file():
    if not DATA_DIR.is_dir():
        pytest.skip(f"{DATA_DIR} is not present on this machine")
    files = sorted(
        DATA_DIR.glob("*_EncodingHamiltonianSpectroscopyExperiment.h5")
    )
    if not files:
        pytest.skip("no EncodingHamiltonianSpectroscopy HDF5 files found")
    return files[0]


def test_library_loader_reads_a_saved_encspec_file():
    """`from_h5file` is the canonical path the repo documents."""
    from experiments.qsim.floquet_dark_mode_readout import (
        EncodingHamiltonianSpectroscopyExperiment,
    )

    path = one_encspec_file()
    expt = EncodingHamiltonianSpectroscopyExperiment.from_h5file(str(path))
    assert expt is not None
    assert hasattr(expt, "cfg")


def test_notebook_local_loader_agrees_with_the_library_on_the_config():
    """The split found a second HDF5 loader living in the notebook.

    `mbr_disorder_h5` carries a 248-line notebook-local reimplementation
    (source cell 261) that runs parallel to `from_h5file`. This pass
    deliberately did not reconcile them, so this test pins down what they
    currently agree on: the embedded experiment config. If someone later
    deletes one loader, this is the check that says whether they were
    interchangeable for that much.
    """
    from experiments.qsim.floquet_dark_mode_readout import (
        EncodingHamiltonianSpectroscopyExperiment,
    )
    from experiments.qsim.notebook_helpers import mbr_disorder_h5

    path = one_encspec_file()

    library = EncodingHamiltonianSpectroscopyExperiment.from_h5file(str(path))
    header = mbr_disorder_h5.read_h5_header(str(path))

    assert header, "the notebook-local loader returned an empty header"

    library_cfg = getattr(library, "cfg", None)
    assert library_cfg is not None
    library_expt_cfg = dict(library_cfg.get("expt", {}))

    # read_h5_header returns a SimpleNamespace with the config under `cfg`,
    # not a mapping -- one of the small ways the two loaders differ in shape
    # while reading the same bytes.
    header_cfg = getattr(header, "cfg", None)
    assert header_cfg is not None, "notebook-local header exposed no cfg"
    header_expt_cfg = (
        header_cfg.get("expt") if hasattr(header_cfg, "get")
        else getattr(header_cfg, "expt", None)
    )
    if header_expt_cfg is None:
        pytest.skip(
            "the notebook-local header does not expose an expt config for "
            "this file; nothing to compare"
        )
    header_expt_cfg = dict(header_expt_cfg)

    shared = set(library_expt_cfg) & set(header_expt_cfg)
    assert shared, (
        "the two loaders share no expt config keys, which would mean they "
        "are not reading the same thing at all"
    )

    # Where they overlap, they should agree. Compare only scalars: the array
    # valued entries differ in container type between the two paths, which is
    # a shape difference rather than a disagreement about the data.
    mismatches = []
    for key in sorted(shared):
        left, right = library_expt_cfg[key], header_expt_cfg[key]
        if isinstance(left, (str, int, float, bool)) and isinstance(
            right, (str, int, float, bool)
        ):
            if left != right:
                mismatches.append(f"{key}: library={left!r} notebook={right!r}")
    assert not mismatches, (
        "the two HDF5 loaders disagree on scalar config values:\n"
        + "\n".join(mismatches)
    )
