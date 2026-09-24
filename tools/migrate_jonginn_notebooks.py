# -*- coding: utf-8 -*-
"""Re-point jonginn's two notebooks at the split MBR stage classes.

Run:  pixi run python tools/migrate_jonginn_notebooks.py [--check]

Why a script and not hand edits
-------------------------------
`qsim_experiments.ipynb` and `data_postprocess.ipynb` are 374 and 306 cells.
The edits are repetitive and every one of them is load-bearing for a
measurement, so each is declared as an exact old/new string with the number of
occurrences it must match. A miscount is an error, not a silent no-op -- which
is the failure mode of doing this with a regex and an eyeball.

`--check` reports whether the notebooks already match the post-edit state
without writing, so this is safe to re-run and can be asserted in a test.

What is migrated
----------------
`EncodingHamiltonianSpectroscopyExperiment.analyze(stage=...)` is gone: four
unrelated aggregate analyses became four Experiment classes. The loading layer
(`from_job_ids`, `from_job_files`, `_from_expts`, `_saved_parameters`,
`hardware_parameters`) and the shared numerics (`analyze_spectrum`,
`merge_spectra`, `analyze_matrix_pencil*`, `build_phase_correction`) stayed on
the god class and are inherited by all four stages, so a variable bound to a
stage class can still do all of it.

What is deliberately NOT migrated
---------------------------------
`ExptClass=EncSpec` in every `BatchRunner`, and the `f"{job_id}_{EncSpec.__name__}.h5"`
filename construction. Both are provenance: the queue records the class each
job was submitted under and the HDF5 file is named after it, so jonginn's
existing data is named `..._EncodingHamiltonianSpectroscopyExperiment.h5`.
Changing the acquisition class would orphan it. `from_batch` exists precisely
so acquisition can keep recording the old class while analysis uses the new
one.

Also not migrated: the `check_program_class(expt, "KerrCavityRamseyExcursionExperiment")`
string literals. Those name the class historical data recorded, not a class to
import.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "measurement_notebooks" / "jonginn"

# The import block every migrated cell needs. Inserted once per notebook, into
# the cell that already binds the god class, so the notebook stays runnable
# top-to-bottom and out of order.
STAGE_IMPORTS = """# The four aggregate MBR stages. `EncodingHamiltonianSpectroscopyExperiment`
# is still the loading layer and the shared numerics, and is still the class
# every job here was acquired under -- so it stays, and these four sit beside
# it. See analysis_notebooks/guan/MBR_analysis.py for the worked example.
from experiments.qsim.floquet_dark_mode_readout import (
    EncodingHamiltonianSpectroscopyExperiment,
)
from experiments.qsim.legacy_mbr import MBRPhaseCorrectionExperiment
from experiments.qsim.legacy_mbr import MBRSpectrumExperiment
from experiments.qsim.legacy_mbr import MBROrthogonalityExperiment
from experiments.qsim.legacy_mbr import MBRPropagatorExperiment"""


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path, notebook):
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _code_cells(notebook):
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _source(cell):
    return "".join(cell["source"])


def _set_source(cell, text):
    """Keep the one-string-per-line shape nbformat writes."""
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


class Edit:
    """One exact replacement, with the occurrence count it must match."""

    def __init__(self, cell, old, new, count=1, note=""):
        self.cell, self.old, self.new, self.count, self.note = (
            cell, old, new, count, note)

    def apply(self, text):
        found = text.count(self.old)
        if found != self.count:
            raise AssertionError(
                f"cell {self.cell}: expected {self.count} occurrence(s) of\n"
                f"  {self.old!r}\nbut found {found}")
        return text.replace(self.old, self.new)

    def already_applied(self, text):
        return text.count(self.old) == 0 and self.new in text


# ---------------------------------------------------------------------------
# Rule 1: the `stage=` argument goes away
# ---------------------------------------------------------------------------
# Matches the argument wherever it sits -- inline, on its own line in a
# multi-line call, or as a key in an `analyze_kwargs` dict -- and takes the
# preceding newline plus indent with it so no blank line is left behind.
STAGE_ARGUMENT = re.compile(
    r"\n?[ \t]*stage[ \t]*=[ \t]*['\"](?:calibration|spectrum|orthogonality|propagator)['\"][ \t]*,?")

# Rule 2: an alias bound to the god class, rebound to the stage that owns the
# analysis it is used for. Checked per notebook against what the alias is
# actually used for -- see the module docstring's note on the loading layer
# and the shared numerics being inherited.
GOD = "EncodingHamiltonianSpectroscopyExperiment"

# Rule 3: class-level methods that left the god class, and who owns them now.
MOVED_METHODS = {
    "calibration_batch": "MBRPhaseCorrectionExperiment",
    "analyze_calibration": "MBRPhaseCorrectionExperiment",
    "phase_correction_from_calibration": "MBRPhaseCorrectionExperiment",
    "display_cycle_phase": "MBRPhaseCorrectionExperiment",
    "display_calibration_results": "MBRPhaseCorrectionExperiment",
    "display_calibration_summary": "MBRPhaseCorrectionExperiment",
    "_calibration_data": "MBRPhaseCorrectionExperiment",
    "spectroscopy_batch": "MBRSpectrumExperiment",
    "subsample_spectroscopy_shots": "MBRSpectrumExperiment",
    "reconstruct_spectroscopy": "MBRSpectrumExperiment",
    "reconstruct_pair_spectroscopy": "MBRSpectrumExperiment",
    "display_occupation": "MBRSpectrumExperiment",
    "display_occupations": "MBRSpectrumExperiment",
    "display_local_density_of_states": "MBRSpectrumExperiment",
    "display_result": "MBRSpectrumExperiment",
    "display_matrix_pencil": "MBRSpectrumExperiment",
    "display_matrix_pencil_occupation": "MBRSpectrumExperiment",
    "display_level_statistics": "MBRSpectrumExperiment",
    "orthogonality_batch": "MBROrthogonalityExperiment",
    "display_orthogonality": "MBROrthogonalityExperiment",
    "propagator_batch": "MBRPropagatorExperiment",
}


# ---------------------------------------------------------------------------
# Per-notebook plans
# ---------------------------------------------------------------------------
# `aliases`  -- name -> stage class to rebind it to. Only names whose whole
#               class-level use is the loading layer, the shared numerics, or
#               methods owned by that one stage. An alias used for two stages'
#               batch builders cannot be rebound and is handled by rule 3.
# `edits`    -- exact one-off replacements, each with its occurrence count.
# `expect`   -- counts that must hold before the run, so a notebook that has
#               drifted fails loudly instead of being half-migrated.

POSTPROCESS = dict(
    notebook="data_postprocess.ipynb",
    import_into=1,            # its real import cell; cell 2 already needs these
    aliases={
        # None of these five is used for a calibration, orthogonality or
        # propagator classmethod -- only loading, shared numerics, and three
        # spectrum-owned display/subsample helpers. So the spectrum stage is
        # the honest owner and rebinding fixes the classmethods and the
        # `.analyze()`/`.display()` on everything built from them at once.
        "EncSpec": "MBRSpectrumExperiment",
        "ReplotEncSpec": "MBRSpectrumExperiment",
        "SavedEncSpec": "MBRSpectrumExperiment",
        "DiagPreviewEncSpec": "MBRSpectrumExperiment",
        "DiagStatsEncSpec": "MBRSpectrumExperiment",
    },
    # Receivers that come back from a notebook-local loader rather than a
    # constructor, so no tracer follows them. Each names the function and the
    # expression inside it that does build with a stage class -- checked
    # below, so the exemption is verified rather than asserted. Both of these
    # loaders take the class as an argument or close over a rebound alias, so
    # they are already spectrum-side.
    returned_by={
        "spectroscopy_expt": ("load_encoding_spectroscopy",
                              "spectroscopy_expt = EncSpec._from_expts("),
        "mpm_spectroscopy_expt": ("load_encoding_spectroscopy",
                                  "spectroscopy_expt = EncSpec._from_expts("),
        "replot_N2_supp_expt": ("replot_load_sector",
                                "spectroscopy_expt = ReplotEncSpec._from_expts("),
        "diag_preview_expt": ("(constructed directly)",
                              "diag_preview_expt = DiagPreviewEncSpec.from_job_ids("),
    },
    extra_stages={},
    edits=[
        # `isinstance(expt, EncSpec)` asks "is this already a loaded
        # aggregate". Rebinding EncSpec to the spectrum stage would narrow it:
        # the saved children were acquired under the god class and are not
        # instances of any stage subclass, so the check would start returning
        # False and silently drop every job it used to accept. This is the one
        # place rebinding is not safe on its own.
        Edit(2,
             "isinstance(expt, EncSpec)",
             f"isinstance(expt, {GOD})",
             note="keep the aggregate check against the acquired class"),
        # `SavedSpectroscopyExperiment` exists to override
        # `_postprocess_reconstruction` and `_saved_parameters` for datasets
        # saved before the decoder/physical-clock change. Both overrides only
        # mean anything on the spectrum path, and the method it overrides now
        # lives on `MBRSpectrumExperiment` -- so the base moves and all 28 of
        # its call sites keep working untouched.
        Edit(202,
             f"class SavedSpectroscopyExperiment({GOD}):",
             "class SavedSpectroscopyExperiment(MBRSpectrumExperiment):",
             note="re-parent the saved-data subclass onto the spectrum stage"),
        # Its one calibration call. `analyze_calibration` went to the
        # calibration stage, and is not inherited from the spectrum one.
        Edit(204,
             "SavedSpectroscopyExperiment.analyze_calibration(children)",
             "MBRPhaseCorrectionExperiment.analyze_calibration(children)",
             note="analyze_calibration is the calibration stage's"),
        # The aggregate that receives it, for the same reason.
        Edit(204,
             "phase_calibration = SavedSpectroscopyExperiment._from_expts(children, job_ids=loaded_calibration_ids)",
             "phase_calibration = MBRPhaseCorrectionExperiment._from_expts(children, job_ids=loaded_calibration_ids)",
             note="the calibration aggregate is not a spectrum object"),
        # `load_dark_experiments` derives the HDF5 filename tag from
        # `ExpClass.__name__` -- its own docstring says so -- and the files on
        # disk are `JOB-..._EncodingHamiltonianSpectroscopyExperiment.h5`.
        # Handing it a rebound alias would send it looking for
        # `..._MBRSpectrumExperiment.h5` and it would find nothing. Name the
        # acquired class outright, so the tag no longer rides on an alias.
        Edit(180,
             "ExpClass=SavedEncSpec,",
             f"ExpClass={GOD},  # names the saved files, not the analysis",
             note="filename tag must name the acquired class"),
    ],
    expect={},
)

EXPERIMENTS = dict(
    notebook="qsim_experiments.ipynb",
    import_into=1,            # its import cell; first migrated cell is 225
    aliases={
        # Neither alias is rebound here, and the reason is the same for both:
        # each is also used to derive an HDF5 filename from `__name__`, and
        # those files are on disk named after the class the jobs were acquired
        # under. `EncSpec` additionally builds all four stages' batches and is
        # what every BatchRunner records. Rule 3 re-addresses their moved
        # classmethods one call at a time, which leaves `__name__` alone.
    },
    edits=[],
    # Aggregates whose `analyze(stage=...)` is in a *later* cell, or is reached
    # through a record, so rule 4 cannot see the assignment and the call
    # together. Declared as {cell: {variable: stage}} and merged into what
    # rule 4 found, so the rewrite still goes through the AST and gets the
    # parentheses right -- `execute(...)` here is variously with and without a
    # trailing comma, which string surgery would get wrong.
    #
    # `--check` reports any receiver that used to name a stage and is no
    # longer traceable to one, which is how these three were found.
    returned_by={},
    extra_stages={
        233: {"orthogonality_expt": "orthogonality"},   # analysed in cell 234
        248: {"diag_expt": "spectrum"},                 # via a dict record, cell 252
        258: {"d72_expt": "spectrum"},                  # via an AttrDict record, cell 259
    },
    expect={},
)

# `qsim_experiments_highkerr_untracked_refactored.ipynb` is a near-duplicate of
# `qsim_experiments.ipynb` -- 223 of its 263 non-comment code cells are
# byte-identical -- so it needs the same plan with the cell numbers shifted.
# Unlike its twin it carries 179 stored figures, which is why it is migrated
# rather than left behind or converted.
HIGHKERR = dict(
    EXPERIMENTS,
    notebook="qsim_experiments_highkerr_untracked_refactored.ipynb",
    extra_stages={
        231: {"orthogonality_expt": "orthogonality"},   # analysed in cell 232
        247: {"diag_expt": "spectrum"},                 # via a dict record
        257: {"d72_expt": "spectrum"},                  # via an AttrDict record
    },
    # Its `D72EncSpec.__name__` filename is in cell 255, not 256, but no edit
    # is needed: neither alias is rebound here either, so `__name__` is left
    # alone by construction.
    edits=[],
)

PLANS = [POSTPROCESS, EXPERIMENTS, HIGHKERR]


# Classes the notebook defines on top of the god class. Their constructions
# must keep naming the subclass; what changes is what the subclass inherits.
LOCAL_SUBCLASSES = {"SavedSpectroscopyExperiment"}

def _binding_patterns(alias):
    """The two shapes `<alias> = <god class>` is written in, and no others.

    Anchored at `^` so rebinding `EncSpec` cannot touch the line that binds
    `SavedEncSpec`, and the unparenthesised form is anchored at end-of-line so
    the match cannot swallow the newline and glue the next statement on.
    """
    name = re.escape(alias)
    god = re.escape(GOD)
    return [
        # SavedEncSpec = (\n    floquet_dark_mode_readout.EncSpec...\n)
        re.compile(rf"^{name}[ \t]*=[ \t]*\([\s]*floquet_dark_mode_readout"
                   rf"[\s]*\.[\s]*{god}[\s]*\)", re.M),
        # EncSpec = floquet_dark_mode_readout.EncSpec...
        re.compile(rf"^{name}[ \t]*=[ \t]*floquet_dark_mode_readout"
                   rf"[ \t]*\.[ \t]*{god}(?=[ \t]*$)", re.M),
    ]


STAGE_OWNER = {
    "calibration": "MBRPhaseCorrectionExperiment",
    "spectrum": "MBRSpectrumExperiment",
    "orthogonality": "MBROrthogonalityExperiment",
    "propagator": "MBRPropagatorExperiment",
}

# Constructors that return an aggregate of whatever class they are called on.
LOADERS = ("from_job_files", "from_job_ids", "_from_expts")


def analyzed_as(source):
    """-> {receiver expression: stage} for every `X.analyze(stage=...)` here.

    The receiver is kept as source text (`record["expt"]`, `record.expt`,
    `spectroscopy_expt`), because that is what has to be matched against an
    assignment target.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    found = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("analyze", "display")):
            continue
        for keyword in node.keywords:
            if keyword.arg == "stage" and isinstance(keyword.value, ast.Constant):
                found[ast.unparse(node.func.value)] = keyword.value.value
    return found


def rewrite_constructions(source, stage_by_receiver):
    """Rule 4/5: build each analysed aggregate with the class that analyses it.

    Two shapes, because there are two ways an aggregate arrives:

      var = <Alias>.from_job_ids(...)      -> var = <Stage>.from_job_ids(...)
      var = <runner>.execute(...)          -> var = <Stage>.from_batch(<runner>.execute(...))

    The second is why `from_batch` exists: `BatchRunner` returns an instance of
    its `ExptClass`, which stays the acquired class for provenance, so the
    aggregate has to be re-wrapped rather than built differently.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    lines = source.splitlines(keepends=True)
    patches, notes = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = ast.unparse(node.targets[0])
        stage = stage_by_receiver.get(target)
        if stage is None or not isinstance(node.value, ast.Call):
            continue
        owner = STAGE_OWNER[stage]
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            continue
        start = _offset(lines, node.value.lineno, node.value.col_offset)
        end = _offset(lines, node.value.end_lineno, node.value.end_col_offset)
        text = source[start:end]
        if call.func.attr in LOADERS:
            receiver = ast.unparse(call.func.value)
            if receiver == owner:
                continue                      # already migrated
            if receiver in LOCAL_SUBCLASSES:
                # `SavedSpectroscopyExperiment` overrides `_saved_parameters`
                # and `_postprocess_reconstruction`; re-addressing the call to
                # its new base would silently drop both, which is the entire
                # reason the subclass exists. Its *base* is repointed instead.
                continue
            patches.append((start, end, f"{owner}.{call.func.attr}"
                            + text[len(receiver) + 1 + len(call.func.attr):]))
            notes.append(f"{target}: {receiver}.{call.func.attr} -> {owner}")
        elif call.func.attr == "execute":
            patches.append((start, end, f"{owner}.from_batch({text})"))
            notes.append(f"{target}: wrapped runner aggregate in "
                          f"{owner}.from_batch")
    for start, end, replacement in sorted(patches, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source, notes


def _offset(lines, lineno, col):
    return sum(len(line) for line in lines[:lineno - 1]) + col


def rewrite_moved_methods(source, aliases_left_on_god):
    """Rule 3: `<Alias>.<moved method>` -> `<NewOwner>.<moved method>`.

    Only for aliases still bound to the god class. An alias that was rebound
    to a stage already resolves its own stage's methods, and rewriting those
    would replace a correct name with a different correct name for no reason.
    """
    notes = []
    # Longest alias first, and \b on both ends: `EncSpec` is a substring of
    # `D72EncSpec` and `SavedEncSpec`, so a plain `str.replace` would rewrite
    # the middle of a different name.
    for alias in sorted(aliases_left_on_god, key=len, reverse=True):
        for method, owner in MOVED_METHODS.items():
            pattern = re.compile(rf"(?<![\w.]){re.escape(alias)}\.{re.escape(method)}\b")
            count = len(pattern.findall(source))
            if not count:
                continue
            source = pattern.sub(f"{owner}.{method}", source)
            notes.append(f"{alias}.{method} -> {owner}.{method} (x{count})")
    return source, notes


def migrate(plan, write):
    path = NOTEBOOKS / plan["notebook"]
    notebook = _load(path)
    cells = _code_cells(notebook)
    report = []

    # Everything the notebook now names has to be importable in it.
    target = cells[plan["import_into"] - 1]
    text = _source(target)
    if "from experiments.qsim.legacy_mbr import" not in text:
        assert re.search(r"^\s*(import|from)\s", text, re.M), (
            f"cell {plan['import_into']} has no imports; the stage imports "
            f"belong in an import cell, and it must run before the first "
            f"migrated cell")
        text = text.rstrip("\n") + "\n\n" + STAGE_IMPORTS + "\n"
        _set_source(target, text)
        report.append(f"cell {plan['import_into']}: stage imports added")

    aliases_on_god = set()
    for index, cell in enumerate(cells, 1):
        source = _source(cell)
        before = source

        # Rule 2, first: later rules ask which aliases are still on the god
        # class, and rebinding is what answers that.
        for alias, owner in plan["aliases"].items():
            for binding in _binding_patterns(alias):
                if binding.search(source):
                    source = binding.sub(f"{alias} = {owner}", source)
                    report.append(f"cell {index}: {alias} -> {owner}")
                    break
        for alias in set(re.findall(r"^(\w*EncSpec)\s*=", source, re.M)):
            if alias not in plan["aliases"]:
                aliases_on_god.add(alias)

        for edit in plan["edits"]:
            if edit.cell != index:
                continue
            if edit.already_applied(source):
                continue
            source = edit.apply(source)
            report.append(f"cell {index}: {edit.note or 'exact edit'}")

        source, notes = rewrite_moved_methods(source, aliases_on_god)
        report += [f"cell {index}: {n}" for n in notes]

        stage_by_receiver = dict(analyzed_as(source))
        stage_by_receiver.update(plan.get("extra_stages", {}).get(index, {}))
        source, notes = rewrite_constructions(source, stage_by_receiver)
        report += [f"cell {index}: {n}" for n in notes]

        stages = STAGE_ARGUMENT.findall(source)
        stripped = STAGE_ARGUMENT.sub("", source)
        if stripped != source:
            n = len(STAGE_ARGUMENT.findall(source))
            source = stripped
            report.append(f"cell {index}: dropped {n} stage= argument(s)")

        if source != before:
            _set_source(cell, source)

    # A cell that parsed before must still parse. Magics make some cells
    # unparseable to begin with; those are excluded rather than excused.
    broken = []
    for index, cell in enumerate(_code_cells(_load(path)), 1):
        pass
    for index, cell in enumerate(cells, 1):
        try:
            ast.parse(_source(cell))
        except SyntaxError as error:
            broken.append((index, error))
    if write:
        _save(path, notebook)
    return report, broken, notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report without writing")
    args = parser.parse_args()

    failed = False
    for plan in PLANS:
        path = NOTEBOOKS / plan["notebook"]
        if not path.exists():
            # The stage-2 notebook split retired qsim_experiments.ipynb and
            # data_postprocess.ipynb after distributing their cells into the
            # themed entry points under */202609_qsim_migration/. A plan with
            # no notebook left is done, not broken.
            print(f"\n=== {plan['notebook']} ===")
            print("  retired by the stage-2 split; nothing to migrate")
            continue
        parsed_before = _parseable(_load(path))
        report, broken, notebook = migrate(plan, write=not args.check)
        parsed_after = _parseable(notebook)
        print(f"\n=== {plan['notebook']} ===")
        for line in report:
            print("  " + line)
        if not report:
            print("  nothing to do")
        newly_broken = parsed_before - parsed_after
        if newly_broken:
            failed = True
            print(f"  *** {len(newly_broken)} cell(s) stopped parsing: "
                  f"{sorted(newly_broken)}")
        leftover = _remaining_stage_calls(notebook)
        if leftover:
            failed = True
            print(f"  *** stage= still present in cells {leftover}")
        leaked = rebound_alias_in_provenance(notebook, plan["aliases"])
        if leaked:
            failed = True
            print("  *** a rebound alias is used to build a provenance "
                  "string; it must name the acquired class:")
            for index, alias, line in leaked:
                print(f"        cell {index}: {alias} in  {line}")
        untraced = _untraced_stage_receivers(_load(path), notebook,
                                             aliases=plan["aliases"],
                                             returned_by=plan.get("returned_by"))
        if untraced:
            failed = True
            print("  *** these were analysed with stage= and are not built by "
                  "a stage class; add them to the plan's extra_stages:")
            for receiver, stage in sorted(untraced.items()):
                print(f"        {receiver}  (was stage={stage!r})")
    return 1 if failed else 0


def _parseable(notebook):
    good = set()
    for index, cell in enumerate(_code_cells(notebook), 1):
        try:
            ast.parse(_source(cell))
            good.add(index)
        except SyntaxError:
            pass
    return good


def _remaining_stage_calls(notebook):
    return [index for index, cell in enumerate(_code_cells(notebook), 1)
            if STAGE_ARGUMENT.search(_source(cell))]


# An alias that was rebound must not be the thing a provenance string is
# derived from. Both of these read a class name that has to match what is on
# disk or in the queue, not what the analysis uses:
#   f"{job_id}_{Alias.__name__}.h5"      -- the saved HDF5 filename
#   ExptClass=Alias / ExpClass=Alias     -- what the queue records, and what
#                                           load_dark_experiments tags files with
PROVENANCE_USE = (
    re.compile(r"\b(\w+)\.__name__"),
    re.compile(r"\bExpt?Class\s*=\s*(\w+)"),
)


def rebound_alias_in_provenance(notebook, aliases):
    """-> [(cell, alias, line)] where a rebound alias feeds a provenance string.

    This is the bug that does not raise: the notebook runs, the loader looks
    for `JOB-..._MBRSpectrumExperiment.h5`, finds nothing, and reports missing
    data for jobs that are sitting right there.
    """
    found = []
    for index, cell in enumerate(_code_cells(notebook), 1):
        for line in _source(cell).splitlines():
            for pattern in PROVENANCE_USE:
                for name in pattern.findall(line):
                    if name in aliases:
                        found.append((index, name, line.strip()))
    return found


def stage_receivers(notebook):
    """-> {receiver: stage} for every `X.analyze(stage=...)` in the notebook.

    Run on the *pre*-migration notebook, this is the list of variables that
    have to end up being built by a stage class.
    """
    found = {}
    for cell in _code_cells(notebook):
        found.update(analyzed_as(_source(cell)))
    return found


def traced_to_a_stage(notebook, aliases=()):
    """-> {variable: stage class} for aggregates built by a stage class.

    Follows one level of indirection, because a campaign loop keeps its
    aggregate in a record: `record["expt"] = diag_expt` where `diag_expt`
    came from `MBRSpectrumExperiment.from_batch(...)`.
    """
    # A rebound alias resolves to its stage class: rule 2 changes the binding
    # line, not the call sites, so `DiagPreviewEncSpec.from_job_ids(...)` is
    # still spelled with the alias and is still a spectrum aggregate.
    owners = set(STAGE_OWNER.values()) | set(aliases)
    direct, indirect = {}, {}
    for cell in _code_cells(notebook):
        try:
            tree = ast.parse(_source(cell))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = ast.unparse(node.targets[0])
            value = node.value
            if (isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id in owners):
                direct[target] = value.func.value.id
            elif isinstance(value, ast.Name):
                indirect[target] = value.id
        # dict(expt=diag_expt) / AttrDict(dict(expt=d72_expt))
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and isinstance(node.value, ast.Name):
                indirect.setdefault(f"record[{node.arg!r}]", node.value.id)
                indirect.setdefault(f"record.{node.arg}", node.value.id)
    for target, source_name in indirect.items():
        if source_name in direct:
            direct.setdefault(target, direct[source_name])
    return direct


def _untraced_stage_receivers(before, after, aliases=(), returned_by=None):
    """Receivers that used to name a stage and are no longer built by one.

    This is the check that finds the case rule 4 structurally cannot: an
    `analyze(stage=...)` whose aggregate is assigned in a different cell, or
    reached through a record. Without it, stripping `stage=` from such a call
    leaves it running against the god class, which raises -- but only when
    that cell is next run.
    """
    traced = traced_to_a_stage(after, aliases)
    text = "\n".join(_source(cell) for cell in _code_cells(after))
    exempt = {}
    for receiver, (function, builder) in (returned_by or {}).items():
        assert builder in text, (
            f"{receiver} is declared as built by {function} via {builder!r}, "
            f"but that expression is not in the notebook any more -- "
            f"re-check the exemption instead of keeping it")
        exempt[receiver] = function
    return {receiver: stage
            for receiver, stage in stage_receivers(before).items()
            if receiver not in traced and receiver not in exempt}


if __name__ == "__main__":
    sys.exit(main())
