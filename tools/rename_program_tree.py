# -*- coding: utf-8 -*-
"""Rename the qsim Program/Experiment classes of step 10F in code and notebooks.

``docs/qsim/program_tree_plan.md``, section 5. Only identifiers change, not
comments or docstrings (in ``.py`` files): notes about the history keep the old
names. Imports from ``experiments.qsim.dark_base``, which step 10F deletes,
are rewritten to the modules the names live in now.

In ``.ipynb`` files the code cells change as whole words (they may hold IPython
magics that the tokenizer rejects), comments included.

    pixi run python tools/rename_program_tree.py PATH ...          # report
    pixi run python tools/rename_program_tree.py --write PATH ...  # change the files

A reference the tool cannot rewrite (an attribute path through the deleted
module, such as ``meas.qsim.dark_base.X``) is reported, not changed.
"""
import argparse
import io
import json
import re
import sys
import tokenize
from pathlib import Path

RENAMES = {
    "QsimBaseProgram": "QsimProgram",
    "DarkBaseProgram": "DarkModeProgram",      # it was DarkModeProgram with an old name
    "DarkBaseRProgram": "QsimRProgram",
    "QsimBaseExperiment": "QsimExperiment",
    "DarkBaseExperiment": "QsimExperiment",    # it was the same driver with an old name
    "QsimWignerBaseExperiment": "QsimWignerExperiment",
    "WignerExperiment": "QsimWignerExperiment",    # its name for part of 2026-09-29
    "SidebandScrambleDarkProgramNewNew": "DarkModeScrambleProgram",
    "SidebandStarkAmplificationModifiedProgram": "StorageSwapStarkPhaseProgram",
}

# Where each name that experiments.qsim.dark_base used to give lives now.
MODULE_OF = {
    "QsimProgram": "experiments.qsim.qsim_base",
    "QsimRProgram": "experiments.qsim.qsim_base",
    "QsimExperiment": "experiments.qsim.qsim_base",
    "classify_two_parity_readouts": "experiments.qsim.qsim_base",
    "readout_lane_count": "experiments.qsim.qsim_base",
    "readout_mode": "experiments.qsim.qsim_base",
    "DarkModeProgram": "experiments.qsim.dark_mode_encoding",
}

DELETED_MODULE = "experiments.qsim.dark_base"
_IMPORT = re.compile(
    r"^(?P<indent>[ \t]*)from experiments\.qsim\.dark_base import "
    r"(?:\((?P<multi>[^)]*)\)|(?P<single>[^\n]*))", re.M)
_WORD = re.compile(r"\b(" + "|".join(sorted(RENAMES, key=len, reverse=True)) + r")\b")


def rename_python(text):
    """-> text with the NAME tokens renamed (comments and strings untouched)."""
    edits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NAME and tok.string in RENAMES:
            edits.append((tok.start, tok.end, RENAMES[tok.string]))
    lines = text.splitlines(keepends=True)
    for (row, col), (_, end_col), new in reversed(edits):
        line = lines[row - 1]
        lines[row - 1] = line[:col] + new + line[end_col:]
    return "".join(lines), len(edits)


def rewrite_dark_base_imports(text):
    """-> text with ``from experiments.qsim.dark_base import ...`` pointing at the new modules."""
    def replace(match):
        body = match.group("multi") if match.group("multi") is not None else match.group("single")
        names = []
        for part in body.replace("\n", ",").split(","):
            part = part.split("#")[0].strip()
            if part and part not in names:
                names.append(part)
        by_module = {}
        for name in names:
            base = name.split(" as ")[0].strip()
            if base not in MODULE_OF:
                raise ValueError(f"no new module for {base!r} (from {DELETED_MODULE})")
            by_module.setdefault(MODULE_OF[base], []).append(name)
        indent = match.group("indent")
        return "\n".join(f"{indent}from {module} import {', '.join(items)}"
                         for module, items in by_module.items())
    return _IMPORT.subn(replace, text)


def unresolved(text):
    """-> lines that still address the deleted module (to fix by hand)."""
    return [line.strip() for line in text.splitlines()
            if "dark_base." in line or ("dark_base" in line and "import" in line)]


def process_py(path, write):
    text = path.read_text(encoding="utf-8")
    new, n_names = rename_python(text)
    new, n_imports = rewrite_dark_base_imports(new)
    return text, new, n_names + n_imports, unresolved(new)


def process_ipynb(path, write):
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    count, todo = 0, []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new, n = _WORD.subn(lambda m: RENAMES[m.group(1)], src)
        new, k = rewrite_dark_base_imports(new)
        count += n + k
        todo += unresolved(new)
        if new != src:
            cell["source"] = new.splitlines(keepends=True)
    indent = 1 if raw.startswith('{\n "') else 2
    new_raw = json.dumps(nb, indent=indent, ensure_ascii=False) + "\n"
    return raw, (new_raw if count else raw), count, todo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--write", action="store_true", help="change the files (default: report)")
    args = parser.parse_args(argv)

    files = []
    for path in args.paths:
        files += sorted(path.rglob("*.py")) + sorted(path.rglob("*.ipynb")) if path.is_dir() else [path]
    total = 0
    for path in files:
        if ".ipynb_checkpoints" in path.parts:
            continue
        process = process_ipynb if path.suffix == ".ipynb" else process_py
        old, new, count, todo = process(path, args.write)
        if count:
            total += count
            print(f"{count:4d}  {path}")
            if args.write and new != old:
                path.write_text(new, encoding="utf-8")
        for line in todo:
            print(f"      by hand: {path}: {line}")
    print(f"{total} renames{'' if args.write else ' (report only; --write to change)'}")


if __name__ == "__main__":
    sys.exit(main())
