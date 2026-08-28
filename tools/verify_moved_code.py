"""Prove moved code is textually identical to its pre-move original.

Compares AST-normalized statement lists, so whitespace, dedent and comment
reflow cannot mask or manufacture a difference. Stronger than a runtime check
on one dataset: it covers every branch, not only the ones the characterization
data reaches.
"""
import ast, copy, subprocess, sys
from pathlib import Path

ROOT = Path(r"C:\python\multimode_expts_guan")
GOD = "experiments/qsim/floquet_dark_mode_readout.py"

# Each moved function, with the commit whose god file still contained it, the
# module it landed in, and -- when the move renamed it -- its new name.
MOVED = [
    ("analyze_matrix_pencil",       "04cea3a", "matrix_pencil"),
    ("analyze_matrix_pencil_trace", "04cea3a", "matrix_pencil"),
    ("analyze_level_statistics",    "c1867d6", "level_statistics"),
    ("analyze_sff",                 "c1867d6", "level_statistics"),
    ("analyze_spectrum",            "5365e4f", "mbr_spectrum"),
    ("_cycle_branches",             "718a5db", "mbr_phase", "cycle_branches"),
    ("build_phase_correction",      "718a5db", "mbr_phase"),
    ("_unwrap_cycle_phase",         "718a5db", "mbr_phase", "unwrap_cycle_phase"),
    ("_saved_correction",           "718a5db", "mbr_phase", "saved_correction"),
]

# Rows carry an optional fourth field; normalize to a fixed shape once.
MOVED = [row if len(row) == 4 else (*row, row[0]) for row in MOVED]

# Retired rows: moved verbatim, then intentionally changed by a later behavior
# fix. A row here can never match again, and leaving it in MOVED makes this
# tool report REVIEW NEEDED forever -- which is how a tool stops being read.
# Kept as a record of which functions are no longer their pre-move originals.
DIVERGED = [
    ("merge_spectra", "level_statistics", "2da9cdc",
     "now refuses off-diagonal merges instead of rebuilding theory from the "
     "pre-generalization return amplitude"),
]

def at(rev):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{GOD}"],
                          capture_output=True, text=True, encoding="utf-8").stdout

def find_fn(source, name, cls=None):
    tree = ast.parse(source)
    scope = tree
    if cls:
        scope = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
        if scope is None:
            return None
    return next((n for n in scope.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)

def _is_docstring(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _strip_docstrings(node):
    """Drop docstrings everywhere, including from nested functions.

    Dedenting a moved block reflows the indentation inside nested docstrings,
    which is a cosmetic difference. Left in, it makes this tool report a
    difference on every move, and a tool that always says REVIEW NEEDED is a
    tool nobody reads.
    """
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if isinstance(body, list) and body and _is_docstring(body[0]) and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            child.body = body[1:] or [ast.Pass()]
    return node


def statements(fn):
    """Body statements, docstrings dropped throughout, each AST-normalized."""
    fn = _strip_docstrings(copy.deepcopy(fn))
    return [ast.unparse(s) for s in fn.body]

def normalize(text):
    """Erase the edits we made on purpose: the self->data signature change and
    the class-qualified internal call."""
    return (text.replace("EncodingHamiltonianSpectroscopyExperiment.", "")
                .replace("self.data", "data"))

sources = {rev: at(rev) for rev in {r for _, r, _, _ in MOVED}}
new_src = {m: (ROOT / f"fitting/qsim/{m}.py").read_text(encoding="utf-8")
           for m in {m for _, _, m, _ in MOVED}}

ok = True
for name, rev, module, new_name in MOVED:
    old_fn = find_fn(sources[rev], name, "EncodingHamiltonianSpectroscopyExperiment")
    new_fn = find_fn(new_src[module], new_name)
    if old_fn is None or new_fn is None:
        print(f"  {name}: NOT FOUND (old={old_fn is not None}, new={new_fn is not None})")
        ok = False; continue

    old_stmts = [normalize(s) for s in statements(old_fn)]
    new_stmts = [normalize(s) for s in statements(new_fn)]
    # Drop the self.data defaulting that moved into the wrapper.
    drop = {"if data is None:\n    data = data", "data = data if data is None else data"}
    old_stmts = [s for s in old_stmts if s not in drop]

    if old_stmts == new_stmts:
        print(f"  {name}: IDENTICAL ({len(new_stmts)} statements)")
    else:
        print(f"  {name}: DIFFERS ({len(old_stmts)} vs {len(new_stmts)} statements)")
        for i, (a, b) in enumerate(zip(old_stmts, new_stmts)):
            if a != b:
                print(f"      stmt {i}:\n        old {a.splitlines()[0][:80]!r}\n        new {b.splitlines()[0][:80]!r}")
                break
        ok = False

print("\nVERDICT:", "all moved code textually identical" if ok else "REVIEW NEEDED")
sys.exit(0 if ok else 1)
