"""Prove moved code is textually identical to its pre-move original.

Compares AST-normalized statement lists, so whitespace, dedent and comment
reflow cannot mask or manufacture a difference. Stronger than a runtime check
on one dataset: it covers every branch, not only the ones the characterization
data reaches.
"""
import ast, subprocess, sys
from pathlib import Path

ROOT = Path(r"C:\python\multimode_expts_guan")
GOD = "experiments/qsim/floquet_dark_mode_readout.py"

# Each moved function, with the commit whose god file still contained it.
MOVED = [
    ("analyze_matrix_pencil",       "04cea3a", "matrix_pencil"),
    ("analyze_matrix_pencil_trace", "04cea3a", "matrix_pencil"),
    ("merge_spectra",               "c1867d6", "level_statistics"),
    ("analyze_level_statistics",    "c1867d6", "level_statistics"),
    ("analyze_sff",                 "c1867d6", "level_statistics"),
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

def statements(fn):
    """Body statements, docstring dropped, each AST-normalized."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return [ast.unparse(s) for s in body]

def normalize(text):
    """Erase the edits we made on purpose: the self->data signature change and
    the class-qualified internal call."""
    return (text.replace("EncodingHamiltonianSpectroscopyExperiment.", "")
                .replace("self.data", "data"))

sources = {rev: at(rev) for rev in {r for _, r, _ in MOVED}}
new_src = {m: (ROOT / f"fitting/qsim/{m}.py").read_text(encoding="utf-8")
           for m in {m for _, _, m in MOVED}}

ok = True
for name, rev, module in MOVED:
    old_fn = find_fn(sources[rev], name, "EncodingHamiltonianSpectroscopyExperiment")
    new_fn = find_fn(new_src[module], name if name != "analyze_matrix_pencil_occupation"
                     else "refit_occupation")
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
