"""One-shot path migrator for measurement_notebooks/QEC/*.ipynb.

Rules applied to CODE CELL source only (markdown cells and `outputs` arrays
are left alone):

  1. config_dir = '<old>' / "<old>"   ->  self-contained repo-relative form
     where <old> is any of:
         D:/python/multimode_expts/configs
         C:/python/multimode_expts/configs
         D:\python\multimode_expts\configs   (and \\ escape variants)
     New value:
         from pathlib import Path; config_dir = str(Path.cwd().parent.parent / 'configs')

  2. Other D:/python/multimode_expts -> C:/python/multimode_expts (+ \ variants)
  3. D:/experiments              -> C:/experiments              (+ \ variants)
  4. D:/closed_loop_runs         -> C:/closed_loop_runs         (+ \ variants)
  5. H:\Shared drives            -> G:\Shared drives            (all escape variants)
  6. Bare H: drive (H:\ or H:/ followed by alpha) -> G:  (catch-all)

Usage:
    python _migrate_qec_notebooks.py            # dry-run, prints summary
    python _migrate_qec_notebooks.py --apply    # write changes
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
QEC_DIR = REPO / "measurement_notebooks" / "QEC"

NOTEBOOKS = [
    "single_qubit_autocalibrate_v2.ipynb",
    "multiphoton_calibration_v2.ipynb",
    "T2_ac_stark_shift.ipynb",
    "coupler_lookup_analysis.ipynb",
    "coupler_systematic_study.ipynb",
    "coupler_systematic_study_v2.ipynb",
    "cpmg_noise_spectroscopy.ipynb",
    "f0g1_stark_flux_sensitivity.ipynb",
    "f0g1_stark_flux_sensitivity_lookup.ipynb",
    "Dual_Rail.ipynb",
    "Dual_Rail_V2.ipynb",
]

CONFIG_DIR_REPLACEMENT = (
    "from pathlib import Path; config_dir = str(Path.cwd().parent.parent / 'configs')"
)

CONFIG_DIR_PATTERNS = [
    re.compile(r"""config_dir\s*=\s*['"]D:/python/multimode_expts/configs/?['"]"""),
    re.compile(r"""config_dir\s*=\s*['"]C:/python/multimode_expts/configs/?['"]"""),
    re.compile(r"""config_dir\s*=\s*r?['"]D:\\python\\multimode_expts\\configs\\?['"]"""),
    re.compile(r"""config_dir\s*=\s*['"]D:\\\\python\\\\multimode_expts\\\\configs\\\\?['"]"""),
]

# Replacements use lambdas so the backslashes in the new string are taken
# literally rather than interpreted by `re.sub` as backreferences.
SUBS = [
    (re.compile(r"D:/python/multimode_expts"), lambda m: "C:/python/multimode_expts"),
    (re.compile(r"D:\\\\python\\\\multimode_expts"), lambda m: r"C:\\python\\multimode_expts"),
    (re.compile(r"D:\\python\\multimode_expts"), lambda m: r"C:\python\multimode_expts"),
    (re.compile(r"D:/experiments"), lambda m: "C:/experiments"),
    (re.compile(r"D:\\\\experiments"), lambda m: r"C:\\experiments"),
    (re.compile(r"D:\\experiments"), lambda m: r"C:\experiments"),
    (re.compile(r"D:/closed_loop_runs"), lambda m: "C:/closed_loop_runs"),
    (re.compile(r"D:\\\\closed_loop_runs"), lambda m: r"C:\\closed_loop_runs"),
    (re.compile(r"D:\\closed_loop_runs"), lambda m: r"C:\closed_loop_runs"),
    (re.compile(r"H:\\\\\\\\(?=[A-Za-z])"), lambda m: r"G:\\\\"),
    (re.compile(r"H:\\\\(?=[A-Za-z])"), lambda m: r"G:\\"),
    (re.compile(r"H:\\(?=[A-Za-z])"), lambda m: "G:\\"),
    (re.compile(r"H:/(?=[A-Za-z])"), lambda m: "G:/"),
]


def fix_source_str(src: str) -> str:
    for pat in CONFIG_DIR_PATTERNS:
        src = pat.sub(CONFIG_DIR_REPLACEMENT, src)
    for pat, repl in SUBS:
        src = pat.sub(repl, src)
    return src


def process(nb_path: Path, apply: bool) -> dict:
    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)
    cells_changed = 0
    line_changes: list[tuple[int, str, str]] = []
    for ci, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            new_lines = []
            cell_changed = False
            for line in src:
                new_line = fix_source_str(line)
                if new_line != line:
                    cell_changed = True
                    line_changes.append((ci, line.rstrip("\n"), new_line.rstrip("\n")))
                new_lines.append(new_line)
            if cell_changed:
                cells_changed += 1
                cell["source"] = new_lines
        else:
            new_src = fix_source_str(src)
            if new_src != src:
                cells_changed += 1
                line_changes.append((ci, src[:200], new_src[:200]))
                cell["source"] = new_src
    if apply and cells_changed:
        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
    return {"cells_changed": cells_changed, "line_changes": line_changes}


def main():
    apply = "--apply" in sys.argv
    print(f"{'APPLYING' if apply else 'DRY-RUN'} migration on {len(NOTEBOOKS)} notebooks under {QEC_DIR}\n")
    total_cells = 0
    total_lines = 0
    for name in NOTEBOOKS:
        p = QEC_DIR / name
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        result = process(p, apply)
        total_cells += result["cells_changed"]
        total_lines += len(result["line_changes"])
        print(f"  {name}: {result['cells_changed']} cells, {len(result['line_changes'])} line changes")
        for ci, old, new in result["line_changes"][:6]:
            print(f"    [cell {ci}]")
            print(f"      - {old.strip()[:160]}")
            print(f"      + {new.strip()[:160]}")
        if len(result["line_changes"]) > 6:
            print(f"    ... and {len(result['line_changes']) - 6} more")
    print(f"\nTOTAL: {total_cells} cells, {total_lines} line changes")
    if not apply:
        print("Run with --apply to write changes.")


if __name__ == "__main__":
    main()
