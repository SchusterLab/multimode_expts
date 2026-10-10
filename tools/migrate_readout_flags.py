"""Move the retired readout flags to the `readout` key in notebook code cells (plan 10F)."""
import json, re, sys
from pathlib import Path
FLAGS = {"multiparity_readout": "multiparity", "parity_readout": "parity",
         "slow_pi_ge_readout": "slow_pi_ge"}
DROP = re.compile(r"^\s*(perform_wigner|parity_readout|multiparity_readout|slow_pi_ge_readout)"
                  r"\s*=\s*False\s*,?\s*(#.*)?$")
WIG_TRUE = re.compile(r"^\s*perform_wigner\s*=\s*True\s*,?\s*(#.*)?$")
SET_TRUE = re.compile(r"^(\s*)(multiparity_readout|parity_readout|slow_pi_ge_readout)\s*=\s*True(\s*,?.*)$")
GET = re.compile(r"([\w.\[\]]+)\.get\(\s*['\"](multiparity_readout|parity_readout|perform_wigner)['\"]\s*,\s*False\s*\)")
MODE = {"multiparity_readout": "multiparity", "parity_readout": "parity", "perform_wigner": "wigner"}
IMPORT = "from experiments.qsim.qsim_base import saved_readout_mode\n"

def fix(src):
    out, used = [], False
    for line in src.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        if DROP.match(body) or WIG_TRUE.match(body):
            continue
        m = SET_TRUE.match(body)
        if m:
            line = f"{m[1]}readout='{FLAGS[m[2]]}'{m[3]}" + line[len(body):]
        if GET.search(line):
            line = GET.sub(lambda g: f"(saved_readout_mode({g[1]}) == '{MODE[g[2]]}')", line)
            used = True
        out.append(line)
    new = "".join(out)
    if used and IMPORT.strip() not in new:
        new = IMPORT + new
    return new

for p in map(Path, sys.argv[2:]):
    raw = p.read_text(encoding="utf-8"); nb = json.loads(raw)
    indent = 1 if raw.startswith('{\n "') else 2
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code": continue
        src = "".join(c["source"]); new = fix(src)
        if new != src:
            n += 1; c["source"] = new.splitlines(keepends=True)
    print(f"{n:4d} cells  {p}")
    if n and sys.argv[1] == "--write":
        p.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
