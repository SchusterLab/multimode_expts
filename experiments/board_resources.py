"""Board resource limits that only the QICK drivers check, at upload.

QICK's compile does not check the envelope memory, the mux tone count or the
tProc program memory: the board drivers do, when the program is loaded. A
program that is too large compiles in mock mode and fails on the board. This
module does the same checks with the limits in ``soccfg``, so that the mock
(``MockQickSoc``) fails where the board would. See
docs/lab_workflow_tooling_spec.md item D.

The readout buffer limits depend on the acquire mode (averaged or decimated)
and the number of shots, not only on the program. ``MockQickSoc`` checks them in
the soc calls; ``check_program_resources`` does not.
"""

from collections import Counter


class BoardResourceError(RuntimeError):
    """A program needs more of a board resource than ``soccfg`` has."""


def envelope_report(prog, ch: int, max_rows: int = 12) -> str:
    """-> the envelopes on generator ``ch``, grouped by size, largest total first."""
    envs = prog.envelopes[ch]["envs"]
    sizes = Counter(len(env["data"]) for env in envs.values())
    names_by_size: dict = {}
    for name, env in envs.items():
        names_by_size.setdefault(len(env["data"]), []).append(name)
    rows = sorted(sizes.items(), key=lambda kv: kv[0] * kv[1], reverse=True)
    lines = []
    for size, count in rows[:max_rows]:
        names = names_by_size[size]
        shown = ", ".join(names[:3]) + (f", ... ({len(names) - 3} more)" if len(names) > 3 else "")
        lines.append(f"    {count} x {size} samples = {count * size}: {shown}")
    if len(rows) > max_rows:
        rest = sum(size * count for size, count in rows[max_rows:])
        lines.append(f"    ... {len(rows) - max_rows} more sizes, {rest} samples")
    return "\n".join(lines)


def program_resource_problems(prog, soccfg) -> list:
    """-> one message per resource that ``prog`` uses past the ``soccfg`` limit."""
    problems = []
    for ch, env in enumerate(prog.envelopes):
        used = env["next_addr"]
        gen = soccfg["gens"][ch]
        if used > gen["maxlen"]:
            problems.append(
                f"gen ch {ch} ({gen['type']}): envelopes use {used} of {gen['maxlen']} samples "
                f"({used - gen['maxlen']} too many).\n" + envelope_report(prog, ch)
            )
    for ch, cfg in prog.gen_chs.items():
        n_tones = soccfg["gens"][ch].get("n_tones")
        if "mux_tones" in cfg and n_tones is not None and len(cfg["mux_tones"]) > n_tones:
            problems.append(
                f"gen ch {ch}: {len(cfg['mux_tones'])} mux tones, the generator has {n_tones}."
            )
    binprog = prog.binprog
    if binprog is not None:
        tproc = soccfg["tprocs"][0]
        # tProc v1: a list of 64-bit words; pmem_size is in 64-bit words too.
        # The v1 data memory is not loaded with the program, so there is no
        # dmem check for v1.
        pmem = binprog["pmem"] if isinstance(binprog, dict) else binprog
        if len(pmem) > tproc["pmem_size"]:
            problems.append(f"tProc: program is {len(pmem)} words, pmem_size is {tproc['pmem_size']}.")
        if isinstance(binprog, dict) and len(binprog.get("dmem", [])) > tproc["dmem_size"]:
            problems.append(
                f"tProc: data memory is {len(binprog['dmem'])} words, dmem_size is {tproc['dmem_size']}."
            )
    return problems


def check_program_resources(prog, soccfg) -> None:
    """Raise ``BoardResourceError`` if ``prog`` does not fit the board in ``soccfg``.

    Compiles ``prog`` first if it is not compiled: envelopes can be declared in
    ``make_program``.
    """
    if prog.binprog is None:
        prog.compile()
    problems = program_resource_problems(prog, soccfg)
    if problems:
        raise BoardResourceError(
            f"{type(prog).__name__} does not fit the board:\n" + "\n".join(problems)
        )
