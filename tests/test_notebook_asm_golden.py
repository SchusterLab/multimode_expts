"""The calibration notebooks still play the programs they played when pinned.

A characterization pin (see ``tests/notebook_asm_golden.py`` and
``docs/testing.md``): a failure says the pulses changed, not that they are
wrong. If the change is intended, regenerate with
``pixi run python -m tests.notebook_asm_golden`` in the same commit, and read
the diff.

Run:  pixi run python -m pytest tests/test_notebook_asm_golden.py -v
"""
import difflib
import re

import pytest

from tests import notebook_asm_golden as golden

pytest.importorskip("qick")


@pytest.fixture(scope="module", params=golden.NOTEBOOKS)
def captured(request):
    """(name, text now, pinned text) for one notebook; about 15 s each."""
    name = request.param
    return name, golden.render_in_subprocess(name), golden.read(name)


def test_notebook_programs_match_the_golden(captured):
    name, now, pinned = captured
    if now == pinned:
        return
    diff = list(difflib.unified_diff(
        pinned.splitlines(), now.splitlines(), "pinned", "now", lineterm="", n=2))
    shown = "\n".join(diff[:80])
    more = f"\n... {len(diff) - 80} more diff lines" if len(diff) > 80 else ""
    pytest.fail(f"{name}: the compiled programs changed.\n{shown}{more}")


def test_every_pinned_program_has_instructions(captured):
    """Guards against the capture recording empty programs and passing."""
    name, _, pinned = captured
    groups = re.split(r"^## program group ", pinned, flags=re.M)[1:]
    assert groups, f"{name}: no program groups pinned"
    for group in groups:
        for section in re.split(r"^### (?:first|last): ", group, flags=re.M)[1:]:
            asm = section.split("# asm", 1)[1]
            assert re.search(r"^\s+\w+ ", asm, flags=re.M), (
                f"{name}: an empty program in group {group.splitlines()[0]}")
