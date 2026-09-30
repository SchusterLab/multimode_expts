# -*- coding: utf-8 -*-
"""The non-MBR qsim Programs still play what they played when pinned.

A characterization pin (see ``tests/program_asm_golden.py``): a failure says
the pulses changed, not that they are wrong. If the change is intended,
regenerate with ``pixi run python -m tests.program_asm_golden`` in the same
commit, and read the diff.

Run:  pixi run python -m pytest tests/test_program_asm_golden.py -v
"""
import contextlib
import difflib
import io

import pytest

from tests import program_asm_golden as golden

pytest.importorskip("qick")

KEYS = [case.key for case in golden.CASES]


def test_every_case_is_pinned():
    pinned = sorted(p.name[: -len(".txt.gz")] for p in golden.GOLDEN_DIR.glob("*.txt.gz"))
    assert pinned == sorted(KEYS), (
        "the pinned files and CASES differ: regenerate with "
        "`pixi run python -m tests.program_asm_golden` and commit the change on purpose")


def test_keys_are_unique():
    assert len(set(KEYS)) == len(KEYS)


@pytest.fixture(scope="module")
def compiled():
    """Every case compiled once for the whole module (the programs print)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return dict(golden.rendered())


@pytest.mark.parametrize("key", KEYS)
def test_program_matches_golden(compiled, key):
    actual = compiled[key]
    expected = golden.read(key)
    if actual == expected:
        return
    diff = list(difflib.unified_diff(
        expected.splitlines(), actual.splitlines(),
        fromfile=f"{key} (golden)", tofile=f"{key} (now)", lineterm="", n=2))
    pytest.fail(
        f"{key}: compiled program differs from the golden in "
        f"{sum(1 for line in diff if line[:1] in '+-') - 2} lines.\n"
        f"If the change is intended, regenerate with "
        f"`pixi run python -m tests.program_asm_golden` and review that diff.\n"
        + "\n".join(diff[:80]))
