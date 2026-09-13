# -*- coding: utf-8 -*-
"""The pulse layer must compile to the same tProc program it compiled before.

This is the net under the pulse-base extraction. Moving a swap train into its
own module cannot be checked by "does it still acquire" -- it always does. It
can be checked exactly, because the whole observable output of that code is
the instruction listing plus the waveform table, and both are deterministic
given a pinned config set.

See ``tests/asm_golden.py`` for what is rendered and how to regenerate.
"""
import difflib

import pytest

from tests import asm_golden

KEYS = sorted(p.name[: -len(".txt.gz")]
              for p in asm_golden.GOLDEN_DIR.glob("*.txt.gz"))

# Pinned so a config set or stage disappearing fails loudly instead of
# silently shrinking the checked surface to nothing.
EXPECTED_KEY_COUNT = 24


def test_every_pinned_program_is_covered():
    assert len(KEYS) == EXPECTED_KEY_COUNT, (
        f"golden holds {len(KEYS)} programs, expected {EXPECTED_KEY_COUNT}: "
        f"regenerate with `pixi run python -m tests.asm_golden` and commit "
        f"the change on purpose")


@pytest.fixture(scope="module")
def compiled():
    """Every stage of every pinned set, compiled once for the whole module."""
    from experiments.qsim.mbr_campaign import pinned_sets

    out = {}
    for set_name in sorted(pinned_sets()):
        for key, prog in asm_golden.programs(set_name):
            out[key] = asm_golden.render(prog)
    return out


@pytest.mark.parametrize("key", KEYS)
def test_program_matches_golden(compiled, key):
    actual = compiled[key]
    expected = asm_golden.read(key)
    if actual == expected:
        return

    diff = list(difflib.unified_diff(
        expected.splitlines(), actual.splitlines(),
        fromfile=f"{key} (golden)", tofile=f"{key} (now)", lineterm="", n=2))
    pytest.fail(
        f"{key}: compiled program differs from the golden in "
        f"{sum(1 for line in diff if line[:1] in '+-') - 2} lines.\n"
        f"If the change is intended, regenerate with "
        f"`pixi run python -m tests.asm_golden` and review that diff.\n"
        + "\n".join(diff[:80]))


def test_golden_detects_a_pulse_change(monkeypatch):
    """The golden is sensitive, not vacuous.

    Without this, a render that silently stopped including the ASM would keep
    every comparison above green. One unit of swap gain, taken from the
    calibration the way the real program takes it, must show up.
    """
    from experiments.dataset import StorageManSwapDataset

    original = StorageManSwapDataset.get_gain
    monkeypatch.setattr(StorageManSwapDataset, "get_gain",
                        lambda self, stor_name: original(self, stor_name) + 1)

    key, prog = next(asm_golden.programs("preload_current"))
    assert asm_golden.render(prog) != asm_golden.read(key), (
        "a one-unit gain change left the rendered program identical -- the "
        "golden is not looking at the pulses")
