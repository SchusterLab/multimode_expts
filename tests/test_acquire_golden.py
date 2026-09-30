# -*- coding: utf-8 -*-
"""The qsim sweep drivers still ask for and slice what they did when pinned.

A characterization pin (see ``tests/acquire_golden.py``): a failure says a
driver changed how many readouts it asks for, the order of its sweep points,
or which lane lands where; not that the change is wrong. If it is intended
(the plan's 7.3 lists the expected ones), regenerate with
``pixi run python -m tests.acquire_golden`` in the same commit, and read the
diff.

Run:  pixi run python -m pytest tests/test_acquire_golden.py -v
"""
import json

import pytest

from tests import acquire_golden as golden

pytest.importorskip("qick")

KEYS = [case.key for case in golden.CASES]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    return golden.records(tmp_path_factory.mktemp("acquire_golden"))


def test_every_case_is_pinned():
    assert sorted(golden.read()) == sorted(KEYS), (
        "the golden and CASES differ: regenerate with "
        "`pixi run python -m tests.acquire_golden` and commit the change on purpose")


@pytest.mark.parametrize("key", KEYS)
def test_driver_matches_golden(recorded, key):
    # Through JSON, so the comparison sees what the file holds.
    now = json.loads(json.dumps(recorded[key], sort_keys=True))
    pinned = golden.read()[key]
    if now == pinned:
        return
    for part in ("builds", "read_num", "data"):
        assert now[part] == pinned[part], (
            f"{key}: '{part}' differs from the golden. If intended, regenerate with "
            f"`pixi run python -m tests.acquire_golden` and review the diff.")
