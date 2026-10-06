# -*- coding: utf-8 -*-
"""Alias of ``tools/legacy_saved_jobs.py``, the frozen converter's raw-file reader.

The deprecated classes and their tests import it under this name. Deprecated
code may import live code (docs/qsim/mbr_step7_plan.md, section 2); ``tools/``
is not a package, so the file is loaded by path, once, as ``legacy_saved_jobs``.
"""
import importlib.util
import sys
from pathlib import Path

_NAME = "legacy_saved_jobs"
if _NAME not in sys.modules:
    _path = Path(__file__).resolve().parents[3] / "tools" / f"{_NAME}.py"
    _spec = importlib.util.spec_from_file_location(_NAME, _path)
    sys.modules[_NAME] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules[_NAME])
sys.modules[__name__] = sys.modules[_NAME]
