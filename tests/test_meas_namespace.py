# -*- coding: utf-8 -*-
"""The flat `experiments` namespace holds no deprecated MBR class.

`experiments/__init__.py` copies every class that a walked module holds into
`experiments` (the `meas.` namespace), including classes the module only
imported. It walks one level below each subpackage, so modules under
`experiments/qsim/deprecated/` are not walked. But if a walked module imports
a deprecated class by name, that class is exported again, and it can shadow
the new class of the same name (docs/qsim/mbr_redesign.md, section 2). Import
the deprecated module instead and use its attributes.

Run:  pixi run python -m pytest tests/test_meas_namespace.py -v
"""
import inspect

import experiments as meas


def test_no_meas_class_comes_from_a_deprecated_module():
    leaked = {name: obj.__module__
              for name, obj in vars(meas).items()
              if inspect.isclass(obj) and ".deprecated." in obj.__module__
              and name.startswith("MBR")}
    assert not leaked, (
        f"deprecated MBR classes in the experiments namespace: {leaked}. "
        f"A walked module imports them by name; import the module instead.")
