import importlib
import inspect
import os
import sys

"""
module_path: parent directory path with . as subfile dividers
f: filename to import modules from
"""


def import_modules_from_files(module_path, f):
    if f[0] != "_" and f[0] != ".":
        module_name = module_path + "." + f.split(".")[0]
        m = importlib.import_module(module_name)
        # print("imported", module_name)
        for name, obj in inspect.getmembers(m):
            if inspect.isclass(obj):
                setattr(thismodule, name, getattr(m, name))
        del m
        del name
        del obj


path = __path__[0]
thismodule = sys.modules[__name__]
files = os.listdir(path)
module_path = os.path.split(path)[-1]
skipped_files = []
for f in files:
    fpath = os.path.join(path, f)
    if f[0] == "_" or f[0] == ".":
        continue

    if os.path.isdir(fpath):
        print("Importing all classes from ", fpath)
        subfiles = os.listdir(fpath)
        submodule_path = module_path + "." + os.path.split(fpath)[-1]
        for subf in subfiles:
            import_modules_from_files(submodule_path, subf)
    else:
        # stick to the convention that only sub modules contain Program and Experiment classes
        # top level modules are for generic, shared utilities
        skipped_files.append(f)

print(f'Skipped top-level files {skipped_files}')

del f
del thismodule
del files

# Explicitly export the runner modules (these are top-level utilities)
from experiments.station import MultimodeStation
from experiments.characterization_runner import (
    CharacterizationRunner,
    PreProcessor,
    PostProcessor,
    default_preprocessor,
    default_postprocessor,
)
from experiments.sweep_runner import SweepRunner, register_analysis_class

