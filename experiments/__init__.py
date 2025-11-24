import importlib
import inspect
import os
import sys

"""
Traverse through all the modules and import everything and every class into a flat hierarchy
"""


def import_modules_from_files(module_path, f, thismodule):
    if f[0] != "_" and f[0] != ".":
        module_name = module_path + "." + f.split(".")[0]
        try:
            m = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            assert "multimode_expts" in " ".join(sys.path), "path to multimode_expts needs to be in PYTHONPATH"
            raise e
        print("imported", module_name)
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
for f in files:
    fpath = os.path.join(path, f)
    if f[0] == "_" or f[0] == ".":
        continue

    print("importing from path ", fpath)
    if os.path.isdir(fpath):
        subfiles = os.listdir(fpath)
        submodule_path = module_path + "." + os.path.split(fpath)[-1]
        for subf in subfiles:
            import_modules_from_files(submodule_path, subf, thismodule)
    else:
        import_modules_from_files(module_path, f, thismodule)

del f
del thismodule
del files

