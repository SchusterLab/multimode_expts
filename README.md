# Multimode Experiment Code

Migrating to pixi. Install pixi on your machine and `pixi install` to set up the environment. 
Then `pixi run jupyter lab` or set up a Jupyter kernel to allow VS Code execute notebooks in this envrionment (details to be added).

slab code that is actively used in this repo is now vendored inside slab as a module.

`pyproject.toml` takes care of making sure the python modules inside this repo can be correctly imported without hacking path variables.
Just modify the include and exclude lists therein to control which directories contain code that is allowed to be imported.


