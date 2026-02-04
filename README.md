# Multimode Experiment Code

## Installation

[Install pixi](https://pixi.prefix.dev/latest/installation/) on your machine and `pixi install` to set up the environment. 
Then `pixi run jupyter lab` or set up a Jupyter kernel to allow VS Code to execute notebooks using the default pixi environmnet (LLMs can tell you how).

## Running

On BF5, just double click `start_all.bat` and follow the instructions to start the Pyro nameserver, the job server and the worker

Alternatively, `pixi run [nameserver|server|worker]` to start these things individually.

You can pass the `--mock` option to the worker to use mock mode (use mock QICK and Yoko instead of real hardware and gitignored local folders instead of real D drive locations on BF5).

## Code organization

Pending major refactoring, `slab` code that is actively used in this repo is now vendored inside `slab` as a module.

`pyproject.toml` takes care of making sure the python modules inside this repo can be correctly imported without hacking path variables:
If you add "abc" to the "include" list under `[tool.setuptools.packages.find]`, the folder `abc` will become an importable module (such that you can `import abc` in your code anywhere).
No `import multimode_expts` or `import .anything` or `sys.path` hacking!

## TODO

- [x] Experiment name should be passed from client instead of always using the worker default
- [x] Add function to interrupt an executing job 
- [x] Versioned cfg files and the job database should be git ignored and synced to S drive
- [ ] Consider SSH forwarding so we can run things from our own desktops (and move log files to S)
- [ ] Delete the binary files that don't belong in git. hdf5 data, images etc. Move them to the data/log folder accompanying each experiment.
- [ ] The few `slab` files are simply vendored in their original cursed state. If we straighten up the Experiment class and allow say proper file-reading functionalities, we can actually easily use the `analyze` functions in each child class to do the analysis instead of collecting everything in `fitting_display_classes` or such.
- [ ] Migrate to tProcv2 at some point
- [ ] Consider using pydantic for complex config dictionaries.
