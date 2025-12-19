# Multimode Experiment Code

Migrating to pixi. Install pixi on your machine and `pixi install` to set up the environment. 
Then `pixi run jupyter lab` or set up a Jupyter kernel to allow VS Code execute notebooks in this envrionment (details to be added).

slab code that is actively used in this repo is now vendored inside slab as a module.

`pyproject.toml` takes care of making sure the python modules inside this repo can be correctly imported without hacking path variables.
Just modify the include and exclude lists therein to control which directories contain code that is allowed to be imported.


## TODO

- [ ] Delete the binary files that don't belong in git. hdf5 data, images etc. Move them to the data/log folder accompanying each experiment.
- [ ] The few `slab` files are simply vendored in their original cursed state. If we straighten up the Experiment class and allow say proper file-reading functionalities, we can actually easily use the `analyze` functions in each child class to do the analysis instead of collecting everything in `fitting_display_classes` or such.
- [ ] Migrate to tProcv2 at some point
- [ ] Consider using pydantic for complex config dictionaries.

