Overall structure:

- We use pixi now so you need to use pixi when calling python.
- We mostly run the Jupyter notebooks in measurement_notebooks as the main interactive interface.
- Inside the submodules under experiments, we have many .py files that should in theory be rather structured. Each should in principle define a NameProgram class and a NameExperiment class with Name being the name of a specific experiment. The Program class inherits both experiments.MM_base.MM_base and an external library that handles communication with hardware. The Experiment class should primarily provide acquire, analyze, display methods as the common interface. 

Known problems: 

- In general, it would be nice to migrate to lmfit for fitting but a complete refactoring is a pretty big undertaking. So we will leave the old files be for the moment but consider doing lmfit when we implement a new fitting function from scratch.
- We have several different config file formats under configs (yaml, csv). We haven't made up our minds on whether we should consolidate and if so, how. Leave them be for now unless this becomes a productivity drag.

