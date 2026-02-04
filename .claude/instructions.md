Overall structure:

- We use pixi now so you need to use pixi when calling python.
- We mostly run the Jupyter notebooks in measurement_notebooks as the main interactive interface.
- Inside the submodules under experiments, we have many .py files that should in theory be rather structured. Each should in principle define a NameProgram class and a NameExperiment class with Name being the name of a specific experiment. The Program class inherits both experiments.MM_base.MM_base and an external library that handles communication with hardware. The Experiment class should primarily provide acquire, analyze, display methods as the common interface. 

Known problems: 

- Although the ideal pattern is to use the analyze() methods under each child Experiment class for analysis and display() for plotting, many of these have been collected under fitting.fit_display_classes which contains usually the most updated versions of analysis and plotting code. We would, however, like to gradually enforce more discipline and move back to using the analyze() and display() under each Experiment as the common interface. At the current stage, we can first switch direct calls to the fit_display_classes classes to OOP patterns ie each Expeirment object calling their own analyze() and display() on their own data inside which the core logic is delegated to the fit_display_classes code. See experiments/single_qubit/single_shot.py and experiments/single_qubit/t2_ramsey.py for examples of such delegation.
- In general, it would be nice to migrate to lmfit for fitting but a complete refactoring is a pretty big undertaking. So we will leave the old files be for the moment but consider doing lmfit when we implement a new fitting function from scratch.
- We have several different config file formats under configs (yaml, csv). We haven't made up our minds on whether we should consolidate and if so, how. Leave them be for now unless this becomes a productivity drag.

Notes for claude:
This machine is on windows. Mistaking it for a real unix system will cause creation of a file named "nul" instead of piping the outputs and this is very difficult to delete! Beware not to create nul files.
