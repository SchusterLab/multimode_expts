20251215

Switching to pixi for a few reasons:

- C drive was running out of space (20GB) while D was empty (900GB). Pixi is local to each project so we can easily put the repos and the accompanying large files in D. This is easily many tens of GB of space saved from C.
- Our conda environment was very cursed. It's installed both machine-wide (C ProgramData Anaconda3) and in user space (home/.anaconda) and it varies from env to env. Our python version 3.8 was also getting too old. New features (eg some type hint syntax) are not supported, we cannot use new tools such as MCP in the slab environment, and it was triggering deprecation warnings from tools such as VS Code. Conda is also just slow. Given this broken state, making a new conda env without a clean slate isn't worth it.
- The repo/package organization was so chaotic that we rely on a number of ways to inject various things into the python path: Windows environment variable (but only did so for user env vars and not system env vars so it works in some runtimes and but not others depending on eg if you use VS Code or from the command line or SSH or whatever), `sys.path.append()` lines at the beginning of notebook initialization, etc.

---

So we installed pixi, uninstalled conda and system-wide python 3.13, and moved files from C:\_Lib\python to D:\python. Because we will start modifying D:\python, a second copy was made to D:\CLIBPYTHONBACKUPDONOTTOUCH. A third copy was made as a simple zip of this folder also under D. A fourth copy was made to S:\_BlueFors5\C Lib python backup. C:\_Lib\python is now gone to prevent accidental work on it. 

The PYTHONPATH env vars are also deleted from Windows. Before the deletion, the user PYTHONPATH was:
C:\_Lib\python\multimode_expts;C:\_Lib\python\slab;C:\_Lib\python;C:\_Lib\python\qick\qick_lib;C:\_Lib\python\rfsoc\rfsoc_multimode\example_expts;

and system PYTHONPATH was:
C:\_Lib\python\multimode_expts;C:\_Lib\python\slab;C:\_Lib\python

---

To start using pixi, we make a pyproject.toml for multimode_expts which is where most of our development takes place. This includes the dependencies and such. We are trying python 3.12 but in principle pixi allows this to be changed easily. I started populating the dependency list by just running some sample notebooks we commonly use and manually adding packages that are not found during imports. 

Two important but tricky dependencies to deal with are slab and qick. These also lived under C:\_Lib and contain changes untracked by Git, especially slab (281 untracked changes many of which seemed to be direct copies of a proprietary SDK). The priority here was to make as little change as humanly possible to make pixi run first, and then we'll clean up the tech debt eldritch horror. The process roughly went like this:

- qick was relatively easy. Just specify it as an editable pypi dependency at a relative path: ..\qick. This assumes that when we clone multimode_expts on other machines, qick will also be alongside it under the same parent dir. It seems this is already the convention so this should be fine (unless someone pip installed qick directly from PyPI in which case you want to undo that and git clone it to the same parent dir as other SLab repos). 
- slab is a bit more tricky. Broadly we take the same approach, declare it as pypi dependency at ..\slab, but pixi cannot run without modification. This is because the setup.py is way too old and no longer compatible. I tried a few different things and ended up with the current way of adding a pyproject.toml, moving all the setup and repo metadata there and leaving only a bare husk in setup.py. The original setup.py is backed up to old_setup.py because we've not been gitting for literal years at this stage.
- We also declare the multimode_expts repo itself a pypi editable dependency so that we can directly import the modules living under multimode_expts as is done widely throughout the code base. Note that we do have to specify the package structure in `tools.setuptools.packages.find` as this is a flat, non-src layout. 
- Now we can actually pixi install and start running something in a jupyter notebook. But the LSP tooling still doesn't know slab despite the python runtime being able to resolve package imports. This is because again slab is also a flat package layout and the packages and modules live under D:\python\slab instead of D:\python\slab\slab which the tools expect. The path of least resistance is to give the tools we use a little help. Inside `multimode_expts/.vscode/settings.json` we tell python analysis to add the parent dir of multimode_expts as an extra path so when it looks for the slab package it sees our slab repo. Inside the pyrightconfig.json we do the same for pyright running in neovim. 
- Also on the tooling side, VS Code doesn't support pixi very well yet. It was easy enough to get LSP to work just by pointing the python interpreter path to the pixi default env one, but this doesn't fly with Jupyter notebook execution simply because VS Code tries to shoehorn the environment activation strategy of other tools and it doesn't work so that times out after a long while and it just decides to accept that the python runtime doesn't have any PATH env var populated. This leads directly to kernel crash when you import anything because there's no PATH and it cannot find the DLLs. The solution that is compatible with most of our workflows including VS Code remote seems to be to make a manual Jupyter kernel spec under C:\Users\26049\AppData\Roaming\jupyter\kernels for each workspace (see the json there for detail). Since we only have one active workspace now, let's just live with this approach for now until the VS Code people solve their compatibility issue (it seems the GitHub issue is actively being looked into literally this week).

Now the init sections of the single qubit autocalibrate notebook seems to run and I tried running resonator spectroscopy and single shot and we got back data from RFSoC no problem although fridge was warm so if there's any issue with the data (I guess that's unlikely but who knows) we don't know. tqdm doesn't work yet though. To do.

As a note, please refrain from using sys.path.append whenever a package/module under multimode_expts or slab cannot be found. There's usually a better, more sustainable solution than that. Talk to people or LLMs. It takes a little bit more time than the few seconds of typing that one line but you spare your fellow colleagues from having to clean after horrendous piles of tech debt in the future. Please be nice to those you work with. It also forces you to write better code which, even if you don't realize it yet, means being more efficient at your own work.


--- 

Edit 20251217:

The current state of the slab package is beyond rescue. 
I'm vendoring the useful code into multimode_expts/slab and deleting D:\python\slab since it's already backed up 3 times in the C lib python backups.

