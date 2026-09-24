# Overall pattern of many body Ramsey experiments

A survey of existing measurements say what we do is basically sweeping various combinations and parts of the grid spanned by the direct product of the following params:
	- Disorder realization
	- Prepared (also termed "encoded") Fock state
	- Unprepared (also termed "decoded") Fock state
	- Experiment (fwd-fwd-fwd…) vs calibration (fwd-bwd-fwd-bwd…)
	- Trotterized time steps
	- Initial pi/2 phase
	- Final pi/2 phase
The last two params (the 4 phase combinations) is essentially always swept over because together they produce one complex matrix element

Architectural choices to make in this meeting:
	- What does each Experiment do in the context of the grid above -> this allows analyze/display functions to be meaningful
	- What does each HDF5 file contain
	- What does each Job submission consist of 

# Ideal target

Proposed ground rules
	- Each combination of the grid above should be a different Experiment (so one Experiment for say an ensemble with lots of disorder realizations, one Experiment for one disorder realization consisting of many init/final Fock states, etc etc..)
		○ MBRDisorderEnsembleExperiment (disorder * init_fock==final_fock * time, omitting the 4 phase combo sweeps in this notation because physics always needs the complex number constructed from them, same below)
		○ MBRHamTomoExperiment (init_fock * final_fock * time)
		○ MBRSpectrumExperiment (init_fock == final_fock * time)
		○ MBRTimeTraceExperiment (time)
		○ MBRStarkCalExperiment (time). Note: up to here, the experiments follow a neat hierarchy. Each one builds from/depends on the one below it.
		○ MBROrthogonalityExperiment (init_fock * final_fock)
		○ MBROrthoColumnExperiment (final_fock) Note: these last two don't need AC Stark shift phase calibration experiments because they don't evolve the Hamiltonian. The last one really only exists because we agree below that each job is a 1D param sweep at most (not counting the 4 phase combos) so the full orthogonality sweep is one dimension too high and thus need an intermediate layer.
		
	- Job submissions happen through Runners, and ideally we will have two types of runners:
		○ 1D (currently CharacterizationRunner): 1 job == 1 HDF5 file == 1 Experiment
		○ 2D (currently SweepRunner): 1 run == 1 Mother Experiment == 1 aggregated HDF5 file, which contains N jobs == N child HDF5 files == N lower-dimensional Experiment objects
			§ To investigate: does the Sweep runner class accept arbitrary constructions of how to assemble N 1D objects/files into one 2D file/object? In other words can we hook OrthoColumn -> Orthogonality and TimeTrace->Spectrum constructions into the runner? This might not be trivial 

Decisions
	- What does one job/HDF5 file contain? Each job-generated file is always a 1D array of complex numbers after processing
		○ EITHER final Fock state OR time steps
		○ Initial pi/2 phase
		○ Final pi/2 phase
	- How do we deal with dimensions higher than SweepRunner (eg disorder times time times occupation):
		○ Say I want a disorder-ensemble experiment object (sweep over realization * Fock states * time), how do construct such an experiment object whose analyze function does ensemble average and the display function plots the level statistics? 
			§ Acquire data: no longer achieved by the acquire() method of this class, but rather just use for-loops in notebook cells. Humans should manually make sure that while taking data, we also generate a manifest of this collection of data in yaml (because it allows us to add comments/annotations manually). These yamls should live C:\experiments\[name]\data_manifests or the like and contain all necessary metadata (job IDs, HDF5 filenames, etc)
			§ Load existing data: a new constructor that reads the manifest yaml and pulls in all the HDF5 files. 


# Migration step

To be implemented. Eg walk over all old HDF5s and save them to C:\experiments\[name]\converted_data

Backwards compatibility: no need for shims (just redirect an old function to the new name). Just migrate 

Migration code should live in a migration script/module, eg under tools/, not in new classes

