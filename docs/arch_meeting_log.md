> History and reasons only. The current spec is `docs/qsim/mbr_redesign.md`.
> Where the two differ, the spec wins.

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
			§ Investigated 2026-09-24: no, and we do not use SweepRunner for MBR. See "Decisions, 2026-09-24" below: one runner (CharacterizationRunner absorbs BatchRunner), and the aggregate class does the assembly.

Decisions
	- What does one job/HDF5 file contain? Each job-generated file is always a 1D array of complex numbers after processing
		○ EITHER final Fock state OR time steps
		○ Initial pi/2 phase
		○ Final pi/2 phase
	- How do we deal with dimensions higher than SweepRunner (eg disorder times time times occupation):
		○ Say I want a disorder-ensemble experiment object (sweep over realization * Fock states * time), how do construct such an experiment object whose analyze function does ensemble average and the display function plots the level statistics? 
			§ Acquire data: no longer achieved by the acquire() method of this class, but rather just use for-loops in notebook cells. Humans should manually make sure that while taking data, we also generate a manifest of this collection of data in yaml (because it allows us to add comments/annotations manually). These yamls should live C:\experiments\[name]\assembled_data (updated 2026-09-24, see below) and contain all necessary metadata (job IDs, HDF5 filenames, etc). It should be built and written/read by the Experiment object representing that level of data.
			§ Load existing data: a new constructor that reads the manifest yaml and pulls in all the HDF5 files. 


# Migration step

To be implemented. Eg walk over all old HDF5s and save them to C:\experiments\[name]\converted_data

Backwards compatibility: do not leave behind shims that redirect old methods. Just migrate the consumers as well.

Migration code should live in a migration script/module, eg under tools/, not in new classes

One obvious gap is that old data stored the 2*2 phase combos in two HDF5 files (separating the final pi/2 0 vs 90deg phase or in other words real and imag quadratures of the complex matel). This is a global schema migration on both acquisition and analysis/display.

Another obvious issue is old jobs recorded EncodingHamiltonianSpectroscopyExperiment (the original god class) as basically the class for every experiment/job/data file and behavior is routed via keyword flag selections inside the giant methods of that class. Now that we are moving towards proper OOPs instead of switchboard functions/classes, this will change (for the better).

Two more layouts to migrate (from the 2026-09-24 decisions below):
- Old chunked time traces (one trace split over several jobs) are joined into one TimeTrace file.
- Old propagator jobs (`propagator_batch`: one encoder × all cycles × all decoders in one file) are split along the axis that the new HamTomo child uses.

# Decisions, 2026-09-24

## One runner

- BatchRunner is retired. Its function merges into `CharacterizationRunner`: `execute(configs=None, batch_size=..., **kwargs)`.
  - No `configs`: one job, as today.
  - `configs` given (a list of override dicts): one job per dict, bounded queue submission, returns a plain list of per-job Experiments.
- Both modes (direct dispatch and job queue) and both instrument types (real and mock) reuse the existing `run_local`/`run` paths. No second copy of submit/wait/load.
- One mock + queue policy for both paths. Prefer BatchRunner's current behavior: raise unless explicitly allowed.
- `BatchRunner._aggregate` (a placeholder ExptClass instance that only holds `batch_expts`) and `from_batch` are deleted. Consumers migrate (no shims).
- SweepRunner stays as it is for simple one-scalar sweeps where parent and child have the same class. MBR does not use it.

## Aggregate Experiments

- The aggregate class does the assembly, not the runner and not the notebook. The dependencies point down: the aggregate knows its child class and uses a runner; the child class knows nothing above it; the runner knows neither.
- Aggregates keep the acquire -> analyze -> display (-> save) lifecycle with the same method names, but `acquire` takes a runner, because acquisition means submitting N child jobs from the notebook process. Only child classes go to the worker.
- Aggregates are not slab `Experiment` subclasses (they never need soccfg, config_file or an instrument manager). Add a small shared base only when a second aggregate needs it.
- The 2D/3D line:
  - `acquire(runner)`: when acquisition is one continuous unit of work, that is one `execute` call (Spectrum, CalibrationSet, Orthogonality).
  - `from_parts(...)`: when acquisition continues over hours or days with human decisions in between (HamTomo, DisorderEnsemble). The notebook loops over the 2D level.
- Sketch:

```python
spectrum = MBRSpectrumExperiment(occupations=..., cycles=..., calibration=cal)
spectrum.acquire(runner, batch_size=10)   # configs -> runner.execute -> assemble
spectrum.save()                           # assembled HDF5 + manifest YAML
spectrum.analyze(); spectrum.display()

spectrum = MBRSpectrumExperiment.from_manifest(path)   # reload: re-assemble from raw job files
```

## Assembled data

- Each aggregate writes two derived files to `C:\experiments\[name]\assembled_data`:
  - A manifest YAML: class name, job IDs, raw HDF5 paths, calibration reference, human notes.
  - An assembled HDF5: the assembled arrays (complex where possible), with the manifest path and code version as provenance.
- The manifest is the source of truth. `from_manifest` re-assembles from the raw job files. The assembled HDF5 is a convenience copy: rebuild it, never edit it.
- Raw job HDF5s are never changed.

## Calibration provenance

- Each TimeTrace (and HamTomo child) job records the path of the CalibrationSet manifest it took its phase correction from, in `cfg.expt`, beside the correction value itself.

# Unresolved

- Existing MBRPhaseCorrectionExperiment is in fact an *aggregate* of an entire calibration dataset, not the 1D time trace calibration sweep per policy above.
  - proposed solution: we make two calibration Experiment classes. MBRStarkCalExperiment stays the 1D job-level sibling to MBRTimeTraceExperiment. We add a new MBRCalibrationSetExperiment as the sibling to the MBRSpectrumExperiment which acquires/displays an entire calibration set and provides easy phase correction lookups for TimeTrace/Spectrum experiments.
- Currently there is another abstraction level in the existing code base: one can decide to chop up a large time-sweep axis into chunks. Do we retain this capability?
  - proposed solution: no for now. This feature was added potentially because sweeping over many time steps might make one job too long. But this is a human coordination/experiment run time control issue, the right fix is not necessarily introduce more complexity at the acquisition code level.
- the HamTomo description above was not clear enough. To clarify, this is a highest-dimensional dataset (same dimensionality as disorder ensemble) which also requires assembly inside notebook cells. One other gap to close is that for TimeTrace to be able to support Spectrum and HamTomo experiments, we need to allow TimeTrace to take init fock != final fock settings.
  - proposed solution: either A) add a thin intermediate in between TimeTrace and Spectrum eg called DiagTimeTrace The general TimeTrace then can accept off diagonal elements in the init-final matrix. or B) simply make TimeTrace be able to take care of both diag and off-diag cases. Choose whichever is easier/cleaner at actual refactoring time. As for the init * final * time assembly mechanism: given that the closest match in existing code is this EncodingPropagatorProgram which is essentially a generalized Orthogonality experiment (not just time=0, but also two more small steps at time=q, 2q), it might be better to build this object from that instead of TimeTrace. 
- We haven't talked about the names of the Programs. In general, we'd like to stick to the MBR (many-body ramsey) family of names now that the Experiments follow this convention. Old names often contain the words Encoding (referring to the "encoding"/state prep pulses that transfer the superposition from qubit to cavities) and Spectroscopy. I personally find this less illuminating as names because Encoding/Decoding sort of suggest two-way mappings which we don't do (qubit |e> is always the source which straightforwardly gets mapped to the target |N> state in each experiment so there's not per se a code to be found in this procedure), and using half of the procedure to refer to the thing is also less clear than Ramsey which refers to the whole thing. Spectroscopy is also a bit too specific: as we see Spectrum is just one out of the family of experiments. I haven't mapped the entire renaming surface but just a note that we shouldn't stick to bad names.

