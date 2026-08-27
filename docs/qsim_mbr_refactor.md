# Qsim many-body Ramsey and Floquet/dark-mode refactor specification

**experiments/qsim/floquet_dark_mode_readout.py** was an 8,272-line
mixed-responsibility module containing 31 classes. Its two largest classes,
**EncodingHamiltonianSpectroscopyExperiment** and **DarkBaseProgram**, account
for approximately 67% of the file, but the refactor covers the entire module
rather than only those two classes.

It is now 5,665 lines: the numerical spine moved to **fitting/qsim/** and seven
whole measurement families moved to their own modules (section 7.6). What
remains is the dark-mode pulse base, the MBR acquisition programs, the legacy
scramble variants, **BatchRunner**, and the god Experiment.

This specification defines:

- the target ownership and module boundaries;
- the Experiment and Program classes required for each physical measurement;
- the separation between reusable pulse construction, experiment workflow, and
  numerical analysis;
- raw and derived HDF5 data contracts;
- compatibility requirements for existing jobs, notebooks, and serialized
  data; and
- a behavior-preserving migration order and acceptance criteria.

The target retains the repository's established Experiment workflow:
acquisition, analysis, persistence, and display remain associated with the
Experiment representing the physical measurement or derived result. Reusable
numerical work may be delegated to importable analysis modules.

Findings are based on a source audit and reproduction of the August 15–17,
2026 N=3 and disorder analysis. That dataset becomes a characterization
fixture for the migration; it is evidence for current behavior, not the sole
definition of correct behavior.

Treat this spec as guidance that describes the general shape of desired behavior
and not a hard rule to stick to, as there might be details on the ground that
affect implementation that this coarse sweep did not surface.

New sessions: read section 0 (why this refactor exists, and what counts as
done) first, then section 13 (where the work runs), appendix B (what is
verified, not assumed), and appendix C (configuration traps). The last three
come from re-running the reference path against live data.

## 0. Purpose: correctness, and how it is established

### 0.1 The deliverable

The deliverable of this refactor is physics that can be trusted and published.
Every structural goal below is a means to that, not an end. When a structural
change and a correctness gain conflict, the correctness gain wins.

This has to be stated because the alternative reading has already cost us. The
extraction in **217c1eb** met every acceptance criterion this spec had: the
lines moved verbatim, the golden baseline passed, no behavior changed. The code
it moved was wrong, before and after -- see the LDOS defect recorded in section
0.4. The baseline did not merely miss the defect, it *pinned* it.

So "behavior-preserving" is a check on the *move*, not a claim about the
*physics*. Both are needed, and they must not be confused for one another.

### 0.2 Watertight abstraction is what makes verification possible

We call `np.fft.ifft` without reading its source. That is not faith. It rests on
a boundary: a named contract, a stable meaning, and evidence behind it. Because
the boundary holds, the function can be used without being re-examined.

A component that can be trusted without being read is simultaneously the unit of
reuse and the unit of verification. It is the same boundary serving both. This is
the whole reason the structural work matters:

- Tangled code has no such boundaries, so it offers nothing to verify. The only
  check available for a 6,000-line module is "does the whole thing still produce
  the same numbers" -- which is exactly the check that certified the LDOS defect.
- Every boundary this spec draws is therefore a thing that can be verified once
  and afterwards assumed. That, and not tidiness, is why locality and module
  ownership appear in the goals.

Watertightness is the property to aim for. A boundary that leaks -- a shared
mutable intermediate, a quantity whose meaning depends on its caller, a function
that needs its consumers read to be understood -- is not a component, whatever
file it lives in.

### 0.3 The inspectability target

One physical measurement should be one file that can be read in a sitting: on
the order of 200 lines, holding the content specific to that measurement plus
named calls into layers whose correctness is already established.

The second half is what makes the first half honest. A line budget met by moving
work into imports buys nothing; it is only legitimate when the callee is itself
verified. Locality and hierarchy are the same requirement seen from two sides.

### 0.4 The hierarchy, and why it needs a ledger

Correctness is established per layer and assumed by callers. Verify the
multi-photon-manipulation functions, and their callers may then assume them.

This works only if what was established is recorded, *and against which
definition*. The August 2026 LDOS defect is the failure mode:
`eigenstate_weights` was correct and inspected as the probability
`|<b|E_k>|^2`; **93c1b20** then redefined it in place as the signed amplitude
product `|<f|E_k><E_k|b>|`, while every consumer went on treating it as the old
probability. Nothing failed, because the two agree for diagonal rows and all
data was diagonal at the time. The established status silently became false.

The rule that follows: changing the meaning of a shared quantity invalidates the
established status of everything downstream, even when the name, shape and dtype
are unchanged. Redefinition in place is a contract break and must be treated as
one.

### 0.5 Ways to establish correctness

The goal is correctness. The following are tools that have done real work in this
codebase -- **examples, not an exhaustive list and not a checklist**. Reach for
whatever exposes the specific way a thing could be wrong, and expect to invent
modes not named here.

- **Equivalence and characterization.** Identical output to the pre-refactor
  path: identical assembly from a mock station fixture, golden analysis
  baselines. Cheap, runs anywhere, and the right tool for proving a move changed
  nothing. It answers "did I change this", never "is this right".
- **Invariance and other properties.** Assert what must leave a quantity
  unchanged: eigenvector gauge, units, permutation of identical modes, time
  translation, normalization. This is what exposed the LDOS defect. It needs no
  device, no data and no known answer, and it does not go stale.
- **Synthetic ground truth.** Feed data whose correct answer is known, and also
  data that is known to be bad, then require the analysis to respond correctly to
  both. Rejecting what should be rejected is part of the contract.
- **On-device interrogation.** Physics correctness ultimately requires the
  device: change something, predict the response, check it. Nothing above
  substitutes for this, and it is the scarcest resource, which is a reason to
  let the cheaper modes carry everything they can.

The question to ask of any component is not "which of these did I run" but "what
would expose this being wrong, and have I done that".

### 0.6 Current practice is the starting specification

Acquisition and analysis both currently live in **measurement_notebooks/**, not
in the sandbox paths this spec designates. Those notebooks gesture at an intended
design and then implement by the path of least resistance and accumulated debt.
They are the authority on how the work is done today.

The concrete acceptance test for this refactor is therefore: from the refactored
entry point, reproduce what those notebooks do. Start a new measurement with a
few lines of config adjustment and standard job submission, and inspect the
result through the canonical analysis API.

As with the August dataset, they are evidence of current behavior, not a
definition of correct behavior. Reproducing them is necessary and not sufficient.

## 1. Scope

### 1.1 Goals

These are the means to section 0. Each exists because it creates a boundary
that can be verified once and then assumed.

- Replace the stage-directed god Experiment with Experiment types describing
  concrete acquired measurements and concrete derived results.
- Make HDF5 the canonical scientific input and output format.
- Remove live station, FastAPI, job-database, and implicit pickle dependencies
  from offline MBR analysis.
- Extract the narrow Floquet pulse-sequence behavior genuinely shared by MBR
  and dark-mode measurements.
- Move reusable numerical analysis into importable modules.
- Preserve the normal runner lifecycle: acquire, analyze, save, display, and
  vault logging.
- Give every supported Experiment and Program a globally unique, descriptive
  class name.
- Permit a clean parallel implementation without forcing an immediate rewrite
  of the legacy acquisition notebook.

### 1.2 Non-goals

- This refactor does not redesign the repository-wide Experiment API.
- It does not redesign the flattened **experiments** export mechanism.
- It does not implement the repository-wide stamping of configuration version
  IDs into HDF5. That is an upstream station/worker/Experiment persistence
  change whose output this refactor consumes.
- It does not specify the exact steps taken to deprecate the superseded legacy
  classes and top-level acquisition notebooks. Those are decisions to be made
  while implementing this spec based on ground level details.
- It does not require a repository-wide typed-configuration conversion.
- It does not require every intermediate aggregate to be written to disk, only
  that every aggregate Experiment can save and reload a reproducible result.

## 2. Local correctness and robustness fixes

The following defects were discovered while reproducing the MBR workflow. They
vary in severity but are all narrow, independently actionable fixes. They do
not determine the target architecture and must not be used to justify moving
unrelated code. Each fix gets a focused regression test.

### 2.1 Preserve unscaled numerical results

Spectrum analysis currently rescales theoretical spectra so their summed peak
matches the measured peak. The original audit correctly identified the silent
mutation but overclaimed that equal time-zero normalization requires equal FFT
peaks. Diagonal measured traces are normalized at time zero; off-diagonal
matrix elements are intentionally not, and finite-window FFT maxima need not
match even for equally normalized time traces.

Required behavior:

- analysis returns measured and theoretical spectra using their physically
  defined normalization without fitting one to the other;
- any peak matching is an explicit display option operating on a copy;
- the display records the applied scale factor and normalization mode;
- display normalization never replaces saved unscaled arrays; and
- tests cover diagonal and off-diagonal matrix elements separately.

The audit must cover both aggregate scaling sites in the current class, not
only the first occurrence.

### 2.2 Never substitute current station timing for historical timing

The current HDF5 fallback reconstructs Floquet timing from today's station
configuration. For the audited August data, this selected different
**m1s_pi_fracs** and produced a substantially wrong cycle time without an
error. Confirmed: between **CFG-FL-20260814-00076** and
**CFG-FL-20260825-00076** the swap dataset moved `gauss_sigma` 0.04 → 0.02 us,
so today's station gives roughly half the historical cycle time.

Required behavior:

- ordinary offline analysis has no station argument;
- historical timing is resolved from the configuration-version references
  described in section 3; and
- failure to resolve an unambiguous historical source is an error.

**Historical timing is fully reconstructible**, from four already-persisted
inputs:

- the versioned Floquet swap CSV — all of `retrieve_swap_parameters`;
- the versioned hardware YAML — `ramp_sigma`, for the `flat_top` branch only;
- committed **configs/soccfg_snapshot.json**, as a real `QickConfig`, for
  `us2cycles`/`cycles2us`; and
- the HDF5's embedded `expt` config — `swap_stors`, `scramble_sync_cycles`,
  and any waveform override.

Verified for `JOB-20260815-00009`: `floquet_cycle_us == 0.7340315934065934`
and `m1s_pi_fracs == [40] * 7`, equal to the values recovered from the pickle.
The pickled program computed the same quantity from the same configs, so it was
a cache, not an independent measurement.

Couplings follow: `couplings_MHz = 1 / (4 * pi_fracs * floquet_cycle_us)`. One
`resolve_floquet_hardware(version_ids, expt_cfg, soccfg)` replaces **both**
branches of the current `_saved_parameters` dispatch. No legacy timing
descriptor is needed; sections 9.1 and 11.1 are corrected.

Station remains a valid acquisition dependency. It is not an aggregate or
numerical-analysis dependency. Section 13 makes the same rule an isolation
guarantee.

### 2.3 Validate aggregate compatibility

The current **_saved_parameters** documentation says it checks sister
experiments, but it primarily adopts the first configuration and the first
available saved Program. Confirmed by inspection: the per-expt loop
`break`s on the first source that carries a usable `prog`, and no
cross-source comparison happens anywhere in the method.

Before reconstruction, every aggregate must compare all invariants not
intentionally varied by that aggregate. Depending on the Experiment, these
include:

- mode and storage ordering;
- photon-number sector;
- detunings;
- pulse and Floquet timing definition;
- preparation and analyzer conventions;
- readout-calibration interpretation;
- phase-application sign;
- schema and configuration versions; and
- coordinate-grid meaning.

Errors identify the offending source files and fields instead of silently
using the first file's values.

### 2.4 Validate sampling grids and state coverage

Every reconstructed matrix element must have:

- complete real and imaginary analyzer information;
- non-overlapping cycle chunks;
- the required identical sorted cycle grid;
- a declared initial and final occupation;
- no duplicate state contribution; and
- no missing or ragged row.

Paired arrays and metadata sequences have their lengths and shapes checked
before combination. Truncating iteration is never used as validation.

### 2.5 Keep aggregate reconstruction non-mutating

Leaf Experiment analysis may cache analyzed leaf data on itself. Aggregate
reconstruction helpers treat loaded sources as immutable. Reusing a leaf file
in two analyses must produce results independent of call order.

In particular, quadrature extraction must not add **Pe** or
**return_quadrature** to a source object as a side effect of aggregate
construction.

### 2.6 Give collection inputs an explicit contract

A filename, Path, job ID, Experiment, or source descriptor is one input.
Supported nested collections may be flattened without iterating strings or
paths. Unsupported input types fail at the public boundary with a descriptive
error.

This replaces the current **flatten_exp_lists** behavior that shreds strings
and rejects Path scalars.

### 2.7 Select file formats explicitly

- **.h5** and **.hdf5** are canonical scientific inputs.
- **.pkl** are for ephemeral debugging and shouldn't be invoked outside
  scratch pads.
- Missing files and unsupported suffixes fail before deserialization.
- In cases where HDF5 metadata are incomplete, it is permissible to
  manually supply missing info in top level analysis notebooks as
  long as the provenance is annotated ("we forgot to persist this value
  to hdf5 so did a DB query, a pkl check etc to find out"). In such cases
  a TODO must be be added to fix this in future HDF5 writers ASAP.

## 3. Configuration and data provenance contract

### 3.1 Ownership and scope

Stamping configuration version IDs into HDF5 is owned by the common
station/worker/Experiment persistence layer, not by this MBR refactor.

This refactor defines and consumes the resulting contract. It removes the
current MBR class's FastAPI, job-database, current-station, and implicit-pickle
fallbacks.

**The contract is already satisfied — in the wrong file.** `jobs.db` table
`jobs` records per job: all four `*_version_id` fields,
`experiment_class`/`module`, `program_class`/`module`, `data_file_path`,
`expt_pickle_path`, timestamps, and a `station_config` snapshot. A saved HDF5
file carries **exactly one attribute, `config`** — no job ID, no class name, no
version IDs, no timestamp, no code revision.

Two consequences:

- Stamping is mechanical, not research. The worker already computes the
  version-ID dict; the HDF5 write is one choke point in `slab/experiment.py`
  (the two `attrs['config'] = json.dumps(...)` sites).
- Until it lands, `from_h5file()` cannot tell which Experiment wrote a file, so
  criterion 11.3 is blocked on stamping, not on this refactor.

The problem is location, not redundancy. A live database on the acquisition
workstation blocks offline analysis and cannot be published. Portable derived
artifacts (section 3.3) are the fix, hence their phase-2 priority.

### 3.2 Raw measurement provenance

Reconstructing a raw measurement requires only:

- the experiment configuration stored in its HDF5 file;
- the four globally unique configuration snapshot IDs used for the job;
- the immutable versioned files identified by those IDs; and
- the identified Program/Experiment implementation or a compatible revision.

The four references are:

- hardware configuration;
- multiphoton configuration;
- Floquet storage-swap configuration; and
- manipulate storage-swap configuration.

The embedded experiment configuration is authoritative for parameters
constructed or modified in memory for that acquisition. The versioned files
provide the hardware and dataset-backed configuration omitted from HDF5
serialization.

Configuration IDs resolve against the file-based **configs/versions/** archive.
Offline scientific analysis does not query **jobs.db** or the FastAPI server.
A read-only share or synchronized copy may expose the archive away from the
production workstation. The archive location is deployment configuration; its
immutable contents and unique IDs form the provenance contract.

Archive reads open `configs/versions/<type>/<ID>` as plain files. They do not
go through `ConfigVersionManager` or `get_database()`, which reach the live
database and can mutate live rig state.

#### Historical data: no raw-file mutation

Existing raw HDF5 files predate the stamp and **are never modified**. The risk
of damaging them outweighs the tidiness; derived artifacts are cheap to
regenerate, raw files are not.

Pre-stamp references come from a **one-time, read-only export** of `jobs.db`
(`job_id` to the four version IDs, class names, data path). The aggregate
manifest then carries them, marked as injected. This is the case section 2.7
already licenses, applied systematically instead of ad hoc.

Use a read-only connection — the live worker holds the same database open in
WAL mode:

~~~python
sqlite3.connect("file:///C:/.../job_server/jobs.db?mode=ro", uri=True)
# writes raise OperationalError: attempt to write a readonly database
~~~

After the export, offline analysis reads only HDF5 files plus the archive, and
the prohibition above holds unconditionally. Stamped data needs no export.

The HDF5 file also identifies, when available:

- Experiment and Program class names;
- job or run identifier;
- acquisition timestamp;
- data schema version; and
- source-code revision.

Resolved timing values may be cached in HDF5 as a consistency check, but the
configuration snapshots remain the authoritative source.

### 3.3 Derived and aggregate provenance

An aggregate Experiment may have no hardware **acquire()** and may implement
its own **save_data()**. Its derived HDF5 result records:

- aggregate Experiment class and schema version;
- every leaf source's HDF5 path or portable source identifier;
- leaf job IDs, Experiment types, and stable fingerprints where available;
- leaf configuration version IDs;
- calibration and timing sources;
- all processing parameters and interactive decisions;
- phase-frame, analyzer-sign, and branch conventions;
- windowing, zero-padding, normalization, fit-window, filtering, and exclusion
  choices;
- source-code and analysis implementation revisions;
- reconstructed intermediate quantities needed to audit the result; and
- final derived quantities.

A result rebuilt from the same sources and recorded processing description
must agree within declared numerical tolerances.

Aggregate provenance is recursive: a spectrum may cite saved matrix-element
artifacts or matrix-element objects built in memory, but its saved provenance
must ultimately identify the raw leaf HDF5 files.

The record also separates values read from the raw files from values
**injected** during aggregation — version IDs from the database export,
reconstructed timing, manual entries — each labelled with how it was obtained.
An injected value is never presented as raw-file provenance.

### 3.4 Derived artifact location and lifecycle

Derived artifacts go to `{output_root}/{experiment_name}/processed_data/`,
beside the existing `data/`, `expt_objs/`, `logs/` and `plots/`. No such
directory exists yet. Its path is a plain function of
`(output_root, experiment_name)`, not a station attribute, so the aggregate
track can compute it without a station (section 13).

Derived artifacts are **cheap to re-derive and safe to overwrite or export**.
Hence the asymmetry: raw files are immutable, and everything expensive to
reconstruct by hand — source manifest, injected provenance, analysis
parameters, interactive decisions — lives in the derived file.

### 3.5 Vault logging

Vault entries are an index and presentation layer, not the primary provenance
record. A vault entry links to the persisted raw or aggregate HDF5 artifact and
summarizes its identifying metadata. Scientific reconstruction does not depend
on parsing a vault note.

## 4. Floquet sequence semantics

The shared pulse implementation and all consumers use these terms
consistently:

- A **palindromic cycle** reverses storage-mode traversal within a cycle to
  implement the selected Trotter symmetrization.
- An **adjoint cycle** reverses the relevant pulse order and phase convention
  to implement the inverse or echo portion of a sequence.
- A **closed cycle pair** is the defined forward/adjoint construction. It is
  not described merely as a reversed cycle.
- A **physical cycle** is one application of the physical Floquet evolution.
  It is distinct from a closed-pair count or an acquisition index.
- A **phase branch** selects one representative of the phase slope left
  ambiguous by the forward/adjoint measurement.

The cycle-phase calibration Experiment documents the physical origin, units,
and Hamiltonian-frame effect of branch selection.

Tests of the shared sequence layer verify pulse ordering and phase-frame
advancement for ordinary, palindromic, adjoint, closed-pair, and fractional
paths.

## 5. Legacy and compatibility policy

All support and deletion decisions remain pending an audit of the top-level
data-acquisition notebook and ground-level behavior on the experiment.
Static source-reference counts are evidence, not liveness decisions.

The refactor may choose either of two strategies:

1. incrementally redirect the existing module and acquisition notebook; or
2. leave both frozen and build new modules plus a minimal new acquisition
   consumer alongside them.

The second, parallel path is a first-class option. The old module and consumer
may remain a frozen reference island until the new path succeeds on saved data
and hardware. Only then is a decision made to migrate the old consumer or
retire the old island as a unit.

During migration, an old symbol may be:

- unchanged in the frozen legacy module;
- a compatibility alias to a one-to-one replacement; or
- a compatibility adapter when the old class combined several new types.

**EncodingHamiltonianSpectroscopyExperiment** requires an adapter rather than
a simple alias because its stage argument currently selects unrelated
analysis products.

No legacy implementation is copied into two independently evolving versions.
No class is deleted merely because a repository search finds no caller.

The existing **floquet_dark_mode_readout.py** eventually becomes either:

- a compatibility façade with no independent implementation; or
- part of the retired legacy island.

That decision is deferred until acquisition validation.

## 6. Naming

The repository dynamically flattens discovered classes into the top-level
**experiments** namespace. Duplicate names are silently overwritten in
filesystem iteration order. Therefore every public Experiment and Program
class introduced by this refactor must be unique across the entire
**experiments** package.

The naming convention is:

~~~
<Domain or project><Measured quantity or operation><Program|Experiment>
~~~

Names describe physical behavior or the derived result, never implementation
chronology.

Forbidden in supported target names:

- **New**, **NewNew**, **newold**, or **Modified**;
- **Debug** or **old** as a scientific identity;
- **BaseR** or other inheritance-history labels; and
- **Encoding** when encoding is only a state-preparation detail.

Program and Experiment partners share the same descriptive prefix. Generic
helpers do not acquire Program or Experiment suffixes.

Candidate target names include:

| Responsibility | Candidate target name |
|---|---|
| Shared Floquet pulse primitives | **QsimFloquetSequenceProgram** |
| One MBR analyzer quadrature | **MBRQuadratureProgram/Experiment** |
| One MBR Stark-phase calibration sweep | **MBRStarkShiftCalProgram/Experiment** |
| One entire-cycle phase sweep | **MBRCyclePhaseCalProgram/Experiment** |
| One orthogonality-matrix column | **MBROrthogonalityColumnProgram/Experiment** |
| One propagator column | **MBRPropagatorColumnProgram/Experiment** |
| Derived phase-correction map | **MBRPhaseCorrectionExperiment** |
| Derived complex matrix element | **MBRMatrixElementExperiment** |
| Derived orthogonality matrix | **MBROrthogonalityExperiment** |
| Derived propagator matrix | **MBRPropagatorExperiment** |
| Derived spectrum | **MBRSpectrumExperiment** |
| Derived ensemble statistics | **MBRLevelStatisticsExperiment** |
| Dark-mode readout | **DarkModeReadoutProgram/Experiment** |
| Dark-mode T1 | **DarkModeT1Program/Experiment** |
| Dark-mode multiparity chevron | **DarkModeMultiparityChevronProgram/Experiment** |
| Storage-swap phase calibration | **StorageSwapPhaseCalProgram/Experiment** |
| Floquet displacement Kerr | **FloquetDisplacementKerrProgram/Experiment** |

These names are finalized only after checking what each pulse sequence
actually measures. For example, the current **FloquetPhaseAccumulationProgram**
measures a closed N=1 access path; its supported name should say that.

Modules avoid binding unrelated imported class objects in their namespace
where practical, because the current flattened exporter also discovers
imported classes. Changing that exporter is a separate repository-wide task;
this refactor works safely under its current behavior.

## 7. Target architecture

The target separates:

1. shared pulse-sequence primitives;
2. concrete acquired measurements;
3. derived or aggregate Experiments; and
4. reusable numerical analysis.

Public Experiment classes retain ownership of workflow-level **analyze()**,
**save_data()**, and **display()** behavior. They may delegate numerical work
but do not delegate their scientific identity to a stage flag or notebook.

### 7.1 Dependency direction

~~~
QsimBaseProgram
    └── QsimFloquetSequenceProgram
            ├── MBR leaf Programs
            ├── dark-mode Programs
            └── other Floquet measurement Programs

QsimBaseExperiment
    ├── acquired leaf Experiments ──save──> raw HDF5
    └── aggregate Experiments ──────save──> derived HDF5
             │
             └── fitting/qsim numerical routines

raw HDF5
    → matrix elements and calibration results
    → spectra, orthogonality matrices, or propagators
    → ensemble statistics
~~~

### 7.2 Shared pulse-sequence layer

Proposed module:

~~~
experiments/qsim/floquet_sequence.py
    QsimFloquetSequenceProgram
~~~

This class owns only primitives shared by multiple physical measurements:

- resolving Floquet timing and swap parameters;
- tracking logical phase offsets;
- playing ordered and palindromic cycles;
- playing adjoint and closed-cycle pairs;
- playing fractional M1-storage trains; and
- updating storage and decoder phase frames.

It does not own:

- a generic acquisition body;
- dark-mode state selection;
- multiparity readout;
- reset-policy dispatch;
- MBR reconstruction;
- plotting;
- stage-directed analysis; or
- large-dark configuration branches.

Every concrete Program supplies its own **initialize()** and **body()** around
the shared primitives.

There is no target **dark_base.py** containing the current god base, and no
target **dark_large_support.py** merely grouping methods by a shared name.

### 7.3 Acquired MBR measurements

Candidate modules:

~~~
experiments/qsim/
    mbr_quadrature.py
    mbr_stark_shift_cal.py
    mbr_cycle_phase_cal.py
    mbr_orthogonality.py
    mbr_propagator.py
~~~

Each contains the Program and Experiment representing one hardware
measurement.

| Acquired unit | Candidate leaf classes |
|---|---|
| One MBR return quadrature | **MBRQuadratureProgram/Experiment** |
| One occupation/storage Stark-phase sweep | **MBRStarkShiftCalProgram/Experiment** |
| One occupation's closed-cycle phase sweep | **MBRCyclePhaseCalProgram/Experiment** |
| One encoded overlap-matrix column | **MBROrthogonalityColumnProgram/Experiment** |
| One raw propagator column | **MBRPropagatorColumnProgram/Experiment** |

Each leaf Experiment:

- acquires one documented physical measurement;
- validates its own configuration schema;
- performs the corresponding immediate analysis;
- displays enough information to judge acquisition quality; and
- saves a raw HDF5 artifact usable without its pickle.

There is no stage argument selecting unrelated analyses.

### 7.4 Aggregate MBR Experiments

Candidate modules:

~~~
experiments/qsim/
    mbr_phase_correction.py
    mbr_matrix_element.py
    mbr_orthogonality.py
    mbr_propagator.py
    mbr_spectrum.py
    mbr_level_statistics.py
~~~

Candidate classes:

- **MBRPhaseCorrectionExperiment**
- **MBRMatrixElementExperiment**
- **MBROrthogonalityExperiment**
- **MBRPropagatorExperiment**
- **MBRSpectrumExperiment**
- **MBRLevelStatisticsExperiment**

These may have no Program and no hardware **acquire()**. They accept source
HDF5 files or already-loaded matching Experiment results, validate
compatibility, run analysis, display the result, and save a derived HDF5 file
with recursive provenance.

No general aggregate base class is introduced initially. Common machinery is
extracted only after the first concrete aggregate types demonstrate stable
duplication.

### 7.5 Reusable numerical analysis

Proposed modules:

~~~
fitting/qsim/
    mbr_reconstruction.py
    mbr_phase.py
    mbr_spectrum.py
    mbr_hamiltonian.py
    matrix_pencil.py
    level_statistics.py
~~~

- **mbr_reconstruction.py** owns quadrature combination, source grouping,
  compatibility validation, grid validation, and matrix reconstruction.
- **mbr_phase.py** owns phase unwrapping, correction construction, and
  phase-frame transformations.
- **mbr_spectrum.py** owns FFT/windowing and spectrum transforms.
- **mbr_hamiltonian.py** owns fixed-photon-number basis construction,
  Hamiltonian construction, diagonalization, and theoretical amplitudes.
- **matrix_pencil.py** owns Matrix Pencil analysis.
- **level_statistics.py** owns spectrum merging, spacing statistics, and
  spectral form factor.

Hamiltonian construction does not remain an authoritative notebook script:
**MBRSpectrumExperiment** is a production consumer. Conversely, generic HDF5
loading does not belong in **fitting/qsim/mbr_io.py**. Concrete Experiment
types own their semantic loading and validation boundaries.

Notebook-only exploratory helpers may stay under **analysis_notebooks/**, but
production Experiments do not import notebook implementations.

### 7.6 Remaining measurement families

The remainder of the old module has these prospective owners:

~~~
experiments/qsim/
    dark_mode_readout.py
    dark_mode_t1.py
    dark_mode_multiparity_chevron.py
    dark_mode_broadband_ge_validation.py
    central_boson_local_return.py
    storage_swap_phase_cal.py
    sideband_stark_shift_cal.py
    floquet_displacement_kerr.py
~~~

This is an ownership map, not authorization to extract or delete these
classes. Exact boundaries and supported variants remain pending the
acquisition-notebook audit.

**Done, as pure moves.** Seven of these modules now exist, each holding one
measurement family's whole acquire/analyze/display triple, verbatim:

| Module | Moved in |
|---|---|
| **dark_mode_t1.py** | **DarkT1Program/Experiment** |
| **dark_mode_multiparity_chevron.py** | **DarkBaseRProgram**, **ManStorMultiparityChevronRProgram/Experiment** |
| **dark_mode_broadband_ge_validation.py** | **BroadbandGeValidationProgram** |
| **floquet_displacement_kerr.py** | **FloquetDisplacementKerrProgram/Experiment** |
| **central_boson_local_return.py** | **CentralBosonLocalReturnProgram/Experiment**, the central-return config validators |
| **storage_swap_phase_cal.py** | **StorageSwapPhaseAccumulationProgram** |
| **sideband_stark_shift_cal.py** | the three **SidebandStarkAmplificationModified** variants |

Nothing was renamed, merged, deleted, or reclassified; the names still violate
section 6 and the variant fates are still pending. Grouping a family's variants
in one file is not a decision about which of them survives.

`tests/test_qsim_measurement_split.py` pins every moved definition to its
pre-split AST, so a later edit cannot be mistaken for part of the move.

Two mechanisms make these moves safe under the flattening exporter:

- Each new module imports the base classes it needs *from*
  **floquet_dark_mode_readout.py**. That is backwards against section 7.1 and
  temporary: the direction inverts when **DarkBaseProgram** decomposes into the
  shared sequence layer (section 7.2).
- **floquet_dark_mode_readout.py** keeps the old attribute addresses alive
  through a module-level `__getattr__` (PEP 562), because the acquisition
  notebooks say `meas.qsim.floquet_dark_mode_readout.<Name>`. It has to be lazy
  -- a top-level re-import would close the cycle above -- and being invisible to
  `inspect.getmembers` is the point: each name is exported to the
  **experiments** namespace by its owning module, exactly once.

**dark_mode_readout.py** is not created yet. Its content is the dark-mode half
of **DarkBaseProgram**/**DarkBaseExperiment**, which is a decomposition rather
than a move and so waits on the phase-4 acquisition audit.

### 7.7 Runner responsibility

The current **BatchRunner** does not belong in a physics module. Its eventual
replacement belongs near **CharacterizationRunner**, with a globally unique
name such as **BatchExperimentRunner**. Depending on detailed sweeps, it may
also be folded into existing runners or become part of a bigger Runner
class refactoring.

The runner may:

- expand configurations into leaf jobs;
- submit and monitor jobs;
- collect resulting HDF5 paths;
- construct the declared aggregate Experiment; and
- invoke its analysis, save, display, and logging workflow.

It does not contain MBR phase correction, reconstruction, fitting, or plotting
logic. Runner extraction occurs only after the new Experiment interfaces are
known.

### 7.8 Display ownership

**The acquire/analyze/display triple is the unit of locality. It is the target
shape, not a legacy accident. Do not propose separating it.**

This has been raised twice in review, so the reasoning is recorded here in
full.

#### The contract

`slab/experiment.py:170` defines the lifecycle every Experiment implements:

~~~
Experiment.go(save, analyze, display, progress)
    data = self.acquire(progress)
    if analyze:  data = self.analyze(data)
    if save:     self.save_data(data)
    if display:  self.display(data)
~~~

`acquire`, `analyze`, `display` and `save_data` are the four stubs on the base
class. One measurement implements all four, in one class, in one file.

#### The contract is actively enforced

The runners call `analyze()` and `display()` after acquisition and capture the
output to file. A measurement whose display lives elsewhere produces no record.

- `experiments/characterization_runner.py:523` calls `expt.go(...)`, then
  `:300` calls `experiment.display(**call_kwargs)` with the kwargs filtered to
  the display signature. `:307` catches and reports a display failure.
- `experiments/sweep_runner.py:230` and `:235` call `mother_expt.analyze()`
  then `mother_expt.display()` for live plotting; `:276`-`:289` repeat that for
  the final analysis and the logged record.

#### The exemplar

`experiments/single_qubit/error_amplification.py` is the shape to copy: one
Program, one Experiment, and the four methods, in about 180 lines. `acquire`
runs the sweep, `analyze` scales by `Ig`/`Ie` and fits, `display` plots the
data with the fitted curve. Reusable mathematics is delegated out
(`fitter.fitgaussian`); the measurement-specific glue stays local.

The result is what inspection should feel like: you open one file, and the
whole story of one measurement — what was pulsed, what was fitted, what was
plotted — is in front of you.

#### What is actually wrong with the god Experiment

Not that display is attached to it. That **one** class carries the triples of
four different measurements (calibration, spectroscopy, orthogonality,
propagator) plus the aggregate analyses, dispatched by a `stage` argument. No
single triple is local, because each is interleaved with three others.

The fix is therefore to **split the class into several Experiments, each
keeping its own triple**, per sections 7.3 and 7.4 — not to split the triple.

#### Ownership after the split

- Leaf acquisition-quality displays belong to leaf Experiments.
- Matrix-element display belongs to **MBRMatrixElementExperiment**.
- Spectrum display belongs to **MBRSpectrumExperiment**.
- Level-statistics and SFF displays belong to
  **MBRLevelStatisticsExperiment**.

Private plotting helpers may be shared, but callers interact through the
owning Experiment's **display()** method.

#### Rejected alternatives

- A standalone display module or display package. It breaks the runner
  contract above and inverts the target shape.
- Splitting the file along an acquisition/analysis line so that the analysis
  half can be read without the Program classes. The same reasoning applies:
  the desired locality is per **measurement**, not per **lifecycle stage**.
  Splitting by module is correct only once the classes are split by
  measurement, at which point each module holds a whole triple.

## 8. MBR data-product hierarchy

Every level is a typed scientific data product with provenance, persistence,
analysis, and display. Numerical transformations are delegated to
**fitting/qsim**, but levels 1–3 remain Experiments rather than bare fitting
objects.

| Level | Data product | Typical sources | Owning Experiment |
|---|---|---|---|
| 0 | Acquired quadrature or calibration sweep | Hardware acquisition | Corresponding leaf Experiment |
| 1 | Complex matrix element or calibrated phase result | Compatible level-0 files | **MBRMatrixElementExperiment** or **MBRPhaseCorrectionExperiment** |
| 2 | Orthogonality matrix, propagator matrix, or spectrum | Level-0/1 results plus calibration and timing provenance | Corresponding aggregate Experiment |
| 3 | Ensemble and level statistics | Multiple compatible spectra | **MBRLevelStatisticsExperiment** |

The current statement that every matrix element is exactly two files is too
rigid. The diagonal path presently uses separate analyzer-phase acquisitions,
while some off-diagonal paths interleave decoder and analyzer settings
differently.

The invariant is:

> A matrix-element aggregate receives complete real and imaginary quadrature
> information with declared analyzer conventions. The acquisition type
> determines source cardinality, which is validated explicitly.

**MBRMatrixElementExperiment** preserves:

- raw real and imaginary quadratures;
- acquired complex amplitude;
- normalized amplitude as a separate result, when requested;
- initial and final occupations;
- common cycle grid;
- source provenance; and
- analyzer and phase-frame conventions.

**MBRSpectrumExperiment** preserves both:

- the acquired reconstruction built directly from measured quadratures; and
- the optionally rephased reconstruction used for spectrum analysis.

Level 3 is a separate type because merged spectra may not share a complex time
grid. Its supported analyses and displays therefore differ from a single
spectrum.

### 8.1 Relation between MBR and its accompanying calibration

The many body Ramsey sequence consists of, from outer to inner layers:

- qubit half-pi pulses;
- qubit pi full-swap pulses that transfer superposition between qubit and cavities
- the many-body Hamiltonian evolution on the cavities

The third (innermost) sequence of pulses exert AC Stark shifts on the other two,
so we need to measure the phase accumulation simply due to that instead of
Hamiltonian evolution alone. This is done by an accompanying experiment replacing
the trotterized forward-forward-forward pulses with forward-backward-forward-backward...
The measured AC Stark accrued phase is then fed into the live MBR experiment
so phase correction is already done at pulse time. However, we should record the
provenance of the accrued phases with a reference to its accompanying calibration file
in each MBR experiment HDF5. This is something to be added if not present yet.

## 9. Implementation strategy

The work divides into offline and acquisition tracks with different risk.

### 9.1 Offline analysis track

This track can proceed from saved HDF5 data without touching the acquisition
notebook:

- characterize the current minimal MBR workflow;
- extract pure reconstruction and analysis functions;
- introduce aggregate Experiment classes;
- implement derived HDF5 persistence and provenance;
- replace server and implicit-pickle loading with typed HDF5 inputs; and
- reproduce the audited datasets and plots.

Existing data lacking configuration-ID stamps resolves its timing through the
section 2.2 resolver, using version IDs from the one-time database export
(section 3.2). No legacy timing descriptor is needed, and nothing falls back to
current station state.

Confirmed executable: the reference analysis
(`analysis_notebooks/guan/MBR_analysis.py` over `JOB-20260815-00009..16`) runs
end to end from HDF5 alone, returns all sixteen analysis products with
`hardware.source == "saved program"`, and reproduces the spectrum figure.
Phase 0 needs no new infrastructure.

Two portability seams belong to this track:

- **Job-ID to path resolution needs the subdirectory, and nothing more.**
  A job's file is `{output_root}/{subdir}/data/{JOB-ID}_*.h5`, where `subdir`
  is the station's `experiment_name` at acquisition time -- so it tracks
  neither the job date nor the project, and cannot be derived from the ID.
  Three ways to obtain it, behind one `resolve_job_paths(ids)`:

  - **Recorded (`provenance`, the default).** The subdirectory is an immutable
    fact about the job, and `tools/export_job_provenance.py` already records it
    in `tests/data/job_provenance.json`. Reading it is a JSON read: no walk, no
    vault, no database, and publishable, since the database will not ship with
    the paper. Jobs acquired since the last export fall through to the glob.
  - **Glob (`index`).** Correct anywhere, instant on the workstation, but the
    walk spans all projects; over SMB it measured ~3 minutes for 148 project
    directories, against 0.18 s for the same eight jobs from the record.
  - **Vault scrape (`vault`).** Retained but **not** a general backend: it only
    knows runs whose acquirer set `station.log_measurements` (default False),
    so entire campaigns are missing from it -- the August 2026 MBR campaign
    among them, on the workstation's own vault copy as well as any synced one.
    Do not rely on it for a dataset you did not personally log.

  Querying the job database for the subdirectory, which the pre-refactor
  loader did, is ruled out on three counts: it dominated load time (about half
  of 40 s for eight files), it contends with the job server for a resource that
  belongs to submission and execution, and it cannot be published.
  This makes the section 12.2 naming follow-up cosmetic.
- **Roots are per-machine.** The configs record absolute Windows data and vault
  roots. Off-workstation readers need environment overrides.

### 9.2 Acquisition track

This track begins only when the top-level acquisition notebook can be audited:

- determine which Programs it actually selects;
- identify runtime configuration conventions invisible to static search;
- choose in-place extraction or parallel replacement;
- extract the minimum shared Floquet sequence layer;
- implement new leaf Program/Experiment pairs;
- validate compiled pulse ordering and phase behavior; and
- run controlled hardware comparisons.

No legacy class or consumer is deleted during the offline track.

### 9.3 Configuration contracts

The refactor does not require a repository-wide dataclass conversion. Every
new leaf Experiment nevertheless documents and validates:

- required keys;
- optional keys and defaults;
- keys intentionally varied by a runner;
- values persisted as acquisition provenance; and
- incompatible combinations.

The current 108-key whole-file switchboard is not carried into the target
classes.

## 10. Migration order

From this section on, everything below is a proposal based on findings during the
scoping round. The implementation details may depend on realities on the ground.

### Phase 0: freeze and characterize the reference path

1. Preserve the current module, acquisition notebook, and audited analysis
   workflow as reference artifacts.
2. Record reconstructed amplitudes, phase corrections, spectra, and plots for
   the August characterization dataset.
3. Distinguish expected changes from defects, especially theory-display
   scaling and validation failures.
4. Add narrow tests for the local defects in section 2.
5. Export the `job_id` to configuration-version mapping once, read-only, from
   the database (section 3.2), so later phases have provenance to carry.

No class movement or deletion occurs in this phase.

Step 2 is confirmed reachable (appendix B). Write the section 2.2 timing
resolver test first: its expected value is already known exactly.

### Phase 1: extract numerical seams

1. Extract reconstruction validation and quadrature combination.
2. Extract phase unwrapping and frame correction.
3. Extract FFT spectrum analysis and fixed-N Hamiltonian construction.
4. Extract Matrix Pencil and level-statistics analysis.
5. Make the old class delegate to the extracted functions where doing so is
   behavior-preserving.

The characterization workflow must continue to run after each extraction.

### Phase 2: introduce aggregate Experiments

1. Implement typed HDF5 source loading without station, FastAPI, job database,
   or implicit pickle access.
2. Add **MBRMatrixElementExperiment** and
   **MBRPhaseCorrectionExperiment**.
3. Add **MBRSpectrumExperiment**, followed by orthogonality, propagator, and
   level-statistics aggregates as required.
4. Implement aggregate **save_data()**, reload, provenance, and display.
5. Replace the scratch analysis workflow with a minimal notebook using the new
   aggregate types.

This phase may complete before any acquisition Program is moved.

### Phase 3: satisfy the upstream provenance contract

The common persistence layer stamps the four configuration IDs and relevant
code identity into new raw HDF5 files. This is a separate implementation
project but an integration dependency for fully automatic offline loading.

Verify that a new raw HDF5 file plus **configs/versions/** resolves timing
without **jobs.db**, FastAPI, station, or pickle.

**Consider bringing this forward.** It is numbered third because it is
externally owned, not because it is large (section 3.1). It is the only thing
blocking criterion 11.3, and landing it early stops the database export from
having to cover future data. It does need a merge to the primary checkout and
a worker restart (section 13.4).

### Phase 4: audit acquisition and choose migration strategy

1. Audit the top-level acquisition notebook and its builders.
2. Record every selected Program, configuration transformation, and runner
   assumption.
3. Decide between in-place extraction and a parallel clean consumer.
4. Freeze that decision in a short acquisition migration note before changing
   pulse code.

### Phase 5: extract shared pulse primitives

1. Characterize the Program methods actually used by new MBR leaf
   measurements.
2. Introduce **QsimFloquetSequenceProgram** with only those shared primitives.
3. Compare compiled pulse/event traces for representative ordinary,
   palindromic, adjoint, closed-pair, and fractional paths.
4. Do not move legacy dark-mode dispatch into the shared class.

### Phase 6: introduce leaf MBR measurements

1. Implement new leaf Program/Experiment pairs.
2. Give each pair a documented configuration contract.
3. Connect them to the selected runner or minimal new acquisition consumer.
4. Verify acquire/analyze/save/display/log behavior.
5. Run controlled hardware comparisons against the reference path.

### Phase 7: address the rest of the legacy module

Only after the MBR replacement is established:

1. inspect the remaining dark-mode, calibration, diagnostic, and Kerr
   consumers;
2. choose supported target classes and compatibility aliases;
3. extract one physical measurement family at a time; and
4. decide whether the legacy island is migrated or retired.

Deletion is the last step, not the first.

## 11. Verification and completion criteria

The criteria below are mostly of the equivalence and characterization kind
(section 0.5): they establish that a move changed nothing. That is necessary and
not sufficient. A component is not done until something has been done that would
expose it being *wrong*, not merely *changed* -- and the established status is
recorded against the definition it was established for (section 0.4).

### 11.1 Offline analysis

- The August characterization dataset loads from HDF5 without FastAPI,
  **jobs.db**, live station state, or implicit pickle loading.
- Existing unstamped data resolves timing through the section 2.2 resolver.
  A regression test asserts the resolver reproduces
  `floquet_cycle_us == 0.7340315934065934` and `m1s_pi_fracs == [40] * 7` for
  `JOB-20260815-00009` from versioned configs alone.
- No offline code path constructs a station, reaches an instrument manager, or
  opens a writable database handle.
- Raw HDF5 files are byte-identical before and after any analysis run.
- Reconstructed raw quadratures and acquired complex amplitudes match the
  characterization baseline where behavior is intended to remain unchanged.
- Intentional correctness changes are documented and tested separately.
- Diagonal and off-diagonal grids, state coverage, and configuration
  compatibility fail clearly when malformed.
- Source leaf objects and arrays are unchanged after aggregate analysis.
- Theory spectra remain unscaled in analysis output.

### 11.2 Persistence

- Every aggregate Experiment can save and reload its result.
- Reloaded results reproduce analysis and display without source pickles.
- A derived artifact reloads and re-displays with no database, no station, no
  vault, and no pickle available. This is the publication-portability test.
- Aggregate HDF5 provenance resolves recursively to the raw leaf files.
- All live processing decisions and calibration references survive the
  round-trip.
- New raw HDF5 files satisfy the external configuration-ID contract once the
  upstream stamping work lands.

### 11.3 Experiment workflow

- Every acquired measurement has a concrete Program/Experiment pair.
- Every derived scientific product has a concrete aggregate Experiment.
- Runners call normal Experiment analysis, save, display, and logging hooks.
- No supported target Experiment selects unrelated behavior with a stage
  string.
- Displays remain callable after **Experiment.from_h5file()** or the
  aggregate's corresponding loader.

### 11.4 Pulse behavior

- Shared Floquet primitives have pulse/event-trace tests.
- Compiled-sequence tests pin the committed **configs/soccfg_snapshot.json**
  explicitly rather than taking whatever soccfg a station hands them, so a
  trace comparison cannot silently depend on which machine ran it (see
  section 13).
- New MBR Programs do not inherit dark-mode measurement dispatch merely to
  obtain shared helpers.
- Representative compiled sequences agree with the intended reference
  behavior.
- Hardware comparisons cover phase sign, cycle counting, branch convention,
  and at least one diagonal and off-diagonal acquisition.

### 11.5 Naming and compatibility

- Every new public Program and Experiment name is globally unique.
- Supported target names contain no implementation-history suffixes.
- Old module/class paths required during migration remain resolvable.
- The legacy acquisition path remains untouched until its audit and explicit
  migration decision.

### 11.6 Structural completion

The refactor is complete when:

- the minimal MBR workflow no longer imports
  **EncodingHamiltonianSpectroscopyExperiment**;
- MBR analysis and display are owned by the new Experiment types;
- MBR leaf Programs depend only on focused shared pulse infrastructure;
- the old god class contains no authoritative MBR implementation; and
- the disposition of the remaining legacy island has been explicitly decided.

Line count alone is not an acceptance criterion. The operative criteria are
cohesive ownership, explicit contracts, testable seams, and absence of
configuration-directed god dispatch.

## 12. Open decisions and external follow-ups

### 12.1 Decisions made during implementation

- Exact final class names after a domain-language review.
- Whether off-diagonal quadratures use a distinct leaf Experiment or a
  declared acquisition mode of **MBRQuadratureExperiment**.
- Whether orthogonality and propagator column acquisitions share any helper
  below their concrete Programs.
- Whether the acquisition migration is in-place or parallel.
- Which Matrix Pencil and level-statistics functionality belongs in the first
  supported MBR replacement.
- The stable fingerprint used for aggregate source files.
- The exact storage representation for nested aggregate provenance.

### 12.2 Upstream or operational follow-ups

- Stamp the four configuration version IDs into raw HDF5 in the common
  persistence layer.
- Expose **configs/versions/** through a read-only share or synchronized local
  copy.
- Record Program/Experiment source revision in new raw HDF5 files.
- Improve data-directory naming so experiment paths do not depend on stale
  station session names. Downgraded to cosmetic by the `resolve_job_paths`
  seam in section 9.1.
- Retire the multiphoton configuration leg of the four-ID contract. Its
  archive holds exactly one file, **CFG-MP-20260121-00001.yml**, which every
  job since January cites, so the reference is inert. It is known dead weight
  slated for removal; the refactor should record it where present but must not
  build logic that depends on it varying.
- Consider a repository-wide change making the flattened **experiments**
  exporter include only locally defined classes and raise on collisions.

These follow-ups do not move into the MBR implementation merely because the
audit exposed them.

## 13. Execution environment

### 13.1 Decision

The refactor runs **on the acquisition workstation, in the `guan` worktree**
(`C:\python\multimode_expts_guan`, branch `guan`).

Working off-workstation was rejected: its only gain is that it cannot touch
live rig state, and the rule in 13.3 buys that more cheaply than losing local
access to the data tree and the archive. Off-workstation stays useful as a
second seat for compiled-trace work, which needs neither.

`main..guan` differs only in `docs/` and notebooks — no code divergence in
`experiments/`, `fitting/`, `slab/`, `job_server/`. So work develops on `guan`
and merges to `main` without conflict risk.

### 13.2 What the worktree shares with the live tree

The worktree isolates version control, not rig state. Two paths link back to
the primary checkout:

~~~
configs/versions   -> <primary>\configs\versions      (directory junction)
job_server/jobs.db -> <primary>\job_server\jobs.db    (symlink)
~~~

They behave coherently, not divergently: a snapshot taken here resolves its
config directory from its own module path, follows the junction, and lands in
the one real archive with its row in the one real database. Nothing forks.

So **the worktree can mutate live rig state as easily as the primary checkout
can.** Reading is safe; writing is not made safe by the worktree. Three
hazards:

- A **non-mock** station here connects to the live instrument manager and,
  because this is the production host, writes the *tracked*
  `configs/soccfg_snapshot.json`. Mock mode does neither.
- **Mock mode on the production host** takes its soccfg from the live proxy,
  not the committed snapshot. Harmless to the hardware, but it makes
  compiled-trace comparison machine-dependent — hence criterion 11.4.
- `MMDataset.create_snapshot()` and the main-version setters write to the
  shared archive and database. They have no place in the offline track.

### 13.3 The isolation rule

> The offline analysis track constructs no station, ever.

Section 2.2 already requires this for correctness. It is also the isolation
guarantee: no station means no instrument manager, no snapshot write, no
config-version mutation, no job submission. With direct archive file reads
(section 3.2) and a read-only database handle for the export, the offline
track is read-only against live state by construction, not by care.

Compiled-sequence work follows the same rule: load
`configs/soccfg_snapshot.json` into a `QickConfig` directly rather than asking
a station.

### 13.4 What requires the primary checkout

Only two things: **hardware job submission** (the worker runs code from the
primary checkout, so new data means merging first) and **landing the stamping
change**.

Phases 0, 1 and 2 run in the worktree, as does phase-5 compiled-trace
validation. Mock mode is confirmed working here
(`tests/test_mock_mode.py` passes; `MultimodeStation(mock=True)` constructs
against the linked archive and database), so old-versus-new Program
equivalence testing does not wait on hardware.

## 14. Working order

Section 10 gives the architectural phases. This is the operational view: what
must happen in order, and what can be taken in any order.

There are only **two real ordering constraints**:

1. The golden characterization test comes before any extraction, or the moves
   have no safety net.
2. Every pure move happens before the first section 2 behavior fix. Extraction
   is free only while the golden stays green; the first re-blessing spends that.

### Invariants

Hold these constant through every step. Each has been proposed for violation at
least once.

1. **The acquire/analyze/display triple stays together on one Experiment.**
   It is the canonical slab shape and the runners enforce it. Reorganize by
   splitting the god class into several Experiments, each keeping its whole
   triple; never by separating analysis from display, or analysis from
   acquisition. Full reasoning, enforcement points and the exemplar file are in
   section 7.8.
2. **The offline track constructs no station.** Sections 2.2 and 13.3.
3. **Reusable mathematics moves to fitting/qsim; measurement-specific glue
   stays on the owning Experiment.** Section 7.5. The test of "reusable" is
   whether the function takes arrays and returns arrays with no knowledge of
   how the data was acquired.

### Spine

1. **Golden test** on current behavior — `tests/test_mbr_analysis_golden.py`.
2. **Pure numerical extractions.** Matrix pencil, level statistics, SFF,
   Hamiltonian and basis construction. Golden green throughout.
3. **Section 2 behavior fixes**, one per commit, each re-blessing the golden
   and carrying its own test for the new behavior.
4. **Aggregate Experiments** and derived persistence (sections 3.3, 3.4).

### Pool

Free to take in any order. None changes numerical output, so the golden stays
green:

- section 2.6 source-collection contract, replacing `flatten_exp_lists`;
- section 2.7 explicit file-format selection;
- the one-time read-only `jobs.db` provenance export (section 3.2);
- the `resolve_job_paths` seam (section 9.1) — **done**, `experiments/job_paths.py`;
- the section 6 naming review;
- splitting whole measurement families out of the god module (section 7.6) --
  **done** for seven of the eight.

### Gated externally

- HDF5 stamping: needs a merge to the primary checkout and a worker restart, so
  it can run in parallel whenever convenient.
- Anything touching pulse code: needs the phase-4 acquisition audit.

## Appendix A. Whole-file ownership inventory

This inventory prevents the refactor from treating the 3,805-line god
Experiment as the whole problem. Target ownership is provisional until the
acquisition notebook audit; no row authorizes deletion.

| Current class | Prospective owner or role | Decision state |
|---|---|---|
| **DarkBaseExperiment** | Legacy dark-mode acquisition façade or decomposed leaf behavior | Pending consumer audit |
| **DarkBaseProgram** | Source of shared Floquet primitives plus legacy dark-mode behavior | Decompose; do not move intact |
| **DarkBaseRProgram** | Dark-mode RAverager infrastructure | Moved to dark_mode_multiparity_chevron.py, its only subclass |
| **ManStorMultiparityChevronRProgram** | Dark-mode multiparity chevron | Moved to dark_mode_multiparity_chevron.py |
| **ManStorMultiparityChevronRExperiment** | Dark-mode multiparity chevron | Moved to dark_mode_multiparity_chevron.py |
| **DarkT1Program** | Dark-mode T1 | Moved to dark_mode_t1.py |
| **DarkT1Experiment** | Dark-mode T1 | Moved to dark_mode_t1.py |
| **SidebandScrambleDarkProgramNewNew** | Legacy dark-mode/scramble bridge used as an MBR base | Remove MBR dependency; legacy fate pending |
| **BroadbandGeValidationProgram** | Dark-mode broadband ge validation | Moved to dark_mode_broadband_ge_validation.py |
| **NPhotonHamiltonianSpectroscopyProgram** | MBR quadrature acquisition | Replace with descriptively named leaf |
| **EncodingOrthogonalityProgram** | MBR orthogonality-column acquisition | Replace with descriptively named leaf |
| **EncodingPropagatorProgram** | MBR propagator-column acquisition | Replace with descriptively named leaf |
| **EncodingStarkShiftCalibrationProgram** | MBR Stark-phase calibration | Replace with descriptively named leaf |
| **EntireFloquetCyclePhaseCalibrationProgram** | MBR cycle-phase calibration | Replace with descriptively named leaf |
| **FloquetDisplacementKerrProgram** | Floquet displacement Kerr | Moved to floquet_displacement_kerr.py |
| **SinglePhotonFloquetSpectroscopyProgram** | Compatibility wrapper for old N=1 name | Compatibility-only candidate |
| **FloquetPhaseAccumulationProgram** | Closed N=1 access-path phase measurement | Rename after domain review |
| **CentralBosonLocalReturnProgram** | Central-boson local return | Moved to central_boson_local_return.py |
| **CentralBosonLocalReturnExperiment** | Central-boson local return | Moved to central_boson_local_return.py |
| **SidebandScrambleDarkProgramNew** | Legacy dark-mode variant | Pending consumer audit |
| **ManStorScrambleProgram** | Legacy scramble variant | Pending consumer audit |
| **SidebandScrambleDarkProgramDebug** | Legacy diagnostic variant | Pending consumer audit |
| **KerrWaitProgramDark** | Legacy Kerr-wait path | Pending consumer audit |
| **SidebandScrambleDarkProgram** | Legacy dark-mode path; also imported elsewhere | Pending consumer audit |
| **SidebandStarkAmplificationModifiedProgram_old** | Live sideband calibration despite suffix | Moved to sideband_stark_shift_cal.py; still needs a descriptive replacement |
| **StorageSwapPhaseAccumulationProgram** | Storage-swap phase calibration | Moved to storage_swap_phase_cal.py |
| **SidebandStarkAmplificationModifiedProgram** | Sideband calibration | Moved to sideband_stark_shift_cal.py; variant comparison still pending |
| **SidebandStarkAmplificationModifiedProgram_newold** | Legacy sideband calibration variant | Moved to sideband_stark_shift_cal.py; consumer audit still pending |
| **BatchRunner** | Generic acquisition orchestration | Move only after new interfaces stabilize |
| **EncodingHamiltonianSpectroscopyExperiment** | Multiple MBR leaf and aggregate workflows | Replace with concrete Experiment types |
| **FloquetDisplacementKerrExperiment** | Floquet displacement Kerr | Moved to floquet_displacement_kerr.py |

Top-level helper functions receive the same treatment:

- **flatten_exp_lists** is replaced by an explicit source-collection contract.
- central-return validation helpers moved with the central-boson measurement
  family. **classify_two_parity_readouts** stayed behind: it is a generic
  two-parity classifier and **DarkBaseExperiment.analyze_multiparity** calls it.
- large configuration-mutating metadata helpers move behind the concrete
  Experiment that owns those parameters.

## Appendix B. Characterization evidence retained from the audit

- The audited HDF5 measurement arrays round-trip identically to their pickle
  counterparts for the checked datasets.
- The embedded experiment configuration was complete for the checked
  experiment-specific keys.
- Floquet dataset objects were omitted by serialization, motivating the
  configuration-version contract.
- Current station timing differed materially from the timing used for the
  audited historical data.
- The existing minimal analysis notebook demonstrates the desired top-level
  shape—select sources, load, analyze, display—but currently needs a fake
  Program object and manual timing because of the missing provenance stamps.

Added after re-running the reference path in the worktree:

- The reference analysis runs end to end from HDF5 alone — all sixteen
  analysis products, `hardware.source == "saved program"`, spectrum figure
  reproduced. No station, database or pickle needed.
- Historical Floquet timing reconstructs bit-for-bit from versioned configs,
  the committed soccfg snapshot and the embedded experiment config.
- All four versioned artifacts cited by `JOB-20260815-00009` are present:
  `CFG-HW-20260814-00074`, `CFG-MP-20260121-00001`, `CFG-FL-20260814-00076`,
  `CFG-M1-20260814-00121`.
- Saved raw HDF5 files carry exactly one attribute, `config`, while `jobs.db`
  records every field the section 3.2 contract asks for.

These observations are regression evidence, not permanent architectural
assumptions.

## Appendix C. Configuration source-of-truth traps

Both cost real time to find, and neither is visible from the code that trips
over them.

### C.1 The version-controlled configuration files are stale decoys

`configs/*.csv` and `configs/*.yml` are tracked in git and were the live
configuration before versioning arrived. They are no longer live: the station
resolves configuration through the database's main-version pointers into
`configs/versions/`.

They have since drifted badly. `configs/floquet_storage_swap_dataset.csv` is
frozen at 2026-01-08, with `pi_frac` 50/30/40 where August used 40 throughout,
and it **lacks the `waveform`, `gauss_sigma` and `gauss_n_sigma` columns** that
the August Gaussian-envelope data needs.

The trap: `MMDataset` and its subclasses default to `parent_path='configs'`, so
any new module that omits an explicit path silently reads January. Every new
module here passes its archive path explicitly.

### C.2 Timing conversions need a real QickConfig

`us2cycles`/`cycles2us` are firmware-dependent and are not the identity. Use
the committed `configs/soccfg_snapshot.json`; a hand-rolled stub silently
changes every reconstructed time.
