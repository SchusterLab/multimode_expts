# Qsim many-body Ramsey and Floquet/dark-mode refactor specification

**experiments/qsim/floquet_dark_mode_readout.py** is an 8,272-line
mixed-responsibility module containing 31 classes. Its two largest classes,
**EncodingHamiltonianSpectroscopyExperiment** and **DarkBaseProgram**, account
for approximately 67% of the file, but the refactor covers the entire module
rather than only those two classes.

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

## 1. Scope

### 1.1 Goals

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
error.

Required behavior:

- ordinary offline analysis has no station argument;
- historical timing is resolved from the HDF5 configuration-version
  references described in section 3;
- a temporary legacy loader may accept an explicitly supplied historical
  timing record;
- any legacy timing source is labeled and persisted in derived output; and
- failure to resolve an unambiguous historical source is an error.

Station remains a valid acquisition dependency. It is not an aggregate or
numerical-analysis dependency.

### 2.3 Validate aggregate compatibility

The current **_saved_parameters** documentation says it checks sister
experiments, but it primarily adopts the first configuration and the first
available saved Program.

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

### 3.4 Vault logging

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

There is no public standalone display subsystem.

- Leaf acquisition-quality displays belong to leaf Experiments.
- Matrix-element display belongs to **MBRMatrixElementExperiment**.
- Spectrum display belongs to **MBRSpectrumExperiment**.
- Level-statistics and SFF displays belong to
  **MBRLevelStatisticsExperiment**.

Private plotting helpers may be shared, but callers interact through the
owning Experiment's **display()** method.

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

Existing data lacking configuration-ID stamps may use an explicit,
documented legacy timing descriptor while this track is developed. It never
falls back to current station state.

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

No class movement or deletion occurs in this phase.

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

### 11.1 Offline analysis

- The August characterization dataset loads from HDF5 without FastAPI,
  **jobs.db**, live station state, or implicit pickle loading.
- Existing unstamped data uses only an explicit legacy timing descriptor.
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
  station session names.
- Consider a repository-wide change making the flattened **experiments**
  exporter include only locally defined classes and raise on collisions.

These follow-ups do not move into the MBR implementation merely because the
audit exposed them.

## Appendix A. Whole-file ownership inventory

This inventory prevents the refactor from treating the 3,805-line god
Experiment as the whole problem. Target ownership is provisional until the
acquisition notebook audit; no row authorizes deletion.

| Current class | Prospective owner or role | Decision state |
|---|---|---|
| **DarkBaseExperiment** | Legacy dark-mode acquisition façade or decomposed leaf behavior | Pending consumer audit |
| **DarkBaseProgram** | Source of shared Floquet primitives plus legacy dark-mode behavior | Decompose; do not move intact |
| **DarkBaseRProgram** | Dark-mode RAverager infrastructure | Pending consumer audit |
| **ManStorMultiparityChevronRProgram** | Dark-mode multiparity chevron | Prospective distinct measurement |
| **ManStorMultiparityChevronRExperiment** | Dark-mode multiparity chevron | Prospective distinct measurement |
| **DarkT1Program** | Dark-mode T1 | Prospective distinct measurement |
| **DarkT1Experiment** | Dark-mode T1 | Prospective distinct measurement |
| **SidebandScrambleDarkProgramNewNew** | Legacy dark-mode/scramble bridge used as an MBR base | Remove MBR dependency; legacy fate pending |
| **BroadbandGeValidationProgram** | Dark-mode broadband ge validation | Prospective distinct measurement |
| **NPhotonHamiltonianSpectroscopyProgram** | MBR quadrature acquisition | Replace with descriptively named leaf |
| **EncodingOrthogonalityProgram** | MBR orthogonality-column acquisition | Replace with descriptively named leaf |
| **EncodingPropagatorProgram** | MBR propagator-column acquisition | Replace with descriptively named leaf |
| **EncodingStarkShiftCalibrationProgram** | MBR Stark-phase calibration | Replace with descriptively named leaf |
| **EntireFloquetCyclePhaseCalibrationProgram** | MBR cycle-phase calibration | Replace with descriptively named leaf |
| **FloquetDisplacementKerrProgram** | Floquet displacement Kerr | Distinct measurement family |
| **SinglePhotonFloquetSpectroscopyProgram** | Compatibility wrapper for old N=1 name | Compatibility-only candidate |
| **FloquetPhaseAccumulationProgram** | Closed N=1 access-path phase measurement | Rename after domain review |
| **CentralBosonLocalReturnProgram** | Central-boson local return | Distinct measurement family |
| **CentralBosonLocalReturnExperiment** | Central-boson local return | Distinct measurement family |
| **SidebandScrambleDarkProgramNew** | Legacy dark-mode variant | Pending consumer audit |
| **ManStorScrambleProgram** | Legacy scramble variant | Pending consumer audit |
| **SidebandScrambleDarkProgramDebug** | Legacy diagnostic variant | Pending consumer audit |
| **KerrWaitProgramDark** | Legacy Kerr-wait path | Pending consumer audit |
| **SidebandScrambleDarkProgram** | Legacy dark-mode path; also imported elsewhere | Pending consumer audit |
| **SidebandStarkAmplificationModifiedProgram_old** | Live sideband calibration despite suffix | Requires supported descriptive replacement |
| **StorageSwapPhaseAccumulationProgram** | Storage-swap phase calibration | Prospective distinct measurement |
| **SidebandStarkAmplificationModifiedProgram** | Sideband calibration | Pending variant comparison |
| **SidebandStarkAmplificationModifiedProgram_newold** | Legacy sideband calibration variant | Pending consumer audit |
| **BatchRunner** | Generic acquisition orchestration | Move only after new interfaces stabilize |
| **EncodingHamiltonianSpectroscopyExperiment** | Multiple MBR leaf and aggregate workflows | Replace with concrete Experiment types |
| **FloquetDisplacementKerrExperiment** | Floquet displacement Kerr | Distinct measurement family |

Top-level helper functions receive the same treatment:

- **flatten_exp_lists** is replaced by an explicit source-collection contract.
- central-return classification and validation helpers move with the
  central-boson measurement family.
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

These observations are regression evidence, not permanent architectural
assumptions.
