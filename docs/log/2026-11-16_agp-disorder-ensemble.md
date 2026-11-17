# Disorder-ensemble AGP implementation

Implemented locally for jonginn, following the request to keep existing
measurement/Floquet modules unchanged. No hardware jobs, remote writes, commits,
or package installations were performed.

## Acquisition

- New FockPopulationEnsembleProgram inherits existing Fock preparation,
  optional prep heralds, and ordinary multiparity readout.
- Existing register-bank and pulse-emission helpers are imported unchanged.
  The bank and persistent DMEM pointer are initialized before the outer shot
  loop. Each repetition loads one table row before preparation, resets phase
  accumulators, and executes the preloaded Floquet loop. No extra idle is added.
- One shot per realization uses reps equal to table length and rounds=1.
  Time points remain a software sweep. The table is not regenerated per Fock
  state, time, or W; W scales the supplied shared patterns.
- Corrected notebook pattern generation to match jonginn's separate
  multimode_thermalization_sim/notebooks/jonginn/occupation_chaos_mechanism.ipynb
  (AGP cells 32-33): Gaussian draws, subtract each storage row's mean, divide
  by its centered RMS. The hub is excluded. The previous independent uniform
  draws without row normalization described a different disorder ensemble.
  The configured seed and realization count remain user-controlled.
  Checked the generation cell alone against row-wise reference normalization
  for 500 four-storage realizations: row means are zero and row RMS values
  are one within 1e-15. Notebook syntax passes; no measurement cells were run.
- Initial scope is one table fitting actual configured DMEM, with no automatic
  batching or sequential multi-mode readout.
- Review follow-up: consolidated ensemble configuration and DMEM checks in
  validate_ensemble_config(), called before parent initialization. Waveform
  support is read from the already attached calibration dataset, so it does
  not depend on the parent's derived waveform-mode list. Backend register and
  waveform resource checks remain delegated. All eight hardware-path tests pass.
- Added English comments tracing the detuning-to-pulse path and row-major DMA
  upload order.
- Clarified register-number allocation versus runtime pointer arithmetic:
  cycle_register + 1 selects another register, whereas mathi increments the
  DMEM address held in that register during each shot. Comments only.
- Frequency convention review: the local name carrier meant the existing
  calibrated pulse frequency word. Renamed it calibrated_frequency_word, with
  explicit detuning_word and dds_word_modulus names. The existing +detuning
  pulse rule and the model's -detuning onsite rule are unchanged. All eight
  acquisition tests pass, including positive/negative fixed-path equivalence.
- Reuse follow-up supersedes the local frequency-word calculation above:
  each table row now calls the inherited _detuned_floquet_pulse_args() itself.
  Its frequency values are assigned to the uint32 table directly. Original
  detuning vectors and exact frequency words are saved.
  This avoids maintaining a second implementation of the frequency convention.
- Removed the redundant astype(np.uint32) on table assignment and the
  view(np.int32) before upload. The existing v1 driver copies the uint32 input
  into its int32 DMA buffer; the ensemble module does not repeat that conversion.
  All eight acquisition tests pass after this simplification.
- At jonginn's request, removed the extra freq2int/reg2freq round trip and
  disorder_applied_detunings_mhz from acquisition, HDF5 assembly and test fixtures.
  Theory now uses the original saved disorder_detunings_mhz table directly.
  All 12 tests in the acquisition and assembled-ensemble suites pass after removal.
- Raw child HDF5s preserve original detunings, exact DDS frequency
  words, realization IDs, unfiltered decoded labels, herald masks, raw IQ,
  occupations, storage ordering, and time axes.

## Analysis and persistence

- New ensemble AGP Experiment/scan classes use the normal named manifests and
  assembled HDF5 lifecycle. One JOB covers one contributing initial Fock state.
- The estimator uses existing confusion-matrix conventions and kernel weights.
  Bootstrap samples resample realization IDs jointly across Fock states and
  time points. Shared calibration uncertainty is propagated separately when
  calibration shot counts are available. With one shot per realization,
  sampling errors combine shot noise and finite-ensemble variation.
- The default cutoff is g L/D in cyclic kHz. Stored AGP is g^2 ||A_mu||^2.
- Theory uses the original saved vectors through the existing Hamiltonian sign
  convention. Infinite-time, analytic finite-window, and sampled-window theory
  remain separate from the measured estimator. There are no model-derived
  experimental corrections or fitted decay/background terms.
- New English Jupytext notebook:
  measurement_notebooks/jonginn/agp_ensemble_measurement.py.
- Added the missing pre-batch quantum-walk verification requested by jonginn:
  separate calibration-display, walk-settings, walk-acquisition and walk-display
  cells precede the AGP batch. Each mode uses the same confusion calibration,
  herald setting and existing ordinary population runner, with zero detunings.
  Defaults match the ordinary walk: M1 initially populated, all modes read,
  zero detunings, 1000 repetitions and a short cycle sweep. The walk does not
  use the ensemble runner or a disorder table.
- Ordinary notebook walk cells plot raw/corrected populations and call the
  existing fixed_n_hamiltonian and fock_number_traces for the separate theory
  overlay. No new walk helper module or ensemble Experiment method is needed.
- Fixed the new notebook's station construction to use the existing
  storage_man_file/floquet_file argument names for the selected config IDs.
  Notebook syntax and all four walk cells passed with synthetic ordinary
  population jobs, including figure rendering and theory photon-number
  conservation. No hardware or queue jobs were submitted. The ensemble
  acquisition module has no changes from the user's committed version.

## Validation

80 targeted tests passed: the three new ensemble suites plus existing AGP
kernel, theory, Experiment, scan, population-set, and phase-frame tests.

The hardware tests verify row order, phase reset, zero-time pointer advancement,
500-by-4 DMEM capacity and constant program length, exact pulse-parameter and
scheduled-time agreement with the fixed-disorder path, readout-slot preservation,
and HDF5 reload without IQ. Syntax checks passed for all new modules/notebook;
all code is English and the longest function is 54 lines.

Existing additional hardware tests have nine environment failures because this
checkout lacks the external firmware ctrl.sv used by the old timing checker.
That checker is disabled only in the new test's fixed-path comparator. Mock and
instruction tracing do not validate real FPGA instruction latency; a device
pilot remains necessary before a measurement campaign.

The local Pixi executable cannot parse the root lockfile version. Tests used the
existing tmp/agp_python_runner/pixi.toml with --as-is --no-completions, activating
the existing environment's DLL directories in the process PATH. No environment
files or packages were changed.
