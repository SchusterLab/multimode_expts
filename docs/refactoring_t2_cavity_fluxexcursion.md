# Historical record: the t2_cavity_fluxexcursion split, as originally specified

> **Status: historical. Superseded as guidance; the work it describes has
> landed and then been reworked.**
>
> This was the spec for splitting classes out of `t2_cavity_fluxexcursion.py`.
> What actually happened differs, so do not use this as a map of the tree:
>
> - It promised a pure move ("copying their existing code unchanged"). The
>   move also renamed 13 of the 14 classes and added docstrings. The bodies
>   were later verified identical modulo those renames.
> - Destinations here read `experiments/calibration/...` etc. The code landed
>   under `experiments/qsim/<role>/...` and was then flattened to
>   `experiments/qsim/*.py`: subpackages are grouped by project, not by role,
>   and `experiments/__init__.py` only exports one level deep.
> - The "notebook copies" section describes `*_refactored.ipynb` duplicates.
>   Those were whole-file forks; they have replaced their originals and the
>   suffixed copies are gone.
> - Its closing question, whether to delete the originals, is answered: they
>   are deleted outright. Old names are not aliased: experiment pickles are
>   ephemeral by policy, so an old one is loaded by checking out the commit
>   it was written under.
>
> Anything below that reads as a plan, an intention, or a loose end is a
> record of what was thought at the time, **not an instruction for present or
> future work**. Do not act on it. Current direction lives in
> [`qsim_refactor_surface_map.md`](qsim_refactor_surface_map.md), with its
> evidence in
> [`qsim_notebook_surface_inventory.md`](qsim_notebook_surface_inventory.md);
> those two supersede every other refactor doc in this repo.

## Refactoring t2_cavity_fluxexcursion

## Scope

Move the selected classes into focused modules by copying their existing code unchanged. Adjust only the imports required by the new locations. Keep the original modules, notebooks, and `experiments/__init__.py` in place.

New notebooks are copies of the existing notebooks, with only module imports and references changed. Bug fixes, fitting changes, shared-helper extraction, and class redesign are outside this phase. Review deletion of the originals only after validation.

## Class decisions

### 1. AmplitudeCalibration — keep and move

This was used to measure how spectroscopy resonance changes with applied flux-drive gain. The triangular boundary gives a gain-versus-frequency relation, which can be combined with the frequency-versus-DC-current calibration to obtain gain versus current. Unmodulated spectroscopy provides a reference when large drive amplitudes produce split spectral structure.

Keep it as a separate calibration program. The class performs acquisition; the existing notebook analysis stays where it is.

Destination: `experiments/calibration/flux_amplitude_calibration.py`.

### 2. ExcursionTransitionDebuggingProgram — keep and move

This investigates the response to a flux excursion. Its existing branches cover a prepared manipulate photon and population mapping, direct flux-induced qubit response, and Wigner measurements. Its role is broader than a single qubit-transition test.

Keep the complete class unchanged as a diagnostic program.

Destination: `experiments/diagnostics/flux_excursion.py`.

### 3. Manf0g1RamseyProgram — keep and move

This creates f0/g1 coherence and measures the relative phase accumulated during the flux excursion. The notebook uses it to extract gain-dependent Ramsey frequency shifts.

It is not a direct cavity self-Kerr measurement: the usual cavity self-Kerr term vanishes for both zero and one photon. Keep the program as an f0/g1 Ramsey measurement.

Destination: `experiments/cavity_ramsey/man_f0g1.py`.

### 4. Cavity Ramsey excursion classes — move together for now

Move these three classes together:

- `KerrCavityRamseyExcursionProgram`
- `KerrCavityRamseyExperimentMod`
- `KerrCavityRamseyExcursionExperiment`

The program supplies the actual flux-excursion sequence. The experiment classes select that program and provide analysis/display interfaces used by existing notebooks.

The original question about whether both experiment wrappers are necessary remains a separate cleanup decision. Replacing them with generic acquisition or another fitter would go beyond moving code and changing notebook import paths. For this phase, copy all three unchanged.

Destination: `experiments/cavity_ramsey/cavity_excursion.py`.

### 5. SlowLengthRabiProgram and SlowPiGeRamseyProgram — keep and move

These are actively used for slow ge-pulse length calibration and Ramsey measurements. Their callers include the manual-calibration and multiphoton notebooks.

Move the pair into one calibration module. The existing notebook preprocessing, fitting, and calibration-update code remains unchanged.

Destination: `experiments/calibration/slow_pi_ge.py`.

### 6. MActiveResetVerificationProgram — keep and move

This prepares the test excitation before active reset and measures the result, with optional mapping pulses. Keep it as a dedicated reset-verification diagnostic.

Destination: `experiments/diagnostics/man_reset.py`.

### 7. SidebandGeneralAmpProgram and SidebandGeneralAmpExperiment — keep and move

These are not part of the current routine workflow, but the amplitude-sweep capability is worth retaining as requested. The experiment class owns its acquisition loop and analysis, while the program supplies the pulse sequence.

Move both unchanged into the same module.

Destination: `experiments/calibration/sideband_general_amp.py`.

### 8. FloquetChevronAmpProgram and FloquetAmpChevronExperiment — keep and move

These are essential for Floquet gain-chevron calibration in the main and high-Kerr qsim notebooks. Keep both the pulse program and experiment class.

They already inherit from the existing Qsim bases. Moving them requires no new Floquet base program or dark-mode implementation.

Destination: `experiments/floquet/floquet_gain_chevron.py`.

### 9. Qsimf0g1Sepctroscopy — omit from the new modules

No exact-name caller was found in the searched Python and jonginn notebook source. The spectroscopy workflows identified in the notebooks use `AmplitudeCalibration`.

Do not copy this class into the new modules. This does not imply that it was never used historically.

### 10. SidebandScrambleDarkProgram — omit from the new modules

This was used historically, but it is a deprecated protocol and is not retained in this t2 split. Do not copy it or create a replacement dark-mode module.

Its original definition and existing notebook cells stay untouched during this phase. Other programs in `floquet_dark_mode_readout.py` are outside this move.

### 11. SlowLengthRabiQsimExperiment — omit from the new modules

This was a temporary wrapper for the older worker interface. The current runner supports explicit program selection, and no current caller of this wrapper was found.

Do not copy it into the new modules.

### 12. Other diagnostics in the file

`ParityDebuggingProgram` has a caller in `qsim_wigner_jonginn.ipynb`. Keep it and move it unchanged to `experiments/diagnostics/parity.py`.

`ActiveResetVerificationProgram` appears only as commented alternatives in the inspected notebook source. Omit it from the new modules. It has different preparation/reset ordering from the M-prefixed class, so it is not an interchangeable replacement.

Omitting classes from new modules does not delete their original definitions or remove references from notebook copies.

## Notebook copies

Copy each affected notebook as a whole. Change only the imports and corresponding module/class references needed to select the moved code.

| Existing notebook | Relevant classes or workflow |
|---|---|
| `multiphoton_calibration_v2_jonginn.ipynb` | Amplitude calibration, excursion diagnostics, f0/g1 Ramsey, cavity excursion, slow-pulse calibration |
| `single_qubit_manualcalibrate.ipynb` | Slow-pulse calibration, M-prefixed reset verification, sideband amplitude |
| `qsim_experiments.ipynb` | Floquet gain chevron |
| `qsim_experiments_highkerr_untracked.ipynb` | Corresponding Floquet gain chevron |
| `qsim_wigner_jonginn.ipynb` | Parity verification and relevant experiment imports |
| `data_postprocess.ipynb` | Cavity excursion experiment imports |

These notebooks are under `measurement_notebooks/jonginn/`. Keep all other source, cell structure, settings, analysis, plots, outputs, execution counts, and metadata unchanged. Do not split notebooks, clear outputs, fix existing issues, or remove deprecated cells.

Use full defining-module imports, for example:

```python
from experiments.calibration.slow_pi_ge import SlowLengthRabiProgram
```

Replace the corresponding old reference in the existing call; leave its arguments unchanged. Keep unrelated imports and references as they are.

Both original and new modules may be loaded by the existing package scanner. This is accepted. Full-path imports select the intended class without changing `__init__.py`.

## Implementation and verification

1. Copy the selected classes and connect their imports, including dependencies between classes moved together.
2. Copy the existing notebooks and update only the relevant imports and references.
3. Compare class implementations with the originals and check that notebook differences contain only those import/reference edits.
4. Verify imports and existing calls, including worker resolution of the new module paths.

After the moved code works through the existing notebook calls, review whether to delete the original files. Deletion is not part of this phase.
