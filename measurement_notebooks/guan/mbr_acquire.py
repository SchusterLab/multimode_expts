# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Multimode (direct remote)
#     language: python
#     name: multimode-direct
# ---

# %% [markdown]
# # MBR acquisition, after the refactor
#
# The worked example for acquiring a many-body-Ramsey campaign. Point an agent
# at this file, or copy the cells you need into your own notebook.
#
# `analysis_notebooks/guan/MBR_analysis.py` is the other half; this file
# acquires, that one loads and analyses.
#
# ## What changed
#
# **No class was renamed.** Class names are recorded in job provenance --
# saved files are `JOB-<id>_<ClassName>.h5` -- so renaming one would orphan
# data. Many jobs go through
# `CharacterizationRunner(ExptClass=..., ExptProgram=...).execute(overrides=...)`
# (`BatchRunner` merged into it on 2026-09-24).
#
# **Modules did move**, as of 2026-09-12: each stage's Program now lives
# beside its Experiment. The old aggregate Experiments are in
# `experiments/qsim/deprecated/legacy_mbr.py` since 2026-09-24.
# Every old address still resolves -- `floquet_dark_mode_readout` forwards
# moved names through a module-level `__getattr__`, which matters for more
# than notebooks, since the queue records `program_module` per job and
# re-imports by that string. Prefer the new address in new cells.
#
# The other change is older: the four aggregate *analysis* stages became four
# Experiment classes and `EncSpec.analyze(stage="...")` is gone. Each stage
# owns its own batch builder, so this file addresses stages by name and lets
# `experiments/qsim/mbr_campaign.py` map name to (owner, program, builder):
#
# | stage           | owner class                    | program                                     | both now live in            |
# |-----------------|--------------------------------|---------------------------------------------|-----------------------------|
# | `calibration`   | `MBRPhaseCorrectionExperiment` | `EntireFloquetCyclePhaseCalibrationProgram` | `mbr_phase_correction.py`   |
# | `spectrum`      | `MBRSpectrumExperiment`        | `NPhotonHamiltonianSpectroscopyProgram`     | program in `mbr_spectroscopy_program.py` |
# | `propagator`    | `MBRPropagatorExperiment`      | `EncodingPropagatorProgram`                 | `mbr_propagator.py`         |
# | `orthogonality` | `MBROrthogonalityExperiment`   | `EncodingOrthogonalityProgram`              | `mbr_orthogonality.py`      |
#
# ## Two things that will bite you
#
# **Never build a Program yourself.** Go through the Experiment. Stage configs
# carry plural keys (`cycle_decoder_analyzers`) and program bodies read the
# singular (`cycle_decoder_analyzer`); the expansion lives in
# `DarkBaseExperiment.acquire`.
#
# **Never set `floquet_waveform` to force an envelope.** The envelope is per
# mode, read from the swap dataset row (`gauss`, `flat_top` or
# `preload_flattop`). Overriding it in the expt config no longer changes what
# plays, so it only makes the recorded config lie about the data -- and worse
# than lie: `floquet_timing.py` still *honours* it when it recomputes the
# cycle duration offline. Set it and acquisition and analysis will disagree
# about how long a Floquet cycle was, which divides straight into every
# coupling rate.

# %%
import matplotlib.pyplot as plt

from experiments.qsim.mbr_campaign import (
    mbr_defaults,
    mock_station,
    pinned_config_set,
    run_stage,
    smoke,
)

# %% [markdown]
# ## 1. Station
#
# On the measurement PC, build the station and job client the usual way and
# skip to section 2:
#
# ```python
# from experiments.station import MultimodeStation
# from job_server.client import JobClient
#
# station = MultimodeStation(user="jonginn")   # main config versions from the DB
# client = JobClient()
# ```
#
# Off-prod, a mock station runs the whole acquisition path with no hardware:
# real program build, real ASM compile, `MockQickSoc` only at the FPGA
# boundary. It needs a *matched* set of config versions, because the swap
# dataset decides the Floquet envelope; the working files in `configs/` drift.
# Two sets are committed, so this needs no mount and no env vars.

# %%
station = mock_station(**pinned_config_set("preload_current"))
client = None  # no queue off-prod; run_stage acquires in-process
print("mock:", station.is_mock)

# %% [markdown]
# ## 2. The campaign, in one place
#
# `mbr_defaults` is the single definition of the shared expt config. Override
# from it rather than redefining keys, so there is one place to read.
#
# Note what is absent: no `floquet_waveform`, per the warning above.

# %%
SWAP_STORS = [1, 2, 3, 4]
OCCUPATIONS = [
    [0, 0, 0, 0, 3],
    [1, 0, 0, 0, 2],
    [0, 1, 1, 1, 0],
]
CYCLES = [0, 4, 8]

defaults = mbr_defaults(SWAP_STORS, reps=1000)
print({k: defaults[k] for k in
       ("swap_stors", "scramble_sync_cycles", "floquet_hardware_loop", "reps")})

# %% [markdown]
# ## 3. Calibration
#
# Every other stage needs the per-occupation phase correction this produces, so
# run it first. `phase_correction_from_calibration` turns the fit into the
# `phase_by_occupation` mapping the later stages take.

# %%
calibration_expts = run_stage(
    station, "calibration", defaults, SWAP_STORS, OCCUPATIONS,
    job_client=client, cycle_pairs=[0, 1, 2], reps=1500,
)

# %% [markdown]
# `run_stage` returns a list of job Experiments. Wrap it in the stage class,
# then analysis is the three-line pattern:
#
# ```python
# calibration_expts = MBRPhaseCorrectionExperiment._from_expts(calibration_expts)
# calibration_expts.analyze()
# calibration_expts.display()
# correction = MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
#     calibration_expts, cycle_branches=0)
# phase_by_occupation = correction.phase_by_occupation
# ```
#
# Off-prod there is no queue, so a list of acquired Experiments comes back
# instead and there is nothing to fit from ten shots of mock zeros. Use a
# neutral correction to keep going.

# %%
if client is None:
    phase_by_occupation = {tuple(o): 0.0 for o in OCCUPATIONS}
    print(f"{len(calibration_expts)} calibration jobs acquired (mock)")
else:
    from experiments.qsim.deprecated.legacy_mbr import MBRPhaseCorrectionExperiment

    calibration_expts.analyze()
    calibration_expts.display()
    phase_by_occupation = (
        MBRPhaseCorrectionExperiment.phase_correction_from_calibration(
            calibration_expts, cycle_branches=0).phase_by_occupation)
    plt.show()

# %% [markdown]
# ## 4. Spectrum, propagator, orthogonality
#
# All three take the same correction. Each returns its own stage class, so
# `analyze()` and `display()` need no `stage=` argument.

# %%
spectrum_expts = run_stage(
    station, "spectrum", defaults, SWAP_STORS, OCCUPATIONS,
    job_client=client, cycles=CYCLES,
    phase_by_occupation=phase_by_occupation, reps=1000,
)

# %%
propagator_expts = run_stage(
    station, "propagator", defaults, SWAP_STORS, OCCUPATIONS,
    job_client=client, cycles=CYCLES,
    phase_by_occupation=phase_by_occupation, reps=1000,
)

# %%
orthogonality_expts = run_stage(
    station, "orthogonality", defaults, SWAP_STORS, OCCUPATIONS,
    job_client=client, reps=1000,
)

# %% [markdown]
# On prod each of these is an aggregate, so:
#
# ```python
# spectrum_expts.analyze(occupations=OCCUPATIONS, spectrum_method="mpm")
# spectrum_expts.display(spectrum_method="mpm")
# plt.show()
# ```
#
# Re-analysing saved jobs later does not need a runner at all -- see
# `analysis_notebooks/guan/MBR_analysis.py`, which loads by job ID and never
# acquires.

# %%
for label, acquired in [("spectrum", spectrum_expts),
                        ("propagator", propagator_expts),
                        ("orthogonality", orthogonality_expts)]:
    n = len(acquired) if isinstance(acquired, list) else "aggregate"
    print(f"{label:<14} {n}")

# %% [markdown]
# ## 5. The same thing as one call
#
# `smoke()` runs every stage at negligible depth. This is what
# `tests/test_mbr_acquire_mock.py` drives, and it is the fastest way to find
# out whether an edit to a program still builds and compiles:
#
# ```
# pixi run pytest tests/test_mbr_acquire_mock.py
# ```

# %%
print(smoke())
