# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (pixi env of cwd)
#     language: python
#     name: pixi-cwd
# ---

# %% [markdown]
# # Wide common-detuning scan with S4-S7 swapped (guan + Claude, 2026-11-07)
#
# Repeat of jonginn's `JOB-20261107-00441` (`FockPopulation2DExperiment`, Fock 1 in M1, the
# same detuning on S4-S7, read M1) over a wider range: +-150 kHz in 31 points instead of
# +-10 kHz in 5. Over +-10 kHz the oscillation frequency changes by only about 1 kHz, because
# the scale is 2 g_eff (about 80 kHz). The wide range shows the full frequency curve, and
# so the residual detuning (offset) and whether the response is symmetric.
#
# First run: `JOB-20261107-00492` (100 reps, about 36 min). Kernel: `guan_2d_wide`
# (Lab server, port 8765).

# %% Cell 1: queue, and the station on the config versions of 00441 (19:15 today).
# jonginn made newer versions after 00441 (Stark and error amplification, f0g1 Rabi; jobs
# 00467-00491). Against CFG-HW-20261107-00020, the readout phase of 00005 differs by 6.7 deg
# (122.8 -> 116.1) and the qubit pi gains by < 1 %; the readout frequency is the same.
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
from slab import AttrDict

from experiments import CharacterizationRunner, MultimodeStation
from experiments.qsim.fock_population_2d import FockPopulation2DExperiment
from job_server import JobClient

client = JobClient()
print("Server:", client.health_check()["status"])
client.print_queue()

config_dict = {
    "hardware_config": "CFG-HW-20261106-00005",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20261106-00001",
    "floquet_storage_swap": "CFG-FL-20261105-00030",
}
station = MultimodeStation(
    user="guan",
    experiment_name="261027AGPlowKerr",
    project="AGP",
    log_measurements=True,
    hardware_config=config_dict["hardware_config"],
    multiphoton_config=config_dict["multiphoton_config"],
    storage_man_file=config_dict["man1_storage_swap"],
    floquet_file=config_dict["floquet_storage_swap"],
)

# %% Cell 2: defaults = the cfg.expt of 00441 (without the sweep keys), and the runner.
# This keeps every setting of 00441, including its confusion matrix (JOB-20261106-00101),
# cycles 0-1195 in steps of 5 and 100 reps. Only the detunings change.
REF_FNAME = r"C:\experiments\261027AGPlowKerr\data\JOB-20261107-00441_FockPopulation2DExperiment.h5"
with h5py.File(REF_FNAME, "r") as f:
    ref_expt = json.loads(f.attrs["config"])["expt"]
wide_defaults = AttrDict({k: v for k, v in ref_expt.items()
                          if k not in ("floquet_cycle", "detuningss", "swept_params")})
print("cycles:", wide_defaults.floquet_cycles[0], "...", wide_defaults.floquet_cycles[-1],
      len(wide_defaults.floquet_cycles), "| reps:", wide_defaults.reps,
      "| swap:", wide_defaults.swap_stors, "| occ:", wide_defaults.occupations,
      "| cm:", wide_defaults.multiparity_confusion_matrix["fname"])

population_2d_runner = CharacterizationRunner(
    station=station,
    ExptClass=FockPopulation2DExperiment,
    default_expt_cfg=wide_defaults,
    job_client=client,
    use_queue=True,
    show=False,
)

WIDE_DETUNINGS = np.round(np.linspace(-0.150, 0.150, 31), 6)  # MHz, the same on S4-S7
wide_detuningss = [[float(d)] * 4 for d in WIDE_DETUNINGS]
print("detunings (kHz):", 1e3 * WIDE_DETUNINGS)

# %% Cell 3: mock check at the two edge detunings (compile and board resources), then back
# to the real instruments.
station.use_mock_instruments()
try:
    mock_expt = population_2d_runner.execute(
        reps=1,
        detuningss=[wide_detuningss[0], wide_detuningss[-1]],
        floquet_cycles=[0, 5, 1195],
        show=False,
    )
    print("mock ok:", type(mock_expt).__name__, "avgi shape", np.shape(mock_expt.data.get("avgi")))
finally:
    station.use_real_instruments()

# %% Cell 4: submit. One job, about 70 s per detuning at 100 reps (about 36 min for 31).
# Time estimate from today's 2D jobs: about 0.19 s per sweep point + 1.05 ms per shot.
client.print_queue()
population_2d_wide = population_2d_runner.execute(
    reps=100,
    detuningss=wide_detuningss,
    postprocess=False,
    show=False,
    log=True,
)
print(population_2d_wide.fname)

# %% Cell 5: the class figure (population map, FFT, sine and FFT frequency fits).
# At large detuning the oscillation is small, and the class FFT peak finder can choose the
# low bin of the slow decay; then the sine fit and the dispersion fit fail (see Cell 6).
fig, axes = population_2d_wide.display()
plt.show()

# %% Cell 6: residual of each term in the full train (Claude, 22:24; jobs 00493-00500).
# The 00492 map has two lines on the + side (+20..+40 kHz): the 4 terms are not equivalent.
# All 4 pulses play (same Stark environment). The scanned term goes first (a cyclic rotation of
# [4, 5, 6, 7]: the same periodic train, and the class x axis is then its detuning); the other 3
# are parked at +1 MHz, then at -1 MHz. Each parked term pulls M1 by about g^2/Delta = 0.4 kHz;
# the mean of the two park signs removes the pull. Fit per job: f = sqrt((d + r)^2 + 4 g^2).
PERTERM_SCAN = np.round(np.linspace(-0.060, 0.060, 7), 6)  # MHz, on the scanned term
PARK_MHZ = 1.0
BASE_ORDER = [4, 5, 6, 7]


def perterm_override(active, park):
    """Scanned term first (cyclic rotation of BASE_ORDER), the other 3 parked at `park` MHz."""
    i = BASE_ORDER.index(active)
    order = BASE_ORDER[i:] + BASE_ORDER[:i]
    detuningss = [[float(d)] + [park] * 3 for d in PERTERM_SCAN]
    return dict(swap_stors=order, storage_reset=order, detuningss=detuningss)


perterm_overrides = [perterm_override(s, sign * PARK_MHZ) for sign in (+1, -1) for s in BASE_ORDER]
perterm_expts = population_2d_runner.execute(overrides=perterm_overrides, batch_size=2, reps=100,
                                             postprocess=False, show=False, log=True)

# %% Cell 7: pulse-order test (Claude, 23:32; jobs 00501, 00502, 20261108-00001).
# Same common scan in other orders; [4, 5, 6, 7] again at the end as a drift check.
ORDER_SCAN = np.round(np.linspace(-0.060, 0.060, 13), 6)  # MHz, common to all 4 terms
ORDERS = [[4, 6, 5, 7], [7, 6, 5, 4], [4, 5, 6, 7]]
order_overrides = [dict(swap_stors=o, storage_reset=o, detuningss=[[float(d)] * 4 for d in ORDER_SCAN])
                   for o in ORDERS]
order_expts = population_2d_runner.execute(overrides=order_overrides, batch_size=2, reps=100,
                                           postprocess=False, show=False, log=True)

# %% Cell 8: fine single-term scans +30..+66 kHz (Claude, 00:18; jobs from 20261108-00002).
# Looks for the extra level X: both good models put its anticrossing near +40..+50 kHz, only in
# the scan of the term that carries it.
FINE_SCAN = np.round(np.arange(0.030, 0.0661, 0.003), 6)  # MHz, on the scanned term


def fine_override(active, park=1.0):
    i = BASE_ORDER.index(active)
    order = BASE_ORDER[i:] + BASE_ORDER[:i]
    return dict(swap_stors=order, storage_reset=order,
                detuningss=[[float(d)] + [park] * 3 for d in FINE_SCAN])


fine_overrides = [fine_override(s) for s in BASE_ORDER]
fine_expts = population_2d_runner.execute(overrides=fine_overrides, batch_size=2, reps=100,
                                          postprocess=False, show=False, log=True)

# %% Cell 9: Floquet recalibration at another gap (inline helper; see module gaps).
# jonginn's floquet_calibration.py cells 2 and 3 (settings of his jobs 00460-00479), with the gap
# as a parameter. A fit is accepted only inside its scan and with a small error (phase < 0.5 deg).
from copy import deepcopy

import experiments as meas
from experiments.branch_manager import BranchManager
from experiments.qsim.notebook_helpers.defaults import ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS

PAIRS = [(a, b) for a in (4, 5, 6, 7) for b in (4, 5, 6, 7) if a != b]
RECAL_LOG = []  # (name, parameter, old, fitted, accepted)


def make_recal_runners(station, client, gap):
    floq = AttrDict(dict(FLOQUET_DEFAULTS, scramble_sync_cycles=gap, floquet_sync_delay=gap))
    ea_defaults = AttrDict(dict(
        reps=100, rounds=1, qubits=[0], active_reset=True, man_mode_no=1, stor_is_dump=False,
        man_reset=True, storage_reset=True, relax_delay=200, expts=50, qubit_start_storage="g",
    ))
    ea_defaults.update(ACTIVE_RESET_DEFAULTS)
    ea_defaults.update(floq)
    coarse = {"gain": dict(n_pulses=8, expts=60), "frequency": dict(n_pulses=7, expts=50)}

    def ea_preproc(station, default_expt_cfg, **kwargs):
        cfg = deepcopy(default_expt_cfg)
        cfg.update(coarse[kwargs["parameter_to_test"]])
        cfg.update(kwargs)
        name = f"M1-S{cfg.stor_mode_no}"
        cfg.pulse_type = ["floquet", name, f"pi/{station.ds_floquet.get_pi_frac(name)}", 0]
        if cfg.parameter_to_test == "frequency":
            cfg.start = station.ds_floquet.get_freq(name) - cfg.span / 2
            cfg.step = cfg.span / (cfg.expts - 1)
        else:
            cfg.start = int(station.ds_floquet.get_gain(name) - cfg.span / 2)
            cfg.step = int(cfg.span / (cfg.expts - 1))
        return cfg

    def ea_postproc(station, expt):
        expt.analyze(data=expt.data, state_fin="e")
        e = expt.cfg.expt
        opt, err = expt.data["fit_avgi"][2], float(np.asarray(expt.data["fit_prod_avgi_err"])[2])
        name = f"M1-S{e.stor_mode_no}"
        old = station.ds_floquet.get_gain(name) if e.parameter_to_test == "gain" else station.ds_floquet.get_freq(name)
        lo, hi = e.start, e.start + e.step * (e.expts - 1)
        ok = np.isfinite(opt) and lo < opt < hi and np.isfinite(err) and err < (hi - lo) / 4
        RECAL_LOG.append((name, e.parameter_to_test, old, opt, ok))
        if ok:
            (station.ds_floquet.update_gain if e.parameter_to_test == "gain" else station.ds_floquet.update_freq)(name, opt)
        print(f"[recal gap {gap}] {e.parameter_to_test} {name}: {old} -> {opt} ({'accepted' if ok else 'NOT accepted'})")

    ea_runner = CharacterizationRunner(
        station=station, ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
        default_expt_cfg=ea_defaults, preprocessor=ea_preproc, postprocessor=ea_postproc,
        job_client=client, use_queue=True, show=False)

    st_defaults = AttrDict(dict(
        expts=1, reps=100, rounds=1, qubits=[0], f0g1_cavity=1, init_stor=0, ro_stor=0,
        n_pulses=[0, 8, 16, 24, 32, 40], swept_params=["n_pulse", "advance_phase"], normalize=False,
        active_reset=True, man_reset=True, storage_reset=False, prepulse=True, postpulse=True, init_fock=True,
        relax_delay=100,
    ))
    st_defaults.update(ACTIVE_RESET_DEFAULTS)
    st_defaults.update(floq)

    def st_preproc(station, default_expt_cfg, **kwargs):
        cfg = deepcopy(default_expt_cfg)
        cfg.update(kwargs)
        cur = station.ds_floquet.get_phase_from(f"M1-S{cfg.stor_A}", f"M1-S{cfg.stor_B}")
        half = cfg.pop("phase_half_span", 3.0)
        cfg.advance_phases = np.linspace(cur - half, cur + half, 51).tolist()
        cfg.storage_reset = [cfg.stor_A, cfg.stor_B]
        return cfg

    def st_postproc(station, expt):
        a, b = f"M1-S{expt.cfg.expt.stor_A}", f"M1-S{expt.cfg.expt.stor_B}"
        expt.analyze(fit=True)
        opt, err = expt.data["fit_avgi"][2], float(np.asarray(expt.data["fit_prod_avgi_err"])[2])
        old = station.ds_floquet.get_phase_from(a, b)
        phases = expt.cfg.expt.advance_phases
        ok = np.isfinite(opt) and min(phases) < opt < max(phases) and np.isfinite(err) and err < 0.5
        RECAL_LOG.append((a, f"phase_from {b}", old, opt, ok))
        if ok:
            station.ds_floquet.update_phase_from(a, b, opt)
        print(f"[recal gap {gap}] phase_from {a} <- {b}: {old:.4f} -> {opt:.4f} ({'accepted' if ok else 'NOT accepted'})")

    st_runner = CharacterizationRunner(
        station=station, ExptClass=meas.SidebandStarkAmplificationExperiment,
        ExptProgram=meas.SidebandStarkAmplificationProgram,
        default_expt_cfg=st_defaults, preprocessor=st_preproc, postprocessor=st_postproc,
        job_client=client, use_queue=True, show=False)
    return ea_runner, st_runner


def freq_gain_overrides(station):
    """jonginn's fine pass: frequency span 0.05 MHz, gain span 15 % of the gain; one job at a time."""
    out = []
    for s in (4, 5, 6, 7):
        kw = dict(stor_mode_no=s, storage_reset=[s], go_kwargs=dict(analyze=False, progress=True, display=False))
        out.append(dict(kw, parameter_to_test="frequency", span=0.05))
        out.append(dict(kw, parameter_to_test="gain", span=int(0.15 * station.ds_floquet.get_gain(f"M1-S{s}"))))
    return out


def gap_test(gap, detuningss):
    """Recalibrate at `gap` on config branch crosstalk_gap<gap>, scan, then go back to crosstalk_gap1_00441."""
    bm = BranchManager(station)
    try:
        ea_runner, st_runner = make_recal_runners(station, client, gap=gap)
        for o in freq_gain_overrides(station):
            ea_runner.execute(**o)
        for a, b in PAIRS:
            st_runner.execute(stor_A=a, stor_B=b)
        n_ok = sum(1 for row in RECAL_LOG[-20:] if row[4])
        bm.commit(f"crosstalk_gap{gap}", note=f"Floquet freq, gain, pairwise phases at gap {gap}; {n_ok}/20 accepted")
        if n_ok == 20:
            return population_2d_runner.execute(reps=100, detuningss=detuningss, scramble_sync_cycles=gap,
                                                postprocess=False, show=False, log=True)
    finally:
        bm.checkout("crosstalk_gap1_00441", force=True)


# %% Cell 10: gap 10 (Claude, 01:20; recal 20261108-00006..00025, scan 20261108-00026).
BranchManager(station).commit("crosstalk_gap1_00441", note="configs of JOB-20261107-00441 (gap 1)")
gap10_expt = gap_test(10, wide_detuningss)

# %% Cell 11: pair scans (Claude, 02:16; jobs from 20261108-00027). 2 terms at a common detuning
# 0..+60 kHz, the other 2 parked at +1 MHz; the rotation starts at the first term of the pair.
PAIR_SCAN = np.round(np.arange(0.0, 0.0601, 0.005), 6)
PAIRS2 = [(4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)]


def pair_override(a, b, park=1.0):
    i = BASE_ORDER.index(a)
    order = BASE_ORDER[i:] + BASE_ORDER[:i]
    return dict(swap_stors=order, storage_reset=order,
                detuningss=[[float(d) if s in (a, b) else park for s in order] for d in PAIR_SCAN])


pair_expts = population_2d_runner.execute(overrides=[pair_override(a, b) for a, b in PAIRS2], batch_size=2,
                                          reps=100, postprocess=False, show=False, log=True)

# %% Cell 12: gap 2 (Claude, after the pair scans): common scan -90..+90 kHz in 10 kHz steps.
gap2_expt = gap_test(2, [[float(d)] * 4 for d in np.round(np.linspace(-0.090, 0.090, 19), 6)])

# %% Cell 13: S4+S5 with a relative detuning d4 - d5 = -20 / +20 kHz (Claude, 04:35; jobs 20261108-00054, -00055).
# Test of the M2 hypothesis: the extra level sits at c + d4 - d5 and couples to S5.
DELTA_SCAN = np.round(np.arange(0.0, 0.0901, 0.005), 6)  # MHz, detuning of S5
delta_overrides = [dict(swap_stors=[4, 5, 6, 7], storage_reset=[4, 5, 6, 7],
                        detuningss=[[float(d5 + D), float(d5), 1.0, 1.0] for d5 in DELTA_SCAN])
                   for D in (-0.020, 0.020)]
delta_expts = population_2d_runner.execute(overrides=delta_overrides, batch_size=2, reps=100,
                                           postprocess=False, show=False, log=True)

# %% [markdown]
# ## Cell 14: module gaps
#
# - `FockPopulation2DExperiment.analyze()` (`experiments/qsim/fock_population_2d.py`): the FFT
#   peak that starts the sine fit is taken over all bins above 0. The slow T1 decay puts a
#   large amplitude in the lowest bin (3.3 kHz for a 306 us window), so the peak finder can
#   choose it. Seen in 00438 and 00439 at +25 kHz (sine fit 0.8 kHz instead of about 81 kHz,
#   then the dispersion fit failed), and in 00492 at most |d| >= 50 kHz. Fix: remove a slow
#   background (polynomial or exponential) before the FFT, or search only above a minimum
#   frequency; and fit a damped sine with a slow background. The exploratory version is in
#   `.cache/crosstalk_20261107/osc_fit.py` (not in git).
# - The same class fits one hyperbola. With an extra level (gap 1) that fit has no meaning;
#   a symmetric-pair residual (f(+d)^2 - f(-d)^2)/4d per |d| shows it directly.
# - Cell 9 (Floquet recalibration with guards) belongs in a module next to jonginn's
#   floquet_calibration.py runners (for example `experiments/qsim/notebook_helpers/`), so the
#   gap can be changed without copying the cells.
# - The per-term / pair / extra-level analysis (`.cache/crosstalk_20261107/analyze_*.py`,
#   `fit_extra*.py`) is exploratory; the model (one photon, M1 + storages + extra level) belongs in
#   `theory/qsim/central_boson.py` and the fits in `fitting/qsim/` if we keep using it.
