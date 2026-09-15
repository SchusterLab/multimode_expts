"""Helpers for the multiphoton calibration measurement notebook.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
7-57 and 61-68 by the stage-2 notebook decomposition. Primary caller:
`measurement_notebooks/202609_qsim_migration/multiphoton_calibration.py`.

Two kinds of thing live here. The preprocessor/postprocessor hooks
(`broadband_amprabi_preproc`, `singleshot_postproc`) were already top-level
`def`s in the notebook and moved unchanged. The rest are the long procedural
cells, each wrapped into one function so the notebook reads as a sequence of
named steps.

Several of those wrappings collapse cells that were copy-paste duplicates
differing only in a scan width, which the stage-2 instructions allow because
the duplication is confirmed rather than assumed:

- `fit_broadband_frequency` is cells 18 and 21. Their only difference is
  `band` (5.0 coarse, 1.0 fine).
- `fit_broadband_gain` is cells 19 and 22, differing only in
  `gain_half_band` (3000 coarse, 1500 fine) and whether the job was logged.
- `scan_return_error` is cells 42, 45, 48 and 51 -- coarse/fine frequency and
  coarse/fine gain of the swap error amplification. Same body throughout:
  sweep, score each point by its mean IQ distance from the depth-zero row,
  take the argmin, plot.
- `even_gain_grid` is the gain-range arithmetic shared by cells 45 and 48.

None of these functions writes to the station. Accepting a fit into
`ds_storage` or into the `pi_ge_broadband` config stayed in the notebook,
because which candidate to accept after looking at a plot is the scientific
choice the notebook is supposed to keep.

Temporary home, per the stage-2 instructions: no attempt was made to reconcile
these with `fitting/` or with the existing autocalibration runners.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

import fitting.fitting as fitter


# --------------------------------------------------------------------------
# Runner hooks -- moved unchanged from cells 11 and 66.
# --------------------------------------------------------------------------


def broadband_amprabi_preproc(station, default_expt_cfg, **kwargs):
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    expt_cfg.qubits = [int(expt_cfg.qubit)]
    expt_cfg.prepulse = bool(expt_cfg.pre_sweep_pulse)
    return expt_cfg


def singleshot_postproc(station, expt):
    expt.analyze(plot=False, station=station, subdir=station.autocalib_path)
    fids = expt.data['fids']
    confusion_matrix = expt.data['confusion_matrix']
    thresholds_new = expt.data['thresholds']
    angle = expt.data['angle']
    print(fids)

    hardware_cfg = station.hardware_cfg
    hardware_cfg.device.readout.phase = [hardware_cfg.device.readout.phase[0] + angle]
    hardware_cfg.device.readout.threshold = thresholds_new
    hardware_cfg.device.readout.threshold_list = [thresholds_new]
    hardware_cfg.device.readout.Ie = [np.median(expt.data['Ie_rot'])]
    hardware_cfg.device.readout.Ig = [np.median(expt.data['Ig_rot'])]
    if expt.cfg.expt.active_reset:
        hardware_cfg.device.readout.confusion_matrix_with_active_reset = confusion_matrix
    else:
        hardware_cfg.device.readout.confusion_matrix_without_reset = confusion_matrix
    print('Updated readout!')


# --------------------------------------------------------------------------
# Broadband ge calibration (cells 12-26).
# --------------------------------------------------------------------------


def ge_population_transfer(z, z_g, z_e):
    """Project IQ points onto one photon number's own g/e readout axis.

    Returns the population-transfer coordinate: 0 at the |g> reference, 1 at
    the |e> reference. Cells 13 and 26 both did this inline. As the notebook
    markdown is careful to say, this is a transfer coordinate from that (n)'s
    own references, not a leakage-resolved gate fidelity.
    """
    readout_axis = z_e - z_g
    return np.real((z - z_g) * np.conj(readout_axis)) / abs(readout_axis) ** 2


def run_broadband_rabi_scan(runner, photon_numbers, broadband_sigma,
                            broadband_frequency, show_each=True):
    """Amplitude Rabi from |g,n> and |e,n> at each photon number (cell 12).

    Two jobs per photon number. These are pure-state preparations, so
    conditional shelving is not needed.

    Returns (rabi_from_g, rabi_from_e), each a list indexed by photon number.
    """
    rabi_from_g = [None] * len(photon_numbers)
    rabi_from_e = [None] * len(photon_numbers)

    for photon_number in photon_numbers:
        prep_gN = []
        for n in range(photon_number):
            prep_gN += [
                ['multiphoton', f'g{n}-e{n}', 'pi', 0.0],
                ['multiphoton', f'e{n}-f{n}', 'pi', 0.0],
                ['multiphoton', f'f{n}-g{n + 1}', 'pi', 0.0],
            ]
        prep_eN = prep_gN + [[
            'multiphoton', f'g{photon_number}-e{photon_number}', 'pi', 0.0
        ]]

        print(f'Running broadband Rabi from |g,{photon_number}>')
        rabi_from_g[photon_number] = runner.execute(
            sigma_test=broadband_sigma,
            user_defined_freq=[True, broadband_frequency],
            pre_sweep_pulse=prep_gN,
            show=False,
            log=True,
        )
        if show_each:
            rabi_from_g[photon_number].display()

        print(f'Running broadband Rabi from |e,{photon_number}>')
        rabi_from_e[photon_number] = runner.execute(
            sigma_test=broadband_sigma,
            user_defined_freq=[True, broadband_frequency],
            pre_sweep_pulse=prep_eN,
            show=False,
            log=True,
        )
        if show_each:
            rabi_from_e[photon_number].display()

    return rabi_from_g, rabi_from_e


def plot_broadband_rabi_transfer(rabi_from_g, rabi_from_e, photon_numbers):
    """Put every photon number on its own g/e readout axis and plot (cell 13).

    Returns (gain, g_to_e, e_to_g); the two arrays are indexed
    [photon number row, gain point].
    """
    gain = np.asarray(rabi_from_g[0].data['xpts'], dtype=float)
    g_to_e = np.zeros((len(photon_numbers), len(gain)))
    e_to_g = np.zeros_like(g_to_e)

    for row, photon_number in enumerate(photon_numbers):
        z_from_g = (
            np.asarray(rabi_from_g[photon_number].data['avgi']) +
            1j * np.asarray(rabi_from_g[photon_number].data['avgq'])
        )
        z_from_e = (
            np.asarray(rabi_from_e[photon_number].data['avgi']) +
            1j * np.asarray(rabi_from_e[photon_number].data['avgq'])
        )
        z_g = z_from_g[0]
        z_e = z_from_e[0]

        g_to_e[row] = ge_population_transfer(z_from_g, z_g, z_e)
        e_to_g[row] = 1 - ge_population_transfer(z_from_e, z_g, z_e)

    plt.figure(figsize=(10, 6))
    for row, photon_number in enumerate(photon_numbers):
        plt.plot(gain, g_to_e[row], label=f'|g,{photon_number}> to e')
        plt.plot(gain, e_to_g[row], '--', label=f'|e,{photon_number}> to g')
    plt.axhline(1, color='0.7')
    plt.xlabel('gain')
    plt.ylabel('population-transfer coordinate')
    plt.ylim(-0.1, 1.1)
    plt.legend(ncol=2)
    plt.show()

    return gain, g_to_e, e_to_g


def fit_broadband_frequency(runner, center, band, expts=51, reps=50,
                            n_pulses=10, log=True):
    """Error-amplified broadband frequency scan and Gaussian fit.

    Cells 18 (band=5.0, coarse) and 21 (band=1.0, fine), which were otherwise
    identical. Does not write the result into the config -- accepting the fit
    is left to the notebook.

    Returns (expt, best_frequency, best_frequency_err) in MHz.
    """
    expt = runner.execute(
        parameter_to_test='frequency',
        start=center - band,
        step=2 * band / (expts - 1),
        expts=expts,
        reps=reps,
        n_pulses=n_pulses,
        pulse_type=['qubit', 'ge_broadband', 'pi', 0],
        postprocess=False,
        show=False,
        log=log,
    )

    x = np.asarray(expt.data['x_pts'], dtype=float)
    y = np.asarray(expt.data['prod_avgi'], dtype=float)

    p, pcov = fitter.fitgaussian(x, y, periodic=False)
    best_frequency = float(p[2])
    best_frequency_err = float(np.sqrt(np.diag(pcov))[2])

    expt.data['fit_avgi'] = p
    expt.data['prod_avgi_fit'] = fitter.gaussianfunc(x, *p)
    expt.data['fit_prod_avgi_err'] = np.sqrt(np.diag(pcov))

    expt.display()
    print(f'current broadband frequency: {center:.6f} MHz')
    print(
        f'N=0 fitted frequency: {best_frequency:.6f} '
        f'+/- {best_frequency_err:.6f} MHz'
    )
    return expt, best_frequency, best_frequency_err


def fit_broadband_gain(runner, current_gain, gain_half_band, gain_limit,
                       gain_expts=26, reps=50, n_pulses=10, log=True):
    """Error-amplified broadband gain scan and Gaussian fit.

    Cells 19 (gain_half_band=3000, coarse) and 22 (1500, fine). Their only
    other difference was that cell 19 passed log=False, deferring the log
    until after the refit; that is now the `log` argument.

    Returns (expt, best_gain, best_gain_err, fitted_gain). `best_gain` is the
    integer candidate, clipped to the scanned range.
    """
    gain_width = 2 * gain_half_band
    gain_start = max(0, min(current_gain - gain_half_band, gain_limit - gain_width))
    gain_step = gain_width // (gain_expts - 1)
    gain_stop = gain_start + gain_step * (gain_expts - 1)

    print(
        f'broadband gain scan: {gain_start} ... {gain_stop} '
        f'(step {gain_step}, limit {gain_limit})'
    )

    expt = runner.execute(
        parameter_to_test='gain',
        start=gain_start,
        step=gain_step,
        expts=gain_expts,
        reps=reps,
        n_pulses=n_pulses,
        pulse_type=['qubit', 'ge_broadband', 'pi', 0],
        postprocess=False,
        show=False,
        log=log,
    )

    x_gain = np.asarray(expt.data['x_pts'], dtype=float)
    y_gain = np.asarray(expt.data['prod_avgi'], dtype=float)

    p_gain, pcov_gain = fitter.fitgaussian(x_gain, y_gain, periodic=False)
    fitted_gain = float(p_gain[2])
    best_gain = int(round(np.clip(fitted_gain, gain_start, gain_stop)))
    best_gain_err = float(np.sqrt(np.diag(pcov_gain))[2])

    expt.data['fit_avgi'] = p_gain
    expt.data['prod_avgi_fit'] = fitter.gaussianfunc(x_gain, *p_gain)
    expt.data['fit_prod_avgi_err'] = np.sqrt(np.diag(pcov_gain))

    expt.display()
    print(f'current broadband gain: {current_gain}')
    print(f'N=0 fitted gain: {fitted_gain:.1f} +/- {best_gain_err:.1f}')
    print(f'capped integer gain candidate: {best_gain}')
    return expt, best_gain, best_gain_err, fitted_gain


def plot_broadband_validation(expt, photon_numbers, validation_cases):
    """Exact-pulse-path validation of pi_ge_broadband (cell 26).

    Case 0/1 are the |g,n>/|e,n> references; case 2/3 are the two transfers.
    Returns `validation_transfer`, shaped (photon number, 2) as
    [g->e, e->g].
    """
    validation_iq = (
        np.asarray(expt.data['avgi'], dtype=float) +
        1j * np.asarray(expt.data['avgq'], dtype=float)
    ).reshape(len(photon_numbers), len(validation_cases))

    validation_transfer = np.zeros((len(photon_numbers), 2))
    for row, photon_number in enumerate(photon_numbers):
        z_g, z_e, z_g_to_e, z_e_to_g = validation_iq[row]
        validation_transfer[row, 0] = ge_population_transfer(z_g_to_e, z_g, z_e)
        validation_transfer[row, 1] = 1 - ge_population_transfer(z_e_to_g, z_g, z_e)

    plt.figure(figsize=(7, 4))
    plt.plot(photon_numbers, validation_transfer[:, 0], 'o-', label='g to e')
    plt.plot(photon_numbers, validation_transfer[:, 1], 'o--', label='e to g')
    plt.axhline(1, color='0.7')
    plt.xlabel('manipulate photon number')
    plt.ylabel('population-transfer coordinate')
    plt.xticks(photon_numbers)
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()

    for row, photon_number in enumerate(photon_numbers):
        print(
            f'N={photon_number}: '
            f'g->e={validation_transfer[row, 0]:.4f}, '
            f'e->g={validation_transfer[row, 1]:.4f}'
        )
    return validation_transfer


# --------------------------------------------------------------------------
# N-photon M1-storage swap calibration (cells 32-55).
# --------------------------------------------------------------------------


def build_swap_pulse_sequences(station, photon_number):
    """Preparation and endpoint-decoder pulses for one swap row (cell 32).

    The decoder uses only the top two transitions, so only an exact return to
    |g,N> is marked bright and lower M1 occupations stay dark rather than
    being mistaken for a successful return.

    Returns a dict with the four sequences the later cells use:
    `prep_descriptions`, `extra_prep` (what ErrorAmplificationProgram does not
    already supply), `endpoint_decoder_descriptions`, `prep_pulse` and
    `endpoint_decoder`.
    """
    prep_descriptions = [
        ['qubit', 'ge', 'pi', 0.0],
        ['qubit', 'ef', 'pi', 0.0],
        ['man', 'M1', 'pi', 0.0],
    ]
    for n in range(1, photon_number):
        prep_descriptions += [
            ['multiphoton', f'g{n}-e{n}', 'pi', 0.0],
            ['multiphoton', f'e{n}-f{n}', 'pi', 0.0],
            ['multiphoton', f'f{n}-g{n + 1}', 'pi', 0.0],
        ]

    # ErrorAmplificationProgram already supplies the first three storage-prep
    # pulses.
    extra_prep = prep_descriptions[3:]

    n_top = photon_number - 1
    endpoint_decoder_descriptions = [
        ['multiphoton', f'f{n_top}-g{n_top + 1}', 'pi', 0.0],
        ['multiphoton', f'e{n_top}-f{n_top}', 'pi', 0.0],
    ]

    # Imported here rather than at module scope: importing experiments.* at
    # module import time is what the notebook did, but it is slow and this
    # module is otherwise import-light.
    from experiments.MM_dual_rail_base import MM_dual_rail_base

    pulse_builder = MM_dual_rail_base(station.hardware_cfg, soccfg=station.soccfg)
    prep_pulse = pulse_builder.get_prepulse_creator(
        prep_descriptions
    ).pulse.tolist()
    endpoint_decoder = pulse_builder.get_prepulse_creator(
        endpoint_decoder_descriptions
    ).pulse.tolist()

    print('preparation:')
    for pulse in prep_descriptions:
        print(' ', pulse)
    print('endpoint decoder:')
    for pulse in endpoint_decoder_descriptions:
        print(' ', pulse)

    return {
        'prep_descriptions': prep_descriptions,
        'extra_prep': extra_prep,
        'endpoint_decoder_descriptions': endpoint_decoder_descriptions,
        'prep_pulse': prep_pulse,
        'endpoint_decoder': endpoint_decoder,
    }


def analyze_swap_chevron(expt, station, pulse_name):
    """Fit the frequency-length Chevron (cell 36).

    Prints candidates only; the accept cell in the notebook is what writes
    them into ds_storage.

    Returns (analysis, frequency_candidate_MHz, pi_candidate_us).
    """
    from fitting.fit_display_classes import ChevronFitting

    chevron_time = np.asarray(expt.data['xpts'], dtype=float)
    if chevron_time.ndim > 1:
        chevron_time = chevron_time[0]

    analysis = ChevronFitting(
        frequencies=np.asarray(expt.data['freq_sweep'], dtype=float),
        time=chevron_time,
        response_matrix=np.asarray(expt.data['avgi'], dtype=float),
        config=station.hardware_cfg,
        station=station,
    )
    analysis.analyze()
    analysis.display_results(title=expt.fname)

    frequency_candidate_MHz = float(
        analysis.results['best_frequency_contrast']
    )
    pi_candidate_us = abs(
        np.pi / analysis.results['best_fit_params_period']['omega']
    )
    print('Chevron candidate:', frequency_candidate_MHz, 'MHz,',
          pi_candidate_us, 'us')
    print('Inspect the plot before running the accept cell.')
    return analysis, frequency_candidate_MHz, pi_candidate_us


def even_gain_grid(center, half_span, step, gain_limit):
    """Even-valued gain sweep grid (cells 45 and 48).

    Start and stop are rounded to even gains because the flat-top program also
    uses a half-gain register, and the stop is capped at `gain_limit`.

    Returns (start, stop, step, points).
    """
    start = max(0, 2 * round((center - half_span) / 2))
    stop = min(gain_limit, 2 * round((center + half_span) / 2))
    points = (stop - start) // step + 1
    stop = start + step * (points - 1)
    return start, stop, step, points


def scan_return_error(runner, parameter_to_test, start, step, expts,
                      n_pulses, reps, xlabel, title, as_int=False):
    """Sweep one swap parameter and score it by return IQ error.

    Cells 42, 45, 48 and 51 -- coarse frequency, coarse gain, fine gain, fine
    frequency -- were the same body with different scan settings. Every depth
    uses an even number of physical swaps, and row 0 is the in-situ
    preparation/readout reference, so the score is the mean IQ distance of the
    deeper rows from that reference. Lower is better.

    Prints a candidate only; the notebook's accept cell writes it.

    Returns (expt, x, return_error, candidate).
    """
    expt = runner.execute(
        parameter_to_test=parameter_to_test,
        start=start,
        step=step,
        expts=expts,
        n_start=0,
        n_step=1,
        n_pulses=n_pulses,
        reps=reps,
        postprocess=False,
        show=False,
        log=True,
    )
    expt.display(fit=False)

    x = np.asarray(expt.data['x_pts'], dtype=float)
    z = (
        np.asarray(expt.data['avgi'], dtype=float)
        + 1j * np.asarray(expt.data['avgq'], dtype=float)
    )
    return_error = np.mean(np.abs(z[1:] - z[0]) ** 2, axis=0)
    candidate = x[np.argmin(return_error)]
    candidate = int(candidate) if as_int else float(candidate)

    plt.figure(figsize=(6, 3.5))
    plt.plot(x, return_error, 'o-')
    plt.axvline(candidate, color='black', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel('mean return IQ error')
    plt.title(f'{title}; inspect before accepting')
    plt.grid()
    plt.show()
    print(f'{title} candidate:', candidate)
    return expt, x, return_error, candidate


def plot_iq_endpoints(expt, labels, title):
    """IQ scatter of a two-point validation sweep (cells 54 and 55).

    The odd check compares zero and one swap: the endpoint marker should go
    bright to dark. The even check compares zero and two swaps: it should
    return to the same IQ point. Together they separate coherent transfer from
    plain photon loss.

    Returns (z, separation) where separation is |z[1] - z[0]|.
    """
    z = (
        np.asarray(expt.data['avgi'], dtype=float).reshape(-1)
        + 1j * np.asarray(expt.data['avgq'], dtype=float).reshape(-1)
    )
    plt.figure(figsize=(5, 4))
    plt.plot(z.real, z.imag, 'o-')
    for label, value in zip(labels, z):
        plt.annotate(label, (value.real, value.imag))
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title(title)
    plt.grid()
    plt.show()
    separation = abs(z[1] - z[0])
    return z, separation
