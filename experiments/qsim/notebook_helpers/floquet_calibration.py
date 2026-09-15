"""Hooks for the Floquet pulse calibration notebook.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
58-60 and 69-122 by the stage-2 notebook decomposition. Primary caller:
`measurement_notebooks/202609_qsim_migration/floquet_calibration.py`.

The source ran the whole calibration twice: once for the legacy flat-top pulse
(cells 71-96) and once for the Gaussian/preloaded envelope (cells 100-122), and
it redefined the hooks identically in both halves. Only that duplication is
collapsed here:

- `error_amp_floquet_preproc` / `error_amp_floquet_postproc` -- cells 78 and
  107 defined these twice with byte-identical bodies, so there is one copy.
  Same for `sideband_stark_error_amp_preproc` / `..._postproc` (cells 88 and
  115) and for `floquet_cycle_list_gen` (cells 151, 152 and 287, plus a fourth
  copy in data_postprocess cell 4 -- all four identical).
- The Chevron hooks stay as two pairs. `floquet_freq_chev_postproc` accepts a
  pi/N *length* from the period fit and `floquet_gain_chev_postproc` a pi/N
  *gain* from the contrast fit, so they are not duplicates.

**The sweep loops belong in the notebook, not here.** An earlier pass pulled
them into `run_freq_chevron_sweep`, `run_gain_chevron_sweep`,
`run_floquet_error_amp_sweep` and `run_phase_accumulation_pairs`. Each took a
closed list of keyword arguments and forwarded a hand-picked subset to
`runner.execute`, so a notebook cell could no longer pass `use_queue`,
`priority`, `go_kwargs` or any other expt_cfg override -- it got a TypeError
instead. They also restated defaults that the defaults dicts already held.
The loops are back inline in the notebook, where defaults -> runner -> execute
reads end to end. Do not hoist them again: a `for` loop around
`runner.execute(...)` is the canonical notebook cell, not duplication.

The defaults dicts stay in the notebook too. They are the settings a
calibration session edits, and the two halves genuinely differ in them -- cell
107 applies `floquet_default_dict` to the error-amplification defaults where
cell 78 set `floquet_waveform` alone.

`error_amp_floquet_preproc` needs the two coarse-span dicts that cell 78 had
at notebook scope. They are arguments rather than module constants, so those
spans stay visible and editable in the notebook; the notebook binds them with
`functools.partial` before handing the hook to the runner.

Temporary home, per the stage-2 instructions. Nothing here was reconciled with
the existing library calibration code.
"""

from copy import deepcopy

import numpy as np

from experiments.MM_dual_rail_base import MM_dual_rail_base


# --------------------------------------------------------------------------
# Cell 60.
# --------------------------------------------------------------------------


def get_floquet_parameters(station, man_mode_no, stor_mode_no):
    """
    Get pulse parameters for a given storage mode.
    Also returns prepulse and postpulse (single photon prep and meas for ge meas)

    Args:
        station: MultimodeStation object for managing frequency data.
        man_mode_no: Manipulation mode number.
        stor_mode_no: Storage mode number.

    Returns:
        A tuple containing freq, gain, ch, prepulse, and postpulse.
    """
    stor_name = 'M' + str(man_mode_no) + '-S' + str(stor_mode_no)
    freq = station.ds_floquet.get_freq(stor_name)
    gain = station.ds_floquet.get_gain(stor_name)
    length = station.ds_floquet.get_len(stor_name)
    pi_frac = station.ds_floquet.get_pi_frac(stor_name)
    ch = 'low' if freq < 1000 else 'high'

    mm_base_dummy = MM_dual_rail_base(station.hardware_cfg, station.soccfg)
    prep_man_pi = mm_base_dummy.prep_man_photon(man_mode_no)
    prepulse = mm_base_dummy.get_prepulse_creator(prep_man_pi).pulse.tolist()
    postpulse = mm_base_dummy.get_prepulse_creator(prep_man_pi[-1:-3:-1]).pulse.tolist() # for ge meas, only do f0g1 and ef pi

    return freq, gain, length, pi_frac, ch, prepulse, postpulse


# --------------------------------------------------------------------------
# Chevron hooks (cells 71 and 100). The two postprocessors look alike but are
# not duplicates: the frequency-length one accepts a pi/N *length* from the
# period fit, the gain one accepts a pi/N *gain* from the contrast fit. Kept
# separate, per the instruction not to decide which algorithm is better.
# --------------------------------------------------------------------------


def floquet_freq_chev_preproc(station, default_expt_cfg, **kwargs):
    assert 'init_stor' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    ds_floquet = station.ds_floquet
    init_stor = kwargs.pop('init_stor') # storage mode number to initialize to n=1 Fock state
    lengths = np.linspace(0.01, 3.0 * ds_floquet.get_len(f'M1-S{init_stor}'), 10).tolist()

    expt_cfg.init_stor = init_stor
    expt_cfg.lengths = lengths
    expt_cfg.update(kwargs)
    print(expt_cfg)
    return expt_cfg


def floquet_freq_chev_postproc(station, expt):
    expt_cfg = expt.cfg.expt
    stor_name = f'M{expt_cfg.f0g1_cavity}-S{expt_cfg.init_stor}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=np.array(expt.data['ypts']),
        time=np.array(expt.data['xpts']),
        response_matrix=expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()

    best_detune = chevron_analysis.results.get('best_frequency_contrast')

    if best_detune is not None:
        pi_frac = station.ds_floquet.get_pi_frac(stor_name)
        print(f"Best detune found: {best_detune:.4f} MHz")
        current_freq = station.ds_floquet.get_freq(stor_name)
        new_freq = current_freq + best_detune
        station.ds_floquet.update_freq(stor_name, new_freq)
        print(f"Updated {stor_name} frequency to {new_freq:.4f} MHz")
        frac_pi_len = abs(np.pi / chevron_analysis.results['best_fit_params_period']['omega'])
        station.ds_floquet.update_len(stor_name, frac_pi_len)
        print(f'Updated the pi/{pi_frac} length from {station.ds_floquet.get_len(stor_name):.4f} to {frac_pi_len:.4f}')

    chevron_analysis.display_results()
    expt.analysis = chevron_analysis
    station.snapshot_floquet_storage_swap(update_main=False)


def floquet_gain_chev_preproc(station, default_expt_cfg, **kwargs):
    assert 'init_stor' in kwargs
    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    ds_floquet = station.ds_floquet
    init_stor = kwargs.pop('init_stor') # storage mode number to initialize to n=1 Fock state
    # gains = np.linspace(0.01, 3.0 * ds_floquet.get_gain(f'M1-S{init_stor}'), 200).tolist()
    max_gain = expt_cfg.get("max_gain", 15000)
    gain_expts = expt_cfg.get("gain_expts", 11)
    if max_gain != 15000:
        gains = np.linspace(0, max_gain, gain_expts).astype(int).tolist()
    else:
        gains = np.linspace(0, np.minimum(10.0 * ds_floquet.get_gain(f'M1-S{init_stor}'), max_gain), gain_expts).astype(int).tolist()

    expt_cfg.init_stor = init_stor
    expt_cfg.gains = gains
    print(expt_cfg)
    return expt_cfg


def floquet_gain_chev_postproc(station, expt):
    expt_cfg = expt.cfg.expt
    stor_name = f'M{expt_cfg.f0g1_cavity}-S{expt_cfg.init_stor}'

    from fitting.fit_display_classes import ChevronFitting

    chevron_analysis = ChevronFitting(
        frequencies=np.array(expt.data['ypts']),
        time=np.array(expt.data['xpts']),
        response_matrix=expt.data['avgi'],
        config=station.hardware_cfg,
        station=station,
    )

    chevron_analysis.analyze()

    best_detune = chevron_analysis.results.get('best_frequency_contrast')

    if best_detune is not None:
        pi_frac = station.ds_floquet.get_pi_frac(stor_name)
        print(f"Best detune found: {best_detune:.4f} MHz")
        current_freq = station.ds_floquet.get_freq(stor_name)
        new_freq = current_freq + best_detune
        station.ds_floquet.update_freq(stor_name, new_freq)
        print(f"Updated {stor_name} frequency to {new_freq:.4f} MHz")
        frac_pi_gain = abs((np.pi)/ chevron_analysis.results['best_fit_params_contrast']['omega'])
        station.ds_floquet.update_gain(stor_name, frac_pi_gain)
        print(f'Updated the pi/{pi_frac} gain from {station.ds_floquet.get_gain(stor_name):.4f} to {frac_pi_gain:.4f}')

    chevron_analysis.display_results()
    expt.analysis = chevron_analysis
    station.snapshot_floquet_storage_swap(update_main=False)


# --------------------------------------------------------------------------
# Error-amplification hooks (cells 78 and 107, identical bodies).
# --------------------------------------------------------------------------


def error_amp_floquet_preproc(station, default_expt_cfg,
                              gain_coarse_defaults=None,
                              freq_coarse_defaults=None,
                              **kwargs):
    """Pick the coarse span block, then resolve the scan start and step.

    `gain_coarse_defaults` and `freq_coarse_defaults` were notebook-scope
    dicts in cells 78 and 107. They are arguments so the spans stay visible in
    the notebook; bind them with functools.partial before passing this to a
    runner. Both are required in practice -- they are keyword arguments with
    None defaults only so the signature matches the runner's calling
    convention.
    """
    assert 'stor_mode_no' in kwargs
    assert 'parameter_to_test' in kwargs

    # construct the defaults
    expt_cfg = deepcopy(default_expt_cfg)
    if kwargs['parameter_to_test'] == 'gain':
        if gain_coarse_defaults is None:
            raise ValueError('gain_coarse_defaults is required for a gain scan')
        expt_cfg.update(gain_coarse_defaults)
    elif kwargs['parameter_to_test'] == 'frequency':
        if freq_coarse_defaults is None:
            raise ValueError('freq_coarse_defaults is required for a frequency scan')
        expt_cfg.update(freq_coarse_defaults)
    # override with the passed kwargs
    expt_cfg.update(kwargs)

    freq, gain, length, pi_frac, ch, prepulse, postpulse = get_floquet_parameters(station, expt_cfg.man_mode_no, expt_cfg.stor_mode_no)
    pulse_type = ['floquet', f'M{expt_cfg.man_mode_no}-{"D" if expt_cfg.stor_is_dump else "S"}{expt_cfg.stor_mode_no}', f'pi/{pi_frac}', 0]
    # freq = 695.7
    # gain = 10000
    if expt_cfg.parameter_to_test == 'frequency':
        start = freq - expt_cfg.span / 2
        step = expt_cfg.span / (expt_cfg.expts - 1)
    elif expt_cfg.parameter_to_test == 'gain':
        start = int(gain - expt_cfg.span / 2)
        step = int(expt_cfg.span / (expt_cfg.expts - 1))
    else:
        raise ValueError("parameter_to_test must be either 'frequency' or 'gain'.")
    expt_cfg.start = start
    expt_cfg.step = step
    expt_cfg.pulse_type = pulse_type
    return expt_cfg


def error_amp_floquet_postproc(station, expt):
    expt.analyze(data=expt.data, state_fin='e')

    opt_val = expt.data['fit_avgi'][2]
    stor_name = 'M1-S' + str(expt.cfg.expt.stor_mode_no)
    if expt.cfg.expt.parameter_to_test == 'gain':
        station.ds_floquet.update_gain(stor_name, opt_val)
        print(f'Updated gain for {stor_name} to {opt_val}')
    elif expt.cfg.expt.parameter_to_test == 'frequency':
        station.ds_floquet.update_freq(stor_name, opt_val)
        print(f'Updated frequency for {stor_name} to {opt_val}')
    station.snapshot_floquet_storage_swap(update_main=False)


# --------------------------------------------------------------------------
# Stark-shift phase accumulation hooks (cells 88 and 115, identical bodies).
# --------------------------------------------------------------------------


def sideband_stark_error_amp_preproc(station, default_expt_cfg, **kwargs):
    assert 'stor_A' in kwargs
    assert 'stor_B' in kwargs

    expt_cfg = deepcopy(default_expt_cfg)
    expt_cfg.update(kwargs)
    print(expt_cfg)
    return expt_cfg


def sideband_stark_error_amp_postproc(station, expt):
    storA = expt.cfg.expt.stor_A
    storB = expt.cfg.expt.stor_B
    stor_name = 'M1-S' + str(storA)
    from_stor_name = 'M1-S' + str(storB)
    expt.analyze(fit=True)
    expt.display(fit=True)
    # opt_phase = expt.data['fit_avgi'][2] / 2 # divide by 2 since did pi/12, -pi/12 on the from_stor swap
    opt_phase = expt.data['fit_avgi'][2] # not dividing seems to give the correct result somehow? TODO: figure out why
    print("Opt phase on", stor_name, "from", from_stor_name, ":", opt_phase)
    station.ds_floquet.update_phase_from(stor_name, from_stor_name, opt_phase)
    # Legacy block: snapshot intentionally disabled.


# --------------------------------------------------------------------------
# Cell 151/152/287: identical in all three places.
# --------------------------------------------------------------------------


def floquet_cycle_list_gen(start,
                           stop,
                           chunk,
                           step = 1):
    _to_return = []
    _q = (stop-start)//chunk
    _start_idx = start
    for _ in range(_q):
        _to_return.append(np.arange(_start_idx,
                                    _start_idx+chunk,
                                    step))
        _start_idx += chunk
    if _start_idx < stop:
        _to_return.append(np.arange(_start_idx,
                                    stop,
                                    step))
    return _to_return
