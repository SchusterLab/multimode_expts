"""Hooks and sweep loops for the Floquet pulse calibration notebook.

Hoisted out of `measurement_notebooks/jonginn/qsim_experiments.ipynb` cells
58-60 and 69-122 by the stage-2 notebook decomposition. Primary caller:
`measurement_notebooks/202609_qsim_migration/floquet_calibration.py`.

The source ran the whole calibration twice: once for the legacy flat-top pulse
(cells 71-96) and once for the Gaussian/preloaded envelope (cells 100-122).
That is why so much here is one function where the notebook had several cells.
Each collapse below was checked against the source rather than assumed, and the
differences that were real became arguments:

- `error_amp_floquet_preproc` / `error_amp_floquet_postproc` -- cells 78 and
  107 defined these twice with byte-identical bodies, so there is one copy.
  Same for `sideband_stark_error_amp_preproc` / `..._postproc` (cells 88 and
  115) and for `floquet_cycle_list_gen` (cells 151, 152 and 287, plus a fourth
  copy in data_postprocess cell 4 -- all four identical).
- `run_floquet_error_amp_sweep` is cells 81, 83, 109 and 113. Four copies of
  one loop. What actually differed: the fine passes halve both spans
  (`span_divisor`), and the two halves disagreed on `reset_dump_mode` and on
  whether `relax_delay` was derived from `active_reset`.
- `run_phase_accumulation_pairs` is cells 89, 91, 116, 119 and 121. Five
  copies. What differed: the mode lists, the `advance_phases` grid, which
  Floquet buffer flags were forwarded, and -- only in cell 121 -- skipping
  pairs where both modes were already calibrated (`pre_existing_modes`).
- `run_freq_chevron_sweep` is cells 72 and 76; cell 76 added a per-mode span
  table, which is now the optional `freq_spans` argument.

The defaults dicts stayed in the notebook. They are the settings a calibration
session edits, and the two halves genuinely differ in them -- cell 107 applies
`floquet_default_dict` to the error-amplification defaults where cell 78 set
`floquet_waveform` alone -- so folding them together here would have merged a
real difference.

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


# --------------------------------------------------------------------------
# The sweep loops.
# --------------------------------------------------------------------------


def run_freq_chevron_sweep(runner, station, stor_modes, freq_spans=None,
                           default_span=1.0, reps=100, relax_delay=200,
                           active_reset=True, reset_dump_mode=1,
                           always_display=False):
    """Frequency-length Chevron for each storage mode (cells 72 and 76).

    Cell 72 swept a fixed +/-1 MHz for every mode; cell 76 added a per-mode
    table, which is `freq_spans` here -- indexed by `stor - 1`, with None
    meaning "use `default_span`". Cell 72 always called display(); cell 76
    only did so when the station was not logging, which is the default here.

    Returns the list of experiments, in `stor_modes` order.
    """
    expts = [None] * len(stor_modes)
    for i, init_stor in enumerate(stor_modes):
        span = default_span
        if freq_spans is not None and freq_spans[init_stor - 1] is not None:
            span = freq_spans[init_stor - 1]
        print(f'Running Floquet Frequency vs Length Chevron for Storage Mode {init_stor}')
        expts[i] = runner.execute(
            init_stor=init_stor,
            detunes=np.linspace(-span, span, 51).tolist(),
            reps=reps,
            relax_delay=relax_delay,
            active_reset=active_reset,
            man_reset=True,
            storage_reset=[init_stor],
            reset_dump_mode=reset_dump_mode,
        )
        if always_display or not station.log_measurements:
            expts[i].display()
    return expts


def run_gain_chevron_sweep(runner, station, stor_modes, detune_span=0.5,
                           reps=50, gain_expts=21, max_gain=14000,
                           relax_delay=200, active_reset=True,
                           reset_dump_mode=1, debug=False):
    """Frequency-gain Chevron for each storage mode (cell 101).

    Returns the list of experiments, in `stor_modes` order.
    """
    expts = [None] * len(stor_modes)
    for i, init_stor in enumerate(stor_modes):
        print(f'Running Floquet Frequency vs Gain Chevron for Storage Mode {init_stor}')
        expts[i] = runner.execute(
            init_stor=init_stor,
            detunes=np.linspace(-detune_span, detune_span, 21).tolist(),
            reps=reps,
            gain_expts=gain_expts,
            max_gain=max_gain,
            relax_delay=relax_delay,
            active_reset=active_reset,
            man_reset=True,
            storage_reset=[init_stor],
            reset_dump_mode=reset_dump_mode,
            debug=debug,
        )
        if not station.log_measurements:
            expts[i].display()
    return expts


def run_floquet_error_amp_sweep(runner, station, stor_modes,
                                freq_span_list, gain_span_list,
                                reset_dump_mode,
                                freq_span_default=0.1, gain_span_default=0.3,
                                span_divisor=1,
                                do_freq_erroramp=True, do_gain_erroramp=True,
                                active_reset=True, relax_delay=200,
                                gain_expts=60):
    """Error-amplify frequency and gain for each storage mode.

    Cells 81, 83, 109 and 113 -- the coarse and fine passes of both the
    flat-top and the Gaussian halves -- were four copies of this loop. The
    fine passes halve both spans, which is `span_divisor=2`. The two halves
    disagreed about `reset_dump_mode` (a literal 1 in the flat-top half, the
    shared default in the Gaussian half), so it is a required argument rather
    than something guessed here.

    `freq_span_list` and `gain_span_list` are indexed by `stor - 1`, with None
    meaning "use the corresponding default".

    Returns (freq_expts, gain_expts_out), each a list in `stor_modes` order
    with None where that scan was skipped.
    """
    freq_expts = [None] * len(stor_modes)
    gain_expts_out = [None] * len(stor_modes)

    for i, stor_i in enumerate(stor_modes):
        stor_idx = stor_i - 1
        stor_name = 'M1-S' + str(stor_i)

        freq_span = (freq_span_default if freq_span_list[stor_idx] is None
                     else freq_span_list[stor_idx])
        gain_span = (gain_span_default if gain_span_list[stor_idx] is None
                     else gain_span_list[stor_idx])

        if do_freq_erroramp:
            freq_expts[i] = runner.execute(
                stor_mode_no=stor_i,
                parameter_to_test='frequency',
                go_kwargs=dict(analyze=False, progress=True, display=False),
                # 0.2 is the program default, but we are already close.
                span=freq_span / span_divisor,
                relax_delay=relax_delay,
                active_reset=active_reset,
                man_reset=True,
                storage_reset=[stor_i],
                reset_dump_mode=reset_dump_mode,
            )
            if not station.log_measurements:
                freq_expts[i].display()

        if do_gain_erroramp:
            gain_expts_out[i] = runner.execute(
                stor_mode_no=stor_i,
                parameter_to_test='gain',
                go_kwargs=dict(analyze=False, progress=True, display=False),
                # 0.7 is the program default.
                span=int(station.ds_floquet.get_gain(stor_name)
                         * gain_span / span_divisor),
                expts=gain_expts,
                relax_delay=relax_delay,
                active_reset=active_reset,
                man_reset=True,
                storage_reset=[stor_i],
                reset_dump_mode=reset_dump_mode,
            )
            if not station.log_measurements:
                gain_expts_out[i].display()

    return freq_expts, gain_expts_out


def run_phase_accumulation_pairs(runner, stor_modes_to, stor_modes_from,
                                 advance_phases, reset_dump_mode,
                                 floquet_settings,
                                 phase_expts=None,
                                 reps=100, relax_delay=100, active_reset=True,
                                 pre_existing_modes=None,
                                 forward_pi_half_buffer=True,
                                 forward_sync_cycles=True):
    """Measure the Stark-shift phase each mode accumulates from another.

    Cells 89, 91, 116, 119 and 121 were five copies of this double loop. What
    genuinely differed between them and is therefore an argument here: the two
    mode lists, the `advance_phases` grid, `reset_dump_mode`, which Floquet
    buffer flags were forwarded to the runner, and -- only in cell 121 --
    skipping pairs where both modes were already calibrated.

    `floquet_settings` is the notebook's `floquet_default_dict`. Only cells 116
    onward forwarded `include_10cycles_buffer_in_pi_half` and
    `scramble_sync_cycles`; cells 89 and 91 did not, so those two are behind
    flags rather than always sent.

    `pre_existing_modes` reproduces cell 121: a pair is skipped when both of
    its modes are in that list, so only the newly added modes are measured.

    `phase_expts` is the 7x7 result grid; a fresh one is made if not given.
    Returns it, indexed [stor_A - 1][stor_B - 1].
    """
    if phase_expts is None:
        phase_expts = [[None for _ in range(7)] for _ in range(7)]
    pre_existing = set(pre_existing_modes or ())

    for init_storA in stor_modes_to:
        for init_storB in stor_modes_from:
            if init_storA == init_storB:
                continue
            if init_storA in pre_existing and init_storB in pre_existing:
                continue

            print("Starting experiment for storage modes:", init_storA,
                  "from", init_storB)

            extra = {}
            if forward_pi_half_buffer:
                extra['include_10cycles_buffer_in_pi_half'] = floquet_settings[
                    "include_10cycles_buffer_in_pi_half"
                ]
            if forward_sync_cycles:
                extra['scramble_sync_cycles'] = floquet_settings[
                    "scramble_sync_cycles"
                ]

            qbe = runner.execute(
                stor_A=init_storA,
                stor_B=init_storB,
                relax_delay=relax_delay,
                active_reset=active_reset,
                man_reset=True,
                storage_reset=[init_storA, init_storB],
                reset_dump_mode=reset_dump_mode,
                reps=reps,
                include_10cycles_buffer=floquet_settings[
                    "include_10cycles_buffer"
                ],
                advance_phases=advance_phases,
                **extra,
            )
            phase_expts[init_storA - 1][init_storB - 1] = qbe

    return phase_expts
