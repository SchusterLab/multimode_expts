"""Build every stage-2 measurement theme against mocked instruments.

Per `docs/reference/mock_mode_architecture.md` the qick program build and ASM
compile are real in mock mode, so these tests exercise the layer a notebook
refactor is most likely to break: whether each theme's config still assembles
into a program the qick validators accept. Nothing reaches the FPGA and
nothing is submitted to the job queue.

Two shapes of test, because the runners differ:

- Themes driven by `CharacterizationRunner` run end to end: its `execute()`
  switches to `run_local()` when `station.is_mock`.
- The MBR themes use `BatchRunner`, which is queue-only. They are validated by
  building their configs through the refactored helpers and then instantiating
  and compiling the program directly -- the same qick path, without a queue.

These are slow-ish (each builds a station from versioned configs) and depend
on config versions present on the measurement PC, so they skip cleanly
elsewhere.
"""
import numpy as np
import pytest
from slab import AttrDict

CONFIG_DICT = {
    "hardware_config": "CFG-HW-20260904-00019",
    "multiphoton_config": "CFG-MP-20260121-00001",
    "man1_storage_swap": "CFG-M1-20260904-00014",
    "floquet_storage_swap": "CFG-FL-20260904-00042",
}


@pytest.fixture(scope="module")
def mock_session():
    """A mock station plus job client, built once for the module."""
    pytest.importorskip("qick")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from experiments.qsim.notebook_helpers.qsim_session import open_session

    try:
        session = open_session(
            user="pytest",
            experiment_name="260818_qsim_spectroscopy",
            project="EncSpec",
            config_dict=CONFIG_DICT,
            log_measurements=False,
            verbose=False,
            mock=True,
        )
    except Exception as exc:  # missing config versions, no soccfg snapshot, ...
        pytest.skip(f"cannot build a mock station here: {type(exc).__name__}: {exc}")
    assert session.station.is_mock, "open_session(mock=True) did not mock"
    return session


@pytest.fixture(scope="module")
def defaults():
    from experiments.qsim.notebook_helpers.qsim_session import (
        ACTIVE_RESET_DEFAULTS,
        FLOQUET_DEFAULTS,
        MEASUREMENT_CONFIG_DEFAULTS,
    )
    return ACTIVE_RESET_DEFAULTS, FLOQUET_DEFAULTS, MEASUREMENT_CONFIG_DEFAULTS


# --------------------------------------------------------------------------
# CharacterizationRunner themes: these run end to end under mock.
# --------------------------------------------------------------------------


def test_broadband_amplitude_rabi_builds(mock_session, defaults):
    """multiphoton_calibration.py's broadband ge calibration."""
    import experiments as meas
    from experiments import CharacterizationRunner
    from experiments.qsim.notebook_helpers.multiphoton_calibration import (
        broadband_amprabi_preproc,
    )

    station, client = mock_session.station, mock_session.client
    pi_ge = station.hardware_cfg.device.qubit.pulses.pi_ge
    ge_freqs = np.asarray(
        station.hardware_cfg.device.multiphoton.pi["gn-en"].frequency[:4],
        dtype=float,
    )
    broadband_frequency = 0.5 * (ge_freqs.min() + ge_freqs.max())
    broadband_sigma = float(pi_ge.sigma[0] / 3)

    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.AmplitudeRabiExperiment,
        default_expt_cfg=AttrDict(dict(
            start=0, step=2000, expts=4, reps=20, rounds=1,
            sigma_test=broadband_sigma, qubit=0, qubits=[0],
            pulse_type="gauss", flat_length=0,
            user_defined_freq=[True, broadband_frequency],
            checkZZ=False, checkEF=False, pulse_ge_init=False,
            pulse_ge_after=False, normalize=False, single_shot=False,
            prepulse=False, pre_sweep_pulse=[], postpulse=False,
            post_sweep_pulse=[], gate_based=True, active_reset=False,
            relax_delay=2500,
        )),
        preprocessor=broadband_amprabi_preproc,
        job_client=client,
        show=False,
    )
    expt = runner.execute(
        sigma_test=broadband_sigma,
        user_defined_freq=[True, broadband_frequency],
        pre_sweep_pulse=[], show=False, log=False,
    )
    assert expt is not None


def test_single_shot_histogram_builds(mock_session, defaults):
    """multiphoton_calibration.py's single-shot readout calibration."""
    import experiments as meas
    from experiments import CharacterizationRunner
    from experiments.qsim.notebook_helpers.multiphoton_calibration import (
        singleshot_postproc,
    )

    _, _, measurement_defaults = defaults
    station, client = mock_session.station, mock_session.client
    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.HistogramExperiment,
        default_expt_cfg=AttrDict(dict(
            reps=200, relax_delay=500, check_f=False, active_reset=False,
            man_reset=False, storage_reset=False, qubit=0,
            pulse_manipulate=False, prepulse=False, pre_sweep_pulse=None,
            gate_based=True, qubits=[0],
        )),
        postprocessor=singleshot_postproc,
        job_client=client,
        show=False,
    )
    expt = runner.execute(
        check_f=False, active_reset=False, relax_delay=2000,
        avoid_yoko=measurement_defaults["avoid_yoko"],
        postprocess=False, log=False, show=False,
    )
    assert expt is not None


def test_floquet_error_amplification_sweep_builds(mock_session, defaults):
    """floquet_calibration.py's error-amplification cells.

    The loop is inline here because it is inline in the notebook: the cell
    body is what this test has to keep working.
    """
    from functools import partial

    import experiments as meas
    from experiments import CharacterizationRunner
    from experiments.qsim.notebook_helpers.floquet_calibration import (
        error_amp_floquet_postproc,
        error_amp_floquet_preproc,
    )

    active_reset_defaults, floquet_defaults, _ = defaults
    station, client = mock_session.station, mock_session.client

    cfg = AttrDict(dict(
        reps=20, rounds=1, qubits=[0], active_reset=False, man_mode_no=1,
        stor_is_dump=False, man_reset=True, storage_reset=True,
        relax_delay=2500, expts=5, qubit_start_storage="g",
        floquet_waveform=floquet_defaults["floquet_waveform"],
        floquet_hardware_loop=floquet_defaults["floquet_hardware_loop"],
        scramble_sync_cycles=floquet_defaults["scramble_sync_cycles"],
    ))
    cfg.update(active_reset_defaults)

    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.single_qubit.error_amplification.ErrorAmplificationExperiment,
        default_expt_cfg=cfg,
        preprocessor=partial(
            error_amp_floquet_preproc,
            gain_coarse_defaults=AttrDict(dict(n_pulses=2, span=4000, expts=5)),
            freq_coarse_defaults=AttrDict(dict(n_pulses=2, span=0.25, expts=5)),
        ),
        postprocessor=error_amp_floquet_postproc,
        job_client=client,
        show=False,
    )
    stor_modes_to_run = [1]
    freq_span_list = [0.1] * 7
    gain_span_list = [None] * 7
    freq_span_default = 0.1
    gain_span_default = 0.3
    span_divisor = 1

    freq_expts = [None] * len(stor_modes_to_run)
    gain_expts = [None] * len(stor_modes_to_run)
    for i, stor_i in enumerate(stor_modes_to_run):
        stor_name = f"M1-S{stor_i}"
        freq_span = freq_span_list[stor_i - 1]
        if freq_span is None:
            freq_span = freq_span_default
        gain_span = gain_span_list[stor_i - 1]
        if gain_span is None:
            gain_span = gain_span_default

        freq_expts[i] = runner.execute(
            stor_mode_no=stor_i,
            parameter_to_test="frequency",
            go_kwargs=dict(analyze=False, progress=False, display=False),
            span=freq_span / span_divisor,
            relax_delay=200,
            active_reset=True,
            man_reset=True,
            storage_reset=[stor_i],
            reset_dump_mode=active_reset_defaults["reset_dump_mode"],
        )
        gain_expts[i] = runner.execute(
            stor_mode_no=stor_i,
            parameter_to_test="gain",
            go_kwargs=dict(analyze=False, progress=False, display=False),
            span=int(station.ds_floquet.get_gain(stor_name)
                     * gain_span / span_divisor),
            expts=5,
            relax_delay=200,
            active_reset=True,
            man_reset=True,
            storage_reset=[stor_i],
            reset_dump_mode=active_reset_defaults["reset_dump_mode"],
        )

    assert freq_expts[0] is not None
    assert gain_expts[0] is not None


def test_bare_scramble_sweep_builds(mock_session, defaults):
    """floquet_calibration.py's bare dark-mode readout check.

    Inline, like the notebook cell it stands in for.
    """
    import experiments as meas
    from experiments import CharacterizationRunner
    from experiments.qsim.notebook_helpers.floquet_bare_readout import (
        sideband_scramble_preproc,
    )
    from experiments.qsim.notebook_helpers.floquet_calibration import (
        floquet_cycle_list_gen,
    )

    active_reset_defaults, floquet_defaults, _ = defaults
    station, client = mock_session.station, mock_session.client

    cfg = AttrDict(dict(
        expts=1, reps=20, rounds=1, qubits=[0], ro_stor=0, init_fock=True,
        normalize=False, post_select_pre_pulse=False, active_reset=False,
        man_reset=False, storage_reset=False, prepulse=True, postpulse=True,
    ))
    cfg.update(active_reset_defaults)
    cfg.update(floquet_defaults)

    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.QsimBaseExperiment,
        ExptProgram=meas.SidebandScrambleDarkProgramNewNew,
        default_expt_cfg=cfg,
        preprocessor=sideband_scramble_preproc,
        postprocessor=None,
        job_client=client,
        show=False,
    )
    swap_stors = [1, 2, 3, 4]
    meas_stors = [0, 1]
    dark_swaps = [4, 5]
    floquet_cycles_list = floquet_cycle_list_gen(0, 4, 4, 2)
    detunings = [0] * len(swap_stors)
    reset_stors = meas_stors[1:]

    expts = []
    for meas_stor in meas_stors:
        sub_expts = []
        for floquet_cycles in floquet_cycles_list:
            sub_expts.append(runner.execute(
                reps=20,
                init_fock=True,
                init_stor=0,
                ro_stor=meas_stor,
                relax_delay=200,
                active_reset=True,
                pre_relax_delay=100,
                man_reset=True,
                storage_reset=reset_stors,
                reset_dump_mode=active_reset_defaults["reset_dump_mode"],
                dump_reset_iter_num=active_reset_defaults["dump_reset_iter_num"],
                swap_stors=swap_stors,
                update_phases=True,
                detunings=detunings,
                floquet_cycles=floquet_cycles,
                swept_params=["floquet_cycle"],
                custom_prepulse=False,
                custom_postpulse=False,
                debug=False,
                swap_man_dark=False,
                dark_swap_order=dark_swaps,
                second_rel_phase=180,
                map_to_qubit_ge=True,
                prepulse=True,
                postpulse=True,
                palindrome_scramble=floquet_defaults["palindrome_scramble"],
                scramble_sync_cycles=floquet_defaults["scramble_sync_cycles"],
            ))
        expts.append(sub_expts)

    assert expts and expts[0]


def test_displacement_kerr_builds_without_the_uncalibrated_mode(
        mock_session, defaults):
    """floquet_displacement_kerr.py, minus storage mode 6.

    The notebook sweeps modes [4, 5, 6, 7], but `M1-S6` has `pi=nan` in
    `CFG-M1-20260904-00014`, which raises "cannot convert float NaN to
    integer" inside acquire(). That is a gap in the pinned config, not in the
    notebook split, so this pins down that the theme builds fine on the three
    calibrated modes. See `test_storage_mode_6_has_no_calibrated_pi_length`.
    """
    from experiments import CharacterizationRunner
    from experiments.qsim import floquet_dark_mode_readout as fdm

    active_reset_defaults, floquet_defaults, _ = defaults
    station, client = mock_session.station, mock_session.client
    modes = [4, 5, 7]

    runner = CharacterizationRunner(
        station=station,
        ExptClass=fdm.FloquetDisplacementKerrExperiment,
        ExptProgram=fdm.FloquetDisplacementKerrProgram,
        default_expt_cfg=AttrDict(dict(
            expts=1, rounds=1, reps=20, qubits=[0], active_reset=True,
            man_reset=True, storage_reset=modes, pre_relax_delay=100,
            relax_delay=200,
            reset_dump_mode=active_reset_defaults["reset_dump_mode"],
            dump_reset_iter_num=active_reset_defaults["dump_reset_iter_num"],
            use_qubit_man_reset=False, normalize=False, swap_stors=modes,
            scramble_sync_cycles=floquet_defaults["scramble_sync_cycles"],
            floquet_hardware_loop=floquet_defaults["floquet_hardware_loop"],
            update_phases=True, zero_floquet_gain=False, man_mode_no=1,
            perform_wigner=False, do_g_and_e=False, ramsey_freq=0.2,
            displace_gains=np.arange(2000, 4001, 1000),
            n_cycle_pairs=np.arange(0, 3, dtype=int),
            swept_params=["displace_gain", "n_cycle_pair"],
        )),
        job_client=client,
        show=False,
    )
    expt = runner.execute(postprocess=False, log=False, show=False)
    assert expt is not None


def test_storage_mode_6_has_no_calibrated_pi_length(mock_session):
    """Pin the config gap that stops the Kerr theme's full mode list.

    If someone recalibrates M1-S6 and this starts failing, the Kerr notebook
    can go back to sweeping [4, 5, 6, 7] and the test above can be widened.
    """
    station = mock_session.station
    pi_length = station.ds_storage.get_pi("M1-S6")
    assert np.isnan(pi_length), (
        "M1-S6 now has a calibrated pi length "
        f"({pi_length}); widen the Kerr theme back to modes [4, 5, 6, 7]"
    )


@pytest.mark.parametrize(
    "waveform,expected",
    [("flat_top", "ok"), ("preload_flattop", "rejected"), ("gauss", "rejected")],
)
def test_floquet_chevron_only_accepts_the_legacy_flat_top(
        mock_session, defaults, waveform, expected):
    """`FloquetChevronProgram` sets `length` unconditionally.

    `experiments/qsim/floquet_chevron.py` line 15 does
    `m1s_kwarg['length'] = ...` for every waveform, but qick only accepts a
    `length` parameter for the const/flat_top pulse style. So the frequency
    chevron cannot build under `preload_flattop` -- which is what
    `FLOQUET_DEFAULTS` selects and what source cell 62 writes into every
    ds_floquet row.

    This is library code the stage-2 split did not touch, and it predates it.
    The test records the actual behaviour so the eventual fix is visible.
    """
    import experiments as meas
    from experiments import CharacterizationRunner
    from experiments.qsim.notebook_helpers.floquet_calibration import (
        floquet_freq_chev_preproc,
    )

    active_reset_defaults, floquet_defaults, _ = defaults
    station, client = mock_session.station, mock_session.client

    for stor in range(1, 8):
        station.ds_floquet.update_waveform(f"M1-S{stor}", waveform)

    cfg = AttrDict(dict(
        expts=1, reps=20, rounds=1, qubits=[0], ro_stor=0, f0g1_cavity=1,
        detunes=[-0.1, 0.0, 0.1], swept_params=["detune", "length"],
        normalize=False, active_reset=False, man_reset=False,
        storage_reset=False, prepulse=True, postpulse=True, init_fock=True,
    ))
    cfg.update(active_reset_defaults)
    cfg.update(dict(floquet_defaults, floquet_waveform=waveform))

    runner = CharacterizationRunner(
        station=station,
        ExptClass=meas.FloquetChevronExperiment,
        ExptProgram=meas.FloquetChevronProgram,
        default_expt_cfg=cfg,
        preprocessor=floquet_freq_chev_preproc,
        postprocessor=None,
        job_client=client,
        show=False,
    )
    run = lambda: runner.execute(
        init_stor=1, reps=20, relax_delay=200, active_reset=True,
        man_reset=True, storage_reset=[1], reset_dump_mode=1,
        postprocess=False, log=False, show=False,
    )

    if expected == "ok":
        assert run() is not None
    else:
        with pytest.raises(RuntimeError, match="unsupported pulse parameter"):
            run()


# --------------------------------------------------------------------------
# BatchRunner is queue-only, so the MBR themes get a different shape.
# --------------------------------------------------------------------------


def test_batch_runner_refuses_the_queue_in_mock_mode(mock_session, defaults):
    """A mock session must not submit real jobs.

    `BatchRunner.execute` overrides `CharacterizationRunner.execute` and has
    no `run_local` path, so before this guard existed a mock MBR run submitted
    to the production queue -- where the worker runs whatever is checked out
    at the main path, against real hardware unless it was started with
    --mock.
    """
    from experiments.qsim.notebook_helpers.mbr_campaign import (
        build_campaign,
        fixed_n_occupations,
    )
    from experiments.qsim.mbr_phase_correction import (
        MBRPhaseCorrectionExperiment,
    )

    active_reset_defaults, floquet_defaults, measurement_defaults = defaults
    station, client = mock_session.station, mock_session.client

    campaign = build_campaign(
        station=station, client=client,
        floquet_settings=floquet_defaults,
        active_reset_settings=active_reset_defaults,
        measurement_settings=measurement_defaults,
        modes=[1, 2, 3, 4], calibration_job_ids={}, reps=20,
    )
    occupations = fixed_n_occupations(1, len(campaign.mode_labels))
    batch = MBRPhaseCorrectionExperiment.calibration_batch(
        campaign.defaults, campaign.modes, occupations,
        np.arange(0, 3, dtype=int),
        sync_cycles=campaign.sync_cycles, repeats=1, reps=20,
    )
    runner = campaign.BatchRunner(
        station=station, ExptClass=campaign.EncSpec,
        ExptProgram=campaign.floquet_dark_mode_readout
        .EntireFloquetCyclePhaseCalibrationProgram,
        default_expt_cfg=batch.default_expt_cfg,
        job_client=client, show=False,
    )
    with pytest.raises(RuntimeError, match="no local mock path"):
        runner.execute(batch.configs[:1], batch_size=1, log=False, show=False)


MBR_BATCHES = [
    "phase_calibration",
    "orthogonality",
    "propagator",
    "spectroscopy",
]


@pytest.mark.parametrize("which", MBR_BATCHES)
def test_mbr_batch_builds_and_compiles(mock_session, defaults, which):
    """Each MBR batch's configs assemble into a compilable qick program.

    This is the layer the stage-2 split actually changed: `build_campaign` and
    the four `*_batch` classmethods it feeds. The program is instantiated
    directly rather than through `BatchRunner.execute`, which would queue.
    """
    from experiments.qsim.notebook_helpers.mbr_campaign import (
        build_campaign,
        fixed_n_occupations,
    )
    from experiments.qsim.notebook_helpers.floquet_calibration import (
        floquet_cycle_list_gen,
    )
    from experiments.qsim.mbr_orthogonality import MBROrthogonalityExperiment
    from experiments.qsim.mbr_phase_correction import (
        MBRPhaseCorrectionExperiment,
    )
    from experiments.qsim.mbr_propagator import MBRPropagatorExperiment
    from experiments.qsim.mbr_spectrum import MBRSpectrumExperiment

    active_reset_defaults, floquet_defaults, measurement_defaults = defaults
    station, client = mock_session.station, mock_session.client

    campaign = build_campaign(
        station=station, client=client,
        floquet_settings=floquet_defaults,
        active_reset_settings=active_reset_defaults,
        measurement_settings=measurement_defaults,
        modes=[1, 2, 3, 4], calibration_job_ids={}, reps=20,
    )
    fdm = campaign.floquet_dark_mode_readout
    # N=1 is the smallest complete sector: five occupations, not thirty-five.
    occupations = fixed_n_occupations(1, len(campaign.mode_labels))
    phases = {tuple(o): 0.0 for o in occupations}

    if which == "phase_calibration":
        batch = MBRPhaseCorrectionExperiment.calibration_batch(
            campaign.defaults, campaign.modes, occupations,
            np.arange(0, 3, dtype=int),
            sync_cycles=campaign.sync_cycles, repeats=1, reps=20)
        ProgramClass = fdm.EntireFloquetCyclePhaseCalibrationProgram
    elif which == "orthogonality":
        batch = MBROrthogonalityExperiment.orthogonality_batch(
            campaign.defaults, campaign.modes, occupations,
            sync_cycles=campaign.sync_cycles, reps=20)
        ProgramClass = fdm.EncodingOrthogonalityProgram
    elif which == "propagator":
        batch = MBRPropagatorExperiment.propagator_batch(
            campaign.defaults, campaign.modes, occupations, [0, 2],
            phase_by_occupation=phases,
            sync_cycles=campaign.sync_cycles, reps=20)
        ProgramClass = fdm.EncodingPropagatorProgram
    else:
        batch = MBRSpectrumExperiment.spectroscopy_batch(
            campaign.defaults, campaign.modes, occupations,
            floquet_cycle_list_gen(0, 4, 4, 2), phases,
            detunings=[0.0] * len(campaign.modes),
            sync_cycles=campaign.sync_cycles, reps=20,
            final_occupations=[list(o) for o in occupations])
        ProgramClass = batch.program

    configs = list(batch.configs)
    assert configs, f"{which} produced no configs"

    runner = campaign.BatchRunner(
        station=station, ExptClass=campaign.EncSpec,
        ExptProgram=ProgramClass,
        default_expt_cfg=batch.default_expt_cfg,
        job_client=client, show=False,
    )
    cfg = AttrDict(dict(
        runner.preprocessor(station, runner.default_expt_cfg, **configs[0])
    ))

    # acquire()'s sweep loop sets each singular swept key from its plural
    # list; building the program directly skips that, so stand in for it.
    for name in list(cfg.get("swept_params") or []):
        if name not in cfg:
            values = cfg.get(f"{name}s")
            assert values is not None and len(values), (
                f"{which}: swept param {name!r} has no {name}s list"
            )
            cfg[name] = list(values)[0]

    full = AttrDict(dict(station.hardware_cfg))
    full.expt = cfg
    program = ProgramClass(soccfg=station.soccfg, cfg=full)
    program.compile()
