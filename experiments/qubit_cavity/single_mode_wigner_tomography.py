from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qick import *
from qick.helpers import gauss
from qutip import fock
from slab import AttrDict, Experiment, dsfit
from tqdm import tqdm_notebook as tqdm

import warnings

import fitting.fitting as fitter
from fitting.fit_display_classes import GeneralFitting
from fitting.state_tomography import as_confusion_matrix, correct_readout_probs
from fitting.wigner import WignerAnalysis
from experiments.MM_base import MMAveragerProgram, MM_base


# ====================================================== #
#            sigma_z probe helpers (shared)              #
# ====================================================== #
# The sigma_z probe is a post-prep transmon readout inserted by lane_layout at
# idx_sigma_z (see MM_base.lane_layout). These helpers turn that lane into the
# reported scalar and, for Tier 2, into a per-shot keep-mask for the parity grid.
# Shot/lane layout matches WignerAnalysis.bin_ss_data: flat ordering is
# (rounds, expts, reps, read_num) with the lane index innermost; keep-masks are
# shaped (rounds*reps, expts) via transpose(0, 2, 1).

def _sigma_z_lane_shots(i0_array, read_num, idx_sigma_z,
                        idx_pre_selection=None, threshold=None):
    '''I values at the sigma_z lane, flattened over all shots/alphas/copies.

    Works for any raw buffer whose innermost axis is the read lane, regardless
    of the alpha/rep/pulse-correction grouping ahead of it.

    If idx_pre_selection/threshold are given, keep only the pre-selection-passed
    shots (qubit |g> at the active-reset herald lane(s), i.e. I < threshold) so the
    sigma_z scalar is computed on the SAME shot population that bin_ss_data uses
    for the parity. Without this, sigma_z averages over shots the experiment
    discards (the herald rejects) -- which inflates the apparent |e> fraction,
    since a poor in-sequence readout makes the conditional-reset pi inject |e>
    into exactly those rejected shots.

    idx_pre_selection may be a single lane index or a list of lane indices; when
    a list is given, a shot is kept only if it passes ALL pre-selection lanes
    (mirrors the AND in bin_ss_data).
    '''
    flat = np.asarray(i0_array).ravel()
    n_groups = flat.size // read_num
    per_shot = flat.reshape(n_groups, read_num)
    lane = per_shot[:, idx_sigma_z]
    if idx_pre_selection is not None and threshold is not None:
        pre_idx = idx_pre_selection if np.ndim(idx_pre_selection) else [idx_pre_selection]
        if len(pre_idx):
            keep = np.all(per_shot[:, pre_idx] < threshold, axis=1)
            lane = lane[keep]
    return lane


def _compute_sigma_z(lane_shots, threshold, cfg):
    '''(sigma_z_corrected, sigma_z_raw) from sigma_z-lane single shots.

    sigma_z = P_g - P_e with |g> = (I < threshold). The corrected value applies
    the readout confusion matrix, picking the with/without-active-reset matrix to
    match whether the shot used active reset; if no matrix is in cfg, the corrected
    value falls back to the raw value (with a one-time warning).
    '''
    shots = np.asarray(lane_shots).ravel()
    if shots.size == 0:
        return None, None
    P_e = float(np.mean(shots > threshold))
    sigma_z_raw = float((1 - P_e) - P_e)

    if cfg.expt.get('active_reset', False):
        conf = cfg.device.readout.get('confusion_matrix_with_active_reset', None)
    else:
        conf = cfg.device.readout.get('confusion_matrix_without_reset', None)

    if conf is not None:
        Pg_c, Pe_c = correct_readout_probs([1 - P_e, P_e], as_confusion_matrix(conf))
        sigma_z = float(Pg_c - Pe_c)
    else:
        warnings.warn(
            "sigma_z: no confusion matrix in cfg.device.readout "
            "(confusion_matrix_with_active_reset / _without_reset); "
            "reporting the uncorrected value.", stacklevel=2)
        sigma_z = sigma_z_raw
    return sigma_z, sigma_z_raw


def _sigma_z_filter_map(i0_array, cfg, layout):
    '''Tier-2 keep-mask (|g>) from the sigma_z lane, shaped (rounds*reps, expts).

    Mirrors the pre-selection mask construction in bin_ss_data so the two compose
    cleanly when both are active.
    '''
    rounds = cfg['expt'].get('rounds', 1)
    reps = cfg['expt']['reps']
    expts = cfg['expt']['expts']
    read_num = layout['read_num']
    threshold = cfg.device.readout.threshold[0]
    I_4d = np.reshape(np.asarray(i0_array).ravel(), (rounds, expts, reps, read_num))
    mask = I_4d[:, :, :, layout['idx_sigma_z']] < threshold   # keep |g>
    return np.reshape(np.transpose(mask, (0, 2, 1)), (rounds * reps, expts))


class WignerTomography1ModeProgram(MMAveragerProgram):
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = self.qubits[0]
        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1

        
        # define the displace sigma for calibration     
        self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[self.man_mode_idx], 
                                      gen_ch=self.man_ch[self.man_mode_idx])
        self.displace_sigma = self.us2cycles(cfg.device.manipulate.displace_sigma[self.man_mode_idx],
                                              gen_ch=self.man_ch[self.man_mode_idx])

        self.add_gauss(ch=self.man_ch[self.man_mode_idx], name="displace",
                        sigma=self.displace_sigma, length=self.displace_sigma*4)
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest],gen_ch = self.man_ch[0]),
                                  gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        self.sync_all(200)


        # Inline IQ_table routing — see class docstring for the contract.
        self.waveforms_opt_ctrl = None
        self.waveforms_opt_ctrl_prepulse = None
        if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
            waveform_names = self.load_opt_ctrl_pulse(pulse_conf=cfg.expt.opt_pulse,
                                IQ_table=cfg.expt.IQ_table,
                                )
            self.waveforms_opt_ctrl = waveform_names
        elif cfg.expt.get("IQ_table") is not None:
            # IQ_table set without opt_pulse: allow routing via prepulse if it has
            # an opt_cont row; otherwise reject (the canonical NPZ would silently play).
            opt_cont_shapes = []
            if cfg.expt.get("prepulse") and cfg.expt.get("pre_sweep_pulse"):
                shapes = cfg.expt.pre_sweep_pulse[5]
                opt_cont_shapes = [s for s in shapes
                                   if isinstance(s, list) and s and s[0] == 'opt_cont']
            if not opt_cont_shapes:
                raise ValueError(
                    "cfg.expt.IQ_table is set but cfg.expt.opt_pulse is not, and no "
                    "['opt_cont', enc, state] row was found in cfg.expt.pre_sweep_pulse. "
                    "Either set opt_pulse=[['opt_cont', enc, state]] (preferred) or put "
                    "an opt_cont row in pre_sweep_pulse — otherwise the inline IQ_table "
                    "cannot be applied (the canonical NPZ at "
                    "cfg.device.optimal_control[...].filename would silently play instead)."
                )
            pulse_conf = [opt_cont_shapes[0]]
            self.waveforms_opt_ctrl_prepulse = self.load_opt_ctrl_pulse(
                pulse_conf=pulse_conf, IQ_table=cfg.expt.IQ_table)


    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        # phase reset
        self.reset_and_sync()

        if 'active_reset' in cfg.expt and cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        #  prepulse
        if cfg.expt.prepulse:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else:
                # B3: if an inline IQ_table was loaded for the prepulse opt_cont row,
                # pass its waveform names through so custom_pulse uses the inline data
                # instead of loading the canonical NPZ.
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre_',
                                  sync_zero_const=True,
                                  waveform_preload=self.waveforms_opt_ctrl_prepulse)


        if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
            creator = self.get_prepulse_creator(cfg.expt.opt_pulse)
            self.custom_pulse(cfg, creator.pulse.tolist(),
                              waveform_preload=self.waveforms_opt_ctrl)

        # sigma_z probe: post-prep transmon readout (before displacement+parity).
        # 'reset' (Tier 1) measures then conditionally pi-pulses the ancilla back
        # to |g>; 'postselect' (Tier 2) is a bare herald measurement (no feedback);
        # 'measure' (Tier 3) skips displace+parity entirely so the final readout is
        # the sigma_z readout -- non-invasive, parity comes from a separate run.
        sigma_z_mode = MM_base.lane_layout(cfg)['sigma_z_mode']
        if sigma_z_mode == 'reset':
            self.parity_active_reset(register_label='sigmaz_reset', play_parity=False,
                                     man_idx=self.man_mode_idx + 1, final_sync=False)
        elif sigma_z_mode == 'postselect':
            # measure_parity=True -> herald is a QND parity (syndrome) measurement
            # (play the manipulate parity pulse, then read); otherwise a bare
            # transmon herald. Either way it's one readout lane (idx_sigma_z) and
            # the Wigner grid is post-selected on |g> via _sigma_z_filter_map.
            self.post_selection_measure(
                parity=cfg.expt.get('measure_parity', False),
                man_idx=self.man_mode_idx + 1,
                parity_fast=cfg.expt.get('parity_fast', False),
                prefix='sigmaz_')

        if sigma_z_mode != 'measure':
            self.setup_and_pulse(ch=self.man_ch[self.man_mode_idx], style="arb", freq=self.f_cavity,
                                phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch = self.man_ch[self.man_mode_idx]),
                                gain=self.cfg.expt.amp_placeholder, waveform="displace")

            # self.sync_all(self.us2cycles(0.05))
            self.sync_all()

            # Parity pulse
            self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.phase_second_pulse,
                                    fast=self.cfg.expt.parity_fast)
        self.measure_wrapper()
    
    def collect_shots(self):
        # collect shots for 1 adc and I and Q channels
        read_num = MM_base.lane_layout(self.cfg)['read_num']
        shots_i0 = self.di_buf[0].reshape((1, read_num*self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape((1, read_num*self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        return shots_i0, shots_q0

# ====================================================== #
                      
class WignerTomography1ModeExperiment(Experiment):
    """
    1-mode Wigner tomography.

    Pulse-source routing (CRITICAL — see also `experiments/MM_base.py: load_opt_ctrl_pulse`
    and `experiments/MM_base.py: custom_pulse` opt_cont branch):

    There are two ways to play an optimal-control pulse before the displacement+parity:

    (1) Canonical NPZ path. Set `cfg.expt.prepulse = True` and put an opt_cont row in
        `cfg.expt.pre_sweep_pulse`. `custom_pulse` will load the IQ samples from
        `cfg.device.optimal_control[encoding][state].filename` (an .npz on disk).

    (2) Inline IQ_table path. Set `cfg.expt.opt_pulse = [['opt_cont', enc, state]]`
        AND `cfg.expt.IQ_table = {'times', 'I_c', 'Q_c', 'I_q', 'Q_q'}`. The IQ table
        is loaded once in `initialize()` via `load_opt_ctrl_pulse`, then fired via
        `custom_pulse(..., waveform_preload=self.waveforms_opt_ctrl)` in `body()`.

    Setting `IQ_table` WITHOUT `opt_pulse` is rejected with a ValueError — it used to
    silently fall back to (1), masking the inline samples and leading to the wrong pulse.

    Q-sign convention: both paths apply DAC `qdata = -Q_input`. The negation lives in
    `custom_pulse` (L530) for path 1 and in `load_opt_ctrl_pulse` (L887-890) for path 2.
    Upstream IQ tables should follow the closed-system convention `α̇(t) = -i·ε(t)`
    (Eickbusch 2022).

    Experimental Config:
    expt = dict(
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        displace_length: scaling for displacement gain
        displacement_path: .npy path of complex alphas
        opt_pulse: optional inline pulse spec, e.g. [['opt_cont', 'fock', '1']]
        IQ_table: optional dict with keys 'times', 'I_c', 'Q_c', 'I_q', 'Q_q'
                  (required iff opt_pulse is set)
        prepulse, pre_sweep_pulse, gate_based: canonical NPZ-prepulse path
    )
    """

    def __init__(self, soccfg=None, path='', prefix='WignweTomography1Mode', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def _measure_sigma_z_once(self):
        '''Tier 3: a single transmon sigma_z measurement (prep -> readout, no
        displacement/parity). sigma_z is displacement-independent, so one run
        suffices. Returns (sigma_z_corrected, sigma_z_raw), herald-masked and
        confusion-corrected the same way as the in-line probe.'''
        cfg_sz = AttrDict(deepcopy(self.cfg))
        cfg_sz.expt.sigma_z_mode = 'measure'
        # placeholders the body references only inside the (skipped) displace block
        cfg_sz.expt.phase_second_pulse = 180
        cfg_sz.expt.amp_placeholder = 0
        cfg_sz.expt.phase_placeholder = 0.0
        layout = MM_base.lane_layout(cfg_sz)
        prog = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=cfg_sz)
        prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True,
                     progress=False, readouts_per_experiment=layout['read_num'])
        i0, _ = prog.collect_shots()
        threshold = self.cfg.device.readout.threshold[0]
        lane = _sigma_z_lane_shots(i0, layout['read_num'], layout['idx_sigma_z'],
                                   idx_pre_selection=layout['idx_pre_selection_list'],
                                   threshold=threshold)
        return _compute_sigma_z(lane, threshold, self.cfg)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        qTest = self.cfg.expt.qubits[0]

        self.pulse_correction = self.cfg.expt.get('pulse_correction', False)
        self.cfg.expt.parity_fast = self.cfg.expt.get('parity_fast', False)

        # measure_parity: insert a QND parity (syndrome) measurement on the
        # manipulate mode BEFORE the Wigner displace+parity, and post-select the
        # reconstruction on even (|g>). Realized through the Tier-2 'postselect'
        # herald with a parity pulse (see body + _sigma_z_filter_map). Auto-resolve
        # sigma_z_mode so the user only needs measure_parity=True.
        if self.cfg.expt.get('measure_parity', False) and \
                str(self.cfg.expt.get('sigma_z_mode', 'off')).lower() in ('off', 'none'):
            self.cfg.expt.sigma_z_mode = 'postselect'

        read_num = MM_base.lane_layout(self.cfg)['read_num']

        # extract displacement list from file path
        if 'alpha_list' in self.cfg.expt:
            alpha_2d = np.array(self.cfg.expt.alpha_list)
            alpha_list = alpha_2d[:, 0] + 1j * alpha_2d[:, 1]
        else:
            alpha_list = np.load(self.cfg.expt["displacement_path"])  # complex ndarray

        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1
        gain2alpha = self.cfg.device.manipulate.gain_to_alpha[self.man_mode_idx] 

        data={"alpha":[],"avgi":[], "avgq":[], "amps":[], "phases":[], "i0":[], "q0":[]}

        # Tier 3 ('measure'): sigma_z is displacement-independent, so measure it
        # ONCE here (prep -> transmon readout, displace+parity skipped), then run
        # the standard Wigner sweep below with the probe OFF so the parity grid is
        # native -- never perturbed by the transmon readout. One job, both outputs.
        self.sigma_z_mode = MM_base.lane_layout(self.cfg)['sigma_z_mode']
        if self.sigma_z_mode == 'measure':
            sz, sz_raw = self._measure_sigma_z_once()
            data["sigma_z"] = np.nan if sz is None else sz
            data["sigma_z_raw"] = np.nan if sz_raw is None else sz_raw
            # Run the sweep + analyze as 'off' so parity is native and analyze
            # does not try to re-extract sigma_z from the (probe-less) sweep shots.
            self.cfg.expt.sigma_z_mode = 'off'
            read_num = MM_base.lane_layout(self.cfg)['read_num']

        pre_selection = ('active_reset' in self.cfg.expt and self.cfg.expt.active_reset
                         and self.cfg.expt.get('pre_selection_reset', False))
        if pre_selection:
            threshold = self.cfg.device.readout.threshold[self.cfg.expt.qubits[0]]

        for alpha in tqdm(alpha_list, disable=not progress):
            self.cfg.expt.phase_second_pulse = 180 # reset the phase of the second pulse
            _alpha = np.conj(alpha) # convert to conjugate to respect qick convention
            self.cfg.expt.amp_placeholder =  int(np.abs(_alpha)/gain2alpha) # assumes you calibrated the gain2alpha with the same displace_sigma as will be played here... it really should be!
            self.cfg.expt.phase_placeholder = np.angle(_alpha)/np.pi*180 - 90 # 90 is needed since da/dt = -i*drive
            wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = wigner
            avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                        readouts_per_experiment=read_num,
                                            #  debug=debug
                                             )
            # collect single shots
            i0, q0 = wigner.collect_shots()

            if pre_selection:
                avgi_val, avgq_val = GeneralFitting.filter_shots_per_point(
                    i0.flatten(), q0.flatten(), read_num,
                    threshold=threshold, pre_selection=True, cfg=self.cfg)
            else:
                avgi_val = avgi[0][-1]
                avgq_val = avgq[0][-1]

            amp = np.abs(alpha) # Calculating the magnitude
            phase = np.angle(alpha) # Calculating the phase
            data["alpha"].append(alpha)
            data["avgi"].append(avgi_val)
            data["avgq"].append(avgq_val)
            data["amps"].append(amp)
            data["phases"].append(phase)
            data["i0"].append(i0)
            data["q0"].append(q0)

            if self.pulse_correction:
                self.cfg.expt.phase_second_pulse = 0
                wigner = WignerTomography1ModeProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = wigner.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                            readouts_per_experiment=read_num,
                                                #  debug=debug
                                                )
                i0, q0 = wigner.collect_shots()

                if pre_selection:
                    avgi_val, avgq_val = GeneralFitting.filter_shots_per_point(
                        i0.flatten(), q0.flatten(), read_num,
                        threshold=threshold, pre_selection=True, cfg=self.cfg)
                else:
                    avgi_val = avgi[0][-1]
                    avgq_val = avgq[0][-1]

                data["avgi"].append(avgi_val)
                data["avgq"].append(avgq_val)
                data["i0"].append(i0)
                data["q0"].append(q0)

        self.cfg.expt['expts'] = len(data["alpha"])

        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        self.pulse_correction = self.cfg.expt.get('pulse_correction', False)
        mode_state_num = kwargs.get('mode_state_num', 10)

        # sigma_z probe: report the post-prep transmon scalar, and (Tier 2) build
        # a per-shot keep-mask so the parity grid uses only |g>-heralded shots.
        layout = MM_base.lane_layout(self.cfg)
        sigma_z_mode = layout['sigma_z_mode']
        threshold = self.cfg.device.readout.threshold[0]
        if sigma_z_mode != 'off':
            # Compute sigma_z on the SAME herald-passed shots the parity uses
            # (bin_ss_data masks the parity on idx_pre_selection; mirror it here).
            lane = _sigma_z_lane_shots(data["i0"], layout['read_num'], layout['idx_sigma_z'],
                                       idx_pre_selection=layout['idx_pre_selection_list'],
                                       threshold=threshold)
            sz, sz_raw = _compute_sigma_z(lane, threshold, self.cfg)
            # Store nan (not None) so the HDF5 save in Experiment.save_data succeeds
            # (a Python None has object dtype and has no native HDF5 equivalent).
            data["sigma_z"] = np.nan if sz is None else sz
            data["sigma_z_raw"] = np.nan if sz_raw is None else sz_raw

        if sigma_z_mode == 'measure':
            # Tier 3 (non-invasive): displace+parity were skipped, so there is no
            # parity in this run -- it comes from a separate sigma_z='off' run.
            # Fill nan (saveable) and return; only sigma_z is meaningful here.
            n_alpha = len(data["alpha"])
            data["pe"] = np.full(n_alpha, np.nan)
            data["parity"] = np.full(n_alpha, np.nan)
            return data

        if self.pulse_correction:
            # if pulse correction i0 = [i_minus0, i_plus0, i_minus1, i_plus1, ...]
            #
            # Hand bin_ss_data the raw lane-interleaved shots; it does the
            # final-lane extraction AND pre_selection_reset masking via lane_layout.

            data_minus = {"i0": data["i0"][0::2], "q0": data["q0"][0::2]}
            data_plus  = {"i0": data["i0"][1::2], "q0": data["q0"][1::2]}

            wigner_analysis_minus = WignerAnalysis(data=data_minus,
                                                   config=self.cfg,
                                                    mode_state_num=mode_state_num,
                                                    alphas=data["alpha"])

            wigner_analysis_plus = WignerAnalysis(data=data_plus,
                                                  config=self.cfg,
                                                  mode_state_num=mode_state_num,
                                                  alphas=data["alpha"])

            fmap_minus = fmap_plus = None
            if sigma_z_mode == 'postselect':
                fmap_minus = _sigma_z_filter_map(data["i0"][0::2], self.cfg, layout)
                fmap_plus  = _sigma_z_filter_map(data["i0"][1::2], self.cfg, layout)
                data["sigma_z_discard_frac"] = float(
                    1 - 0.5 * (fmap_minus.mean() + fmap_plus.mean()))

            pe_plus = wigner_analysis_plus.bin_ss_data(filter_map=fmap_plus)
            pe_minus = wigner_analysis_minus.bin_ss_data(filter_map=fmap_minus)
            parity_plus = (1 - pe_plus) - pe_plus
            parity_minus = (1 - pe_minus) - pe_minus
            parity = (parity_minus - parity_plus) / 2

            # apply scale
            scale_parity = self.cfg.device.manipulate.alpha_scale[self.man_mode_idx]

            data["pe_plus"] = pe_plus
            data["pe_minus"] = pe_minus
            data["parity_plus"] = parity_plus
            data["parity_minus"] = parity_minus
            data["parity"] = parity / scale_parity

        else:
            # Hand bin_ss_data the raw lane-interleaved shots; it handles
            # final-lane extraction AND pre_selection_reset masking via lane_layout.
            data_wigner = {
                "i0": data["i0"],
                "q0": data["q0"],
            }

            wigner_analysis = WignerAnalysis(data=data_wigner,
                                              config=self.cfg,
                                              mode_state_num=mode_state_num,
                                              alphas=data["alpha"])

            filter_map = None
            if sigma_z_mode == 'postselect':
                filter_map = _sigma_z_filter_map(data["i0"], self.cfg, layout)
                data["sigma_z_discard_frac"] = float(1 - filter_map.mean())

            pe = wigner_analysis.bin_ss_data(filter_map=filter_map)
            data["pe"] = pe
            data["parity"] = (1 - pe) - pe

        return data

    def display(self, data=None, mode_state_num=None, initial_state=None, rotate=None, state_label='', debug_components=False, station=None, save_fig=False, **kwargs):
        """
        Display using WignerAnalysis reconstruction pipeline.

        Parameters:
        - mode_state_num: Hilbert space cutoff (default from cfg.expt.display_mode_state_num or 5)
        - initial_state: qutip.Qobj (ket) used for reconstruction; if None, defaults to |0> in the chosen dimension
        - rotate: whether to rotate the Wigner frame (default from cfg.expt.display_rotate or False)
        - state_label: optional label for the plotted state
        - debug_components: when True and pulse correction data are available, plot pe_plus/minus and parity_plus/minus vs |alpha|
        - station: MultimodeStation instance for saving plots (optional)
        - save_fig: if True and station is provided, save the figure using station.save_plot()
        """
        if data is None:
            data = self.data

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', False))

        # Basic validation
        if 'parity' not in data or 'alpha' not in data:
            raise ValueError('Expected keys "parity" and "alpha" in data for Wigner display.')
        parity = data['parity']
        alphas = data['alpha']
        if len(parity) != len(alphas):
            raise ValueError(f'Length mismatch: parity ({len(parity)}) vs alpha ({len(alphas)}).')
        # Tier 3 ('measure') runs carry no parity (displace+parity were skipped);
        # the Wigner must come from a separate sigma_z_mode='off' run.
        if not np.all(np.isfinite(np.asarray(parity, dtype=float))):
            raise ValueError(
                "parity contains NaN/inf -- this looks like a sigma_z_mode='measure' "
                "(Tier 3) run, which has no parity grid. Display the Wigner from a "
                "separate sigma_z_mode='off' run; use this run only for "
                "data['sigma_z'].")

        # Default initial state if not provided
        if initial_state is None:
            initial_state = qt.fock(mode_state_num, 0).unit()

        # Build analysis and plot
        wigner_analysis = WignerAnalysis(data=data, config=self.cfg, mode_state_num=mode_state_num, alphas=alphas, station=station)
        results = wigner_analysis.wigner_analysis_results(parity, initial_state=initial_state, rotate=rotate)
        fig = wigner_analysis.plot_wigner_reconstruction_results(results, initial_state=initial_state, state_label=state_label)

        data['rho'] = results['rho']
        data['rho_rotated'] = results['rho_rotated']
        data['fidelity'] = results['fidelity']
        data['W_fit'] = results['W_fit']
        data['W_ideal'] = results['W_ideal']
        data['alpha_wigner'] = results['x_vec']
        data['theta_opt'] = results['theta_max']
        data['target_state'] = initial_state

        # Optional debug components plot (requires pulse-correction products)
        if debug_components:
            has_pe = ('pe_plus' in data) and ('pe_minus' in data) and (data.get('pe_plus') is not None) and (data.get('pe_minus') is not None)
            has_par = ('parity_plus' in data) and ('parity_minus' in data) and (data.get('parity_plus') is not None) and (data.get('parity_minus') is not None)
            if has_pe and has_par:
                try:
                    pe_plus = np.asarray(data['pe_plus'])
                    pe_minus = np.asarray(data['pe_minus'])
                    parity_plus = np.asarray(data['parity_plus'])
                    parity_minus = np.asarray(data['parity_minus'])
                    parity = np.asarray(data['parity'])
                    alpha_abs = np.abs(alphas)

                    fig_dbg, ax = plt.subplots(1, 2, figsize=(12, 6))
                    pe = pe_plus + pe_minus
                    ax[0].plot(alpha_abs, pe, 'o', label='pe')
                    ax[0].plot(alpha_abs, pe_plus, 'o', label='pe_plus')
                    ax[0].plot(alpha_abs, pe_minus, 'o', label='pe_minus')

                    ax[1].plot(alpha_abs, 2/np.pi * parity, 'o', label='parity')
                    ax[1].plot(alpha_abs, 2/np.pi * parity_plus, 'o', label='parity_plus')
                    ax[1].plot(alpha_abs, 2/np.pi * parity_minus, 'o', label='parity_minus')

                    ax[0].set_xlabel('Alpha')
                    ax[0].set_ylabel('Probability')
                    ax[1].set_xlabel('Alpha')
                    ax[1].set_ylabel('Parity')
                    ax[0].legend()
                    ax[1].legend()
                    plt.tight_layout()
                except Exception as e:
                    print(f"Debug components plot skipped due to: {e}")
            else:
                print("Debug components requested, but pe+/pe- or parity+/parity- not found in data.")

        # Save figure if requested
        if save_fig and station is not None:
            filename = f"wigner_{state_label}.png" if state_label else "wigner_reconstruction.png"
            station.save_plot(fig, filename=filename)
        elif save_fig and station is None:
            print("Warning: save_fig=True but no station provided. Plot not saved.")

        return fig

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


# ====================================================== #
#                Process Tomography (1 mode)             #
# ====================================================== #

class ProcessTomographyProgram(MMAveragerProgram):
    """
    Program sequence per shot:
    - Optional active reset
    - State preparation via custom_pulse (cfg.expt.state_prep) respecting cfg.expt.gate_based
    - Repeat block executed cfg.expt.repeat_count times (default: 1):
        - Optional prepulse (cfg.expt.prepulse)
        - Wait for cfg.expt.wait_time [us]
        - Optional postpulse (cfg.expt.postpulse)
    - Displacement (amp_placeholder/phase_placeholder)
    - Parity pulse (phase set via cfg.expt.phase_second_pulse)
    - Measure
    """

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = self.qubits[0]

        # Manipulation channel and displacement GAUSS
        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1
        self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[self.man_mode_idx],
                                       gen_ch=self.man_ch[self.man_mode_idx])
        self.displace_sigma = self.us2cycles(cfg.device.manipulate.displace_sigma[self.man_mode_idx],
                                              gen_ch=self.man_ch[self.man_mode_idx])
        self.add_gauss(ch=self.man_ch[self.man_mode_idx],
                        name="displace", sigma=self.displace_sigma,
                          length=self.displace_sigma*4)

        # Readout setup
        self.set_pulse_registers(
            ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest],
            phase=self.deg2reg(cfg.device.readout.phase[qTest], gen_ch=self.man_ch[self.man_mode_idx]),
            gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest]
        )

        self.sync_all(200)

        if "opt_pulse" in cfg.expt and cfg.expt.opt_pulse:
            waveform_names = self.load_opt_ctrl_pulse(
                pulse_conf=cfg.expt.opt_pulse, IQ_table=cfg.expt.IQ_table,
            )
            self.waveforms_opt_ctrl = waveform_names
        elif cfg.expt.get("IQ_table") is not None:
            raise ValueError(
                "cfg.expt.IQ_table is set but cfg.expt.opt_pulse is not. "
                "The inline IQ_table is only consumed when opt_pulse is set "
                "(see WignerTomography1ModeExperiment docstring for the contract). "
                "Set opt_pulse=[['opt_cont', enc, state]] alongside the IQ_table — "
                "otherwise the canonical NPZ at cfg.device.optimal_control[...].filename "
                "would silently play instead."
            )

    def body(self):
        cfg = AttrDict(self.cfg)
        # reset phases
        self.reset_and_sync()
        repeat_count = int(cfg.expt.get('repeat_count', 1))

        # Optional active reset
        if 'active_reset' in cfg.expt and cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # State preparation
        if 'state_prep' in cfg.expt and cfg.expt.state_prep:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.state_prep)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix='state_')
            else:
                self.custom_pulse(cfg, cfg.expt.state_prep, prefix='state_')

        # sigma_z probe: post-prep transmon readout, once per shot, before the
        # repeat/displacement/parity block. 'reset' (Tier 1) measures then resets
        # the ancilla to |g>; 'postselect' (Tier 2) is a bare herald measurement.
        sigma_z_mode = MM_base.lane_layout(cfg)['sigma_z_mode']
        if sigma_z_mode == 'reset':
            self.parity_active_reset(register_label='sigmaz_reset_pt', play_parity=False,
                                     man_idx=self.man_mode_idx + 1, final_sync=False)
        elif sigma_z_mode == 'postselect':
            self.post_selection_measure(parity=False, prefix='sigmaz_')

        for _ in range(repeat_count):
            # Prepulse
            if 'prepulse' in cfg.expt and cfg.expt.prepulse:
                if cfg.expt.gate_based:
                    creator = self.get_prepulse_creator(cfg.expt.prepulse)
                    self.custom_pulse(cfg, creator.pulse.tolist(), prefix='pre_')
                else:
                    self.custom_pulse(cfg, cfg.expt.prepulse, prefix='pre_')

            # Wait
            wait_us = cfg.expt.get('wait_time', 0.0) or 0.0
            if wait_us > 0:
                self.sync_all(self.us2cycles(wait_us))

            # Postpulse
            if 'postpulse' in cfg.expt and cfg.expt.postpulse:
                if cfg.expt.gate_based:
                    creator = self.get_prepulse_creator(cfg.expt.postpulse)
                    self.custom_pulse(cfg, creator.pulse.tolist(), prefix='post_')
                else:
                    self.custom_pulse(cfg, cfg.expt.postpulse, prefix='post_')

            if 'parity_shot' in cfg.expt and cfg.expt.parity_shot:
                self.parity_active_reset(register_label='label_%i'%_, play_parity=True)


        # Displacement for Wigner tomography
        self.setup_and_pulse(
            ch=self.man_ch[self.man_mode_idx], style="arb", freq=self.f_cavity,
            phase=self.deg2reg(self.cfg.expt.phase_placeholder, gen_ch=self.man_ch[self.man_mode_idx]),
            gain=self.cfg.expt.amp_placeholder, waveform="displace"
        )
        self.sync_all()

        # Parity pulse and measure
        self.play_parity_pulse(self.man_mode_idx, second_phase=self.cfg.expt.phase_second_pulse,
                                fast=self.cfg.expt.parity_fast)
        self.measure_wrapper()

    def collect_shots(self):
        # collect shots for 1 adc and I and Q channels
        read_num = MM_base.lane_layout(self.cfg)['read_num']
        shots_i0 = self.di_buf[0].reshape((1, read_num*self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        shots_q0 = self.dq_buf[0].reshape((1, read_num*self.cfg["reps"]), order='F') / self.readout_lengths_adc[0]
        return shots_i0, shots_q0


class ProcessTomographyExperiment(Experiment):
    """
    Process tomography over 4 cardinal states and a wait-time sweep, performing 1-mode Wigner tomography for each condition.

    expt keys:
    - qubits: [q]
    - cardinal_states: list of N pulses (list-of-lists) accepted by custom_pulse
    - gate_based: bool — applies to state_prep, prepulse, postpulse
    - prepulse: optional pulse (list-of-lists)
    - postpulse: optional pulse (list-of-lists)
    - wait_start, wait_step, wait_expts (us): defines wait_list
    - repeat_count: integer number of times to repeat (prepulse -> wait -> postpulse) per shot (default 1)
    - repeat_list OR repeat_start, repeat_step, repeat_expts: defines sweep over repeat_count values
    - displacement_path: .npy path of complex alphas
    - displace_length: scaling ref, re-used from Wigner experiment
    - reps, rounds, active_reset, opt_pulse, IQ_table: standard
    - pulse_correction: bool — if True, perform two acquisitions per alpha with phase_second_pulse=180 then 0 (non-inline)
    """

    def __init__(self, soccfg=None, path='', prefix='ProcessTomography', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def _build_wait_list(self):
        ex = self.cfg.expt
        if 'wait_list' in ex and ex.wait_list is not None:
            return np.array(ex.wait_list, dtype=float)
        start = float(ex.wait_start)
        step = float(ex.wait_step)
        expts = int(ex.wait_expts)
        return start + step * np.arange(expts, dtype=float)

    def _build_repeat_list(self):
        """Construct list of repeat counts to sweep. Defaults to [repeat_count or 1]."""
        ex = self.cfg.expt
        # Accept multiple key names for flexibility
        if 'repeat_list' in ex and ex.repeat_list is not None:
            return np.array(ex.repeat_list, dtype=int)
        if 'n_repeat_list' in ex and ex.n_repeat_list is not None:
            return np.array(ex.n_repeat_list, dtype=int)
        # Parametric definition
        keys = ('repeat_start', 'repeat_step', 'repeat_expts')
        if all(k in ex for k in keys):
            start = int(ex.repeat_start)
            step = int(ex.repeat_step)
            expts = int(ex.repeat_expts)
            return start + step * np.arange(expts, dtype=int)
        # Fallback: single value from repeat_count or 1
        if 'repeat_count' in ex and ex.repeat_count:
            rc = int(ex.repeat_count)
        else:
            rc = 1

        return np.array([rc], dtype=int)

    def acquire(self, progress=False, debug=False):
        # Expand singleton config entries
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        # Tier 3 ('measure') skips displace+parity and is only meaningful for the
        # single-shot Wigner program; the process-tomography repeat/parity_shot
        # structure has no well-defined 'measure' semantics. Fail fast rather than
        # silently read the parity lane as sigma_z.
        if MM_base.lane_layout(self.cfg)['sigma_z_mode'] == 'measure':
            raise NotImplementedError(
                "sigma_z_mode='measure' (Tier 3) is only supported by "
                "WignerTomography1ModeExperiment. Use 'off' / 'reset' / "
                "'postselect' for ProcessTomographyExperiment.")

        # Read control flags
        self.pulse_correction = bool(self.cfg.expt.get('pulse_correction', False))
        self.parity_fast = bool(self.cfg.expt.get('parity_fast', False))
        # Displacements

        # extract displacement list from file path
        if 'alpha_list' in self.cfg.expt:
            alpha_2d = self.cfg.expt.alpha_list  # 2d list 
            # convert list to array 
            alpha_2d = np.array(alpha_2d)
            alpha_list = alpha_2d[:, 0] + 1j * alpha_2d[:, 1]
        else:
            alpha_list = np.load(self.cfg.expt["displacement_path"])  # complex ndarray


        man_mode_no = self.cfg.expt.get('man_mode_no', 1)
        self.man_mode_idx = man_mode_no - 1
        gain2alpha = self.cfg.device.manipulate.gain_to_alpha[self.man_mode_idx]
        displace_sigma = self.cfg.device.manipulate.displace_sigma[self.man_mode_idx]

        # Wait sweep and repeat sweep and states
        wait_list = self._build_wait_list()
        repeat_list = self._build_repeat_list()
        states = self.cfg.expt.cardinal_states
        # Prepare data containers
        nS, nW, nR, nA = len(states), len(wait_list), len(repeat_list), len(alpha_list)
        pc_factor = 2 if self.pulse_correction else 1

        data = {
            "alpha": np.array(alpha_list),
            "wait_list": np.array(wait_list, dtype=float),
            "repeat_list": np.array(repeat_list, dtype=int),
            "avgi": np.empty((nS, nW, nR, nA*pc_factor), dtype=float),
            "avgq": np.empty((nS, nW, nR, nA*pc_factor), dtype=float),
            "amps": np.empty((nS, nW, nR, nA*pc_factor), dtype=float),
            "phases": np.empty((nS, nW, nR, nA*pc_factor), dtype=float),
            "i0": np.empty((nS, nW, nR, nA*pc_factor), dtype=object),
            "q0": np.empty((nS, nW, nR, nA*pc_factor), dtype=object),
        }

        # Iterate: states -> waits -> repeats -> alphas
        for si, state_prep in enumerate(states):
            for wi, wait_us in enumerate(wait_list):
                # reset phase for default parity (minus)
                self.cfg.expt.phase_second_pulse = 180
                self.cfg.expt.state_prep = state_prep
                self.cfg.expt.wait_time = float(wait_us)

                for ri, repeat_count in enumerate(repeat_list):
                    # set repeat count for this condition
                    self.cfg.expt.repeat_count = int(repeat_count)

                    # Readouts per shot — single source of truth in MM_base.lane_layout
                    read_num = MM_base.lane_layout(self.cfg)['read_num']

                    for ai, alpha in enumerate(tqdm(alpha_list, disable=not progress)):
                        # Map alpha to amp/phase placeholders
                        self.cfg.expt.phase_second_pulse = 180
                        scale = displace_sigma
                        _alpha = np.conj(alpha)
                        self.cfg.expt.amp_placeholder = int(np.abs(_alpha)/gain2alpha * scale / self.cfg.expt.displace_length)
                        self.cfg.expt.phase_placeholder = np.angle(_alpha)/np.pi*180 - 90

                        # First acquisition: parity minus
                        prog = ProcessTomographyProgram(soccfg=self.soccfg, cfg=self.cfg)
                        self.prog = prog
                        avgi, avgq = prog.acquire(
                            self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                            readouts_per_experiment=read_num,
                        )
                        avgi = avgi[0][0]
                        avgq = avgq[0][0]
                        amp = np.abs(alpha)
                        phase = np.angle(alpha)
                        i0, q0 = prog.collect_shots()
 
                        # Index position for minus
                        idx = ai*pc_factor
                        data["avgi"][si, wi, ri, idx] = avgi
                        data["avgq"][si, wi, ri, idx] = avgq
                        data["amps"][si, wi, ri, idx] = amp
                        data["phases"][si, wi, ri, idx] = phase
                        data["i0"][si, wi, ri, idx] = i0
                        data["q0"][si, wi, ri, idx] = q0


                        # Second acquisition: parity plus (if enabled)
                        if self.pulse_correction:
                            self.cfg.expt.phase_second_pulse = 0
                            prog = ProcessTomographyProgram(soccfg=self.soccfg, cfg=self.cfg)
                            avgi2, avgq2 = prog.acquire(
                                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False,
                                readouts_per_experiment=read_num,
                            )
                            avgi2 = avgi2[0][0]
                            avgq2 = avgq2[0][0]
                            idx2 = idx + 1
                            data["avgi"][si, wi, ri, idx2] = avgi2
                            data["avgq"][si, wi, ri, idx2] = avgq2
                            data["amps"][si, wi, ri, idx2] = amp
                            data["phases"][si, wi, ri, idx2] = phase
                            i02, q02 = prog.collect_shots()
                            data["i0"][si, wi, ri, idx2] = i02
                            data["q0"][si, wi, ri, idx2] = q02

        # Store meta
        self.data = data
        return data

    def analyze(self, data=None, mode_state_num=None, target_states=None, rotate=None, debug=False, data_stacked=False, **kwargs):
        if data is None:
            data = self.data
        
        if data_stacked:
            """Reconstruct object-array i0/q0 from padded stacks and shape metadata.

            Expects keys written by save_data with stack_2d_variable:
            - data['i0']:  float array (nS, nW, nR, nApc, Rmax, Smax)
            - data['i0_rounds'], data['i0_shots']: int arrays (nS, nW, nR, nApc)
            Same structure for q0.
            """
            nS = len(self.cfg.expt.cardinal_states)
            nW = len(data['wait_list'])
            nR = len(data['repeat_list'])
            nApc = data['avgi'].shape[3]

            i0_stacked = data['i0']
            q0_stacked = data['q0']
            i0_rounds = data.get('i0_rounds')
            i0_shots = data.get('i0_shots')
            q0_rounds = data.get('q0_rounds')
            q0_shots = data.get('q0_shots')

            if i0_rounds is None or i0_shots is None or q0_rounds is None or q0_shots is None:
                raise ValueError("data_stacked=True but shape metadata (i0_rounds/i0_shots/q0_rounds/q0_shots) is missing.")

            data_reshaped = {
                'alpha': data['alpha'],
                'wait_list': data['wait_list'],
                'repeat_list': data['repeat_list'],
                'i0': np.empty((nS, nW, nR, nApc), dtype=object),
                'q0': np.empty((nS, nW, nR, nApc), dtype=object),
                'avgi': data['avgi'],
                'avgq': data['avgq'],
            }

            for si in range(nS):
                for wi in range(nW):
                    for ri in range(nR):
                        for ai in range(nApc):
                            R = int(i0_rounds[si, wi, ri, ai])
                            S = int(i0_shots[si, wi, ri, ai])
                            Rq = int(q0_rounds[si, wi, ri, ai])
                            Sq = int(q0_shots[si, wi, ri, ai])
                            # Guard against zero-sized entries
                            if R > 0 and S > 0:
                                data_reshaped['i0'][si, wi, ri, ai] = i0_stacked[si, wi, ri, ai, :R, :S]
                            else:
                                data_reshaped['i0'][si, wi, ri, ai] = None
                            if Rq > 0 and Sq > 0:
                                data_reshaped['q0'][si, wi, ri, ai] = q0_stacked[si, wi, ri, ai, :Rq, :Sq]
                            else:
                                data_reshaped['q0'][si, wi, ri, ai] = None

            data = data_reshaped



        # Read flags
        pulse_correction = bool(self.cfg.expt.get('pulse_correction', False))

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', True))

        if 'target_states' in self.cfg.expt and self.cfg.expt.target_states is not None:
            target_states = self.cfg.expt.target_states
        if target_states is None:
            # if no initial states provided, use the vacuum state times the number of cardinal states
            target_states = [qt.fock(mode_state_num, 0).unit() for _ in range(len(self.cfg.expt.cardinal_states))]

        parity_shot = bool(self.cfg.expt.get('parity_shot', False))


        nS, nW, nR, nApc = data['avgi'].shape
        pc_factor = 2 if pulse_correction else 1

        # sigma_z probe mode (resolved once; lane indices come from lane_layout below)
        sigma_z_mode = MM_base.lane_layout(self.cfg)['sigma_z_mode']
        threshold_val = self.cfg.device.readout.threshold[0]

        # Containers for parity results
        results = {
            'parity': np.empty((nS, nW, nR), dtype=object),
            'rho': np.empty((nS, nW, nR), dtype=object),
            'rho_rotated': np.empty((nS, nW, nR), dtype=object),
            'theta_opt': np.empty((nS, nW, nR), dtype=object),
        }
        if sigma_z_mode != 'off':
            results['sigma_z'] = np.empty((nS, nW, nR), dtype=object)
            results['sigma_z_raw'] = np.empty((nS, nW, nR), dtype=object)
            if sigma_z_mode == 'postselect':
                results['sigma_z_discard_frac'] = np.empty((nS, nW, nR), dtype=object)
        if parity_shot:
            results.update({
                'parity_shot': np.empty((nS, nW, nR), dtype=object),
                'parity_filtered': np.empty((nS, nW, nR), dtype=object),
                'rho_filtered': np.empty((nS, nW, nR), dtype=object),
                'rho_rotated_filtered': np.empty((nS, nW, nR), dtype=object),
                'theta_opt_filtered': np.empty((nS, nW, nR), dtype=object),
                'filter_map': np.empty((nS, nW, nR), dtype=object),
            })

        if pulse_correction:
            results.update({
                'pe_plus': np.empty((nS, nW, nR), dtype=object),
                'pe_minus': np.empty((nS, nW, nR), dtype=object),
                'parity_plus': np.empty((nS, nW, nR), dtype=object),
                'parity_minus': np.empty((nS, nW, nR), dtype=object),
            })
            if parity_shot:
                results.update({
                    'parity_shot_plus': np.empty((nS, nW, nR), dtype=object),
                    'parity_shot_minus': np.empty((nS, nW, nR), dtype=object),
                    'parity_filtered': np.empty((nS, nW, nR), dtype=object),
                    'rho_filtered': np.empty((nS, nW, nR), dtype=object),
                    'rho_rotated_filtered': np.empty((nS, nW, nR), dtype=object),
                    'theta_opt_filtered': np.empty((nS, nW, nR), dtype=object),
                    'filter_map_plus': np.empty((nS, nW, nR), dtype=object),
                    'filter_map_minus': np.empty((nS, nW, nR), dtype=object),
                })


        # Ensure expts is set for downstream single-shot binning
        nA = len(data['alpha'])

        for si in range(nS):
            for wi in range(nW):
                for ri in range(nR):
                    # Per-ri cfg clone with the right repeat_count so lane_layout (called
                    # inside bin_ss_data) computes read_num correctly for this slice.
                    cfg_local = AttrDict(deepcopy(self.cfg))
                    cfg_local.expt.expts = nA
                    cfg_local.expt.repeat_count = int(data['repeat_list'][ri])

                    layout = MM_base.lane_layout(cfg_local)
                    read_num = layout['read_num']
                    idx_step = read_num
                    idx_sz = layout['idx_sigma_z']
                    # Original behavior: all non-final lanes are treated as herald
                    # candidates -- but the sigma_z probe lane is NOT a parity herald,
                    # so exclude it (it is consumed for the sigma_z scalar / Tier 2 mask).
                    idx_parity_shot = np.arange(layout['idx_final'])
                    if idx_sz is not None:
                        idx_parity_shot = idx_parity_shot[idx_parity_shot != idx_sz]

                    # sigma_z scalar for this (state, wait, repeat): aggregate the
                    # sigma_z lane over all alphas (and both pulse-correction copies).
                    if idx_sz is not None:
                        _sz_buf = np.concatenate(
                            [np.asarray(data['i0'][si, wi, ri, k][0]).ravel()
                             for k in range(nApc)])
                        # Herald-mask sigma_z onto the same shots the parity uses.
                        _sz_lane = _sigma_z_lane_shots(_sz_buf, read_num, idx_sz,
                                                       idx_pre_selection=layout['idx_pre_selection_list'],
                                                       threshold=threshold_val)
                        _sz, _sz_raw = _compute_sigma_z(_sz_lane, threshold_val, cfg_local)
                        results['sigma_z'][si, wi, ri] = _sz
                        results['sigma_z_raw'][si, wi, ri] = _sz_raw

                    if pulse_correction:
                        shot_minus = {
                            'i0': np.stack([data['i0'][si, wi, ri, k][0] for k in range(0, nApc, 2)], axis=0),
                            'q0': np.stack([data['q0'][si, wi, ri, k][0] for k in range(0, nApc, 2)], axis=0),
                        }
                        shot_plus = {
                            'i0': np.stack([data['i0'][si, wi, ri, k][0] for k in range(1, nApc, 2)], axis=0),
                            'q0': np.stack([data['q0'][si, wi, ri, k][0] for k in range(1, nApc, 2)], axis=0),
                        }

                        # Herald sub-bins: pre-stride a single lane so bin_ss_data treats
                        # it as a plain single-readout slice (cfg_herald disables
                        # active_reset/parity_shot so lane_layout returns read_num=1).
                        data_parity_minus = {idx: {
                            'i0': shot_minus['i0'][:, idx::idx_step],
                            'q0': shot_minus['q0'][:, idx::idx_step],
                            'alpha': data['alpha'],
                        } for idx in idx_parity_shot}
                        data_parity_plus = {idx: {
                            'i0': shot_plus['i0'][:, idx::idx_step],
                            'q0': shot_plus['q0'][:, idx::idx_step],
                            'alpha': data['alpha'],
                        } for idx in idx_parity_shot}

                        # Main parity: hand bin_ss_data the raw lane-interleaved shots; it
                        # extracts the final lane AND applies pre_selection_reset masking.
                        data_minus = {'i0': shot_minus['i0'], 'q0': shot_minus['q0']}
                        data_plus  = {'i0': shot_plus['i0'],  'q0': shot_plus['q0']}

                        # Tier 2 (postselect): keep-mask (|g>) from the sigma_z lane,
                        # shaped (reps, nA) to match the parity_shot filter_map convention.
                        sz_fmap_minus = sz_fmap_plus = None
                        if sigma_z_mode == 'postselect' and idx_sz is not None:
                            sz_fmap_minus = (shot_minus['i0'][:, idx_sz::idx_step] < threshold_val).T
                            sz_fmap_plus  = (shot_plus['i0'][:, idx_sz::idx_step] < threshold_val).T
                            results['sigma_z_discard_frac'][si, wi, ri] = float(
                                1 - 0.5 * (sz_fmap_minus.mean() + sz_fmap_plus.mean()))

                        # plot the i0 and q0 data for debugging
                        if debug:
                            # Final-lane view for the I/Q scatter plot
                            final_m_i = shot_minus['i0'][:, idx_step-1::idx_step]
                            final_m_q = shot_minus['q0'][:, idx_step-1::idx_step]
                            final_p_i = shot_plus['i0'][:, idx_step-1::idx_step]
                            final_p_q = shot_plus['q0'][:, idx_step-1::idx_step]

                            fig, ax = plt.subplots(2, 1, sharex=True)
                            ax[0].plot(np.mean(final_m_i, axis=1).T, np.mean(final_m_q, axis=1).T, linestyle='', marker='o', label='data minus')
                            [ax[0].plot(np.mean(data_parity_minus[i]['i0'], axis=1).T, np.mean(data_parity_minus[i]['q0'], axis=1).T, label=f'parity minus {i}',  linestyle='', marker='o') for i in idx_parity_shot]
                            [ax[1].plot(np.mean(data_parity_plus[i]['i0'], axis=1).T, np.mean(data_parity_plus[i]['q0'], axis=1).T, label=f'parity plus {i}', linestyle='', marker='o') for i in idx_parity_shot]
                            ax[1].plot(np.mean(final_p_i, axis=1).T, np.mean(final_p_q, axis=1).T, label='data plus',  linestyle='', marker='o')

                            # mark the default g / e states
                            ie = self.cfg.device.readout.Ie
                            ig = self.cfg.device.readout.Ig
                            ax[0].plot(ig[0], 0, label='g', marker='o')
                            ax[0].plot(ie[0], 0, label='e', marker='o')
                            ax[1].plot(ig[0], 0, label='g', marker='o')
                            ax[1].plot(ie[0], 0, label='e', marker='o')

                            for a in ax:
                                a.legend()
                                a.axis('equal')
                            ax[0].legend()
                            ax[1].legend()
                            fig.show()

                        if parity_shot:
                            # cfg for herald sub-bins: disable active_reset & parity_shot so
                            # lane_layout returns read_num=1, matching the pre-strided shape.
                            cfg_herald = AttrDict(deepcopy(cfg_local))
                            cfg_herald.expt.active_reset = False
                            cfg_herald.expt.parity_shot = False

                            nA = len(data['alpha'])
                            nSel = len(idx_parity_shot)
                            parity_shot_plus = np.empty((nA, nSel))
                            parity_shot_minus = np.empty((nA, nSel))
                            filter_map_plus = np.ones((self.cfg.expt.reps, nA), dtype=bool)
                            filter_map_minus = np.ones((self.cfg.expt.reps, nA), dtype=bool)
                            # Compose Tier 2 sigma_z keep-mask into the parity_shot filter.
                            if sz_fmap_minus is not None:
                                filter_map_minus &= sz_fmap_minus
                                filter_map_plus &= sz_fmap_plus
                            for j, idx in enumerate(idx_parity_shot):
                                _wtemp_minus = WignerAnalysis(data=data_parity_minus[idx], config=cfg_herald, mode_state_num=mode_state_num, alphas=data['alpha'])
                                _wtemp_plus = WignerAnalysis(data=data_parity_plus[idx], config=cfg_herald, mode_state_num=mode_state_num, alphas=data['alpha'])
                                _pe_minus, _shots_minus = _wtemp_minus.bin_ss_data(return_shots=True)
                                _pe_plus, _shots_plus = _wtemp_plus.bin_ss_data(return_shots=True)
                                filter_map_minus &= (_shots_minus == 0)
                                filter_map_plus &= (_shots_plus == 0)

                                _parity_plus = (1 - _pe_plus) - _pe_plus
                                _parity_minus = (1 - _pe_minus) - _pe_minus
                                parity_shot_plus[:, j] = _parity_plus
                                parity_shot_minus[:, j] = _parity_minus

                        wa_minus = WignerAnalysis(data=data_minus, config=cfg_local, mode_state_num=mode_state_num, alphas=data['alpha'])
                        wa_plus = WignerAnalysis(data=data_plus, config=cfg_local, mode_state_num=mode_state_num, alphas=data['alpha'])
                        scale_parity = self.cfg.device.manipulate.alpha_scale[self.man_mode_idx]

                        if parity_shot:
                            # apply filter map to shots
                            pe_minus_filtered = wa_minus.bin_ss_data(filter_map=filter_map_minus)
                            pe_plus_filtered = wa_plus.bin_ss_data(filter_map=filter_map_plus)
                            parity_minus_filtered = (1 - pe_minus_filtered) - pe_minus_filtered
                            parity_plus_filtered = (1 - pe_plus_filtered) - pe_plus_filtered
                            parity_filtered = (parity_minus_filtered - parity_plus_filtered) / 2 / scale_parity 
                            # raise an error if one of the mask is true, ie no shot are available for this 
                            #displacement 
                            mask = parity_filtered.mask
                            if np.any(mask):
                                print(f'Warning: No valid shots available after filtering for some alphas at state {si}, wait {wi}, repeat {ri}. Setting parity to NaN for these alphas.')
                                # add the zero at the corresponding position
                                parity_filtered = np.ma.filled(parity_filtered, np.nan)

                            data_slice_filtered = {'alpha': data['alpha'], 'parity': parity_filtered}
                            wigner_analysis_filtered = WignerAnalysis(data=data_slice_filtered, config=cfg_local,
                                                    mode_state_num=mode_state_num, alphas=data['alpha'])
                            wigner_result_filtered = wigner_analysis_filtered.wigner_analysis_results(parity_filtered,
                                                                        initial_state=target_states[si],
                                                                        rotate=rotate)



                        # Tier 2 masks the MAIN parity by the sigma_z keep-mask;
                        # sz_fmap_* is None unless sigma_z_mode == 'postselect'.
                        pe_minus = wa_minus.bin_ss_data(filter_map=sz_fmap_minus)
                        pe_plus = wa_plus.bin_ss_data(filter_map=sz_fmap_plus)

                        if sz_fmap_minus is not None:
                            pe_minus = np.ma.filled(pe_minus, np.nan)
                            pe_plus = np.ma.filled(pe_plus, np.nan)

                        parity_minus = (1 - pe_minus) - pe_minus
                        parity_plus = (1 - pe_plus) - pe_plus
                        parity = (parity_minus - parity_plus) / 2 / scale_parity
                        data_slice = {'alpha': data['alpha'], 'parity': parity}

                        wigner_analysis = WignerAnalysis(data=data_slice, config=cfg_local,
                                             mode_state_num=mode_state_num, alphas=data['alpha'])

                        wigner_result = wigner_analysis.wigner_analysis_results(parity,
                                                                    initial_state=target_states[si],
                                                                    rotate=rotate)


                        results['pe_minus'][si, wi, ri] = pe_minus
                        results['pe_plus'][si, wi, ri] = pe_plus
                        results['parity_minus'][si, wi, ri] = parity_minus
                        results['parity_plus'][si, wi, ri] = parity_plus
                        results['parity'][si, wi, ri] = parity
                        results['rho'][si, wi, ri] = wigner_result['rho']
                        results['rho_rotated'][si, wi, ri] = wigner_result['rho_rotated']
                        results['theta_opt'][si, wi, ri] = wigner_result['theta_max']

                        if parity_shot:
                            results['parity_filtered'][si, wi, ri] = parity_filtered
                            results['rho_filtered'][si, wi, ri] = wigner_result_filtered['rho']
                            results['rho_rotated_filtered'][si, wi, ri] = wigner_result_filtered['rho_rotated']
                            results['theta_opt_filtered'][si, wi, ri] = wigner_result_filtered['theta_max']
                            results['parity_shot_plus'][si, wi, ri] = parity_shot_plus
                            results['parity_shot_minus'][si, wi, ri] = parity_shot_minus
                            results['filter_map_plus'][si, wi, ri] = filter_map_plus
                            results['filter_map_minus'][si, wi, ri] = filter_map_minus


                    else:
                        # No correction: hand bin_ss_data the raw lane-interleaved shots;
                        # it handles final-lane extraction AND pre_selection_reset masking.
                        data_w = {
                            'i0': np.stack([data['i0'][si, wi, ri, k][0] for k in range(0, nApc)], axis=0),
                            'q0': np.stack([data['q0'][si, wi, ri, k][0] for k in range(0, nApc)], axis=0),
                        }
                        wa = WignerAnalysis(data=data_w, config=cfg_local, mode_state_num=mode_state_num, alphas=data['alpha'])

                        # Tier 2 (postselect): keep-mask (|g>) from the sigma_z lane.
                        sz_fmap = None
                        if sigma_z_mode == 'postselect' and idx_sz is not None:
                            sz_fmap = (data_w['i0'][:, idx_sz::idx_step] < threshold_val).T
                            results['sigma_z_discard_frac'][si, wi, ri] = float(1 - sz_fmap.mean())

                        pe = wa.bin_ss_data(filter_map=sz_fmap)
                        if sz_fmap is not None:
                            pe = np.ma.filled(pe, np.nan)
                        parity = (1 - pe) - pe
                        results['parity'][si, wi, ri] = parity

                        data_slice = {'alpha': data['alpha'], 'parity': parity}
                        wigner_analysis = WignerAnalysis(data=data_slice, config=cfg_local,
                                                mode_state_num=mode_state_num, alphas=data['alpha'])
                        wigner_result = wigner_analysis.wigner_analysis_results(parity,
                                                                    initial_state=target_states[si],
                                                                    rotate=rotate)
                        results['rho'][si, wi, ri] = wigner_result['rho']
                        results['rho_rotated'][si, wi, ri] = wigner_result['rho_rotated']
                        results['theta_opt'][si, wi, ri] = wigner_result['theta_max']


        # Merge into data
        for k, v in results.items():
            data[k] = v
        return data

    def display(self, data=None, state_idx=0, wait_idx=0, repeat_idx=0,
                 mode_state_num=None, target_state=None, rotate=None,
                   state_label='', debug_components=False, filtered=False, **kwargs):
        """
    Display a selected (state, wait, repeat) slice using WignerAnalysis reconstruction.

        Parameters:
        - state_idx: index in expt.cardinal_states (0..3)
        - wait_idx: index in wait_list
        - repeat_idx: index in repeat_list
        - mode_state_num: Hilbert space cutoff (default cfg.expt.display_mode_state_num or 5)
        - initial_state: qutip.Qobj (ket); if None, defaults to |0>
        - rotate: default cfg.expt.display_rotate or False
        - state_label: annotation for plot
        """
        if data is None:
            data = self.data

        # Defaults
        if mode_state_num is None:
            mode_state_num = int(getattr(self.cfg.expt, 'display_mode_state_num', 5))
        if rotate is None:
            rotate = bool(getattr(self.cfg.expt, 'display_rotate', False))

        alphas = data['alpha']
        if filtered:
            # first check if filtered data exists
            if 'parity_filtered' in data:
                parity = data['parity_filtered'][state_idx, wait_idx, repeat_idx]
            else:
                print("No filtered parity data available, falling back to unfiltered.")
                parity = data['parity'][state_idx, wait_idx, repeat_idx]
        else:
            parity = data['parity'][state_idx, wait_idx, repeat_idx]
        if parity is None:
            print("No parity data available for the selected slice.")
            return
        if len(parity) != len(alphas):
            raise ValueError(f'Length mismatch: parity ({len(parity)}) vs alpha ({len(alphas)}).')

        if target_state is None:
            if 'target_states' in self.cfg.expt:
                target_state = self.cfg.expt.target_states[state_idx]
            else:
                target_state = qt.fock(mode_state_num, 0).unit()

        # Minimal dict for analysis API compatibility if needed
        data_slice = {'alpha': alphas, 'parity': parity}
        wigner_analysis = WignerAnalysis(data=data_slice, config=self.cfg, mode_state_num=mode_state_num, alphas=alphas)
        results = wigner_analysis.wigner_analysis_results(parity, initial_state=target_state, rotate=rotate)
        fig = wigner_analysis.plot_wigner_reconstruction_results(results, initial_state=target_state, state_label=state_label)

        # Optional debug components: show pe_plus/minus and parity_plus/minus vs |alpha|
        if debug_components:
            has_pe = ('pe_plus' in data) and ('pe_minus' in data)
            has_par = ('parity_plus' in data) and ('parity_minus' in data)
            if has_pe and has_par:
                pe_plus = data['pe_plus'][state_idx, wait_idx, repeat_idx] if data['pe_plus'][state_idx, wait_idx, repeat_idx] is not None else None
                pe_minus = data['pe_minus'][state_idx, wait_idx, repeat_idx] if data['pe_minus'][state_idx, wait_idx, repeat_idx] is not None else None
                parity_plus = data['parity_plus'][state_idx, wait_idx, repeat_idx] if data['parity_plus'][state_idx, wait_idx, repeat_idx] is not None else None
                parity_minus = data['parity_minus'][state_idx, wait_idx, repeat_idx] if data['parity_minus'][state_idx, wait_idx, repeat_idx] is not None else None
                if pe_plus is not None and pe_minus is not None and parity_plus is not None and parity_minus is not None:
                    try:
                        alpha_abs = np.abs(alphas)
                        fig_dbg, ax = plt.subplots(1, 2, figsize=(12, 6))
                        pe = pe_plus + pe_minus
                        ax[0].plot(alpha_abs, pe, 'o', label='pe')
                        ax[0].plot(alpha_abs, pe_plus, 'o', label='pe_plus')
                        ax[0].plot(alpha_abs, pe_minus, 'o', label='pe_minus')

                        ax[1].plot(alpha_abs, 2/np.pi * np.asarray(parity), 'o', label='parity')
                        ax[1].plot(alpha_abs, 2/np.pi * np.asarray(parity_plus), 'o', label='parity_plus')
                        ax[1].plot(alpha_abs, 2/np.pi * np.asarray(parity_minus), 'o', label='parity_minus')

                        ax[0].set_xlabel('Alpha')
                        ax[0].set_ylabel('Probability')
                        ax[1].set_xlabel('Alpha')
                        ax[1].set_ylabel('Parity')
                        ax[0].legend()
                        ax[1].legend()
                        plt.tight_layout()
                    except Exception as e:
                        print(f"Debug components plot skipped due to: {e}")
                else:
                    print("Debug components requested, but components for selected slice are missing.")
            else:
                print("Debug components requested, but pe+/pe- or parity+/parity- not found in data.")
        return fig

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data

        serial = {}

        # Metadata
        for key in ("alpha", "wait_list", "repeat_list"):
            if key in data:
                serial[key] = np.asarray(data[key])

        # Numeric tensors
        for key in ("amps", "phases"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype != object:
                serial[key] = data[key]

        def stack_1d(obj_arr):
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = elem
                    break
            if sample is None:
                return np.zeros((*obj_arr.shape, 0), dtype=float)
            arr_sample = np.asarray(sample)
            # Handle scalar or 0-D array
            L = int(arr_sample.size) if arr_sample.ndim == 0 else int(arr_sample.shape[-1])
            out = np.zeros((*obj_arr.shape, L), dtype=float)
            for idx in np.ndindex(obj_arr.shape):
                if obj_arr[idx] is not None:
                    v = np.asarray(obj_arr[idx]).reshape(-1)
                    out[idx] = v[:L] if v.shape[0] >= L else np.pad(v, (0, L-v.shape[0]))
            return out
        
        # I should probably generalize this to N-dimensions at some point
        # and also move it to slab or MM_base for reuse

        def stack_2d(obj_arr):
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = elem
                    break
            if sample is None:
                return np.zeros((*obj_arr.shape, 0), dtype=float)
            arr_sample = np.asarray(sample)
            # Handle scalar or shape mismatch
            if arr_sample.ndim == 0:
                M = 1
            elif arr_sample.ndim == 1:
                M = arr_sample.shape[0]
            else:
                M = arr_sample.shape[-1]
            out = np.zeros((*obj_arr.shape, M), dtype=float)
            for idx in np.ndindex(obj_arr.shape):
                if obj_arr[idx] is not None:
                    a = np.asarray(obj_arr[idx])
                    v = a[0].reshape(-1) if a.ndim >= 2 else a.reshape(-1)
                    out[idx] = v[:M] if v.shape[0] >= M else np.pad(v, (0, M-v.shape[0]))
            return out
        
        def stack_2d_variable(obj_arr, fill_value=np.nan):
            """
            Stack an object array of 2D arrays with variable shapes into a single
            6D tensor with padding.

            Input:
                obj_arr: np.ndarray of dtype=object, shape (nS, nW, nR, nK)
                        each element A[idx] is a 2D array of shape (rounds, shots)
            Output:
                stacked: float array, shape (nS, nW, nR, nK, Rmax, Smax)
                rounds_shape: int array, shape (nS, nW, nR, nK)
                shots_shape: int array, shape (nS, nW, nR, nK)
            """
            if not isinstance(obj_arr, np.ndarray) or obj_arr.dtype != object:
                raise ValueError("stack_2d_variable expects an object ndarray")

            # Find a non-None sample to infer ndim and dtype
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = np.asarray(elem)
                    break

            if sample is None:
                # No data at all: return empty arrays with appended dims = 0
                base_shape = obj_arr.shape + (0, 0)
                stacked = np.full(base_shape, fill_value, dtype=float)
                rounds_shape = np.zeros(obj_arr.shape, dtype=int)
                shots_shape = np.zeros(obj_arr.shape, dtype=int)
                return stacked, rounds_shape, shots_shape

            sample_arr = np.asarray(sample)
            if sample_arr.ndim < 1:
                # Promote scalars to (1, 1)
                Rmax, Smax = 1, 1
            elif sample_arr.ndim == 1:
                # Treat as (1, L)
                Rmax, Smax = 1, sample_arr.shape[0]
            else:
                Rmax, Smax = sample_arr.shape[0], sample_arr.shape[1]

            # First pass: determine global maxima over all entries
            for elem in np.ravel(obj_arr):
                if elem is None:
                    continue
                a = np.asarray(elem)
                if a.ndim == 0:
                    r, s = 1, 1
                elif a.ndim == 1:
                    r, s = 1, a.shape[0]
                else:
                    r, s = a.shape[0], a.shape[1]
                if r > Rmax:
                    Rmax = r
                if s > Smax:
                    Smax = s

            stacked = np.full(obj_arr.shape + (Rmax, Smax), fill_value, dtype=float)
            rounds_shape = np.zeros(obj_arr.shape, dtype=int)
            shots_shape = np.zeros(obj_arr.shape, dtype=int)

            # Second pass: copy with padding
            it = np.nditer(obj_arr, flags=['multi_index', 'refs_ok'])
            for x in it:
                idx = it.multi_index
                if x.item() is None:
                    continue
                a = np.asarray(x.item())
                if a.ndim == 0:
                    r, s = 1, 1
                    a2 = a.reshape(1, 1)
                elif a.ndim == 1:
                    r, s = 1, a.shape[0]
                    a2 = a.reshape(1, -1)
                else:
                    r, s = a.shape[0], a.shape[1]
                    a2 = a
                r_use = min(r, Rmax)
                s_use = min(s, Smax)
                stacked[idx + (slice(0, r_use), slice(0, s_use))] = a2[:r_use, :s_use]
                rounds_shape[idx] = r
                shots_shape[idx] = s

            return stacked, rounds_shape, shots_shape
        
        def stack_3d(obj_arr):
            """Stack object array of 2D matrices to (obj_arr.shape, rows, cols) tensor"""
            sample = None
            for elem in np.ravel(obj_arr):
                if elem is not None:
                    sample = elem
                    break
            if sample is None:
                return np.zeros((*obj_arr.shape, 0, 0), dtype=complex)
            arr_sample = np.asarray(sample)
            # Determine matrix dimensions
            if arr_sample.ndim == 2:
                rows, cols = arr_sample.shape
            else:
                # Handle edge case: flatten and take sqrt to estimate square matrix
                size = arr_sample.size
                rows = cols = int(np.sqrt(size))
            
            out = np.zeros((*obj_arr.shape, rows, cols), dtype=complex)
            for idx in np.ndindex(obj_arr.shape):
                if obj_arr[idx] is not None:
                    mat = np.asarray(obj_arr[idx])
                    if mat.ndim == 2:
                        r, c = mat.shape
                        out[tuple(idx) + (slice(None, min(r, rows)), slice(None, min(c, cols)))] = mat[:rows, :cols]
                    else:
                        # Try to reshape if it's flat
                        mat_flat = mat.reshape(-1)
                        out[tuple(idx)] = mat_flat[:rows*cols].reshape(rows, cols)
            return out

        # Apply stacking
        if "avgi" in data and isinstance(data["avgi"], np.ndarray) and data["avgi"].dtype == object:
            serial["avgi"] = stack_1d(data["avgi"])
        if "avgq" in data and isinstance(data["avgq"], np.ndarray) and data["avgq"].dtype == object:
            serial["avgq"] = stack_1d(data["avgq"])

        if "i0" in data and isinstance(data["i0"], np.ndarray) and data["i0"].dtype == object:
            serial["i0"], serial["i0_rounds"], serial["i0_shots"] = stack_2d_variable(data["i0"])

        if "q0" in data and isinstance(data["q0"], np.ndarray) and data["q0"].dtype == object:
            serial["q0"], serial["q0_rounds"], serial["q0_shots"] = stack_2d_variable(data["q0"])



        # Parity arrays
        for key in ("parity", "pe_plus", "pe_minus", "parity_plus", "parity_minus"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                serial[key] = stack_1d(data[key])
            elif key in data:
                serial[key] = np.asarray(data[key])


        serial['theta_opt'] = stack_1d(data['theta_opt']) if 'theta_opt' in data else None


        if "parity_shot" in self.cfg.expt and self.cfg.expt.parity_shot:

            # check for filtered versions
            for key in ("parity_filtered", "pe_filtered", "parity_plus_filtered", "parity_minus_filtered"):
                if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                    serial[key] = stack_1d(data[key])
                elif key in data:
                    serial[key] = np.asarray(data[key])


            if "filter_map_plus" in data and isinstance(data["filter_map_plus"], np.ndarray) and data["filter_map_plus"].dtype == object:
                serial["filter_map_plus"] = stack_2d(data["filter_map_plus"])
            if "filter_map_minus" in data and isinstance(data["filter_map_minus"], np.ndarray) and data["filter_map_minus"].dtype == object:
                serial["filter_map_minus"] = stack_2d(data["filter_map_minus"])
            serial['theta_opt_filtered'] = stack_1d(data['theta_opt_filtered']) if 'theta_opt_filtered' in data else None


        # Density matrices (2D)
        for key in ("rho", "rho_rotated"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                serial[key] = stack_3d(data[key])
            elif key in data:
                serial[key] = np.asarray(data[key])

        # check for filtered versions
        for key in ("rho_filtered", "rho_rotated_filtered"):
            if key in data and isinstance(data[key], np.ndarray) and data[key].dtype == object:
                serial[key] = stack_3d(data[key])
            elif key in data:
                serial[key] = np.asarray(data[key])

        super().save_data(data=serial)
