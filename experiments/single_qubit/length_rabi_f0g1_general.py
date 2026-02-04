import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

from qick import *
# from qick.helpers import gauss, sin2, tanh, flat_top_gauss
from slab import Experiment, dsfit, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import *

"""
Measures Rabi oscillations by sweeping over the duration of the qubit drive pulse.
This is a preliminary measurement to prove that we see Rabi oscillations.
This measurement is followed up by the Amplitude Rabi experiment.
"""


class LengthRabiF0g1GeneralProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        qTest = self.cfg.expt.qubits[0]
        self.MM_base_initialize()
        self.drive_freq = self.cfg.expt.freq
        # print(f"Using drive frequency {self.drive_freq} MHz for qubit {qTest}")
        self.test_pulse_str =  [[self.drive_freq], [self.cfg.expt.gain], [self.cfg.expt.length_placeholder], [0],
                      [self.f0g1_ch[qTest]], ["flat_top"], [self.cfg.device.manipulate.ramp_sigma]]    # flux drive = [low/high (ch), freq, gain, ramp_sigma(us)] RF flux modulation, gaussian flat top pulse
        # print(self.test_pulse_str)

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]


        # phase reset
        self.reset_and_sync()

        # Active Reset
        if cfg.expt.active_reset:
            self.active_reset(man_reset = True, storage_reset = True)

        #  prepulse
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='prepulse')

        self.sync_all()  # align channels

        # pre-rotation
        if self.cfg.expt.pi_ge_before: # g0 to e0
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[qTest],
                                 phase=0, gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            self.sync_all()

        if self.cfg.expt.pi_ef_before: # g1 - e1
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ef_reg[qTest],
                                 phase=0, gain=self.pi_ef_gain, waveform="pi_qubit_ef")
            self.sync_all()


        for i in range(2*self.cfg.expt.err_amp_reps+1):
            if self.cfg.expt.length_placeholder>0:
                self.custom_pulse(cfg, self.test_pulse_str, prefix='pi_test_ramp')
                self.sync_all()  # align channels

        if self.cfg.expt.pi_ge_after:  
            self.setup_and_pulse(ch=self.qubit_chs[qTest],
                                 style="arb",
                                 freq=self.f_ef_reg[qTest],
                                 phase=0,
                                 gain=self.pi_ef_gain,
                                 waveform="pi_qubit_ef")
            self.sync_all()

        if self.cfg.expt.swap_lossy:
            self.man_reset(man_idx = self.cfg.expt.check_man_reset[1])
            self.sync_all()
        # self.custom_pulse(cfg, cfg.expt.check_man_reset_pi, prefix='pi3')
        # if self.cfg.expt.postpulse: 
        #     self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix='postpulse')

        # align channels and wait 50ns and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure_wrapper()

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        qTest = 0
        cfg=AttrDict(self.cfg)
        # print(np.average(self.di_buf[0]))
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[qTest]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[qTest]
        return shots_i0, shots_q0


class LengthRabiGeneralF0g1Experiment(Experiment):
    """
    Length Rabi Experiment
    Experimental Config
    expt = dict(
        start: start length [us],
        step: length step, 
        expts: number of different length experiments, 
        reps: number of reps,
        gain: gain to use for the qubit pulse
        pulse_type: 'gauss' or 'const'
        checkZZ: True/False for putting another qubit in e (specify as qA)
        checkEF: does ramsey on the EF transition instead of ge
        qubits: if not checkZZ, just specify [1 qubit]. if checkZZ: [qA in e , qB sweeps length rabi]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='LengthRabiGeneralF0g1', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update(
                                    {key3: [value3]*num_qubits_sample})
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        lengths = self.cfg.expt["start"] + \
            self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])

        data = {"xpts": [], "idata": [], "qdata": [], "avgi": [], "avgq": []}

        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        if self.cfg.expt.check_man_reset[0]: read_num = 1

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = LengthRabiF0g1GeneralProgram(
                soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(
                self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug, readouts_per_experiment=read_num)
            avgi = avgi[0][-1]
            avgq = avgq[0][-1]
            idata, qdata = lengthrabi.collect_shots()
            # amp = np.abs(avgi+1j*avgq)  # Calculating the magnitude
            # phase = np.angle(avgi+1j*avgq)  # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            if self.cfg.expt.active_reset or self.cfg.expt.check_man_reset[0]:
                #print('getting i data')
                data["idata"].append(idata)
                data["qdata"].append(qdata)

        for k, a in data.items():
            data[k] = np.array(a)


        if self.cfg.expt.normalize:
            from experiments.single_qubit.normalize import normalize_calib
            g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)

            data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
            data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
            data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]

        self.data = data
        return data

    def analyze(self, data=None, fit=True, fitparams=None, **kwargs):
        if data is None:
            data = self.data

        station = kwargs.pop('station', None)

        # Detect 2D sweep data (from SweepRunner)
        # 2D data has 'freq_sweep' key and avgi is a 2D array
        is_2d = (
            'freq_sweep' in data
            and 'avgi' in data
            and hasattr(data['avgi'], 'ndim')
            and data['avgi'].ndim == 2
        )

        if is_2d:
            from fitting.fit_display_classes import ChevronFitting
            # Extract time axis - handle both 1D and 2D xpts
            time = data['xpts'][0] if data['xpts'].ndim > 1 else data['xpts']
            analysis = ChevronFitting(
                frequencies=data['freq_sweep'],
                time=time,
                response_matrix=data['avgi'],
                config=self.cfg,
                station=station,
            )
            analysis.analyze()
            self._chevron_analysis = analysis
            return data

        # Original 1D case - delegate to LengthRabiFitting
        from fitting.fit_display_classes import LengthRabiFitting
        analysis = LengthRabiFitting(data, fit=fit, fitparams=fitparams, config=self.cfg, station=station)
        analysis.analyze()

        # Store results in self for access by postprocessor
        self._length_rabi_analysis = analysis

        # add the pi and pi/2 lengths to the experiment config for easy access
        p_fit = analysis.results.get('fit_avgi', None)
        if p_fit is not None:
            if p_fit[2] > 180:
                p_fit[2] = p_fit[2] - 360
            elif p_fit[2] < -180:
                p_fit[2] = p_fit[2] + 360
            if p_fit[2] < 0:
                pi_length = (1/2 - p_fit[2]/180)/2/p_fit[1]
            else:
                pi_length = (3/2 - p_fit[2]/180)/2/p_fit[1]
            pi2_length = pi_length/2
            self._length_rabi_analysis.results['pi_length'] = pi_length
            self._length_rabi_analysis.results['pi2_length'] = pi2_length

        return data

    def display(self, data=None, fit=True, title_str='Length Rabi General F0g1', **kwargs):
        if data is None:
            data = self.data

        # 2D case - use ChevronFitting's display_results
        if hasattr(self, '_chevron_analysis'):
            # Note: ChevronFitting uses display_results() not display()
            self._chevron_analysis.display_results(
                save_fig=kwargs.get('save_fig', False),
                title=title_str,
            )
            return

        # Original 1D case - delegate to LengthRabiFitting
        if hasattr(self, '_length_rabi_analysis'):
            self._length_rabi_analysis.display(title_str=title_str, **kwargs)
        else:
            # Fallback: create a new analysis object
            from fitting.fit_display_classes import LengthRabiFitting
            station = kwargs.pop('station', None)
            analysis = LengthRabiFitting(data, config=self.cfg, station=station)
            analysis.display(title_str=title_str, **kwargs)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname
