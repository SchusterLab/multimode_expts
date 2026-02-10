import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import time

from qick import *
from qick.helpers import gauss
from slab import Experiment, dsfit, AttrDict

import fitting.fitting as fitter
from experiments.MM_base import *

"""

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely"  narrow peaks!
"""
class FluxSpectroscopyF0g1Program(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.qubits = self.cfg.expt.qubit

        qTest = self.qubits[0]

        self.MM_base_initialize()

        print('flux drive:', self.cfg.expt.flux_drive)
        if self.cfg.expt.flux_drive[0] == 'low':
            self.rf_ch = cfg.hw.soc.dacs.flux_low.ch
            self.rf_ch_types = cfg.hw.soc.dacs.flux_low.type
        elif self.cfg.expt.flux_drive[0] == 'high':
            self.rf_ch = cfg.hw.soc.dacs.flux_high.ch
            self.rf_ch_types = cfg.hw.soc.dacs.flux_high.type
        else:
            raise ValueError(f"Invalid flux drive option {self.cfg.expt.flux_drive[0]}. Must be 'low' or 'high'.")


        # get register page for qubit_chs
        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]
        self.rf_rps = [self.ch_page(ch) for ch in self.rf_ch]

        self.f_ge_reg = [self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_ef_reg = [self.freq2reg(
            cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])]

        # self.f_ge_resolved_reg = [self.freq2reg(
        #     self.cfg.expt.qubit_resolved_pi[0], gen_ch=self.qubit_chs[qTest])]

        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.f_rf_reg = [self.freq2reg(self.cfg.expt.flux_drive[1], gen_ch=self.rf_ch[0])]

        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(
            self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(
            self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []

        # declare res dacs
        mask = None
        mixer_freq = 0  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = None
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest],
                         mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest],
                             freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(
                    ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # define pi_test_ramp as the pulse that we are calibrating with ramsey, update in outer loop over averager program
        self.pi_test_ramp = self.us2cycles(
            cfg.device.qubit.ramp_sigma[qTest], gen_ch=self.qubit_chs[qTest])
        self.rf_gain_test = self.cfg.expt.flux_drive[2]  # gain we are trying to play

        # define pisigma_ge as the ge pulse for the qubit that we are calibrating the pulse on
        self.pisigma_ge = self.us2cycles(
            cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ge value
        self.pisigma_ef = self.us2cycles(
            cfg.device.qubit.pulses.pi_ef.sigma[qTest], gen_ch=self.qubit_chs[qTest])  # default pi_ef value
        # self.pisigma_resolved = self.us2cycles(
        #     self.cfg.expt.qubit_resolved_pi[3], gen_ch=self.qubit_chs[qTest])  # default resolved pi value

        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.f_ef_init_reg = self.f_ef_reg[qTest]
        # self.f_ge_resolved_int_reg = self.f_ge_resolved_reg[qTest]
        self.rf_freq_reg = self.f_rf_reg[qTest]

        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_ef_init = self.cfg.device.qubit.pulses.pi_ef.gain[qTest]

        self.frequency = cfg.expt.frequency

        if self.cfg.expt.flux_drive[0] == 'low':
            self.declare_gen(ch=self.rf_ch[0], nqz=cfg.hw.soc.dacs.flux_low.nyquist[0], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=self.rf_ch[0])
        elif self.cfg.expt.flux_drive[0] == 'high':
            self.declare_gen(ch=self.rf_ch[0], nqz=cfg.hw.soc.dacs.flux_high.nyquist[0], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=self.rf_ch[0])
        else:
            raise ValueError(f"Invalid flux drive option {self.cfg.expt.flux_drive[0]}. Must be 'low' or 'high'.")
        self.freqreg = self.freq2reg(self.frequency, gen_ch=self.rf_ch[0])


        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_test_ramp", sigma=self.pi_test_ramp,
                       length=self.pi_test_ramp*2*cfg.device.qubit.ramp_sigma_num[qTest])
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge",
                       sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef",
                       sigma=self.pisigma_ef, length=self.pisigma_ef*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_resolved",
        #                sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)

        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        ## ALL ACTIVE RESET REQUIREMENTS
        # read val definition
        self.r_read_q = 3   # ge active reset register
        self.r_read_q_ef = 4   # ef active reset register
        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # threshold definition
        self.r_thresh_q = 5  # Define a location to store the threshold info

        # # multiplication bc the readout is summed, so need common thing to compare to
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))

        # Define a location to store a counter for how frequently the condj is triggered
        self.r_counter = 7
        self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0

        self.sync_all(self.us2cycles(0.2))

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = self.qubits[0]

        # active reset 
        if cfg.expt.active_reset:
            self.active_reset()

        self.sync_all()
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre')
        self.sync_all()
        # RF flux modulation

        print(f'Playing flux drive at frequency {self.frequency} MHz with gain {self.rf_gain_test}')
        print('channel', self.rf_ch[0])

        self.setup_and_pulse(ch=self.rf_ch[0],
                             style="const",
                             freq=self.freqreg,
                             phase=0,
                             gain=self.cfg.expt.flux_drive[2],
                             length=self.us2cycles(self.cfg.expt.flux_drive[3]))

        self.sync_all()  # align channels

        # post pulse
        if cfg.expt.postpulse:
            self.custom_pulse(cfg, cfg.expt.post_sweep_pulse, prefix = 'post')

        # align channels and wait 50ns and measure
        self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs,
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[0])
        )

# ====================================================== #

class FluxSpectroscopyF0g1Experiment(Experiment):
    """
    RF Spectroscopy Experiment
    Experimental Config
    expt = dict(
        start: start frequency (MHz), 
        step: frequency step (MHz), 
        expts: number of experiments, 
        pulse_e: boolean to add e pulse prior to measurement
        pulse_f: boolean to add f pulse prior to measurement
        reps: number of reps
        )
    """

    def __init__(self, soccfg=None, path='', prefix='FluxSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        xpts=self.cfg.expt["start"] + self.cfg.expt["step"]*np.arange(self.cfg.expt["expts"])

        # Calculate read_num to account for active_reset measurements
        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

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

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], "idata":[], "qdata":[]}
        for f in tqdm(xpts, disable=not progress):
            self.cfg.expt.frequency = f
            rspec = FluxSpectroscopyF0g1Program(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = rspec
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc],
                                       load_pulses=True,
                                       progress=False,
                                       debug=debug,
                                       readouts_per_experiment=read_num)

            avgi = avgi[0][-1]
            avgq = avgq[0][-1]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["xpts"].append(f)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)
            if self.cfg.expt.active_reset:
                idata, qdata = rspec.collect_shots()
                data["idata"].append(idata)
                data["qdata"].append(qdata)

        for k, a in data.items():
            data[k]=np.array(a)

        self.data=data

        return data

    def analyze(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data

        # Delegate to the newer implementation in fit_display_classes
        from fitting.fit_display_classes import Spectroscopy
        spec_analysis = Spectroscopy(data, signs=signs, config=self.cfg, station=None)
        spec_analysis.analyze(fit=fit)

        # Store analysis object for display() to use
        self._spec_analysis = spec_analysis

        return data

    def display(self, data=None, fit=True, signs=[1,1,1], title='Storage Spectroscopy', **kwargs):
        if data is None:
            data=self.data

        # Delegate to the newer implementation in fit_display_classes
        # Use the analysis object created in analyze() if available
        if hasattr(self, '_spec_analysis'):
            vlines = kwargs.get('vlines', None)
            self._spec_analysis.display(title=title, vlines=vlines, fit=fit)
        else:
            # Fallback: create a new analysis object
            from fitting.fit_display_classes import Spectroscopy
            spec_analysis = Spectroscopy(data, signs=signs, config=self.cfg, station=None)
            vlines = kwargs.get('vlines', None)
            spec_analysis.display(title=title, vlines=vlines, fit=fit)

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)


