import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, dsfit, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter


class DemolutionMeasurementProgram(AveragerProgram):
    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        
        qTest = self.qubits[0]

        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type

        self.man_chs = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_types = cfg.hw.soc.dacs.manipulate_in.type

        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        self.f_ge_init_reg = self.f_ge_reg[qTest]
        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.gain_hge_init = self.cfg.device.qubit.pulses.hpi_ge.gain[qTest]
        self.gain_ge_middle = self.cfg.device.qubit.pulses.pi_ge_middle.gain[qTest]

        gen_chs = []
        
        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_types[qTest] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[qTest]

        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest], mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest], freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # define the displace sigma for calibration
        # self.displace_sigma = self.us2cycles(cfg.expt.displace_sigma)
        self.hpisigma_ge = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value
        self.pisigma_ge_middle = self.us2cycles(cfg.device.qubit.pulses.pi_ge_middle.sigma[qTest], gen_ch=self.qubit_chs[qTest]) # default pi_ge value

        if cfg.expt.cavity_name == 0:
            self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[0], gen_ch=self.man_chs[0])
        else:
            self.f_cavity = self.freq2reg(cfg.device.manipulate.f_ge[1], gen_ch=self.man_chs[0])

         # add readout pulses to respective channels
        # if self.res_ch_types[qTest] == 'mux4':
        #     self.set_pulse_registers(ch=self.res_chs[qTest], style="const", length=self.readout_lengths_dac[qTest], mask=mask)
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])
        
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit", sigma=self.hpisigma_ge, length=self.hpisigma_ge*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_middle_qubit", sigma=self.pisigma_ge_middle, length=self.pisigma_ge_middle*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="pi_middle_qubit", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        # self.add_gauss(ch=self.man_chs[0], name="displace", sigma=self.displace_sigma, length=self.displace_sigma*4)

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]
        self.wait_time_r = self.cfg.expt.time_placeholder

        # prepare g+e, then do two repeated readout, pi pulse the qubit as specified
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_hge_init, waveform="hpi_qubit")
        # self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit")
            
        # align channels and measure
        self.sync_all(self.us2cycles(0.2))
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(self.wait_time_r)
        )
        self.sync_all(self.us2cycles(0.05))
        if cfg.expt.pi_pulse:
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_init_reg, phase=0, gain=self.gain_ge_middle, waveform="pi_middle_qubit")
            self.sync_all(self.us2cycles(0.05))
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )


    def collect_shots(self):
        # collect shots for 2 adcs (0 and 1 indexed) and I and Q channels
        cfg = self.cfg
        # print(self.di_buf[0])
        shots_i0 = self.di_buf[0].reshape((2, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]
        # print(shots_i0)
        shots_q0 = self.dq_buf[0].reshape((2, self.cfg["reps"]),order='F') / self.readout_lengths_adc[0]

        return shots_i0, shots_q0

    # def update(self):
    #     pass
    #     # self.mathi(self.q_rp, self.dummy1, self.dummy1, '+', 0)  # update frequency list index


class DemolutionMeasurementExperiment(Experiment):
    """Demolution measurement
       Experimental Config
        expt = {"reps": 10, "rounds": 200, "parity_number": 5, "storage_ge":True}
         }
    """

    def __init__(self, soccfg=None, path='', prefix='DemolutionMeasurement', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not(isinstance(value3, list)):
                                value2.update({key3: [value3]*num_qubits_sample})                                
                elif not(isinstance(value, list)):
                    subcfg.update({key: [value]*num_qubits_sample})

        qTest = self.cfg.expt.qubits[0]

        waiting_list = self.cfg.expt["time_start"] + self.cfg.expt["time_step"] * np.arange(self.cfg.expt["time_expts"])

        data = {"xpts": [], "avgi": [], "avgq": [], "i0":[], "i1":[], "q0":[], "q1":[]}
        self.cfg.expt.readout_no_placeholder = 2
        for time_value in tqdm(waiting_list, disable=not progress):
            self.cfg.expt.time_placeholder = time_value
            rspec = DemolutionMeasurementProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = rspec
            avgi, avgq = rspec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False, debug=False,\
                                    readouts_per_experiment=2,save_experiments=np.arange(0,2,1))


            data["xpts"].append(time_value)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)

            i0, q0 = rspec.collect_shots()
            data["i0"].append(i0)
            data["q0"].append(q0)

        for k, a in data.items():
            data[k]=np.array(a)

        self.data = data

        return data

    def analyze(self, data=None, **kwargs):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

        return data

    def display(self, data=None, **kwargs):
        print("Not working")

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
