import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter
from MM_base import * 
class ParityDelayProgram(MMAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize() # should take care of all the MM base (channel names, pulse names, readout )
        cfg = AttrDict(self.cfg)

        self.sync_all(200)

    

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0] 

        # phase reset
        self.reset_and_sync()

        # active reset 
        if cfg.expt.active_reset:
            self.active_reset(man_reset=self.cfg.expt.man_reset, storage_reset=self.cfg.expt.storage_reset)

        self.sync_all(self.us2cycles(0.2))

        # if cfg.expt.prepulse:
        #     creator = self.get_prepulse_creator(cfg.expt.pre_gate_sweep_pulse)
        #     self.custom_pulse(cfg, creator.pulse.tolist(), prefix = '')
        #     # self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')


        

        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[qTest], phase=self.deg2reg(0), gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")
        
        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="const", freq=self.f_ge_reg[qTest], phase=self.deg2reg(0), gain=0, length=self.us2cycles(cfg.expt.length_placeholder, gen_ch=self.qubit_chs[qTest]))


        self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[qTest], phase=self.deg2reg(180, self.qubit_chs[qTest]), gain=self.hpi_ge_gain, waveform="hpi_qubit_ge")
        # self.wait_all(self.us2cycles(0.01)) # wait for the time stored in the wait variable register
        self.measure_wrapper()


class ParityDelayExperiment(Experiment):
    """
    ParityDelay Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubits[0]
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


        lengths = self.cfg.expt["start"] + self.cfg.expt["step"] * np.arange(self.cfg.expt["expts"])
        read_num = 1
        if self.cfg.expt.active_reset: read_num = 4

        data={"xpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[], "idata":[], "qdata":[]}

        for length in tqdm(lengths, disable=not progress):
            self.cfg.expt.length_placeholder = float(length)
            lengthrabi = ParityDelayProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = lengthrabi
            avgi, avgq = lengthrabi.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug,
                                            readouts_per_experiment=read_num )      
            idata, qdata = lengthrabi.collect_shots()  
            data["idata"].append(idata)
            data["qdata"].append(qdata)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase
            data["xpts"].append(length)
            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)                             

        # t1 = ParityDelayProaram(soccfg=self.soccfg, cfg=self.cfg)
        # x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        # avgi = avgi[0][0]
        # avgq = avgq[0][0]
        # amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        # phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        # data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        # if self.cfg.expt.normalize:
        #     from experiments.single_qubit.normalize import normalize_calib
        #     g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)
            
        #     data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
        #     data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
        #     data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]
        
        
        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data
            
        # fitparams=[y-offset, amp, x-offset, decay rate]
        # Remove the last point from fit in case weird edge measurements
        data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
        data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
        data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        # plt.figure(figsize=(12, 8))
        # plt.subplot(111,title="$T_1$", xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
        # plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
        # if fit:
        #     p = data['fit_amps']
        #     pCov = data['fit_err_amps']
        #     captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        #     plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_amps"]), label=captionStr)
        #     plt.legend()
        #     print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

        plt.figure(figsize=(10,10))
        plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
        if fit:
            p = data['fit_avgi']
            pCov = data['fit_err_avgi']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
        plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
        plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
        if fit:
            p = data['fit_avgq']
            pCov = data['fit_err_avgq']
            captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
            plt.legend()
            print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname

# class ParityDelayProgram(MMRAveragerProgram):
#     def __init__(self, soccfg, cfg):
#         self.cfg = AttrDict(cfg)
#         self.cfg.update(self.cfg.expt)

#         # copy over parameters for the acquire method
#         self.cfg.reps = cfg.expt.reps
#         self.cfg.rounds = cfg.expt.rounds
        
#         super().__init__(soccfg, self.cfg)

#     def initialize(self):
#         cfg = AttrDict(self.cfg)
#         self.cfg.update(cfg.expt)
        
#         self.adc_ch = cfg.hw.soc.adcs.readout.ch
#         self.res_ch = cfg.hw.soc.dacs.readout.ch
#         self.res_ch_type = cfg.hw.soc.dacs.readout.type
#         self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
#         self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
#         self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
#         self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
#         self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
#         self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
#         self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
#         self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
#         self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
#         self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
#         self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
#         self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

#         self.man_chs = cfg.hw.soc.dacs.manipulate_in.ch
#         self.man_ch_types = cfg.hw.soc.dacs.manipulate_in.type

#         self.q_rp = self.ch_page(self.qubit_ch) # get register page for qubit_ch
#         self.r_wait = 3
#         self.safe_regwi(self.q_rp, self.r_wait, self.us2cycles(cfg.expt.start))
        
#         self.f_ge = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
#         self.f_ef = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
#         self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)
#         self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
#         self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
#         self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

#         # declare res dacs
#         mask = None
#         mixer_freq = 0 # MHz
#         mux_freqs = None # MHz
#         mux_gains = None
#         ro_ch = self.adc_ch
#         if self.res_ch_type == 'int4':
#             mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
#         elif self.res_ch_type == 'mux4':
#             assert self.res_ch == 6
#             mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
#             mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
#             mux_freqs = [0]*4
#             mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
#             mux_gains = [0]*4
#             mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
#         self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

#         #f0g1 sideband
#         self.f0g1 = self.freq2reg(cfg.device.QM.pulses.f0g1.freq[cfg.expt.f0g1_cavity-1], gen_ch=self.qubit_ch)
#         self.f0g1_length = self.us2cycles(cfg.device.QM.pulses.f0g1.length[cfg.expt.f0g1_cavity-1], gen_ch=self.qubit_ch)
#         self.pif0g1_gain = cfg.device.QM.pulses.f0g1.gain[cfg.expt.f0g1_cavity-1]
        
#         # declare qubit dacs
#         mixer_freq = 0
#         if self.qubit_ch_type == 'int4':
#             mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
#         self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

#         # declare adcs
#         self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

#         self.pi_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch)
#         self.hpi_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma, gen_ch=self.qubit_ch)
#         self.hpi_sigma_fast = self.us2cycles(cfg.device.qubit.pulses.hpi_ge_fast.sigma, gen_ch=self.qubit_ch)
#         self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)

#         self.pi_gain = cfg.device.qubit.pulses.pi_ge.gain
#         self.pief_gain = cfg.device.qubit.pulses.pi_ef.gain

#         # add qubit and readout pulses to respective channels
#         if self.cfg.device.qubit.pulses.pi_ge.type.lower() == 'gauss':
            
#             self.add_gauss(ch=self.qubit_ch, name="pi_qubit", sigma=self.pi_sigma, length=self.pi_sigma*4)
#             self.add_gauss(ch=self.qubit_ch, name="hpi_qubit", sigma=self.hpi_sigma, length=self.hpi_sigma*4)
#             self.add_gauss(ch=self.qubit_ch, name="hpi_qubit_fast", sigma=self.hpi_sigma_fast, length=self.hpi_sigma_fast*4)
#             self.add_gauss(ch=self.qubit_ch, name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
#             self.add_gauss(ch=self.f0g1_ch, name="f0g1",
#                        sigma=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma), length=self.us2cycles(self.cfg.device.QM.pulses.f0g1.sigma)*4)
#             #self.set_pulse_registers(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")
#         else:
#             self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_ge, phase=0, gain=cfg.expt.start, length=self.pi_sigma)


#         # if self.res_ch_type == 'mux4':
#         #     self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
#         self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

#         self.sync_all(200)

#     # def reset_and_sync(self):
#     #     # Phase reset all channels except readout DACs 

#     #     # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
#     #     # self.setup_and_pulse(ch=self.qubit_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[0]), phase=0, gain=5, length=10, phrst=1)
#     #     # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
#     #     # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
#     #     # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
#     #     # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
#     #     # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)


#     #     #initialize the phase to be 0
#     #     self.set_pulse_registers(ch=self.qubit_ch, freq=self.f_ge,
#     #                              phase=0, gain=0, length=10, style="const", phrst=1)
#     #     self.pulse(ch=self.qubit_ch)
#     #     self.set_pulse_registers(ch=self.man_chs, freq=self.f_ge,
#     #                              phase=0, gain=0, length=10, style="const", phrst=1)
#     #     self.pulse(ch=self.man_chs)
#     #     # self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cavity,
#     #     #                          phase=0, gain=0, length=10, style="const", phrst=1)
#     #     # self.pulse(ch=self.storage_ch[0])
#     #     self.set_pulse_registers(ch=self.flux_low_ch, freq=self.f_ge,
#     #                              phase=0, gain=0, length=10, style="const", phrst=1)
#     #     self.pulse(ch=self.flux_low_ch)
#     #     self.set_pulse_registers(ch=self.flux_high_ch, freq=self.f_ge,
#     #                              phase=0, gain=0, length=10, style="const", phrst=1)
#     #     self.pulse(ch=self.flux_high_ch)
#     #     self.set_pulse_registers(ch=self.f0g1_ch, freq=self.f_ge,
#     #                              phase=0, gain=0, length=10, style="const", phrst=1)
#     #     self.pulse(ch=self.f0g1_ch)

#     #     self.sync_all(10)

#     def body(self):
#         cfg=AttrDict(self.cfg)

#         # phase reset
#         self.reset_and_sync()

#         # active reset 
#         if cfg.expt.active_reset:
#             self.active_reset(man_reset=self.cfg.expt.man_reset, storage_reset=self.cfg.expt.storage_reset)

#         if cfg.expt.prepulse:
#             self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix='pre')

#         if cfg.expt.f0g1_cavity > 0:
#             self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=0, gain=self.pi_gain, waveform="pi_qubit")
#             self.sync_all() # align channels
#             self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef, phase=0, gain=self.pief_gain, waveform="pief_qubit")
#             self.sync_all() # align channels
#             self.setup_and_pulse(
#                     ch=self.f0g1_ch,
#                     style="flat_top",
#                     freq=self.f0g1,
#                     length=self.f0g1_length,
#                     phase=0,
#                     gain=self.pif0g1_gain, 
#                     waveform="f0g1")
#             self.sync_all() # align channels

#         self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=self.deg2reg(0), gain=cfg.device.qubit.pulses.hpi_ge_fast.gain, waveform="hpi_qubit_fast")
#         self.sync_all() # align channels
#         self.sync(self.q_rp, self.r_wait) # wait for the time stored in the wait variable register
#         self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge, phase=self.deg2reg(180), gain=cfg.device.qubit.pulses.hpi_ge_fast.gain, waveform="hpi_qubit_fast")
#         self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
#         self.measure(pulse_ch=self.res_ch, 
#              adcs=[self.adc_ch],
#              adc_trig_offset=cfg.device.readout.trig_offset,
#              wait=True,
#              syncdelay=self.us2cycles(cfg.device.readout.relax_delay))

#     def update(self):
#         self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.us2cycles(self.cfg.expt.step)) # update wait time
#         self.wait_all(self.us2cycles(0.01)) # wait 10ns


# class ParityDelayExperiment(Experiment):
#     """
#     ParityDelay Experiment
#     Experimental Config:
#     expt = dict(
#         start: wait time sweep start [us]
#         step: wait time sweep step
#         expts: number steps in sweep
#         reps: number averages per experiment
#         rounds: number rounds to repeat experiment sweep
#     )
#     """

#     def __init__(self, soccfg=None, path='', prefix='T1', config_file=None, progress=None):
#         super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

#     def acquire(self, progress=False, debug=False):
#         q_ind = self.cfg.expt.qubit
#         for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
#             for key, value in subcfg.items() :
#                 if isinstance(value, list):
#                     subcfg.update({key: value[q_ind]})
#                 elif isinstance(value, dict):
#                     for key2, value2 in value.items():
#                         for key3, value3 in value2.items():
#                             if isinstance(value3, list):
#                                 value2.update({key3: value3[q_ind]})                                

#         t1 = ParityDelayProgram(soccfg=self.soccfg, cfg=self.cfg)
#         x_pts, avgi, avgq = t1.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

#         avgi = avgi[0][0]
#         avgq = avgq[0][0]
#         amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
#         phases = np.angle(avgi+1j*avgq) # Calculating the phase        

#         data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
#         if self.cfg.expt.normalize:
#             from experiments.single_qubit.normalize import normalize_calib
#             g_data, e_data, f_data = normalize_calib(self.soccfg, self.path, self.config_file)
            
#             data['g_data'] = [g_data['avgi'], g_data['avgq'], g_data['amps'], g_data['phases']]
#             data['e_data'] = [e_data['avgi'], e_data['avgq'], e_data['amps'], e_data['phases']]
#             data['f_data'] = [f_data['avgi'], f_data['avgq'], f_data['amps'], f_data['phases']]
        
        
#         self.data=data
#         return data

#     def analyze(self, data=None, **kwargs):
#         if data is None:
#             data=self.data
            
#         # fitparams=[y-offset, amp, x-offset, decay rate]
#         # Remove the last point from fit in case weird edge measurements
#         data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
#         data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
#         data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
#         return data

#     def display(self, data=None, fit=True, **kwargs):
#         if data is None:
#             data=self.data 
        
#         # plt.figure(figsize=(12, 8))
#         # plt.subplot(111,title="$T_1$", xlabel="Wait Time [us]", ylabel="Amplitude [ADC level]")
#         # plt.plot(data["xpts"][:-1], data["amps"][:-1],'o-')
#         # if fit:
#         #     p = data['fit_amps']
#         #     pCov = data['fit_err_amps']
#         #     captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
#         #     plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_amps"]), label=captionStr)
#         #     plt.legend()
#         #     print(f'Fit T1 amps [us]: {data["fit_amps"][3]}')

#         plt.figure(figsize=(10,10))
#         plt.subplot(211, title="$T_1$", ylabel="I [ADC units]")
#         plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
#         if fit:
#             p = data['fit_avgi']
#             pCov = data['fit_err_avgi']
#             captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
#             plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
#             plt.legend()
#             print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
#         plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
#         plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
#         if fit:
#             p = data['fit_avgq']
#             pCov = data['fit_err_avgq']
#             captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
#             plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
#             plt.legend()
#             print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

#         plt.show()
        
#     def save_data(self, data=None):
#         print(f'Saving {self.fname}')
#         super().save_data(data=data)
#         return self.fname