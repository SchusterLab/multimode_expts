import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import experiments.fitting as fitter

class PulseProbeEgGfSpectroscopyProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    """
    Qubit A: E<->G
    Qubit B: g<->f
    Drive applied on qubit B
    """
    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)
        
        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.qubits = self.cfg.expt.qubits
        qA, qB = self.qubits
        
        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type
        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_chs
        self.f_ge_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        gen_chs = []
        
        # declare res dacs
        mask = None
        if self.res_ch_types[0] == 'mux4': # only supports having all resonators be on mux, or none
            assert np.all([ch == 6 for ch in self.res_chs])
            mask = range(4) # indices of mux_freqs, mux_gains list to play
            mux_freqs = [0 if i not in self.qubits else cfg.device.readout.frequency[i] for i in range(4)]
            mux_gains = [0 if i not in self.qubits else cfg.device.readout.gain[i] for i in range(4)]
            self.declare_gen(ch=6, nqz=cfg.hw.soc.dacs.readout.nyquist[0], mixer_freq=cfg.hw.soc.dacs.readout.mixer_freq[0], mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=0)
            gen_chs.append(6)
        else:
            for q in self.qubits:
                mixer_freq = 0
                if self.res_ch_types[q] == 'int4':
                    mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq[q]
                self.declare_gen(ch=self.res_chs[q], nqz=cfg.hw.soc.dacs.readout.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.res_chs[q])

        # declare qubit dacs
        for q in self.qubits:
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # declare swap dac indexed by qA (since the the drive is always applied to qB)
        mixer_freq = 0
        if self.swap_ch_types[qA] == 'int4':
            mixer_freq = cfg.hw.soc.dacs.swap.mixer_freq[qA]
        if self.swap_chs[qA] not in gen_chs: 
            self.declare_gen(ch=self.swap_chs[qA], nqz=cfg.hw.soc.dacs.swap.nyquist[qA], mixer_freq=mixer_freq)
        gen_chs.append(self.swap_chs[qA])

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])

        # drive is applied on qB via the swap channel indexed by qA
        self.r_freq_swap = self.sreg(self.swap_chs[qA], "freq") # get frequency register for swap_ch 
        self.r_freq_swap_update = 4 # register to hold the current sweep frequency

        self.pi_sigmaA = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qA], gen_ch=self.qubit_chs[qA])
        self.pi_ge_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[qB], gen_ch=self.qubit_chs[qB])
        self.pi_ef_sigmaB = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[qB], gen_ch=self.qubit_chs[qB])

        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.swap_chs[qA])
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.swap_chs[qA])

        # send start frequency to r_freq_swap_update
        self.safe_regwi(self.ch_page(self.swap_chs[qA]), self.r_freq_swap_update, self.f_start)

        # add qubit pulses to respective channels
        self.add_gauss(ch=self.qubit_chs[qA], name="pi_qubitA", sigma=self.pi_sigmaA, length=self.pi_sigmaA*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ge_qubitB", sigma=self.pi_ge_sigmaB, length=self.pi_ge_sigmaB*4)
        self.add_gauss(ch=self.qubit_chs[qB], name="pi_ef_qubitB", sigma=self.pi_ef_sigmaB, length=self.pi_ef_sigmaB*4)

        # add readout pulses to respective channels
        if self.res_ch_types[0] == 'mux4':
            self.set_pulse_registers(ch=6, style="const", length=max(self.readout_lengths_dac), mask=mask)
        else:
            for q in self.qubits:
                self.set_pulse_registers(ch=self.res_chs[q], style="const", freq=self.f_res_reg[q], phase=0, gain=cfg.device.readout.gain[q], length=self.readout_lengths_dac[q])

        self.sync_all(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qA, qB = self.qubits
 
        # initialize qubit A to E: expect to end in Eg
        self.setup_and_pulse(ch=self.qubit_chs[qA], style="arb", phase=0, freq=self.f_ge_reg[qA], gain=cfg.device.qubit.pulses.pi_ge.gain[qA], waveform="pi_qubitA")
        self.sync_all(5)

        # apply Eg -> Gf pulse on B: expect to end in Gf
        self.set_pulse_registers(
            ch=self.swap_chs[qA],
            style="const",
            freq=0, # freq set by update
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.swap_chs[qA]))
        self.mathi(self.ch_page(self.swap_chs[qA]), self.r_freq_swap, self.r_freq_swap_update, "+", 0)
        self.pulse(ch=self.swap_chs[qA])
        self.sync_all(5)

        # take qubit B f->e: expect to end in Ge (or Eg if incomplete Eg-Gf)
        self.setup_and_pulse(ch=self.qubit_chs[qB], style="arb", freq=self.f_ef_reg[qB], phase=0, gain=cfg.device.qubit.pulses.pi_ef.gain[qB], waveform="pi_ef_qubitB")

        self.sync_all(5)
        measure_chs = self.res_chs
        if self.res_ch_types[0] == 'mux4': measure_chs = self.res_chs[0]
        self.measure(
            pulse_ch=measure_chs, 
            adcs=self.adc_chs,
            adc_trig_offset=cfg.device.readout.trig_offset[0],
            wait=True,
            syncdelay=self.us2cycles(max([cfg.device.readout.relax_delay[q] for q in self.qubits])))

    def update(self):
        qA, qB = self.qubits
        self.mathi(self.ch_page(self.swap_chs[qA]), self.r_freq_swap_update, self.r_freq_swap_update, '+', self.f_step) # update frequency list index
        
# ====================================================== #

class PulseProbeEgGfSpectroscopyExperiment(Experiment):
    """
    Pulse Probe Eg-Gf Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
        qubits: qubit 0 goes E->G, apply drive on qubit 1 (g->f)
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEgGfSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits

        # expand entries in config that are length 1 to fill all qubits
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

        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]

        qspec_EgGf = PulseProbeEgGfSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        xpts, avgi, avgq = qspec_EgGf.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
        
        data=dict(
            xpts=xpts,
            avgi=(avgi[adcA_ch][0], avgi[adcB_ch][0]),
            avgq=(avgq[adcA_ch][0], avgq[adcB_ch][0]),
            amps=(np.abs(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]),
                  np.abs(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0])),
            phases=(np.angle(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]),
                    np.angle(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0])),
        )

        self.data=data
        return data

    def analyze(self, data=None, fit=True, signs=[[1, 1], [1, 1]], **kwargs):
        # signs of fit: [iA, qA], [iB, qB]
        if data is None: data=self.data
        self.signs = signs
        if fit:
            data['fitA_avgi'], data['fitA_err_avgi'] = fitter.fitlor(data["xpts"], signs[0][0]*data['avgi'][0])
            data['fitA_avgq'], data['fitA_err_avgq'] = fitter.fitlor(data["xpts"], signs[0][1]*data['avgq'][0])
            data['fitB_avgi'], data['fitB_err_avgi'] = fitter.fitlor(data["xpts"], signs[1][0]*data['avgi'][1])
            data['fitB_avgq'], data['fitB_err_avgq'] = fitter.fitlor(data["xpts"], signs[1][1]*data['avgq'][1])
        return data

    def display(self, data=None, fit=True, signs=None, **kwargs):
        # signs of fit: [iA, qA], [iB, qB]
        if data is None: data=self.data 
        if signs is None: signs = self.signs
        plt.figure(figsize=(14,8))
        plt.suptitle(f"Pulse Probe Eg-Gf Spectroscopy")

        test_f = 2410
        
        plt.subplot(221, title=f'Qubit A ({self.cfg.expt.qubits[0]})', ylabel="I [adc level]")
        plt.plot(data["xpts"][0:-1], data["avgi"][0][0:-1],'o-')
        # plt.axvline(test_f, color='r')
        if fit:
            plt.plot(data["xpts"], signs[0][0]*fitter.lorfunc(data["xpts"], *data["fitA_avgi"]))
            print(f'Found peak in avgi data (qubit A) at [MHz] {data["fitA_avgi"][2]}, HWHM {data["fitA_avgi"][3]}')
        plt.subplot(223, xlabel="Pulse Frequency [MHz]", ylabel="Q [adc levels]")
        plt.plot(data["xpts"][0:-1], data["avgq"][0][0:-1],'o-')
        # plt.axvline(test_f, color='r')
        if fit:
            plt.plot(data["xpts"], signs[0][1]*fitter.lorfunc(data["xpts"], *data["fitA_avgq"]))
            print(f'Found peak in avgq data (qubit A) at [MHz] {data["fitA_avgq"][2]}, HWHM {data["fitA_avgq"][3]}')


        plt.subplot(222, title=f'Qubit B ({self.cfg.expt.qubits[1]})')
        plt.plot(data["xpts"][0:-1], data["avgi"][1][0:-1],'o-')
        if fit:
            plt.plot(data["xpts"], signs[1][0]*fitter.lorfunc(data["xpts"], *data["fitB_avgi"]))
            print(f'Found peak in avgi data (qubit B) at [MHz] {data["fitB_avgi"][2]}, HWHM {data["fitB_avgi"][3]}')
        # plt.axvline(test_f, color='r')
        plt.subplot(224, xlabel="Pulse Frequency [MHz]")
        plt.plot(data["xpts"][0:-1], data["avgq"][1][0:-1],'o-')
        # plt.axvline(test_f, color='r')
        if fit:
            plt.plot(data["xpts"], signs[1][1]*fitter.lorfunc(data["xpts"], *data["fitB_avgq"]))
            print(f'Found peak in avgq data (qubit B) at [MHz] {data["fitB_avgq"][2]}, HWHM {data["fitB_avgq"][3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #
                      
class PulseProbeEgGfSweepSpectroscopyExperiment(Experiment):
    """
    Pulse probe EgGf length sweep spectroscopy experiment
    Experimental Config
        expt = dict(
        start_f: start probe frequency [MHz]
        step_f: step probe frequency
        expts_f: number experiments freq stepping from start
        start_len: start const pulse len [us]
        step_len
        expts_len
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        gain: ef const pulse gain [dac units]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEgGfSweep', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        qA, qB = self.cfg.expt.qubits

        # expand entries in config that are length 1 to fill all qubits
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

        adcA_ch = self.cfg.hw.soc.adcs.readout.ch[qA]
        adcB_ch = self.cfg.hw.soc.adcs.readout.ch[qB]

        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"]*np.arange(self.cfg.expt["expts_len"])

        data=dict(
            fpts=fpts,
            lenpts=lenpts,
            avgi=([], []),
            avgq=([], []),
            amps=([], []),
            phases=([], []),
        )

        for length in tqdm(lenpts):
            self.cfg.expt.length = length
            self.cfg.expt.start = self.cfg.expt.start_f
            self.cfg.expt.step = self.cfg.expt.step_f
            self.cfg.expt.expts = self.cfg.expt.expts_f
            qspec_EgGf = PulseProbeEgGfSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            xpts, avgi, avgq = qspec_EgGf.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)        
        
            data['avgi'][0].append(avgi[adcA_ch][0])
            data['avgq'][0].append(avgq[adcA_ch][0])
            data['amps'][0].append(np.abs(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]))
            data['phases'][0].append(np.angle(avgi[adcA_ch][0]+1j*avgq[adcA_ch][0]))

            data['avgi'][1].append(avgi[adcB_ch][0])
            data['avgq'][1].append(avgq[adcB_ch][0])
            data['amps'][1].append(np.abs(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0]))
            data['phases'][1].append(np.angle(avgi[adcB_ch][0]+1j*avgq[adcB_ch][0]))

        for k, a in data.items():
            data[k] = np.array(a)

        self.data=data
        return data

    def analyze(self, data=None, fit=True, signs=[[1, 1], [1, 1]], **kwargs):
        if data is None:
            data=self.data
        return data

    def display(self, data=None, fit=True, signs=None, **kwargs):
        if data is None:
            data=self.data 

        x_sweep = data['fpts']
        y_sweep = data['lenpts'] 

        # Qubit 0
        avgi = data['avgi'][0]
        avgq = data['avgq'][0]

        plt.figure(figsize=(10,12))
        plt.subplot(211, title=f"Pulse Probe EgGf Spectroscopy Length Sweep Qubit A ({self.cfg.expt.qubits[0]})", ylabel="Pulse Length [us]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='I [adc level]')

        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Pulse Length [us]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Q [radians]')
        
        plt.show()    

        # Qubit 1
        avgi = data['avgi'][1]
        avgq = data['avgq'][1]

        plt.figure(figsize=(10,12))
        plt.subplot(211, title=f"Pulse Probe EgGf Spectroscopy Length Sweep Qubit B ({self.cfg.expt.qubits[1]})", ylabel="Pulse Length [us]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='I [adc level]')

        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Pulse Length [us]")
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Q')
        
        plt.show()    

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname