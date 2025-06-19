import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
import time

import experiments.fitting as fitter
from MM_base import MMRAveragerProgram

class PulseProbeSpectroscopyProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = 0

       
        ## Initialize the register and page
        self.q_rp=self.ch_page(self.qubit_chs[qTest]) # get register page for qubit_ch
        self.r_freq=self.sreg(gen_ch=self.qubit_chs[qTest], name = "freq") # get frequency register for qubit_ch  
        print(f"qubit register page: {self.q_rp}, frequency register: {self.r_freq}")
        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_chs[qTest]) # get start/step frequencies
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_chs[qTest])

        ## qubit 
        self.ramp = self.us2cycles(cfg.expt.sigma, gen_ch=self.qubit_chs[qTest]) # get length of qubit pulse
        self.add_gauss(ch=self.qubit_chs[qTest], name="ramp", sigma=self.ramp, length=self.ramp*4)

        self.r_freq2 = 4 # dummy register for freq  (since multiple qubit pulses)
        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start) # set dummygain register to start value

       
        self.synci(200) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = 0

        if cfg.expt.prepulse:
            if cfg.expt.gate_based: 
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix = 'pre_')
            else: 
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_')
       
        
        # add qubit and readout pulses to respective channels
        # self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_start, phase=0, gain=cfg.expt.gain, length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch))
        self.set_pulse_registers(ch=self.qubit_chs[qTest], style="flat_top", 
                                 freq=self.f_start, # dummy 
                                  phase=0, gain=cfg.expt.gain, 
                            length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_chs[qTest]), waveform="ramp")
        self.mathi(self.q_rp,  self.r_freq, self.r_freq2, '+', 0 ) # update gain register
        
        
        self.pulse(ch=self.qubit_chs[qTest]) # play probe pulse
        
        self.measure_wrapper()
    
    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index
        # self.mathi(self.qubit_chs[qTest],  self.r_freq, self.r_freq2, '+', 0 ) # update gain register
 
# ====================================================== #

class PulseProbeSpectroscopyExperiment(Experiment):
    """
    PulseProbe Spectroscopy Experiment
    Experimental Config:
        start: Qubit frequency [MHz]
        step
        expts: Number of experiments stepping from start
        reps: Number of averages per point
        rounds: Number of start to finish sweeps to average over
        length: Qubit probe constant pulse length [us]
        gain: Qubit pulse gain [DAC units]
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment( num_qubits_sample)                          

        qspec = PulseProbeSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        self.prog = qspec
        xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq)
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        
        
        data={'xpts':xpts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
        self.data=data
        return data

    def analyze(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data
        if fit:
            xdata = data['xpts'][1:-1]
            data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
            data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
            data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])
        return data

    def display(self, data=None, fit=True, signs=[1,1,1], **kwargs):
        if data is None:
            data=self.data 

        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts'][1:-1]
        else: 
            xpts = data['xpts'][1:-1]

        plt.figure(figsize=(9, 11))
        plt.subplot(311, title=f"Qubit {self.cfg.expt.qubit} Spectroscopy (Gain {self.cfg.expt.gain})", ylabel="Amplitude [ADC units]")
        plt.plot(xpts, data["amps"][1:-1],'o-')
        if fit:
            plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
            print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

        plt.subplot(312, ylabel="I [ADC units]")
        plt.plot(xpts, data["avgi"][1:-1],'o-')
        if fit:
            plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
            print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
        plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
        plt.plot(xpts, data["avgq"][1:-1],'o-')
        # plt.axvline(3476, c='k', ls='--')
        # plt.axvline(3376+50, c='k', ls='--')
        # plt.axvline(3376, c='k', ls='--')
        if fit:
            plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
            # plt.axvline(3593.2, c='k', ls='--')
            print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)

# ====================================================== #

from experiments.single_qubit.resonator_spectroscopy import ResonatorSpectroscopyExperiment
class PulseProbeVoltSweepSpectroscopyExperiment(Experiment):
    """
    PulseProbe Spectroscopy Experiment Sweep Voltage
    Experimental Config:
        start_qf: start qubit frequency (MHz), 
        step_qf: frequency step (MHz), 
        expts_qf: number of experiments in frequency,
        length: Qubit probe constant pulse length [us]
        gain: Qubit pulse gain [DAC units]
        dc_ch: channel on dc_instr to sweep voltage

        start_rf: start resonator frequency (MHz), 
        step_rf: frequency step (MHz), 
        expts_rf: number of experiments in frequency,

        start_volt: start volt, 
        step_volt: voltage step, 
        expts_volt: number of experiments in voltage sweep,

        reps_q: Number of averages per point for pulse probe
        rounds_q: Number of start to finish freq sweeps to average over

        reps_r: Number of averages per point for resonator spectroscopy
    """

    def __init__(self, soccfg=None, path='', dc_instr=None, prefix='PulseProbeVoltSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(path=path, soccfg=soccfg, prefix=prefix, config_file=config_file, progress=progress)
        self.dc_instr = dc_instr
        self.path = path
        self.config_file = config_file

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items() :
                if isinstance(value, list):
                    subcfg.update({key: value[q_ind]})
                elif isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if isinstance(value3, list):
                                value2.update({key3: value3[q_ind]})                                
        
        voltpts = self.cfg.expt["start_volt"] + self.cfg.expt["step_volt"]*np.arange(self.cfg.expt["expts_volt"])
        data=dict(
            xpts=[],
            voltpts=[],
            avgi=[],
            avgq=[],
            amps=[],
            phases=[],
            rspec_avgi=[],
            rspec_avgq=[],
            rspec_amps=[],
            rspec_phases=[],
            rspec_fits=[]
        )

        self.cfg.expt.start = self.cfg.expt.start_qf
        self.cfg.expt.step = self.cfg.expt.step_qf
        self.cfg.expt.expts = self.cfg.expt.expts_qf
        self.cfg.expt.reps = self.cfg.expt.reps_q
        self.cfg.expt.rounds = self.cfg.expt.rounds_q

        for volt in tqdm(voltpts):
            self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=volt)
            time.sleep(0.5)

            # Get readout frequency
            rspec = ResonatorSpectroscopyExperiment(
                soccfg=self.soccfg,
                path=self.path,
                config_file=self.config_file,
            )
            rspec.cfg.expt = dict(
                start=self.cfg.expt.start_rf,
                step=self.cfg.expt.step_rf,
                expts=self.cfg.expt.expts_rf,
                reps=self.cfg.expt.reps_r,
                pi_pulse=False,
                qubit=self.cfg.expt.qubit,
            )
            rspec.go(analyze=False, display=False, progress=False, save=False)
            rspec.analyze(fit=True, verbose=False)
            readout_freq = rspec.data['fit'][0]

            self.cfg.device.readout.frequency = readout_freq
            print(f'readout at {readout_freq} at voltage {volt}')

            qspec = PulseProbeSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            xpts, avgi, avgq = qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)        
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amps = np.abs(avgi+1j*avgq)
            phases = np.angle(avgi+1j*avgq) # Calculating the phase        

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)

            data["rspec_avgi"].append(rspec.data['avgi'])
            data["rspec_avgq"].append(rspec.data['avgq'])
            data["rspec_amps"].append(rspec.data['amps'])
            data["rspec_phases"].append(rspec.data['phases'])
            data["rspec_fits"].append(rspec.data['fit'])

            time.sleep(0.5)
        # self.dc_instr.initialize()
        self.dc_instr.set_voltage(channel=self.cfg.expt.dc_ch, voltage=0)

        data["rspec_xpts"] = rspec.data['xpts']
        data['xpts'] = xpts
        data['voltpts'] = voltpts
        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data=self.data

        # data.update(
        #     dict(
        #     rspec_avgi=[],
        #     rspec_avgq=[],
        #     rspec_amps=[],
        #     rspec_phases=[],
        #     rspec_fits=[]
        #     )
        # )
        # data["rspec_xpts"] = data['rspec_data'][0]['xpts']
        # for rspec_data in data['rspec_data']:
        #     data["rspec_avgi"].append(rspec_data['avgi'])
        #     data["rspec_avgq"].append(rspec_data['avgq'])
        #     data["rspec_amps"].append(rspec_data['amps'])
        #     data["rspec_phases"].append(rspec_data['phases'])
        #     data["rspec_fits"].append(rspec_data['fit'])

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        freqs_q = data['xpts']
        freqs_r = data['rspec_xpts']
        x_sweep = 1e3*data['voltpts']
        amps = data['amps']
        # for amps_volt in amps:
        #     amps_volt -= np.average(amps_volt)
        
        # THIS IS THE FIXED EXTENT LIMITS FOR 2D PLOTS
        plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,2])
        plt.subplot(gs[0], title="Pulse Probe Voltage Sweep", ylabel="Resonator Frequency [MHz]")
        y_sweep = freqs_r
        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(data['rspec_amps']), 0), cmap='viridis')
        rfreqs = [data['rspec_fits'][i][0] for i in range(len(data['voltpts']))]
        plt.scatter(x_sweep, rfreqs, marker='o', color='r')
        if 'add_data' in kwargs:
            for add_data in kwargs['add_data']:
                plt.pcolormesh(
                    1e3*add_data['voltpts'], add_data['rspec_xpts'], np.flip(np.rot90(add_data['rspec_amps']), 0), cmap='viridis')
                rfreqs = [add_data['rspec_fits'][i][0] for i in range(len(add_data['voltpts']))]
                plt.scatter(1e3*add_data['voltpts'], rfreqs, marker='o', color='r')
        plt.xlim(min(x_sweep), max(x_sweep))
        # plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')

        plt.subplot(gs[1], xlabel=f"DC Voltage (DAC ch {self.cfg.expt.dc_ch}) [mV]", ylabel="Qubit Frequency [MHz]")
        y_sweep = freqs_q
        plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(amps), 0), cmap='viridis')
        plt.xlim(min(x_sweep), max(x_sweep))
        if 'add_data' in kwargs:
            for add_data in kwargs['add_data']:
                y_sweep = add_data['xpts']
                x_sweep = 1e3*add_data['voltpts']
                amps = add_data['amps']
                # for amps_volt in amps:
                #     amps_volt -= np.average(amps_volt)
                plt.pcolormesh(x_sweep, y_sweep, np.flip(np.rot90(amps), 0), cmap='viridis')
        plt.axvline(2.55)
        # plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps [ADC level]')
        
        # if fit: pass
        plt.show()
        
    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname