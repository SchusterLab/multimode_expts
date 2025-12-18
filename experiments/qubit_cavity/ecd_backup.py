import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from qick import *

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
import time

import fitting.fitting as fitter

class ECDProgram(RAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg=AttrDict(self.cfg)
        self.cfg.update(cfg.expt)

        self.adc_ch = cfg.hw.soc.adcs.readout.ch
        self.res_ch = cfg.hw.soc.dacs.readout.ch
        self.res_ch_type = cfg.hw.soc.dacs.readout.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
       # self.storage_ch = cfg.hw.soc.dacs.storage_in.ch

        self.q_rp=self.ch_page(self.qubit_ch) # get register page for qubit_ch
        self.r_freq=self.sreg(self.qubit_ch, "freq") # get frequency register for qubit_ch    
        self.f_res_reg = self.freq2reg(cfg.device.readout.frequency, gen_ch=self.res_ch, ro_ch=self.adc_ch)

        # Dummy Register to store frequency **** USING STORAGE as dummy; don't do ECD on storage!!! *****
        # self.d_rp=self.ch_page(self.storage_ch) # get register page for dummy_ch
        # self.d_freq=self.sreg(self.storage_ch, "freq") # get frequency register for dummy_ch  
        self.d_reg = 10 #dummy register to store frequency

        self.f_ge_reg = self.freq2reg(cfg.device.qubit.f_ge, gen_ch=self.qubit_ch)
        self.f_ef_reg = self.freq2reg(cfg.device.qubit.f_ef, gen_ch=self.qubit_ch)
        
        self.readout_length_dac = self.us2cycles(cfg.device.readout.readout_length, gen_ch=self.res_ch)
        self.readout_length_adc = self.us2cycles(cfg.device.readout.readout_length, ro_ch=self.adc_ch)
        self.readout_length_adc += 1 # ensure the rounding of the clock ticks calculation doesn't mess up the buffer

        # declare res dacs
        mask = None
        mixer_freq = 0 # MHz
        mux_freqs = None # MHz
        mux_gains = None
        ro_ch = None
        if self.res_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
        elif self.res_ch_type == 'mux4':
            assert self.res_ch == 6
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mixer_freq = cfg.hw.soc.dacs.readout.mixer_freq
            mux_freqs = [0]*4
            mux_freqs[cfg.expt.qubit] = cfg.device.readout.frequency
            mux_gains = [0]*4
            mux_gains[cfg.expt.qubit] = cfg.device.readout.gain
            ro_ch=self.adc_ch
        self.declare_gen(ch=self.res_ch, nqz=cfg.hw.soc.dacs.readout.nyquist, mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)

        # declare qubit dacs
        mixer_freq = 0
        if self.qubit_ch_type == 'int4':
            mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq
        self.declare_gen(ch=self.qubit_ch, nqz=cfg.hw.soc.dacs.qubit.nyquist, mixer_freq=mixer_freq)

        # declare adcs
        self.declare_readout(ch=self.adc_ch, length=self.readout_length_adc, freq=cfg.device.readout.frequency, gen_ch=self.res_ch)

        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_ch) # get start/step frequencies 
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_ch)
        self.pisigma_ge = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma, gen_ch=self.qubit_ch) # default pi_ge value
        self.pief_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma, gen_ch=self.qubit_ch)
        self.ramp = self.us2cycles(cfg.expt.sigma, gen_ch=self.qubit_ch)

        #cavity frequency 
        self.f_cav = self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.cavity_name], gen_ch=self.man_ch)
        #qubit frequency 
        #qtest = self.qubits[0]
        self.f_q = self.f_ge_reg
        
        
        self.add_gauss(ch=self.qubit_ch, name="pi_qubit_ge", sigma=self.pisigma_ge, length=self.pisigma_ge*4)
        self.add_gauss(ch=self.qubit_ch, name="pief_qubit", sigma=self.pief_sigma, length=self.pief_sigma*4)
        self.add_gauss(ch=self.qubit_ch, name="ramp", sigma=self.ramp, length=self.ramp*4)

        self.gain_ge_init = self.cfg.device.qubit.pulses.pi_ge.gain
        self.gain_ef_init = self.cfg.device.qubit.pulses.pi_ef.gain

        # add qubit and readout pulses to respective channels
        # self.set_pulse_registers(ch=self.qubit_ch, style="const", freq=self.f_start, phase=0, gain=cfg.expt.gain, length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch))
        # self.set_pulse_registers(ch=self.qubit_ch, style="flat_top", freq=self.f_start, phase=0, gain=cfg.expt.gain, 
        #                     length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch), waveform="ramp")

        if self.res_ch_type == 'mux4':
            self.set_pulse_registers(ch=self.res_ch, style="const", length=self.readout_length_dac, mask=mask)
        else: self.set_pulse_registers(ch=self.res_ch, style="const", freq=self.f_res_reg, phase=self.deg2reg(cfg.device.readout.phase), gain=cfg.device.readout.gain, length=self.readout_length_dac)

        # load ECD pulse file data
        with open(cfg.expt.pulse_fname + '.npy', 'rb') as f:
            self.cavity_dac_gauss= np.load(f)   
            self.qubit_dac_gauss = np.load(f)


        # setup dummy channel (i will never play this pulse; just want to store frequency here)
        #self.regwi(self.q_rp, self.d_reg, self.f_start)
        #f_start_reg = self.freq2reg(cfg.expt.start, gen_ch=self.storage_ch)
        # self.set_pulse_registers(ch=self.storage_ch, style="const", freq=self.f_start, phase=0, gain=0, 
        #                     length=self.us2cycles(0.01))
        
        # self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.storage_ch)

        print(self.f_start)
        #print(self.d_freq)
        #reg_value = self.read(self.d_rp, self.d_freq)
        #f_dummy = self.reg2freq(reg_value, gen_ch=self.storage_ch)
        #print(f_dummy)
        #print()

        self.set_pulse_registers(ch=self.qubit_ch, style="flat_top", phase=0, gain=cfg.expt.gain, freq = self.f_start,
                            length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch), waveform="ramp")
        
        # set the frequency stored in dummy register to be same as f_start (the one stored in qubit frequency register)
        self.mathi(self.q_rp, self.d_reg, self.r_freq, '+', 0)


        self.synci(200) # give processor some time to configure pulses
    
    def body(self):
        cfg=AttrDict(self.cfg)

        # if cfg.expt.qubit_ge:
        #     self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge_reg, phase=0, gain=self.gain_ge_init, waveform="pi_qubit_ge")
        #     self.sync_all()

        # if cfg.expt.qubit_ef:
        #     self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ef_reg, phase=0, gain=self.gain_ef_init, waveform="pief_qubit")
        #     self.sync_all()

        # if cfg.expt.cavity_drive:
        #     self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.freq2reg(cfg.device.manipulate.f_ge[cfg.expt.cavity_name], gen_ch=self.man_ch), phase=0, gain=cfg.expt.cavity_gain, length=self.us2cycles(cfg.expt.cavity_length, gen_ch=self.man_ch))
        # #     # self.setup_and_pulse(ch=self.man_ch, style="const", freq=cfg.device.manipulate.f_ge[cfg.expt.cavity_name], phase=0, gain=cfg.expt.cavity_gain, length=self.us2cycles(cfg.expt.cavity_length))
        self.sync_all()    

        # --------------------------------------------------------------------------------
        # Iterate over ECD pulses
        for idx, cav_arr in enumerate(self.cavity_dac_gauss): 
            qub_arr = self.qubit_dac_gauss[idx]

            amp_c = cav_arr[0]
            sigma_c = self.us2cycles(cav_arr[1].real * 1e-3) 

            amp_q = qub_arr[0]
            sigma_q = self.us2cycles(qub_arr[1].real * 1e-3)

            name = 'gauss' + str(idx)

            #Pathological Case 1 0 length pulses
            if sigma_c == 0:
                continue 

            # Case 1: qubit off, cavity off  (** replace with sync command)
            if int(np.abs(amp_c)) == 0 and int(np.abs(amp_q)) == 0 : 
            
                self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_q, phase=0, gain=0, length=sigma_q)
                self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_cav, phase=0, gain=0, length=sigma_c)
            
            # Case 2: qubit on, cavity off
            elif int(np.abs(amp_c)) == 0 and int(np.abs(amp_q)) != 0  : 
                
                self.add_gauss_ecd_specific(ch = self.qubit_ch, name = name, sigma = [sigma_q, sigma_q],
                                        length = [4*sigma_q, 4*sigma_q], maxv = [int(np.real(amp_q)), int(np.imag(amp_q))])
                self.setup_and_pulse(ch = self.qubit_ch, style = "arb", freq=self.f_q, phase=0, gain = 1, waveform = name)

                self.setup_and_pulse(ch=self.man_ch, style="const", freq=self.f_cav, phase=0, gain=0, length=sigma_c)
            
            # Case 3: qubit off, cavity on
            elif int(np.abs(amp_c)) != 0 and int(np.abs(amp_q)) == 0  :
                
            
                self.setup_and_pulse(ch=self.qubit_ch, style="const", freq=self.f_q, phase=0, gain=0, length=sigma_q)

                self.add_gauss_ecd_specific(ch = self.man_ch, name = name, sigma = [sigma_c, sigma_c],
                                        length = [4*sigma_c, 4*sigma_c], maxv = [int(np.real(amp_c)), int(np.imag(amp_c))])
                self.setup_and_pulse(ch = self.man_ch, style = "arb",  freq=self.f_cav, phase=0, gain = int(np.abs(amp_c)),waveform = name)

                print('cavity on')
                print('amp is ' + str(amp_c))
                print('sigma is ' + str(sigma_c))
            
            self.sync_all()
        #------------------------------------------------------------------------------------
        
        # Normal Qubit Spectroscopy here ----------------------------------------------------

        

        self.set_pulse_registers(ch=self.qubit_ch, style="flat_top", phase=0, gain=cfg.expt.gain, freq = self.f_start, # I will change this freqency in mathi command below
                            length=self.us2cycles(cfg.expt.length, gen_ch=self.qubit_ch), waveform="ramp")
        
        # get updated frequency from dummy register and put into qubit frequency register
        self.mathi(self.q_rp, self.r_freq, self.d_reg, '+', 0)

        self.pulse(ch=self.qubit_ch) # play probe pulse
        if cfg.expt.wait_qubit:
            self.sync_all(cfg.device.qubit.cycles_add_to_R) # align channels and wait designated time
        else:
            self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(pulse_ch=self.res_ch, 
             adcs=[self.adc_ch],
             adc_trig_offset=cfg.device.readout.trig_offset,
             wait=True,
             syncdelay=self.us2cycles(cfg.device.readout.relax_delay))
    
    def update(self):
        self.mathi(self.q_rp, self.d_reg, self.r_freq, '+', self.f_step) # update frequency list index
 
# ====================================================== #

class ECDExperiment(Experiment):
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
        self.qspec = None

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

        self.qspec = ECDProgram(soccfg=self.soccfg, cfg=self.cfg)
        #print(self.qspec)
        xpts, avgi, avgq = self.qspec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        
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