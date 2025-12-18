import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from MM_base import MMRAveragerProgram
from experiments.single_qubit.pulse_probe_ef_spectroscopy import PulseProbeEFSpectroscopyProgram

class PulseProbeF0g1SpectroscopyProgram(MMRAveragerProgram):
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
        qTest = cfg.expt.qubits[0]

        self.q_rp=self.ch_page(self.f0g1_ch[qTest]) # get register page for f0g1_ch
        self.r_freq=self.sreg(self.f0g1_ch[qTest], "freq") # get frequency register for qubit_ch 
        self.r_freq2 = 4
        self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.qubit_chs[qTest])
        self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.qubit_chs[qTest])
        

        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start) # send start frequency to r_freq2
        self.synci(200)

    def body(self):
        cfg=AttrDict(self.cfg)
        qTest = self.qubits[0]

        #prepulse : 
        self.sync_all()
        if cfg.expt.prepulse:
            self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix = 'pre_ar_')

        
        if self.cfg.expt['qubit_f']:
            # init to qubit excited state
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ge_reg[0], phase=0, gain=self.pi_ge_gain, waveform="pi_qubit_ge")
            # None
            self.setup_and_pulse(ch=self.qubit_chs[qTest], style="arb", freq=self.f_ef_reg[0], phase=0, gain=self.pi_ef_gain, waveform="pi_qubit_ef")
        # print(self.pief_gain)
        # setup and play ef probe pulse
        self.set_pulse_registers(
            ch=self.f0g1_ch[qTest],
            style="const",
            freq=0, # freq set by update
            phase=0,
            gain=cfg.expt.gain,
            length=self.us2cycles(cfg.expt.length, gen_ch=self.f0g1_ch[qTest]))
        self.mathi(self.q_rp, self.r_freq, self.r_freq2, "+", 0)
        self.pulse(ch=self.f0g1_ch[qTest])

        # go back to ground state if in e to distinguish between e and f
        # self.setup_and_pulse(ch=self.qubit_ch, style="arb", freq=self.f_ge_reg, phase=0, gain=cfg.device.qubit.pulses.pi_ge.gain, waveform="pi_qubit")

        self.sync_all(self.us2cycles(0.05)) # align channels and wait 50ns
        self.measure(
            pulse_ch=self.res_chs[qTest], 
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(cfg.device.readout.relax_delay[qTest])
        )
    
    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index
        

class PulseProbeF0g1SpectroscopyExperiment(Experiment):
    """
    PulseProbe EF Spectroscopy Experiment
    Experimental Config:
    expt = dict(
        start: start ef probe frequency [MHz]
        step: step ef probe frequency
        expts: number experiments stepping from start
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
        gain: ef const pulse gain [dac units]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeF0g1Spectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
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
        read_num = 1                             

        qspec_ef=PulseProbeF0g1SpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = qspec_ef.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress, debug=debug)        

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
        phases = np.angle(avgi+1j*avgq) # Calculating the phase        

        data={'xpts': x_pts, 'avgi':avgi, 'avgq':avgq, 'amps':amps, 'phases':phases}
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

    def display(self, data=None, fit=True, signs=[1,1, 1], **kwargs):
        if data is None:
            data=self.data 

        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            xpts = self.cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts'][1:-1]
        else: 
            xpts = data['xpts'][1:-1]

        plt.figure(figsize=(9, 11))
        plt.subplot(311, title=f" Spectroscopy (Gain {self.cfg.expt.gain})", ylabel="Amplitude [ADC units]")
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




class PulseProbeEFPowerSweepSpectroscopyExperiment(Experiment):
    """
    Pulse probe EF power sweep spectroscopy experiment
    Experimental Config
        expt = dict(
        start_f: start ef probe frequency [MHz]
        step_f: step ef probe frequency
        expts_f: number experiments freq stepping from start
        start_gain: start ef const pulse gain (dac units)
        step_gain
        expts_gain
        reps: number averages per experiment
        rounds: number repetitions of experiment sweep
        length: ef const pulse length [us]
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeEFPowerSweepSpectroscopy', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"]*np.arange(self.cfg.expt["expts_f"])
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"]*np.arange(self.cfg.expt["expts_gain"])
        
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
       
        data={"fpts":[], "gainpts":[], "avgi":[], "avgq":[], "amps":[], "phases":[]}
        for gain in tqdm(gainpts):
            self.cfg.expt.gain = gain
            self.cfg.expt.start = self.cfg.expt.start_f
            self.cfg.expt.step = self.cfg.expt.step_f
            self.cfg.expt.expts = self.cfg.expt.expts_f
            spec = PulseProbeEFSpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
            self.prog = spec
            x_pts, avgi, avgq = spec.acquire(self.im[self.cfg.aliases.soc], load_pulses=True, progress=False)
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amp)
            data["phases"].append(phase)

        data["fpts"] = fpts
        data["gainpts"] = gainpts
        
        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        if data is None:
            data=self.data

        # Lorentzian fit at highgain [DAC units] and lowgain [DAC units]
        # if fit:
        #     if highgain == None: highgain = data['gainpts'][-1]
        #     if lowgain == None: lowgain = data['gainpts'][0]
        #     i_highgain = np.argmin(np.abs(data['gainpts']-highgain))
        #     i_lowgain = np.argmin(np.abs(data['gainpts']-lowgain))
        #     fit_highpow=dsfit.fitlor(data["fpts"], data["avgi"][i_highgain])
        #     fit_lowpow=dsfit.fitlor(data["fpts"], data["avgi"][i_lowgain])
        #     data['fit'] = [fit_highpow, fit_lowpow]
        #     data['fit_gains'] = [highgain, lowgain]
        #     data['lamb_shift'] = fit_highpow[2] - fit_lowpow[2]

        return data

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 

        x_sweep = data['fpts']
        y_sweep = data['gainpts'] 
        avgi = data['avgi']
        avgq = data['avgq']
        for avgi_gain in avgi:
            avgi_gain -= np.average(avgi_gain)
        for avgq_gain in avgq:
            avgq_gain -= np.average(avgq_gain)


        plt.figure(figsize=(10,12))
        plt.subplot(211, title="Pulse Probe EF Spectroscopy Power Sweep", ylabel="Pulse Gain [adc level]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Amps-Avg [adc level]')

        plt.subplot(212, xlabel="Pulse Frequency (MHz)", ylabel="Pulse Gain [adc level]")
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.clim(vmin=None, vmax=None)
        plt.colorbar(label='Phases-Avg [radians]')
        
        plt.show()    

        # if fit:
        #     fit_highpow, fit_lowpow = data['fit']
        #     highgain, lowgain = data['fit_gains']
        #     plt.axvline(fit_highpow[2], linewidth=0.5, color='0.2')
        #     plt.axvline(fit_lowpow[2], linewidth=0.5, color='0.2')
        #     plt.plot(x_sweep, [highgain]*len(x_sweep), linewidth=0.5, color='0.2')
        #     plt.plot(x_sweep, [lowgain]*len(x_sweep), linewidth=0.5, color='0.2')
        #     print(f'High power peak [MHz]: {fit_highpow[2]}')
        #     print(f'Low power peak [MHz]: {fit_lowpow[2]}')
        #     print(f'Lamb shift [MHz]: {data["lamb_shift"]}')

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


# ----------------------------------------------------------------------
# 2D Experiment: PulseProbe f0g1 spectroscopy vs DC flux (single H5 file)
# ----------------------------------------------------------------------
class PulseProbeF0g1SpectroscopyFluxSweepExperiment(Experiment):
    """
    2D sweep of f0g1 spectroscopy vs DC flux current using Yokogawa GS200.

    Experimental Config
    expt = dict(
        # Frequency axis (handled inside the f0g1 program)
        start: start frequency (Hz or MHz per your setup)
        step: frequency step
        expts: number of frequency points
        reps: averages per frequency point
        rounds: sweep repetitions
        length: f0g1 const pulse length [us]
        gain: f0g1 const pulse gain [DAC units]
        qubits: [qubit_index]

        # Current axis (outer sweep)
        curr_start: starting current [A]
        curr_step: step [A]
        curr_expts: number of current points

        # Optional safety/driver
        yokogawa_address: '192.168.137.148'
        sweeprate: 0.002  # A/s
        safety_limit: 0.03  # |A|
    )
    """

    def __init__(self, soccfg=None, path='', prefix='PulseProbeF0g1SpectroscopyFluxSweep', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)

    def acquire(self, progress=False):
        # Resolve current sweep params with defaults
        ex = self.cfg.expt
        curr_start = float(getattr(ex, 'curr_start', 0.0))*1e-3
        curr_step = float(getattr(ex, 'curr_step', 0.001))*1e-3
        curr_expts = int(getattr(ex, 'curr_expts', 11))
        address = getattr(ex, 'yokogawa_address', '192.168.137.148')
        sweeprate = float(getattr(ex, 'sweeprate', 2))*1e-3
        safety_limit = float(getattr(ex, 'safety_limit', 30))*1e-3

        currents = curr_start + curr_step * np.arange(curr_expts, dtype=float) # to mA
        # Initialize Yokogawa
        try:
            from slab.instruments import YokogawaGS200
        except Exception as e:
            raise ImportError("YokogawaGS200 instrument driver not available. Ensure slab/instruments are installed.") from e

        dcflux = YokogawaGS200(address=address)
        dcflux.set_output(True)
        dcflux.set_mode('current')
        dcflux.ramp_current(0.000, sweeprate=sweeprate)

        avgi_rows = []
        avgq_rows = []
        amps_rows = []
        phases_rows = []
        xpts_ref = None

        try:
            for idx, target in enumerate(currents):
                # safety clip
                clipped = max(min(float(target), safety_limit), -safety_limit)
                if clipped != float(target):
                    print(f"[WARN] Requested current {target:.6f} A exceeds ±{safety_limit:.3f} A. Clipped to {clipped:.6f} A.")
                    target = clipped

                print(f"{idx}: Setting DC flux to {target*1e3:.3f} mA")
                dcflux.ramp_current(target, sweeprate=sweeprate)

                # Run the f0g1 frequency sweep program at this current
                # Pass current through cfg so it can be saved
                try:
                    ex['current'] = float(target)
                except Exception:
                    setattr(ex, 'current', float(target))

                spec = PulseProbeF0g1SpectroscopyProgram(soccfg=self.soccfg, cfg=self.cfg)
                x_pts, avgi, avgq = spec.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=progress)

                # Flatten to 1D
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amps = np.abs(avgi + 1j * avgq)
                phases = np.angle(avgi + 1j * avgq)

                if xpts_ref is None:
                    xpts_ref = np.array(x_pts).copy()

                avgi_rows.append(np.array(avgi))
                avgq_rows.append(np.array(avgq))
                amps_rows.append(np.array(amps))
                phases_rows.append(np.array(phases))
        finally:
            try:
                dcflux.ramp_current(0.000, sweeprate=sweeprate)
            except Exception:
                pass

        # Stack into 2D arrays (currents x freqs)
        data = {
            'fpts': np.array(xpts_ref),
            'currents': np.array(currents),
            'avgi': np.array(avgi_rows),
            'avgq': np.array(avgq_rows),
            'amps': np.array(amps_rows),
            'phases': np.array(phases_rows),
        }

        self.data = data
        return data

    def analyze(self, data=None, fit=False, **kwargs):
        # Placeholder: 2D fits can be added later; right now we just return the data.
        if data is None:
            data = self.data
        return data

    def display(self, data=None, which: str = 'amps', **kwargs):
        if data is None:
            data = self.data

        fpts = data['fpts']
        currents = data['currents']*1e3
        Z = data.get(which, data['amps'])

        # Frequency axis: add mixer_freq if present (similar to 1D display)
        if 'mixer_freq' in self.cfg.hw.soc.dacs.qubit:
            faxis = self.cfg.hw.soc.dacs.qubit.mixer_freq + fpts
        else:
            faxis = fpts

        plt.figure(figsize=(10, 6))
        plt.title(f"f0g1 Spectroscopy vs Flux ({which})")
        plt.xlabel("Pulse Frequency")
        plt.ylabel("Flux current [mA]")
        plt.imshow(
            np.flip(Z, 0),
            cmap='viridis',
            extent=[faxis[0], faxis[-1], currents[0], currents[-1]],
            aspect='auto')
        plt.colorbar(label=which)
        plt.tight_layout()
        plt.show()

        

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname


# ---------------------------------------------------------------
# Convenience runner: instantiate the 2D Experiment and save one H5
# ---------------------------------------------------------------
def run_pulseprobe_f0g1_spectroscopy_vs_flux(
    soccfg=None,
    path: str = '',
    prefix: str = 'PulseProbeF0g1SpectroscopyFluxSweep',
    config_file: str = None,
    progress: bool = False,
    curr_start: float = 0.0,
    curr_step: float = 0.001,
    curr_expts: int = 11,
    yokogawa_address: str = '192.168.137.148',
    sweeprate: float = 0.002,
    safety_limit: float = 0.03,
    save: bool = True,
):
    # Prepare an experiment instance and push sweep params into cfg.expt
    exp = PulseProbeF0g1SpectroscopyFluxSweepExperiment(
        soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress
    )
    exp.cfg.expt.curr_start = curr_start
    exp.cfg.expt.curr_step = curr_step
    exp.cfg.expt.curr_expts = curr_expts
    exp.cfg.expt.yokogawa_address = yokogawa_address
    exp.cfg.expt.sweeprate = sweeprate
    exp.cfg.expt.safety_limit = safety_limit

    exp.go(analyze=False, display=False, progress=progress, save=save)
    return getattr(exp, 'fname', None)