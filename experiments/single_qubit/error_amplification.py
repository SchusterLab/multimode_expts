import numpy as np
import matplotlib.pyplot as plt
from qick import *
from qick.helpers import gauss
from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm
import experiments.fitting as fitter
from copy import deepcopy
from MM_base import MMRAveragerProgram


class ErrorAmplificationProgram(MMRAveragerProgram):
    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.rounds
        
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        cfg = AttrDict(self.cfg)
        self.MM_base_initialize()
        qTest = 0 

        # what pulse do we want to calibrate?
        # use the pre_pulse_creator to define pulse parameters
        # I should add user define pulse later for more flexibility
        self.pulse_to_test = self.get_prepulse_creator(cfg.expt.pulse_type).pulse.tolist()

        # add the pulse to test to the channel
        if self.pulse_to_test[5] == 'gauss' and self.pulse_to_test[6] > 0:
            self.add_gauss(ch=self.pulse_to_test[4],
                           name="pulse_to_test",
                           sigma=self.pulse_to_test[6],
                           length=self.pulse_to_test[6]*4, # take 4 sigma cutoff
                           )

        # initialize registers
        if cfg.expt.parameter_to_test == 'gain':
            if self.pulse_to_test[5] == "flat_top":
                self.r_gain = self.sreg(self.pulse_to_test[4], "gain") # get gain register for qubit_ch
                self.r_gain2 = self.sreg(self.pulse_to_test[4], "gain2") # get gain register for qubit_ch
            else:
                self.r_gain = self.sreg(self.pulse_to_test[4], "gain") # get gain register for qubit_ch

            self.r_gain3 = 4 # I am taking this from amplitude rabi but I am not sure why 4
            self.channel_page = self.ch_page(self.pulse_to_test[4])
            self.safe_regwi(self.channel_page, self.r_gain3, self.cfg.expt.start)

        if cfg.expt.parameter_to_test == 'frequency':
            self.channel_page = self.ch_page(self.pulse_to_test[4])
            self.r_freq = self.sreg(self.pulse_to_test[4], "freq")
            self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.pulse_to_test[4])
            self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.pulse_to_test[4])
            self.r_freq2 = 4
            self.safe_regwi(self.channel_page, self.r_freq2, self.f_start)

        self.sync_all(200)

    def body(self):

        cfg=AttrDict(self.cfg)

        # initializations as necessary TBD 
        self.reset_and_sync()

        # set the prepulse sequence depending on the pulse to calibrate 
        # TO DO: replace everything with the multiphoton def 
        if cfg.expt.pulse_type[0] == 'qubit':
            if self.pulse_to_test[1] =='ef':
                self.creator = self.get_prepulse_creator(
                    [['qubit', 'ge', 'pi', 0]]
                )
                self.custom_pulse(cfg, self.creator.pulse.tolist(), prefix='pre_')
        
        # this will be deleted once we replace everything with the multiphoton def
        elif cfg.expt.pulse_type[0] == 'man':
            self.creator = self.get_prepulse_creator(
                    [['qubit', 'ge', 'pi', 0],
                     ['qubit', 'ef', 'pi', 0]]
                )
            self.custom_pulse(cfg, self.creator.pulse.tolist(), prefix='pre_')

        elif cfg.expt.pulse_type[0] == 'storage':
            man_idx = cfg.expt.pulse_type[1][1]
            self.creator = self.get_prepulse_creator(
                [['qubit', 'ge', 'pi', 0],
                 ['qubit', 'ef', 'pi', 0],
                 ['man', f'M{man_idx}', 'pi', 0]]
            )
            self.custom_pulse(cfg, self.creator.pulse.tolist(), prefix='pre_')

        elif cfg.expt.pulse_type[0] == 'multiphoton':
            photon_no = int(cfg.expt.pulse_type[1][1])
            qubit_state_start = cfg.expt.pulse_type[1][0]
            prep_pulses = self.prep_man_photon(photon_no)
            if qubit_state_start == 'e':
                prep_pulses += [['qubit', 'g' + str(photon_no) + '-e' + str(photon_no), 'pi', 0]]
            elif qubit_state_start == 'f':
                prep_pulses += [['qubit', 'g' + str(photon_no) + '-e' + str(photon_no), 'pi', 0]]
                prep_pulses += [['qubit', 'e' + str(photon_no) + '-f' + str(photon_no), 'pi', 0]]
            else :
                raise ValueError("Invalid qubit state start. Must be 'e' or 'f'.")
            self.creator = self.get_prepulse_creator(prep_pulses)
            self.custom_pulse(cfg, self.creator.pulse.tolist(), prefix='pre_')



        else:
            raise ValueError("Invalid pulse type. Must be 'qubit', 'man', 'storage', or 'multiphoton'.")

        print("pulse preparation: ", self.creator.pulse)

        # set the pulse register to test 
        if self.pulse_to_test[5] == 'gauss':
            pulse_style = "arb"
        elif self.pulse_to_test[5] == 'flat_top':
            pulse_style = "flat_top"
        else:
            raise ValueError("Invalid pulse style. Must be 'gauss' or 'flat_top'.")


        if cfg.expt.parameter_to_test == 'gain':
            self.set_pulse_registers(
                ch=self.pulse_to_test[4],
                style = pulse_style,
                freq=self.pulse_to_test[0],
                phase = 0,
                gain = 0, # dummy
                waveform = "pulse_to_test",
            )

            self.mathi(self.channel_page, self.r_gain, self.r_gain3, "+", 0)
            if self.pulse_to_test[5] == "flat_top":
                self.mathi(self.channel_page, self.r_gain2, self.r_gain3, "+", 0)

        elif cfg.expt.parameter_to_test == 'frequency':
            self.set_pulse_registers(
                ch=self.pulse_to_test[4],
                style = pulse_style,
                freq=self.pulse_to_test[0], # dummy 
                phase = 0,
                gain = self.pulse_to_test[1], 
                waveform = "pulse_to_test",
            )
            self.mathi(self.channel_page, self.r_freq, self.r_freq2, "+", 0)

        else:
            raise ValueError("Invalid parameter to test. Must be 'gain' or 'frequency'.")


        # set the number of pulse to be played and start playing
        n_pulses = 1
        if "n_pulses" in cfg.expt:
            n_pulses = cfg.expt.n_pulses
        if cfg.expt.pulse_type[2] == 'hpi':
            n_pulses *=2
        for i in range(n_pulses):
            self.pulse(ch = self.pulse_to_test[4])

        self.sync_all()

        # post pulse sequence 

        if cfg.expt.pulse_type[0] == 'qubit':
            if self.pulse_to_test[1] == 'ef':
                post_pulse = self.creator.pulse.tolist()[0] # ge
                self.custom_pulse(cfg, [post_pulse], prefix='post_')
        elif cfg.expt.pulse_type[0] == 'man':
            post_pulse = self.creator.pulse.tolist()[-1] # ef 
            self.custom_pulse(cfg, [post_pulse], prefix='post_')
        elif cfg.expt.pulse_type[0] == 'storage':
            post_pulse = self.creator.pulse.tolist()[-2:]
            post_pulse = post_pulse[::-1]
            self.custom_pulse(cfg, post_pulse, prefix='post_')
        elif cfg.expt.pulse_type[0] == 'multiphoton':
            qubit_state_start = cfg.expt.pulse_type[1][0]
            if qubit_state_start == 'e':
                post_pulse = self.creator.pulse.tolist()[-0] #ge
                self.custom_pulse(cfg, [post_pulse], prefix='post_')
            elif qubit_state_start == 'f':
                post_pulse = self.creator.pulse.tolist()[-1] # ef
                self.custom_pulse(cfg, [post_pulse], prefix='post_')
        else:
            raise ValueError("Invalid pulse type. Must be 'qubit', 'man', 'storage', or 'multiphoton'.")

        self.sync_all()
        # align channel and measure
        self.measure_wrapper()
 
    def update(self):

        step = self.cfg.expt.step
        if self.cfg.expt.parameter_to_test == 'gain':
            self.mathi(self.channel_page, self.r_gain, self.r_gain3, "+", step)
        elif self.cfg.expt.parameter_to_test == 'frequency':
            self.mathi(self.channel_page, self.r_freq, self.r_freq2, "+", step)
        else:
            raise ValueError("Invalid parameter to test. Must be 'gain' or 'frequency'.")


    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        self.readout_length_adc = self.readout_lengths_adc[0]
        shots_i0 = self.di_buf[0] / self.readout_length_adc
        shots_q0 = self.dq_buf[0] / self.readout_length_adc
        return shots_i0, shots_q0


class ErrorAmplificationExperiment(Experiment):
    """
    Experiment to test the error amplification by changing
    the gain or frequency of a pulse.
    Experiment parameters:
    expt = dict(
        parameter_to_test='gain',  # 'gain' or 'frequency'
        pulse_type=['type', 'transition', 'pi/hpi', 'phase'],  # pulse parameters
        start,  # start value for gain or frequency
        step,  # step size for gain or frequency
        reps,  # number of repetitions
        rounds,  # number of rounds
    )
    """

    def __init__(self, soccfg=None, path='', prefix='ErrorAmplification', config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)


    
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


        cfg = deepcopy(self.cfg)
        adc_ch = cfg.hw.soc.adcs.readout.ch
        n_pts = np.arange(1, cfg.expt.n_pulses) 
        x_pts = np.arange(cfg.expt.start, cfg.expt.start + cfg.expt.step * len(n_pts), cfg.expt.step)
        
        data = {"npts":[],"x_pts":[], "avgi":[], "avgq":[], "amps":[], "phase":[]}
        for pt in n_pts:
            cfg.expt.n_pulses = pt
            prog = ErrorAmplificationProgram(soccfg=self.soccfg, cfg=cfg)
            xpts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug)
        
            avgi = avgi[adc_ch][0]
            avgq = avgq[adc_ch][0]
            amps = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phases = np.angle(avgi+1j*avgq) # Calculating the phase        

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amps"].append(amps)
            data["phases"].append(phases)

        data["N_pts"] = n_pts
        data["x_pts"] = x_pts

        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data
    
    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data
        pass

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        x_sweep = data['x_pts']
        y_sweep = data['N_pts']
        avgi = data['avgi']
        avgq = data['avgq']

        if self.cfg.expt.parameter_to_test == 'gain':
            ylabel = "Gain [dac units]"
        elif self.cfg.expt.parameter_to_test == 'frequency':
            ylabel = "Frequency [MHz]"
        else:
            raise ValueError("Invalid parameter to test. Must be 'gain' or 'frequency'.")



        title= f"Error Amplification: {self.cfg.expt.pulse_type[0]}-{self.cfg.expt.pulse_type[1]}"

        plt.figure(figsize=(10,8))
        plt.subplot(211, title=title, ylabel=ylabel)
        plt.imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='I [ADC level]')
        plt.clim(vmin=None, vmax=None)
        # plt.axvline(1684.92, color='k')
        # plt.axvline(1684.85, color='r')

        plt.subplot(212, title=title, ylabel=ylabel, xlabel='N pulse')
        plt.imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        plt.colorbar(label='Q [ADC level]')
        plt.clim(vmin=None, vmax=None)
        
        if fit: pass

        plt.tight_layout()
        plt.show()
    


        