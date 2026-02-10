from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.helpers import gauss
from slab import AttrDict, Experiment
from tqdm import tqdm_notebook as tqdm

import fitting.fitting as fitter
from experiments.MM_base import MMRAveragerProgram
from experiments.dataset import FloquetStorageSwapDataset, StorageManSwapDataset


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


        self.pulse_to_test = self.get_prepulse_creator([cfg.expt.pulse_type], cfg=cfg).pulse.tolist()
        # flatten the list
        self.pulse_to_test = [item for sublist in self.pulse_to_test for item in sublist]
        print("pulse_to_test:", self.pulse_to_test)
        # add the pulse to test to the channel
        if self.pulse_to_test[5] == 'gauss' and self.pulse_to_test[6] > 0:
            _sigma = self.us2cycles(self.pulse_to_test[6], gen_ch=self.pulse_to_test[4])
            self.add_gauss(ch=self.pulse_to_test[4],
                           name="pulse_to_test",
                           sigma=_sigma,
                           length=_sigma*4, # take 4 sigma cutoff
                           )
        elif self.pulse_to_test[5] == 'flat_top' and  self.pulse_to_test[6] > 0: 
            _sigma = self.us2cycles(self.pulse_to_test[6], gen_ch=self.pulse_to_test[4])
            self.add_gauss(ch=self.pulse_to_test[4],
                              name="pulse_to_test",
                              sigma=_sigma,
                              length=_sigma*4, # take 4 sigma cutoff
                              )

        # initialize registers
        if cfg.expt.parameter_to_test == 'gain':
            if self.pulse_to_test[5] == "flat_top":
                self.r_gain = self.sreg(self.pulse_to_test[4], "gain") # get gain register for qubit_ch ramp part
                self.r_gain2 = self.sreg(self.pulse_to_test[4], "gain2") # get gain register for qubit_ch const part
            else:
                self.r_gain = self.sreg(self.pulse_to_test[4], "gain") # get gain register for qubit_ch
            self.r_gain3 = 4 # update register for the ramp part
            self.channel_page = self.ch_page(self.pulse_to_test[4])
            self.safe_regwi(self.channel_page, self.r_gain3, int(self.cfg.expt.start))
            if self.pulse_to_test[5] == "flat_top":
                assert 5 not in (self.r_gain, self.r_gain2)
                self.r_gain4 = 5 # update register for the const part, which needs to be 1/2 the ramp part
                self.safe_regwi(self.channel_page, self.r_gain4, int(self.cfg.expt.start/2))

        if cfg.expt.parameter_to_test == 'frequency':
            self.channel_page = self.ch_page(self.pulse_to_test[4])
            self.r_freq = self.sreg(self.pulse_to_test[4], "freq")
            self.f_start = self.freq2reg(cfg.expt.start, gen_ch=self.pulse_to_test[4])
            self.f_step = self.freq2reg(cfg.expt.step, gen_ch=self.pulse_to_test[4])
            self.r_freq2 = 4
            self.safe_regwi(self.channel_page, self.r_freq2, self.f_start)
            # define phase register to update it later for pi, -pi pulses
            self.r_phase= self.sreg(self.pulse_to_test[4], "phase")

        self.sync_all(200)


    def body(self):

        cfg=AttrDict(self.cfg)

        # initializations as necessary TBD 
        self.reset_and_sync()

        if cfg.expt.pulse_type[0] in ['storage', 'floquet']:
            # get the qubit start state for storage expt, this is to calibrate chi for a given swap amplitude (i.e. taking into account Stark shift)
            qubit_start_storage = cfg.expt.get('qubit_state_start', 'g')
            print("qubit_start_storage:", qubit_start_storage)

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

        elif cfg.expt.pulse_type[0] in ['storage', 'floquet']:
            man_idx = cfg.expt.pulse_type[1][1]

            pulse_list = [['qubit', 'ge', 'pi', 0],
                            ['qubit', 'ef', 'pi', 0], 
                            ['man', f'M{man_idx}', 'pi', 0]]
            
            if qubit_start_storage == 'e':
                # add ['multiphoton', 'g1-e1', 'pi', 0] at the start
                pulse_list = pulse_list + [['multiphoton', 'g1-e1', 'pi', 0]]
            
            self.creator = self.get_prepulse_creator(
                pulse_list
            )
            self.custom_pulse(cfg, self.creator.pulse.tolist(), prefix='pre_')

        elif cfg.expt.pulse_type[0] == 'multiphoton':
            photon_no = int(cfg.expt.pulse_type[1][1])
            qubit_state_start = cfg.expt.pulse_type[1][0]
            prep_pulses = self.prep_man_photon(photon_no)
            if qubit_state_start == 'g':
                prep_pulses += []
            elif qubit_state_start == 'e':
                prep_pulses += [['multiphoton', 'g' + str(photon_no) + '-e' + str(photon_no), 'pi', 0]]
            elif qubit_state_start == 'f':
                prep_pulses += [['multiphoton', 'g' + str(photon_no) + '-e' + str(photon_no), 'pi', 0]]
                prep_pulses += [['multiphoton', 'e' + str(photon_no) + '-f' + str(photon_no), 'pi', 0]]
            else :
                raise ValueError("Invalid qubit state start. Must be 'g', 'e' or 'f'.")
            # print("prep_pulses:", prep_pulses)
            self.creator = self.get_prepulse_creator(prep_pulses)
            self.custom_pulse(cfg, self.creator.pulse.tolist(), prefix='pre_')

        else:
            raise ValueError("Invalid pulse type. Must be 'qubit', 'man', 'storage', or 'multiphoton'.")


        # set the pulse register to test 
        if self.pulse_to_test[5] == 'gauss':
            pulse_style = "arb"
        elif self.pulse_to_test[5] == 'flat_top':
            pulse_style = "flat_top"
        else:
            raise ValueError("Invalid pulse style. Must be 'gauss' or 'flat_top'.")

        _freq = self.freq2reg(self.pulse_to_test[0], gen_ch=self.pulse_to_test[4])
        if self.pulse_to_test[5] == "gauss":
            self.set_pulse_registers(
                ch=self.pulse_to_test[4],
                style = pulse_style,
                freq=_freq,
                phase = 0,
                gain = int(self.pulse_to_test[1]),
                waveform = "pulse_to_test",
            )
        elif self.pulse_to_test[5] == "flat_top":
            _length = self.us2cycles(self.pulse_to_test[2], gen_ch=self.pulse_to_test[4])
            self.set_pulse_registers(
                ch=self.pulse_to_test[4],
                style = pulse_style,
                freq=_freq,
                length = _length,
                phase = 0,
                gain = int(self.pulse_to_test[1]),
                waveform = "pulse_to_test",
            )
        if cfg.expt.parameter_to_test == 'frequency':    
            self.mathi(self.channel_page, self.r_freq, self.r_freq2, "+", 0)
        elif cfg.expt.parameter_to_test == 'gain':
            self.mathi(self.channel_page, self.r_gain, self.r_gain3, "+", 0) # the arb part
            if self.pulse_to_test[5] == "flat_top":
                self.mathi(self.channel_page, self.r_gain2, self.r_gain4, "+", 0) #  the const part is renamed to gain2
        else:
            raise ValueError("Invalid parameter to test. Must be 'gain' or 'frequency'.")


        # set the number of pulse to be played and start playing
        n_pulses = 1
        if "n_pulses" in cfg.expt:
            n_pulses = cfg.expt.n_pulses



        if cfg.expt.pulse_type[2] == 'pi':
            pi_frac = 1
        elif cfg.expt.pulse_type[2] == 'hpi':
            pi_frac = 2
        elif cfg.expt.pulse_type[2][2] == '/': # pi/pi_frac
            pi_frac = int(cfg.expt.pulse_type[2][3:])
        else:
            pi_frac = 0 # this should not happen
            assert False, "Invalid pulse type. Must be 'pi', 'hpi' or 'pi/pi_frac'."


        if cfg.expt.parameter_to_test == 'gain':
            for i in range((n_pulses * 2) * pi_frac):
                self.pulse(ch = self.pulse_to_test[4])

        elif cfg.expt.parameter_to_test == 'frequency':
            # set the phase register to the initial value
            phase = self.pulse_to_test[3]
            for i in range(n_pulses):
                # for p in range(2):
                #     # play the pulse
                #     self.pulse(ch = self.pulse_to_test[4])
                #     # update the phase modulo 360
                #     phase += 180
                #     phase = phase % 360
                #     _phase_reg = self.deg2reg(phase, gen_ch=self.pulse_to_test[4])
                #     self.safe_regwi(self.channel_page, self.r_phase, _phase_reg)

                for repeat in range(2):
                    for p in range(pi_frac):
                        # play the pulse
                        self.pulse(ch = self.pulse_to_test[4])

                    # update the phase modulo 360
                    phase += 180
                    phase = phase % 360
                    _phase_reg = self.deg2reg(phase, gen_ch=self.pulse_to_test[4])
                    self.safe_regwi(self.channel_page, self.r_phase, _phase_reg)


        self.sync_all()

        # post pulse sequence 

        if cfg.expt.pulse_type[0] == 'qubit':
            if self.pulse_to_test[1] == 'ef':
                post_pulse = self.creator.pulse.tolist() # ge
                self.custom_pulse(cfg, post_pulse, prefix='post_')
        elif cfg.expt.pulse_type[0] in ('man', 'storage', 'floquet'):
            pulse_list = self.creator.pulse.tolist()
            # reverse the pulse sequence
            post_pulse = [sublist[:0:-1] for sublist in pulse_list]
            # when qubit starts in e, it plays g1-e1 which is not ideal since photon can be in storage, but I dont know how to do better for now
            self.custom_pulse(cfg, post_pulse, prefix='post_')
        elif cfg.expt.pulse_type[0] == 'multiphoton':
            qubit_state_start = cfg.expt.pulse_type[1][0]
            if qubit_state_start == 'g':
                post_pulse = []
            if qubit_state_start == 'e':
                last_pulse = [[sublist[-1]] for sublist in  self.creator.pulse.tolist()]
                print("post_pulse:", last_pulse)
                self.custom_pulse(cfg, last_pulse, prefix='post_')
            elif qubit_state_start == 'f':
                print("post_pulse:", self.creator.pulse.tolist())
                last_pulse = [[sublist[-1]] for sublist in  self.creator.pulse.tolist()]
                # print("post_pulse:", last_pulse)
                self.custom_pulse(cfg, last_pulse, prefix='post_')
        else:
            raise ValueError("Invalid pulse type. Must be 'qubit', 'man', 'storage', 'floquet', or 'multiphoton'.")

        self.sync_all()
        # align channel and measure
        self.measure_wrapper()
 
    def update(self):

        step = self.cfg.expt.step
        if self.cfg.expt.parameter_to_test == 'gain':
            self.mathi(self.channel_page, self.r_gain3, self.r_gain3, "+", int(step))
            if self.pulse_to_test[5] == "flat_top":
                self.mathi(self.channel_page, self.r_gain4, self.r_gain4, "+", int(step/2))
        elif self.cfg.expt.parameter_to_test == 'frequency':
            _step = self.freq2reg(step, gen_ch=self.pulse_to_test[4])
            self.mathi(self.channel_page, self.r_freq2, self.r_freq2, "+", _step)
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

        print("cfg at start of acquire", self.cfg.expt)

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


        cfg = deepcopy(self.cfg)
        adc_ch = cfg.hw.soc.adcs.readout.ch
        n_start = 1 if "n_start" not in cfg.expt else cfg.expt.n_start
        n_step = 1 if "n_step" not in cfg.expt else cfg.expt.n_step
        n_pts = np.arange(n_start, cfg.expt.n_pulses + n_step, n_step)

        # Calculate read_num to account for active_reset measurements
        read_num = 1
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        data = {"npts":[],"x_pts":[], "avgi":[], "avgq":[], "amp":[], "phase":[]}
        for pt in tqdm(n_pts):
            cfg.expt.n_pulses = pt
            prog = ErrorAmplificationProgram(soccfg=self.soccfg, cfg=cfg)
            xpts, avgi, avgq = prog.acquire(self.im[self.cfg.aliases.soc], threshold=None, load_pulses=True, progress=False, debug=debug, readouts_per_experiment=read_num)
            avgi = avgi[adc_ch[0]][-1]  # Get last readout (actual measurement after active_reset)
            avgq = avgq[adc_ch[0]][-1]
            amp = np.abs(avgi+1j*avgq) # Calculating the magnitude
            phase = np.angle(avgi+1j*avgq) # Calculating the phase        

            data["avgi"].append(avgi)
            data["avgq"].append(avgq)
            data["amp"].append(amp)
            data["phase"].append(phase)

        data["N_pts"] = n_pts
        data["x_pts"] = xpts

        for k, a in data.items():
            data[k] = np.array(a)
        self.data=data
        return data
    
    def analyze(self, data=None, fit=True, state_fin='g', **kwargs):
        if data is None:
            data=self.data

        # use the fitting process implemented by MIT 
        # https://arxiv.org/pdf/2406.08295
        
        # for avgi, avgq, amp and phase take the product of the raws and

        # prod_avgi = np.abs(np.prod(data['avgi'], axis=0))
        # prod_avgq = np.abs(np.prod(data['avgq'], axis=0))
        # prod_amp = np.abs(np.prod(data['amp'], axis=0))
        # prod_phase = np.abs(np.prod(data['phase'], axis=0))


        Ie = self.cfg.device.readout.Ie[0]
        Ig = self.cfg.device.readout.Ig[0]

        # rescale avgi so that when equal to v_e it is 0 and when equal to v_g it is 1
        if state_fin == 'g':
            data_avgi_scaled = (data['avgi'] - Ie) / (Ig - Ie)
        elif state_fin == 'e':
            data_avgi_scaled = (data['avgi'] - Ig) / (Ie - Ig)
        else:
            raise ValueError("Invalid state_fin. Must be 'g' or 'e'.")

        prod_avgi = np.prod(data_avgi_scaled, axis=0)/ np.prod(data_avgi_scaled, axis=0).max()  # normalize the product
        data['prod_avgi'] = prod_avgi  # normalize the product

        if fit:
            p_avgi, pCov_avgi = fitter.fitgaussian(data['x_pts'], data['prod_avgi'])
            data['prod_avgi_fit'] = fitter.gaussianfunc(data['x_pts'], *p_avgi)
            # add the fit parameters to the data dictionary
            data['fit_avgi'] = p_avgi
            data['fit_prod_avgi_err'] = np.sqrt(np.diag(pCov_avgi))
    

    def display(self, data=None, fit=True, **kwargs):
        if data is None:
            data=self.data 
        
        x_sweep = data['x_pts']
        y_sweep = data['N_pts']
        avgi = data['avgi']
        avgq = data['avgq']

        if self.cfg.expt.parameter_to_test == 'gain':
            xlabel = "Gain [dac units]"
        elif self.cfg.expt.parameter_to_test == 'frequency':
            xlabel = "Frequency [MHz]"
        else:
            raise ValueError("Invalid parameter to test. Must be 'gain' or 'frequency'.")



        title= f"Err Ampl: {self.cfg.expt.pulse_type[0]}-{self.cfg.expt.pulse_type[1]}-{self.cfg.expt.pulse_type[2]}"
        fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, sharey=True)
        ax[0].imshow(
            np.flip(avgi, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        ax[0].set_title(title)
        ax[0].set_xlabel(xlabel)
        ax[0].set_ylabel('N pulse')
        fig.colorbar(ax[0].imshow(np.flip(avgi, 0), cmap='viridis', extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]], aspect='auto'), ax=ax[0], label='I [ADC level]')

        ax[1].imshow(
            np.flip(avgq, 0),
            cmap='viridis',
            extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]],
            aspect='auto')
        ax[1].set_title(title)
        ax[1].set_xlabel(xlabel)
        ax[1].set_ylabel('N pulse')
        fig.colorbar(ax[1].imshow(np.flip(avgq, 0), cmap='viridis', extent=[x_sweep[0], x_sweep[-1], y_sweep[0], y_sweep[-1]], aspect='auto'), ax=ax[1], label='Q [ADC level]')

        if fit: 
            if 'fit_avgi' in data:
                x_opt = data['fit_avgi'][2]
                ax[0].axvline(x_opt, color='black', linestyle='--')

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(x_sweep, data['prod_avgi'], label='Avg I Product')
            # add the fit line if available
            if 'prod_avgi_fit' in data:
                ax2.plot(x_sweep, data['prod_avgi_fit'], label='Fit Avg I Product', color='black')
                # add a text annotation for the optimal point if available and put it in the upper left corner
                x_opt = data['fit_avgi'][2]
                if self.cfg.expt.parameter_to_test == 'gain':
                    text = f"Optimal Gain: {x_opt:.2f} DAC units"
                elif self.cfg.expt.parameter_to_test == 'frequency':
                    text = f"Optimal Frequency: {x_opt:.2f} MHz"
                ax2.axvline(x_opt, color='black', linestyle='--')
                ax2.text(0.05, 0.95, text, transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

            ax2.set_xlabel(xlabel)
            ax2.set_ylabel('Avg I Product')
            ax2.legend(loc='lower left')
            ax2.grid()

            if 'fit_avgq' in data:
                ax[1].fill_between(x_sweep, 
                                 data['fit_avgq'] - data['fit_avgq_err'], 
                                 data['fit_avgq'] + data['fit_avgq_err'], 
                                 alpha=0.2, color='black')
                ax[1].set_xlabel(xlabel)
                ax[1].set_ylabel('Avgq')
        plt.show()


    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        super().save_data(data=data)        

            
    


        
