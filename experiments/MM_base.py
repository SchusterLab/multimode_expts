import logging
from typing import List, Optional, Union
import yaml
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from qick import AveragerProgram, QickProgram, RAveragerProgram
from slab import AttrDict

from experiments.dataset import FloquetStorageSwapDataset, StorageManSwapDataset

logger = logging.getLogger("qick.qick_asm")
logger.setLevel(logging.ERROR)


def print_debug():
    import inspect
    print(inspect.getfile(QickProgram))
    print(inspect.getfile(AveragerProgram))
    print(inspect.getfile(RAveragerProgram))


class MM_base:
    """
    Methods and handy properties that are useful for both averager and raverager programs
    Prepares the commonly used pulses in multimode experiments
    such as qubit ge, ef, f0g1, M1-Sx π and π/2 pulses,
    such that child classes can directly use the waveforms (gaussians) added here.
    Also provides a more generic way to create custom pulses and many convenience functions.
    """
    def __init__(self, cfg: AttrDict, soccfg: AttrDict):
        self.cfg = cfg
        self.soccfg = soccfg
        # raise NotImplementedError("""
        # Don't instantiate this directly.
        # Inherit together with a QickProgram.
        # See eg MMAveragerProgram or MMRAveragerProgram for usage.
        # """)

    def parse_config(self):
        '''
        "Software" initialization: parses the cfg and stores parameters in self for easy access
        such as channel info, frequency, gain for various pulses.
        This is called by self.MM_base_initialize() during intialization
        '''
        cfg = self.cfg
        # self.cfg = cfg
        self.cfg.update(cfg.expt) # this is to make things like reps, expts available

        # self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        try:
            self.qubits = self.cfg.expt.qubits
        except AttributeError:
            self.qubits = self.cfg.expt.qubit

        qTest = self.qubits[0]

        # -----------channel info-----------
        self.adc_chs = cfg.hw.soc.adcs.readout.ch
        self.res_chs = cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = cfg.hw.soc.dacs.readout.type
        self.qubit_chs = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = cfg.hw.soc.dacs.qubit.type
        self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
        self.man_ch = cfg.hw.soc.dacs.manipulate_in.ch
        self.man_ch_type = cfg.hw.soc.dacs.manipulate_in.type
        self.flux_low_ch = cfg.hw.soc.dacs.flux_low.ch
        self.flux_low_ch_type = cfg.hw.soc.dacs.flux_low.type
        self.flux_high_ch = cfg.hw.soc.dacs.flux_high.ch
        self.flux_high_ch_type = cfg.hw.soc.dacs.flux_high.type
        self.f0g1_ch = cfg.hw.soc.dacs.sideband.ch
        self.f0g1_ch_type = cfg.hw.soc.dacs.sideband.type
        self.storage_ch = cfg.hw.soc.dacs.storage_in.ch
        self.storage_ch_type = cfg.hw.soc.dacs.storage_in.type

        # ----------register page for qubit chs----------
        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs]
        # self.rf_rps = [self.ch_page(ch) for ch in self.rf_ch]

        # --------------frequencies (register values)----------
        self.f_ge = self.freq2reg(cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_ch[qTest])
        self.f_ef = self.freq2reg(cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_ch[qTest])
        self.f_ge_pi_fast = self.freq2reg(cfg.device.multiphoton.pi['gn-en'].frequency[0], gen_ch=self.qubit_ch[qTest])
        self.f_ge_hpi_fast = self.freq2reg(cfg.device.multiphoton.hpi['gn-en'].frequency[0], gen_ch=self.qubit_ch[qTest])
        # -----------freqeuncies: (same as above but diff name and as lists...)-----------
        self.f_ge_reg = [self.freq2reg(
            cfg.device.qubit.f_ge[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_ef_reg = [self.freq2reg(
            cfg.device.qubit.f_ef[qTest], gen_ch=self.qubit_chs[qTest])]
        self.f_res_reg = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(
            cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        # self.f_rf_reg = [self.freq2reg(self.cfg.expt.flux_drive[1], gen_ch=self.rf_ch[0])]
        # self.f_ge_resolved_reg = [self.freq2reg(
        #     self.cfg.expt.qubit_resolved_pi[0], gen_ch=self.qubit_chs[qTest])]

        # --------------readout lengths---------
        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(
            self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(
            self.cfg.device.readout.readout_length, self.adc_chs)]

        # --------------qubit pulse parameters: sigma----------
        self.pi_ge_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ge.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.hpi_ge_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ge.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.pi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.pi_ef.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.hpi_ef_sigma = self.us2cycles(cfg.device.qubit.pulses.hpi_ef.sigma[0], gen_ch=self.qubit_chs[qTest])
        self.pi_ge_sigma_fast = self.us2cycles(cfg.device.multiphoton.pi['gn-en'].sigma[0], gen_ch=self.qubit_chs[qTest])
        self.hpi_ge_sigma_fast = self.us2cycles(cfg.device.multiphoton.hpi['gn-en'].sigma[0], gen_ch=self.qubit_chs[qTest])
        # --------------qubit pulse parameters: gain----------
        self.pi_ge_gain = cfg.device.qubit.pulses.pi_ge.gain[qTest]
        self.hpi_ge_gain = cfg.device.qubit.pulses.hpi_ge.gain[qTest]
        self.pi_ef_gain = cfg.device.qubit.pulses.pi_ef.gain[qTest]
        self.hpi_ef_gain = cfg.device.qubit.pulses.hpi_ef.gain[qTest]
        self.pi_ge_gain_fast = cfg.device.multiphoton.pi['gn-en'].gain[0]
        self.hpi_ge_gain_fast = cfg.device.multiphoton.hpi['gn-en'].gain[0]

        # -------------f0g1 and M1-S sigmas-------
        #TODO: Get from dataset
        self.dataset = StorageManSwapDataset(cfg.device.storage.storage_man_file)
        self.pi_m1_sigma_low = self.us2cycles(self.cfg.device.manipulate.ramp_sigma, gen_ch=self.flux_low_ch[qTest])
        self.pi_m1_sigma_high = self.us2cycles(self.cfg.device.manipulate.ramp_sigma, gen_ch=self.flux_high_ch[qTest])

        # ------------ Multiphoton Config ---------- 
        f_path = self.cfg.device.multiphoton_config.file
        f_path = Path(__file__).parent.parent / 'configs' / f_path
        with f_path.open('r') as f:
            self.multiphoton_cfg = AttrDict(yaml.safe_load(f))


    def initialize_idling_dataset(self):
        '''
        Create a dictionary that will keep a record of idling times

        dict= {'key = transition' : value = []}
        '''

    def get_prepulse_creator(self, sweep_pulse: Optional[List[List[Union[str,int]]]] = None, cfg: Optional[AttrDict] = None):
        '''
        sweep_pulse:
            [name of channel (qubit, multiphoton, man, storage),
            name of transition of cavity name like 'ge', 'ef' or 'M1', 'M1-S1',
            name of pulse like pi, hpi, or parity_M1 or parity_M2,
            phase  (int form )]
        Returns:
            an instance of prepulse creator class


            *** MULTIPHOTON in not a channel but  redirects to qubit and f0 g1
            **** Man is not refferring to man channel but f0g1 channel

        '''
        if cfg is None:
            cfg = self.cfg

        if 'floquet_man_stor_file' not in cfg.device.storage:
            cfg.device.storage.floquet_man_stor_file = None

        creator = prepulse_creator2(cfg, cfg.device.storage.storage_man_file, cfg.device.storage.floquet_man_stor_file)
        # creator = prepulse_creator2(cfg, cfg.device.storage.storage_man_file)
        if sweep_pulse is not None:
            for pulse_idx, pulse in enumerate(sweep_pulse):
                # for each pulse
                pulse_param = list(pulse[1:])
                channel_name = pulse[0]
                eval(f"creator.{channel_name}({pulse_param})")

        return creator

    def compound_storage_gate(self, input = True, storage_no = 1, man_no = 1):
        '''
        input: if True, then the storage gate is on, else output to storage mode

        input from ge state

        returns gate based prepulse string
        '''
        prepulse_str = [ ['qubit', 'ef', 'pi',0],
                    ['man', 'M1' , 'pi',0 ],
                    ['storage', 'M' + str(man_no) + '-S' + str(storage_no), 'pi',0]]
        if not input:
            prepulse_str = prepulse_str [::-1]
            for idx in range(len(prepulse_str)):
                prepulse_str[idx][-1] = 180
        return prepulse_str

    def prep_man_photon(self, photon_no):
        ''' prepare a photon in the manipulate mode '''

        pulse_seq = []
        for i in range(photon_no):
            pulse_seq += [['multiphoton', 'g' + str(i) + '-e' + str(i), 'pi', 0]]
            pulse_seq += [['multiphoton', 'e' + str(i) + '-f' + str(i), 'pi', 0]]
            pulse_seq += [['multiphoton', 'f' + str(i) + '-g' + str(i+1), 'pi', 0]]
        return pulse_seq


    def MM_base_initialize(self):
        '''
        This is effectively the actual initialization function of this class,
        as when inherited after eg RAveragerProgram,
        the __init__ of this class never gets called due to MRO
        where as the initialize() of the child classes does.
        First calls parse_config() to get the parameters
        Then does "hardware" initialization: declares gen/ro channels and adds waveforms
        '''
        self.parse_config()  # parse the cfg to get the parameters
        cfg = self.cfg
        qTest = self.qubits[0]
        # ------ declare res dacs -------
        gen_chs = []
        mask = None
        mixer_freq = None  # MHz
        mux_freqs = None  # MHz
        mux_gains = None
        ro_ch = None
        self.declare_gen(ch=self.res_chs[qTest], nqz=cfg.hw.soc.dacs.readout.nyquist[qTest],
                         mixer_freq=mixer_freq, mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=ro_ch)
        self.declare_readout(ch=self.adc_chs[qTest], length=self.readout_lengths_adc[qTest],
                             freq=cfg.device.readout.frequency[qTest], gen_ch=self.res_chs[qTest])

        # --------declare qubit dacs-------
        for q in self.qubits:
            mixer_freq = None
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in gen_chs:
                self.declare_gen(
                    ch=self.qubit_chs[q], nqz=cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
                gen_chs.append(self.qubit_chs[q])

        # -------add gaussian envelopes------
        self.initialize_waveforms()

        # define ramp pulses
        # self.add_gauss(ch=self.qubit_chs[qTest], name="ramp_up_ge", sigma=self.pi_sigma_ramp, length=self.pi_sigma_ramp*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="ramp_up_ef", sigma=self.pief_sigma_ramp, length=self.pief_sigma_ramp*4)
        # self.add_gauss(ch=self.qubit_chs[qTest], name="ramp_up_hge", sigma=self.hpi_sigma_ramp, length=self.hpi_sigma_ramp*4)

        # ---------- readout pulse parameters -----------
        self.set_pulse_registers(ch=self.res_chs[qTest], style="const", freq=self.f_res_reg[qTest], phase=self.deg2reg(
            cfg.device.readout.phase[qTest]), gain=cfg.device.readout.gain[qTest], length=self.readout_lengths_dac[qTest])

        # Useful pulse sequences
        self.parity_pulse = self.get_parity_str(1, return_pulse=True, second_phase=180)

        # self.wait_all(self.us2cycles(0.2))
        self.sync_all(self.us2cycles(0.2))


    def get_total_time(self, test_pulse, gate_based = False, cycles = False, cycles2us = 0.0023251488095238095):
        '''
        Takes in pulse str of form
        # [[frequency], [gain], [length (us)], [phases], [drive channel], [shape], [ramp sigma]]s
        '''
        if gate_based:
            test_pulse = self.get_prepulse_creator(test_pulse).pulse
        t = 0
        for i in range(len(test_pulse[0])):
            if test_pulse[5][i] == 'g' or test_pulse[5][i] == 'gauss' or test_pulse[5][i] == 'gaussian':
                t += test_pulse[-1][i] * 4
            elif test_pulse[5][i] == 'flat_top' or test_pulse[5][i] == 'f':
                t += test_pulse[-1][i] * 6 + test_pulse[2][i]
            t+= 0.01 # 10ns delay
        if cycles:
            # QickConfig(im[yaml_cfg['aliases']['soc']].get_cfg())
            return int(round(t / cycles2us))
        return t

    def initialize_waveforms(self):
        '''
        Initialize waveforms for ge, ef_new, f0g1 and sidebands
        '''
        qTest = 0

        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge", sigma=self.pi_ge_sigma, length=self.pi_ge_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit_ge", sigma=self.hpi_ge_sigma, length=self.hpi_ge_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ef", sigma=self.pi_ef_sigma, length=self.pi_ef_sigma*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit_ef", sigma=self.hpi_ef_sigma, length=self.hpi_ef_sigma*4)
        # add fast pi pulse
        self.add_gauss(ch=self.qubit_chs[qTest], name="hpi_qubit_ge_fast", sigma=self.hpi_ge_sigma_fast, length=self.hpi_ge_sigma_fast*4)
        self.add_gauss(ch=self.qubit_chs[qTest], name="pi_qubit_ge_fast", sigma=self.pi_ge_sigma_fast, length=self.pi_ge_sigma_fast*4)


        self.add_gauss(ch=self.flux_low_ch[qTest], name="pi_m1si_low", sigma=self.pi_m1_sigma_low, length=self.pi_m1_sigma_low*6)
        self.add_gauss(ch=self.flux_high_ch[qTest], name="pi_m1si_high", sigma=self.pi_m1_sigma_high, length=self.pi_m1_sigma_high*6)


    def measure_wrapper(self):
        """
        Aligns channels and performs a measurement on the first qubit specified in the experiment configuration.
        This method synchronizes all channels, then triggers a measurement pulse on the readout channel
        associated with the first qubit (`qTest`). The measurement is performed with the specified ADC channel,
        trigger offset, and a synchronization delay based on the configured relaxation delay.
        Steps:
            1. Selects the first qubit from the experiment configuration.
            2. Synchronizes all channels with a fixed delay. (This may not correct definition of sync)
            3. Initiates a measurement pulse with parameters:
                - Readout pulse channel for the selected qubit.
                - ADC channel for the selected qubit.
                - ADC trigger offset from device configuration.
                - Waits for measurement completion.
                - Synchronization delay based on relaxation delay.
        Returns:
            None
        """
        # align channels and measure
        qTest = self.cfg.expt.qubits[0]
        self.sync_all(10)
        self.measure(
            pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=self.cfg.device.readout.trig_offset[qTest],
            wait=True,
            syncdelay=self.us2cycles(self.cfg.device.readout.relax_delay[qTest])
        )


    def reset_and_sync(self):
        # Phase reset all channels except readout DACs

        # self.setup_and_pulse(ch=self.res_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.res_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.qubit_chs[qTest]s[0], style='const', freq=self.freq2reg(18, gen_ch=self.qubit_chs[qTest]s[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.man_chs[0], style='const', freq=self.freq2reg(18, gen_ch=self.man_chs[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_low_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_low_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.flux_high_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.flux_high_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.f0g1_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.f0g1_ch[0]), phase=0, gain=5, length=10, phrst=1)
        # self.setup_and_pulse(ch=self.storage_ch[0], style='const', freq=self.freq2reg(18, gen_ch=self.storage_ch[0]), phase=0, gain=5, length=10, phrst=1)
        cfg = self.cfg

        # dirty patch to cope with old code that doesn't run MM_base_initialize() in the initialize method...
        try:
            self.qubit_ch
        except AttributeError:
            self.parse_config()

        # some dummy variables
        qTest = 0
        self.f_cav = self.freq2reg(5000, gen_ch=self.man_ch[0])

        #initialize the phase to be 0
        self.set_pulse_registers(ch=self.qubit_ch[0], freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.qubit_ch[0])
        self.set_pulse_registers(ch=self.man_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.man_ch[0])
        # self.set_pulse_registers(ch=self.storage_ch[0], freq=self.f_cav,
        #                          phase=0, gain=0, length=10, style="const", phrst=1)
        # self.pulse(ch=self.storage_ch[0])
        self.set_pulse_registers(ch=self.flux_low_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_low_ch[0])
        self.set_pulse_registers(ch=self.flux_high_ch[0], freq=self.f_cav,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.flux_high_ch[0])
        self.set_pulse_registers(ch=self.f0g1_ch[0], freq=self.f_ge,
                                 phase=0, gain=0, length=10, style="const", phrst=1)
        self.pulse(ch=self.f0g1_ch[0])
        self.sync_all(10)

    def channel_table(self, channel_idx):

        channel_mapping = {
            0: self.f0g1_ch,
            1: self.flux_low_ch,
            2: self.qubit_ch,
            3: self.flux_high_ch,
            4: self.man_ch,
            6: self.storage_ch
        }
        ch = channel_mapping.get(channel_idx)
        if isinstance(ch, list):
            return ch[0]
        return ch

    def custom_pulse(self,
                     cfg, # not used but in order not to break old API
                     pulse_data: Optional[Union[List[List[float]], np.ndarray]]=None,
                     advance_qubit_phase: float=0,
                     sync_zero_const: bool=False,
                     waveform_preload: Optional[List[str]]=None,
                     prefix: str='pre'):
        '''
        Executes prepulse or postpulse
        pulse data:
            [[frequency], [gain], [length (us)], [phases],
            [drive channel], [shape], [ramp sigma]]
        where drive channel=
            1 (flux low), 2 (qubit), 3 (flux high),
            4 (storage),  0 (f0g1),  6 (manipulate)
        '''
        if pulse_data is None:
            return None

        sync_all_flag=True

        pulse_data[3] = [x + advance_qubit_phase for x in pulse_data[3]]

        if waveform_preload is not None:
            # check that the waveform length = pulse_data
            if len(waveform_preload) != len(pulse_data[0]):
                raise ValueError("Waveform preload length does not match pulse data length")
            # check that the pulse and only of the kind opt_cont
            else:
                if not all(isinstance(x, list) and x[0] == 'opt_cont' for x in pulse_data[5]):
                    raise ValueError("Waveform preload must be of kind 'opt_cont'")

        for jj in range(len(pulse_data[0])):

            self.tempch = pulse_data[4][jj]
            if type(self.tempch) == list:
                self.tempch = self.tempch[0]
            if pulse_data[5][jj] == "gaussian" or pulse_data[5][jj] == "gauss" or pulse_data[5][jj] == "g":
                self.pisigma_resolved = self.us2cycles(
                    pulse_data[6][jj], gen_ch=self.tempch)
                self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                    sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                # self.wait_all(self.us2cycles(0.01))
                self.sync_all(self.us2cycles(0.01))

                self.setup_and_pulse(ch=self.tempch, style="arb",
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                    gain=pulse_data[1][jj],
                                    waveform="temp_gaussian"+str(jj)+prefix)


            elif pulse_data[5][jj] == "flat_top" or pulse_data[5][jj] == "f":
                self.pisigma_resolved = self.us2cycles(
                    pulse_data[6][jj], gen_ch=self.tempch)
                if self.tempch==0 or self.tempch == 1 or self.tempch == 3: # for f0g1
                    self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                    sigma=self.pisigma_resolved, length=self.pisigma_resolved*6)
                else:
                    self.add_gauss(ch=self.tempch, name="temp_gaussian"+str(jj)+prefix,
                    sigma=self.pisigma_resolved, length=self.pisigma_resolved*4)
                self.sync_all(self.us2cycles(0.01))

                self.setup_and_pulse(ch=self.tempch, style="flat_top",
                                freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                gain=pulse_data[1][jj],
                                length=self.us2cycles(pulse_data[2][jj],
                                                    gen_ch=self.tempch),
                                waveform="temp_gaussian"+str(jj)+prefix)

            # check if pulse_data[5][jj] is a list or not
            elif isinstance(pulse_data[5][jj], list) and pulse_data[5][jj][0] == 'opt_cont':
                if waveform_preload is None:
                    encoding = pulse_data[5][jj][1] # fock, gkp, etc.
                    state = pulse_data[5][jj][2] # '0', '0+2' / 'Z', 'X', etc.

                    # load the corresponding waveform
                    filename = self.cfg.device.optimal_control[encoding][state]['filename']
                    # load only once the waveform
                    data = np.load(filename, allow_pickle=True)

                    if self.tempch == self.qubit_ch[0]:
                        times = data['times']
                        I = data[f'I_q']
                        Q = data[f'Q_q']
                        sync_all_flag = False # dont sync for qubit channel

                    elif self.tempch == self.man_ch[0]:
                        times = data['times']
                        I = data[f'I_c']
                        Q = data[f'Q_c']

                    # convert in us and MHz
                    times_us = times * 1e-3
                    I_mhz = I
                    Q_mhz = -Q

                    gencfg = self.soccfg["gens"][self.tempch]
                    maxv = gencfg["maxv"] * gencfg["maxv_scale"] - 1
                    samps_per_clk = gencfg["samps_per_clk"]


                    num_samps_tot = samps_per_clk * self.us2cycles(times_us[-1], gen_ch=self.tempch)
                    times_samps = np.arange(0, int(num_samps_tot))
                    times_samps_interp = np.linspace(0, num_samps_tot, len(times_us))


                    IQ_scale = max((np.max(np.abs(I_mhz)), np.max(np.abs(Q_mhz))))
                    if IQ_scale == 0:
                        # skip this pulse if the IQ scale is zero
                        continue
                    I_func = sp.interpolate.interp1d(times_samps_interp,
                                                    I_mhz / IQ_scale, kind="linear", fill_value=0)
                    Q_func = sp.interpolate.interp1d(times_samps_interp,
                                                    Q_mhz / IQ_scale, kind="linear", fill_value=0)
                    iamps = I_func(times_samps)
                    qamps = Q_func(times_samps)
                    # fig, ax = plt.subplots(1,1)
                    # ax.plot(times_samps, iamps, label='I')
                    # ax.plot(times_samps, qamps, label='Q')
                    # ax.legend()
                    # ax.set_xlabel('Time (us)')
                    # ax.set_ylabel('Amplitude (MHz)')
                    # ax.set_title('Optimal Control Pulse')
                    # fig.show()

                    waveform = encoding + '_' + state
                    if self.tempch == self.qubit_ch[0]:
                        waveform = encoding + '_' + state + '_q'
                    elif self.tempch == self.man_ch[0]:
                        waveform = encoding + '_' + state + '_m'

                    self.add_pulse(ch=self.tempch,
                                name=waveform,idata=maxv * iamps, qdata=maxv * qamps)

                    # if sync_all_flag:
                    #     self.sync_all(self.us2cycles(0.01))
                    # else:
                    #     sync_all_flag=True

                    self.setup_and_pulse(ch=self.tempch, style="arb",
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                    gain=pulse_data[1][jj],
                                    waveform=waveform)

                else:

                    if self.tempch == self.qubit_ch[0]:
                        # look for the waveform_preload ending by _qb
                        waveform = [w for w in waveform_preload if w.endswith('_qb')]
                        sync_all_flag = False # dont sync for qubit channels
                        if waveform:
                            waveform = waveform[0]
                        else:
                            raise ValueError("No waveform found for qubit channel in waveform_preload")
                    elif self.tempch == self.man_ch[0]:
                        # look for the waveform_preload ending by _man
                        waveform = [w for w in waveform_preload if w.endswith('_man')]
                        if waveform:
                            waveform = waveform[0]
                        else:
                            raise ValueError("No waveform found for manipulation channel in waveform_preload")

                    self.setup_and_pulse(ch=self.tempch, style="arb",
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                    gain=pulse_data[1][jj],
                                    waveform=waveform)


            else: # this is for parity measurement wait time, either wait or do constant pulse of 0 amplitude
                if sync_zero_const and pulse_data[1][jj] ==0:
                    self.sync_all(self.us2cycles(pulse_data[2][jj])) #,
                                                        #gen_ch=self.tempch))
                else:
                    self.setup_and_pulse(ch=self.tempch, style="const",
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                    gain=pulse_data[1][jj],
                                    length=self.us2cycles(pulse_data[2][jj],
                                                        gen_ch=self.tempch))
            # self.wait_all(self.us2cycles(0.01))
            if sync_all_flag:
                self.sync_all(self.us2cycles(0.01))
            else:
                sync_all_flag=True

    def custom_pulse_with_preloaded_wfm(self, cfg, pulse_data, advance_qubit_phase = None, sync_zero_const = False, prefix='pre',
                                        same_storage = False, same_qubit_pulse = False, storage_no=1):
        '''
        Executes prepulse or postpulse

        # [[frequency], [gain], [length (us)], [phases], [drive channel],
        #  [shape], [ramp sigma]],
        #  drive channel=1 (flux low),
        # 2 (qubit),3 (flux high),4 (storage),0 (f0g1),6 (manipulate),

        same_storage: if True, then the storage mode is not changed, we can reuse already prgrammed pulse
        '''
        if pulse_data is None:
            return None

        for jj in range(len(pulse_data[0])):
            # translate ch id to ch
            if pulse_data[4][jj] == 1:
                self.tempch = self.flux_low_ch
            elif pulse_data[4][jj] == 2:
                self.tempch = self.qubit_ch
            elif pulse_data[4][jj] == 3:
                self.tempch = self.flux_high_ch
            elif pulse_data[4][jj] == 6:
                self.tempch = self.storage_ch
            elif pulse_data[4][jj] == 0:   # used to be 5
                self.tempch = self.f0g1_ch
            elif pulse_data[4][jj] == 4:
                self.tempch = self.man_ch
            if type(self.tempch) == list:
                self.tempch = self.tempch[0]
            # determine the pulse shape

            waveform_name = None

            if pulse_data[5][jj] == "gaussian" or pulse_data[5][jj] == "gauss" or pulse_data[5][jj] == "g":
                # likely a qubit pulse on ge space with 35 ns sigma
                waveform_name = "pi_qubit_ge"
                if self.cfg.expt.preloaded_pulses and self.tempch == 2 and same_qubit_pulse:
                    self.pulse(ch=self.tempch)
                else:
                    self.setup_and_pulse(ch=self.tempch, style="arb",
                                freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                gain=pulse_data[1][jj],
                                waveform=waveform_name)

            elif pulse_data[5][jj] == "flat_top" or pulse_data[5][jj] == "f":
                if self.tempch == 0 :
                    waveform_name = "pi_f0g1"
                elif self.tempch == 1:
                    waveform_name = "pi_m1si_low"
                elif self.tempch == 3:
                    waveform_name = "pi_m1si_high"
                # elif self.tempch == 2:
                #     waveform_name = "pi_qubit_ef_ftop"

                # self.sync_all(self.us2cycles(0.01))
                if self.cfg.expt.preloaded_pulses and self.tempch == 0: # f0g1 resuse
                    self.safe_regwi(self.page_f0g1_phase, self.r_f0g1_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                    self.pulse(ch=self.tempch)

                elif self.cfg.expt.preloaded_pulses and self.tempch == (1 or 3) and same_storage: # storage reuse
                    if self.tempch == 1:
                        self.safe_regwi(self.page_flux_low_phase, self.r_flux_low_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                    else:
                        self.safe_regwi(self.page_flux_high_phase, self.r_flux_high_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                    self.pulse(ch=self.tempch)

                # elif self.cfg.expt.preloaded_pulses and self.tempch == 2: # qubit reuse
                #     self.safe_regwi(self.page_qubit_phase, self.r_qubit_phase, self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch))
                #     self.pulse(ch=self.tempch)
                else:
                    # using arb waveform for flat top pulse

                    if self.cfg.expt.use_arb_waveform:
                        if self.tempch == 0:  # f0g1
                            self.setup_and_pulse(ch=self.tempch, style="arb",
                                            freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                            phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                            gain=pulse_data[1][jj],
                                        waveform="pi_f0g1_arb")
                        else:  # M1-Si, need to specify storage number
                            self.setup_and_pulse(ch=self.tempch, style="arb",
                                                freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                                phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                                gain=pulse_data[1][jj],
                                            waveform="pi_m1s" + str(storage_no) + "_arb")
                    else:
                        # using standard flat top pulse
                        self.setup_and_pulse(ch=self.tempch, style="flat_top",
                                            freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                            phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                            gain=pulse_data[1][jj],
                                            length=self.us2cycles(pulse_data[2][jj],
                                                                gen_ch=self.tempch),
                                        waveform=waveform_name)
            else:
                if sync_zero_const and pulse_data[1][jj] ==0:
                    self.sync_all(self.us2cycles(pulse_data[2][jj])) #,
                                                        #gen_ch=self.tempch))
                else:
                    self.setup_and_pulse(ch=self.tempch, style="const",
                                    freq=self.freq2reg(pulse_data[0][jj], gen_ch=self.tempch),
                                    phase=self.deg2reg(pulse_data[3][jj], gen_ch=self.tempch),
                                    gain=pulse_data[1][jj],
                                    length=self.us2cycles(pulse_data[2][jj],
                                                        gen_ch=self.tempch))
            # self.wait_all(self.us2cycles(0.01))
            self.sync_all(self.us2cycles(0.01))

    def load_opt_ctrl_pulse(self, pulse_conf, IQ_table):
        '''
        Load IQ pulse from configuration
        '''
        # self.IQ_pulse = pulse_conf
        self.IQ_table = IQ_table

        pulse_param = self.get_prepulse_creator(pulse_conf).pulse
        # # we load the IQ waveform with the param
        qb_channel = pulse_param[4][0]
        cav_channel = pulse_param[4][1]


        qb_cfg = self.soccfg["gens"][qb_channel]
        qb_maxv = qb_cfg["maxv"] * qb_cfg["maxv_scale"] - 1
        qb_samps_per_clk = qb_cfg["samps_per_clk"]

        cav_cfg = self.soccfg["gens"][cav_channel]
        cav_maxv = cav_cfg["maxv"] * cav_cfg["maxv_scale"] - 1
        cav_samps_per_clk = cav_cfg["samps_per_clk"]


        times = IQ_table['times']
        Ic = IQ_table['I_c']
        Qc = IQ_table['Q_c']
        Iq = IQ_table['I_q']
        Qq = IQ_table['Q_q']

        qb_num_samps_tot = qb_samps_per_clk * self.us2cycles(times[-1], gen_ch=qb_channel)
        qb_times_samps = np.arange(0, int(qb_num_samps_tot))
        qb_times_samps_interp = np.linspace(0, qb_num_samps_tot, len(times))

        cav_num_samps_tot = cav_samps_per_clk * self.us2cycles(times[-1], gen_ch=cav_channel)
        cav_times_samps = np.arange(0, int(cav_num_samps_tot))
        cav_times_samps_interp = np.linspace(0, cav_num_samps_tot, len(times))

        Ic_func = sp.interpolate.interp1d(cav_times_samps_interp,Ic, kind="linear", fill_value=0)
        Qc_func = sp.interpolate.interp1d(cav_times_samps_interp,Qc, kind="linear", fill_value=0)
        Iq_func = sp.interpolate.interp1d(qb_times_samps_interp,Iq, kind="linear", fill_value=0)
        Qq_func = sp.interpolate.interp1d(qb_times_samps_interp,Qq, kind="linear", fill_value=0)

        ic_amps = Ic_func(cav_times_samps)
        qc_amps = Qc_func(cav_times_samps)
        iq_amps = Iq_func(qb_times_samps)
        qq_amps = Qq_func(qb_times_samps)

        encoding = pulse_conf[0][1]
        state = pulse_conf[0][2]

        waveform_cav = encoding + '_' + state + '_man'
        waveform_qb = encoding + '_' + state + '_qb'

        self.add_pulse(ch=cav_channel, name=waveform_cav,
                       idata=cav_maxv * ic_amps, qdata=cav_maxv * qc_amps)

        self.sync_all(self.us2cycles(0.01))

        self.add_pulse(ch=qb_channel, name=waveform_qb,
                       idata=qb_maxv * iq_amps, qdata=qb_maxv * qq_amps)

        self.sync_all(self.us2cycles(0.01))

        return [waveform_cav, waveform_qb]


    def man_reset(self, man_idx, chi_dressed = True ):
        '''
        Reset manipulate mode by swapping it to lossy mode

        chi_dressed: if man freq shifted due to pop in qubit e, f states.
        '''
        qTest = 0
        cfg=AttrDict(self.cfg)

        assert man_idx==1, "only supports resetting man1 now"

        M1D1_freq = self.dataset.get_freq('M1-D1')
        M1D1_gain = self.dataset.get_gain('M1-D1')
        N = 2 if chi_dressed else 0
        chi_ge = cfg.device.manipulate.chi_ge[qTest]
        chi_ef = cfg.device.manipulate.chi_ef[qTest]

        self.sideband_sigma_high = self.us2cycles(0.005, gen_ch=self.flux_high_ch[qTest])
        self.add_gauss(ch=self.flux_high_ch[qTest],
                       name="ramp_high",# + str(man_idx),
                       sigma=self.sideband_sigma_high,
                       length=self.sideband_sigma_high*4)
        # self.wait_all(self.us2cycles(0.1))
        self.sync_all(self.us2cycles(0.1))

        chis = [chi_ge, chi_ge+chi_ef] if chi_dressed else [0]
        ch = self.flux_high_ch[qTest]
        for n in range(0, N+1): # works when M1D1 freq goes down (chi<0, bare freq+chi*n)
            for chi in chis:
                freq_chi_shifted = M1D1_freq + (n * chi)
                self.set_pulse_registers(ch=ch,
                                        freq=self.freq2reg(freq_chi_shifted,gen_ch=ch),
                                        style="flat_top",
                                        phase=self.deg2reg(0),
                                        length=self.us2cycles(10, gen_ch=ch),
                                        gain=M1D1_gain,
                                        waveform="ramp_high" )
                self.pulse(ch=ch)
                self.sync_all(self.us2cycles(0.025))
        # self.wait_all(self.us2cycles(0.25))
        self.sync_all(self.us2cycles(2))


    def man_stor_swap(self, man_idx: int, stor_idx: int):
        '''
        Perform Swap (pi pulse only) between manipulate mode and storage mode
        '''
        sweep_pulse = [['storage', 'M'+ str(man_idx) + '-' + 'S' + str(stor_idx), 'pi', 0], ]
        creator = self.get_prepulse_creator(sweep_pulse)
        # self.sync_all(self.us2cycles(0.2))
        self.custom_pulse(self.cfg, creator.pulse, prefix='Storage' + str(stor_idx) + 'dump')
        self.sync_all(self.us2cycles(0.2)) # without this sideband rabi of storage mode 7 has kinks


    def coup_stor_swap(self, man_idx):
        '''
        Perform Swap between manipulate mode and  storage mode
        '''
        sweep_pulse = [['storage', 'M'+ str(man_idx) + '-' + 'C', 'pi', 0],
                       ]
        creator = self.get_prepulse_creator(sweep_pulse)
        # self.sync_all(self.us2cycles(0.2))
        self.custom_pulse(self.cfg, creator.pulse, prefix='Coupler')
        self.sync_all(self.us2cycles(0.2)) # without this sideband rabi of storage mode 7 has kinks

    def parity_active_reset(self, name='parity_reset', register_label='base', play_parity=True):
        '''
        Perform a parity measurement and if the qubit end up in e do a pi pulse reset
        '''

        # import the config and set qubit number, by default 0 since we have only one, but should be done better
        cfg=AttrDict(self.cfg)
        qTest = self.cfg.expt.qubits[0]
        self.r_read_q = 9
        self.r_thresh_q = 11 
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))

        # parity pulses, for now I will do something hacky, 
        # i.e. will only load the waveform once, should rewrite custom_pulse to be more general
        if play_parity: 
            parity_str = self.get_parity_str(1, return_pulse=True, second_phase=180, fast=False)
            self.custom_pulse(cfg, parity_str, prefix=name)

        # measurement
        self.measure(pulse_ch=self.res_chs[qTest],
            adcs=[self.adc_chs[qTest]],
            adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                t='auto', wait=True)
        # I dont exactly get why I need a wait instead of sync here, but ok, this is the minimal wait for read to be done    
        self.wait_all(self.us2cycles(0.10))        
        # # syntax is read(input_ch, page, upper/lower, reg) where lower is I, upper is Q
        self.read(0, 0, "lower", self.r_read_q) # stores I in (0,0) into r_read_q
        # # first if 
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   register_label)  # compare the value recorded above to the value stored in threshold.
        self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                 freq=self.f_ge_reg[qTest],
                                 style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_ge_gain,
                                 waveform='pi_qubit_ge')
        self.pulse(ch=self.qubit_chs[qTest])
        self.label(register_label)  # location to be jumped to
        print("needs a pretty long sync here I should be smatter with parity pulse timing")
        self.sync_all(self.us2cycles(1))



    def active_reset(self, man_reset = False, storage_reset = False, coupler_reset = False,
                      ef_reset = True, pre_selection_reset = True, prefix = 'base'):
        '''
        Performs active reset on g,e,f as well as man/storage modes
        Includes post selection measurement
        '''
        cfg = self.cfg
        qTest = 0

        # Prepare Active Reset
        ## ALL ACTIVE RESET REQUIREMENTS
        # read val definition
        self.r_read_q = 9  # ge active reset register
        self.r_read_q_ef = 10   # ef active reset register
        self.safe_regwi(0, self.r_read_q, 0)  # init read val to be 0
        self.safe_regwi(0, self.r_read_q_ef, 0)  # init read val to be 0

        # threshold definition
        self.r_thresh_q = 11  # Define a location to store the threshold info

        # # multiplication bc the readout is summed, so need common thing to compare to
        self.safe_regwi(0, self.r_thresh_q, int(cfg.device.readout.threshold[qTest] * self.readout_lengths_adc[qTest]))

        # Define a location to store a counter for how frequently the condj is triggered
        self.r_counter = 12
        self.safe_regwi(0, self.r_counter, 0)  # init counter val to 0

        # self.sync_all(self.us2cycles(0.2))
        self.sync_all(self.us2cycles(0.05))

        # First Reset Manipulate Modes
        # =====================================
        if man_reset:
            # self.man_reset(0)
            self.man_reset(1)

        # Reset ge level
        # ======================================================
        cfg=AttrDict(self.cfg)
        self.measure(pulse_ch=self.res_chs[qTest],
                    adcs=[self.adc_chs[qTest]],
                    adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                    #  t='auto', wait=True, syncdelay=self.us2cycles(2.0))
                     t='auto', wait=True)

        self.wait_all(self.us2cycles(0.25))  # to allow the read to be complete might be reduced

        self.read(0, 0, "lower", self.r_read_q)  # read data from I buffer, QA, and store
        # self.wait_all(self.us2cycles(0.05))  # to allow the read to be complete might be reduced
        self.sync_all(self.us2cycles(0.05)) # EG: this is not doing anything

        # perform Qubit active reset comparison, jump if condition is true to the label1 location
        self.condj(0, self.r_read_q, "<", self.r_thresh_q,
                   prefix + "LABEL_1")  # compare the value recorded above to the value stored in threshold.

        self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                 freq=self.f_ge_reg[qTest],
                                 style="arb",
                                 phase=self.deg2reg(0),
                                 gain=self.pi_ge_gain,
                                 waveform='pi_qubit_ge')
        self.pulse(ch=self.qubit_chs[qTest])
        self.label(prefix + "LABEL_1")  # location to be jumped to
        self.wait_all(self.us2cycles(0.05))
        self.sync_all(self.us2cycles(0.25))

        # Reset ef level
        # ======================================================
        if  ef_reset:
            # Map f level to e level
            self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                    freq=self.f_ef_reg[qTest],
                                    style="arb",
                                    phase=self.deg2reg(0),
                                    gain=self.pi_ef_gain,
                                    waveform='pi_qubit_ef')
            self.pulse(ch=self.qubit_chs[qTest])
            # self.wait_all(self.us2cycles(0.05))
            self.sync_all(self.us2cycles(0.05))
            self.measure(pulse_ch=self.res_chs[qTest],
                        adcs=[self.adc_chs[qTest]],
                        adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                        t='auto', wait=True, syncdelay=self.us2cycles(2))  # self.us2cycles(1))

            self.wait_all(self.us2cycles(0.2))  # to allow the read to be complete might be reduced

            self.read(0, 0, "lower", self.r_read_q_ef)  # read data from I buffer, QA, and store
            # self.wait_all(self.us2cycles(0.05))  # to allow the read to be complete might be reduced
            self.sync_all(self.us2cycles(0.05))

            # perform Qubit active reset comparison, jump if condition is true to the label1 location
            self.condj(0, self.r_read_q_ef, "<", self.r_thresh_q,
                    prefix + "LABEL_2")  # compare the value recorded above to the value stored in threshold.

            #play pi pulse if condition is false (ie, if qubit is in excited state), to pulse back to ground.
            self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                    freq=self.f_ge_reg[qTest],
                                    style="arb",
                                    phase=self.deg2reg(0),
                                    gain=self.pi_ge_gain,
                                    waveform='pi_qubit_ge')
            self.pulse(ch=self.qubit_chs[qTest])
            self.label(prefix + "LABEL_2")  # location to be jumped to
            # self.wait_all(self.us2cycles(0.05))
            self.sync_all(self.us2cycles(0.25))

        # Dump storage population to manipulate, then to lossy mode
        # ======================================================
        if storage_reset:
            for ii in range(7):
                man_idx = 0
                stor_idx = ii
                self.man_stor_swap(man_idx=man_idx+1, stor_idx=stor_idx+1) #self.man_stor_swap(1, ii+1)
                # self.man_reset(0, chi_dressed = False)
                self.man_reset(1, chi_dressed = False)

        if coupler_reset:
            self.coup_stor_swap(man_idx=1) # M1
            self.man_reset(0, chi_dressed = False)
            self.man_reset(1, chi_dressed = False)

        # post selection
        # ======================================================
        if pre_selection_reset:
            self.sync_all(self.us2cycles(self.cfg.device.active_reset.relax_delay[0]))
            self.measure(pulse_ch=self.res_chs[qTest],
                        adcs=[self.adc_chs[qTest]],
                        adc_trig_offset=cfg.device.readout.trig_offset[qTest],
                        t='auto', wait=True, syncdelay=self.us2cycles(2.0))  # self.us2cycles(1))
            # self.sync_all(self.us2cycles(self.cfg.device.active_reset.relax_delay[0]))
            self.sync_all(self.us2cycles(0.2))


    def get_parity_str(self, man_mode_no, return_pulse=False, second_phase = 0, fast = False):
        '''
        Create parity pulse
        '''
        parity_str = [['qubit', 'ge', 'hpi', 0],
                    ['qubit', 'ge', 'parity_M' + str(man_mode_no), 0],
                    ['qubit', 'ge', 'hpi', second_phase]]
        if fast:
            # take the AC stark phase
            freq_AC = self.cfg.device.manipulate.revival_stark_shift[man_mode_no-1]
            revival_time = self.cfg.device.manipulate.revival_time[man_mode_no-1]
            theta_2 = second_phase + 2*np.pi*freq_AC * revival_time * 180/np.pi
            theta_2 = theta_2 % 360

            parity_str = [
                    ['multiphoton', 'g0-e0', 'hpi', 0],
                    ['qubit', 'ge', 'parity_M' + str(man_mode_no), 0],
                    ['multiphoton', 'g0-e0', 'hpi', theta_2]
                    ]
        if return_pulse:
            # mm_base = MM_rb_base(cfg = self.cfg)
            creator = self.get_prepulse_creator(parity_str)
            return creator.pulse.tolist()

        return parity_str


    def play_parity_pulse(self, man_mode_no, qubit_no=0, second_phase=180, fast=False):
        '''
        Play parity pulse
        '''
        cfg=AttrDict(self.cfg)
        qTest = self.cfg.expt.qubits[qubit_no] 
        revival_time = cfg.device.manipulate.revival_time[man_mode_no]
        revival_cycles = self.us2cycles(revival_time)
        reg_page = self.ch_page(self.qubit_chs[qTest])
        reg_phase =self.sreg(self.qubit_chs[qTest], "phase")
        if fast: 
            freq_pi = self.f_ge_hpi_fast
            gain_pi = self.hpi_ge_gain_fast
            waveform_pi = 'hpi_qubit_ge_fast'
            freq_AC = self.cfg.device.manipulate.revival_stark_shift[man_mode_no]
            theta_2 = second_phase + 2*np.pi*freq_AC * revival_time * 180/np.pi
            theta_2 = theta_2 % 360
        else:
            freq_pi = self.f_ge
            gain_pi = self.hpi_ge_gain
            theta_2 = second_phase
            waveform_pi = 'hpi_qubit_ge'

        #first pi/2 pulse
        self.set_pulse_registers(ch=self.qubit_chs[qTest],
                                 freq=freq_pi,
                                 style="arb",
                                 phase=self.deg2reg(0),
                                 gain=gain_pi,
                                 waveform=waveform_pi)
        self.pulse(ch=self.qubit_chs[qTest])
        self.sync_all()
        # wait based on revival time 
        self.sync_all(revival_cycles)
        # second pi/2 pulse, if fast take into account AC stark phase
        # here we can just update the phase of the waveform
        self.safe_regwi(reg_page, reg_phase, self.deg2reg(theta_2))
        self.pulse(ch=self.qubit_chs[qTest])


    def get_gain_optimal_pulse(self, pulse=None, pulse_IQ=None, plot=False):
        if pulse is not None:
            encoding = pulse[1]
            state = pulse[2]
        # open the file with the optimal pulse
            filename = self.cfg.device.optimal_control[encoding][state]['filename']
            data = np.load(filename, allow_pickle=True)
            I_q = data['I_q']*2*np.pi*1e3
            Q_q = data['Q_q']*2*np.pi*1e3
            max_q = max(np.max(np.abs(I_q)), np.max(np.abs(Q_q)))
            I_c = data['I_c']*2*np.pi*1e3
            Q_c = data['Q_c']*2*np.pi*1e3
            max_c = max(np.max(np.abs(I_c)), np.max(np.abs(Q_c)))
        elif pulse_IQ is not None:
            if 'I_q' in pulse_IQ.keys():
                I_q = pulse_IQ['I_q']*2*np.pi*1e3
                Q_q = pulse_IQ['Q_q']*2*np.pi*1e3
                I_c = pulse_IQ['I_c']*2*np.pi*1e3
                Q_c = pulse_IQ['Q_c']*2*np.pi*1e3
            elif 'Iq' in pulse_IQ.keys():
                I_q = pulse_IQ['Iq']*2*np.pi*1e3
                Q_q = pulse_IQ['Qq']*2*np.pi*1e3
                I_c = pulse_IQ['Ic']*2*np.pi*1e3
                Q_c = pulse_IQ['Qc']*2*np.pi*1e3
            else:
                raise ValueError("pulse_IQ must contain keys 'I_q', 'Q_q', 'I_c', 'Q_c' or 'Iq', 'Qq', 'Ic', 'Qc'")
            max_q = max(np.max(np.abs(I_q)), np.max(np.abs(Q_q)))
            max_c = max(np.max(np.abs(I_c)), np.max(np.abs(Q_c)))
        else:
            raise ValueError("Either pulse or pulse_IQ must be provided")
        
        if plot: 
            times = data['times']
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            axs[0].plot(times, I_q, label='I_q')
            axs[0].plot(times, Q_q, label='Q_q')
            axs[0].set_title('Qubit Pulse')
            axs[0].legend()
            axs[1].plot(times, I_c, label='I_c')
            axs[1].plot(times, Q_c, label='Q_c')
            axs[1].set_title('Cavity Pulse')
            axs[1].legend()
            plt.tight_layout()
            plt.show()

        # import pi pulse parameters
        gain_qb_pi_pulse = self.cfg.device.qubit.pulses.pi_ge.gain[0]
        sigma_qb_pi_pulse = self.cfg.device.qubit.pulses.pi_ge.sigma[0]
        pi_pulse_type = self.cfg.device.qubit.pulses.pi_ge.type[0]
        assert pi_pulse_type == 'gauss', "Only gaussian pi pulse is supported for now"
        n=4 # assume the pulse is 4 sigma long
        theta_to_gain = np.pi/2/gain_qb_pi_pulse
        drive_to_gain_qb = sigma_qb_pi_pulse * np.sqrt(np.pi)/theta_to_gain * sp.special.erf(n/2)
        # do the same for the manipulate pulse
        alpha_to_gain = self.cfg.device.manipulate.gain_to_alpha[0]
        sigma_cav = self.cfg.device.manipulate.displace_sigma[0]
        drive_to_gain_cav = sigma_cav * np.sqrt(np.pi)/alpha_to_gain * sp.special.erf(n/2)

        gain_qb = round(max_q * drive_to_gain_qb, 0)
        gain_qb = int(gain_qb)
        gain_cav = round(max_c * drive_to_gain_cav, 0)
        gain_cav = int(gain_cav)
        return gain_qb, gain_cav

    def collect_shots(self):
        # collect shots for the relevant adc and I and Q channels
        qTest = 0
        cfg=AttrDict(self.cfg)
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[qTest]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[qTest]
        return shots_i0, shots_q0

    # def post_select_histogram(self):


    # --------------------------------- Single shot analysis code  ---------------------------------
    # hmm do these really belong here or in a separate single shot file?
    @staticmethod
    def filter_data_IQ(II, IQ, threshold, readout_per_experiment=2):
        # assume the last one is experiment data, the last but one is for post selection
        result_Ig = []
        result_Ie = []

        for k in range(len(II) // readout_per_experiment):
            index_4k_plus_2 = readout_per_experiment * k + readout_per_experiment-2
            index_4k_plus_3 = readout_per_experiment * k + readout_per_experiment-1

            # Ensure the indices are within the list bounds
            if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
                # Check if the value at 4k+2 exceeds the threshold
                if II[index_4k_plus_2] < threshold:
                    # Add the value at 4k+3 to the result list
                    result_Ig.append(II[index_4k_plus_3])
                    result_Ie.append(IQ[index_4k_plus_3])

        return np.array(result_Ig), np.array(result_Ie)

    def hist(self, data, plot=False, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4.3):
        """
        span: histogram limit is the mean +/- span
        """
        if active_reset:
            Ig, Qg = self.filter_data_IQ(data['Ig'], data['Qg'], threshold, readout_per_experiment=readout_per_round)
            # Qg = filter_data(data['Qg'], threshold, readout_per_experiment=readout_per_round)
            Ie, Qe = self.filter_data_IQ(data['Ie'], data['Qe'], threshold, readout_per_experiment=readout_per_round)
            # Qe = filter_data(data['Qe'], threshold, readout_per_experiment=readout_per_round)
            plot_f = False
            if 'If' in data.keys():
                plot_f = True
                If, Qf = self.filter_data_IQ(data['If'], data['Qf'], threshold, readout_per_experiment=readout_per_round)
        else:
            Ig = data['Ig']
            Qg = data['Qg']
            Ie = data['Ie']
            Qe = data['Qe']
            plot_f = False
            if 'If' in data.keys():
                plot_f = True
                If = data['If']
                Qf = data['Qf']

        numbins = 200

        xg, yg = np.median(Ig), np.median(Qg)
        xe, ye = np.median(Ie), np.median(Qe)
        if plot_f: xf, yf = np.median(If), np.median(Qf)

        if verbose:
            print('Unrotated:')
            print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
            print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
            if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

        if plot:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
            fig.tight_layout()

            axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.', s=1)
            axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.', s=1)

            if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.', s=1)
            axs[0,0].scatter(xg, yg, color='k', marker='o')
            axs[0,0].scatter(xe, ye, color='k', marker='o')
            if plot_f: axs[0,0].scatter(xf, yf, color='k', marker='o')

            axs[0,0].set_xlabel('I [ADC levels]')
            axs[0,0].set_ylabel('Q [ADC levels]')
            axs[0,0].legend(loc='upper right')
            axs[0,0].set_title('Unrotated')
            axs[0,0].axis('equal')

        """Compute the rotation angle"""
        theta = -np.arctan2((ye-yg),(xe-xg))
        if plot_f: theta = -np.arctan2((ye-yf),(xe-xf))

        """Rotate the IQ data"""
        Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
        Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta)

        Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
        Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

        if plot_f:
            If_new = If*np.cos(theta) - Qf*np.sin(theta)
            Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(Ig_new), np.median(Qg_new)
        xe, ye = np.median(Ie_new), np.median(Qe_new)
        if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
        if verbose:
            print('Rotated:')
            print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
            print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
            if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

        if span is None:
            span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
        midpoint = (np.max(np.concatenate((Ie_new, Ig_new))) + np.min(np.concatenate((Ie_new, Ig_new))))/2
        xlims = [midpoint-span, midpoint+span]
        ylims = [yg-span, yg+span]

        if plot:
            axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.', s=1)
            axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.', s=1)
            if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.', s=1)
            axs[0,1].scatter(xg, yg, color='k', marker='o')
            axs[0,1].scatter(xe, ye, color='k', marker='o')
            if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')

            axs[0,1].set_xlabel('I [ADC levels]')
            axs[0,1].legend(loc='upper right')
            axs[0,1].set_title('Rotated')
            axs[0,1].axis('equal')

            """X and Y ranges for histogram"""

            ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5, density=True)
            ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5, density=True)
            if plot_f:
                nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5, density=True)
            axs[1,0].set_ylabel('Counts')
            axs[1,0].set_xlabel('I [ADC levels]')
            axs[1,0].legend(loc='upper right')

        else:
            ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
            ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
            if plot_f:
                nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

        """Compute the fidelity using overlap of the histograms"""
        fids = []
        thresholds = []
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        confusion_matrix = [np.cumsum(ng)[tind]/ng.sum(),
                            1-np.cumsum(ng)[tind]/ng.sum(),
                            np.cumsum(ne)[tind]/ne.sum(),
                            1-np.cumsum(ne)[tind]/ne.sum()]   # Pgg (prepare g measured g), Pge (prepare g measured e), Peg, Pee
        if plot_f:
            contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
            tind=contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])

            contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
            tind=contrast.argmax()
            thresholds.append(binsg[tind])
            fids.append(contrast[tind])

        if plot:
            axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
            axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
            if plot_f:
                axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
                axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

            axs[1,1].set_title('Cumulative Counts')
            axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
            axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
            axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
            if plot_f:
                axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
                axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
                axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
            axs[1,1].legend()
            axs[1,1].set_xlabel('I [ADC levels]')

            plt.subplots_adjust(hspace=0.25, wspace=0.15)
            plt.show()

        return fids, thresholds, theta*180/np.pi, confusion_matrix # fids: ge, gf, ef

    # g states for q0


class MMAveragerProgram(AveragerProgram, MM_base):
    def __init__(self, soccfg, cfg):
        AveragerProgram.__init__(self, soccfg, cfg)

    def acquire(self, soc, threshold=None, load_pulses=False, progress=False, debug=False, readouts_per_experiment = 1):
        """
        Acquire data from the device, applying the necessary pulses and post-processing.

        note the soc object is proxy soc not QIckConfig soc
        """
        return super().acquire(soc=soc, threshold=threshold, load_pulses=load_pulses, progress=progress,
                       readouts_per_experiment=readouts_per_experiment)


class MMRAveragerProgram(RAveragerProgram, MM_base):
    def __init__(self, soccfg, cfg):
        RAveragerProgram.__init__(self, soccfg, cfg)

    def acquire(self, soc, threshold=None, load_pulses=False, progress=False, debug=False, readouts_per_experiment = 1):
        """
        Acquire data from the device, applying the necessary pulses and post-processing.

        note the soc object is proxy soc not QIckConfig soc
        """
        return super().acquire(soc=soc, threshold=threshold,load_pulses=load_pulses,
                               progress=progress, readouts_per_experiment=readouts_per_experiment)


# prepulse_creator2(self.cfg, self.cfg.device.storage.storage_man_file, multiphoton_cfg)

class prepulse_creator2:
    def __init__(self, cfg, storage_man_file, floquet_man_stor_file=None):
        '''
        Takes pulse param of form
            [name of transition of cavity name like 'ge', 'ef' or 'M1', 'M1-S1',
            name of pulse like pi, hpi, or parity_M1 or parity_M2,
            phase  (int form )]

        Creates pulses of the form
            [[frequency], [gain], [length (us)], [phases],
            [drive channel], [shape], [ramp sigma]]
        where drive channel should be looked up in the hardware config file
            1 (flux low & high), 2 (qubit), 3 (manipulate),
            4 (storage),  0 (f0g1), as of fall 2025
        '''
        # config
        # with open(config_file, 'r') as cfg_file:
        #     yaml_cfg = yaml.safe_load(cfg_file)
        self.cfg = cfg#AttrDict(yaml_cfg)

        # print(
        #     "multiphoton_cfg", multiphoton_cfg.pulses["pi_g0-e0"]['frequency'] if multiphoton_cfg else "No multiphoton config provided"
        # )

        # man storage swap data
        self.dataset = StorageManSwapDataset(storage_man_file)

        # man storage floquet swap data
        self.dataset_floquet = None
        if floquet_man_stor_file is not None:
            self.dataset_floquet = FloquetStorageSwapDataset(floquet_man_stor_file)

        # initialize pulse
        self.pulse = np.array([[],[],[],[],[],[],[]], dtype = object)

    def flush(self):
        '''re initializes to empty array'''
        self.pulse = np.array([[],[],[],[],[],[],[]], dtype = object)

    def append(self, pulse):
        self.pulse = np.concatenate((self.pulse, pulse), axis=1)
        return None

    def channel_assign(self, transition_name):
        ''' if transition name contains g,e then use qubit channel self.cfg hw.soc.dacs.qubit.ch
        if transition name contains f,g use sideband channel hw.soc.dacs.sideband.ch '''
        if ("g" in transition_name and "e" in transition_name) or ("e" in transition_name and "f" in transition_name):
            ch = self.cfg.hw.soc.dacs.qubit.ch
            if type(ch) is list :
                ch = ch[0]  # use the first qubit channel if multiple are defined
            return int(ch)

        elif "f" in transition_name and "g" in transition_name:
            ch =  self.cfg.hw.soc.dacs.sideband.ch
            if type(ch) is list:
                ch = ch[0]  # use the first qubit channel if multiple are defined
            return int(ch)
        else:
            raise ValueError(f"Unknown transition name: {transition_name}")


    def multiphoton(self, pulse_param):
        ''' pulse name comes from yaml file '''
        transition_name, pulse_name, phase = pulse_param
        # frequency
        # pulse_full_name = pulse_name + '_' + transition_name
        # cfg = self.multiphoton_cfg

        # qubit_pulse = np.array([[cfg.pulses[pulse_full_name]['frequency']],
        #             [cfg.pulses[pulse_full_name]['gain']],
        #             [cfg.pulses[pulse_full_name]['length']],
        #             [phase],
        #             [self.channel_assign(transition_name)],
        #             [cfg.pulses[pulse_full_name]['type']],
        #             [cfg.pulses[pulse_full_name]['sigma']]], dtype = object)
        # self.pulse = np.concatenate((self.pulse, qubit_pulse), axis=1)

        cfg = self.cfg

        state_start = transition_name[0]
        state_end = transition_name[3]
        photon_no_start = int(transition_name[1])
        if state_start =='g':
            assert state_end == 'e', "transition name should be gn-en"
            transition = 'gn-en'
        elif state_start == 'e':
            assert state_end == 'f', "transition name should be en-fn"
            transition = 'en-fn'
        elif state_start == 'f':
            assert state_end == 'g', "transition name should be fn-gn+1"
            transition = 'fn-gn+1'
            # if photon_no_start == 0:
            #     pulse_param = ('M1', pulse_name, phase)
            #     return self.man(pulse_param)


        qubit_pulse = np.array([[cfg.device.multiphoton[pulse_name][transition]['frequency'][photon_no_start]],
                               [cfg.device.multiphoton[pulse_name][transition]['gain'][photon_no_start]],
                               [cfg.device.multiphoton[pulse_name][transition]['length'][photon_no_start]],
                               [phase],
                               [self.channel_assign(transition_name)],
                               [cfg.device.multiphoton[pulse_name][transition]['type'][photon_no_start]],
                               [cfg.device.multiphoton[pulse_name][transition]['sigma'][photon_no_start]]], dtype = object)


        self.pulse = np.concatenate((self.pulse, qubit_pulse), axis=1)
        return None


    def optimal_control(self, pulse_param):

        encoding, state, channel_idx = pulse_param
        cfg = self.cfg

        # qubit pulse
        pulse = np.array([[cfg.device.optimal_control[encoding][state]['frequency'][0]],
                        [cfg.device.optimal_control[encoding][state]['gain'][0]],
                        [0],
                        [cfg.device.optimal_control[encoding][state]['phase'][0]],
                        [int(self.cfg.hw.soc.dacs.qubit.ch[channel_idx[0]])],
                        [['opt_cont', encoding, state]],
                        [0]], dtype = object)

        self.pulse = np.concatenate((self.pulse, pulse), axis=1)

        # pulse manipulate
        pulse = np.array([[cfg.device.optimal_control[encoding][state]['frequency'][1]],
                        [cfg.device.optimal_control[encoding][state]['gain'][1]],
                        [0],
                        [cfg.device.optimal_control[encoding][state]['phase'][1]],
                        # int(self.cfg.hw.soc.dacs.manipulate_in.ch[channel_idx[1]]),
                        [int(self.cfg.hw.soc.dacs.manipulate_in.ch[channel_idx[0]])],
                        [['opt_cont', encoding, state]],
                        [0]], dtype = object)

        self.pulse = np.concatenate((self.pulse, pulse), axis=1)

        return None


    def qubit(self, pulse_param): #(self, transition_name, pulse_name, man_idx = 0):
        ''' pulse name comes from yaml file '''
        transition_name, pulse_name, phase = pulse_param
        # frequency
        if transition_name[:2] == 'ge':
            freq = self.cfg.device.qubit.f_ge[0]
        else:
            freq = self.cfg.device.qubit.f_ef[0]

        if pulse_name[:6] != 'parity':
            pulse_full_name = pulse_name + '_' + transition_name # like pi_ge or pi_ef or pi_ge_new or pi_ef_new

            qubit_pulse = np.array([[freq],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['gain'][0]],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['length'][0]],
                    [phase],
                    [2],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['type'][0]],
                    [self.cfg.device.qubit.pulses[pulse_full_name]['sigma'][0]]], dtype = object)

            self.pulse = np.concatenate((self.pulse, qubit_pulse), axis=1)

        else: # parity string is 'parity_M1' or 'parity_M2'
            man_idx = int(pulse_name[-1:]) -1 # 1 for man1, 2 for man2

            if self.cfg.device.manipulate.revival_time[man_idx] != 0: # if revival time is not zero, then use it
                qubit_pulse = np.array([[freq],
                        [0],
                        [self.cfg.device.manipulate.revival_time[man_idx] ], # parity delay experiment doesn't involve 10 ns syncs
                        [phase],
                        [2],
                        ['const'],
                        [0.0]], dtype = object)
                self.pulse = np.concatenate((self.pulse, qubit_pulse), axis=1)
        return None


    def man(self, pulse_param):
        '''name can be pi or hpi
        man_idx is not irrelvant
        '''
        # connie: this whole function should really be calling multiphoton with f0-g1
        cav_name, pulse_name, phase = pulse_param

        if pulse_name == 'pi':
            length = self.dataset.get_pi(cav_name)
        else:
            length = self.dataset.get_h_pi(cav_name)

        f0g1  = np.array([[self.dataset.get_freq(cav_name)],
                [ self.dataset.get_gain(cav_name)],
                [length],
                [phase],
                [0], # f0g1 pulse
                ['flat_top'],
                [self.cfg.device.manipulate.ramp_sigma]], dtype = object)

        self.pulse = np.concatenate((self.pulse, f0g1), axis=1)

        return None

    def buffer(self, pulse_param):
        '''here the last parameter is time  (NOT THE MODE; this is just empty pulse; buffer tyime between gates )'''
        buffer = np.array([[0],
                [0],
                [pulse_param[-1]],
                [0],
                [1],
                ['const'],
                [self.cfg.device.storage.ramp_sigma]], dtype = object)
        self.pulse = np.concatenate((self.pulse, buffer), axis=1)
        return None

    def storage(self, pulse_param):
        '''
        plays sideband pulse on storage via coupler rf flux
        name can be pi or hpi'''
        stor_name, pulse_name, phase = pulse_param

        if pulse_name == 'pi':
            length = self.dataset.get_pi(stor_name)
        else:
            length = self.dataset.get_h_pi(stor_name)
        freq = self.dataset.get_freq(stor_name)
        flux_low_ch = self.cfg.hw.soc.dacs.flux_low.ch[0]
        flux_high_ch = self.cfg.hw.soc.dacs.flux_high.ch[0]
        ch = flux_low_ch if freq<1000 else flux_high_ch

        storage_pulse = np.array([[self.dataset.get_freq(stor_name)],
                [self.dataset.get_gain(stor_name)],
                [length],
                [phase],
                [ch],
                ['flat_top'],
                [self.cfg.device.storage.ramp_sigma]], dtype = object)

        self.pulse = np.concatenate((self.pulse, storage_pulse), axis=1)
        return None

    def floquet(self, pulse_param):
        stor_name, pi_frac, phase = pulse_param
        length = self.dataset_floquet.get_len(stor_name)
        freq = self.dataset_floquet.get_freq(stor_name)
        flux_low_ch = self.cfg.hw.soc.dacs.flux_low.ch[0]
        flux_high_ch = self.cfg.hw.soc.dacs.flux_high.ch[0]
        ch = flux_low_ch if freq<1000 else flux_high_ch

        storage_pulse = np.array([
            [self.dataset_floquet.get_freq(stor_name)],
            [self.dataset_floquet.get_gain(stor_name)],
            [length],
            [phase],
            [ch],
            ['flat_top'],
            [self.dataset_floquet.get_ramp_sigma(stor_name)]
            ], dtype = object)

        self.pulse = np.concatenate((self.pulse, storage_pulse), axis=1)
        return None

    def wait(self, pulse_param):
        wait_len = float(pulse_param[0])

        wait = np.array([[self.cfg.device.manipulate.f_ge[0]], #dummy freq
                [0], #dummy gain
                [wait_len], #wait len
                [0.0], #dummy phase
                [self.cfg.hw.soc.dacs.manipulate_in.ch[0]], #man channel
                ['const'],
                [0]], dtype = object)
        self.pulse = np.concatenate((self.pulse, wait), axis=1)
        return None
