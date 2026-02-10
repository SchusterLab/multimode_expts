from slab import Experiment, AttrDict
from experiments.MM_base import MMAveragerProgram, MM_base
from fitting.fit_display_classes import Histogram

# TODO: add gef analysis and rotation/threshold for g vs f

class HistogramProgram(MMAveragerProgram):
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps

        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()

        self.sync_all(200)  # not sure if this is needed

    # def set_gen_delays(self):
    #     for ch in self.gen_chs:
    #         delay_ns = self.cfg.hw.soc.dacs.delay_chs.delay_ns[
    #             np.argwhere(np.array(self.cfg.hw.soc.dacs.delay_chs.ch) == ch)[0][0]
    #         ]
    #         delay_cycles = self.us2cycles(delay_ns * 1e-3, gen_ch=ch)
    #         self.gen_delays[ch] = delay_cycles
    # def sync_all(self, t=0, gen_t0=None):
    #     if gen_t0 is None:
    #         gen_t0 = self.gen_delays
    #     super().sync_all(t=t, gen_t0=gen_t0)

    def collect_shots(self, read_num=1):
        """
        Collect raw shots from buffer. Returns ALL measurements (active reset + final).
        The Histogram class will handle filtering via filter_data_IQ() using readout_per_round.
        """
        qTest = 0

        # Return raw buffer - do NOT slice/filter here
        # Histogram.filter_data_IQ() expects the full buffer with all measurements
        shots_i0 = self.di_buf[0] / self.readout_lengths_adc[qTest]
        shots_q0 = self.dq_buf[0] / self.readout_lengths_adc[qTest]

        return shots_i0, shots_q0

    def body(self):
        cfg = AttrDict(self.cfg)
        qTest = 0

        # initializations as necessary
        self.reset_and_sync()
        # self.sync_all()

        # do the active reset
        if cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        
        # Prepulse
        if cfg.expt.prepulse:
            if cfg.expt.gate_based:
                creator = self.get_prepulse_creator(cfg.expt.pre_sweep_pulse)
                self.custom_pulse(cfg, creator.pulse.tolist(), prefix="pre_")
            else:
                self.custom_pulse(cfg, cfg.expt.pre_sweep_pulse, prefix="pre_")



        if self.cfg.expt.pulse_e or self.cfg.expt.pulse_f:
            self.setup_and_pulse(
                ch=self.qubit_chs[0],
                style="arb",
                freq=self.f_ge_reg[qTest],
                phase=0,
                gain=self.pi_ge_gain,
                waveform="pi_qubit_ge",
            )

        self.sync_all()
        self.wait_all(self.us2cycles(0.01))

        if self.cfg.expt.pulse_f:
            self.setup_and_pulse(
                ch=self.qubit_chs[qTest],
                style="arb",
                freq=self.f_ef_reg[qTest],
                phase=0,
                gain=self.pi_ef_gain,
                waveform="pi_qubit_ef",
            )
        self.sync_all()
        self.wait_all(self.us2cycles(0.01))

        self.measure_wrapper()


class HistogramExperiment(Experiment):
    """
    Histogram Experiment
    expt = dict(
        reps: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(
        self, soccfg=None, path="", prefix="Histogram", config_file=None, progress=None
    ):
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubits[0]
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not (isinstance(value3, list)):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not (isinstance(value, list)):
                    subcfg.update({key: [value] * num_qubits_sample})
        data = dict()

        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        # Ground state shots
        cfg = self.cfg
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False
        histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
        avgi, avgq = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            readouts_per_experiment=read_num,
        )
        data["Ig"], data["Qg"] = histpro.collect_shots()

        # Excited state shots
        if "check_e" not in self.cfg.expt:
            self.check_e = True
        else:
            self.check_e = self.cfg.expt.check_e
        if self.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                readouts_per_experiment=read_num,
            )
            data["Ie"], data["Qe"] = histpro.collect_shots()
        self.prog = histpro

        # Excited state shots
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = True
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
                debug=debug,
                readouts_per_experiment=read_num,
            )
            data["If"], data["Qf"] = histpro.collect_shots()

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=True, **kwargs):
        if data is None:
            data = self.data

        # Calculate read_num to pass to Histogram for proper filtering
        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        hist_fitter = Histogram(
            data=data,
            span=span,
            verbose=verbose,
            config=self.cfg,
            station=kwargs.pop("station", None),
            readout_per_round=read_num,
        )
        hist_fitter.analyze(plot=False, subdir=kwargs.pop("subdir", None))

        data["fids"] = hist_fitter.results["fids"]
        data["angle"] = hist_fitter.results["angle"]
        data["thresholds"] = hist_fitter.results["thresholds"]
        data["confusion_matrix"] = hist_fitter.results["confusion_matrix"]

        return data

    def display(
        self, station=None, data=None, span=None, verbose=True, plot_e=True, plot_f=False, **kwargs
    ):
        if data is None:
            data = self.data

        # Calculate read_num to pass to Histogram for proper filtering
        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        hist_fitter = Histogram(
            data=data, span=span, verbose=verbose, config=self.cfg, station=station,
            readout_per_round=read_num,
        )
        hist_fitter.analyze(plot=True)

        fids = hist_fitter.results["fids"]
        thresholds = hist_fitter.results["thresholds"]
        angle = hist_fitter.results["angle"]

        print(f"ge fidelity (%): {100*fids[0]}")
        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f"gf fidelity (%): {100*fids[1]}")
            print(f"ef fidelity (%): {100*fids[2]}")
        print(f"rotation angle (deg): {angle}")
        # print(f'set angle to (deg): {-angle}')
        print(f"threshold ge: {thresholds[0]}")
        if self.cfg.expt.check_f:
            print(f"threshold gf: {thresholds[1]}")
            print(f"threshold ef: {thresholds[2]}")

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
