"""
SingleShotGainSweepExperiment: Sweep readout gain while measuring single-shot fidelity.

This experiment internally loops over readout gain values, running HistogramProgram
(g and e state shots) at each gain point and analyzing fidelity via the Histogram class.

Designed to be used with SweepRunner for 2D optimization:
- Gain: swept inside this experiment's acquire() loop
- Frequency: swept by SweepRunner (sweep_param='readout_freq')
- Length: swept in the notebook (outer Python loop)

Usage (standalone):
    expt = SingleShotGainSweepExperiment(soccfg=soc, path=data_path, config_file=config_file)
    expt.cfg = AttrDict(deepcopy(hardware_cfg))
    expt.cfg.expt = AttrDict(dict(
        reps=2000, gain_start=1000, gain_stop=5000, gain_npts=9,
        check_f=False, active_reset=False, qubits=[0], ...
    ))
    expt.go()

Usage (with SweepRunner for freq sweep):
    runner = SweepRunner(
        station=station,
        ExptClass=meas.SingleShotGainSweepExperiment,
        default_expt_cfg=defaults,  # includes gain_start/stop/npts
        sweep_param='readout_freq',
        job_client=client,
    )
    result = runner.run(sweep_start=7249, sweep_stop=7251, sweep_npts=11, batch=True)
"""

import numpy as np
from tqdm import tqdm

from slab import Experiment, AttrDict
from experiments.MM_base import MMAveragerProgram, MM_base
from experiments.single_qubit.single_shot import HistogramProgram
from fitting.fit_display_classes import Histogram


class SingleShotGainSweepExperiment(Experiment):
    """
    Sweep readout gain and measure single-shot fidelity at each point.

    expt config keys:
        reps: number of shots per gain point
        gain_start: starting readout gain (DAC units)
        gain_stop: ending readout gain (DAC units)
        gain_npts: number of gain points
        readout_freq: (optional) override readout frequency [MHz]
        readout_length: (optional) override readout length [us]
        check_f: whether to also measure f state
        active_reset: whether to use active reset
        qubits: list of qubit indices
        + all other HistogramProgram config keys (prepulse, gate_based, etc.)
    """

    def __init__(
        self, soccfg=None, path="", prefix="SingleShotGainSweep", config_file=None, progress=None
    ):
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=False):

        # Ensure config values are list-typed (same pattern as HistogramExperiment)
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        for subcfg in (self.cfg.device.readout, self.cfg.device.qubit, self.cfg.hw.soc):
            for key, value in subcfg.items():
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        for key3, value3 in value2.items():
                            if not isinstance(value3, list):
                                value2.update({key3: [value3] * num_qubits_sample})
                elif not isinstance(value, list):
                    subcfg.update({key: [value] * num_qubits_sample})

        # Apply readout frequency/length overrides from expt config
        if hasattr(self.cfg.expt, 'readout_freq'):
            self.cfg.device.readout.frequency = [self.cfg.expt.readout_freq]
        if hasattr(self.cfg.expt, 'readout_length'):
            self.cfg.device.readout.readout_length = [self.cfg.expt.readout_length]

        # Calculate read_num for active reset
        read_num = 1
        if self.cfg.expt.active_reset:
            params = MM_base.get_active_reset_params(self.cfg)
            read_num += MMAveragerProgram.active_reset_read_num(**params)

        # Generate gain sweep values
        gains = np.linspace(
            self.cfg.expt.gain_start,
            self.cfg.expt.gain_stop,
            self.cfg.expt.gain_npts,
        )
        print(f'[GainSweep] gains array: {gains}')

        check_e = self.cfg.expt.get('check_e', True)
        check_f = self.cfg.expt.get('check_f', False)

        # Accumulate results
        data = {
            'xpts': [],
            'fids': [],
            'contrast': [],
            'thresholds': [],
            'angle': [],
        }

        for gain in tqdm(gains, disable=not progress, desc='Gain sweep'):
            # Override readout gain for this point
            self.cfg.device.readout.gain = [int(gain)]
            shot_data = dict()

            # Ground state shots
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = False
            cfg.expt.pulse_f = False
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
            avgi, avgq = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
                readouts_per_experiment=read_num,
            )
            shot_data['Ig'], shot_data['Qg'] = histpro.collect_shots()

            # Excited state shots
            if check_e:
                cfg = AttrDict(self.cfg.copy())
                cfg.expt.pulse_e = True
                cfg.expt.pulse_f = False
                histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
                avgi, avgq = histpro.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                    readouts_per_experiment=read_num,
                )
                shot_data['Ie'], shot_data['Qe'] = histpro.collect_shots()

            # F state shots
            if check_f:
                cfg = AttrDict(self.cfg.copy())
                cfg.expt.pulse_e = True
                cfg.expt.pulse_f = True
                histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)
                avgi, avgq = histpro.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                    readouts_per_experiment=read_num,
                )
                shot_data['If'], shot_data['Qf'] = histpro.collect_shots()

            self.prog = histpro

            # Analyze this gain point
            hist = Histogram(
                data=shot_data,
                verbose=False,
                config=self.cfg,
                readout_per_round=read_num,
            )
            hist.analyze(plot=False)

            fid = hist.results['fids'][0]  # ge fidelity
            contrast_val = np.median(shot_data.get('Ie_rot', shot_data.get('Ie', [0]))) - \
                           np.median(shot_data.get('Ig_rot', shot_data.get('Ig', [0])))
            threshold = hist.results['thresholds'][0]
            angle = hist.results['angle']

            data['xpts'].append(gain)
            data['fids'].append(fid)
            data['contrast'].append(contrast_val)
            data['thresholds'].append(threshold)
            data['angle'].append(angle)

        # Convert to numpy arrays
        for k in data:
            data[k] = np.array(data[k])

        self.data = data
        return data

    def _is_2d(self, data):
        """Check if data was stacked by SweepRunner (2D: freq x gain)."""
        return (
            'readout_freq_sweep' in data
            and 'fids' in data
            and hasattr(data['fids'], 'ndim')
            and data['fids'].ndim == 2
        )

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        if self._is_2d(data):
            # 2D case: stacked by SweepRunner (freq x gain)
            fids = data['fids']
            best_idx = np.unravel_index(np.argmax(fids), fids.shape)
            gains = data['xpts'][0]  # gain values (same for each freq)
            freqs = data['readout_freq_sweep']
            data['best_gain'] = gains[best_idx[1]]
            data['best_freq'] = freqs[best_idx[0]]
            data['best_fid'] = fids[best_idx]
            print(f'Best fidelity: {data["best_fid"]:.4f} at gain={data["best_gain"]:.0f}, freq={data["best_freq"]:.3f}')
        else:
            # 1D case: standalone gain sweep
            best_idx = np.argmax(data['fids'])
            data['best_gain'] = data['xpts'][best_idx]
            data['best_fid'] = data['fids'][best_idx]
            print(f'Best fidelity: {data["best_fid"]:.4f} at gain={data["best_gain"]:.0f}')

        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data

        import matplotlib.pyplot as plt

        if self._is_2d(data):
            # 2D heatmap: freq x gain
            freqs = data['readout_freq_sweep']
            gains = data['xpts'][0]
            fids = data['fids']
            contrast = data['contrast']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            im0 = axes[0].pcolormesh(gains, freqs, fids, shading='auto', cmap='viridis')
            plt.colorbar(im0, ax=axes[0], label='Fidelity')
            axes[0].set_xlabel('Readout Gain [DAC units]')
            axes[0].set_ylabel('Readout Freq [MHz]')
            axes[0].set_title('Fidelity')

            im1 = axes[1].pcolormesh(gains, freqs, contrast, shading='auto', cmap='viridis')
            plt.colorbar(im1, ax=axes[1], label='Contrast')
            axes[1].set_xlabel('Readout Gain [DAC units]')
            axes[1].set_ylabel('Readout Freq [MHz]')
            axes[1].set_title('Contrast')

            # Mark optimum
            if 'best_gain' in data and 'best_freq' in data:
                for ax in axes:
                    ax.plot(data['best_gain'], data['best_freq'], 'r*', markersize=15)

            plt.tight_layout()
            plt.show()
        else:
            # 1D line plot: fidelity vs gain
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(data['xpts'], data['fids'], 'o-')
            axes[0].set_xlabel('Readout Gain [DAC units]')
            axes[0].set_ylabel('ge Fidelity')
            axes[0].set_title('Fidelity vs Readout Gain')

            axes[1].plot(data['xpts'], data['contrast'], 'o-')
            axes[1].set_xlabel('Readout Gain [DAC units]')
            axes[1].set_ylabel('Contrast (Ie_rot - Ig_rot)')
            axes[1].set_title('Contrast vs Readout Gain')

            axes[0].set_yscale('log')

            if 'best_gain' in data:
                for ax in axes:
                    ax.axvline(data['best_gain'], color='r', linestyle='--', alpha=0.5,
                               label=f"Best: {data['best_gain']:.0f}, Fid: {data['best_fid']:.4f}")
                    ax.legend()

            plt.tight_layout()
            plt.show()

    def save_data(self, data=None):
        print(f"Saving {self.fname}")
        super().save_data(data=data)
