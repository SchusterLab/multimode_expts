'''
Fock-State Decoder for a lossy-beamsplitter channel.

System = manipulate mode (logical qubit a|0> + b|2>), environment = a storage
mode holding |1>, coupled by a partial beamsplitter (length-scaled M-S swap).

This module currently implements the **parity-syndrome characterization**
(Phase 2b / Section 3a): prepare the environment + the encoded system, apply
the partial beamsplitter at transmissivity eta, then read the manipulate-mode
parity (even {|0>,|2>} vs odd {|1>,|3>}). Sweeping eta maps out the syndrome.

Body sequence:
  1. phase reset
  2. (optional) active reset
  3. environment prep: |1> into the env storage mode (env_prep_seq)
  4. system encode: a|0> + b|2> in the manipulate mode (encode_seq, with the
     calibrated encoder phase folded in)
  5. partial beamsplitter: M <-> S_env swap, flat length scaled by
     eta_to_swap_ratio(eta)
  6. parity syndrome pulse on the manipulate mode
  7. ge readout (the syndrome bit)

The physics-specific gate sequences (env_prep_seq, encode_seq,
channel_swap_seq) are built in the notebook with MM_dual_rail_base helpers and
passed via cfg.expt -- the program stays generic and just compiles/plays them
(like StateTomography1QExperiment's state_prep_seq).

The decode + tomography path (readout_mode='decode', Phase 2c) is stubbed for a
later section.

Seb 06/2026
'''

import numpy as np
from copy import deepcopy

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import *
from experiments.transduction.decoder import eta_to_swap_ratio, scale_swap_length


class FockDecoderProgram(MMAveragerProgram):
    '''One eta-setting of the Fock-decoder channel (parity-syndrome mode).'''
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)
        super().__init__(soccfg, self.cfg)

    def _compile(self, seq):
        if not seq:
            return None
        return self.get_prepulse_creator(seq).pulse.tolist()

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)
        self.man_no = cfg.expt.get('man_no', 1)

        self.env_pulse = self._compile(cfg.expt.get('env_prep_seq'))
        self.encode_pulse = self._compile(cfg.expt.get('encode_seq'))

        # Partial beamsplitter: compile the full SWAP, scale flat length by ratio(eta).
        self.channel_pulse = None
        swap_seq = cfg.expt.get('channel_swap_seq')
        if swap_seq:
            full = self.get_prepulse_creator(swap_seq).pulse.tolist()
            ratio = eta_to_swap_ratio(cfg.expt.get('eta', 1.0))
            self.channel_pulse = scale_swap_length(full, ratio)

        # Parity syndrome pulse (maps manipulate parity onto the ge qubit).
        self.parity_pulse = self.get_parity_str(
            self.man_no, return_pulse=True, second_phase=180,
            fast=cfg.expt.get('parity_fast', False))

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)

        self.reset_and_sync()

        if cfg.expt.get('active_reset', False):
            self.active_reset(**MM_base.get_active_reset_params(cfg))

        if self.env_pulse is not None:
            self.custom_pulse(cfg, self.env_pulse, prefix='env_')
        if self.encode_pulse is not None:
            self.custom_pulse(cfg, self.encode_pulse, prefix='enc_')
        if self.channel_pulse is not None:
            self.custom_pulse(cfg, self.channel_pulse, prefix='chan_')

        mode = cfg.expt.get('readout_mode', 'parity')
        if mode == 'parity':
            if cfg.expt.get('parity_direct', False):
                # Hand-written ASM parity, byte-for-byte the path the Wigner uses
                # (play_parity_pulse). A/B against the compiled get_parity_str
                # path to isolate readout-implementation differences. man_no-1
                # matches the Wigner's man_mode_idx revival_time lookup.
                self.play_parity_pulse(self.man_no - 1, second_phase=180,
                                       fast=cfg.expt.get('parity_fast', False))
            else:
                self.custom_pulse(cfg, self.parity_pulse, prefix='parity_')
            self.measure_wrapper()
        else:
            raise NotImplementedError(
                f"readout_mode '{mode}' not implemented yet "
                "(decode + tomography is Phase 2c / Section 4)")

    def _calculate_read_num(self):
        cfg = self.cfg
        read_num = MM_base.lane_layout(cfg)['n_active_reset']
        read_num += 1  # final syndrome readout
        return read_num

    def collect_shots(self):
        read_num = self._calculate_read_num()
        total_reps = self.cfg.expt.reps * self.cfg.expt.get('rounds', 1)
        buf_i = self.di_buf[0] / self.readout_lengths_adc[0]
        buf_q = self.dq_buf[0] / self.readout_lengths_adc[0]
        return (buf_i.reshape((total_reps, read_num)),
                buf_q.reshape((total_reps, read_num)), read_num)


class FockDecoderExperiment(Experiment):
    '''Parity-syndrome characterization of the lossy-beamsplitter channel vs eta.

    expt = dict(
        reps, rounds, qubits=[0],
        man_no=1, env_storage=3,
        eta_list: list of transmissivities to sweep (e.g. high-eta > 2/3),
        env_prep_seq: gate list loading |1> into the env storage (build_env_prep_seq),
        encode_seq:   gate list encoding the system logical state (prep_fock_state
                      + encoder phase correction); [] -> empty system (|0>),
        channel_swap_seq: gate list for the full M-S swap (channel_swap_gate);
                      its flat length is scaled by eta_to_swap_ratio(eta),
        parity_fast: bool,
        readout_mode: 'parity' (only mode implemented),
        active_reset + flags (active_reset_dict),
        dedupe_waveforms: True recommended for the long env+encode sequence.
    )

    "Even fraction" = shots whose syndrome readout is below threshold (the parity
    pulse with second_phase=180 maps even {|0>,|2>} -> |g>). Confirm the
    convention from the eta->1 point (no swap -> stays even -> even_frac ~ 1).
    '''

    def __init__(self, soccfg=None, path='', prefix='FockDecoder',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        threshold = self.cfg.device.readout.threshold[0]
        eta_list = list(self.cfg.expt.eta_list)
        data = {'threshold': threshold, 'eta_list': eta_list,
                'reps': self.cfg.expt.reps,
                'rounds': self.cfg.expt.get('rounds', 1)}

        for eta in tqdm(eta_list, desc='eta', disable=not progress):
            self.cfg.expt.eta = float(eta)
            prog = FockDecoderProgram(soccfg=self.soccfg, cfg=self.cfg)
            read_num = prog._calculate_read_num()
            prog.acquire(self.im[self.cfg.aliases.soc], threshold=None,
                         load_pulses=True, progress=progress,
                         readouts_per_experiment=read_num)
            i0, _, _ = prog.collect_shots()
            data[f'i0_{eta:.4f}'] = i0
            data['read_num'] = read_num

        self.data = data
        return data

    def _shot_indices(self):
        cfg = self.cfg
        idx = 0
        indices = {}
        _ar = MM_base.lane_layout(cfg)
        if _ar.get('idx_pre_selection_list'):
            indices['ar_pre_selection'] = [idx + p for p in _ar['idx_pre_selection_list']]
        idx += _ar['n_active_reset']
        indices['syndrome'] = idx
        return indices

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        threshold = data['threshold']
        indices = self._shot_indices()

        even_frac, counts = [], []
        for eta in data['eta_list']:
            per_shot = data[f'i0_{eta:.4f}']
            if 'ar_pre_selection' in indices:
                mask = np.all(per_shot[:, indices['ar_pre_selection']] < threshold, axis=1)
                per_shot = per_shot[mask]
            syn = per_shot[:, indices['syndrome']]
            n = len(syn)
            n_even = int(np.sum(syn < threshold))  # even -> |g| (below threshold)
            even_frac.append(n_even / n if n else np.nan)
            counts.append(n)

        data['even_frac'] = np.array(even_frac)
        data['counts'] = np.array(counts)
        print(f"{'eta':>7} {'even_frac':>10} {'shots':>7}")
        for eta, ef, n in zip(data['eta_list'], even_frac, counts):
            print(f"{eta:7.3f} {ef:10.3f} {n:7d}")
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        import matplotlib.pyplot as plt
        ef = data.get('even_frac')
        if ef is None:
            print('run analyze() first'); return
        plt.figure(figsize=(7, 4))
        plt.plot(data['eta_list'], ef, 'o-')
        plt.xlabel('transmissivity eta'); plt.ylabel('parity even fraction')
        plt.title('Channel syndrome vs eta')
        plt.ylim(0, 1.05); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data
        save_data = {}
        for key, value in data.items():
            if value is None or isinstance(value, dict):
                continue
            if isinstance(value, str):
                save_data[key] = np.bytes_(value)
                continue
            if isinstance(value, (list, tuple)):
                arr = np.array(value)
                if arr.dtype.kind in ('U', 'S'):
                    save_data[key] = arr.astype('S')
                elif arr.dtype.kind == 'O':
                    continue
                else:
                    save_data[key] = arr
                continue
            if isinstance(value, np.ndarray) and value.dtype == object:
                continue
            save_data[key] = value
        super().save_data(data=save_data)
        return self.fname
