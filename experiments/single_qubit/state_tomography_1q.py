'''
Single-Qubit State Tomography

Reconstructs the density matrix of a single transmon qubit prepared by an
arbitrary (gate-based) prepulse, by measuring in the X, Y and Z bases and
running maximum-likelihood reconstruction (``fitting/state_tomography.py``).

Sequence per basis:
  1. phase reset
  2. (optional) active reset
  3. state-prep prepulse  (the state to characterize)
  4. (optional) state-prep post-selection readout
  5. basis pre-rotation on the ge manifold:
       Z -> identity (no pulse)
       X -> hpi about Y   (maps |+x> onto |g>)
       Y -> hpi about X   (maps |+y> onto |g>)
  6. ge readout

The three programs are run sequentially from the Experiment; counts are
thresholded into (n_g, n_e) per basis and handed to the reconstruction core.

NOTE on conventions: the pre-rotation phases that realize the X/Y axes are
configurable (``cfg.expt.tomo_phases``) and default to {'X': 90, 'Y': 0}.
The intended validation workflow is to prepare known cardinal states
(|0>, |1>, |+>, |+i>, ...) and confirm the reconstruction; if an axis comes
out swapped or sign-flipped, adjust these phases. The reconstruction math
itself is unit-tested offline in tests/test_state_tomography_1q.py.

Seb 06/2026
'''

import numpy as np
from copy import deepcopy

from slab import Experiment, AttrDict
from tqdm import tqdm_notebook as tqdm

from experiments.MM_base import *
from fitting.state_tomography import reconstruct_single_qubit, state_fidelity


# Default pre-rotation phases (deg) realizing each measurement axis.
# X basis: hpi about Y (phase 90); Y basis: hpi about X (phase 0).
DEFAULT_TOMO_PHASES = {'X': 90, 'Y': 0}
TOMO_BASES = ('Z', 'X', 'Y')


class StateTomography1QProgram(MMAveragerProgram):
    '''Single basis-setting tomography program.

    Reads ``cfg.expt.tomo_basis`` in {'Z', 'X', 'Y'} to choose the
    pre-rotation appended after state prep.
    '''
    _pre_selection_filtering = True

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)
        self.cfg.reps = cfg.expt.reps
        self.cfg.rounds = cfg.expt.get('rounds', 1)
        super().__init__(soccfg, self.cfg)

    def initialize(self):
        self.MM_base_initialize()
        cfg = AttrDict(self.cfg)

        # State-prep prepulse. Accept either a gate-based list (compiled here)
        # or an already-compiled 7-row pulse array (e.g. when a partial swap
        # whose length was scaled outside the gate framework is included).
        # Gate entries start with a string channel name; compiled rows start
        # with numbers.
        prep_seq = cfg.expt.get('state_prep_seq', None)
        if prep_seq:
            if (isinstance(prep_seq[0], list) and prep_seq[0]
                    and isinstance(prep_seq[0][0], str)):
                self.state_prep_pulse = self.get_prepulse_creator(
                    prep_seq).pulse.tolist()
            else:
                self.state_prep_pulse = [list(r) for r in prep_seq]  # pre-compiled
        else:
            self.state_prep_pulse = None

        # Inline optimal-control pulse state-prep (closed-loop IQ_table path).
        # Mirrors WignerTomography1ModeProgram: opt_pulse names a config slot used
        # for the carrier frequency + gain placeholder, IQ_table is the inline
        # envelope played in its place. Played BEFORE the gate-based state_prep_seq
        # (if any) so an opt-control prep and gate prep compose.
        self.waveforms_opt_ctrl = None
        if cfg.expt.get('opt_pulse') and cfg.expt.get('IQ_table') is not None:
            self.waveforms_opt_ctrl = self.load_opt_ctrl_pulse(
                pulse_conf=cfg.expt.opt_pulse, IQ_table=cfg.expt.IQ_table)

        # Basis pre-rotation pulse (None for Z)
        basis = cfg.expt.get('tomo_basis', 'Z')
        phases = dict(DEFAULT_TOMO_PHASES)
        phases.update(cfg.expt.get('tomo_phases', {}))
        if basis == 'Z':
            self.basis_pulse = None
        else:
            self.basis_pulse = self.get_prepulse_creator(
                [['qubit', 'ge', 'hpi', phases[basis]]]).pulse.tolist()

        self.sync_all(200)

    def body(self):
        cfg = AttrDict(self.cfg)

        # 1. phase reset
        self.reset_and_sync()

        # 2. active reset
        if cfg.expt.get('active_reset', False):
            params = MM_base.get_active_reset_params(cfg)
            self.active_reset(**params)

        # 3. state preparation: inline opt-control pulse (if given) then the
        #    gate-based state_prep_seq (if given).
        if cfg.expt.get('opt_pulse'):
            creator = self.get_prepulse_creator(cfg.expt.opt_pulse)
            self.custom_pulse(cfg, creator.pulse.tolist(), prefix='opt_prep_',
                              waveform_preload=self.waveforms_opt_ctrl)
        if self.state_prep_pulse is not None:
            self.custom_pulse(cfg, self.state_prep_pulse, prefix='state_prep_')

        # 4. state-prep post-selection
        if cfg.expt.get('state_prep_postselect', False):
            self.post_selection_measure(
                parity=False,
                prefix='state_prep_ps')

        # 5. basis pre-rotation
        if self.basis_pulse is not None:
            self.custom_pulse(cfg, self.basis_pulse, prefix='tomo_rot_')

        # 6. ge readout
        self.measure_wrapper()

    def _calculate_read_num(self):
        cfg = self.cfg
        read_num = MM_base.lane_layout(cfg)['n_active_reset']
        if cfg.expt.get('state_prep_postselect', False):
            read_num += 1
        read_num += 1  # final tomography readout
        return read_num

    def collect_shots(self):
        '''Return (i0, q0, read_num) with shape (reps*rounds, read_num).'''
        read_num = self._calculate_read_num()
        reps = self.cfg.expt.reps
        rounds = self.cfg.expt.get('rounds', 1)
        total_reps = rounds * reps

        buf_i = self.di_buf[0] / self.readout_lengths_adc[0]
        buf_q = self.dq_buf[0] / self.readout_lengths_adc[0]
        shots_i0 = buf_i.reshape((total_reps, read_num))
        shots_q0 = buf_q.reshape((total_reps, read_num))
        return shots_i0, shots_q0, read_num


class StateTomography1QExperiment(Experiment):
    '''Single-qubit state tomography.

    Runs three basis settings (Z, X, Y), thresholds the final readout into
    g/e counts per basis (after optional active-reset / state-prep
    post-selection), and reconstructs the density matrix.

    expt = dict(
        reps, rounds,
        qubits=[0],
        state_prep_seq: gate-based prepulse list for the state to characterize
            (e.g. [['qubit','ge','hpi',0]] for |+>). None -> |g>.
        opt_pulse / IQ_table: optional inline optimal-control state-prep, same
            contract as WignerTomography1ModeExperiment. opt_pulse =
            [['opt_cont', enc, state]] names the config slot (carrier freq + gain
            placeholder); IQ_table = {'times','I_c','Q_c','I_q','Q_q'} is the
            inline envelope. Played before state_prep_seq. Used by the closed-loop
            server to characterize the transmon state an optimized pulse prepares.
        tomo_phases: optional {'X': deg, 'Y': deg} override for axis pulses.
        active_reset: bool,
        state_prep_postselect: bool,
        confusion: optional 2x2 readout assignment matrix for correction,
        target_state: optional ket (list of 2 complex) for fidelity report,
        recon_method: 'fast' (default) | 'cholesky' | 'linear',
        bases: optional subset of ['Z','X','Y'] (e.g. ['X','Y'] azimuth-only),
        dedupe_waveforms: bool (default False). When True, custom_pulse shares
            identical (channel, sigma) envelopes instead of loading one per gate
            -- required for long prepulses (e.g. (E+D)^N round-trips) that would
            otherwise overflow the generator's 32768-sample waveform buffer.
    )
    '''

    def __init__(self, soccfg=None, path='', prefix='StateTomography1Q',
                 config_file=None, progress=None):
        super().__init__(soccfg=soccfg, path=path, prefix=prefix,
                         config_file=config_file, progress=progress)

    def acquire(self, progress=False, debug=False):
        num_qubits_sample = len(self.cfg.device.qubit.f_ge)
        self.format_config_before_experiment(num_qubits_sample)

        threshold = self.cfg.device.readout.threshold[0]
        reps = self.cfg.expt.reps
        rounds = self.cfg.expt.get('rounds', 1)
        total_reps = reps * rounds

        # Which measurement bases to run. Default = full Z/X/Y tomography;
        # pass e.g. ['X', 'Y'] for an azimuth-only (phase) measurement.
        bases = list(self.cfg.expt.get('bases', TOMO_BASES))
        data = {'threshold': threshold, 'reps': reps, 'rounds': rounds,
                'bases': bases}

        for basis in tqdm(bases, desc='basis', disable=not progress):
            self.cfg.expt.tomo_basis = basis
            prog = StateTomography1QProgram(soccfg=self.soccfg, cfg=self.cfg)
            read_num = prog._calculate_read_num()
            prog.acquire(self.im[self.cfg.aliases.soc], threshold=None,
                         load_pulses=True, progress=progress,
                         readouts_per_experiment=read_num)
            i0, q0, _ = prog.collect_shots()
            data[f'i0_{basis}'] = i0
            data[f'read_num'] = read_num

        self.data = data
        return data

    def _basis_counts(self, data, basis):
        '''Threshold the final readout of one basis into (n_g, n_e),
        after active-reset / state-prep pre-selection filtering.'''
        i0 = data[f'i0_{basis}']
        read_num = data['read_num']
        threshold = data['threshold']
        per_shot = i0  # already (total_reps, read_num)

        indices = self._get_shot_indices()

        if 'ar_pre_selection' in indices:
            mask = per_shot[:, indices['ar_pre_selection']] < threshold
            per_shot = per_shot[mask]
        if 'state_prep_ps' in indices:
            mask = per_shot[:, indices['state_prep_ps']] < threshold
            per_shot = per_shot[mask]

        final = per_shot[:, indices['tomo']]
        n_e = int(np.sum(final > threshold))
        n_g = int(len(final) - n_e)
        return n_g, n_e

    def _get_shot_indices(self):
        cfg = self.cfg
        idx = 0
        indices = {}
        _ar = MM_base.lane_layout(cfg)
        if _ar['idx_pre_selection'] is not None:
            indices['ar_pre_selection'] = idx + _ar['idx_pre_selection']
        idx += _ar['n_active_reset']
        if cfg.expt.get('state_prep_postselect', False):
            indices['state_prep_ps'] = idx
            idx += 1
        indices['tomo'] = idx
        return indices

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        bases = list(data.get('bases', TOMO_BASES))
        counts = {b: self._basis_counts(data, b) for b in bases}
        data['counts'] = counts

        # Per-basis Pauli expectation values (always available).
        # <A> = (n_g - n_e) / (n_g + n_e), in the same convention the
        # reconstruction uses (basis pre-rotation maps +A eigenstate -> |g>).
        expectations = {}
        for b in bases:
            n_g, n_e = counts[b]
            tot = n_g + n_e
            expectations[b] = (n_g - n_e) / tot if tot else 0.0
        data['expectations'] = expectations

        # Azimuthal phase (only needs X, Y) -- the phase-sweep observable.
        if 'X' in expectations and 'Y' in expectations:
            data['azimuth_rad'] = float(np.arctan2(expectations['Y'],
                                                   expectations['X']))
            data['equatorial_contrast'] = float(
                np.hypot(expectations['X'], expectations['Y']))

        # Full density matrix only when all three bases were measured.
        if all(b in counts for b in TOMO_BASES):
            confusion = self.cfg.expt.get('confusion', None)
            if confusion is not None:
                confusion = np.asarray(confusion, dtype=float)
            method = self.cfg.expt.get('recon_method', 'fast')
            rho = reconstruct_single_qubit(counts, confusion=confusion,
                                           method=method)
            data['rho'] = rho
        else:
            data['rho'] = None

        target = self.cfg.expt.get('target_state', None)
        if target is not None and data['rho'] is not None:
            target = np.asarray(target, dtype=complex)
            data['fidelity'] = state_fidelity(data['rho'], target)
            print(f"State fidelity to target: {data['fidelity']:.4f}")

        if data['rho'] is not None:
            print('Reconstructed rho:')
            print(np.round(data['rho'], 4))
        else:
            print(f"Bases measured: {bases} (no full rho).")
            if 'azimuth_rad' in data:
                print(f"azimuth = {np.degrees(data['azimuth_rad']):.2f} deg, "
                      f"contrast = {data['equatorial_contrast']:.3f}")
        print('counts (n_g, n_e) per basis:', counts)
        return data

    def display(self, data=None, **kwargs):
        if data is None:
            data = self.data
        rho = data.get('rho')
        if rho is None:
            print('No reconstructed rho; run analyze() first.')
            return

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, mat, title in (
                (axes[0], np.real(rho), 'Re(rho)'),
                (axes[1], np.imag(rho), 'Im(rho)')):
            im = ax.imshow(mat, cmap='RdBu', vmin=-1, vmax=1)
            ax.set_title(title)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['g', 'e']); ax.set_yticklabels(['g', 'e'])
            for (r, c), v in np.ndenumerate(mat):
                ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fid = data.get('fidelity')
        suptitle = 'Single-qubit state tomography'
        if fid is not None:
            suptitle += f'  (F = {fid:.3f})'
        fig.suptitle(suptitle)
        plt.tight_layout()
        plt.show()

    def save_data(self, data=None):
        print(f'Saving {self.fname}')
        if data is None:
            data = self.data
        save_data = {}
        for key, value in data.items():
            # skip things h5py can't write; they survive in the pickled expt
            if value is None or isinstance(value, dict):
                continue  # None (e.g. rho in azimuth-only), counts/expectations
            if isinstance(value, str):
                save_data[key] = np.bytes_(value)
                continue
            if isinstance(value, (list, tuple)):
                arr = np.array(value)
                if arr.dtype.kind in ('U', 'S'):      # e.g. bases ['X','Y']
                    save_data[key] = arr.astype('S')
                elif arr.dtype.kind == 'O':           # ragged/object -> skip
                    continue
                else:
                    save_data[key] = arr
                continue
            if isinstance(value, np.ndarray):
                if value.dtype.kind in ('U', 'S'):
                    save_data[key] = value.astype('S')
                    continue
                if value.dtype == object:
                    continue
            save_data[key] = value
        super().save_data(data=save_data)
        return self.fname
