"""N=2 Hamiltonian spectroscopy reprocessing from saved job files.

Hoisted out of `measurement_notebooks/jonginn/data_postprocess.ipynb` cells
168-172 by the stage-2 notebook decomposition. Primary caller:
`analysis_notebooks/202609_qsim_migration/mbr.py`.

Cell 172 was a single 291-line cell that did five separate things. It is split
here at its own comment boundaries, so the notebook reads as named steps and
the rigid shifts -- the one genuinely hand-entered input -- stay visible in the
notebook rather than being buried in a function.

The 40 HDF5 paths that cell 168 held inline are now
`analysis_notebooks/202609_qsim_migration/n2_spectroscopy_files.yml`. Read that
file's own note: cell 167's markdown claims fifteen occupations but the list
holds ten, which the analysis tolerates because it builds the full fifteen-state
basis for theory and compares only measured rows against it.

What the reconstruction does, per cell 167's markdown:

    Q(phi) = Pe(0, phi) - Pe(180 deg, phi)
    A_n    = Q(0) + i Q(90 deg)

The decoder phase matrix saved in each file was already applied during
acquisition and is not applied again. Only the occupation-dependent rigid
shifts the user enters are applied on top.

Temporary home, per the stage-2 instructions. Nothing here was reconciled with
`MBRSpectrumExperiment`, which does its own reconstruction and FFT for the
N=3 work in the same notebook -- deciding whether this path should survive at
all is a later, theme-specific task.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.qsim.floquet_dark_mode_readout import DarkBaseExperiment


def load_file_catalog(path):
    """Read the occupation-grouped HDF5 paths and check they all exist.

    Replaces the inline list and the existence check of cell 168. Raises
    rather than silently analyzing a partial set.
    """
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    groups = [
        [[str(fname) for fname in phi_files] for phi_files in occupation_files]
        for occupation_files in doc["occupation_groups"]
    ]
    missing_files = [
        fname
        for occupation_files in groups
        for phi_files in occupation_files
        for fname in phi_files
        if not Path(fname).exists()
    ]
    if missing_files:
        raise FileNotFoundError("\n".join(missing_files))

    print("occupation groups:", len(groups))
    print("HDF5 files:", sum(
        len(phi_files)
        for occupation_files in groups
        for phi_files in occupation_files
    ))
    return groups


def reconstruct_complex_return(spectroscopy_fnames, kerr_override_MHz=-10e-3):
    """Build A_n(t) from the saved analyzer-phase pairs (cell 170).

    Files are placed by the analyzer phase saved inside them rather than by
    list position. The large shot arrays are released as soon as each file's
    trace is reduced, which is why this loads rather than returning experiments.

    `kerr_override_MHz` reproduces a hard-coded line in the source: cell 170
    read the saved M1 Kerr out of the config and then immediately overwrote it
    with -10e-3 MHz. That override is preserved but named, so it is visible
    and can be turned off by passing None to use the saved value instead.

    Returns a dict with `A_raw`, `occupation_strings`, `cycles`, `N`,
    `mode_labels`, `swap_stors`, `decoder_phase_matrix` and
    `physical_kerr_MHz`.
    """
    occupation_strings = []
    A_rows = []
    cycles = None
    decoder_phase_matrix = None
    swap_stors = None
    physical_kerr_MHz = None

    for occupation_files in spectroscopy_fnames:
        Q_by_phi = {}
        saved_occupation = None

        for phi_files in occupation_files:
            cycle_parts = []
            Q_parts = []
            saved_phi = None

            for fname in phi_files:
                expt = DarkBaseExperiment.from_h5file(fname)
                expt_cfg = expt.cfg.expt

                saved_kerr_MHz = float(np.asarray(
                    expt.cfg.device.manipulate.kerr
                ).reshape(-1)[0])
                # Cell 170 read the saved Kerr and then unconditionally
                # overwrote it with -10e-3. Preserved, but named.
                if kerr_override_MHz is not None:
                    saved_kerr_MHz = kerr_override_MHz
                if physical_kerr_MHz is None:
                    physical_kerr_MHz = saved_kerr_MHz
                elif not np.isclose(saved_kerr_MHz, physical_kerr_MHz):
                    raise ValueError('the saved M1 Kerr changed between jobs')

                occupation = tuple(int(n) for n in expt_cfg.spectroscopy_occupations)
                phi = float(expt_cfg.spectroscopy_analyzer_phase) % 360.
                measured_theta = np.asarray(expt.data['xpts'], dtype=float) % 360.
                measured_cycles = np.asarray(expt.data['ypts'], dtype=int)
                signal = np.asarray(expt.data['avgi'], dtype=float).reshape(
                    len(measured_cycles),
                    len(measured_theta),
                )

                theta_0 = np.flatnonzero(np.isclose(measured_theta, 0.))[0]
                theta_180 = np.flatnonzero(np.isclose(measured_theta, 180.))[0]

                readout_Ig = float(np.asarray(
                    expt.cfg.device.readout.Ig
                ).reshape(-1)[0])
                readout_Ie = float(np.asarray(
                    expt.cfg.device.readout.Ie
                ).reshape(-1)[0])
                Pe = (signal - readout_Ig) / (readout_Ie - readout_Ig)

                if saved_occupation is None:
                    saved_occupation = occupation
                elif occupation != saved_occupation:
                    raise ValueError('one occupation group contains different states')

                if saved_phi is None:
                    saved_phi = phi
                elif not np.isclose(phi, saved_phi):
                    raise ValueError('one analyzer group contains different phases')

                saved_matrix = np.asarray(
                    expt_cfg.decoder_phase_matrix,
                    dtype=float,
                )
                if decoder_phase_matrix is None:
                    decoder_phase_matrix = saved_matrix
                    swap_stors = [int(stor) for stor in expt_cfg.swap_stors]
                elif not np.allclose(saved_matrix, decoder_phase_matrix):
                    raise ValueError('the saved decoder phase matrix changed between jobs')

                cycle_parts.append(measured_cycles)
                Q_parts.append(Pe[:, theta_0] - Pe[:, theta_180])
                del expt

            acquired_cycles = np.concatenate(cycle_parts)
            Q = np.concatenate(Q_parts)
            order = np.argsort(acquired_cycles)
            acquired_cycles = acquired_cycles[order]
            Q = Q[order]

            if cycles is None:
                cycles = acquired_cycles
            elif not np.array_equal(acquired_cycles, cycles):
                raise ValueError('the saved Floquet cycles are incomplete or different')

            Q_by_phi[int(round(saved_phi)) % 360] = Q

        if 0 not in Q_by_phi or 90 not in Q_by_phi:
            raise ValueError('each occupation needs analyzer phases 0 and 90 deg')

        occupation_strings.append(saved_occupation)
        A_rows.append(Q_by_phi[0] + 1j * Q_by_phi[90])

    A_raw = np.asarray(A_rows)
    N = int(sum(occupation_strings[0]))
    mode_labels = ['M1'] + [f'S{stor}' for stor in swap_stors]

    print('N =', N)
    print('A_raw shape =', A_raw.shape)
    for occupation, amplitude in zip(occupation_strings, A_raw):
        print(occupation, 'A(0) =', amplitude[0])

    return {
        "A_raw": A_raw,
        "occupation_strings": occupation_strings,
        "cycles": cycles,
        "N": N,
        "mode_labels": mode_labels,
        "swap_stors": swap_stors,
        "decoder_phase_matrix": decoder_phase_matrix,
        "physical_kerr_MHz": physical_kerr_MHz,
    }


def fft_grid_and_raw_dos(cycles, A_raw, occupation_strings,
                         hamiltonian_dt_us, zero_padding):
    """The FFT grid and every un-shifted local spectrum (cell 172, part 1).

    Rows are deliberately *not* divided by A(0): this analysis may translate
    spectra but must not change their weights.

    Returns a dict with `time_us`, `sample_times_us`, `window`, `n_fft`,
    `energy_MHz`, `energy_step_MHz` and `raw_local_DOS`.
    """
    time_us = cycles * hamiltonian_dt_us
    sample_times_us = np.diff(time_us)
    if not np.allclose(sample_times_us, sample_times_us[0]):
        raise ValueError('FFT requires uniformly spaced Floquet cycles')

    window = np.hanning(len(cycles))
    n_fft = zero_padding * len(cycles)
    energy_MHz = np.fft.fftshift(np.fft.fftfreq(
        n_fft,
        d=float(sample_times_us[0]),
    ))
    energy_step_MHz = float(energy_MHz[1] - energy_MHz[0])

    # First construct every raw local spectrum. Do not divide rows by A(0):
    # this cell is allowed to translate spectra, but not to change their weights.
    raw_local_DOS = np.empty((len(occupation_strings), n_fft))
    for occupation_index in range(len(occupation_strings)):
        raw_local_DOS[occupation_index] = abs(np.fft.fftshift(np.fft.ifft(
            A_raw[occupation_index] * window,
            n=n_fft,
        )))

    return {
        "time_us": time_us,
        "sample_times_us": sample_times_us,
        "window": window,
        "n_fft": n_fft,
        "energy_MHz": energy_MHz,
        "energy_step_MHz": energy_step_MHz,
        "raw_local_DOS": raw_local_DOS,
    }


def build_fixed_n_theory(N, mode_labels, swap_stors, floquet_couplings_MHz,
                         physical_kerr_MHz, physical_cycle_us,
                         hamiltonian_dt_us, occupation_strings, grid):
    """The fixed-N target Hamiltonian and its local spectra (cell 172, part 2).

    This is the same Hamiltonian behind the orange theory curve in the source
    figures. Note the Kerr rescaling: the physical per-physical-cycle Kerr is
    converted to the logical Hamiltonian time step by
    `physical_cycle_us / hamiltonian_dt_us`.

    `grid` is the result of `fft_grid_and_raw_dos`.

    Returns a dict with `H_N_MHz`, `fock_basis`, `fock_index`,
    `theory_energies_MHz`, `theory_states`, `effective_kerr_MHz` and
    `theory_local_DOS`.
    """
    time_us = grid["time_us"]
    window = grid["window"]
    n_fft = grid["n_fft"]
    raw_local_DOS = grid["raw_local_DOS"]

    # Build the same fixed-N target Hamiltonian used by the orange qsim curve.
    if len(floquet_couplings_MHz) != len(swap_stors):
        raise ValueError('floquet_couplings_MHz must match swap_stors')

    fock_basis = [
        tuple(occupation)
        for occupation in product(
            range(N + 1),
            repeat=len(mode_labels),
        )
        if sum(occupation) == N
    ]
    fock_index = {
        occupation: index
        for index, occupation in enumerate(fock_basis)
    }

    effective_kerr_MHz = (
        physical_kerr_MHz
        * physical_cycle_us
        / hamiltonian_dt_us
    )
    H_N_MHz = np.zeros((len(fock_basis), len(fock_basis)))

    for column, occupation in enumerate(fock_basis):
        n_M1 = occupation[0]
        H_N_MHz[column, column] = (
            0.5 * effective_kerr_MHz * n_M1 * (n_M1 - 1)
        )

        for mode_index, coupling_MHz in enumerate(
                floquet_couplings_MHz, start=1):
            if n_M1 == 0:
                continue

            final_occupation = list(occupation)
            final_occupation[0] -= 1
            final_occupation[mode_index] += 1
            row = fock_index[tuple(final_occupation)]
            matrix_element_MHz = coupling_MHz * np.sqrt(
                n_M1 * (occupation[mode_index] + 1)
            )
            H_N_MHz[row, column] += matrix_element_MHz
            H_N_MHz[column, row] += matrix_element_MHz

    theory_energies_MHz, theory_states = np.linalg.eigh(H_N_MHz)
    theory_phase = np.exp(
        -2j * np.pi * np.outer(theory_energies_MHz, time_us)
    )
    theory_local_DOS = np.empty_like(raw_local_DOS)

    for occupation_index, occupation in enumerate(occupation_strings):
        basis_row = fock_index[occupation]
        spectral_weights = abs(theory_states[basis_row]) ** 2
        theory_A = spectral_weights @ theory_phase
        theory_local_DOS[occupation_index] = abs(np.fft.fftshift(
            np.fft.ifft(
                theory_A * window,
                n=n_fft,
            )
        ))


    return {
        "H_N_MHz": H_N_MHz,
        "fock_basis": fock_basis,
        "fock_index": fock_index,
        "theory_energies_MHz": theory_energies_MHz,
        "theory_states": theory_states,
        "effective_kerr_MHz": effective_kerr_MHz,
        "theory_local_DOS": theory_local_DOS,
    }


def apply_rigid_shifts(A_raw, occupation_strings, occupation_energy_shift_MHz,
                       hamiltonian_dt_us, grid, theory):
    """Apply one overall energy shift per occupation (cell 172, parts 3-4).

    Nothing is fitted here. The shift only changes the phase ramp of A_n(t);
    it does not stretch or rescale a spectrum. A missing occupation raises,
    per the source and per the stage-2 rule that a missing scientific input
    must fail visibly rather than defaulting to zero.

    Returns a dict with `spectral_shift_MHz`, `A_corrected`,
    `corrected_local_DOS`, `raw_DOS`, `corrected_DOS` and `theory_DOS`.
    """
    window = grid["window"]
    n_fft = grid["n_fft"]
    time_us = grid["time_us"]
    raw_local_DOS = grid["raw_local_DOS"]
    theory_local_DOS = theory["theory_local_DOS"]

    missing_occupations = [
        occupation
        for occupation in occupation_strings
        if occupation not in occupation_energy_shift_MHz
    ]
    if missing_occupations:
        raise KeyError(
            'add overall shifts for these occupations: '
            f'{missing_occupations}'
        )

    spectral_shift_MHz = np.array([
        float(occupation_energy_shift_MHz[occupation])
        for occupation in occupation_strings
    ])

    # Under the current IFFT convention this moves every peak by +shift_MHz.
    A_corrected = A_raw * np.exp(
        -2j * np.pi
        * spectral_shift_MHz[:, None]
        * time_us[None, :]
    )

    corrected_local_DOS = np.empty_like(raw_local_DOS)
    for occupation_index in range(len(occupation_strings)):
        corrected_local_DOS[occupation_index] = abs(np.fft.fftshift(np.fft.ifft(
            A_corrected[occupation_index] * window,
            n=n_fft,
        )))

    raw_DOS = np.sum(raw_local_DOS, axis=0)
    corrected_DOS = np.sum(corrected_local_DOS, axis=0)
    theory_DOS = np.sum(theory_local_DOS, axis=0)
    theory_DOS *= corrected_DOS.max() / theory_DOS.max()

    print('manually entered occupation-resolved rigid shifts')
    for occupation_index, (occupation, shift_MHz) in enumerate(zip(
            occupation_strings,
            spectral_shift_MHz)):
        phase_ramp_deg_per_cycle = (
            -360. * shift_MHz * hamiltonian_dt_us
        )
        print(
            occupation,
            f'spectral shift = {shift_MHz:+.6f} MHz,',
            f'phase ramp = {phase_ramp_deg_per_cycle:+.6f} deg/cycle,',
            f'|A(0)| = {abs(A_raw[occupation_index, 0]):.4f}',
        )

    return {
        "spectral_shift_MHz": spectral_shift_MHz,
        "A_corrected": A_corrected,
        "corrected_local_DOS": corrected_local_DOS,
        "raw_DOS": raw_DOS,
        "corrected_DOS": corrected_DOS,
        "theory_DOS": theory_DOS,
    }


def plot_shifted_spectra(occupation_strings, mode_labels, N, cycles,
                         energy_limit_MHz, grid, theory, shifted,
                         physical_kerr_MHz):
    """Per-occupation spectra, the row heatmap, and the summed DOS (cell 172, part 5)."""
    energy_MHz = grid["energy_MHz"]
    raw_local_DOS = grid["raw_local_DOS"]
    sample_times_us = grid["sample_times_us"]
    theory_local_DOS = theory["theory_local_DOS"]
    theory_energies_MHz = theory["theory_energies_MHz"]
    effective_kerr_MHz = theory["effective_kerr_MHz"]
    corrected_local_DOS = shifted["corrected_local_DOS"]
    raw_DOS = shifted["raw_DOS"]
    corrected_DOS = shifted["corrected_DOS"]
    theory_DOS = shifted["theory_DOS"]
    # Each subplot title carries its own shift.
    spectral_shift_MHz = shifted["spectral_shift_MHz"]

    # Show every manually shifted local spectrum.
    n_columns = 3
    n_rows = int(np.ceil(len(occupation_strings) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(15, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for occupation_index, occupation in enumerate(occupation_strings):
        row, column = divmod(occupation_index, n_columns)
        ax = axes[row, column]
        theory_plot = theory_local_DOS[occupation_index].copy()
        theory_plot *= (
            corrected_local_DOS[occupation_index].max()
            / theory_plot.max()
        )

        ax.plot(
            energy_MHz,
            raw_local_DOS[occupation_index],
            color='0.75',
            label='raw',
        )
        ax.plot(
            energy_MHz,
            corrected_local_DOS[occupation_index],
            color='black',
            label='manual shift',
        )
        ax.plot(
            energy_MHz,
            theory_plot,
            color='tab:orange',
            label='theory (shape scaled)',
        )
        ax.set_xlim(-energy_limit_MHz, energy_limit_MHz)
        ax.set(
            xlabel='energy E/h (MHz)',
            ylabel='spectral magnitude',
            title=(
                f'{occupation}, shift '
                f'{spectral_shift_MHz[occupation_index]:+.4f} MHz'
            ),
        )
        ax.legend(fontsize=8)

    for empty_index in range(len(occupation_strings), n_rows * n_columns):
        row, column = divmod(empty_index, n_columns)
        axes[row, column].axis('off')

    plt.show()

    corrected_heatmap = corrected_local_DOS / np.maximum(
        corrected_local_DOS.max(axis=1, keepdims=True),
        1e-15,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(24, 10),
        constrained_layout=True,
    )

    axes[0].imshow(
        corrected_heatmap,
        origin='lower',
        aspect='auto',
        extent=[
            energy_MHz[0],
            energy_MHz[-1],
            -0.5,
            len(occupation_strings) - 0.5,
        ],
        cmap='magma',
    )
    axes[0].set_yticks(range(len(occupation_strings)))
    axes[0].set_yticklabels([str(occupation) for occupation in occupation_strings])
    axes[0].set_xlim(-energy_limit_MHz, energy_limit_MHz)
    axes[0].set(
        xlabel='energy E/h (MHz)',
        ylabel=f'occupation {mode_labels}',
        title='manually shifted local spectra (row normalized)',
    )

    axes[1].plot(
        energy_MHz,
        raw_DOS,
        color='0.75',
        label='raw',
    )
    axes[1].plot(
        energy_MHz,
        corrected_DOS,
        color='black',
        label='manual shift',
    )
    axes[1].plot(
        energy_MHz,
        theory_DOS,
        color='tab:orange',
        label='theory',
    )
    axes[1].set_xlim(-energy_limit_MHz, energy_limit_MHz)
    axes[1].set(
        xlabel='energy E/h (MHz)',
        ylabel='summed local spectral magnitude',
        title=f'N={N} manually shifted DOS and theory',
    )
    axes[1].legend()
    fig.tight_layout()
    plt.show()

    raw_power = np.sum(raw_local_DOS ** 2, axis=1)
    corrected_power = np.sum(corrected_local_DOS ** 2, axis=1)

    print('saved signed M1 Kerr =', physical_kerr_MHz, 'MHz')
    print('effective theory Kerr =', effective_kerr_MHz, 'MHz')
    print('target eigenenergies =', theory_energies_MHz, 'MHz')
    print('sample time =', sample_times_us[0], 'us')
    print('Nyquist range = +/-', 0.5 / sample_times_us[0], 'MHz')
    print(
        'approximate Hann resolution =',
        2. / (len(cycles) * sample_times_us[0]),
        'MHz',
    )
    print(
        'largest relative row FFT-power change from the phase ramps =',
        np.max(abs(corrected_power - raw_power) / raw_power),
    )
