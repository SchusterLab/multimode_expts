"""The central-boson (star) model in one fixed photon number sector.

The effective Hamiltonian of the Floquet map: M1 (mode 0, the hub, Kerr
nonlinear) coupled to the storage modes (the leaves). In cyclic MHz (H/h),

    H = sum_s (-detuning_s) n_s + (K/2) n_M1 (n_M1 - 1) + sum_s g_s (a_M1^dag a_s + h.c.)

- ``detunings`` are what the pulse program adds to the storage-M1 sidebands;
  the onsite energy is minus that.
- The Kerr term is (K/2) n(n-1), with K the physical Kerr, device sign.
- Basis order: ``itertools.product`` over (M1, *storages), lexicographic.

Moved out of :func:`fitting.qsim.mbr_spectrum.analyze_spectrum` in MBR
redesign step 7b (2026-09-24), and from ``fitting/qsim/mbr_hamiltonian.py``
to ``theory/`` on 2026-10-24, both without changes to the arithmetic.
``tests/test_mbr_analysis_golden.py`` pins the result through ``analyze_spectrum``.

Pure numerics.
"""
from itertools import product

import numpy as np

from slab import AttrDict


def fixed_n_hamiltonian(photon_number, mode_count, detunings, couplings_MHz,
                        physical_kerr_MHz):
    """Build and diagonalize H in the Fock basis of ``photon_number`` photons.

    ``mode_count`` counts M1 and the storage modes; ``detunings`` (MHz) and
    ``couplings_MHz`` are per storage mode. The pulse program adds the
    detuning to the storage-M1 sideband, so the onsite energy is -detuning.

    Returns AttrDict with ``fock_basis`` (list of occupation lists),
    ``fock_index`` ({tuple: row}), ``hamiltonian_MHz``, ``energies_MHz``
    (ascending), ``states`` (columns are eigenstates in the Fock basis) and
    ``basis_eigenstate_weights`` (|states|**2).
    """
    detunings = np.asarray(detunings)
    #Here, the Hamiltonian is directly calculated as a matrix in a Fock basis
    #First, product makes the all possible product states within photon_number
    #and then those are conditionally stored in fock_basis if the number = photon number
    fock_basis = [
        list(occupation) for occupation in product(range(photon_number + 1), repeat=mode_count)
        if sum(occupation) == photon_number
    ]
    #Storing index of each fock basis
    fock_index = {tuple(occupation): index for index, occupation in enumerate(fock_basis)}
    #Making Hamiltonian matrix in a fock basis
    H_MHz = np.zeros((len(fock_basis), len(fock_basis)))
    # The pulse program adds detuning to the positive storage-M1 sideband, so the rotating-frame onsite energy is -detuning.
    onsite_MHz = np.concatenate(([0.], -detunings))
    # updating Hamiltonian indices by estimating
    # <n_i|H_{diag}|n_j> = \delta_{ij}(delta_i n_i+Kerr/2*n_M*(n_M-1) 
    #Specifically, the algorithm is
    #   1. Multiply self Kerr times n_M1
    #   2. Multiply onsize detuning times n_i
    # <n_i|H_{coupling}|n_j>  = g \delta_{n_M+1  n_i-1}\sqrt{n_M+1 n_i}+
    #                           g \delta_{n_M-1  n_i+1}\sqrt{n_M   n_i+1}
    #Specifically, the algorithm is
    #For each column occupation,
    #   1. Loop the iteraction on storage mode index i
    #   2. Find the state with n_M increased by 1 and n_i decreased by 1 
    #      using fock_index dictionary
    #   3. Add g * \sqrt{n_M+1 n_i}
    #   4. Do the same for the state iwth n_M-1 and n_i+1
    for column, occupation in enumerate(fock_basis):
        n_M1 = occupation[0]
        H_MHz[column, column] = np.dot(onsite_MHz, occupation) + 0.5 * physical_kerr_MHz * n_M1 * (n_M1 - 1)
        for mode_index, coupling_MHz in enumerate(couplings_MHz, start=1):
            if n_M1 == 0:
                continue
            final_occupation = occupation.copy()
            final_occupation[0] -= 1
            final_occupation[mode_index] += 1
            row = fock_index[tuple(final_occupation)]
            matrix_element = coupling_MHz * np.sqrt(n_M1 * (occupation[mode_index] + 1))
            H_MHz[row, column] += matrix_element
            H_MHz[column, row] += matrix_element
    #Using np.linalg.eigh, get the eigenvalue of the Hamiltonian Matrix
    #Returns matrix with the index of (f, k), where k being eigenstate index
    #And f being fock state index.
    #So each column is an eigen state in a fock basis
    energies_MHz, states = np.linalg.eigh(H_MHz)

    return AttrDict(dict(
        fock_basis=fock_basis,
        fock_index=fock_index,
        hamiltonian_MHz=H_MHz,
        energies_MHz=energies_MHz,
        states=states,
        basis_eigenstate_weights=np.abs(states) ** 2,
    ))


def disorder_pattern(rng, count, rms_mhz):
    """-> ``count`` storage detunings: iid normal, mean removed, scaled to RMS ``rms_mhz``.

    Uniform on the sphere of fixed RMS in the zero-mean plane. ``disorder_pattern(rng, n, 1)``
    times W is the same draw as ``disorder_pattern(rng, n, W)``.
    """
    pattern = rng.normal(size=count)
    pattern -= pattern.mean()
    return pattern * rms_mhz / np.sqrt(np.mean(pattern**2))


def disorder_values(fock_basis, unit_pattern):
    """-> the diagonal of V = sum_s e_s n_s (the direction of the disorder) in ``fock_basis``."""
    return np.asarray(fock_basis, dtype=float)[:, 1:] @ np.asarray(unit_pattern, dtype=float)


def operator_form_hamiltonian(photon_number, mode_count, detunings, couplings_MHz, physical_kerr_MHz):
    """An independent build of the :func:`fixed_n_hamiltonian` matrix, for checks.

    Truncated boson operators (at most ``photon_number`` per mode) on the full space of all modes,
    H = sum_s -d_s n_s + (K/2) n_0 (n_0 - 1) + sum_s g_s (a_0^+ a_s + h.c.), then restricted to the
    states with ``photon_number`` photons. The full space has (photon_number + 1)^mode_count states.
    Returns AttrDict with ``hamiltonian_MHz`` and ``numbers`` (the sector diagonals of n_k, per mode),
    in the same basis order as :func:`fixed_n_hamiltonian`.
    """
    from functools import reduce

    cut = photon_number + 1
    lower = np.diag(np.sqrt(np.arange(1, cut)), 1)
    full = [reduce(np.kron, [lower if j == k else np.eye(cut) for j in range(mode_count)])
            for k in range(mode_count)]
    number = [a.T @ a for a in full]
    sector = np.where(np.isclose(sum(n.diagonal() for n in number), photon_number))[0]
    H = 0.5 * physical_kerr_MHz * number[0] @ (number[0] - np.eye(cut**mode_count))
    for s, (d, g) in enumerate(zip(detunings, couplings_MHz), start=1):
        H = H - d * number[s] + g * (full[0].T @ full[s] + full[s].T @ full[0])
    return AttrDict(dict(
        hamiltonian_MHz=H[np.ix_(sector, sector)],
        numbers=[n.diagonal()[sector] for n in number],
    ))
