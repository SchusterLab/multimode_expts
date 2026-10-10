"""The fixed-photon-number Hamiltonian of the many-body Ramsey Floquet map.

Moved out of :func:`fitting.qsim.mbr_spectrum.analyze_spectrum` in MBR
redesign step 7b (2026-09-24), without changes to the arithmetic, so that
theory levels can be computed without a measured reconstruction (the disorder
planning and ensemble analysis need only the levels). ``analyze_spectrum``
calls it; ``tests/test_mbr_analysis_golden.py`` pins the result through it.

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
