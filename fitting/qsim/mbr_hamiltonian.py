"""Hamiltonian consturction for theory result reproduction.
"""



from itertools import product

import numpy as np

from slab import AttrDict

def construct_hamiltonian(photon_number,
                          mode_count,
                          detunings,
                          physical_kerr_MHz,
                          couplings_MHz):
    """
    Here, the Hamiltonian is directly calculated as a matrix in a Fock basis
    First, product makes the all possible product states within photon_number
    and then those are conditionally stored in fock_basis if the number = photon number
    Args:
        - photon_number: the total number of photons
        - mode_count: the total number of modes
        - detunings: detuning list
        - physical_kerr_MHz: self Kerr value in MHz
        - coupling_MHz: coupling_list in MHz


    Returns:
        - H_MHz: Hamiltonian matrix
        - couplings_MHz: Fock states and their index upon which H_MHz is defined
    """
    fock_basis = [
        list(occupation) for occupation in product(range(photon_number + 1), repeat=mode_count)
        if sum(occupation) == photon_number
    ]
    
    #Storing index of each fock basis
    fock_index = {tuple(occupation): index 
                  for index, occupation 
                  in enumerate(fock_basis)}
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
    return H_MHz, fock_index