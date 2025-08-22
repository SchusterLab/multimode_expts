from copy import deepcopy

import numpy as np

from experiments.qsim.qsim_base import QsimBaseProgram


class SidebandScrambleProgram(QsimBaseProgram):
    """
    Scramble 1 photon via fractional beam splitters

    
    expt params:
    swap_stors: list of storage modes to apply the floquet swaps to, will go in order of the list
    update_phases: boolean of whether to update each subsequent swap with the calibrated stark shift phase
    """
    def core_pulses(self):
        pulse_args = deepcopy(self.m1s_kwargs[self.cfg.expt.init_stor-1])

        # for kk in range(self.cfg.expt.floquet_cycle):
        #     for jj in range(7):
        #         if jj+1==self.cfg.expt.init_stor:
        #             pulse_args['phase'] = self.deg2reg(self.cfg.expt.advance_phase*kk)
        #             self.setup_and_pulse(**pulse_args)
        #             self.sync_all()
        #         elif jj+1==2:
        #             pulse2 = deepcopy(self.m1s_kwargs[1])
        #             # pulse2['gain'] //= self.cfg.expt.gain_div
        #             # pulse2['length'] //= self.cfg.expt.length_div
        #             pulse2['phase'] = self.deg2reg(3*kk)
        #             self.setup_and_pulse(**pulse2)
        #         else:
        #             # self.sync_all(self.us2cycles(0.3))
        #             pass
        # # self.sync_all(self.us2cycles(0.3 * 2 * self.cfg.expt.floquet_cycle))
        # self.sync_all()


        swap_stors = self.cfg.expt.swap_stors
        swap_stor_phases = np.zeros(len(swap_stors))
        update_phases = self.cfg.expt.update_phases
        for kk in range(self.cfg.expt.floquet_cycle):
            for i_stor, stor in enumerate(swap_stors):
                pulse_args = self.m1s_kwargs[stor - 1]
                pulse_args['phase'] = self.deg2reg(swap_stor_phases[i_stor], gen_ch=pulse_args['ch'])
                # print("phase on storage", stor, swap_stor_phases[i_stor])
                self.setup_and_pulse(**pulse_args)
                self.sync_all()

                # Update the phases for all other swaps using the phases accumulated during this swap
                if update_phases:
                    for j_stor, stor_B in enumerate(swap_stors):
                        if stor_B != stor:
                            stor_B_name = f"M1-S{stor_B}"
                            stor_name = f"M1-S{stor}"
                            swap_stor_phases[j_stor] += self.swap_ds.get_phase_from(stor_B_name, stor_name)
                            swap_stor_phases[j_stor] = swap_stor_phases[j_stor] % 360
        self.sync_all()




