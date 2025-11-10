import os
import subprocess
from typing import Optional
from warnings import warn

import lmfit
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import rfft, rfftfreq
from slab import AttrDict


def ensure_local_file(remote_path, local_path, filename):
    """
    Ensure that a file from remote_path exists in local_path.
    If not, copy it over via sftp
    """
    local_file = os.path.join(local_path, filename)
    if os.path.exists(local_file):
        print(f'Found local file {local_file}')
    else:
        print(f'Fetching remote file {filename} to local path {local_path}')
        remote_file = os.path.join(remote_path, filename)
        subprocess_cmd = f'sftp {remote_file} {local_file}'
        subprocess.run(subprocess_cmd, shell=True, check=True)
    return local_file


def ensure_list_in_cfg(cfg: Optional[AttrDict]):
    """
    Expand entries in config that are length 1 to fill all qubits
    Modifies the cfg in place
    """
    assert cfg, 'Found empty config when trying to convert entries to lists!'

    num_qubits_sample = len(cfg.device.qubit.f_ge)
    for subcfg in (cfg.device.readout, cfg.device.qubit, cfg.hw.soc):
        for key, value in subcfg.items() :
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        if not(isinstance(value3, list)):
                            value2.update({key3: [value3]*num_qubits_sample})                                
            elif not(isinstance(value, list)):
                subcfg.update({key: [value]*num_qubits_sample})

def guess_freq(x, y):
    # note: could also guess phase but need zero-padding
    # just guessing freq seems good enough to escape from local minima in most cases
    yf = rfft(y - np.mean(y))
    xf = rfftfreq(len(x), x[1] - x[0])
    peak_idx = np.argmax(np.abs(yf[1:])) + 1
    return np.abs(xf[peak_idx]), np.angle(yf[peak_idx])

def filter_data_IQ(II, IQ, threshold):
    """
    Deals with active reset measurement data:
    4 shots are qubit ge test, qubit ef test, post-reset verification, data shot
    """
    result_Ig = []
    result_Ie = []

    for k in range(len(II) // 4):
        index_4k_plus_2 = 4 * k + 2
        index_4k_plus_3 = 4 * k + 3

        if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
            if II[index_4k_plus_2] < threshold:
                result_Ig.append(II[index_4k_plus_3])
                result_Ie.append(IQ[index_4k_plus_3])

    return np.array(result_Ig), np.array(result_Ie)


def post_select_raverager_data(data, cfg):
    """
    only deals with 4-shot active reset data now
    needs the cfg for rounds, reps, expts info to know shape
    """
    read_num = 4

    rounds = cfg.expt.rounds
    reps = cfg.expt.reps
    expts = cfg.expt.expts
    I_data = np.array(data['idata'])
    Q_data = np.array(data['qdata'])

    I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds * reps * read_num))
    Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds * reps * read_num))

    Ilist = []
    Qlist = []
    for ii in range(len(I_data)):
        Ig, Qg = filter_data_IQ(I_data[ii], Q_data[ii], cfg.device.readout.threshold[0])
        Ilist.append(np.mean(Ig))
        Qlist.append(np.mean(Qg))

    return Ilist, Qlist


class Cos2dModel(lmfit.Model):
    """
    Incompatible with lmfit api but very short to call
    """
    def __init__(self, *args, **kwargs):
        def cos2d(tau, phi, f, phi0, A, C):
            return A*np.cos(2*np.pi*f*tau + phi/180*np.pi + phi0/180*np.pi) + C
        super().__init__(cos2d, independent_vars=['tau', 'phi'], *args, **kwargs)

    def guess(self, data, tau, phi, **kwargs):
        verbose = kwargs.pop('verbose', None)
        phases = np.unwrap([guess_freq(tau, line)[1] for line in data]) / np.pi*180
        slope_sign = np.sign(np.corrcoef(phi, phases)[0, 1])
        # if we don't take care of the sign of the freq, it only finds the local minimum at f>0
        freq_guess = np.mean([guess_freq(tau, line)[0] for line in data]) * slope_sign
        offset_guess = np.mean(data)
        amp_guess = np.ptp(data) / 2
        if verbose:
            print(freq_guess, offset_guess, amp_guess)
            plt.plot(phi, phases)
        params = self.make_params(
            f=freq_guess,
            phi0=0,
            A=amp_guess,
            C=offset_guess
        )
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

    def fit(self, data, tau, phi, **kwargs):
        Tau, Phi = np.meshgrid(tau, phi)
        params = self.guess(data, tau, phi)
        return super().fit(data.ravel(), params, tau=Tau.ravel(), phi=Phi.ravel())

def fit_cos2d(data, tau, phi, plot=False, **kwargs):
    """
    Fit a 2D cos model to the data
    """
    #TODO: this should handle user supplied param constraints and
    # options such as whether to force phase at (tau, phi)=(0,0) to be 0
    model = Cos2dModel()
    result = model.fit(data, tau=tau, phi=phi, **kwargs)
    if result.rsquared<0.7:
        warn('R rsquared small, fit likely failed')

    if plot:
        fig, axs = plt.subplots(1,2,figsize=(12,5))
        mesh = axs[0].pcolormesh(tau, phi, data)
        fig.colorbar(mesh, ax=axs[0])
        axs[0].set_title('avgi')
        mesh = axs[1].pcolormesh(tau, phi, result.best_fit.reshape(data.shape))
        fig.colorbar(mesh, ax=axs[1])
        axs[1].set_title('best fit')
        for ax in axs:
            ax.set_xlabel('pulse length (us)')
            ax.set_ylabel('advance phase (deg)')

    return result

