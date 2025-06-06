import numpy as np
from slab import AttrDict
from typing import Optional
from scipy.fft import rfft, rfftfreq

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
    return np.abs(xf[peak_idx])

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

