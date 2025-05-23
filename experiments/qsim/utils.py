import json
import os
from typing import Optional, Tuple

import numpy as np
from slab import AttrDict, SlabFile


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


def read_hdf5_data(path: str, filename: str) -> Tuple[dict, AttrDict]:
    """
    Extracts data and attrs from hdf5
    Returns:
        data: dict of numpy arrays
        attrs: dict of parsed json
    """
    with SlabFile(os.path.join(path, filename)) as f:
        attrs = {}
        for key, value in f.attrs.items():
            attrs.update({key: json.loads(value)})
        data = {}
        for key, value in f.items():
            data.update({key: np.array(value)})
    return data, AttrDict(attrs)

