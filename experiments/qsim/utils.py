from slab import AttrDict
from typing import Optional

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

