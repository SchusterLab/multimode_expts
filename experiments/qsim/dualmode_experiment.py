import os
from typing import Optional, Union

from qick import QickConfig
from slab import AttrDict, Experiment
from experiments.qsim.utils import read_hdf5_data


class ReadWriteExperiment(Experiment):
    """
    Slightly modifies the Experiment __init__ to allow reading exisitng hdf5 data
    Ideally you shouldn't have to override the methods defined here at all
    """
    def __init__(self, 
                 soccfg: Optional[QickConfig]=None, 
                 path: str='', 
                 prefix: str='',
                 config_file: Optional[str]=None, 
                 expt_params: Union[dict,AttrDict,None]=None,
                 progress: bool=False,
                 read_mode: bool=False, 
                 hdf5_file: Optional[str]=None):
        """
        Initialize an experiment object in two possible modes:
            - read_mode=False: to start a new measurement.
            - read_mode=True: to read an existing hdf5 file and make use of the analysis and plotting methods
        """
        if not read_mode:
            prefix = prefix or self.__class__.__name__ # prefix defaults to class name
            super().__init__(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file, progress=progress)
            self.cfg.expt = AttrDict(expt_params)
        else:
            if not hdf5_file:
                raise ValueError("Must provide hdf5 file in read mode")
            self.data, attrs = read_hdf5_data(path, hdf5_file)
            self.cfg = attrs.config
            self.fname = os.path.join(path, hdf5_file)


    def save_data(self, data=None):
        # do we really need to ovrride this?
        print(f'Saving {self.fname}')
        super().save_data(data=data)
        return self.fname


