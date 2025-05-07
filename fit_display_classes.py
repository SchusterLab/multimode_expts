# %reload_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt

from qick import *
from qick.helpers import gauss
from tqdm import tqdm_notebook as tqdm

import time
import os
import sys
sys.path.append('/home/xilinx/jupyter_notebooks/')
sys.path.append('C:\\_Lib\\python\\rfsoc\\rfsoc_multimode\\example_expts')
import scipy as sp
import json
from scipy.fft import fft, fftfreq

from slab.instruments import *
from slab.experiment import Experiment
from slab.datamanagement import SlabFile
from slab import get_next_filename, AttrDict
from slab import Experiment, dsfit, AttrDict

# Figure params
plt.rcParams['figure.figsize'] = [10,6]
plt.rcParams.update({'font.size': 14})

from slab.dsfit import *
import os
from scipy.interpolate import griddata
from numpy import mgrid, array, zeros, abs, pi, cos, transpose, linspace
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tempfile import TemporaryFile
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 15}
import json
from h5py import File
from datetime import datetime
import time
from slab.datamanagement import SlabFile
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from slab.dsfit import *
from scipy.optimize import curve_fit
import experiments.fitting as fitter
from numpy.linalg import inv
import pandas as pd
from matplotlib.colors import Normalize

class DataProcessing:
    """
    Class for data normalization and filtering functions
    """
    @staticmethod
    def normalize_data(axi, axq, data, normalize): 
        '''
        Display avgi and avgq data with the g,e,f corresponding i,q values
        
        Parameters:
        -----------
        axi : matplotlib axis
            Axis for I data
        axq : matplotlib axis
            Axis for Q data
        data : dict
            Data dictionary containing I and Q values
        normalize : list
            List containing [bool, min_label, max_label]
            
        Returns:
        --------
        axi, axq : matplotlib axes
            Updated axes with normalized data
        '''
        # change tick labels
        # Get current y-axis ticks
        
        for idx, ax in enumerate([axi, axq]):
            ticks = ax.get_yticks()

            #set limits 
            ax.set_ylim(min(data[normalize[1]][idx], data[normalize[2]][idx]),
                        max(data[normalize[1]][idx], data[normalize[2]][idx]))
            #get tick labels
            ticks = ax.get_yticks()

            # Create new tick labels, replacing the first and last with custom text
            new_labels = list(ticks)#[item.get_text() for item in ax.get_xticklabels()]
            
            if data[normalize[1]][idx] > data[normalize[2]][idx] :
                new_labels[0] = normalize[2][0] # min
                new_labels[-1] = normalize[1][0] # max
            else:
                new_labels[0] = normalize[1][0] # min 
                new_labels[-1] = normalize[2][0] # max

            # Apply the new tick labels
            ax.set_yticks(ax.get_yticks().tolist()) # need to set this first 
            ax.set_yticklabels(new_labels)
        return axi, axq
    
    @staticmethod
    def filter_data(II, threshold, readout_per_experiment=2):
        '''
        Filter data based on threshold
        
        Parameters:
        -----------
        II : array
            Input data array
        threshold : float
            Threshold value for filtering
        readout_per_experiment : int
            Number of readouts per experiment
            
        Returns:
        --------
        result : array
            Filtered data
        '''
        # assume the last one is experiment data, the last but one is for post selection
        result = []
        
        for k in range(len(II) // readout_per_experiment):
            index_4k_plus_2 = readout_per_experiment * k + readout_per_experiment-2
            index_4k_plus_3 = readout_per_experiment * k + readout_per_experiment-1
            
            # Ensure the indices are within the list bounds
            if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
                # Check if the value at 4k+2 exceeds the threshold
                if II[index_4k_plus_2] > threshold:
                    # Add the value at 4k+3 to the result list
                    result.append(II[index_4k_plus_3])
        
        return result

    @staticmethod
    def filter_data_IQ(II, IQ, threshold, readout_per_experiment=2):
        '''
        Filter I and Q data based on threshold
        
        Parameters:
        -----------
        II : array
            I data array
        IQ : array
            Q data array
        threshold : float
            Threshold value for filtering
        readout_per_experiment : int
            Number of readouts per experiment
            
        Returns:
        --------
        result_Ig, result_Ie : arrays
            Filtered I and Q data
        '''
        # assume the last one is experiment data, the last but one is for post selection
        result_Ig = []
        result_Ie = []
        
        for k in range(len(II) // readout_per_experiment):
            index_4k_plus_2 = readout_per_experiment * k + readout_per_experiment-2
            index_4k_plus_3 = readout_per_experiment * k + readout_per_experiment-1
            
            # Ensure the indices are within the list bounds
            if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
                # Check if the value at 4k+2 exceeds the threshold
                if II[index_4k_plus_2] < threshold:
                    # Add the value at 4k+3 to the result list
                    result_Ig.append(II[index_4k_plus_3])
                    result_Ie.append(IQ[index_4k_plus_3])
        
        return np.array(result_Ig), np.array(result_Ie)
    
    @staticmethod
    def normalize_data1(data, min_val, max_val): 
        '''
        Normalize data to range [0, 1]
        
        Parameters:
        -----------
        data : array
            Data to normalize
        min_val : float
            Minimum value for normalization
        max_val : float
            Maximum value for normalization
            
        Returns:
        --------
        normalized_data : array
            Normalized data
        '''
        return (data - min_val) / (max_val - min_val)
    
    @staticmethod
    def post_select_raverager_data(temp_data, attrs, threshold, readouts_per_rep):
        '''
        Post-select and average data based on threshold
        
        Parameters:
        -----------
        temp_data : dict
            Data dictionary
        attrs : dict
            Attributes dictionary
        threshold : float
            Threshold value for filtering
        readouts_per_rep : int
            Number of readouts per repetition
            
        Returns:
        --------
        Ilist, Qlist : arrays
            Post-selected and averaged I and Q data
        '''
        read_num = readouts_per_rep
        rounds = attrs['config']['expt']['rounds']
        reps = attrs['config']['expt']['reps']
        expts = attrs['config']['expt']['expts']
        I_data = np.array(temp_data['idata'])
        Q_data = np.array(temp_data['qdata'])

        # reshape data into (read_num x rounds x reps x expts)
        I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds*reps * read_num))
        Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds*reps * read_num))

        # now we do post selection
        Ilist = []
        Qlist = []
        for ii in range(len(I_data)-1):
            Ig, Qg = DataProcessing.filter_data_IQ(I_data[ii], Q_data[ii], threshold, readout_per_experiment=read_num)
            #print(len(Ig))
            Ilist.append(np.mean(Ig))
            Qlist.append(np.mean(Qg))

        return Ilist, Qlist
    
    @staticmethod
    def post_select_averager_data(data, threshold, readout_per_round=4):
        '''
        Post select the data based on the threshold 
        
        Parameters:
        -----------
        data : array
            Input data array
        threshold : float
            Threshold value for filtering
        readout_per_round : int
            Number of readouts per round
            
        Returns:
        --------
        Ilist, Qlist : arrays
            Post-selected and averaged I and Q data
        '''
        Ilist = []
        Qlist = []
        for ii in range(len(data)):
            Ig, Qg = DataProcessing.filter_data_IQ(data[ii], data[ii], threshold, readout_per_experiment=readout_per_round)
            Ilist.append(np.mean(Ig))
            Qlist.append(np.mean(Qg))
        return Ilist, Qlist
    
    @staticmethod
    def filter_data_single_shot(prev_data, file_list, name='_single_shot_phase_sweep.h5', threshold=-20, active_reset=True, expt_path=''):
        '''
        Returns active reset filtered data 
        
        Parameters:
        -----------
        prev_data : function
            Function to load previous data
        file_list : list
            List of file numbers
        name : str
            File name pattern
        threshold : float
            Threshold value for filtering
        active_reset : bool
            Whether active reset is used
        expt_path : str
            Path to experiment data
            
        Returns:
        --------
        y_list : array
            Filtered data
        '''
        y_list = []
        for file_no in file_list:
            count = 0   # how many points possibly |f>
            full_name = str(file_no).zfill(5)+name
            temp_data, attrs = prev_data(expt_path, full_name)  # ef

            # active reset:
            if active_reset:
                Ig, Qg = DataProcessing.filter_data_IQ(temp_data['I'], temp_data['Q'], threshold, readout_per_experiment=4)
                total_counts = len(Ig)
                # Ig = I_selected
            else:
                Ig = temp_data['I']
                Qg = temp_data['Q']
                total_counts = len(Ig)
            for i in range(len(Ig)):
                if Ig[i] < threshold:
                    count += 1

            y_list.append(count/total_counts)
        return y_list
    
    @staticmethod
    def parity_post_select_modified(data, attrs, readout_threshold, readouts_per_rep):
        '''
        Post select the data based on the threshold, every readouts_per_rep readouts
        (based on preselection measurement pulse during active reset)
        
        Parameters:
        -----------
        data : dict
            Data dictionary
        attrs : dict
            Attributes dictionary
        readout_threshold : float
            Threshold value for readout
        readouts_per_rep : int
            Number of readouts per repetition
            
        Returns:
        --------
        Ilist, Qlist : arrays
            Post-selected I and Q data
        '''
        print('calling parity post select modified')
        Ilist = []
        Qlist = []

        rounds = attrs['config']['expt']['rounds']
        reps = attrs['config']['expt']['reps']
        expts = attrs['config']['expt']['expts']

        I_data = data['idata'] # in shape rounds(1) x expts (1)  x reps   x read_num
        Q_data = data['qdata'] 

        # assume we have made 80 parity measurements
        # reshape data into (reps, read_num)
        read_num = readouts_per_rep 
        I_data_rs = np.reshape(I_data, (reps, read_num))
        Q_data_rs = np.reshape(Q_data, (reps, read_num))

        # for every rep, we have a list of [...83 elements ...]
        # we want to get the first 80 elements of each rep iff the 3rd element is within threshold 

        for idx in range(reps):
            rep_array = I_data_rs[idx] # has 83 elements
            if rep_array[2] < readout_threshold:
                # get the first 80 elements
                Ilist.append(rep_array[3:])
                Qlist.append(Q_data_rs[idx][3:])
        
        return Ilist, Qlist
    
    @staticmethod
    def consistency_post_selection(meas_records, cutoff_n=3): 
        '''
        Check if measurement record is consistent with the parity measurement
        up to cutoff_n measurements
        
        Parameters:
        -----------
        meas_records : list
            List of measurement records
        cutoff_n : int
            Cutoff number for consistency check (should be >2)
            
        Returns:
        --------
        new_records : list
            List of consistent records
        '''
        new_records = []
        for record in meas_records:
            single_photon = False
            consistent_check = True
            if record[0] != record[1]:
                single_photon = True  # this record contains a single photon
            
            current_n = 2
            while current_n <= cutoff_n:
                
                current_idx = current_n - 1
                if single_photon:  # photon detected, so the record should be flipping
                    if record[current_idx-1] == record[current_idx]:
                        consistent_check = False
                        break
                else: # no photon, so the record should be the same
                    if record[current_idx-1] != record[current_idx]:
                        consistent_check = False
                        break
                current_n += 1
            if consistent_check:
                new_records.append(record)
        return new_records


class HistogramAnalysis:
    """
    Class for histogram-related functions
    """
    @staticmethod
    def hist(data, plot=True, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4):
        """
        Generate histogram of I/Q data
        
        Parameters:
        -----------
        data : dict
            Data dictionary
        plot : bool
            Whether to plot the histogram
        span : float or None
            Span for histogram limits
        verbose : bool
            Whether to print verbose output
        active_reset : bool
            Whether active reset is used
        readout_per_round : int
            Number of readouts per round
        threshold : float
            Threshold value for filtering
            
        Returns:
        --------
        fids : list
            List of fidelities
        thresholds : list
            List of thresh
