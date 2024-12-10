
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
from numpy import mgrid, array, zeros,abs,pi,cos,transpose, linspace
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

## Normalize Data 

def normalize_data(axi, axq, data, normalize): 
    '''
    Display avgi and avgq data with the g,e,f corresponding i,q values
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
def filter_data(II, threshold, readout_per_experiment=2):
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

def filter_data_IQ(II, IQ, threshold, readout_per_experiment=2):
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
## histgram
def hist(data, plot=True, span=None, verbose=True, active_reset=True, readout_per_round=2, threshold=-4):
    """
    span: histogram limit is the mean +/- span
    """
    # if active_reset:
    #     Ig = data['Ig'][readout_per_round-1::readout_per_round]
    #     Qg = data['Qg'][readout_per_round-1::readout_per_round]
    #     Ie = data['Ie'][readout_per_round-1::readout_per_round]
    #     Qe = data['Qe'][readout_per_round-1::readout_per_round]
    #     plot_f = False 
    #     if 'If' in data.keys():
    #         plot_f = True
    #         If = data['If'][readout_per_round-1::readout_per_round]
    #         Qf = data['Qf'][readout_per_round-1::readout_per_round]

    if active_reset:
        Ig, Qg = filter_data_IQ(data['Ig'], data['Qg'], threshold, readout_per_experiment=readout_per_round)
        # Qg = filter_data(data['Qg'], threshold, readout_per_experiment=readout_per_round)
        Ie, Qe = filter_data_IQ(data['Ie'], data['Qe'], threshold, readout_per_experiment=readout_per_round)
        # Qe = filter_data(data['Qe'], threshold, readout_per_experiment=readout_per_round)
        print(len(Ig))
        print(len(Ie))
        plot_f = False 
        if 'If' in data.keys():
            plot_f = True
            If, Qf = filter_data_IQ(data['If'], data['Qf'], threshold, readout_per_experiment=readout_per_round)
            # Qf = filter_data(data['Qf'], threshold, readout_per_experiment=readout_per_round)
            print(len(If))
    else:

        Ig = data['Ig']
        Qg = data['Qg']
        Ie = data['Ie']
        Qe = data['Qe']
        plot_f = False 
        if 'If' in data.keys():
            plot_f = True
            If = data['If']
            Qf = data['Qf']

    numbins = 200

    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f: xf, yf = np.median(If), np.median(Qf)

    if verbose:
        print('Unrotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')

    if plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
        fig.tight_layout()
        
        axs[0,0].scatter(Ie, Qe, label='e', color='r', marker='.', s=1)
        axs[0,0].scatter(Ig, Qg, label='g', color='b', marker='.', s=1)
        
        if plot_f: axs[0,0].scatter(If, Qf, label='f', color='g', marker='.', s=1)
        axs[0,0].scatter(xg, yg, color='k', marker='o')
        axs[0,0].scatter(xe, ye, color='k', marker='o')
        if plot_f: axs[0,0].scatter(xf, yf, color='k', marker='o')

        axs[0,0].set_xlabel('I [ADC levels]')
        axs[0,0].set_ylabel('Q [ADC levels]')
        axs[0,0].legend(loc='upper right')
        axs[0,0].set_title('Unrotated')
        axs[0,0].axis('equal')

    """Compute the rotation angle"""
    theta = -np.arctan2((ye-yg),(xe-xg))
    if plot_f: theta = -np.arctan2((ye-yf),(xe-xf))

    """Rotate the IQ data"""
    Ig_new = Ig*np.cos(theta) - Qg*np.sin(theta)
    Qg_new = Ig*np.sin(theta) + Qg*np.cos(theta) 

    Ie_new = Ie*np.cos(theta) - Qe*np.sin(theta)
    Qe_new = Ie*np.sin(theta) + Qe*np.cos(theta)

    if plot_f:
        If_new = If*np.cos(theta) - Qf*np.sin(theta)
        Qf_new = If*np.sin(theta) + Qf*np.cos(theta)

    """New means of each blob"""
    xg, yg = np.median(Ig_new), np.median(Qg_new)
    xe, ye = np.median(Ie_new), np.median(Qe_new)
    if plot_f: xf, yf = np.median(If_new), np.median(Qf_new)
    if verbose:
        print('Rotated:')
        print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')
        print(f'Ie {xe} +/- {np.std(Ie)} \t Qe {ye} +/- {np.std(Qe)} \t Amp e {np.abs(xe+1j*ye)}')
        if plot_f: print(f'If {xf} +/- {np.std(If)} \t Qf {yf} +/- {np.std(Qf)} \t Amp f {np.abs(xf+1j*yf)}')


    if span is None:
        span = (np.max(np.concatenate((Ie_new, Ig_new))) - np.min(np.concatenate((Ie_new, Ig_new))))/2
    xlims = [xg-span, xg+span]
    ylims = [yg-span, yg+span]

    if plot:
        axs[0,1].scatter(Ig_new, Qg_new, label='g', color='b', marker='.', s=1)
        axs[0,1].scatter(Ie_new, Qe_new, label='e', color='r', marker='.', s=1)
        if plot_f: axs[0, 1].scatter(If_new, Qf_new, label='f', color='g', marker='.', s=1)
        axs[0,1].scatter(xg, yg, color='k', marker='o')
        axs[0,1].scatter(xe, ye, color='k', marker='o')    
        if plot_f: axs[0, 1].scatter(xf, yf, color='k', marker='o')    

        axs[0,1].set_xlabel('I [ADC levels]')
        axs[0,1].legend(loc='upper right')
        axs[0,1].set_title('Rotated')
        axs[0,1].axis('equal')

        """X and Y ranges for histogram"""

        ng, binsg, pg = axs[1,0].hist(Ig_new, bins=numbins, range = xlims, color='b', label='g', alpha=0.5, density=True)
        ne, binse, pe = axs[1,0].hist(Ie_new, bins=numbins, range = xlims, color='r', label='e', alpha=0.5, density=True)
        if plot_f:
            nf, binsf, pf = axs[1,0].hist(If_new, bins=numbins, range = xlims, color='g', label='f', alpha=0.5, density=True)
        axs[1,0].set_ylabel('Counts')
        axs[1,0].set_xlabel('I [ADC levels]')       
        axs[1,0].legend(loc='upper right')

    else:        
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

    """Compute the fidelity using overlap of the histograms"""
    fids = []
    thresholds = []
    contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5*ng.sum() + 0.5*ne.sum())))
    tind=contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])

    confusion_matrix = [np.cumsum(ng)[tind]/ng.sum(),
                        1-np.cumsum(ng)[tind]/ng.sum(),
                        np.cumsum(ne)[tind]/ne.sum(),
                        1-np.cumsum(ne)[tind]/ne.sum()]   # Pgg (prepare g measured g), Pge (prepare g measured e), Peg, Pee
    if plot_f:
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(nf)) / (0.5*ng.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        contrast = np.abs(((np.cumsum(ne) - np.cumsum(nf)) / (0.5*ne.sum() + 0.5*nf.sum())))
        tind=contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
        
    if plot: 
        axs[1,0].set_title(f'Histogram (Fidelity g-e: {100*fids[0]:.3}%)')
        axs[1,0].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,0].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,0].axvline(thresholds[2], color='0.2', linestyle='--')

        axs[1,1].set_title('Cumulative Counts')
        axs[1,1].plot(binsg[:-1], np.cumsum(ng), 'b', label='g')
        axs[1,1].plot(binse[:-1], np.cumsum(ne), 'r', label='e')
        axs[1,1].axvline(thresholds[0], color='0.2', linestyle='--')
        if plot_f:
            axs[1,1].plot(binsf[:-1], np.cumsum(nf), 'g', label='f')
            axs[1,1].axvline(thresholds[1], color='0.2', linestyle='--')
            axs[1,1].axvline(thresholds[2], color='0.2', linestyle='--')
        axs[1,1].legend()
        axs[1,1].set_xlabel('I [ADC levels]')
        
        plt.subplots_adjust(hspace=0.25, wspace=0.15)        
        plt.show()

    return fids, thresholds, theta*180/np.pi, confusion_matrix # fids: ge, gf, ef

def hist_analyze(data, span=None, verbose=True, **kwargs):
        
        fids, thresholds, angle, confusion_matrix = hist(data=data, plot=False, span=span, verbose=verbose)
        data['fids'] = fids
        data['angle'] = angle
        data['thresholds'] = thresholds
        data['confusion_matrix'] = confusion_matrix
        
        return data

def hist_display(data, span=None, verbose=True, plot_e=True, plot_f=False, active_reset=True, readout_per_round=2, thresholds=-4, **kwargs):
    
    fids, thresholds_new, angle, confusion_matrix = hist(data=data, plot=True, active_reset=active_reset, threshold=thresholds, readout_per_round=readout_per_round, verbose=verbose, span=span)
        
    print(f'ge fidelity (%): {100*fids[0]}')

    if plot_f:
        print(f'gf fidelity (%): {100*fids[1]}')
        print(f'ef fidelity (%): {100*fids[2]}')
    print(f'rotation angle (deg): {angle}')
    # print(f'set angle to (deg): {-angle}')
    print(f'threshold ge: {thresholds_new[0]}')
    print('Confusion matrix [Pgg, Pge, Peg, Pee]: ',confusion_matrix)
    if plot_f:
        print(f'threshold gf: {thresholds_new[1]}')
        print(f'threshold ef: {thresholds_new[2]}')

def hist_display_sweep(data, span=None, verbose=True, plot_e=True, plot_f=False, **kwargs):
    
    fids, thresholds, angle, confusion_matrix = hist(data=data, plot=False, verbose=verbose, span=span)
        
    # print(f'ge fidelity (%): {100*fids[0]}')

    # if plot_f:
        # print(f'gf fidelity (%): {100*fids[1]}')
        # print(f'ef fidelity (%): {100*fids[2]}')
    # print(f'rotation angle (deg): {angle}')
    # print(f'set angle to (deg): {-angle}')
    # print(f'threshold ge: {thresholds[0]}')
    # print('Confusion matrix [Pgg, Pge, Peg, Pee]: ',confusion_matrix)
    # if plot_f:
        # print(f'threshold gf: {thresholds[1]}')
        # print(f'threshold ef: {thresholds[2]}')
    return fids, thresholds, angle, confusion_matrix
def hist_new(data, threshold1=0, plot=True, span=None, verbose=True,active_reset=True, readout_per_round=2, threshold=-4):
    """
    span: histogram limit is the mean +/- span

    """
    
    if active_reset:
        Ig, Qg = filter_data_IQ(data['I'], data['Q'], threshold, readout_per_experiment=readout_per_round)
    else:
        print('here')
        Ig = data['I']
        Qg = data['Q']



    xg, yg = np.median(Ig), np.median(Qg)

    print(f'Ig {xg} +/- {np.std(Ig)} \t Qg {yg} +/- {np.std(Qg)} \t Amp g {np.abs(xg+1j*yg)}')



    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        fig.tight_layout()
        axs.scatter(Ig, Qg, label='g', color='b', marker='.', s=0.5)
        axs.axvline(threshold1, color='r', linestyle='--')


        axs.set_xlabel('I [ADC levels]')
        axs.set_ylabel('Q [ADC levels]')
        axs.legend(loc='upper right')
        axs.set_title('Unrotated')
        axs.axis('equal')
    return Ig

def hist_new_display(data, threshold1, span=None, verbose=True, plot_e=True, plot_f=False, active_reset=True, readout_per_round=4, threshold=-4, **kwargs):
    counts = 0
    counts_neg = 0
    I_selected = hist_new(data=data, threshold1=threshold1, plot=True, verbose=verbose, span=span, active_reset=active_reset, readout_per_round=readout_per_round, threshold=threshold)
    total_counts = len(I_selected)
    for i in range(total_counts):
        if I_selected[i] > threshold1:
            counts += 1
        else:
            counts_neg += 1
    print(counts_neg)
    print(f'|e> population (%): {100*counts/total_counts}')
    
        
## t1

# def t1_display(data, fit=True, title="$T_1$", **kwargs):
#     data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
#     data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
#     data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
    
#     plt.figure(figsize=(10,10))
#     plt.subplot(211, title=title, ylabel="I [ADC units]")
#     plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
#     if fit:
#         p = data['fit_avgi']
#         pCov = data['fit_err_avgi']
#         captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
#         plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
#         plt.xlabel('Time [us]')
#         plt.legend()
#         print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
#     plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
#     plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
#     if fit:
#         p = data['fit_avgq']
#         pCov = data['fit_err_avgq']
#         captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
#         plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
#         plt.xlabel('Time [us]')
#         plt.legend()
#         print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')

#     plt.show()
## Phase Sweep 
def phase_sweep_display(temp_data, attrs, normalize=[False, 'g_data', 'e_data'], fit = True, fitparams=None):
        # make 3 subplots with first one showing amps, second one showing avgi, and third one showing avgq
    # Sample data
    data = temp_data.copy()

    if fit:
        # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
        # Remove the first and last point from fit in case weird edge measurements
        # fitparams = None
        # fitparams=[8, 0.5, 0, 20, None, None]
        p_avgi, pCov_avgi = fitter.fitsin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
        p_avgq, pCov_avgq = fitter.fitsin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
        p_amps, pCov_amps = fitter.fitsin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
        data['fit_avgi'] = p_avgi   
        data['fit_avgq'] = p_avgq
        data['fit_amps'] = p_amps
        data['fit_err_avgi'] = pCov_avgi   
        data['fit_err_avgq'] = pCov_avgq
        data['fit_err_amps'] = pCov_amps


    x = temp_data['xpts']
    amps = temp_data['amps']
    avgi = temp_data['avgi']
    avgq = temp_data['avgq']

    # Create a figure and three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot amps on the first subplot
    ax1.plot(x, amps, label='Amps')
    ax1.set_title('Amps')
    ax1.set_xlabel('Phase(deg)')
    ax1.legend()

    # Plot avgi on the second subplot
    ax2.plot(x, avgi, label='AvgI', color='orange')
    ax2.plot(data["xpts"][:-1], fitter.sinfunc(data["xpts"][:-1], *data['fit_avgi']), label='fit')
    #add a vertical line at min of avgi 
    ydata = fitter.sinfunc(data["xpts"][:-1], *data['fit_avgi'])
    #print fit_avgi data element wise
    fit_params = ['amp', 'freq', 'phase_offset', '...']
    for i, element in enumerate(data['fit_avgi']):
        print(f'{fit_params[i]}: {element}')
    #print advances phase 
    advanced_phase = attrs['config']['expt']['zz_phase']
    print(f'ge phase per identity: {advanced_phase}')
    ax2.axvline(x[np.argmin(ydata)], color='r', linestyle='--', label = 'Min AvgI: ' + str(round(x[np.argmin(ydata)], 2)))
    ax2.set_title('AvgI')
    ax2.set_xlabel('Phase (deg)')
    ax2.legend()

    # Plot avgq on the third subplot
    ax3.plot(x, avgq, label='AvgQ', color='green')
    ax3.set_title('AvgQ')
    ax3.set_xlabel('Phase (deg)')
    ax3.legend()

    if normalize[0]:
        ax2,ax3 = normalize_data(ax2, ax3, temp_data, normalize)

    # add advance phase to plot title 
    ax1.set_title('Amps' + ' ge phase per identity: ' + str(advanced_phase))
    # Adjust layout
    plt.tight_layout()

    # Display the plots
    plt.show()
## Ramsey
def filter_data(II, threshold, readout_per_experiment=2):
    # assume the last one is experiment data, the last but one is for post selection
    result = []
    
    
    for k in range(len(II) // readout_per_experiment):
        index_4k_plus_2 = readout_per_experiment * k + readout_per_experiment-2
        index_4k_plus_3 = readout_per_experiment * k + readout_per_experiment-1
        
        # Ensure the indices are within the list bounds
        if index_4k_plus_2 < len(II) and index_4k_plus_3 < len(II):
            # Check if the value at 4k+2 exceeds the threshold
            if II[index_4k_plus_2] < threshold:
                # Add the value at 4k+3 to the result list
                result.append(II[index_4k_plus_3])
    
    return result

def filter_data_IQ(II, IQ, threshold, readout_per_experiment=2):
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
def post_select_raverager_data(temp_data, attrs, threshold, readouts_per_rep):
    read_num = readouts_per_rep
    rounds = attrs['config']['expt']['rounds']
    reps = attrs['config']['expt']['reps']
    expts = attrs['config']['expt']['expts']
    I_data = np.array(temp_data['idata'])
    Q_data = np.array(temp_data['qdata'])

    # reshape data into (read_num x rounds x reps x expts)
    # I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (3, 0, 2, 1)), (read_num, rounds*reps, expts))
    # Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (3, 0, 2, 1)), (read_num, rounds*reps, expts))

    I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds*reps * read_num))
    Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds*reps * read_num))

    # ## Read tomography code for reshaping if rounds neq 1
    # I_data_ = np.reshape(I_data, (xpts, reps*read_num))
    # Q_data_ = np.reshape(Q_data, (xpts, reps*read_num))

    # now we do post selection
    Ilist = []
    Qlist = []
    for ii in range(len(I_data)-1):
        Ig, Qg = filter_data_IQ(I_data[ii], Q_data[ii], threshold, readout_per_experiment=read_num)
        #print(len(Ig))
        Ilist.append(np.mean(Ig))
        Qlist.append(np.mean(Qg))


    return Ilist, Qlist
# def plot_sideband_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, normalize = None, title="Sideband Sweep",
#                         active_reset=False, readout_per_round=4, threshold=-4.0, xlabel='Time (us)', ylabel='Alpha '):
#     plt.figure(figsize = (15,6))
#     plt.subplot(111,xlabel='Time (us)',ylabel='Drive freq (MHz)')
#     plt.title(title)

#     if active_reset:
#         # for each time, post select the single shot data
#         znew = []
        
#         for jj in range(len(z_datalist)):
#             Ilist = []
#             for ii in range(len(z_datalist[jj])):
#                 Ig, Qg = filter_data_IQ(z_datalist[jj][ii], z_datalist[jj][ii], threshold, readout_per_experiment=readout_per_round)
#                 Ilist.append(np.mean(Ig))
#             print(len(Ig))
#             znew.append(Ilist)
#         znew = np.array(znew)
        
#     else:
        
#         znew = np.array(z_datalist)


#     cax = plt.pcolormesh(x_timelist,y_freqlist[:znew.shape[0]],znew)
#     cbar = plt.colorbar()
    
    
#     # if normalize: 
#     #     # New normalization limits
#     #     new_min, new_max = 0, 1

#     #     # Apply new normalization
#     #     norm = Normalize(vmin=new_min, vmax=new_max)
#     #     cax.set_norm(norm)
#     #     # Update colorbar to reflect new normalization
#     #     cbar.set_ticks([0.0, 1.0])  # Set ticks at min and max of Z
#     #     cbar.set_ticklabels([r'$g$', r'$f$'])  # Custom labels


#     if vlines is not None:
#         for vline in vlines:
#             plt.axvline(vline, color='r', ls='--', label = str(vline))
#     if hlines is not None:
#         for hline in hlines:
#             plt.axhline(hline, color='r', ls='--', label = str(hline))
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.tight_layout()
def plot_ramsey_sideband(data_list, attrs_list, y_list,
                   active_reset = False, threshold = 4, readouts_per_rep = 4, title='Ramsey', xlabel='Time (us)', ylabel='Alpha', hlines=None, vlines=None):
    '''
    Plots chevron for kerr measurement with ylist as alpha, x list as time, and z as I value 
    '''

    Ilist_full = []
    
    for idx, data in enumerate(data_list): 
        if active_reset:
            attrs = attrs_list[idx]
            Ilist, Qlist = post_select_raverager_data(data, attrs, threshold, readouts_per_rep)
            data['avgi'] = Ilist
            data['avgq'] = Qlist
            data['xpts'] = data['xpts'][:-1]
            data['amps'] = data['amps'][:-1] # adjust since active reset throws away the last data point
            Ilist_full.append(Ilist)
        else:
            Ilist_full.append(data['avgi'])
    z_list = np.array(Ilist_full)
    x_list = data_list[0]['xpts']
    cax = plt.pcolormesh(x_list,y_list,z_list)
    cbar = plt.colorbar()

    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--', label = str(vline))
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--', label = str(hline))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    return x_list, y_list, z_list
    
    
def Ramsey_display(data, attrs, ramsey_freq=0.02, initial_freq=3500, fit=True, fitparams = None, normalize= [False, 'g_data', 'e_data'], 
                   active_reset = False, threshold = 4, readouts_per_rep = 4, return_idata = False, return_all_param = True, title='Ramsey',
                   end_idx = None, start_idx = None):
    '''
    Returns_all_param = True: returns all the parameters of the fit i 

    '''
    try: 
        if attrs['config']['expt']['echoes'][0]: # if there are echoes
            print('Echoes in the data')
            print(data['xpts'][:5])
            data['xpts'] *= (1 + attrs['config']['expt']['echoes'][1]) # multiply by the number of echoes
            print(data['xpts'][:5])
        else:
            print('No echoes in the data')
    except KeyError:
        print('No echoes in the data')
        pass
    if active_reset:
        Ilist, Qlist = post_select_raverager_data(data, attrs, threshold, readouts_per_rep)
        data['avgi'] = Ilist[start_idx:end_idx]
        data['avgq'] = Qlist[start_idx:end_idx]
        data['xpts'] = data['xpts'][:-1][start_idx:end_idx]
        data['amps'] = data['amps'][:-1][start_idx :end_idx] # adjust since active reset throws away the last data point
    if fit:
        # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
        # Remove the first and last point from fit in case weird edge measurements
        # fitparams = None
        # fitparams=[8, 0.5, 0, 20, None, None]
        p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][1:-1], data["avgi"][1:-1], fitparams=fitparams)
        p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][1:-1], data["avgq"][1:-1], fitparams=fitparams)
        p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][1:-1], data["amps"][1:-1], fitparams=fitparams)
        data['fit_avgi'] = p_avgi   
        data['fit_avgq'] = p_avgq
        data['fit_amps'] = p_amps
        data['fit_err_avgi'] = pCov_avgi   
        data['fit_err_avgq'] = pCov_avgq
        data['fit_err_amps'] = pCov_amps

        if isinstance(p_avgi, (list, np.ndarray)): data['f_adjust_ramsey_avgi'] = sorted((ramsey_freq - p_avgi[1], ramsey_freq + p_avgi[1]), key=abs)
        if isinstance(p_avgq, (list, np.ndarray)): data['f_adjust_ramsey_avgq'] = sorted((ramsey_freq - p_avgq[1], ramsey_freq + p_avgq[1]), key=abs)
        if isinstance(p_amps, (list, np.ndarray)): data['f_adjust_ramsey_amps'] = sorted((ramsey_freq - p_amps[1], ramsey_freq + p_amps[1]), key=abs)

    f_pi_test = initial_freq
    title = title
    plt.figure(figsize=(10,9))
    axi = plt.subplot(211, 
        title=f"{title} (Ramsey Freq: {ramsey_freq} MHz)",
        ylabel="I [ADC level]")
    plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
    if fit:
        p = data['fit_avgi']
        # print(p)
        if isinstance(p, (list, np.ndarray)): 
            pCov = data['fit_err_avgi']
            captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.legend()
            print(f'Current pi pulse frequency: {f_pi_test}')
            print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
            if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
            print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
            print(f'T2 Ramsey from fit I [us]: {p[3]}')
            t2 = p[3]
            t2_err = np.sqrt(pCov[3][3])

    axq = plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
    plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
    if fit:
        p = data['fit_avgq']
        if isinstance(p, (list, np.ndarray)): 
            pCov = data['fit_err_avgq']
            captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label=captionStr)
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.legend()
            print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
            if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
            print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                    f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
            print(f'T2 Ramsey from fit Q [us]: {p[3]}')
    
    if normalize[0]:
        axi,axq = normalize_data(axi, axq, data, normalize)
    
    
        
    plt.tight_layout()
    plt.show()
    if fit:
        if return_all_param: 
            return p_avgi, pCov_avgi, data["xpts"][1:-1], data["avgi"][1:-1]
        if return_idata:
            return t2, t2_err, data["xpts"][1:-1], data["avgi"][1:-1]
        else: 
            return t2, t2_err
def multiple_Ramsey_display(prev_data, file_list, label_list, color_list, 
                             active_reset = False, threshold = 4, readouts_per_rep = 4,
                             ramsey_freq=0.02, initial_freq=3500, fit=True, fitparams = None, normalize= [False, 'g_data', 'e_data'], title='Ramsey'):

    ### figure plot
    plt.figure(figsize=(10,9))
    axi = plt.subplot(211, 
        title=f"{title} ",
        ylabel="I [ADC level]")
    axq = plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
    fit_i_list = []
    fit_i_err_list = []
    
    for idx, file_name in enumerate(file_list):
        #full_name = str(file_no).zfill(5)+'_CavityRamseyExperiment.h5'
        data, attrs = prev_data(expt_path, file_name)
        label = label_list[idx]
        color = color_list[idx]

        if active_reset:
            Ilist, Qlist = post_select_raverager_data(data, attrs, threshold, readouts_per_rep)
            data['avgi'] = Ilist
            data['avgq'] = Qlist
            data['xpts'] = data['xpts'][:-1]
            data['amps'] = data['amps'][:-1] # adjust since active reset throws away the last data point
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = None
            # fitparams=[8, 0.5, 0, 20, None, None]
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            if isinstance(p_avgi, (list, np.ndarray)): data['f_adjust_ramsey_avgi'] = sorted((ramsey_freq - p_avgi[1], ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)): data['f_adjust_ramsey_avgq'] = sorted((ramsey_freq - p_avgq[1], ramsey_freq + p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)): data['f_adjust_ramsey_amps'] = sorted((ramsey_freq - p_amps[1], ramsey_freq + p_amps[1]), key=abs)

        f_pi_test = initial_freq

        
        axi.plot(data["xpts"][:-1], data["avgi"][:-1],'o-', color=color)
        if fit:
            p = data['fit_avgi']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgi']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                axi.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label = label , color=color)
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                fit_i_list.append(p[1])
                fit_i_err_list.append(np.sqrt(pCov[1][1]))
                if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        axi.legend()
       
        axq.plot(data["xpts"][:-1], data["avgq"][:-1],'o-', color=color)
        if fit:
            p = data['fit_avgq']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgq']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                axq.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label = label , color=color)
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')
        axq.legend()
    
        if normalize[0]:
            axi,axq = normalize_data(axi, axq, data, normalize)
    plt.legend()
    for idx, label in enumerate(label_list):
        print(label)
        print(f'Fit frequency from I [MHz]: {fit_i_list[idx]} +/- {fit_i_err_list[idx]}')
    print('----- Difference from original ------')
    for idx, label in enumerate(label_list):
        print(label)
        print(f'Fit frequency from I [MHz]: {fit_i_list[idx] - fit_i_list[0]} +/- {fit_i_err_list[idx]}')
        
        
    plt.tight_layout()
    plt.show()

    return np.array(fit_i_list) - fit_i_list[0], np.array(fit_i_err_list)

# def parity_mapping_display(data, fit=True, fitparams=None,
#                             title='sideband_rabi', active_reset = False, threshold = 4, readouts_per_rep = 4):
        
#     # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
#     # Remove the first and last point from fit in case weird edge measurements
#     # fitparams = [None, 1/max(data['xpts']), None, None]
#     # fitparams = None

#     if active_reset:
#         Ilist, Qlist = post_select_averager_data(data['idata'], threshold, readouts_per_rep)
        
#     else:
        
#         Ilist = data["avgi"][0:-1]
#         Qlist = data["avgq"][0:-1]

#     p_avgi, pCov_avgi = fitter.fitdecaysin(
#         data['xpts'], Ilist, fitparams=fitparams)
#     p_avgq, pCov_avgq = fitter.fitdecaysin(
#         data['xpts'], Qlist, fitparams=fitparams)
#     # p_amps, pCov_amps = fitter.fitdecaysin(
#     #     data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
#     data['fit_avgi'] = p_avgi
#     data['fit_avgq'] = p_avgq
#     # data['fit_amps'] = p_amps
#     data['fit_err_avgi'] = pCov_avgi
#     data['fit_err_avgq'] = pCov_avgq
#     # data['fit_err_amps'] = pCov_amps

#     xpts_ns = data['xpts']*1e3

#     plt.figure(figsize=(10, 8))

#     plt.subplot(
#         211, title=title, ylabel="I [adc level]")
#     plt.plot(xpts_ns[1:-1], data["avgi"][1:-1], 'o-')
#     if fit:
#         p = data['fit_avgi']
#         plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
#         if p[2] > 180:
#             p[2] = p[2] - 360
#         elif p[2] < -180:
#             p[2] = p[2] + 360
#         if p[2] < 0:
#             pi_length = (1/2 - p[2]/180)/2/p[1]
#         else:
#             pi_length = (3/2 - p[2]/180)/2/p[1]
#         pi2_length = pi_length/2
#         print('Decay from avgi [us]', p[3])
#         print(f'Pi length from avgi data [us]: {pi_length}')
#         print(f'\tPi/2 length from avgi data [us]: {pi2_length}')
#         plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
#         plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')

#     print()
#     plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
#     plt.plot(xpts_ns[1:-1], data["avgq"][1:-1], 'o-')
#     if fit:
#         p = data['fit_avgq']
#         plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
#         if p[2] > 180:
#             p[2] = p[2] - 360
#         elif p[2] < -180:
#             p[2] = p[2] + 360
#         if p[2] < 0:
#             pi_length = (1/2 - p[2]/180)/2/p[1]
#         else:
#             pi_length = (3/2 - p[2]/180)/2/p[1]
#         pi2_length = pi_length/2
#         print('Decay from avgq [us]', p[3])
#         print(f'Pi length from avgq data [us]: {pi_length}')
#         print(f'Pi/2 length from avgq data [us]: {pi2_length}')
#         plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
#         plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
#     plt.tight_layout()
#     plt.show()


def cross_kerr_display(expt_path, prev_data, file_list, label_list, color_list,  orig_idx = 0,
                             active_reset = False, threshold = 4, readouts_per_rep = 4,
                             ramsey_freq=0.02, initial_freq=3500, fit=True, fitparams = None, normalize= [False, 'g_data', 'e_data'], title='Ramsey'):

    ### figure plot
    plt.figure(figsize=(10,9))
    axi = plt.subplot(211, 
        title=f"{title} ",
        ylabel="I [ADC level]")
    axq = plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC level]")
    fit_i_list = []
    fit_i_err_list = []
    
    for idx, file_no in enumerate(file_list):
        full_name = str(file_no).zfill(5)+'_cross_kerr_sweep.h5'
        print(full_name)
        data, attrs = prev_data(expt_path, full_name)
        label = label_list[idx]
        color = color_list[idx]

        if active_reset:
            Ilist, Qlist = post_select_raverager_data(data, attrs, threshold, readouts_per_rep)
            data['avgi'] = Ilist
            data['avgq'] = Qlist
            data['xpts'] = data['xpts'][:-1]
            data['amps'] = data['amps'][:-1] # adjust since active reset throws away the last data point
        
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = None
            # fitparams=[8, 0.5, 0, 20, None, None]
            p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
            p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
            p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
            data['fit_avgi'] = p_avgi   
            data['fit_avgq'] = p_avgq
            data['fit_amps'] = p_amps
            data['fit_err_avgi'] = pCov_avgi   
            data['fit_err_avgq'] = pCov_avgq
            data['fit_err_amps'] = pCov_amps

            if isinstance(p_avgi, (list, np.ndarray)): data['f_adjust_ramsey_avgi'] = sorted((ramsey_freq - p_avgi[1], ramsey_freq + p_avgi[1]), key=abs)
            if isinstance(p_avgq, (list, np.ndarray)): data['f_adjust_ramsey_avgq'] = sorted((ramsey_freq - p_avgq[1], ramsey_freq + p_avgq[1]), key=abs)
            if isinstance(p_amps, (list, np.ndarray)): data['f_adjust_ramsey_amps'] = sorted((ramsey_freq - p_amps[1], ramsey_freq + p_amps[1]), key=abs)

        f_pi_test = initial_freq

        
        axi.plot(data["xpts"][:-1], data["avgi"][:-1],'o-', color=color)
        if fit:
            p = data['fit_avgi']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgi']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                axi.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label = label , color=color)
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.legend()
                print(f'Current pi pulse frequency: {f_pi_test}')
                print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                fit_i_list.append(p[1])
                fit_i_err_list.append(np.sqrt(pCov[1][1]))
                if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequency from fit I [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgi"][1]}')
                print(f'T2 Ramsey from fit I [us]: {p[3]}')
        axi.legend()
       
        axq.plot(data["xpts"][:-1], data["avgq"][:-1],'o-', color=color)
        if fit:
            p = data['fit_avgq']
            if isinstance(p, (list, np.ndarray)): 
                pCov = data['fit_err_avgq']
                captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
                axq.plot(data["xpts"][:-1], fitter.decaysin(data["xpts"][:-1], *p), label = label , color=color)
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.plot(data["xpts"][:-1], fitter.expfunc(data['xpts'][:-1], p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
                #plt.legend()
                print(f'Fit frequency from Q [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
                if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
                print('Suggested new pi pulse frequencies from fit Q [MHz]:\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][0]}\n',
                        f'\t{f_pi_test + data["f_adjust_ramsey_avgq"][1]}')
                print(f'T2 Ramsey from fit Q [us]: {p[3]}')
        axq.legend()
    
        if normalize[0]:
            axi,axq = normalize_data(axi, axq, data, normalize)
    plt.legend()
    for idx, label in enumerate(label_list):
        print(label)
        print(f'Fit frequency from I [MHz]: {fit_i_list[idx]} +/- {fit_i_err_list[idx]}')
    print('----- Difference from original ------')
    for idx, label in enumerate(label_list):
        print(label)
        new_err = np.sqrt(fit_i_err_list[idx]**2 + fit_i_err_list[orig_idx]**2)
        print(f'Fit frequency from I [MHz]: {fit_i_list[idx] - fit_i_list[orig_idx]} +/- {new_err}')
        
        
    plt.tight_layout()
    plt.show()

    return np.array(fit_i_list) - fit_i_list[orig_idx], np.array(fit_i_err_list)
## Amplitude Rabi
def amp_display(data, sigma = 0.04, fit=True, fitparams=None, vline = None, normalize = [False, 'g_data', 'e_data'], **kwargs):
    if fit:
        # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
        # Remove the first and last point from fit in case weird edge measurements
        xdata = data['xpts']

        p_avgi, pCov_avgi = fitter.fitdecaysin(data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
        p_avgq, pCov_avgq = fitter.fitdecaysin(data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
        p_amps, pCov_amps = fitter.fitdecaysin(data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
        data['fit_avgi'] = p_avgi   
        data['fit_avgq'] = p_avgq
        data['fit_amps'] = p_amps
        data['fit_err_avgi'] = pCov_avgi   
        data['fit_err_avgq'] = pCov_avgq
        data['fit_err_amps'] = pCov_amps


    plt.figure(figsize=(10,10))
    axi = plt.subplot(211, title=f"Amplitude Rabi (Pulse Length {sigma})", ylabel="I [ADC units]")
    plt.plot(data["xpts"][1:-1], data["avgi"][1:-1],'o-')
    if fit:
        p = data['fit_avgi']
        plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        if p[2] > 180: p[2] = p[2] - 360
        elif p[2] < -180: p[2] = p[2] + 360
        if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        period = 1/p[1]
        pi2_gain = pi_gain - period/4 
        # pi2_gain = pi_gain/2
        print(f'Pi gain from avgi data [dac units]: {int(pi_gain)}')
        print(f'\tPi/2 gain from avgi data [dac units]: {int(pi2_gain)}')
        # print(f'\tPi/2 gain from avgi data [dac units]: {int(1/4/p[1])}')
        plt.axvline(pi_gain, color='0.2', linestyle='--')
        plt.axvline(pi2_gain, color='0.2', linestyle='--')
        if not(vline==None):
            plt.axvline(vline, color='0.2', linestyle='--')
    axq = plt.subplot(212, xlabel="Gain [DAC units]", ylabel="Q [ADC units]")
    plt.plot(data["xpts"][1:-1], data["avgq"][1:-1],'o-')
    if fit:
        p = data['fit_avgq']
        plt.plot(data["xpts"][0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
        if p[2] > 180: p[2] = p[2] - 360
        elif p[2] < -180: p[2] = p[2] + 360
        if p[2] < 0: pi_gain = (1/2 - p[2]/180)/2/p[1]
        else: pi_gain= (3/2 - p[2]/180)/2/p[1]
        period = 1/p[1]
        pi2_gain = pi_gain - period/4 
        print(f'Pi gain from avgq data [dac units]: {int(pi_gain)}')
        print(f'\tPi/2 gain from avgq data [dac units]: {int(pi2_gain)}')
        # print(f'\tPi/2 gain from avgq data [dac units]: {int(1/4/p[1])}')
        plt.axvline(pi_gain, color='0.2', linestyle='--')
        plt.axvline(pi2_gain, color='0.2', linestyle='--')
    if normalize[0]:
        axi,axq = normalize_data(axi, axq, data, normalize)

    plt.show()
## Cavity spectroscopy
def spectroscopy_display(data, fit=True, signs=[1,1,1], hlines=None, vlines=None,  title='SPectroscopy'):
    xdata = data['xpts'][1:-1]
    data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
    data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
    data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])

    xpts = data['xpts'][1:-1]

    plt.figure(figsize=(9, 11))
    plt.subplot(311, title=title, ylabel="Amplitude [ADC units]")
    plt.plot(xpts, data["amps"][1:-1],'o-')
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--') 
    if fit:
        plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
        print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

    plt.subplot(312, ylabel="I [ADC units]")
    plt.plot(xpts, data["avgi"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
        print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
    plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
    plt.plot(xpts, data["avgq"][1:-1],'o-')
    # plt.axvline(3476, c='k', ls='--')
    # plt.axvline(3376+50, c='k', ls='--')
    # plt.axvline(3376, c='k', ls='--')
    if fit:
        plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
        # plt.axvline(3593.2, c='k', ls='--')
        print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

    plt.tight_layout()
    plt.show()
    
def cavity_spec_display(data, findpeaks=False, verbose=True, fitparams=None, fit=True):
    if fit:
        # fitparams = [f0, Qi, Qe, phi, scale]
        xdata = data["xpts"][1:-1]
        # ydata = data["avgi"][1:-1] + 1j*data["avgq"][1:-1]
        ydata = data['amps'][1:-1]
        fitparams=fitparams
        data['fit'], data['fit_err'] = fitter.fithanger(xdata, ydata, fitparams=fitparams)
        if isinstance(data['fit'], (list, np.ndarray)):
            f0, Qi, Qe, phi, scale, a0, slope = data['fit']
            if verbose:
                print(f'\nFreq with minimum transmission: {xdata[np.argmin(ydata)]}')
                print(f'Freq with maximum transmission: {xdata[np.argmax(ydata)]}')
                print('From fit:')
                print(f'\tf0: {f0}')
                print(f'\tQi: {Qi}')
                print(f'\tQe: {Qe}')
                print(f'\tQ0: {1/(1/Qi+1/Qe)}')
                print(f'\tkappa [MHz]: {f0*(1/Qi+1/Qe)}')
                print(f'\tphi [radians]: {phi}')
        
    if findpeaks:
        maxpeaks, minpeaks = dsfit.peakdetect(data['amps'][1:-1], x_axis=data['xpts'][1:-1], lookahead=30, delta=5*np.std(data['amps'][:5]))
        data['maxpeaks'] = maxpeaks
        data['minpeaks'] = minpeaks

    xpts = data['xpts'][1:-1]

    plt.figure(figsize=(16,16))
    plt.subplot(311, title=f"Cavity Spectroscopy",  ylabel="Amps [ADC units]")
    plt.plot(xpts, data['amps'][1:-1],'o-')
    if fit:
        plt.plot(xpts, fitter.hangerS21func_sloped(data["xpts"][1:-1], *data["fit"]))
    if findpeaks:
        # for peak in np.concatenate((data['maxpeaks'], data['minpeaks'])):
        for peak in data['minpeaks']:
            plt.axvline(peak[0], linestyle='--', color='0.2')
            print(f'Found peak [MHz]: {peak[0]}')
    # plt.axvline(float(self.cfg.hw.lo.readout.frequency)*1e-6 + self.cfg.device.readout.lo_sideband*(self.cfg.hw.soc.dacs.readout.mixer_freq + 812.37), c='k', ls='--')
    # plt.axvline(7687.5, c='k', ls='--')

    plt.subplot(312, xlabel="Cavity Frequency [MHz]", ylabel="I [ADC units]")
    plt.plot(xpts, data["avgi"][1:-1],'o-')

    plt.subplot(313, xlabel="Cavity Frequency [MHz]", ylabel="Phases [ADC units]")
    plt.plot(xpts, data["phases"][1:-1],'o-')
    plt.show()
## Qubit spectroscopy
def qubit_spectroscopy_display(data, fit=True, signs=[1,1,1], hlines=None, vlines=None,  title='Qubit SPectroscopy'):
    xdata = data['xpts'][1:-1]
    data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
    data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
    data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])

    xpts = data['xpts'][1:-1]

    plt.figure(figsize=(9, 11))
    plt.subplot(311, title=title, ylabel="Amplitude [ADC units]")
    plt.plot(xpts, data["amps"][1:-1],'o-')
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--') 
    if fit:
        plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
        print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

    plt.subplot(312, ylabel="I [ADC units]")
    plt.plot(xpts, data["avgi"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
        print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
    plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
    plt.plot(xpts, data["avgq"][1:-1],'o-')
    # plt.axvline(3476, c='k', ls='--')
    # plt.axvline(3376+50, c='k', ls='--')
    # plt.axvline(3376, c='k', ls='--')
    if fit:
        plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
        # plt.axvline(3593.2, c='k', ls='--')
        print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')

    plt.tight_layout()
    plt.show()

def plot_spectroscopy_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, title="f0-g1 spectroscopy Sweep"):
    plt.figure(figsize = (15,6))
    plt.subplot(111,xlabel='Frequency (MHz)',ylabel='DC flux bias (mA)')
    plt.title(title)
    plt.pcolormesh(x_timelist,y_freqlist[:z_datalist.shape[0]],z_datalist)
    cbar = plt.colorbar()
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--')   

def plot_spectroscopy1(xdata_g, g_data, xdata_e, e_data, fitparams=None, title="Readout"):
    fitparams=fitparams
    data1 = {}
    data1['fit'], data1['fit_err'] = fitter.fithanger(xdata_g, g_data, fitparams=fitparams)
    f0, Qi, Qe, phi, scale, a0, slope = data1['fit']
    print('f0 at |g>:', f0)
    ## separate |e> data into two parts, individually fit peaks
    data2_e1 = {}
    data2_e1['fit'], data2_e1['fit_err'] = fitter.fithanger(xdata_e, e_data, fitparams=fitparams)
    f0, Qi, Qe, phi, scale, a0, slope = data2_e1['fit']
    print('f0 at |e>:', f0)
    plt.figure(figsize=(10,4))
    plt.subplot(111, title=title,  ylabel="Amps [ADC units]")
    plt.plot(xdata_g, g_data,'.',label='Qubit |g>', color='r')
    plt.plot(xdata_g, fitter.hangerS21func_sloped(xdata_g, *data1["fit"]), '--', color='r', label='Fit |g>')
    plt.plot(xdata_e, e_data,'.',label='Qubit |e>', color='b')
    plt.plot(xdata_e, fitter.hangerS21func_sloped(xdata_e, *data2_e1["fit"]), '--', color='b', label='Fit |e>')
    plt.legend()
## cavity DC sweep
def plot_cavity_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, title="Sideband Sweep"):
    plt.figure(figsize = (15,6))
    plt.subplot(111,xlabel='Freq (MHz)',ylabel='Flux quanta')
    plt.title(title)
    plt.pcolormesh(x_timelist,y_freqlist[:z_datalist.shape[0]],z_datalist)
    cbar = plt.colorbar()
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--')
## ECD pulse spectroscopy
def ECD_pulse_spectroscopy( data=None, fit=True, signs=[1,1,1], fitparams = None,normalize= [False, 'g_data', 'e_data'], **kwargs):
    #if data is None:
    data=data 

    # if 'mixer_freq' in cfg.hw.soc.dacs.qubit:
    #     xpts = cfg.hw.soc.dacs.qubit.mixer_freq + data['xpts'][1:-1]
    # else: 
    
    xpts = data['xpts'][1:-1]
    xdata = xpts #redundancy

    #Fit parameters
    signs=[1,1,1]
    data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
    data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
    data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])
        

    plt.figure(figsize=(9, 11))
    plt.subplot(311, title=f"Qubit  Spectroscopy", ylabel="Amplitude [ADC units]")
    plt.plot(xpts, data["amps"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
        print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')

    axi = plt.subplot(312, ylabel="I [ADC units]")
    plt.plot(xpts, data["avgi"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
        print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
    axq = plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
    plt.plot(xpts, data["avgq"][1:-1],'o-')
    # plt.axvline(3476, c='k', ls='--')
    # plt.axvline(3376+50, c='k', ls='--')
    # plt.axvline(3376, c='k', ls='--')
    if fit:
        plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
        # plt.axvline(3593.2, c='k', ls='--')
        print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')
        
    if normalize[0]:
        axi,axq = normalize_data(axi, axq, data, normalize)

    plt.tight_layout()
    plt.show()
## length rabi
# def length_rabi_display(data, fit=True, fitparams=None, title='sideband_rabi', vlines = []):
        
#     # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
#     # Remove the first and last point from fit in case weird edge measurements
#     # fitparams = [None, 1/max(data['xpts']), None, None]
#     # fitparams = None
#     p_avgi, pCov_avgi = fitter.fitdecaysin(
#         data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)
#     p_avgq, pCov_avgq = fitter.fitdecaysin(
#         data['xpts'][:-1], data["avgq"][:-1], fitparams=fitparams)
#     p_amps, pCov_amps = fitter.fitdecaysin(
#         data['xpts'][:-1], data["amps"][:-1], fitparams=fitparams)
#     data['fit_avgi'] = p_avgi
#     data['fit_avgq'] = p_avgq
#     data['fit_amps'] = p_amps
#     data['fit_err_avgi'] = pCov_avgi
#     data['fit_err_avgq'] = pCov_avgq
#     data['fit_err_amps'] = pCov_amps

#     xpts_ns = data['xpts']*1e3

#     plt.figure(figsize=(10, 8))

#     plt.subplot(
#         211, title=title, ylabel="I [adc level]")
#     plt.plot(xpts_ns[1:-1], data["avgi"][1:-1], 'o-')
#     if fit:
#         p = data['fit_avgi']
#         plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
#         if p[2] > 180:
#             p[2] = p[2] - 360
#         elif p[2] < -180:
#             p[2] = p[2] + 360
#         if p[2] < 0:
#             pi_length = (1/2 - p[2]/180)/2/p[1]
#         else:
#             pi_length = (3/2 - p[2]/180)/2/p[1]
#         period = 1/p[1]
#         pi2_length = pi_length - period/4 #fitter.decaysin(, *p)#pi_length/2
#         print('Decay from avgi [us]', p[3])
#         print(f'Pi length from avgi data [us]: {pi_length}')
#         print(f'\tPi/2 length from avgi data [us]: {pi2_length}')
#         plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
#         plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
#         for vline in vlines:
#             plt.axvline(vline, color='0.2', linestyle='--')

#     print()
#     plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
#     plt.plot(xpts_ns[1:-1], data["avgq"][1:-1], 'o-')
#     if fit:
#         p = data['fit_avgq']
#         plt.plot(xpts_ns[0:-1], fitter.decaysin(data["xpts"][0:-1], *p))
#         if p[2] > 180:
#             p[2] = p[2] - 360
#         elif p[2] < -180:
#             p[2] = p[2] + 360
#         if p[2] < 0:
#             pi_length = (1/2 - p[2]/180)/2/p[1]
#         else:
#             pi_length = (3/2 - p[2]/180)/2/p[1]
#         pi2_length = pi_length/2
#         print('Decay from avgq [us]', p[3])
#         print(f'Pi length from avgq data [us]: {pi_length}')
#         print(f'Pi/2 length from avgq data [us]: {pi2_length}')
#         plt.axvline(pi_length*1e3, color='0.2', linestyle='--')
#         plt.axvline(pi2_length*1e3, color='0.2', linestyle='--')
#     plt.tight_layout()
#     plt.show()
def phase_calibration(xpts, ypts, fitparams=None, vlines=None, hlines=None, min_pi = False,
                       title='Phase Calibration', xlabel='Phase (deg)', ylabel='|g> population'):
    '''
    Specifically for RBAM code for cross storage phase calib 

    If min_pi is True, then the pi length is calculated at the minimum of the sin curve
    '''
    data = {}
    #xpts_ns = xpts[0:len(ypts)]
    xpts_ns = xpts
    p_avgi, pCov_avgi = fitter.fitsin(
        xpts_ns, ypts, fitparams=None)
    print(p_avgi)
    data['fit_avgi'] = p_avgi
    data['fit_err_avgi'] = pCov_avgi

    # print frequency 
    print(f'Frequency from fit [MHz]: {p_avgi[1]} +/- {np.sqrt(pCov_avgi[1][1])}')


    # xpts_ns = vary_pts[0:len(ypts)]
    plt.figure(figsize=(10, 4))

    plt.subplot(
        111, title=title, ylabel="|g> population", xlabel=xlabel)
    plt.plot(xpts_ns[0:-1], ypts[0:-1], 'o-')
    p = data['fit_avgi']
    plt.plot(xpts_ns[0:-1], fitter.sinfunc(xpts_ns[0:-1], *p))
    if p[2] > 180:
        p[2] = p[2] - 360
    elif p[2] < -180:
        p[2] = p[2] + 360
    if p[2] < 0:
        pi_length = (1/2 - p[2]/180)/2/p[1]
    else:
        pi_length = (3/2 - p[2]/180)/2/p[1]
    
    # to make sure pi length occurs at minimum of sin not maximum (specifcally for RBAM code)
    if min_pi:
        phase_max = xpts_ns[0:-1][np.argmax(fitter.sinfunc(xpts_ns[0:-1], *p))] 
        if np.abs(pi_length - phase_max) < 30: 
            pi_length = pi_length + 180
        
        print(f'Pi length from avgi data: {pi_length}')
        print(f'phase at max: {phase_max}')
    # ---------------------------------------------
    pi2_length = pi_length/2
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--')   
    print(f'Pi length from avgi data: {pi_length}')
    print(f'\tPi/2 length from avgi data: {pi2_length}')
    plt.axvline(pi_length, color='0.2', linestyle='--')
    # plt.close()
    return pi_length
## RB
def RB_extract(temp_data, conf=False):
    temp_data['confusion_matrix'] = [0.9922999999999998, 0.007700000000000151, 0.024050000000000002, 0.97595]
    avg_readout = []
    for i in range(len(temp_data['Idata'])):
        counting = 0
        for j in temp_data['Idata'][i]:
            if j<temp_data['thresholds']:
                counting += 1
        g_out = counting/len(temp_data['Idata'][i])
        if conf:
            P_matrix = np.matrix([[temp_data['confusion_matrix'][0], temp_data['confusion_matrix'][2]],[temp_data['confusion_matrix'][1], temp_data['confusion_matrix'][3]]])
            
            e_out = 1-g_out
            counts_new = inv(P_matrix)*np.matrix([[g_out],[e_out]])
            g_out = counts_new[0,0]
        avg_readout.append(g_out)
    return avg_readout
def plot_lengthrabi_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, title="Sideband Sweep",
                        active_reset=False, readout_per_round=4, threshold=-4.0, factor=1.0):
    plt.figure(figsize = (15,6*factor))
    plt.subplot(111,xlabel='Time (us)',ylabel='Drive freq (MHz)')
    plt.title(title)

    if active_reset:
        # for each time, post select the single shot data
        znew = []
        
        for jj in range(len(z_datalist)):
            Ilist = []
            for ii in range(len(z_datalist[jj])):
                Ig, Qg = filter_data_IQ(z_datalist[jj][ii], z_datalist[jj][ii], threshold, readout_per_experiment=readout_per_round)
                Ilist.append(np.mean(Ig))
            # print(len(Ig))
            znew.append(Ilist)
        znew = np.array(znew)
        
    else:
        
        znew = np.array(z_datalist)


    cax = plt.pcolormesh(x_timelist,y_freqlist[:znew.shape[0]],znew)
    cbar = plt.colorbar()
    


    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--', label = str(vline))
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--', label = str(hline))
    plt.legend()
    plt.tight_layout()
# def plot_sideband_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, normalize = None, title="Sideband Sweep", xlabel = None, ylabel = None):
#     plt.figure(figsize = (15,6))
#     plt.subplot(111,xlabel=xlabel,ylabel=ylabel)
#     plt.title(title)
#     cax = plt.pcolormesh(x_timelist,y_freqlist[:z_datalist.shape[0]],z_datalist)
#     cbar = plt.colorbar()



#     if vlines is not None:
#         for vline in vlines:
#             plt.axvline(vline, color='r', ls='--', label = str(vline))
#     if hlines is not None:
#         for hline in hlines:
#             plt.axhline(hline, color='r', ls='--', label = str(hline))
#     plt.legend()
#     plt.tight_layout()


# ---------------------------------------------------------------------------------------------------------------------
# Plotting for Sideband Analysis 
# ---------------------------------------------------------------------------------------------------------------------

## length_rabi f0-g1
def normalize_data(axi, axq, data, normalize): 
    '''
    Display avgi and avgq data with the g,e,f corresponding i,q values
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
def filter_data(II, threshold, readout_per_experiment=2):
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

def filter_data_IQ(II, IQ, threshold, readout_per_experiment=2):
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
def post_select_averager_data(data, threshold, readout_per_round=4):
    '''
    Post select the data based on the threshold 
    '''
    Ilist = []
    Qlist = []
    for ii in range(len(data)):
        Ig, Qg = filter_data_IQ(data[ii], data[ii], threshold, readout_per_round)
        Ilist.append(np.mean(Ig))
        Qlist.append(np.mean(Qg))
    return Ilist, Qlist
def length_rabi_display(data, fit=True, fitparams=None,  normalize= [False, 'g_data', 'e_data'], vlines = None, title='sideband_rabi',
                        active_reset=True, readout_per_round=4, threshold=-4.0, 
                        return_fit_params = False, fit_sin = False):
        
    xlist = data['xpts'][0:-1]
    if active_reset:
        try: 
            Ilist, Qlist = post_select_averager_data(data['Idata'][:-1], threshold, readout_per_round)
        except KeyError: 
            Ilist, Qlist = post_select_averager_data(data['idata'][:-1], threshold, readout_per_round)
        
    else:
        
        Ilist = data["avgi"][0:-1]
        Qlist = data["avgq"][0:-1]

    if fit_sin == True:
        fit_func = fitter.fitsin
        func = fitter.sinfunc
    else:
        fit_func = fitter.fitdecaysin
        func = fitter.decaysin
    

    p_avgi, pCov_avgi = fit_func(
        xlist, Ilist, fitparams=fitparams)
    p_avgq, pCov_avgq = fit_func(
        xlist, Qlist, fitparams=fitparams)

    data['fit_avgi'] = p_avgi
    data['fit_avgq'] = p_avgq

    data['fit_err_avgi'] = pCov_avgi
    data['fit_err_avgq'] = pCov_avgq


    xpts_ns = data['xpts']*1e3

    plt.figure(figsize=(10, 8))

    axi = plt.subplot(
        211, title=title, ylabel="I [adc level]")
    plt.plot(xpts_ns[1:-1], Ilist[1:], 'o-')
    if fit:
        p = data['fit_avgi']
        plt.plot(xpts_ns[0:-1], func(xlist, *p))
        if p[2] > 180:
            p[2] = p[2] - 360
        elif p[2] < -180:
            p[2] = p[2] + 360
        if p[2] < 0:
            pi_length = (1/2 - p[2]/180)/2/p[1]
        else:
            pi_length = (3/2 - p[2]/180)/2/p[1]
        
        T = 1/p[1]/2

        pi2_length = pi_length- (T/2)
        data['pi_length'] = pi_length
        data['pi2_length'] = pi2_length
        print(p)
        print('Decay from avgi [us]', p[3])
        print('Rate [MHz]', p[1])
        print(f'Pi length from avgi data [us]: {pi_length}')
        print(f'\tPi/2 length from avgi data [us]: {pi2_length}')
        # plot vline at pi and pi/2 length
        plt.axvline(pi_length*1e3, color='0.2', linestyle='--', label = 'pi')
        plt.axvline(pi2_length*1e3, color='0.2', linestyle='--',    label = 'pi/2')
        if vlines is not None:
            for vline in vlines:
                plt.axvline(vline, color='r', ls='--')
        print('Fit params: ', p)

    print()
    axq = plt.subplot(212, xlabel="Pulse length [ns]", ylabel="Q [adc levels]")
    plt.plot(xpts_ns[1:-1], Qlist[1:], 'o-')
    if fit:
        p = data['fit_avgq']
        plt.plot(xpts_ns[0:-1], func(xlist, *p))
        if p[2] > 180:
            p[2] = p[2] - 360
        elif p[2] < -180:
            p[2] = p[2] + 360
        if p[2] < 0:
            pi_length = (1/2 - p[2]/180)/2/p[1]
        else:
            pi_length = (3/2 - p[2]/180)/2/p[1]
        pi2_length = pi_length/2
        print('Decay from avgq [us]', p[3])
        print('Rate [MHz]', p[1])
        print(f'Pi length from avgq data [us]: {pi_length}')
        print(f'Pi/2 length from avgq data [us]: {pi2_length}')
        plt.axvline(pi_length*1e3, color='0.2', linestyle='--', label = 'pi')
        plt.axvline(pi2_length*1e3, color='0.2', linestyle='--',    label = 'pi/2')
        if vlines is not None:
            for vline in vlines:
                plt.axvline(vline, color='r', ls='--')
        print('Fit params: ', p)
    
    if normalize[0]:
        axi,axq = normalize_data(axi, axq, data, normalize)
    plt.tight_layout()
    plt.legend()
    plt.show()

    if return_fit_params: 
        return p_avgi, pCov_avgi, xlist, Ilist


    return Ilist
## sideband rabi sweep
def plot_sideband_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, normalize = None, title="Sideband Sweep",
                        active_reset=False, readout_per_round=4, threshold=-4.0, factor=1.0):
    plt.figure(figsize = (15,6*factor))
    plt.subplot(111,xlabel='Time (us)',ylabel='Drive freq (MHz)')
    plt.title(title)

    if active_reset:
        # for each time, post select the single shot data
        znew = []
        
        for jj in range(len(z_datalist)):
            Ilist = []
            for ii in range(len(z_datalist[jj])):
                Ig, Qg = filter_data_IQ(z_datalist[jj][ii], z_datalist[jj][ii], threshold, readout_per_experiment=readout_per_round)
                Ilist.append(np.mean(Ig))
            # print(len(Ig))
            znew.append(Ilist)
        znew = np.array(znew)
        
    else:
        
        znew = np.array(z_datalist)


    cax = plt.pcolormesh(x_timelist,y_freqlist[:znew.shape[0]],znew)
    cbar = plt.colorbar()
    
    
    if normalize: 
        # New normalization limits
        new_min, new_max = 0, 1

        # Apply new normalization
        norm = Normalize(vmin=new_min, vmax=new_max)
        cax.set_norm(norm)
        # Update colorbar to reflect new normalization
        cbar.set_ticks([0.0, 1.0])  # Set ticks at min and max of Z
        cbar.set_ticklabels([r'$g$', r'$f$'])  # Custom labels


    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--', label = str(vline))
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--', label = str(hline))
    plt.legend()
    plt.tight_layout()
    
def normalize_data1(data, min_val, max_val): 
    return (data - min_val) / (max_val - min_val)
def post_select_raverager_data(temp_data, attrs, threshold, readouts_per_rep):
    read_num = readouts_per_rep
    rounds = attrs['config']['expt']['rounds']
    reps = attrs['config']['expt']['reps']
    expts = attrs['config']['expt']['expts']
    I_data = np.array(temp_data['idata'])
    Q_data = np.array(temp_data['qdata'])

    # reshape data into (read_num x rounds x reps x expts)
    # I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (3, 0, 2, 1)), (read_num, rounds*reps, expts))
    # Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (3, 0, 2, 1)), (read_num, rounds*reps, expts))

    I_data = np.reshape(np.transpose(np.reshape(I_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds*reps * read_num))
    Q_data = np.reshape(np.transpose(np.reshape(Q_data, (rounds, expts, reps, read_num)), (1, 0, 2, 3)), (expts, rounds*reps * read_num))

    # ## Read tomography code for reshaping if rounds neq 1
    # I_data_ = np.reshape(I_data, (xpts, reps*read_num))
    # Q_data_ = np.reshape(Q_data, (xpts, reps*read_num))

    # now we do post selection
    Ilist = []
    Qlist = []
    for ii in range(len(I_data)-1):
        Ig, Qg = filter_data_IQ(I_data[ii], Q_data[ii], threshold, readout_per_experiment=read_num)
        #print(len(Ig))
        Ilist.append(np.mean(Ig))
        Qlist.append(np.mean(Qg))


    return Ilist, Qlist
def t1_display(data, attrs, fit=True, active_reset = True, threshold = -4, readouts_per_rep = 4, title="$T_1$", **kwargs):
    if active_reset:
        Ilist, Qlist = post_select_raverager_data(data, attrs, threshold, readouts_per_rep)
        data['avgi'] = Ilist
        data['avgq'] = Qlist
        data['xpts'] = data['xpts'][:-1]
        data['amps'] = data['amps'][:-1] # adjust since active reset throws away the last data point
    
    data['fit_amps'], data['fit_err_amps'] = fitter.fitexp(data['xpts'][:-1], data['amps'][:-1], fitparams=None)
    data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
    data['fit_avgq'], data['fit_err_avgq'] = fitter.fitexp(data['xpts'][:-1], data['avgq'][:-1], fitparams=None)
    
    plt.figure(figsize=(10,10))
    plt.subplot(211, title=title, ylabel="I [ADC units]")
    plt.plot(data["xpts"][:-1], data["avgi"][:-1],'o-')
    if fit:
        p = data['fit_avgi']
        pCov = data['fit_err_avgi']
        captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgi"]), label=captionStr)
        plt.xlabel('Time [us]')
        plt.legend()
        print(f'Fit T1 avgi [us]: {data["fit_avgi"][3]}')
    plt.subplot(212, xlabel="Wait Time [us]", ylabel="Q [ADC units]")
    plt.plot(data["xpts"][:-1], data["avgq"][:-1],'o-')
    if fit:
        p = data['fit_avgq']
        pCov = data['fit_err_avgq']
        captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
        plt.plot(data["xpts"][:-1], fitter.expfunc(data["xpts"][:-1], *data["fit_avgq"]), label=captionStr)
        plt.xlabel('Time [us]')
        plt.legend()
        print(f'Fit T1 avgq [us]: {data["fit_avgq"][3]}')
        plt.show()
        return p[3], np.sqrt(pCov[3][3])

    plt.show()
def plot_sideband_sweep_long(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, title="Sideband Sweep"):
    plt.figure(figsize = (15,8))
    plt.subplot(111,ylabel='Time (us)',xlabel='Drive freq (MHz)')
    plt.title(title)
    plt.pcolormesh(y_freqlist[:z_datalist.shape[0]],x_timelist,z_datalist.T)
    cbar = plt.colorbar()
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--')

def plot_f0g1_sweep(x_timelist, y_freqlist, z_datalist, hlines=None, vlines=None, title="Sideband Sweep"):
    plt.figure(figsize = (15,8))
    plt.subplot(111,ylabel='Current (mA)',xlabel='Drive freq (MHz)')
    plt.title(title)
    plt.pcolormesh(x_timelist,y_freqlist[:z_datalist.shape[0]],z_datalist)
    cbar = plt.colorbar()
    if vlines is not None:
        for vline in vlines:
            plt.axvline(vline, color='r', ls='--')
    if hlines is not None:
        for hline in hlines:
            plt.axhline(hline, color='r', ls='--')
## Spectroscopy
def qubit_spectroscopy_display(data, fit=True, signs=[1,1,1], title='Qubit SPectroscopy', vlines=None):
    xdata = data['xpts'][1:-1]
    data['fit_amps'], data['fit_err_amps'] = fitter.fitlor(xdata, signs[0]*data['amps'][1:-1])
    data['fit_avgi'], data['fit_err_avgi'] = fitter.fitlor(xdata, signs[1]*data['avgi'][1:-1])
    data['fit_avgq'], data['fit_err_avgq'] = fitter.fitlor(xdata, signs[2]*data['avgq'][1:-1])

    xpts = data['xpts'][1:-1]

    plt.figure(figsize=(9, 11))
    plt.subplot(311, title=title, ylabel="Amplitude [ADC units]")
    plt.plot(xpts, data["amps"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[0]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_amps"]))
        print(f'Found peak in amps at [MHz] {data["fit_amps"][2]}, HWHM {data["fit_amps"][3]}')
    if vlines:
        for vline in vlines:
            plt.axvline(vline, c='k', ls='--')

    plt.subplot(312, ylabel="I [ADC units]")
    plt.plot(xpts, data["avgi"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[1]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgi"]))
        print(f'Found peak in I at [MHz] {data["fit_avgi"][2]}, HWHM {data["fit_avgi"][3]}')
    if vlines:
        for vline in vlines:
            plt.axvline(vline, c='k', ls='--')

    plt.subplot(313, xlabel="Pulse Frequency (MHz)", ylabel="Q [ADC units]")
    plt.plot(xpts, data["avgq"][1:-1],'o-')
    if fit:
        plt.plot(xpts, signs[2]*fitter.lorfunc(data["xpts"][1:-1], *data["fit_avgq"]))
        print(f'Found peak in Q at [MHz] {data["fit_avgq"][2]}, HWHM {data["fit_avgq"][3]}')
    if vlines:
        for vline in vlines:
            plt.axvline(vline, c='k', ls='--')

    plt.tight_layout()
    plt.show()

def plot_spectroscopy1(xdata_g, g_data, xdata_e, e_data, fitparams=None, title="Readout", vlines=None):
    fitparams=fitparams
    data1 = {}
    data1['fit'], data1['fit_err'] = fitter.fithanger(xdata_g, g_data, fitparams=fitparams)
    f0, Qi, Qe, phi, scale, a0, slope = data1['fit']
    print('f0 at |g>:', f0)
    ## separate |e> data into two parts, individually fit peaks
    data2_e1 = {}
    data2_e1['fit'], data2_e1['fit_err'] = fitter.fithanger(xdata_e, e_data, fitparams=fitparams)
    f0, Qi, Qe, phi, scale, a0, slope = data2_e1['fit']
    print('f0 at |e>:', f0)
    plt.figure(figsize=(10,4))
    plt.subplot(111, title=title,  ylabel="Amps [ADC units]")
    plt.plot(xdata_g, g_data,'.',label='Qubit |g>', color='r')
    plt.plot(xdata_g, fitter.hangerS21func_sloped(xdata_g, *data1["fit"]), '--', color='r', label='Fit |g>')
    plt.plot(xdata_e, e_data,'.',label='Qubit |e>', color='b')
    plt.plot(xdata_e, fitter.hangerS21func_sloped(xdata_e, *data2_e1["fit"]), '--', color='b', label='Fit |e>')
    if vlines:
        for vline in vlines:
            plt.axvline(vline, c='k', ls='--')
    plt.legend()
    plt.show()
## Randomized Benchamarking BeamSplitter Check
import numpy as np
import matplotlib.pyplot as plt

class MM_DualRail_Analysis: 

    def __init__(self): 
        '''Analysis for dual rail experiments '''
    
    # ------------- Analyis for Storage state in presence of spectator BS ------------

    
    def filter_data_for_si_wrt_spec_BS(self, temp_data, attrs, threshold=None): 
        '''
        Filter data (based on active reset preselection) for single rail wrt to spectator BS
        '''
        if threshold == None:
            threshold = temp_data['thresholds']
        avg_idata = []
        for aa, var in enumerate(temp_data['Idata']):
            var_data, _ =  self.filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], None, threshold,post_selection = False)
            avg_data = np.mean(var_data, axis=0) # average wrt to shots
            # print(len(var_data))
            avg_idata.append(avg_data)
            # print(threshold)
        
        bs_gate_nums = attrs['config']['expt']['bs_gate_nums']
        rb_times = attrs['config']['expt']['rb_times']
        
        return avg_idata, bs_gate_nums, rb_times # the second average is over variations
    
    def extract_ramsey_seq_check_target(self, prev_data, expt_path, file_list, name, 
                                        wrong_way=False):
        '''
        Extract the data for the ramsey sequence as a function of gates 

        wrong_way: averaging over a single depth; ignoring that within a given depth, multiple times are covered
        '''

        var_datas = []
        # depth_list = []
        bs_gate_numss = []
        rb_timess = []

        wrong_fids = []
        wrong_bs_gate_nums = []
        wrong_times = []
        depth_list = []
        threshold = 0
        fnot_found_err_bool  = False

        for idx, file_no in enumerate(file_list): 
            try: 
                full_name = str(file_no).zfill(5)+name
                # print(full_name)
                temp_data, attrs = prev_data(expt_path, full_name)
                # analysis = MM_DualRail_Analysis()

                if attrs['config']['expt']['calibrate_single_shot']:
                    threshold = temp_data['thresholds']

                avg_idata, bs_gate_nums, rb_times = self.filter_data_for_si_wrt_spec_BS(temp_data, attrs, threshold)
                var_datas+= avg_idata
                bs_gate_numss += bs_gate_nums
                rb_timess += rb_times

                wrong_fids.append(np.average(avg_idata))
                wrong_bs_gate_nums.append(np.average(bs_gate_nums))
                wrong_times.append(np.average(rb_times))
                depth_list.append(attrs['config']['expt']['rb_depth'])

            except FileNotFoundError:
                print('FileNotFoundError')
                continue
        if wrong_way:
            return wrong_times, wrong_bs_gate_nums, wrong_fids, depth_list
        
        return self.reorganize_var_data_for_ramsey(var_datas, bs_gate_numss, rb_timess, attrs)
        
    def reorganize_var_data_for_ramsey(self, var_datas, bs_gate_numss, rb_timess, attrs, return_df=False, len_threshold = 0):
        # Re organize data so that we average over all the data points for a given BS gate number

        data = {'bs_gate_nums': bs_gate_numss, 'avg_idata': var_datas, 'rb_times': rb_timess}
        df = pd.DataFrame(data)
        if return_df: 
            return df
        bs_nums_range = np.arange(df['bs_gate_nums'].min(), df['bs_gate_nums'].max() + 1, 1)
        bs_nums_for_plot = [] # List to store the BS gate numbers that have a fidelity
        rb_times_for_plot = []
        fids_for_plot = []

        for idx, bs_gate_num in enumerate(bs_nums_range): 
            df_bs_num = df[df['bs_gate_nums'] == bs_gate_num]
            if len(df_bs_num) > len_threshold:
                # plt.plot(df_bs_num['rb_times'], df_bs_num['avg_idata'], '-o', label='BS gate ' + str(bs_gate_num))
                print('len of df_bs_num', len(df_bs_num))
                fids_for_plot.append(np.average(df_bs_num['avg_idata'].values))
                bs_nums_for_plot.append(bs_gate_num)
                rb_times_for_plot.append(np.average(df_bs_num['rb_times'].values))
            # print(f'Average idata for BS gate {bs_gate_num} is {avg_idata[idx]}')
        return bs_nums_for_plot, fids_for_plot, rb_times_for_plot, attrs['config']['expt']['wait_freq']
    def Ramsey_display(self, xdata, ydata, ramsey_freq=0.02, fit=True, fitparams = None, 
                       title='Ramsey'):

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
            # Remove the first and last point from fit in case weird edge measurements
            # fitparams = None
            # fitparams=[8, 0.5, 0, 20, None, None]
            p, pCov = fitter.fitdecaysin(xdata, ydata, fitparams=fitparams)
            
            if isinstance(p, (list, np.ndarray)): f_adjust_ramsey_avgi = sorted((ramsey_freq - p[1], ramsey_freq + p[1]), key=abs)

  

        title = title
        plt.figure(figsize=(10,5))
        axi = plt.subplot(111, 
            title=f"{title} (Ramsey Freq: {ramsey_freq} MHz)",
            ylabel="I [ADC level]")
        plt.plot(xdata, ydata,'o-')
        if fit:
            #p = data['fit_avgi']
            # print(p)
            # if isinstance(p, (list, np.ndarray)): 
                # pCov = data['fit_err_avgi']
            captionStr = f'$T_2$ Ramsey fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
            plt.plot(xdata, fitter.decaysin(xdata, *p), label=captionStr)
            plt.plot(xdata, fitter.expfunc(xdata, p[4], p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.plot(xdata, fitter.expfunc(xdata, p[4], -p[0], p[5], p[3]), color='0.2', linestyle='--')
            plt.legend()
            # print(f'Current pi pulse frequency: {f_pi_test}')
            print(f'Fit frequency from I [MHz]: {p[1]} +/- {np.sqrt(pCov[1][1])}')
            if p[1] > 2*ramsey_freq: print('WARNING: Fit frequency >2*wR, you may be too far from the real pi pulse frequency!')
            # print('Suggested new pi pulse frequency from fit I [MHz]:\n',
            #         f'\t{f_pi_test + f_adjust_ramsey_avgi[0]}\n',
            #         f'\t{f_pi_test + f_adjust_ramsey_avgi[1]}')
            print(f'T2 Ramsey from fit I [us]: {p[3]}')
            return p[3], np.sqrt(pCov[3][3])
    
    # -------------------------------------------------------------------------

    def plot_rb(self, fids_list , fids_post_list , xlist, 
                gg_list , gg_list_err , ge_list, ge_list_err , 
                eg_list, eg_list_err , ee_list , ee_list_err , ebars_list, ebars_post_list,reset_qubit_after_parity = False, parity_meas = True, 
                title='M1-S4 RB Post selection'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        # Exponential fit subplot
        ax1.errorbar(xlist, fids_list, yerr=ebars_list, fmt='o', label='raw', capsize=5, color=colors[0])
        ax1.errorbar(xlist, fids_post_list, yerr=ebars_post_list, fmt='o', label='post selection', capsize=5, color=colors[1])

        # Fitting
        xpts = xlist
        ypts = fids_list
        fit, err = fitter.fitexp(xpts, ypts, fitparams=None)

        ypts = fids_post_list
        fit_post, err_post = fitter.fitexp(xpts, ypts, fitparams=[None, None, None, None])

        p = fit
        pCov = err
        rel_err = 1 / p[3] / p[3] * np.sqrt(pCov[3][3])
        abs_err = rel_err * np.exp(-1 / fit[3])
        fid = np.exp(-1 / fit[3])
        fid_err = abs_err
        captionStr = f'$t$ fit [gates]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}\nFidelity per gate: {np.exp(-1 / fit[3])*100:.6f} $\pm$ {abs_err*100:.6f} %'

        p_post = fit_post
        pCov_post = err_post
        rel_err_post = 1 / p_post[3] / p_post[3] * np.sqrt(pCov_post[3][3])
        abs_err_post = rel_err_post * np.exp(-1 / fit_post[3])
        fid_post = np.exp(-1 / fit_post[3])
        fid_err_post = abs_err_post
        captionStr_post = f'$t$ fit [gates]: {p_post[3]:.3} $\pm$ {np.sqrt(pCov_post[3][3]):.3}\nFidelity per gate: {np.exp(-1 / fit_post[3])*100:.6f} $\pm$ {abs_err_post*100:.6f}%'

        ax1.plot(xpts, fitter.expfunc(xpts, *fit), label=captionStr, color=colors[0])
        ax1.plot(xpts, [fitter.expfunc(x, *fit_post) for x in xpts], label=captionStr_post, color = colors[1])
        ax1.set_xlabel('Time [us]')
        ax1.set_ylabel('Man1 |1> population')
        #ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title('Exponential Fit')
        ax1.set_xlabel('RB depth')
        ax1.set_ylabel('Man1 |1> population')

        # Shots subplot
        gg_label = '|11>'
        ge_label = '|10>'
        eg_label = '|00>'
        ee_label = '|01>'

        if reset_qubit_after_parity:
            gg_label  = '|11>'
            ge_label = '|10>'
            eg_label = '|01>'
            ee_label = '|00>'
        elif not parity_meas: 
            gg_label  = '|00>'
            ge_label = '|01>'
            eg_label = '|10>'
            ee_label = '|11>'

        ax2.errorbar(xlist, gg_list, yerr=gg_list_err, fmt='-o', label=gg_label, capsize=5)
        ax2.errorbar(xlist, ge_list, yerr=ge_list_err, fmt='-o', label=ge_label, capsize=5)
        ax2.errorbar(xlist, eg_list, yerr=eg_list_err, fmt='-o', label=eg_label, capsize=5)
        ax2.errorbar(xlist, ee_list, yerr=ee_list_err, fmt='-o', label=ee_label, capsize=5)

        # print number of 10 and 01 counts
        # print(str(gg_label) + ' counts:', gg_list)
        # print(str(ge_label) + ' counts:', ge_list)
        # print(str(eg_label) + ' counts:', eg_list)
        # print(str(ee_label) + ' counts:', ee_list)

        ax2.set_yscale('log')
        ax2.legend()
        ax2.set_title('Shots')
        ax2.set_xlabel('RB depth')
        ax2.set_ylabel('Population Ratio')

        # Main title
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        return fid, fid_err, fid_post, fid_err_post
    def show_rb(self, prev_data, expt_path, file_list, name = '_SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep.h5', title = 'RB', 
                dual_rail_spec = False, skip_spec_state_idx = None):
        '''show the rb result for a list of files
        
        Args: dual_rail_spec: if True, then we use that rb data extract function 
        skip_spec_state_idx: if dual_rail_spec is True, then we skip the state index in the list
        '''

        Pgg = 0.997573060976843
        Pge = 0.0024269390231570487
        Peg = 0.012984367839503025
        Pee = 0.9870156321604969

        P_matrix = np.matrix([[Pgg, Peg],[Pge, Pee]])
        conf_matrix = inv(P_matrix)

        tensor_product_matrix = np.kron(conf_matrix, conf_matrix)

        fids_list = []
        fids_post_list = []
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        gg_list_err = []
        ge_list_err = []
        eg_list_err = []
        ee_list_err = []
        xlist = []
        depth_list = []
        ebars_list = []
        ebars_post_list = []
        for file_no in file_list:
            full_name = str(file_no).zfill(5)+name
            temp_data, attrs = prev_data(expt_path, full_name)  
            if not dual_rail_spec:
                avg_readout, avg_readout_post, gg, ge, eg, ee = self.RB_extract_postselction_excited(temp_data,attrs, active_reset=True, conf_matrix=tensor_product_matrix)#, start_idx = start_idx)
            else: 
                avg_readout, avg_readout_post, gg, ge, eg, ee = self.RB_extract_postselction_excited_dual_rail_spec(temp_data,attrs, active_reset=True, conf_matrix=tensor_product_matrix, skip_spec_states_idx = skip_spec_state_idx)
            # print(gg[0]+ge[0]+eg[0]+ee[0])
            # print number of gg shots
            
            gg_list.append(np.average(gg))
            ge_list.append(np.average(ge))
            eg_list.append(np.average(eg))
            ee_list.append(np.average(ee))
            fids_list.append(np.average(avg_readout))
            ebars_list.append(np.std(avg_readout)/np.sqrt(len(avg_readout)))
            gg_list_err.append(np.std(ge)/np.sqrt(len(ge)))
            ge_list_err.append(np.std(ge)/np.sqrt(len(ge)))
            eg_list_err.append(np.std(eg)/np.sqrt(len(eg)))
            ee_list_err.append(np.std(ee)/np.sqrt(len(ee)))


            fids_post_list.append(np.average(avg_readout_post))
            ebars_post_list.append(np.std(avg_readout_post)/np.sqrt(len(avg_readout_post)))
            xlist.append(attrs['config']['expt']['rb_depth']*attrs['config']['expt']['bs_repeat'])
            depth_list.append(attrs['config']['expt']['rb_depth']*attrs['config']['expt']['bs_repeat'])

        try: 
            reset_bool = (attrs['config']['expt']['reset_qubit_after_parity'] or attrs['config']['expt']['reset_qubit_via_active_reset_after_first_meas'])
        except KeyError:
            reset_bool = attrs['config']['expt']['reset_qubit_after_parity']
        fid, fid_err, fid_post, fid_post_err = self.plot_rb(fids_list = fids_list, fids_post_list = fids_post_list, xlist=depth_list, 
                    gg_list = gg_list, gg_list_err = gg_list_err, ge_list = ge_list, ge_list_err = ge_list_err, 
                    eg_list = eg_list, eg_list_err = eg_list_err, ee_list = ee_list, ee_list_err = ee_list_err,
                    ebars_list=ebars_list, ebars_post_list=ebars_post_list, reset_qubit_after_parity = reset_bool,
                    parity_meas=attrs['config']['expt']['parity_meas'],
                    title=title)
        return fids_list, fids_post_list, gg_list, ge_list, eg_list, ee_list, gg_list_err, ge_list_err, eg_list_err, ee_list_err, xlist, depth_list, ebars_list, ebars_post_list, fid, fid_err, fid_post, fid_post_err
    def RB_extract_excited(self, temp_data):
        avg_readout = []
        for i in range(len(temp_data['Idata'])):
            counting = 0
            for j in temp_data['Idata'][i]:
                if j>temp_data['thresholds']:
                    counting += 1
            avg_readout.append(counting/len(temp_data['Idata'][i]))
        return avg_readout

    def RB_extract_ground(self, temp_data):
        avg_readout = []
        for i in range(len(temp_data['Idata'])):
            counting = 0
            for j in temp_data['Idata'][i]:
                if j<temp_data['thresholds']:
                    counting += 1
            avg_readout.append(counting/len(temp_data['Idata'][i]))
        return avg_readout


    
    def RB_extract_postselction_excited_old(self, temp_data):
        # remember the parity mapping rule:
        # 00 -> eg, 01 -> ee, 10 -> ge, 11 -> gg
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []
        for aa in range(len(temp_data['Idata'])):
            gg = 0
            ge = 0
            eg = 0
            ee = 0
            for j in range(len(temp_data['Idata'][aa][0])):
                #  check if the counts are the same as initial counts
                if temp_data['Idata'][aa][0][j]>temp_data['thresholds'][0]: # classified as e
                    if temp_data['Idata'][aa][1][j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if temp_data['Idata'][aa][1][j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))
            fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
            fid_post_list.append(ge/(ge+ee))
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list
    def filter_data_BS(self, a1, a2, a3, threshold, post_selection = False):
        # assume the last one  is experiment data, the last but one is for post selection
        '''
        This is for active reset post selection 

        the post selection parameter DOES not refer to active reset post selection
        a1: from active reset pre selection 
        a2: from actual experiment
        a3: from actual experiment post selection
        '''
        result_1 = []
        result_2 = []
        
        
        for k in range(len(a1)):

            if a1[k] < threshold:

                result_1.append(a2[k])
                if post_selection:
                    result_2.append(a3[k])
        
        return np.array(result_1), np.array(result_2)
    def RB_extract_postselction_excited(self, temp_data, attrs, active_reset = False, conf_matrix = None):
        # remember the parity mapping rule:
        # 00 -> eg, 01 -> ee, 10 -> ge, 11 -> gg
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []


        for aa in range(len(temp_data['Idata'])):
            gg = 0
            ge = 0
            eg = 0
            ee = 0

            #  post selection due to active reset
            if active_reset:
                data_init, data_post_select = self.filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], temp_data['Idata'][aa][4], temp_data['thresholds'],post_selection = True)
            else: 
                data_init = temp_data['Idata'][aa][0]
                data_post_select = temp_data['Idata'][aa][1]
            
            # print('len data_init', len(data_init))
            # print('len data_post_select', len(data_post_select))
            
            # beamsplitter post selection 
            for j in range(len(data_init)):
                #  check if the counts are the same as initial counts
                if data_init[j]>temp_data['thresholds'][0]: # classified as e
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            # print('gg', gg)
            # print('ge', ge)
            # print('eg', eg)
            # print('ee', ee)
            # print('total', eg + ge + gg + ee)
            if conf_matrix is not None: ## correct counts from histogram
                gg = gg * conf_matrix[0,0] + ge * conf_matrix[0,1] + eg * conf_matrix[0,2] + ee * conf_matrix[0,3]
                ge = gg * conf_matrix[1,0] + ge * conf_matrix[1,1] + eg * conf_matrix[1,2] + ee * conf_matrix[1,3]
                eg = gg * conf_matrix[2,0] + ge * conf_matrix[2,1] + eg * conf_matrix[2,2] + ee * conf_matrix[2,3]
                ee = gg * conf_matrix[3,0] + ge * conf_matrix[3,1] + eg * conf_matrix[3,2] + ee * conf_matrix[3,3]
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))

            # print('gg_list', gg_list)
            # print('ge_list', ge_list)
            # print('eg_list', eg_list)
            # print('ee_list', ee_list)

            try:
                if attrs['config']['expt']['reset_qubit_after_parity']:
                    # print('reset_qubit_after_parity')
                    # print('using new method to calculate post selection fidelity ')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                elif not attrs['config']['expt']['parity_meas']: 
                    # print('not parity_meas')
                    fid_raw_list.append((ee+eg)/(eg+ge+gg+ee))
                    fid_post_list.append(eg/(ge+eg))
                elif attrs['config']['expt']['reset_qubit_via_active_reset_after_first_meas']:
                    # print('reset_qubit_via_active_reset_after_first_meas')
                    
                                            # gg_label = '|11>'
                                                # ge_label = '|10>'
                                                # eg_label = '|01>'
                                                # ee_label = '|00>'
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                else:
                    # print('using old method to calculate post selection fidelity ')
                    # print('old method')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+ee))
            except KeyError:
                print('using old method to calculate post selection fidelity ')
                fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                fid_post_list.append(ge/(ge+ee))
        print(eg + ge + gg + ee)
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list

    def RB_extract_postselction_excited_dual_rail_spec(self, temp_data, attrs, active_reset = False, conf_matrix = None,
                                        skip_spec_states_idx = None):
        '''
        This is specially for dual rail spectator analysis where we skip over 0 and 1 population of the spectator mode
        '''
        # remember the parity mapping rule:
        # 00 -> eg, 01 -> ee, 10 -> ge, 11 -> gg
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []

        
        for aa in range(len(temp_data['Idata'])):
            if aa%6 in skip_spec_states_idx:
                continue

            gg = 0
            ge = 0
            eg = 0
            ee = 0

            #  post selection due to active reset
            if active_reset:
                data_init, data_post_select = filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], temp_data['Idata'][aa][4], temp_data['thresholds'],post_selection = True)
            else: 
                data_init = temp_data['Idata'][aa][0]
                data_post_select = temp_data['Idata'][aa][1]
            
            # print('len data_init', len(data_init))
            # print('len data_post_select', len(data_post_select))
            
            # beamsplitter post selection 
            for j in range(len(data_init)):
                #  check if the counts are the same as initial counts
                if data_init[j]>temp_data['thresholds'][0]: # classified as e
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            # print('gg', gg)
            # print('ge', ge)
            # print('eg', eg)
            # print('ee', ee)
            # print('total', eg + ge + gg + ee)
            if conf_matrix is not None: ## correct counts from histogram
                gg = gg * conf_matrix[0,0] + ge * conf_matrix[0,1] + eg * conf_matrix[0,2] + ee * conf_matrix[0,3]
                ge = gg * conf_matrix[1,0] + ge * conf_matrix[1,1] + eg * conf_matrix[1,2] + ee * conf_matrix[1,3]
                eg = gg * conf_matrix[2,0] + ge * conf_matrix[2,1] + eg * conf_matrix[2,2] + ee * conf_matrix[2,3]
                ee = gg * conf_matrix[3,0] + ge * conf_matrix[3,1] + eg * conf_matrix[3,2] + ee * conf_matrix[3,3]
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))

            # print('gg_list', gg_list)
            # print('ge_list', ge_list)
            # print('eg_list', eg_list)
            # print('ee_list', ee_list)

            try:
                if attrs['config']['expt']['reset_qubit_after_parity']:
                    # print('reset_qubit_after_parity')
                    # print('using new method to calculate post selection fidelity ')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                elif not attrs['config']['expt']['parity_meas']: 
                    # print('not parity_meas')
                    fid_raw_list.append((ee+eg)/(eg+ge+gg+ee))
                    fid_post_list.append(eg/(ge+eg))
                elif attrs['config']['expt']['reset_qubit_via_active_reset_after_first_meas']:
                    # print('reset_qubit_via_active_reset_after_first_meas')
                    
                                            # gg_label = '|11>'
                                                # ge_label = '|10>'
                                                # eg_label = '|01>'
                                                # ee_label = '|00>'
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+eg))
                else:
                    # print('using old method to calculate post selection fidelity ')
                    # print('old method')
                    fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                    fid_post_list.append(ge/(ge+ee))
            except KeyError:
                print('using old method to calculate post selection fidelity ')
                fid_raw_list.append((ge+gg)/(eg+ge+gg+ee))
                fid_post_list.append(ge/(ge+ee))
        print(eg + ge + gg + ee)
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list

    def RB_extract_postselction_parity_fixed_excited(self, temp_data, active_reset = False):
        # remember the parity mapping rule:
        # 00 -> ee, 01 -> eg, 10 -> ge, 11 -> gg
        gg_list = []
        ge_list = []
        eg_list = []
        ee_list = []
        fid_raw_list = []
        fid_post_list = []


        for aa in range(len(temp_data['Idata'])):
            gg = 0
            ge = 0
            eg = 0
            ee = 0

            #  post selection due to active reset
            if active_reset:
                data_init, data_post_select = filter_data_BS(temp_data['Idata'][aa][2], temp_data['Idata'][aa][3], temp_data['Idata'][aa][4], temp_data['thresholds'],post_selection = True)
            else: 
                data_init = temp_data['Idata'][aa][0]
                data_post_select = temp_data['Idata'][aa][1]

            # print(data_init)
            # print(data_post_select)
            
            # beamsplitter post selection 
            for j in range(len(data_init)):
                #  check if the counts are the same as initial counts
                if data_init[j]>temp_data['thresholds'][0]: # classified as e
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ee += 1
                    else:
                        eg +=1
                else:  # classified as g
                    if data_post_select[j]>temp_data['thresholds'][0]:  # second e
                        ge +=1
                    else:
                        gg += 1
            gg_list.append(gg/(eg+ge+gg+ee))
            ge_list.append(ge/(eg+ge+gg+ee))
            eg_list.append(eg/(eg+ge+gg+ee))
            ee_list.append(ee/(eg+ge+gg+ee))
            
            fid_raw_list.append((gg+ge)/(eg+ge+gg+ee))
            fid_post_list.append(ge/(ge+eg))
        print(eg + ge + gg + ee)
        return fid_raw_list, fid_post_list, gg_list, ge_list, eg_list, ee_list


import random
import numpy as np

# Calculate inverse rotation
matrix_ref = {}
# Z, X, Y, -Z, -X, -Y

# Basis states : |0>, |+>, |+i>, |1>, |->, |-i>
matrix_ref['0'] = np.matrix([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
matrix_ref['1'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 0, 0]])
matrix_ref['2'] = np.matrix([[0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1]])
matrix_ref['3'] = np.matrix([[0, 0, 1, 0, 0, 0],   # rotations are counterclockwise about axes
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0]])
matrix_ref['4'] = np.matrix([[0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1]])
matrix_ref['5'] = np.matrix([[0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0, 0]])
matrix_ref['6'] = np.matrix([[0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 1]])

# create pauli matrices 
# Create rotation gate 
# def Rx(theta):
#     return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
# def Ry(theta):
#     return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])

# X = Rx(np.pi) * 1j
# Y = Ry(np.pi) * 1j
# X_hpi = Rx(np.pi/2) * np.sqrt(1j)
# Y_hpi = Ry(np.pi/2) * np.sqrt(1j)


def no2gate(no):
    g = 'I'
    if no==1:
        g = 'X'
    elif no==2:
        g = 'Y'
    elif no==3:
        g = 'X/2'
    elif no==4:
        g = 'Y/2'
    elif no==5:
        g = '-X/2'
    elif no==6:
        g = '-Y/2'  

    return g

def gate2no(g):
    no = 0
    if g=='X':
        no = 1
    elif g=='Y':
        no = 2
    elif g=='X/2':
        no = 3
    elif g=='Y/2':
        no = 4
    elif g=='-X/2':
        no = 5
    elif g=='-Y/2':
        no = 6

    return no

def generate_sequence(rb_depth, iRB_gate_no=-1, debug=False, matrix_ref=matrix_ref):
    gate_list = []
    for ii in range(rb_depth):
        print(gate_list)
        gate_list.append(random.randint(1, 2))
        if iRB_gate_no > -1:   # performing iRB
            gate_list.append(iRB_gate_no)
    if debug:
        print(gate_list)

    a0 = np.matrix([[1], [0], [0], [0], [0], [0]])
    final_matrix = matrix_ref['0']
    anow = a0
    for i in gate_list:
        anow = np.dot(matrix_ref[str(i)], anow)
        final_matrix = np.dot(matrix_ref[str(i)], final_matrix)
    anow1 = np.matrix.tolist(anow.T)[0] # creates a bra vector out of ket 
    max_index = anow1.index(max(anow1)) # finds which elecment is 1; others are 0 
    # print(anow1)
    # return anow1
    # inverse of the rotation
    inverse_gate_symbol = ['-Y/2', 'X/2', 'X', 'Y/2', '-X/2']
    if max_index == 0:
        pass
    else:
        gate_list.append(gate2no(inverse_gate_symbol[max_index-1]))
        final_matrix = np.dot(matrix_ref[str(gate2no(inverse_gate_symbol[max_index-1]))], final_matrix)
    if debug:
        print(gate_list)
        print(max_index)
    return gate_list, final_matrix
## Automatic calibration code
def find_on_resonant_frequency(y_list, time_points, frequency_points, fitparams=None):
    """
    Finds the on-resonant frequency and its oscillation rate from a Chevron plot using curve fitting.
    
    Parameters:
    y_list (np.ndarray): 2D array where rows correspond to different frequencies and columns correspond to time points.
    time_points (np.ndarray): 1D array of time points corresponding to the columns of y_list.
    frequency_points (np.ndarray): 1D array of frequency points corresponding to the rows of y_list.
    
    Returns:
    tuple: On-resonant frequency and its oscillation rate.
    """
    max_amplitude = 0
    min_rate = 99999
    on_resonant_frequency = None
    on_resonant_rate = None
    
    for i, row in enumerate(y_list):
        # Initial guess for the parameters: amplitude, decay rate, frequency, phase, offset
        # initial_guess = [np.max(row) - np.min(row), 5, 1, 0, np.mean(row)]
        
        try:
            # Perform the curve fitting
            popt, _ = fitter.fitdecaysin(
                    time_points, row, fitparams=fitparams)
            
            
            # Calculate the oscillation rate (as frequency)
            oscillation_rate = popt[1]
            
            # # Check if this frequency has the maximum amplitude
            # if np.abs(popt[0]) > max_amplitude:
            #     max_amplitude = np.abs(popt[0])
            #     on_resonant_frequency = frequency_points[i]
            #     on_resonant_rate = oscillation_rate
            
            # Check if this frequency has the slowst rate (looks more robust)
            if np.abs(oscillation_rate) < min_rate:
                min_rate = np.abs(oscillation_rate)
                on_resonant_frequency = frequency_points[i]
                on_resonant_rate = oscillation_rate
                
        except RuntimeError:
            # If the fit fails, we skip this frequency
            continue
    
    return on_resonant_frequency, on_resonant_rate

## ---------------------------- Calibration for MultiRBAM ---------------------------- ##

def filter_data_single_shot(prev_data, file_list,  name = '_single_shot_phase_sweep.h5', threshold = -20, active_reset = True, expt_path = ''):
    '''
    Returns active reset filtered data 
    '''
    y_list = []
    for file_no in file_list:
        count = 0   # how many points possibly |f>
        full_name = str(file_no).zfill(5)+name
        temp_data, attrs = prev_data(expt_path, full_name)  # ef

        # active reset:
        if active_reset:
            Ig, Qg = filter_data_IQ(temp_data['I'], temp_data['Q'],threshold, readout_per_experiment=4)
            total_counts = len(Ig)
            # Ig = I_selected
        else:

            Ig = temp_data['I']
            Qg = temp_data['Q']
            total_counts = len(Ig)
        for i in range(len(Ig)):
            if Ig[i] < threshold :
                count += 1

        y_list.append(count/total_counts)
    return y_list

def cross_storage_phase_analysis(attrs, gate_length, from_fitting, target_mode_no, add = True, cfg = None):
    self_overhead = attrs['config']['device']['storage']['idling_phase'][target_mode_no - 1][target_mode_no - 1]
    idling_frequency = attrs['config']['device']['storage']['idling_freq'][target_mode_no - 1]
    if cfg is not None:
        self_overhead = cfg['device']['storage']['idling_phase'][target_mode_no - 1][target_mode_no - 1]
        idling_frequency = cfg['device']['storage']['idling_freq'][target_mode_no - 1]
        print('using values from cfg')
        print(self_overhead, idling_frequency)
    else: 
        print('using values from attrs')
        print(self_overhead, idling_frequency)
    if add: 
        cross_phase_overhead = (from_fitting-gate_length*idling_frequency*180*2 -self_overhead*2 ) % 360
    else: 
        cross_phase_overhead = (from_fitting-gate_length*idling_frequency*180*2 -self_overhead*2 ) % -360
    return cross_phase_overhead

import itertools

def generate_mode_combinations(mode_list, num_modes_sim_rb, skip_combos):
    """
    Generate all possible lists of modes where each list has length num_modes_sim_rb.

    Parameters:
    mode_list (list): List of available modes.
    num_modes_sim_rb (int): Length of each combination list.

    Returns:
    list: List of all possible combinations.
    """
    # Generate all possible combinations
    combinations = np.array(list(itertools.combinations(mode_list, num_modes_sim_rb))).tolist()

    for combo in skip_combos:
        if combo in combinations:
            combinations.remove(combo)
    
    return combinations

# for computing gate_length 
from multimode_expts.MM_rb_base import *

def get_spec_idling_time(spec_reps, spectator_mode_no, cfg): 
    '''computes total idling time for the spectator mode pulse sequence'''
    mm_base = MM_rb_base(cfg = cfg)

    # spector pulse 
    qubit_spec_init = [['qubit', 'ge', 'hpi', 0]]
    spectator_pulse_str = mm_base.compound_storage_gate(input = True, storage_no = spectator_mode_no)
    for _ in range(spec_reps-1): 
        spectator_pulse_str += mm_base.compound_storage_gate(input = False, storage_no = spectator_mode_no) + mm_base.compound_storage_gate(input = True, storage_no = spectator_mode_no)

    qubit_spec_init_idling_time = mm_base.get_total_time(qubit_spec_init, gate_based = True)
    spec_idling_time = mm_base.get_total_time(spectator_pulse_str, gate_based = True)
    return spec_idling_time +  qubit_spec_init_idling_time

# ---------------------------- RB of  MultiRBAM sweep depth analysis  ---------------------------- #
def discriminate_for_g(raw_data, threshold): 
    '''
    Return # of g counts
    '''
    counting = 0 
    for j in raw_data:
        if j<threshold:
            counting += 1
    return counting

def RBAM_extract(temp_data, mode_idxs = [1], active_reset = True, post_select = False): 
    '''
    returns rb fidelity , assumes active reset is on 
    '''
    mean_list = []
    err_list = []

    for mode_idx in mode_idxs: 
        # mode_idx = 0
        avg_readout = []
        for i in range(len(temp_data['Idata'][mode_idx])):
            #counting = 0
            
            if active_reset:
                if post_select:  # POST SELECT FOR EXPERIMENT, NOT ACTIVE RESEET
                    raw_data, post_select_data= filter_data_BS(temp_data['Idata'][mode_idx][i][2], temp_data['Idata'][mode_idx][i][3], temp_data['Idata'][mode_idx][i][4], temp_data['thresholds'], post_selection = True)
                else: 
                    raw_data, _= filter_data_BS(temp_data['Idata'][mode_idx][i][2], temp_data['Idata'][mode_idx][i][3], None, temp_data['thresholds'])
            else: 
                raw_data = temp_data['Idata'][mode_idx][i][0]
            # print(len(raw_data))
            
            
            

            if post_select:
                g_counts = discriminate_for_g(raw_data, temp_data['thresholds'])
                e_counts = discriminate_for_g(post_select_data, temp_data['thresholds'])
                # print(post_select_data)
                # print(g_counts, e_counts)
                prob = g_counts/(g_counts+e_counts)
                avg_readout.append(prob)

            else: 
                counting = discriminate_for_g(raw_data, temp_data['thresholds'])
                avg_readout.append(counting/len(raw_data))


        #
        # avg_readout = RB_extract(temp_data)
        mean = np.average(avg_readout)
        err = np.std(avg_readout)/np.sqrt(len(avg_readout))
        # print('Result for Mode ', mode_idx + 1, 'is', mean, err)
        mean_list.append(mean)
        err_list.append(err)
    return mean_list, err_list


def compute_fidelity_list(prev_data, file_list, name, mode_length, expt_path):
    """
    Computes the fidelity list and error bars for each mode from the given file list.

    Parameters:
    file_list (list): List of file numbers to process.
    name (str): Base name of the files.
    mode_length (int): Number of modes to process.
    expt_path (str): Path to the experiment data.

    Returns:
    tuple: A tuple containing:
        - xlist (list): List of depth values.
        - fids_list (list of lists): List of fidelity values for each mode.
        - ebars_list (list of lists): List of error bars for each mode.
    """
    fids_list = [[] for _ in range(mode_length)]
    ebars_list = [[] for _ in range(mode_length)]
    xlist = []
    mode_idxs = [i for i in range(mode_length)]

    for file_no in file_list:
        full_name = str(file_no).zfill(5) + name
        temp_data, attrs = prev_data(expt_path, full_name)
        mean, err = RBAM_extract(temp_data, mode_idxs=mode_idxs, post_select=False)
        for i in range(mode_length):
            fids_list[i].append(mean[i])
            ebars_list[i].append(err[i])
        xlist.append(attrs['config']['expt']['depth_list'][0])

    return xlist, fids_list, ebars_list

def compute_fidelity_list_RB(prev_data, file_list, name, mode_length, expt_path):
    """
    Computes the reference fidelity list and error bars from the given file list.

    Parameters:
    file_list (list): List of file numbers to process.
    name (str): Base name of the reference files.
    mode_length (int): Number of modes to process.
    expt_path (str): Path to the experiment data.

    Returns:
    tuple: A tuple containing:
        - xlist_reference (list): List of depth values.
        - fids_list_reference (list): List of average fidelity values.
        - ebars_list_reference (list): List of error bars.
    """
    fids_list_reference = []
    xlist_reference = []
    ebars_list_reference = []
    for file_no in file_list:
        full_name = str(file_no).zfill(5) + name
        temp_data, attrs = prev_data(expt_path, full_name)  
        avg_readout_reference = RB_extract(temp_data, conf=False)
        fids_list_reference.append(np.average(avg_readout_reference))
        ebars_list_reference.append(np.std(avg_readout_reference) / np.sqrt(len(avg_readout_reference)))
        xlist_reference.append(attrs['config']['expt']['rb_depth'])
    return xlist_reference, fids_list_reference, ebars_list_reference

def fit_fidelity(xlist, fids_list):
    """
    Fits the fidelity data using an exponential model and computes the fidelity.

    Parameters:
    xlist (list): List of depth values.
    fids_list (list of lists): List of fidelity values for each mode.

    Returns:
    tuple: A tuple containing:
        - fidelity_list (list): List of computed fidelities for each mode.
        - fit_params_list (list): List of fit parameters for each mode.
    """
    p_survival_list = []
    fit_params_list = []
    for i in range(len(fids_list)):
        ypts = fids_list[i]
        fit, err = fitter.fitexp(xlist, ypts, fitparams=None)

        p = fit
        pCov = err
        # alpha = np.exp(-1 / fit[3])
        # r = 1 - alpha - (1 - alpha) / 3
        # fidelity = 1 - r

        p_survival_list.append(np.exp(-1/fit[3]))
        fit_params_list.append(fit)

    return p_survival_list, fit_params_list

def fit_fidelity_reference(xlist_ref, fids_list_ref):
    """
    Computes decay parameter (and hence survival probability) for the reference data.

    Parameters:
    xlist_ref (list): List of depth values for the reference data.
    fids_list_ref (list): List of fidelity values for the reference data.

    Returns:
    list: List of fit parameters for the reference data.
    """
    fit_params_ref, err_ref = fitter.fitexp(xlist_ref, fids_list_ref, fitparams=None)
    return fit_params_ref


import numpy as np
import matplotlib.pyplot as plt

# def plot_fidelity(xlist, fids_list, ebars_list, fidelity_list, fit_params_list, xlist_ref, fids_list_ref, ebars_list_ref, fit_params_ref, captionStr_ref, mode_list, close_plt=False, scale_factor=1, scale_factor_ref=1.5):
#     """
#     Plots the fidelity data and the fitted results, including reference data.

#     Parameters:
#     xlist (list): List of depth values.
#     fids_list (list of lists): List of fidelity values for each mode.
#     ebars_list (list of lists): List of error bars for each mode.
#     fidelity_list (list): List of computed fidelities for each mode.
#     fit_params_list (list): List of fit parameters for each mode.
#     xlist_ref (list): List of depth values for the reference data.
#     fids_list_ref (list): List of fidelity values for the reference data.
#     ebars_list_ref (list): List of error bars for the reference data.
#     fit_params_ref (list): List of fit parameters for the reference data.
#     captionStr_ref (str): Caption for the reference data.
#     mode_list (list): List of modes.
#     close_plt (bool): Whether to close the plot after showing.
#     scale_factor (float): Scale factor for xlist.
#     scale_factor_ref (float): Scale factor for xlist_ref.
#     """
#     fig, axs = plt.subplots(1, 2, figsize=(20, 5))

#     # Unscaled plot
#     axs[0].set_ylabel("Fidelity")
#     color_list = ['r', 'g']

#     for i in range(len(fids_list)):
#         axs[0].errorbar(xlist, fids_list[i], yerr=ebars_list[i], fmt='o', capsize=5, label=f'Mode {i+1} Data', color=color_list[i])
#         fit_x = np.linspace(min(xlist), max(xlist), 100)
#         fit_y = fitter.expfunc(fit_x, *fit_params_list[i])
#         axs[0].plot(fit_x, fit_y, color=color_list[i], label=f'Mode {i+1} Fit: Depth={fit_params_list[i][3]:.2f}, Fidelity={fidelity_list[i]:.4f}')

#     axs[0].errorbar(xlist_ref, fids_list_ref, yerr=ebars_list_ref, fmt='o', capsize=5, label=f'{captionStr_ref} Data', color='b')
#     fit_x_ref = np.linspace(min(xlist_ref), max(xlist_ref), 100)
#     fit_y_ref = fitter.expfunc(fit_x_ref, *fit_params_ref)
#     fidelity_ref = 1 - ((1 - np.exp(-1 / fit_params_ref[3])) - (1 - np.exp(-1 / fit_params_ref[3])) / 3)
#     axs[0].plot(fit_x_ref, fit_y_ref, color='b', label=f'{captionStr_ref} Fit: Depth={fit_params_ref[3]:.2f}, Fidelity={fidelity_ref:.4f}')

#     fidelity_list_wrt_ref = [np.sqrt(fidelity / fidelity_ref) for fidelity in fidelity_list]
#     axs[0].set_title('Sim RB on storage modes ' + str(mode_list) + ' with fidelities wrt ref ' + str(np.round(fidelity_list_wrt_ref, 4)))
#     axs[0].set_xlabel('Gates')
#     axs[0].legend()

#     # Scaled plot
#     scaled_xlist = [x * scale_factor for x in xlist]
#     scaled_xlist_ref = [x * scale_factor_ref for x in xlist_ref]

#     for i in range(len(fids_list)):
#         axs[1].errorbar(scaled_xlist, fids_list[i], yerr=ebars_list[i], fmt='o', capsize=5, label=f'Scaled Mode {i+1} Data', color=color_list[i])
#         fit_x = np.linspace(min(scaled_xlist), max(scaled_xlist), 100)
#         fit_y = fitter.expfunc(fit_x / scale_factor, *fit_params_list[i])
#         axs[1].plot(fit_x, fit_y, color=color_list[i], label=f'Scaled Mode {i+1} Fit: Depth={fit_params_list[i][3]:.2f}, Fidelity={fidelity_list[i]:.4f}')

#     axs[1].errorbar(scaled_xlist_ref, fids_list_ref, yerr=ebars_list_ref, fmt='o', capsize=5, label=f'Scaled {captionStr_ref} Data', color='b')
#     fit_x_ref = np.linspace(min(scaled_xlist_ref), max(scaled_xlist_ref), 100)
#     fit_y_ref = fitter.expfunc(fit_x_ref / scale_factor_ref, *fit_params_ref)
#     axs[1].plot(fit_x_ref, fit_y_ref, color='b', label=f'Scaled {captionStr_ref} Fit: Depth={fit_params_ref[3]:.2f}, Fidelity={fidelity_ref:.4f}')

#     axs[1].set_title('Scaled Sim RB on storage modes ' + str(mode_list))
#     axs[1].set_xlabel('Time')
#     axs[1].legend()

#     if close_plt: 
#         plt.close()
#     else: 
#         plt.show()

#     return fidelity_list_wrt_ref

def find_gate_fidelity(p_survival, dim, interleaved = False, p_survival_interleaved_upon = 1):
    '''
    Computes gate fidelity according to https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.109.080505 

    If interleaved, p_survival_interleaved_upon is the survival probability of the sequence without the interleaved gates
    '''
    p = p_survival
    if interleaved: 
        p = p_survival/p_survival_interleaved_upon
    r = (dim-1)/dim * (1-p)
    return 1 - r 

def plot_fidelity(xlist, fids_list, ebars_list, p_survival_list, fit_params_list, xlist_ref, fids_list_ref, ebars_list_ref, fit_params_ref, captionStr_ref, mode_list, close_plt=False, scale_factor=1, scale_factor_ref=1.5):
    """
    Plots the fidelity data and the fitted results, including reference data.

    Parameters:
    xlist (list): List of depth values.
    fids_list (list of lists): List of fidelity values for each mode.
    ebars_list (list of lists): List of error bars for each mode.
    p_survival_list (list): List of computed survival probabilities for each mode.
    fit_params_list (list): List of fit parameters for each mode.
    xlist_ref (list): List of depth values for the reference data.
    fids_list_ref (list): List of fidelity values for the reference data.
    ebars_list_ref (list): List of error bars for the reference data.
    fit_params_ref (list): List of fit parameters for the reference data.
    captionStr_ref (str): Caption for the reference data.
    mode_list (list): List of modes.
    close_plt (bool): Whether to close the plot after showing.
    scale_factor (float): Scale factor for xlist.
    scale_factor_ref (float): Scale factor for xlist_ref.
    """
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))

    # Unscaled plot
    axs[0].set_ylabel("Fidelity")
    color_list =  ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


    for i in range(len(fids_list)):
        axs[0].errorbar(xlist, fids_list[i], yerr=ebars_list[i], fmt='o', capsize=5, label=f'Mode {i+1} Data', color=color_list[i])
        fit_x = np.linspace(min(xlist), max(xlist), 100)
        fit_y = fitter.expfunc(fit_x, *fit_params_list[i])
        axs[0].plot(fit_x, fit_y, color=color_list[i], label=f'Mode {i+1} Fit: Depth={fit_params_list[i][3]:.2f}, p_survival={p_survival_list[i]:.4f}')

    axs[0].errorbar(xlist_ref, fids_list_ref, yerr=ebars_list_ref, fmt='o', capsize=5, label=f'{captionStr_ref} Data', color='b')
    fit_x_ref = np.linspace(min(xlist_ref), max(xlist_ref), 100)
    fit_y_ref = fitter.expfunc(fit_x_ref, *fit_params_ref)
    p_survival_ref = np.exp(-1/fit_params_ref[3])
    fidelity_ref = find_gate_fidelity(p_survival_ref, 3, False)
    axs[0].plot(fit_x_ref, fit_y_ref, color='b', label=f'{captionStr_ref} Fit: Depth={fit_params_ref[3]:.2f}, Fidelity={fidelity_ref:.4f}')

    fidelity_list_wrt_ref = [np.sqrt(find_gate_fidelity(p_survival_list[i], 3, True, p_survival_ref)) for i in range(len(p_survival_list))] # sqrt because each interleaved gate includes 2 swaps
                             #[np.sqrt(fidelity / fidelity_ref) for fidelity in fidelity_list]
    axs[0].set_title('Sim RB on storage modes ' + str(mode_list) + ' with fidelities wrt ref ' + str(np.round(fidelity_list_wrt_ref, 4)))
    axs[0].set_xlabel('Gates')
    axs[0].legend()

    # Scaled plot
    scaled_xlist = [x * scale_factor for x in xlist]
    scaled_xlist_ref = [x * scale_factor_ref for x in xlist_ref]

    for i in range(len(fids_list)):
        axs[1].errorbar(scaled_xlist, fids_list[i], yerr=ebars_list[i], fmt='o', capsize=5, label=f'Scaled Mode {i+1} Data', color=color_list[i])
        fit_x = np.linspace(min(scaled_xlist), max(scaled_xlist), 100)
        fit_y = fitter.expfunc(fit_x / scale_factor, *fit_params_list[i])
        axs[1].plot(fit_x, fit_y, color=color_list[i], label=f'Scaled Mode {i+1} Fit: Depth={fit_params_list[i][3]:.2f}, p_survival={p_survival_list[i]:.4f}')

    axs[1].errorbar(scaled_xlist_ref, fids_list_ref, yerr=ebars_list_ref, fmt='o', capsize=5, label=f'Scaled {captionStr_ref} Data', color='b')
    fit_x_ref = np.linspace(min(scaled_xlist_ref), max(scaled_xlist_ref), 100)
    fit_y_ref = fitter.expfunc(fit_x_ref / scale_factor_ref, *fit_params_ref)
    axs[1].plot(fit_x_ref, fit_y_ref, color='b', label=f'Scaled {captionStr_ref} Fit: Depth={fit_params_ref[3]:.2f}, Fidelity={fidelity_ref:.4f}')

    # axs[1].set_title('Scaled Sim RB on storage modes ' + str(mode_list))
    axs[1].set_xlabel('Time')
    axs[1].legend()
    plt.tight_layout()

    if close_plt: 
        plt.close()
    else: 
        plt.show()

    return fidelity_list_wrt_ref
def get_gate_time_RBAM(target_mode_no, spec_mode_nos, cfg ): 
    '''computes total idling time for the spectator mode pulse sequence'''
    mm_base = MM_rb_base(cfg = cfg)

    full_pulse_str = []
    time = 0 

    for mode_no in [target_mode_no] + spec_mode_nos:
        stor_output = mm_base.compound_storage_gate(input = False, storage_no = mode_no)
        time += mm_base.get_total_time(stor_output, gate_based = True)
        pulse_ge_str = [['qubit', 'ge', 'pi', 0]] 
        time+= mm_base.get_total_time(pulse_ge_str, gate_based = True) * 8/6
        stor_input = mm_base.compound_storage_gate(input = True, storage_no = mode_no)
        time += mm_base.get_total_time(stor_input, gate_based = True)

    return time

def get_f0g1_time(cfg): 
    '''
    computes time for doing gate in ge space and keep in man1
    '''
    time = 0 
    mm_base = MM_rb_base(cfg = cfg)

    mode_no = 1 # mode_no doesnt matter, we ignore storage pulse
    stor_output = mm_base.compound_storage_gate(input = False, storage_no = mode_no)[1:]
    time += mm_base.get_total_time(stor_output, gate_based = True)
    pulse_ge_str = [['qubit', 'ge', 'pi', 0]] 
    time+= mm_base.get_total_time(pulse_ge_str, gate_based = True) * 8/6
    stor_input = mm_base.compound_storage_gate(input = True, storage_no = mode_no)[:-1]
    time += mm_base.get_total_time(stor_input, gate_based = True)

    # print('stor_input', stor_input)
    # print('stor_output', stor_output)
    return time

def parity_post_select_modified(data, attrs, readout_threshold, readouts_per_rep):
    '''
    Post select the data based on the threshold, every readouts_per_rep readouts

    (based on preselection measurement pulse during active reset)
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

def parity_post_select(data, attrs, threshold, readouts_per_rep):
    '''
    Post select the data based on the threshold, every readouts_per_rep readouts

    OLD VERSION: EG couldn't understand here; better code with comments above
    '''
    Ilist = []
    Qlist = []


    # assume the 3rd one is post selection data, 4th to readouts_per_rep is actual data

    II = data['idata']
    IQ = data['qdata']
    
    
    
    for k in range(len(II) // readouts_per_rep):
        index_4k_plus_2 = readouts_per_rep * k + 2
        index_4k_plus_end = readouts_per_rep * k + readouts_per_rep-1

        result_Ig = []
        result_Ie = []
        
        # Ensure the indices are within the list bounds
        if index_4k_plus_2 < len(II) and index_4k_plus_end < len(II):
            # Check if the value at 4k+2 exceeds the threshold
            if II[index_4k_plus_2] < threshold:
                # Add the value at 4k+3 to the result list
                for jj in range(index_4k_plus_2+1, index_4k_plus_end+1):
                    result_Ig.append(II[jj])
                    result_Ie.append(IQ[jj])

            Ilist.append(result_Ig)
            Qlist.append(result_Ie)
    return Ilist, Qlist

def parity_temp_display(data, attrs, active_reset = False, threshold = 4, readouts_per_rep = 4, return_all_param = True, title='Ramsey'):
    '''
    Convert each traces into 0/1 digits 

    '''
    state_string_list = []
    if active_reset:
        Ilist, Qlist = parity_post_select_modified(data, attrs, threshold, readouts_per_rep)
        data['i_selected'] = Ilist
        data['q_selected'] = Qlist

    else:
        Ilist = []
        Qlist = []

        #rounds = attrs['config']['expt']['rounds']
        reps = attrs['config']['expt']['reps']
        #expts = attrs['config']['expt']['expts']

        I_data = data['idata'] # in shape rounds(1) x expts (1)  x reps   x read_num
        Q_data = data['qdata'] 

        # assume we have made 80 parity measurements
        # reshape data into (reps, read_num)
        I_data_rs = np.reshape(I_data, (reps, readouts_per_rep))
        Q_data_rs = np.reshape(Q_data, (reps, readouts_per_rep))
        # for k in range(len(data['idata']) // readouts_per_rep):
        #     result_Ig = []
        #     result_Ie = []
        #     for jj in range(readouts_per_rep):
        #         result_Ig.append(data['idata'][k*readouts_per_rep+jj])
        #         result_Ie.append(data['qdata'][k*readouts_per_rep+jj])
        #     Ilist.append(result_Ig)
        #     Qlist.append(result_Ie)


        data['i_selected'] = I_data_rs
        data['q_selected'] = Q_data_rs

    # now single shot bin shots
    for i in range(len(data['i_selected'])):
        result_Ig = []
        for j in range(len(data['i_selected'][i])):
            if data['i_selected'][i][j] > threshold:
                result_Ig.append(1)
            else:
                result_Ig.append(0)

        state_string_list.append(result_Ig)

    return data, state_string_list