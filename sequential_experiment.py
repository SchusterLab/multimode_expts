import numpy as np
import os
import time
from tqdm import tqdm
import json
from slab.datamanagement import SlabFile
from slab import get_next_filename, AttrDict
from slab.experiment import Experiment
import experiments as meas
from slab.instruments import *
import yaml
from scipy.interpolate import UnivariateSpline
from slab import get_next_filename, get_current_filename

from slab.dsfit import *
from scipy.optimize import curve_fit
import experiments.fitting as fitter
from scipy.fft import fft, fftfreq
from multimode_expts.MM_base import MM_base
from multimode_expts.MM_rb_base import MM_rb_base
from multimode_expts.MM_dual_rail_base import MM_dual_rail_base
from multimode_expts.fit_display import * # for generate combos in MultiRBAM

'''
Updates: 

03/05/2024 : Added sweep of dc flux as we probe a mode
'''

#===========================auto calibration analysis function=========================================#
def RB_fid_limit(data1, data2, fitparams=None):
        
    # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
    # Remove the first and last point from fit in case weird edge measurements
    # fitparams = [None, 1/max(data['xpts']), None, None]
    # fitparams = None
    p_avgi1, pCov_avgi1 = fitter.fitdecaysin(
        data1['xpts'][:-1], data1["avgi"][:-1], fitparams=fitparams)
    p_avgi2, pCov_avgi2 = fitter.fitdecaysin(
        data2['xpts'][:-1], data2["avgi"][:-1], fitparams=fitparams)
    data1['fit_avgi'] = p_avgi1
    data1['fit_err_avgi'] = pCov_avgi1
    data2['fit_avgi'] = p_avgi2
    data2['fit_err_avgi'] = pCov_avgi2

    a1 = p_avgi1
    a2 = p_avgi2
    T = data2['xpts'][0]
    A = a1[0]
    B = a1[4]-a1[0]
    k1 = -np.log((a2[4]-B)/A)/T
    kp = -np.log(a2[0]/A)/T-k1

    print('Fitted k1, kp: ', k1, kp, 'MHz')
    

    if kp<0: kpp=0
    rate = a1[1]
    kbs = k1+kpp/2
    F = 1-np.pi/4*kbs/rate
    print('Fitted g: ', rate, 'MHz')
    print('Fidelity limit: ', F)

    return k1, kp, rate, F

def pi_length_calibration(data, fitparams=None):  
    '''
    Input data from length rabi plot 
    '''
    # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset, decay time offset]
    # Remove the first and last point from fit in case weird edge measurements
    # fitparams = [None, 1/max(data['xpts']), None, None]
    # fitparams = None
    p, pCov = fitter.fitdecaysin(
        data['xpts'][:-1], data["avgi"][:-1], fitparams=fitparams)

    #xpts_ns = data['xpts']*1e3

    if p[2] > 180:
        p[2] = p[2] - 360
    elif p[2] < -180:
        p[2] = p[2] + 360
    if p[2] < 0:
        pi_length = (1/2 - p[2]/180)/2/p[1]
    else:
        pi_length = (3/2 - p[2]/180)/2/p[1]
    pi2_length = pi_length/2
    return pi_length, pi2_length

def fit_t1(data, fit=True, title="$T_1$", **kwargs):
    data['fit_avgi'], data['fit_err_avgi'] = fitter.fitexp(data['xpts'][:-1], data['avgi'][:-1], fitparams=None)
    p = data['fit_avgi']
    pCov = data['fit_err_avgi']
    captionStr = f'$T_1$ fit [us]: {p[3]:.3} $\pm$ {np.sqrt(pCov[3][3]):.3}'
    #print(captionStr)
    return p[3], np.sqrt(pCov[3][3])

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
    on_resonant_id = None
    
    for i, row in enumerate(y_list):
        # Initial guess for the parameters: amplitude, decay rate, frequency, phase, offset
        # initial_guess = [np.max(row) - np.min(row), 5, 1, 0, np.mean(row)]
        print(i)
        
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
            print(np.abs(oscillation_rate))
            if np.abs(oscillation_rate) < min_rate and np.abs(popt[0]) > (0.8 * max_amplitude): 
                # the second condition is to block pathological cases where rsonant frequency is not near where amp is max
                max_amplitude = np.abs(popt[0])
                min_rate = np.abs(oscillation_rate)
                on_resonant_frequency = frequency_points[i]
                on_resonant_rate = oscillation_rate
                on_resonant_id = i
                
        except RuntimeError:
            # If the fit fails, we skip this frequency
            continue
    
    return on_resonant_frequency, on_resonant_rate, on_resonant_id#====================================================================#

def manipulate_dc_flux_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, dcflux = None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.parity_freq'
    experiment_name = 'ParityFreqExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['ParityFreqExperimentDCSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # load YOKO
    dcflux = YokogawaGS200(address="192.168.137.148")
    dcflux.set_output(True)
    dcflux.set_mode('current')
    dcflux.ramp_current(0.000, sweeprate=0.002)


    for index, current in enumerate(np.linspace(loaded['ParityFreqExperimentDCSweep']['flux_start'], 
                                           loaded['ParityFreqExperimentDCSweep']['flux_stop'], 
                                           loaded['ParityFreqExperimentDCSweep']['flux_expts'])):
        loaded[experiment_name]['current'] = current
        current_now = current / 1000
        if abs(current_now) > 0.03: break    # for safety


        print("%d: flux Driving at %.3f mA " % (index, current_now * 1000))
        dcflux.ramp_current(current_now, sweeprate=0.002)


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        run_exp.cfg.device.readout.relax_delay = 400 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

    #After expt is over, set current back to 0.32mA
    dcflux.ramp_current(0.00032, sweeprate=0.002)


def fluxspectroscopy_f0g1_dc_flux_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, dcflux = None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rf_flux_spectroscopy_f0g1'
    experiment_name = 'FluxSpectroscopyF0g1Experiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['FluxSpectroscopyF0g1ExperimentSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # load YOKO
    dcflux = YokogawaGS200(address="192.168.137.148")
    dcflux.set_output(True)
    dcflux.set_mode('current')
    dcflux.ramp_current(0.000, sweeprate=0.002)

    # Sample data
    x_data = np.array(loaded['FluxSpectroscopyF0g1ExperimentSweep']['flux_sample_list'])
    y_data = np.array(loaded['FluxSpectroscopyF0g1ExperimentSweep']['f0g1_freq_sample_list'])
    z_data = np.array(loaded['FluxSpectroscopyF0g1ExperimentSweep']['pi_length_sample_list'])

    # Sort the data
    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]
    z_data = z_data[sorted_indices]

    # Fit a spline to the data
    from scipy.interpolate import CubicSpline, interp1d
    y_spline = CubicSpline(x_data, y_data)
    z_spline = interp1d(x_data, z_data)


    for index, current in enumerate(np.linspace(loaded['FluxSpectroscopyF0g1ExperimentSweep']['flux_start'], 
                                           loaded['FluxSpectroscopyF0g1ExperimentSweep']['flux_stop'], 
                                           loaded['FluxSpectroscopyF0g1ExperimentSweep']['flux_expts'])):
        loaded[experiment_name]['current'] = current
        current_now = current / 1000
        if abs(current_now) > 0.03: break    # for safety

        freq_update = y_spline(current)  # new f0g1 frequency
        pi_length_update = z_spline(current)  # new pi length

        loaded[experiment_name]['pre_sweep_pulse'][0][-1] = freq_update
        loaded[experiment_name]['pre_sweep_pulse'][2][-1] = pi_length_update
        loaded[experiment_name]['post_sweep_pulse'][0][-1] = freq_update
        loaded[experiment_name]['post_sweep_pulse'][2][-1] = pi_length_update

        print("%d: flux Driving at %.3f mA with F0G1 frequency %.6f" % (index, current_now * 1000, freq_update))
        dcflux.ramp_current(current_now, sweeprate=0.002)


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        run_exp.cfg.device.readout.relax_delay = 200 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

    #After expt is over, set current back to 0.32mA
    dcflux.ramp_current(0.00032, sweeprate=0.002)

def cavity_t1_dc_flux_sweep_new(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, dcflux = None):
    '''
    this function is to sweep the dc flux and measure the T1 of the cavity
    with 3 different step sizes 1, 7, 15
    '''
    #====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
    #===================================================================#

    experiment_class = 'single_qubit.t1_cavity'
    experiment_name = 'T1CavityExperiment'   
    sweep_name = 'T1CavityExperiment_DC_sweep_new'

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['T1CavityExperiment_DC_sweep_new'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # load YOKO
    dcflux = YokogawaGS200(address="192.168.137.148")
    dcflux.set_output(True)
    dcflux.set_mode('current')
    sweep_rate_coupler = 0.0005
    dcflux.ramp_current(0.000, sweeprate=sweep_rate_coupler)


    for index, current in enumerate(loaded[sweep_name]['currents_interped']):

        loaded[experiment_name]['current'] = current
        current_now = current / 1000
        if abs(current_now) > 0.03: break    # for safety
        print("%d: flux Driving at %.3f mA" % (index, current_now * 1000))
        dcflux.ramp_current(current_now, sweeprate=sweep_rate_coupler)

        aa = loaded[sweep_name]['f0g1_freq_interped'][index]
        bb = loaded[sweep_name]['pi_length_interped'][index]

        loaded[experiment_name]['f0g1_param'] = [aa, 15000, bb]
        print('f0g1 parameters are ', loaded[experiment_name]['f0g1_param'])

        for idx, step in enumerate(loaded[sweep_name]['step_sizes']):
            if step == 1 and current<0.55: 
                continue
            else:
                if step == 15: 
                    loaded[experiment_name]['expts'] = 60
                else:
                    loaded[experiment_name]['expts'] = 30
                loaded[experiment_name]['step'] = step
                print('config is', loaded[experiment_name])
                run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
                run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")
                run_exp.go(analyze=False, display=False, progress=False, save=True)

    #After expt is over, set current back to 0
    dcflux.ramp_current(0.000, sweeprate=sweep_rate_coupler)


def cavity_t1_dc_flux_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, dcflux = None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.t1_cavity'
    experiment_name = 'T1CavityExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['T1CavityExperiment_DC_sweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # load YOKO
    dcflux = YokogawaGS200(address="192.168.137.148")
    dcflux.set_output(True)
    dcflux.set_mode('current')
    dcflux.ramp_current(0.000, sweeprate=0.002)


    for index, current in enumerate(np.linspace(loaded['T1CavityExperiment_DC_sweep']['flux_start'], 
                                           loaded['T1CavityExperiment_DC_sweep']['flux_stop'], 
                                           loaded['T1CavityExperiment_DC_sweep']['flux_expts'])):
        loaded[experiment_name]['current'] = current
        current_now = current / 1000
        if abs(current_now) > 0.03: break    # for safety
        print("%d: flux Driving at %.3f mA" % (index, current_now * 1000))
        dcflux.ramp_current(current_now, sweeprate=0.002)

        aa = loaded['T1CavityExperiment_DC_sweep']['f0g1_freq_fit_param'][0]
        bb = loaded['T1CavityExperiment_DC_sweep']['f0g1_freq_fit_param'][1]
        cc = loaded['T1CavityExperiment_DC_sweep']['f0g1_freq_fit_param'][2]
        xx = (current-loaded['T1CavityExperiment_DC_sweep']['sweet_spot'])/loaded['T1CavityExperiment_DC_sweep']['period']
        f0g1_freq = aa+bb/(np.sqrt(abs(np.cos(np.pi*xx)))+cc)
        print('New f0g1 freq: ',f0g1_freq, ' (MHz)')
        # loaded[experiment_name]['f0g1_freq'] = f0g1_freq
        loaded[experiment_name]['f0g1_param'] = [f0g1_freq, loaded['T1CavityExperiment_DC_sweep']['f0g1_gain'], loaded['T1CavityExperiment_DC_sweep']['f0g1_pi']]
        # yaml_cfg.device.QM.pulses.f0g1.freq[loaded['T1CavityExperiment_DC_sweep']['cavity']-1] = f0g1_freq
        # yaml_cfg.device.QM.pulses.f0g1.gain[loaded['T1CavityExperiment_DC_sweep']['cavity']-1] = loaded['T1CavityExperiment_DC_sweep']['f0g1_gain']
        # yaml_cfg.device.QM.pulses.f0g1.length[loaded['T1CavityExperiment_DC_sweep']['cavity']-1] = loaded['T1CavityExperiment_DC_sweep']['f0g1_pi']


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 10 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

    #After expt is over, set current back to 0
    dcflux.ramp_current(0.000, sweeprate=0.002)


def pulseprobe_f0g1_dc_flux_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, dcflux = None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.pulse_probe_f0g1_spectroscopy'
    experiment_name = 'PulseProbeF0g1SpectroscopyExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # load YOKO
    dcflux = YokogawaGS200(address="192.168.137.148")
    dcflux.set_output(True)
    dcflux.set_mode('current')
    dcflux.ramp_current(0.000, sweeprate=0.002)

    # initialiaze spline for f0g1 in  prepulse 
    if loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['prepulse_f0g1']:
        x_data = np.array(loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['flux_sample_list'])
        y_data = np.array(loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['f0g1_freq_sample_list'])

        # Sort the data
        sorted_indices = np.argsort(x_data)
        x_data = x_data[sorted_indices]
        y_data = y_data[sorted_indices]

        # Fit a spline to the data
        spline = UnivariateSpline(x_data, y_data)


    for index, current in enumerate(np.linspace(loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['flux_start'], 
                                           loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['flux_stop'], 
                                           loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['flux_expts'])):
        
        loaded[experiment_name]['current'] = current
        current_now = current / 1000
        if abs(current_now) > 0.03: break    # for safety
        print("%d: flux Driving at %.3f mA" % (index, current_now * 1000))
        dcflux.ramp_current(current_now, sweeprate=0.002)

        if loaded['PulseProbeF0g1SpectroscopyFluxSweepExperiment']['prepulse_f0g1']:
            freq_update = spline(current)
            #print the updated freq
            print('New f0g1 freq: ',freq_update, ' (MHz)')
            loaded[experiment_name]['pre_sweep_pulse'][0][-1] = freq_update


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        run_exp.cfg.device.readout.relax_delay = 10 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

    #After expt is over, set current back to 0
    dcflux.ramp_current(0.000, sweeprate=0.002)

def cavity_ramsey_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.t2_cavity'
    experiment_name = 'CavityRamseyExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['CavityRamseySweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, gain in enumerate(np.arange(loaded['CavityRamseySweep']['gain_start'], 
                                           loaded['CavityRamseySweep']['gain_stop'], 
                                           loaded['CavityRamseySweep']['gain_step'])):

        print('Index: %s Gain. = %s MHz' %(index, gain))
        loaded[experiment_name]['user_defined_pulse'][2] = gain
        #print(loaded[experiment_name]['pre_sweep_pulse'])


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)



def ramsey_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.t2_ramsey'
    experiment_name = 'RamseyExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['RamseySweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['RamseySweep']['freq_start'], 
                                           loaded['RamseySweep']['freq_stop'], 
                                           loaded['RamseySweep']['freq_step'])):

        print('Index: %s Freq. = %s MHz' %(index, freq))
        loaded[experiment_name]['pre_sweep_pulse'][0][0] = freq
        #print(loaded[experiment_name]['pre_sweep_pulse'])


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 1000 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def sideband_general_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.sideband_general'
    experiment_name = 'SidebandGeneralExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SidebandGeneralExperimentSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['SidebandGeneralExperimentSweep']['freq_start'], 
                                           loaded['SidebandGeneralExperimentSweep']['freq_stop'], 
                                           loaded['SidebandGeneralExperimentSweep']['freq_step'])):

        print('Index: %s Freq. = %s MHz' %(index, freq))
        loaded[experiment_name]['flux_drive'][1] = freq

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5
        if run_exp.cfg.expt.active_reset: 
            run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def storage_sideband_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.sideband_general'
    experiment_name = 'SidebandGeneralExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['StorageSidebandExperimentSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass
    
    for mode_idx, mode_freq in enumerate(loaded['StorageSidebandExperimentSweep']['mode_freq_list']):
        print('-------------------------------------------------')
        print('Mode Index: %s Freq. = %s MHz' %(mode_idx + 1, mode_freq))
        loaded['StorageSidebandExperimentSweep']['freq_start'] = mode_freq - loaded['StorageSidebandExperimentSweep']['chevron_freq_span']/2
        loaded['StorageSidebandExperimentSweep']['freq_stop'] = mode_freq + loaded['StorageSidebandExperimentSweep']['chevron_freq_span']/2

        for index, freq in enumerate(np.arange(loaded['StorageSidebandExperimentSweep']['freq_start'], 
                                            loaded['StorageSidebandExperimentSweep']['freq_stop'], 
                                            loaded['StorageSidebandExperimentSweep']['freq_step'])):
            
            if freq<1000: 
                loaded[experiment_name]['flux_drive'][0] = 'low'
            else:
                loaded[experiment_name]['flux_drive'][0] = 'high'

            print('Index: %s Freq. = %s MHz' %(index, freq))
            loaded[experiment_name]['flux_drive'][1] = freq
            gain = loaded['StorageSidebandExperimentSweep']['gain_list'][mode_idx]
            loaded[experiment_name]['flux_drive'][2] = gain
            print('Gain: ', gain)

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            # special updates on device_config file
            run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
            # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
            # run_exp.cfg.device.manipulate.readout_length = 5
            # run_exp.cfg.device.storage.readout_length = 5

            run_exp.go(analyze=False, display=False, progress=False, save=True)        




def HistogramExperiment_freq_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.single_shot'
    experiment_name = 'HistogramExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['HistogramExperiment_freq_sweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['HistogramExperiment_freq_sweep']['freq_start'], 
                                           loaded['HistogramExperiment_freq_sweep']['freq_stop'], 
                                           loaded['HistogramExperiment_freq_sweep']['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        loaded[experiment_name]['freq'] = freq
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.qubit.f_ge = [freq]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def HistogramExperiment_ef_freq_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.single_shot'
    experiment_name = 'HistogramExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['HistogramExperiment_ef_freq_sweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['HistogramExperiment_ef_freq_sweep']['freq_start'], 
                                           loaded['HistogramExperiment_ef_freq_sweep']['freq_stop'], 
                                           loaded['HistogramExperiment_ef_freq_sweep']['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        loaded[experiment_name]['freq'] = freq
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.qubit.f_ef = [freq]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def SingleRB_sweep_freq(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rb_ziqian'
    experiment_name = 'SingleRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleRB_sweep_freq'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['SingleRB_sweep_freq']['freq_start'], 
                                           loaded['SingleRB_sweep_freq']['freq_stop'], 
                                           loaded['SingleRB_sweep_freq']['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        loaded[experiment_name]['freq'] = freq
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.qubit.f_ge = [freq]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def SingleRB_sweep_pi_amp(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rb_ziqian'
    experiment_name = 'SingleRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleRB_sweep_pi_amp'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, amp in enumerate(np.arange(loaded['SingleRB_sweep_pi_amp']['amp_start'], 
                                           loaded['SingleRB_sweep_pi_amp']['amp_stop'], 
                                           loaded['SingleRB_sweep_pi_amp']['amp_step'])):

        print('Index: %s Amp. = %s ' %(index, amp))
        loaded[experiment_name]['amp'] = amp
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.qubit.pulses.pi_ge.gain = [amp]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

#sweep depth and prepulse
def SingleRB_sweep_depth_and_prepulse(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================# 

    experiment_class = 'single_qubit.rb_ziqian'
    experiment_name = 'SingleRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleRB_sweep_depth_and_prepulse'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass
    
    for jdx, bool in enumerate(loaded['SingleRB_sweep_depth_and_prepulse']['prepulses_bool']):
        print('-------------------------------------------------')
        print('Index: %s prepulse = %s ' %(jdx, bool))

        # if offset is None: 
        #     print('offset is None')
        #     loaded[experiment_name]['prepulse'] = False
        #     loaded[experiment_name]['postpulse'] = False
        #     loaded[experiment_name]['f0g1_offset'] = 0
        if bool:
            print('offset is not None') 
            loaded[experiment_name]['prepulse'] = True
            loaded[experiment_name]['postpulse'] = True
        else: 
            print('offset is None')
            loaded[experiment_name]['prepulse'] = False
            loaded[experiment_name]['postpulse'] = False
            loaded[experiment_name]['f0g1_offset'] = 0

        loaded[experiment_name]['pre_sweep_pulse'] = loaded['SingleRB_sweep_depth_and_prepulse']['pre_sweep_pulses'][jdx]
        loaded[experiment_name]['post_sweep_pulse'] = loaded['SingleRB_sweep_depth_and_prepulse']['post_sweep_pulses'][jdx]
        loaded[experiment_name]['f0g1_offset'] = loaded['SingleRB_sweep_depth_and_prepulse']['f0g1_offsets'][jdx]

        print('Prepulse: ', loaded[experiment_name]['prepulse'])
        print('Postpulse: ', loaded[experiment_name]['postpulse'])
        print('pre_sweep_pulse: ', loaded[experiment_name]['pre_sweep_pulse'])
        print('post_sweep_pulse: ', loaded[experiment_name]['post_sweep_pulse'])
        print('f0g1_offset: ', loaded[experiment_name]['f0g1_offset'])

        for index, depth in enumerate(np.arange(loaded['SingleRB_sweep_depth_and_prepulse']['depth_start'], 
                                            loaded['SingleRB_sweep_depth_and_prepulse']['depth_stop'], 
                                            loaded['SingleRB_sweep_depth_and_prepulse']['depth_step'])):

            print('Index: %s depth. = %s ' %(index, depth))
            loaded[experiment_name]['rb_depth'] = depth
            

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            # special updates on device_config file
            #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
            run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
            # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
            # run_exp.cfg.device.manipulate.readout_length = 5
            # run_exp.cfg.device.storage.readout_length = 5

            run_exp.go(analyze=False, display=False, progress=False, save=True)


#sweep depth of rb experiment
def SingleRB_sweep_depth(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================# 

    experiment_class = 'single_qubit.rb_ziqian'
    experiment_name = 'SingleRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleRB_sweep_depth'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, depth in enumerate(loaded['SingleRB_sweep_depth']['depth_list']):

        print('Index: %s depth. = %s ' %(index, depth))
        loaded[experiment_name]['rb_depth'] = depth
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
        # run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


class MM_dual_rail_seq_exp:
    def __init__(self):
        '''Contains sequential experiments for dual rail based sequential experiments'''

    def DualRail_sweep_depth_and_single_spec_and_stor(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        '''
        This performs dual rail rb for a given target mode in presence of a single spectator. 
        This function sweeps the single spectator modes and also internally sweeps all the cardinal states 
        that the spectator mode can be in . 
        This function will also sweep the target modes
        '''
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   
        sweep_experiment_name = 'DualRail_sweep_depth_and_single_spec_and_stor'

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass
        
        start_pair = loaded[sweep_experiment_name]['start_pair']
        target_start, spec_start = start_pair
        
        for kdx, target_mode in enumerate(loaded[sweep_experiment_name]['target_mode_list']):
            if target_mode < target_start: continue

            print('----------------------############---------------------------')
            print('Kndex: %s target mode. = %s ' %(kdx, target_mode))
            loaded[sweep_experiment_name]['target_mode'] = target_mode
            mode_list = [1,2,3,4,5,6,7]
            mode_list.remove(target_mode)
            new_mode_list = mode_list.copy()
            for spec in mode_list: 
                if target_mode == target_start and spec < spec_start: 
                    new_mode_list.remove(spec)
            loaded[sweep_experiment_name]['target_spec_list'] = new_mode_list
            # print(new_mode_list)

            loaded[experiment_name]['bs_para'] = loaded[sweep_experiment_name]['bs_para_list'][kdx]
            print(loaded[experiment_name]['bs_para'])

            self.SingleBeamSplitterRBPostSelection_sweep_depth_and_single_spec(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                        prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name, yaml_cfg])
            





    def SingleBeamSplitterRBPostSelection_sweep_depth_and_single_spec(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, 
                                                                    prep_init = False, prep_params = None):
        '''
        This performs dual rail rb for a given target mode in presence of a single spectator. 
        This function sweeps the single spectator modes and also internally sweeps all the cardinal states 
        that the spectator mode can be in . 
        '''
        if prep_init: 
            config_path, loaded, experiment_class, experiment_name, sweep_experiment_name, yaml_cfg = prep_params
            # target_mode = loaded[sexperiment_name]['target_mode']
        else: 
            #====================================================================#
            config_path = config_file
            print('Config will be', config_path)

            with open(config_file, 'r') as cfg_file:
                yaml_cfg = yaml.safe_load(cfg_file)
            yaml_cfg = AttrDict(yaml_cfg)

            with open(exp_param_file, 'r') as file:
                # Load the YAML content
                loaded = yaml.safe_load(file)
            #===================================================================# 

            experiment_class = 'single_qubit.rb_BSgate_postselection'
            experiment_name = 'SingleBeamSplitterRBPostSelection'   
            sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth_and_single_spec'


            for keys in loaded[experiment_name].keys():
                try:
                    loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
                except:
                    pass
            
        
        target_mode = loaded[sweep_experiment_name]['target_mode']

        #depth_array = np.array([1,2,3,4,5,10,20])
        for jdx, target_spec in enumerate(loaded[sweep_experiment_name]['target_spec_list']):
        #for index, depth in enumerate(depth_array):
            print('-------------------------------------------------')
            print('Jndex: %s target spec. = %s ' %(jdx, target_spec))
            # loaded[experiment_name]['ram_prepulse'][1] = #num_occupied_smodes
            # loaded[experiment_name]['ram_prepulse'][3] = loaded[sweep_experiment_name]['prepulse_vars_list'][jdx] 

            dummy = MM_dual_rail_base(cfg = yaml_cfg)
            prepulse_strs = [dummy.prepulse_str_for_random_ram_state(1, [target_mode], target_spec, i) for i in range(1, 7)]
            print(prepulse_strs)
            loaded[experiment_name]['ram_prepulse_strs'] = prepulse_strs
            self.SingleBeamSplitterRBPostSelection_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                        prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])
            


    def SingleBeamSplitterRBPostSelection_sweep_depth_and_ram(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   
        sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth_and_ram'

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        #depth_array = np.array([1,2,3,4,5,10,20])
        for jdx, num_occupied_smodes in enumerate(loaded[sweep_experiment_name]['num_occupied_smodes_list']):
        #for index, depth in enumerate(depth_array):
            print('-------------------------------------------------')
            print('Jndex: %s depth. = %s ' %(jdx, num_occupied_smodes))
            loaded[experiment_name]['ram_prepulse'][1] = num_occupied_smodes
            loaded[experiment_name]['ram_prepulse'][3] = loaded[sweep_experiment_name]['prepulse_vars_list'][jdx] 

            self.SingleBeamSplitterRBPostSelection_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                        prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])

    def SingleBeamSplitterRB_stor_ramsey_spec(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        '''
        Depth sweep over all storage-storage pairs  for ramsey inpresence of beamsplitters
        '''
        #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
        #===================================================================# 
        experiment_class = 'single_qubit.rb_BSgate_check_target'
        experiment_name = 'SingleBeamSplitterRB_check_target'   
        sweep_experiment_name = 'SingleBeamSplitterRB_stor_ramsey_spec'

        # for keys in loaded[experiment_name].keys():
        #     try:
        #         loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
        #     except:
        #         pass

        for idx, stor_no in enumerate(loaded[sweep_experiment_name]['stor_list']):
            for jdx, spec_no in enumerate(loaded[sweep_experiment_name]['spec_list']):
                if stor_no == spec_no: # no self -self pair
                    continue
                if [stor_no, spec_no] in loaded[sweep_experiment_name]['skip_pairs']:
                    continue
                print('-------------------------------------------------')
                print('Index: %s Storage = %s, Spectator = %s ' %(idx, stor_no, spec_no))
                # Frequency 
                loaded[sweep_experiment_name]['wait_freq'] = loaded[sweep_experiment_name]['wait_freq_list'][stor_no -1][spec_no -1]

                # update prepulse/post pulse
                loaded[sweep_experiment_name]['pre_sweep_pulse'][-1][1] = 'M1-S' + str(stor_no)
                loaded[sweep_experiment_name]['post_sweep_pulse'][0][1] = 'M1-S' + str(stor_no)

                # update bs_para
                loaded[sweep_experiment_name]['bs_para'] = loaded[sweep_experiment_name]['bs_para_list'][spec_no -1]

                # print(loaded[sweep_experiment_name])

                _ = self.SingleBeamSplitterRB_check_target_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                                  prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name])



    def SingleBeamSplitterRB_check_target_sweep_depth(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, prep_init = False, prep_params = None):
        '''
            Although this function is uses the daughter function SingleBeamSplitterRBPostSelection, 
            the post selection part is unimportant. 

            This is gate based ramsey experiment (instead of time based) for target state in presence
            spectator beamsplitters.
        '''
    # #====================================================================#
    #     config_path = config_file
    #     print('Config will be', config_path)

    #     with open(config_file, 'r') as cfg_file:
    #         yaml_cfg = yaml.safe_load(cfg_file)
    #     yaml_cfg = AttrDict(yaml_cfg)

    #     with open(exp_param_file, 'r') as file:
    #         # Load the YAML content
    #         loaded = yaml.safe_load(file)
    # #===================================================================# 

    #     experiment_class = 'single_qubit.rb_BSgate_check_target'
    #     experiment_name = 'SingleBeamSplitterRB_check_target'   
    #     sweep_experiment_name = 'SingleBeamSplitterRB_check_target_sweep_depth'

    #     for keys in loaded[experiment_name].keys():
    #         try:
    #             loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
    #         except:
    #             pass
        if prep_init: 
            config_path, loaded, experiment_class, experiment_name, sweep_experiment_name = prep_params
        else: 
        #====================================================================#
            config_path = config_file
            print('Config will be', config_path)

            with open(config_file, 'r') as cfg_file:
                yaml_cfg = yaml.safe_load(cfg_file)
            yaml_cfg = AttrDict(yaml_cfg)

            with open(exp_param_file, 'r') as file:
                # Load the YAML content
                loaded = yaml.safe_load(file)
        #===================================================================# 

            experiment_class = 'single_qubit.rb_BSgate_check_target'
            experiment_name = 'SingleBeamSplitterRB_check_target'   
            sweep_experiment_name = 'SingleBeamSplitterRB_check_target_sweep_depth'

        # NOTe the following code is not part of the "prep function"
        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass
        
        loaded[sweep_experiment_name]['depth_list'] = np.arange(loaded[sweep_experiment_name]['depth_start'],
                                                                loaded[sweep_experiment_name]['depth_stop'],
                                                                loaded[sweep_experiment_name]['depth_step'])
        length = len(loaded[sweep_experiment_name]['depth_list'])
        loaded[sweep_experiment_name]['reps_list'] = [loaded[sweep_experiment_name]['repss'] for _ in range(length)] # * len(loaded[sweep_experiment_name]['depth_list'])

        # print(loaded[sweep_experiment_name])
        self.SingleBeamSplitterRBPostSelection_sweep_depth(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path, exp_param_file=exp_param_file,
                                                    prep_init = True, prep_params = [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name],
                                                    skip_ss = True)
            
    def SingleBeamSplitterRBPostSelection_sweep_depth(self,soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None,
                                                    prep_init = False, prep_params = None, skip_ss = False):
        '''
        Prep_init: True if the config, experiment names are already initialized in some other parent function that calls this as a child
        prep_params: [config_path, loaded, experiment_class, experiment_name, sweep_experiment_name]
        skip_ss: Skip the single shot part of the experiment (True/False) (first depth will have it, later depths will not )
        '''
        if prep_init: 
            config_path, loaded, experiment_class, experiment_name, sweep_experiment_name = prep_params
        else: 
        #====================================================================#
            config_path = config_file
            print('Config will be', config_path)

            with open(config_file, 'r') as cfg_file:
                yaml_cfg = yaml.safe_load(cfg_file)
            yaml_cfg = AttrDict(yaml_cfg)

            with open(exp_param_file, 'r') as file:
                # Load the YAML content
                loaded = yaml.safe_load(file)
        #===================================================================# 

            experiment_class = 'single_qubit.rb_BSgate_postselection'
            experiment_name = 'SingleBeamSplitterRBPostSelection'  
            sweep_experiment_name = 'SingleBeamSplitterRBPostSelection_sweep_depth' 

            for keys in loaded[experiment_name].keys():
                try:
                    loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
                except:
                    pass

        #depth_array = np.array([1,2,3,4,5,10,20])
        for index, depth in enumerate(loaded[sweep_experiment_name]['depth_list']):
        #for index, depth in enumerate(depth_array):
            print('Index: %s depth. = %s ' %(index, depth))
            if index != 0 and skip_ss:
                loaded[experiment_name]['calibrate_single_shot'] = False
            loaded[experiment_name]['rb_depth'] = depth
            loaded[experiment_name]['rb_reps'] = loaded[sweep_experiment_name]['reps_list'][index]
            

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            print(run_exp.cfg.expt)
            run_exp.go(analyze=False, display=False, progress=False, save=True)

    def SingleBeamSplitterRBPostSelection_sweep_depth_storsweep(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep'][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        
        for stor_idx, stor_no in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep']['stor_list']): 

            print('-------------------------------------------------')
            print('Storage Index: %s Storage No. = %s ' %(stor_idx, stor_no))
            man_idx = 2   # 1 or 2

            # create prepulse , postpulse, post selection pulse 
            mm_base = MM_base(cfg = yaml_cfg)
            pre_sweep_pulse_str = [['qubit', 'ge', 'pi'],
                            ['qubit', 'ef', 'pi'],
                                ['man', 'M' + str(man_idx) , 'pi']]
            post_sweeep_pulse_str = [['qubit', 'ge', 'hpi'], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx)], 
                        ['qubit', 'ge', 'hpi']]
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no), 'hpi'], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no), 'hpi'],
                            ['qubit', 'ge', 'hpi'], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx)], 
                            ['qubit', 'ge', 'hpi']]
            bs_para_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no), 'hpi']]
            
            creator = mm_base.get_prepulse_creator(pre_sweep_pulse_str)
            loaded[experiment_name]['pre_sweep_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(post_sweeep_pulse_str)
            loaded[experiment_name]['post_sweep_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            loaded[experiment_name]['post_selection_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(bs_para_str)
            bs_para_pulse = creator.pulse.tolist()
            loaded[experiment_name]['bs_para'] = [bs_para_pulse[0][0], bs_para_pulse[1][0], bs_para_pulse[2][0],  bs_para_pulse[6][0]]

            print('Prepulse: ', loaded[experiment_name]['pre_sweep_pulse'])
            print('Postpulse: ', loaded[experiment_name]['post_sweep_pulse'])
            print('Post Selection Pulse: ', loaded[experiment_name]['post_selection_pulse'])
            print('BS Para: ', loaded[experiment_name]['bs_para'])
            
            
            
            #depth_array = np.array([1,2,3,4,5,10,20])
            for index, depth in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep']['depth_list']):
            #for index, depth in enumerate(depth_array):
                print('Index: %s depth. = %s ' %(index, depth))
                loaded[experiment_name]['rb_depth'] = depth

                loaded[experiment_name]['rb_reps'] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_storsweep']['reps_list'][index]
                

                run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


                run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

                # special updates on device_config file
                #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
                # run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
                # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
                # run_exp.cfg.device.manipulate.readout_length = 5
                # run_exp.cfg.device.storage.readout_length = 5
                run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
                print(run_exp.cfg.expt)
                run_exp.go(analyze=False, display=False, progress=False, save=True)




    def SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep(self, soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.rb_BSgate_postselection'
        experiment_name = 'SingleBeamSplitterRBPostSelection'   

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep'][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        
        for stor_idx, stor_no in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['stor_list']): 

            print('-------------------------------------------------')
            print('Storage Index: %s Storage No. = %s ' %(stor_idx, stor_no))
            man_idx = 1   # 1 or 2

            # create prepulse , postpulse, post selection pulse 
            mm_base = MM_base(cfg = yaml_cfg)
            pre_sweep_pulse_str = [['qubit', 'ge', 'pi', 0],
                            ['qubit', 'ef', 'pi', 0],
                                ['man', 'M' + str(man_idx) , 'pi', 0]]
            
            creator = mm_base.get_prepulse_creator(pre_sweep_pulse_str)
            loaded[experiment_name]['pre_sweep_pulse'] = creator.pulse.tolist()
            loaded[experiment_name]['bs_para'] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['bs_para_list'][stor_no-1]

            print('Prepulse: ', loaded[experiment_name]['pre_sweep_pulse'])
            print('BS Para: ', loaded[experiment_name]['bs_para'])

            for index, depth in enumerate(loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['depth_list']):
                print('Index: %s depth. = %s ' %(index, depth))
                loaded[experiment_name]['rb_depth'] = depth

                loaded[experiment_name]['rb_reps'] = loaded['SingleBeamSplitterRBPostSelection_sweep_depth_defined_storsweep']['reps_list'][index]
                

                run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


                run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

                # special updates on device_config file
                run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
                print(run_exp.cfg.expt)
                run_exp.go(analyze=False, display=False, progress=False, save=True)

def cavity_temperature_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    '''
    Assumiung photon loaded into storage via man  1
    '''
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================# 

    experiment_class = 'single_qubit.t2_ramsey'
    experiment_name = 'RamseyExperiment'   
    sweep_experiment_name = 'cavity_temperature_sweep'

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for targ_idx, targ_label in enumerate(loaded[sweep_experiment_name]['targ_list']): 

        
        # create prepulse , post pulse 
        mm_base = MM_base(cfg = yaml_cfg)
        pre_sweep_pulse_str = []
        if targ_label != 'S0':
            pre_sweep_pulse_str.append(['storage', 'M1-' + str(targ_label), 'pi', 0])
            loaded[experiment_name]['prepulse'] = True
        if targ_label == 'S0':
            loaded[experiment_name]['prepulse'] = False # no prepulse for S0 which is basically bare man pop 
        
        print('Prepulse: ', pre_sweep_pulse_str)
        creator = mm_base.get_prepulse_creator(pre_sweep_pulse_str)
        loaded[experiment_name]['pre_sweep_pulse'] = creator.pulse.tolist()

        
        print(loaded[experiment_name])

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
        print(run_exp.cfg.expt)
        run_exp.go(analyze=False, display=False, progress=False, save=True)


class sweep_cavity_ramsey_expts: 
    def __init__(self):
        '''Contains sequential experiments for cavity ramsey based experiments'''

    
    def cavity_ramsey_with_spectators(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
        '''
        perform cavity ramsey with all spectator storage modes occupied
        '''
    #====================================================================#
        config_path = config_file
        print('Config will be', config_path)

        with open(config_file, 'r') as cfg_file:
            yaml_cfg = yaml.safe_load(cfg_file)
        yaml_cfg = AttrDict(yaml_cfg)

        with open(exp_param_file, 'r') as file:
            # Load the YAML content
            loaded = yaml.safe_load(file)
    #===================================================================# 

        experiment_class = 'single_qubit.t2_cavity'
        experiment_name = 'CavityRamseyExperiment'   
        sweep_experiment_name = 'cavity_ramsey_with_spectators'

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded[sweep_experiment_name][keys]   # overwrite the single experiment file with new paramters
            except:
                pass

        

        # create prepulse , post pulse 
        mode_list = [1,2,3,4,5,6,7]
        pre_sweep_pulse = []
        mm_base = MM_dual_rail_base(cfg = yaml_cfg)
        
        for mode_idx, mode_no in enumerate(mode_list):
            if mode_no != loaded[eexperiment_name]['storage_ramsey'][1]: # skip the target mode
                prep_stor = mm_base_dummy.prep_random_state_mode(3, mode_no)  # prepare the storage state + 
                pre_sweep_pulse += prep_stor
        # prep man1 half pi photon 
        prep_man = mm_base_dummy.prep_random_state_mode(3, 1)[:-1] # prepare the man state + 
        pre_sweep_pulse += prep_man
        post_sweep_pulse = prep_man[::-1]
        print('Prepulse: ', pre_sweep_pulse)
        print('Postpulse: ', post_sweep_pulse)
        loaded[experiment_name]['pre_sweep_pulse'] = mm_base.get_prepulse_creator(pre_sweep_pulse).pulse.tolist()
        loaded[experiment_name]['post_sweep_pulse'] = mm_base.get_prepulse_creator(post_sweep_pulse).pulse.tolist()
        
                
        print(loaded[experiment_name])

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
        print(run_exp.cfg.expt)
        run_exp.go(analyze=False, display=False, progress=False, save=True)

def cross_kerr_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
    '''
    Assumiung photon loaded into storage via man  1
    '''
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================# 

    experiment_class = 'single_qubit.t2_cavity'
    experiment_name = 'CavityRamseyExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['cross_kerr_sweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    loaded[experiment_name]['man_idx'] = 1
    for targ_idx, targ_label in enumerate(loaded['cross_kerr_sweep']['targ_list']): 

        print('-------------------------------------------------')
        print('Target Index: %s Target No. = %s ' %(targ_idx, targ_label))

        loaded[experiment_name]['ramsey_freq'] = loaded['cross_kerr_sweep']['ramsey_freq_list'][targ_idx]

        for spec_idx, spec_label in enumerate(loaded['cross_kerr_sweep']['spec_list']): 

            print('-------------------------------------------------')
            print('Spec Index: %s Spec label = %s ' %(spec_idx, spec_label))

            # create prepulse , post pulse 
            mm_base = MM_base(cfg = yaml_cfg)
            pre_sweep_pulse_str = []
            if targ_label != spec_label:
                pre_sweep_pulse_str.append(['qubit', 'ge', 'pi'])
                pre_sweep_pulse_str.append(['qubit', 'ef', 'pi'])
                pre_sweep_pulse_str.append(['man', 'M1' , 'pi'])
                pre_sweep_pulse_str.append(['storage', 'M1-' + str(spec_label), 'pi'])
        
            pre_sweep_pulse_str.append(['qubit', 'ge', 'hpi']) 
            pre_sweep_pulse_str.append(['qubit', 'ef', 'pi'])
            print('Prepulse: ', pre_sweep_pulse_str)

            post_sweep_pulse_str = []
            
            if targ_label[0] != 'M': # target mode is NOT a manipulate 
                pre_sweep_pulse_str.append(['man', 'M1' , 'pi'])
                # post_sweep_pulse_str.append(['storage', 'M1-' + str(spec_label), 'pi']) # for debugging
                post_sweep_pulse_str.append(['man', 'M1' , 'pi'])

            post_sweep_pulse_str.append(['qubit', 'ef', 'pi'])
            post_sweep_pulse_str.append(['qubit', 'ge', 'hpi']) # for debug

            creator = mm_base.get_prepulse_creator(pre_sweep_pulse_str)
            loaded[experiment_name]['pre_sweep_pulse'] = creator.pulse.tolist()
            creator = mm_base.get_prepulse_creator(post_sweep_pulse_str)
            loaded[experiment_name]['post_sweep_pulse'] = creator.pulse.tolist()

            # make sure target mode pulse is set up 
            loaded[experiment_name]['storage_ramsey'] = [False, int(targ_label[-1])] # this does not matter if target mode is manipulate
            ## get f0g1 params 
            loaded[experiment_name]['user_defined_pulse'] =[True, None, None, None, None, 4]
            print('User Defined Pulse: ', loaded[experiment_name]['user_defined_pulse'])

            if targ_label[0] != 'M': # if target mode is NOT manipulate
                loaded[experiment_name]['storage_ramsey'][0] = True
                loaded[experiment_name]['user_defined_pulse'][0] = False
            else: 
                f0g1_str = [['man', targ_label , 'pi']]
                loaded[experiment_name]['user_defined_pulse'] =[True, pulse[0], gain, ramp_sigma, length, channel]

            
            print(loaded[experiment_name])

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            # special updates on device_config file
            run_exp.cfg.device.readout.relax_delay = 100 # Wait time between experiments [us]
            print(run_exp.cfg.expt)
            run_exp.go(analyze=False, display=False, progress=False, save=True)

def SingleBeamsplitterRB_sweep_depth(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================# 

    experiment_class = 'single_qubit.rb_BSgate'
    experiment_name = 'SingleBeamSplitterRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleBeamSplitterRB_sweep_depth'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    #depth_array = np.array([1,2,3,4,5,10,20])
    for index, depth in enumerate(loaded['SingleBeamSplitterRB_sweep_depth']['depth_list']):
    #for index, depth in enumerate(depth_array):
        print('Index: %s depth. = %s ' %(index, depth))
        loaded[experiment_name]['rb_depth'] = depth
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
        run_exp.cfg.device.readout.relax_delay = 5000 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)



def PhaseSweepAverager_sweep_reps_middlepulse(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.phase_sweep_averager'
    experiment_name = 'PhaseSweepAveragerExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['PhaseSweepAverager_sweep_reps_middlepulse'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    
    for index, reps in enumerate(np.arange(loaded['PhaseSweepAverager_sweep_reps_middlepulse']['reps_start'], 
                                           loaded['PhaseSweepAverager_sweep_reps_middlepulse']['reps_stop'], 
                                           loaded['PhaseSweepAverager_sweep_reps_middlepulse']['reps_step'])):

        # print('Index: %s Phase. = %s ' %(index, phase))
        # loaded[experiment_name]['f0g1_offset'] = phase
        print('Index: %s Reps. = %s ' %(index, reps))
        loaded[experiment_name]['reps_middlepulse'] = reps

        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
        # run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def SingleRB_sweep_f0g1_phase(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rb_ziqian'
    experiment_name = 'SingleRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleRB_sweep_f0g1_phase'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    print('-------------------------------------------------')
    print(np.arange(loaded['SingleRB_sweep_f0g1_phase']['phase_start'], 
                                           loaded['SingleRB_sweep_f0g1_phase']['phase_stop'], 
                                           loaded['SingleRB_sweep_f0g1_phase']['phase_step']))
    print(loaded['SingleRB_sweep_f0g1_phase']['phase_start'])
    print(loaded['SingleRB_sweep_f0g1_phase']['phase_stop'])
    print(loaded['SingleRB_sweep_f0g1_phase']['phase_step'])
    for index, phase in enumerate(np.arange(loaded['SingleRB_sweep_f0g1_phase']['phase_start'], 
                                           loaded['SingleRB_sweep_f0g1_phase']['phase_stop'], 
                                           loaded['SingleRB_sweep_f0g1_phase']['phase_step'])):

        print('Index: %s Phase. = %s ' %(index, phase))
        loaded[experiment_name]['f0g1_offset'] = phase
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def SingleRB_sweep_hpi_amp(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rb_ziqian'
    experiment_name = 'SingleRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleRB_sweep_hpi_amp'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, amp in enumerate(np.arange(loaded['SingleRB_sweep_hpi_amp']['amp_start'], 
                                           loaded['SingleRB_sweep_hpi_amp']['amp_stop'], 
                                           loaded['SingleRB_sweep_hpi_amp']['amp_step'])):

        print('Index: %s Amp. = %s ' %(index, amp))
        loaded[experiment_name]['amp'] = amp
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def length_rabi_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.length_rabi_general'
    experiment_name = 'LengthRabiGeneralExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['LengthRabiGeneralExperimentSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['LengthRabiGeneralExperimentSweep']['freq_start'], 
                                           loaded['LengthRabiGeneralExperimentSweep']['freq_stop'], 
                                           loaded['LengthRabiGeneralExperimentSweep']['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        loaded[experiment_name]['freq'] = freq

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def length_rabi_f0g1_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.length_rabi_f0g1_general'
    experiment_name = 'LengthRabiGeneralF0g1Experiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['LengthRabiGeneralF0g1ExperimentSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['LengthRabiGeneralF0g1ExperimentSweep']['freq_start'], 
                                           loaded['LengthRabiGeneralF0g1ExperimentSweep']['freq_stop'], 
                                           loaded['LengthRabiGeneralF0g1ExperimentSweep']['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        loaded[experiment_name]['freq'] = freq

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        if loaded[experiment_name]['active_reset']:
            print('doesnt make sense to active reset in this exp')
        run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def displace_enhanced_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'qubit_cavity.displacement_enhanced_sideband'
    experiment_name = 'DisplacementEnhancedSidebandExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['DisplacementEnhancedSidebandExperimentPhaseSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['DisplacementEnhancedSidebandExperimentSweep']['sweep_start'], 
                                           loaded['DisplacementEnhancedSidebandExperimentSweep']['sweep_stop'], 
                                           loaded['DisplacementEnhancedSidebandExperimentSweep']['sweep_step'])):

        #print('Index: %s Phase. = %s deg' %(index, phase))
        #loaded[experiment_name]['cavity_disp_pulse'][4] = phase
        #loaded[experiment_name]['hadamard'][1] = phase

        print('Index: %s Freq. = %s MHz' %(index, freq))
        loaded[experiment_name]['wait'][1] = freq

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def dc_flux_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None, dcflux = None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.cavity_spectroscopy'
    experiment_name = 'CavitySpectroscopyExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['DCFluxSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # load YOKO
    dcflux = YokogawaGS200(address="192.168.137.148")
    dcflux.set_output(True)
    dcflux.set_mode('current')
    dcflux.ramp_current(0.000, sweeprate=0.002)


    for index, current in enumerate(np.linspace(loaded['DCFluxSweep']['curr_start'], 
                                           loaded['DCFluxSweep']['curr_stop'], 
                                           loaded['DCFluxSweep']['curr_expts'])):

        print("%d: flux Driving at %.3f mA" % (index, current * 1000))
        dcflux.ramp_current(current, sweeprate=0.002)


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

    #After expt is over, set current back to 0
    dcflux.ramp_current(0.000, sweeprate=0.002)


def gain_displace_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.pulse_probe_spectroscopy'
    experiment_name = 'PulseProbeSpectroscopyExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['GainDisplaceSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass



    for index, gain in enumerate(np.arange(loaded['GainDisplaceSweep']['gain_start'], 
                                           loaded['GainDisplaceSweep']['gain_stop'], 
                                           loaded['GainDisplaceSweep']['gain_step'])):

        print('Index: %s Gain. = %s Dac units' %(index, gain))
        #print(type(loaded[experiment_name]['cavity_name']))
        loaded[experiment_name]['cavity_gain'] = int(gain)


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def storage_cross_ac_phase_correction(cfg, spectator_mode_no, target_mode_no, spec_reps = 1):
    '''
    Can find effect of driving a MX-SY sideband on state of SZ mode 

    returns gate based pulse str 
    '''
    # spectator_mode_no = 2
    # target_mode_no = 1
    man_no = 1

    #cfg = yaml_cfg

    omega_target = cfg.device.storage.idling_freq[target_mode_no -1]
    # omega_f0g1 = cfg.device.manipulate.idling_freq[man_no - 1]
    phi_target = cfg.device.storage.idling_phase[target_mode_no -1][target_mode_no -1]
    phi_spec_on_target = cfg.device.storage.idling_phase[target_mode_no -1][spectator_mode_no -1]
    # phi_f0g1 = cfg.device.manipulate.idling_phase[man_no - 1][man_no - 1]

    mm_base = MM_rb_base(cfg = cfg)

    # create prepulse 
    prepulse_str = [['qubit', 'ge', 'hpi',0]]
    stor_input = mm_base.compound_storage_gate(input = True, storage_no = target_mode_no)
    #stor_ouput = mm_base.compound_storage_gate(input = True, storage_no = target_mode_no)

    # spector pulse 
    qubit_spec_init = [['qubit', 'ge', 'hpi', 0]]
    spectator_pulse_str = mm_base.compound_storage_gate(input = True, storage_no = spectator_mode_no)
    for _ in range(spec_reps-1): 
        spectator_pulse_str += mm_base.compound_storage_gate(input = False, storage_no = spectator_mode_no) + mm_base.compound_storage_gate(input = True, storage_no = spectator_mode_no)

    spectator_pulse_str_rev = mm_base.compound_storage_gate(input = False, storage_no = spectator_mode_no)

    # idling_times 
    qubit_spec_init_idling_time = mm_base.get_total_time(qubit_spec_init, gate_based = True)
    spec_idling_time = mm_base.get_total_time(spectator_pulse_str, gate_based = True)
    targ_idling_time = mm_base.get_total_time([['storage', 'M' + str(man_no) + '-S' + str(target_mode_no), 'pi', 0]], gate_based = True)

    stor_output_str = mm_base.compound_storage_gate(input = False, storage_no = target_mode_no)
    ge_virtual_phase = 0 #omega_target * (spec_idling_time + qubit_spec_init_idling_time) + (2 * phi_target) + ( phi_spec_on_target/2) #omega_f0g1 * targ_idling_time # + phi_target + phi_f0g1
    print('Idling time is ', spec_idling_time + qubit_spec_init_idling_time)

    if target_mode_no == spectator_mode_no: 
        ge_virtual_phase = 0
    post_pulse_str = [['qubit', 'ge', 'hpi', ge_virtual_phase]]

    # Concatenate all lists along axis 0
    concatenated_list = prepulse_str + stor_input #+ stor_output_str + post_pulse_str
    if spectator_mode_no != target_mode_no:
        concatenated_list += qubit_spec_init + spectator_pulse_str #+ spectator_pulse_str_rev + spectator_pulse_str
    concatenated_list += stor_output_str + post_pulse_str

    # print(concatenated_list)
    return concatenated_list



def single_shot_phase_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.single_shot_prepulse'
    experiment_name = 'HistogramPrepulseExperiment'   

    

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['HistogramPrepulseExperimentPhaseSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass
    
    skip_mode_no = loaded['HistogramPrepulseExperimentPhaseSweep']['skip_mode_no']
    
    for targ_idx, target_mode_no in enumerate(loaded['HistogramPrepulseExperimentPhaseSweep']['target_mode_list']):
        print('#-------------------------------------------------')
        print('Target Index: %s Target No. = %s ' %(targ_idx, target_mode_no))

        for spec_idx, spectator_mode_no in enumerate(loaded['HistogramPrepulseExperimentPhaseSweep']['spectator_mode_list']):
            print('##-------------------------------------------------')
            print('Spec Index: %s Spec No. = %s ' %(spec_idx, spectator_mode_no))

            if (spectator_mode_no <= skip_mode_no) and (target_mode_no <= skip_mode_no):
                print('Skipping')
                continue

            loaded['HistogramPrepulseExperimentPhaseSweep']['target_mode_no'] = target_mode_no
            loaded['HistogramPrepulseExperimentPhaseSweep']['spectator_mode_no'] = spectator_mode_no

            for spec_reps_idx, spec_reps in enumerate(loaded['HistogramPrepulseExperimentPhaseSweep']['spec_reps_list']): 
                print('###-------------------------------------------------')
                print('Spec Reps Index: %s Spec Reps = %s ' %(spec_reps_idx, spec_reps))

                loaded['HistogramPrepulseExperimentPhaseSweep']['spec_reps']  = spec_reps

                single_shot_phase_sweep_run(soccfg=soccfg, path=path, prefix=prefix, config_file=config_file,
                                            config_path=config_path,
                                             exp_param_file=exp_param_file, loaded = loaded, 
                                             experiment_name = experiment_name, experiment_class = experiment_class, 
                                             yaml_cfg = yaml_cfg)
                                                  

                
            
def single_shot_phase_sweep_run(soccfg=None, path=None, prefix=None, config_file=None, config_path = None, exp_param_file=None,
                                 loaded = None, experiment_name = None, experiment_class = None, yaml_cfg = None):
    '''
    This is recallable method for running single shot phase sweep experiment on a specifc target and spectator mode
    '''
    if loaded[experiment_name]['gate_based']:
        loaded[experiment_name]['pre_sweep_pulse'] = storage_cross_ac_phase_correction(cfg = yaml_cfg, 
                                                                                       spectator_mode_no = loaded['HistogramPrepulseExperimentPhaseSweep']['spectator_mode_no'],
                                                                                        target_mode_no = loaded['HistogramPrepulseExperimentPhaseSweep']['target_mode_no'], 
                                                                                        spec_reps = loaded['HistogramPrepulseExperimentPhaseSweep']['spec_reps'])
        initial_phase = loaded[experiment_name]['pre_sweep_pulse'][-1][-1]



    for index, phase in enumerate(np.arange(loaded['HistogramPrepulseExperimentPhaseSweep']['phase_start'], 
                                           loaded['HistogramPrepulseExperimentPhaseSweep']['phase_stop'], 
                                           loaded['HistogramPrepulseExperimentPhaseSweep']['phase_step'])):

        print('Index: %s Phase. = %s Deg' %(index, phase))
        #print(type(loaded[experiment_name]['cavity_name']))
        if loaded[experiment_name]['gate_based']:
            loaded[experiment_name]['pre_sweep_pulse'][loaded['HistogramPrepulseExperimentPhaseSweep']['sweep_id']][-1] = initial_phase+   phase
        else: 
            loaded[experiment_name]['pre_sweep_pulse'][3][loaded['HistogramPrepulseExperimentPhaseSweep']['sweep_id']] = phase
        print(loaded[experiment_name]['pre_sweep_pulse'])


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)



def single_shot_time_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#


    experiment_class = 'single_qubit.single_shot_prepulse'
    experiment_name = 'HistogramPrepulseExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['HistogramPrepulseExperimentTimeSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    if loaded[experiment_name]['gate_based']:
        mm_base = MM_base(cfg = yaml_cfg)
        target_mode_no = loaded['HistogramPrepulseExperimentTimeSweep']['target_mode_no']
        qubit_init_str = [['qubit', 'ge', 'hpi', 0]]
        input_str = mm_base.compound_storage_gate(input = True, storage_no = target_mode_no)
        buffer_str = [['buffer', '_', '_',0],]
        output_str = mm_base.compound_storage_gate(input = False, storage_no = target_mode_no)
        qubit_final_str = [['qubit', 'ge', 'hpi', 0]]
        loaded[experiment_name]['pre_sweep_pulse'] = qubit_init_str + input_str + buffer_str + output_str + qubit_final_str

    for index, time in enumerate(np.arange(loaded['HistogramPrepulseExperimentTimeSweep']['time_start'], 
                                           loaded['HistogramPrepulseExperimentTimeSweep']['time_stop'], 
                                           loaded['HistogramPrepulseExperimentTimeSweep']['time_step'])):

        print('Index: %s Time. = %s us' %(index, time))
        #print(type(loaded[experiment_name]['cavity_name']))
        if loaded[experiment_name]['gate_based']:
            print(loaded[experiment_name]['pre_sweep_pulse'][loaded['HistogramPrepulseExperimentTimeSweep']['time2sweep']])
            loaded[experiment_name]['pre_sweep_pulse'][loaded['HistogramPrepulseExperimentTimeSweep']['time2sweep']][-1] = time 
        else:
            loaded[experiment_name]['pre_sweep_pulse'][2][loaded['HistogramPrepulseExperimentTimeSweep']['time2sweep']] = time
        print(loaded[experiment_name]['pre_sweep_pulse'])

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def single_shot_freq_sweep(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.single_shot_prepulse'
    experiment_name = 'HistogramPrepulseExperiment'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['HistogramPrepulseExperimentFreqSweep'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass



    for index, freq in enumerate(np.arange(loaded['HistogramPrepulseExperimentFreqSweep']['freq_start'], 
                                           loaded['HistogramPrepulseExperimentFreqSweep']['freq_stop'], 
                                           loaded['HistogramPrepulseExperimentFreqSweep']['freq_step'])):

        print('Index: %s Freq. = %s MHz' %(index, freq))
        #print(type(loaded[experiment_name]['cavity_name']))
        loaded[experiment_name]['pre_sweep_pulse'][0][loaded['HistogramPrepulseExperimentFreqSweep']['freq2sweep']] = freq


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def SingleBeamSplitterRB_sweep_freq(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rb_BSgate'
    experiment_name = 'SingleBeamSplitterRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleBeamSplitterRB_sweep_freq'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, freq in enumerate(np.arange(loaded['SingleBeamSplitterRB_sweep_freq']['freq_start'], 
                                           loaded['SingleBeamSplitterRB_sweep_freq']['freq_stop'], 
                                           loaded['SingleBeamSplitterRB_sweep_freq']['freq_step'])):

        print('Index: %s Freq. = %s GHz' %(index, freq))
        loaded[experiment_name]['bs_para'][0] = freq
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.bs_freq = [freq]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def SingleBeamSplitterRB_gain_freq(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.rb_BSgate'
    experiment_name = 'SingleBeamSplitterRB'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleBeamSplitterRB_gain_freq'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    for index, gain in enumerate(np.arange(loaded['SingleBeamSplitterRB_gain_freq']['gain_start'], 
                                           loaded['SingleBeamSplitterRB_gain_freq']['gain_stop'], 
                                           loaded['SingleBeamSplitterRB_gain_freq']['gain_step'])):

        print('Index: %s Gain. = %s ' %(index, gain))
        loaded[experiment_name]['bs_para'][1] = gain
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        # run_exp.cfg.device.bs_freq = [freq]
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)

def sideband_fidelity_optimization(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
    #=========================Outer loop: sweeping different drive gain==========================#
    gain2sweep = loaded['sideband_fidelity_optimization']['gain2sweep']
    iterative_rounds = loaded['sideband_fidelity_optimization']['iterative_rounds']  # number of iterations, 2 is enough
    freq_guess = loaded['sideband_fidelity_optimization']['freq_guess']  # initial guess of the resonant frequency
    search_bandwidth = loaded['sideband_fidelity_optimization']['search_bandwidth']  # search bandwidth for the frequency
    search_step = loaded['sideband_fidelity_optimization']['search_step']  # search step for the frequency


    prefix = 'optimization_result_'+loaded['sideband_fidelity_optimization']['sideband_name']
    fname2save = get_next_filename(path, prefix, suffix='.h5')
    print('Results will be saved to', fname2save)
    fidelity_list = []
    kappa1_list = []
    kappa_phi_list = []
    rate_list = []
    gain_list = []
    resonant_freq_list = []
    with SlabFile(path+'\\'+fname2save, 'w') as f:
        f.append_line('Initialize', [0])
    for index1, gain in enumerate(gain2sweep):
        print('Index1: %s Gain. = %s ' %(index1, gain))
        #=========================step 1: sweeping different drive gain==========================#
        freq_now, resonant_freq_list = iterative_chevron_calibration(config_file=config_file, exp_param_file=exp_param_file, freq_guess=freq_guess, 
                                                                      iterative_rounds=iterative_rounds, search_bandwidth=search_bandwidth, search_step=search_step, 
                                                                      gain=gain, resonant_freq_list=resonant_freq_list)

        #=========================step 2 long sideband length rabi to extract kappa and g==========================#      
        #resonant_freq_list.append(freq_now)
        #gain_list.append(gain)
        

        # short sideband length rabi
        experiment_class = 'single_qubit.sideband_general'
        experiment_name = 'SidebandGeneralExperiment'  
        loaded[experiment_name]['flux_drive'][0] = 'low'
        if freq_now > 1000:
            loaded[experiment_name]['flux_drive'][0] = 'high'
        loaded[experiment_name]['flux_drive'][1] = freq_now
        loaded[experiment_name]['flux_drive'][2] = gain
        loaded[experiment_name]['flux_drive'][3] = 0.005  # 2 sigma ramp set by default
        loaded[experiment_name]['start'] = 0.007
        loaded[experiment_name]['step'] = loaded['sideband_fidelity_optimization']['step_fitting']
        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]

        run_exp.go(analyze=False, display=False, progress=False, save=True)
        temp_data1 = run_exp.data

        # long sideband length rabi
        loaded['sideband_fidelity_optimization']['freq_resonant'] = freq_now
        experiment_class = 'single_qubit.sideband_general'
        experiment_name = 'SidebandGeneralExperiment'  
        loaded[experiment_name]['flux_drive'][0] = 'low'
        if freq_now > 1000:
            loaded[experiment_name]['flux_drive'][0] = 'high'
        loaded[experiment_name]['flux_drive'][1] = freq_now
        loaded[experiment_name]['flux_drive'][2] = gain
        loaded[experiment_name]['flux_drive'][3] = 0.005  # 2 sigma ramp set by default
        loaded[experiment_name]['start'] = loaded['sideband_fidelity_optimization']['T1_delay']
        loaded[experiment_name]['step'] = loaded['sideband_fidelity_optimization']['step_fitting']
        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]

        run_exp.go(analyze=False, display=False, progress=False, save=True)
        temp_data2 = run_exp.data

        k1, kp, rate, F = RB_fid_limit(temp_data1, temp_data2)
        fidelity_list.append(F)
        kappa1_list.append(k1)
        kappa_phi_list.append(kp)
        rate_list.append(rate)

        print('Fidelity list is %s' % fidelity_list)
        print('kappa1 list is %s' % kappa1_list)
        print('kappa_phi list is %s' % kappa_phi_list)
        print('rate list is %s' % rate_list)
        print('Resonant freq list is %s' % resonant_freq_list)
        #===================================================================#

        #=========================step 3 Save data, fittings are carried out but needs fine tuning probably==========================#
        with SlabFile(path+'\\'+fname2save, 'w') as f:
            f.append_line('fidelity_list', fidelity_list)
            f.append_line('kappa1_list', kappa1_list)
            f.append_line('kappa_phi_list', kappa_phi_list)
            f.append_line('rate_list', rate_list)
            f.append_line('resonant_freq_list', resonant_freq_list)

def iterative_chevron_calibration(soccfg = None, path = None, prefix = None, config_file=None, exp_param_file=None, freq_guess=None, iterative_rounds=None, 
                                  search_bandwidth=None, search_step=None, gain=None, resonant_freq_list=None, scaling_factors=None):
    '''
    Performs sideband general experiment sweep; with finer frequency search per iteration
    '''
    #====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
    #===================================================#
    #print(soccfg)

    freq_now = freq_guess
    bandwidth_now = search_bandwidth
    search_step_now = search_step
    
    for index1 in range(0, iterative_rounds):
        y_list = []
        freq_list = []
        calibration_file_path = []
        print('Iterative round. = %s ' %(index1))
        experiment_class = 'single_qubit.sideband_general'
        experiment_name = 'SidebandGeneralExperiment'   

        for keys in loaded[experiment_name].keys():
            try:
                loaded[experiment_name][keys] = loaded['sideband_fidelity_optimization'][keys]   # overwrite the single experiment file with new paramters
            except:
                pass


        ## produce the chevron pattern
        for index2, freq in enumerate(np.arange(freq_now-bandwidth_now/2, freq_now+bandwidth_now/2, search_step_now)):

            print('Index2: %s Freq. = %s MHz' %(index2, freq))
            loaded[experiment_name]['flux_drive'][0] = 'low'
            if freq > 1000:
                loaded[experiment_name]['flux_drive'][0] = 'high'
            loaded[experiment_name]['flux_drive'][1] = freq
            loaded[experiment_name]['flux_drive'][2] = gain
            loaded[experiment_name]['flux_drive'][3] = 0.005  # 2 sigma ramp set by default

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            # special updates on device_config file
            run_exp.cfg.device.readout.relax_delay = 1500 # Wait time between experiments [us]

            run_exp.go(analyze=False, display=False, progress=True, save=True)
            temp_data= run_exp.data
            signal_y = temp_data['avgi']
            len_x = temp_data['xpts']
            freq_list.append(freq)
            y_list.append(signal_y)
            calibration_file_path.append(run_exp.fname)
        #======================== analysis===========================================#
        on_resonant_freq, on_resonant_rate, on_resonant_id = find_on_resonant_frequency(np.array(y_list), len_x, freq_list, fitparams=None)
        print('On resonant frequency is %s MHz' % on_resonant_freq)
        print('On resonant rate is %s MHz' % on_resonant_rate)
        freq_now = on_resonant_freq
        bandwidth_now = search_step_now*scaling_factors[0]
        search_step_now = bandwidth_now * scaling_factors[1]
        # with SlabFile(path+'\\'+fname2save, 'a') as f:
        #     f.append_line('calibration_file_path', calibration_file_path)

            
        resonant_freq_list.append(freq_now)

    return freq_now, resonant_freq_list

def storage_t1_optimization(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
    #=========================Outer loop: sweeping different mdoe freqs==========================#

    for mode_idx, storage_name in enumerate(loaded['storage_t1_optimization']['storage_names']):
        print('Index1: %s Storage. = %s ' %(mode_idx, storage_name))
        prefix = 'storage_t1_result_'+storage_name 
        fname2save = get_next_filename(path, prefix, suffix='.h5')
        print('Results will be saved to', fname2save)

        freq_guess = loaded['storage_t1_optimization']['freq_guess'][mode_idx]  # initial guess of the resonant frequency
        iterative_rounds = loaded['storage_t1_optimization']['iterative_rounds']  # number of iterations, 2 is enough
        search_bandwidth = loaded['storage_t1_optimization']['search_bandwidth']  # search bandwidth for the frequency
        search_step = loaded['storage_t1_optimization']['search_step']  # search step for the frequency
        scaling_factors = loaded['storage_t1_optimization']['scaling_factors']  # scaling factors for the drive
        gain = loaded['storage_t1_optimization']['gain_list'][mode_idx]  # gain for the drive
        print('Gain is %s' % gain)
        
        resonant_freq_list = []
        with SlabFile(path+'\\'+fname2save, 'w') as f:
            f.append_line('Initialize', [0])


        #=========================step 1 Iteratively sweeping frequency as rough calibration======================#
        freq_now, resonant_freq_list = iterative_chevron_calibration(soccfg = soccfg, path=path, prefix=prefix, config_file=config_file, exp_param_file=exp_param_file, freq_guess=freq_guess, 
                                                                      iterative_rounds=iterative_rounds, search_bandwidth=search_bandwidth, search_step=search_step, 
                                                                      gain=gain, resonant_freq_list=resonant_freq_list, scaling_factors = scaling_factors)
            

        # short sideband length rabi
        experiment_class = 'single_qubit.sideband_general'
        experiment_name = 'SidebandGeneralExperiment'  
        loaded[experiment_name]['flux_drive'][0] = 'low'
        if freq_now > 1000:
            loaded[experiment_name]['flux_drive'][0] = 'high'
        loaded[experiment_name]['flux_drive'][1] = freq_now
        loaded[experiment_name]['flux_drive'][2] = gain
        loaded[experiment_name]['flux_drive'][3] = 0.005  # 2 sigma ramp set by default
        loaded[experiment_name]['start'] = 0.007
        loaded[experiment_name]['step'] = loaded['sideband_fidelity_optimization']['step_fitting']
        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]

        run_exp.go(analyze=False, display=False, progress=True, save=True)
        temp_data1 = run_exp.data

        # fit data to extract pi pulse length and half pi pulse length 
        pi_length, hpi_length = pi_length_calibration(temp_data1)
        print('Pi length is %s us' % pi_length)
        print('Half pi length is %s us' % hpi_length)



        #========================= T1 experiment ==========================#
        experiment_class = 'single_qubit.sideband_t1_general'
        experiment_name = 'SidebandT1GeneralExperiment'
        loaded[experiment_name]['pre_sweep_pulse'][0][-1] = freq_now
        loaded[experiment_name]['pre_sweep_pulse'][1][-1] = gain
        loaded[experiment_name]['pre_sweep_pulse'][2][-1] = pi_length
        loaded[experiment_name]['pre_sweep_pulse'][4][-1] = 3 if freq_now > 1000 else 1
        loaded[experiment_name]['pre_sweep_pulse'][6][-1] = 0.005 # 2 sigma ramp set by default

        loaded[experiment_name]['post_sweep_pulse'][:][0] = loaded[experiment_name]['pre_sweep_pulse'][:][-1]

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]

        run_exp.go(analyze=False, display=False, progress=True, save=True)
        temp_data2 = run_exp.data
        t1, t1_err = fit_t1(temp_data2)
        print(f'$T_1$ fit [us]: {t1:.3} $\pm$ {t1_err:.3}')


        
        # print('Fidelity list is %s' % fidelity_list)
        # print('kappa1 list is %s' % kappa1_list)
        # print('kappa_phi list is %s' % kappa_phi_list)
        # print('rate list is %s' % rate_list)
        # print('Resonant freq list is %s' % resonant_freq_list)
        #===================================================================#
       

        #=========================step 3 Save data, fittings are carried out but needs fine tuning probably==========================#
        with SlabFile(path+'\\'+fname2save, 'w') as f:
            f.append_line('gain', [gain,0])
            f.append_line('resonant_freq_list', resonant_freq_list)
            f.append_line('pi_length', [pi_length,0 ])
            f.append_line('half_pi_length', [hpi_length,0])
            f.append_line('T1', [t1,0])
            f.append_line('T1_err', [t1_err,0])

def rb_bs_optimization(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
    #=========================Outer loop: sweeping different mdoe freqs==========================#

    for idx, freq in enumerate(loaded['rb_bs_optimization']['freqs']):
        print('Index1: %s Freq. = %s ' %(idx, freq))

        # short sideband length rabi
        experiment_class = 'single_qubit.sideband_general'
        experiment_name = 'SidebandGeneralExperiment'  
        loaded[experiment_name]['flux_drive'][0] = 'low'
        if freq > 1000:
            loaded[experiment_name]['flux_drive'][0] = 'high'
        loaded[experiment_name]['flux_drive'][1] = freq
        loaded[experiment_name]['flux_drive'][2] = loaded['rb_bs_optimization']['gain']
        loaded[experiment_name]['flux_drive'][3] = 0.005  # 2 sigma ramp set by default
        #loaded[experiment_name]['start'] = 0.007
        #loaded[experiment_name]['step'] = loaded['sideband_fidelity_optimization']['step_fitting']
        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]

        run_exp.go(analyze=False, display=False, progress=True, save=True)
        temp_data1 = run_exp.data

        # fit data to extract pi pulse length and half pi pulse length 
        pi_length, hpi_length = pi_length_calibration(temp_data1)
        print('Pi length is %s us' % pi_length)
        print('Half pi length is %s us' % hpi_length)


        length_step = loaded['rb_bs_optimization']['length_step']
        length_points = loaded['rb_bs_optimization']['length_points']
        hpi_lengths = np.linspace(-length_step*length_points/2, length_step*length_points/2, length_points) + hpi_length

        for h_pi_length in hpi_lengths:
            #========================= RB Bs gate expt  ==========================#
            experiment_class = 'single_qubit.rb_BSgate'
            experiment_name = 'SingleBeamSplitterRB'
            loaded[experiment_name]['bs_para'][0]= freq
            loaded[experiment_name]['bs_para'][1] = loaded['rb_bs_optimization']['gain']
            loaded[experiment_name]['bs_para'][2] = h_pi_length
            #loaded[experiment_name]['post_sweep_pulse'][:][0] = loaded[experiment_name]['pre_sweep_pulse'][:][-1]

            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            # special updates on device_config file
            run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]

            run_exp.go(analyze=False, display=False, progress=True, save=True)
            temp_data = run_exp.data

            avg_readout = []
            for i in range(len(temp_data['Idata'])):
                counting = 0
                for j in temp_data['Idata'][i]:
                    if j>temp_data['thresholds']:
                        counting += 1
                avg_readout.append(counting/len(temp_data['Idata'][i]))
            #
            # avg_readout = RB_extract(temp_data)
            mean = np.average(avg_readout)
            err = np.std(avg_readout)/np.sqrt(len(avg_readout))
            print(mean, err)

def rb_bs_dual_rail_optimization(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
    #=========================Outer loop: sweeping different mdoe freqs==========================#

    for idx, freq in enumerate(loaded['rb_bs_dual_rail_optimization']['freqs']):
        print('Index1: %s Freq. = %s ' %(idx, freq))

        # short sideband length rabi
        if not loaded['rb_bs_dual_rail_optimization']['skip_length_for_freq_calib'][0]: 
            experiment_class = 'single_qubit.sideband_general'
            experiment_name = 'SidebandGeneralExperiment'  
            loaded[experiment_name]['flux_drive'][0] = 'low'
            if freq > 1000:
                loaded[experiment_name]['flux_drive'][0] = 'high'
            loaded[experiment_name]['flux_drive'][1] = freq
            loaded[experiment_name]['flux_drive'][2] = loaded['rb_bs_dual_rail_optimization']['gain'][idx]
            loaded[experiment_name]['flux_drive'][3] = 0.005  # 2 sigma ramp set by default
            #loaded[experiment_name]['start'] = 0.007
            #loaded[experiment_name]['step'] = loaded['sideband_fidelity_optimization']['step_fitting']
            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

            # special updates on device_config file
            run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]

            run_exp.go(analyze=False, display=False, progress=True, save=True)
            temp_data1 = run_exp.data

            # fit data to extract pi pulse length and half pi pulse length 
            pi_length, hpi_length = pi_length_calibration(temp_data1)
            print('Pi length is %s us' % pi_length)
            print('Half pi length is %s us' % hpi_length)
        else: 
            hpi_length = loaded['rb_bs_dual_rail_optimization']['skip_length_for_freq_calib'][1][idx]


        length_step = loaded['rb_bs_dual_rail_optimization']['length_step']#[idx]
        length_calib_bool = True

        while length_calib_bool:

            length_points = loaded['rb_bs_dual_rail_optimization']['length_points']
            hpi_lengths = np.linspace(-length_step*length_points/2, length_step*length_points/2, length_points) + hpi_length
            raw_fids = []
            post_fids = []

            for jdx, h_pi_length in enumerate(hpi_lengths):
                print('Index2: %s hpi_length. = %s ' %(idx, h_pi_length))
                #========================= RB Bs gate expt  ==========================#
                experiment_class_ = 'single_qubit.rb_BSgate_postselection'
                experiment_name_ = 'SingleBeamSplitterRBPostSelection'
                loaded[experiment_name_]['bs_para'][0]= freq
                loaded[experiment_name_]['bs_para'][1] = loaded['rb_bs_dual_rail_optimization']['gain'][idx]
                loaded[experiment_name_]['bs_para'][2] = h_pi_length
                #loaded[experiment_name]['post_sweep_pulse'][:][0] = loaded[experiment_name]['pre_sweep_pulse'][:][-1]

                run_exp = eval(f"meas.{experiment_class_}.{experiment_name_}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
                run_exp.cfg.expt = eval(f"loaded['{experiment_name_}']")

                # special updates on device_config file
                # run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]

                run_exp.go(analyze=False, display=False, progress=True, save=True)
                temp_data = run_exp.data
                # attrs = run_exp.attrs
                attrs = {'config': run_exp.cfg}

                avg_readout, avg_readout_post, gg, ge, eg, ee = RB_extract_postselction_excited(temp_data,attrs, active_reset=loaded[experiment_name_]['rb_active_reset'])
                raw_fid = np.average(avg_readout)
                post_fid = np.average(avg_readout_post)
                print('Raw Fidelity:', raw_fid)
                print('Post Fidelity:', post_fid)
                raw_fids.append(raw_fid)
                post_fids.append(post_fid)

            # redo length calib but now zoom in 
            length_step = length_step/4
            hpi_length = hpi_lengths[np.argmax(raw_fids)]
            if length_step < loaded['rb_bs_dual_rail_optimization']['precision']:
                print('----------------------------------------------------')
                print('FINAL hpi_length:', hpi_length)
                length_calib_bool = False





        
        # print('Fidelity list is %s' % fidelity_list)
        # print('kappa1 list is %s' % kappa1_list)
        # print('kappa_phi list is %s' % kappa_phi_list)
        # print('rate list is %s' % rate_list)
        # print('Resonant freq list is %s' % resonant_freq_list)
        #===================================================================#
       

        #=========================step 3 Save data, fittings are carried out but needs fine tuning probably==========================#
        # with SlabFile(path+'\\'+fname2save, 'w') as f:
        #     f.append_line('gain', [gain,0])
        #     f.append_line('resonant_freq_list', resonant_freq_list)
        #     f.append_line('pi_length', [pi_length,0 ])
        #     f.append_line('half_pi_length', [hpi_length,0])
            #f.append_line('T1', [t1,0])
            #f.append_line('T1_err', [t1_err,0])



def single_qubit_tomography(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.single_shot_prepulse'
    experiment_name = 'HistogramPrepulseExperiment'   


    ## tomography measurement pulses:
    measurement_pulse = ['I', 'X', 'Y']

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleQubitTomographyExperiment'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # constructing the correct prepulse+measurement pulse sequence

    vz2add = loaded['SingleQubitTomographyExperiment']['vz']

    for index, qubit_gate in enumerate(measurement_pulse):

        print('Tomography rotation:', qubit_gate)
        #print(type(loaded[experiment_name]['cavity_name']))
        if qubit_gate == 'I':
            pass
        elif qubit_gate == 'X':
            for ii in range(len(loaded[experiment_name]['pre_sweep_pulse'])):

                loaded[experiment_name]['pre_sweep_pulse'][ii].append(loaded['SingleQubitTomographyExperiment']['qubit_hpi_pulse'][ii])
            loaded[experiment_name]['pre_sweep_pulse'][3][-1] = 180+vz2add  # X/2 tomo phase
        else:
            for ii in range(len(loaded[experiment_name]['pre_sweep_pulse'])):
                loaded[experiment_name]['pre_sweep_pulse'][ii].append(loaded['SingleQubitTomographyExperiment']['qubit_hpi_pulse'][ii])
            loaded[experiment_name]['pre_sweep_pulse'][3][-1] = -90+vz2add  # Y/2 tomo phase


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")
        run_exp.go(analyze=False, display=False, progress=False, save=True)

import itertools



def MultiRBAM_sweep_depth(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================# 

    experiment_class = 'single_qubit.rbam'
    experiment_name = 'MultiRBAM'   

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['MultiRBAM_sweep_depth'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass
    
    mode_list = loaded['MultiRBAM_sweep_depth']['full_mode_list']
    num_modes_sim_rb = loaded['MultiRBAM_sweep_depth']['num_modes_sim_rb']
    skip_combos = loaded['MultiRBAM_sweep_depth']['skip_combos']
    all_combinations = generate_mode_combinations(mode_list, num_modes_sim_rb, skip_combos )
    if loaded['MultiRBAM_sweep_depth']['random_selection'][0]:
        all_combinations = random.sample(all_combinations, loaded['MultiRBAM_sweep_depth']['random_selection'][1])
    print('All combinations:', all_combinations)

    for combination in all_combinations:
        print('Combination:', combination)
        loaded[experiment_name]['mode_list'] = combination
        MultiRBAM_sweep_depth_run(soccfg=soccfg, path=path, prefix=prefix, config_path=config_path, exp_param_file=exp_param_file, 
                                  experiment_class=experiment_class, experiment_name=experiment_name, loaded=loaded)

    
def MultiRBAM_sweep_depth_run(soccfg=None, path=None, prefix=None, config_path=None, exp_param_file=None, 
                              experiment_class='single_qubit.rbam', experiment_name='MultiRBAM', loaded=None):

    

    for index, depth in enumerate(loaded['MultiRBAM_sweep_depth']['list_for_depth_list']):

        print('Index: %s depth. = %s ' %(index, depth))
        print('mode list:', loaded[experiment_name]['mode_list'])
        print('depth list:', [depth]*len(loaded[experiment_name]['mode_list']))
        loaded[experiment_name]['depth_list'] = [depth]*len(loaded[experiment_name]['mode_list'])
        

        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")


        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")

        # special updates on device_config file
        #run_exp.cfg.device.qubit.pulses.hpi_ge.gain = [amp]
        # run_exp.cfg.device.readout.relax_delay = 2500 # Wait time between experiments [us]
        # run_exp.cfg.device.readout.relax_delay = 300 # Wait time between experiments [us]
        # run_exp.cfg.device.manipulate.readout_length = 5
        # run_exp.cfg.device.storage.readout_length = 5

        run_exp.go(analyze=False, display=False, progress=False, save=True)


def single_dual_rail_tomography(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.dual_rail_single_shot'
    experiment_name = 'HistogramPrepulseDualRailExperiment'   


    ## tomography measurement pulses:
    measurement_pulse = ['I', 'X', 'Y']

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['SingleDualRailTomographyExperiment'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # constructing the correct prepulse+measurement pulse sequence

    # vz2add = loaded['SingleDualRailTomographyExperiment']['vz']

    for index, qubit_gate in enumerate(measurement_pulse):

        # print('Tomography rotation:', qubit_gate)
        #print(type(loaded[experiment_name]['cavity_name']))
        loaded[experiment_name]['measurement_pulse_list'] = []
        stor_no=loaded['SingleDualRailTomographyExperiment']['dual_rail_storage_id']
        man_idx=1

        mm_base = MM_base(cfg = yaml_cfg)
        post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0]), 'hpi',0], 
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0]), 'hpi',0],
                        ['qubit', 'ge', 'hpi',0], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                        ['qubit', 'ge', 'hpi',0]]# measure]
        creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
        I_gate1 = creator.pulse.tolist()
        post_selection_pulse_str = [
                        ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0], 
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0],
                        ['qubit', 'ge', 'hpi',0], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                        ['qubit', 'ge', 'hpi',0]]# measure]
        creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
        I_gate2 = creator.pulse.tolist()


        I_gate = [I_gate1, I_gate2]

        post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0]), 'hpi',0], 
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0]), 'hpi',0],
                        ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0], 
                        ['qubit', 'ge', 'hpi',0], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                        ['qubit', 'ge', 'hpi',0]]
        creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
        X_gate1 = creator.pulse.tolist()
        post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0], 
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0],
                        ['qubit', 'ge', 'hpi',0], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                        ['qubit', 'ge', 'hpi',0]]
        creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
        X_gate2 = creator.pulse.tolist()

        X_gate = [X_gate1, X_gate2]

        post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0]), 'hpi',0], 
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0]), 'hpi',0],
                        ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',90], 
                        ['qubit', 'ge', 'hpi',0], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                        ['qubit', 'ge', 'hpi',0]]
        creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
        Y_gate1 = creator.pulse.tolist()
        post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0], 
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1]), 'hpi',0],
                        ['qubit', 'ge', 'hpi',0], # Starting parity meas
                        ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                        ['qubit', 'ge', 'hpi',0]]
        creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
        Y_gate2 = creator.pulse.tolist()

        Y_gate = [Y_gate1, Y_gate2]
        if qubit_gate == 'I':
            loaded[experiment_name]['measurement_pulse_list'] = I_gate
            print('Running I gate')
        elif qubit_gate == 'X':
            loaded[experiment_name]['measurement_pulse_list'] = X_gate
            print('Running X gate')
        else:
            loaded[experiment_name]['measurement_pulse_list'] = Y_gate
            print('Running Y gate')


        run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
        run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")
        run_exp.go(analyze=False, display=False, progress=False, save=True)


def two_dual_rail_tomography(soccfg=None, path=None, prefix=None, config_file=None, exp_param_file=None):
#====================================================================#
    config_path = config_file
    print('Config will be', config_path)

    with open(config_file, 'r') as cfg_file:
        yaml_cfg = yaml.safe_load(cfg_file)
    yaml_cfg = AttrDict(yaml_cfg)

    with open(exp_param_file, 'r') as file:
        # Load the YAML content
        loaded = yaml.safe_load(file)
#===================================================================#

    experiment_class = 'single_qubit.dual_rail_single_shot'
    experiment_name = 'HistogramPrepulseDualRailExperiment'   


    ## tomography measurement pulses:
    measurement_pulse1 = ['I', 'X', 'Y']
    measurement_pulse2 = ['I', 'X', 'Y']

    stor_no=loaded['TwoDualRailTomographyExperiment']['dual_rail_storage_id']  # [1st dual rail qubit, 2nd dual rail qubit]

    for keys in loaded[experiment_name].keys():
        try:
            loaded[experiment_name][keys] = loaded['TwoDualRailTomographyExperiment'][keys]   # overwrite the single experiment file with new paramters
        except:
            pass

    # constructing the correct prepulse+measurement pulse sequence

    # vz2add = loaded['SingleDualRailTomographyExperiment']['vz']

    for index, qubit_gate in enumerate(measurement_pulse1):
        
        for index2, qubit_gate2 in enumerate(measurement_pulse2):
            # print('Tomography rotation:', qubit_gate)
            #print(type(loaded[experiment_name]['cavity_name']))
            loaded[experiment_name]['measurement_pulse_list'] = []
            
            man_idx=1

            mm_base = MM_base(cfg = yaml_cfg)
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][0]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][0]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]# measure]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            I_gate1 = creator.pulse.tolist()
            post_selection_pulse_str = [
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]# measure]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            I_gate2 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][0]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][0]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]# measure]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            I_gate3 = creator.pulse.tolist()
            post_selection_pulse_str = [
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]# measure]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            I_gate4 = creator.pulse.tolist()


            I1_gate = [I_gate1, I_gate2]
            I2_gate = [I_gate3, I_gate4]

            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][0]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][0]), 'hpi',0],
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0], 
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            X_gate1 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            X_gate2 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][0]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][0]), 'hpi',0],
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0], 
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            X_gate3 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            X_gate4 = creator.pulse.tolist()

            X1_gate = [X_gate1, X_gate2]
            X2_gate = [X_gate3, X_gate4]

            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][0]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][0]), 'hpi',0],
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',90], 
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            Y_gate1 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[0][1]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            Y_gate2 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][0]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][0]), 'hpi',0],
                            ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',90], 
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            Y_gate3 = creator.pulse.tolist()
            post_selection_pulse_str = [['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0], 
                                ['storage', 'M'+ str(man_idx) + '-S' + str(stor_no[1][1]), 'hpi',0],
                            ['qubit', 'ge', 'hpi',0], # Starting parity meas
                            ['qubit', 'ge', 'parity_M' + str(man_idx),0], 
                            ['qubit', 'ge', 'hpi',0]]
            creator = mm_base.get_prepulse_creator(post_selection_pulse_str)
            Y_gate4 = creator.pulse.tolist()

            Y1_gate = [Y_gate1, Y_gate2]
            Y2_gate = [Y_gate3, Y_gate4]

            gate_name_now = ''

            if qubit_gate == 'I':
                loaded[experiment_name]['measurement_pulse_list'] = I1_gate
                gate_name_now += 'I'
            elif qubit_gate == 'X':
                loaded[experiment_name]['measurement_pulse_list'] = X1_gate
                gate_name_now += 'X'
            else:
                loaded[experiment_name]['measurement_pulse_list'] = Y1_gate
                gate_name_now += 'Y'

            if qubit_gate2 == 'I':
                loaded[experiment_name]['measurement_pulse_list'] += I2_gate
                gate_name_now += 'I'
            elif qubit_gate2 == 'X':
                loaded[experiment_name]['measurement_pulse_list'] += X2_gate
                gate_name_now += 'X'
            else:
                loaded[experiment_name]['measurement_pulse_list'] += Y2_gate
                gate_name_now += 'Y'
            print('Running:', gate_name_now)


            run_exp = eval(f"meas.{experiment_class}.{experiment_name}(soccfg=soccfg, path=path, prefix=prefix, config_file=config_path)")
            run_exp.cfg.expt = eval(f"loaded['{experiment_name}']")
            run_exp.go(analyze=False, display=False, progress=False, save=True)