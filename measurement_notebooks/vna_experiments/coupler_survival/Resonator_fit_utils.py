import easy_lmfit #<--Clone from Andrew's GitHub and run installer
from easy_lmfit import *
import slab 
from slab import fitexp
import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
from h5py import File
import scipy
import scipy.integrate as integ 
from scipy.interpolate import interp1d
from scipy import constants
import glob
import pint
import re
import os
from pint import UnitRegistry, Context
ureg = UnitRegistry()
Q = ureg.Quantity
RE_UNIT = re.compile('([a-z]+)')
RE_VAL = re.compile('([0-9]+)')

def S21_hanger(x, Qi, Qc, f0):
    Q_tot=(1/Qi+1/Qc)
    return 10*np.log10(np.abs(1-(Q_tot*np.exp(1j*gam))/(Qc+2j*Q_tot*Qc*(x-f0)/f0))**2)

def real_imag_S21_hanger(x, Qi, Qc, f0, gam):
    Q_tot=(1/Qi+1/Qc)**-1
    g=1-(Q_tot*np.exp(1j*gam))/(Qc+2j*Q_tot*Qc*(x-f0)/f0)
    return np.imag(g), np.real(g)

def Q_func(x, Qi, Qc, f0, phi):
    mag=((1/Qi)-(np.exp(1j*phi)/Qc)+(2j*(x-f0)/f0))/((1/Qi)+(np.exp(1j*phi)/Qc)+(2j*(x-f0)/f0))
    return 10*np.log10(abs(mag)**2)

def Q_real_imag(x, Qi, Qc, f0, phi):
    mag=((1/Qi)-(np.exp(1j*phi)/Qc)+(2j*(x-f0)/f0))/((1/Qi)+(np.exp(1j*phi)/Qc)+(2j*(x-f0)/f0))
    return np.real(mag), np.imag(mag)

def new_S11_imag_real(x, ki, kc, f0, gam):
    mag = 1-(2*kc*np.exp(1j*gam))/(kc*np.exp(1j*gam)+ki+2*1j*(x-f0))
    return np.imag(mag), np.real(mag)


def IQ_fit(x, xi, ki, kc, f0, gam, a):
    I, Q=new_S11_imag_real(x, ki, kc, f0, gam)
    return a*I*(1-xi)+a*Q*(xi)

def IQ_hanger_fit(x, xi, Qi, Qc, f0, gam, a):
    I, Q = real_imag_S21_hanger(x, Qi, Qc, f0, gam)
    return a*I*(1-xi)+a*Q*(xi)

def mag_phase_IQ(db_mags, phase):
    # phase=phase-phase[0]
    phase_rad=np.pi*phase/180
    lin_mag=10.**((db_mags)/20.)
    Q=lin_mag*np.sin(phase_rad)
    I=lin_mag*np.cos(phase_rad)
    imag=np.imag(2*(I+1j*Q)/(np.mean(I[:2]+1j*Q[:2])+np.mean(I[-2:-1]+1j*Q[-2:-1])))
    real=np.real(2*(I+1j*Q)/(np.mean(I[:2]+1j*Q[:2])+np.mean(I[-2:-1]+1j*Q[-2:-1])))
    return real, imag

def S11_temp_sweep_unpack(cwd, params):
    data_files=[fname for fname in os.listdir(cwd) if fname.split('.')[0][-2::]=='mK']
    T_vals=[]
    fit_vals=[]
    for data in data_files:
        # print(data)
        T_vals.append(Q(data.split('_')[-1].split('.')[0]).to('K').magnitude)
        with File(cwd+data, 'r') as f:
            try:
                mags=np.array(f['mags'][0,:])
                freq=np.array(f['freq'][0,:])
                phase=np.array(f['phases'][0,:])
            except:
                mags=np.array(f['mags'])
                freq=np.array(f['readfreq_pts'][0,:])
                phase=np.array(f['phases'])
        try:
            params['freq']=freq
            params['mag']=mags
            params['phase']=phase
            f0=freq[np.argmin(mags)]
            params['param_override']['f0']=f0
            fit_vals.append(gen_S11_fit(params)) 
        except:
            pass
    zip_sorted=sorted(zip(T_vals, fit_vals), key=lambda x: x[0])
    T_vals, fit_vals=map(list, zip(*zip_sorted))
    return T_vals, fit_vals

def TLS_func(x, tan_del, Q_int_0, f, alpha, S_e):
    h_bar=constants.hbar
    kb=constants.k
    w=2*np.pi*f
    l=10E-9
    eps_r=33
    a=6.35E-3/2
    b=15.875E-3/2
    # S_e=1/(a*np.log(b/a))
    p_diel=l*S_e/eps_r
    return (1/Q_int_0+p_diel*tan_del*np.tanh(alpha*h_bar*w/(2*kb*x)))**-1

def TLS_Q(x, tan_del, f):
    h_bar=constants.hbar
    kb=constants.k
    w=2*np.pi*f
    l=5E-9
    eps_r=33
    a=6.35E-3/2
    b=15.875E-3/2
    S_e=1/(a*np.log(b/a))
    p_diel=l*S_e/eps_r
    return (p_diel*tan_del)**-1


def S11(x, ki, kc, f0, gam):
    kappa_1 = kc
    kappa_2 = 0 #kappa_c
    loss_rate = ki + kappa_2
    return 10 * np.log10(np.abs((1j * (x - f0) + (loss_rate - (kappa_1*np.exp(1j*gam))))/ (1j * (x - f0) +  (loss_rate + (kappa_1*np.exp(1j*gam))))) ** 2)

def S11_inv(x, ki, kc, f0, gam):
    kappa_1 = kc
    kappa_2 = 0 #kappa_c
    loss_rate = ki + kappa_2
    return (10 * np.log10(np.abs((1j * (x - f0) + (loss_rate - (kappa_1*np.exp(1j*gam))))/ (1j * (x - f0) +  (loss_rate + (kappa_1*np.exp(1j*gam))))) ** 2))**-1

def S11_imag_real(x, *p):
    kappa_i, kappa_c, f0, gam = p
    kappa_1 = kappa_c
    kappa_2 = 0 #kappa_c
    loss_rate = kappa_i + kappa_2
    g = (1j * (x - f0) + (loss_rate - (kappa_1*np.exp(1j*gam))))/ (1j * (x - f0) +  (loss_rate + (kappa_1*np.exp(1j*gam))))

    return np.imag(g), np.real(g)

def S21(x, ki, k1, k2, f0, gam1, gam2, A, phi):
        kappa_1 = k1+1j*gam1
        kappa_2 = k2+1j*gam2
        kappa_i=ki
        return np.abs((2 * np.sqrt(kappa_1 * kappa_2) / (x - f0 + 1j * (kappa_1 + kappa_2 + kappa_i)))+A*np.exp(-1j*phi*np.pi/180.)) ** 2

def S21_imag_real(x, ki, k1, k2, f0, gam1, gam2):
    kappa_1 = k1+1j*gam1
    kappa_2 = k2+1j*gam2
    kappa_i=ki
    g=(2.*np.sqrt(kappa_1 * kappa_2) / (x - f0 + 1j * (kappa_1 + kappa_2 + kappa_i)))
    return np.imag(g), np.real(g) 
    
def gen_S11_fit(params):
    
    non_opt_params=['mag', 'freq', 'p0', 'phase']
    opt_params_defaults={'plot':False, 'param_override':None, 'line_sub':False, 'param_domain':None, 'verbose':False, 'savgol':None, 'fit_range':None}
    
    miss_params=[]
    for no_params in non_opt_params:
        if no_params in params:
            pass
        else:
            miss_params.append(no_params)
    if len(miss_params)!=0:
        raise Exception('ERROR: Params missing: %s'%(', '.join(map(str, miss_params))))
    else:
        pass
    
    for keys in iter(opt_params_defaults.keys()):
        if keys in params:
            pass
        else:
            params[keys]=opt_params_defaults[keys]
            
    init_guess=params['p0']
    mags=params['mag']
    freq=params['freq']
    phase=params['phase']
    plot=params['plot']
    p_over=params['param_override']
    line_sub=params['line_sub']
    param_domain=params['param_domain']
    verbose=params['verbose']
    savgol=params['savgol']
    fit_range=params['fit_range']
    
    Qi_guess=init_guess['Qi_guess']
    Qc_guess=init_guess['Qc_guess']
    n_avg=10
    db_mags=mags-max([sum(mags[0:n_avg])/n_avg, sum(mags[-1:len(mags)-(n_avg+1):-1])/n_avg])
    #print('S11_offset in gen S11 fit',max([sum(mags[0:n_avg])/n_avg, sum(mags[-1:len(mags)-(n_avg+1):-1])/n_avg]))
    
    if line_sub==True:
        m,b=np.polyfit([freq[0], freq[-1]], [db_mags[0], db_mags[-1]], 1)
        line=m*freq+b
        db_mags=db_mags-line
    else:
        pass

    if savgol!=None:
        if type(savgol)==list:
            order=int(savgol[0])
            poly=int(savgol[1])
            if order<= poly:
                raise Exception('ERROR: Savgol order must be greater than polynomial, %d>%d'%(poly, order))
            elif (order%2)-1!=0:
                 raise Exception('ERROR: Savgol order must be odd, %d'%(order))
            else:
                pass
            fit_mags=scipy.signal.savgol_filter(db_mags, order, poly)
        else:
            raise Exception('ERROR: Savgol input is %d, must be list type [order, poly]'%type(savgol))
    else:
        fit_mags=db_mags
    

    phase=phase-phase[0]
    phase_rad=np.pi*phase/180
    lin_mag=10.**((db_mags)/20.)
    
    real=lin_mag*np.cos(phase_rad)
    imag=lin_mag*np.sin(phase_rad)
    d =max([abs(np.sort(real)[-1]-np.sort(real)[0]),abs(np.sort(imag)[-1]-np.sort(imag)[0])])
    kappa = 1.0/(2.0/d-1.0)
    
    ind_peak=np.argmin(fit_mags)
    
    f0_guess=freq[ind_peak]

    
    if fit_range!=None:
        fit_domain=[f0_guess-fit_range/2., f0_guess+fit_range/2.]
        df=fit_range
    else:
        fit_domain=None
        df=fit_range
    p0={}
    
    if Qi_guess==None and p_over!=None:
        if p_over['kc']!=None:
            ind_l=np.argmax(fit_mags[0:ind_peak:1]<=fit_mags[ind_peak]/2)
            ind_r=ind_peak+np.argmax(fit_mags[ind_peak::1]>=fit_mags[ind_peak]/2)
            kt_bestguess=(freq[ind_r]-freq[ind_l])/2
            if kt_bestguess>=p_over['kc']:
                p0['ki']=kt_bestguess-p_over['kc']
                p0['kc']=p_over['kc']
            else:
                p0['ki']=kt_bestguess
                p0['kc']=p_over['kc']
        else:
             raise Exception('ERROR: kc param_override is None Qi_guess is none, enter kc override')
    elif Qi_guess!=None:
        p0['ki']=f0_guess/Qi_guess
        p0['kc']=f0_guess/(Qi_guess/kappa)
    else:
        raise Exception('ERROR: Need Qi_guess')
        
    p0['f0']=f0_guess
    p0['gam']=0
    
    fits_dict, ferr=lm_curve_fit(S11, freq, fit_mags, p0, p_over=p_over, param_domain=param_domain, fit_domain=fit_domain, verbose=verbose, plot_fit=False)
    
    if fit_domain!=None:
        if type(fit_domain)==list:
            if len(fit_domain)==2:
                ind=np.searchsorted(freq, fit_domain)
                df=freq[ind[0]:ind[1]]
                db_mags=db_mags[ind[0]:ind[1]]
    else:
        df=freq
        db_mags=db_mags
    
    if plot==True:

        f0=freq[np.argmin(fit_mags)]
        fits=[]
        for key in iter(fits_dict.keys()):
            fits.append(fits_dict[key])
        fig1, (ax1, ax2)=plt.subplots(2,1, figsize=(6,10))
        ax1.plot(real,imag, '.r', label='data')

        imag_fit, real_fit=S11_imag_real(freq, *fits)
        ax1.plot(real_fit, imag_fit-imag_fit[0]/2, label="Fit")
        ax1.set_xlabel('Re(S11)')
        ax1.set_ylabel('Im(S11)')
        ax1.legend(frameon=False)
        ax1.set_title('Imag vs Real Plot Data and Fit')

        
        ax2.plot(df-f0, db_mags, '.b', label="Data")
        
        if savgol!=None:
            ax2.plot(df-f0, fit_mags, 'orange', linewidth=1, label="Savgol filter Ord=%d, Poly=%d"%(order, poly))
        else:
            pass
        
        ax2.plot(df-f0, S11(df, *fits), 'r', label="$Q_c$ = %.2e\n$Q_i$ = %.2e\n f0=%.5f GHz"%(fits[2]/(2*fits[1]), fits[2]/(2*fits[0]), fits[2]/1E9))
        
        ax2.legend(frameon=False)
        ax2.set_title('S11 Data and Fit')
        ax2.set_xlabel('$\delta f$ (Hz)')
        ax2.set_ylabel('dB')
    else: 
        pass
    fit_params={}
    for key in iter(fits_dict.keys()):
        fit_params[key]=fits_dict[key]
        fit_params[key+'_err']=ferr[key]
    
    return fit_params

def S11_val_sweep_fits(params):
    non_opt_params=['mag', 'freq', 'phase', 'sweep_val', 'val_name', 'Qi_guess']
    opt_params_defaults={'kc':None,'gam1':None,'plot':True, 'line_sub':False, 'Qi_domain':None, 'save_plot':False, 'savgol':None, 'fit_check':None}
    
    miss_params=[]
    for no_params in non_opt_params:
        if no_params in params:
            pass
        else:
            miss_params.append(no_params)
            
    if len(miss_params)!=0:
        raise Exception('ERROR: Params missing: %s'%(', '.join(map(str, miss_params))))
    else:
        pass
    
    for keys in iter(opt_params_defaults.keys()):
        if keys in params:
            pass
        else:
            params[keys]=opt_params_defaults[keys]
    
    mags_S11=params['mag']
    freq_S11=params['freq']
    phase_S11=params['phase']
    val=params['sweep_val']
    val_name=params['val_name']
    
    L=mags_S11.shape[0]
    Qi_guess=params['Qi_guess']
    kc=params['kc']
    gam1=params['gam1']
    err_array=np.zeros((L,4))
    fits_array=np.zeros((L,4))
    Qi_array=np.zeros(L)
    Qi_err=np.zeros(L)
    Qi_dom=params['Qi_domain']
    fit_check=params['fit_check']
    line_sub=params['line_sub']
    fit_range=params['fit_range']
    
    p_over={'kc':kc,
            'ki':None,
            'gam':gam1,
            'f0':None,
           }

    p0={'Qi_guess':Qi_guess,
        'Qc_guess':None
        }
    
    param_domain={'kc':[-np.inf, np.inf],
                  'ki':[0.0, np.inf],
                  'gam':[-np.inf, np.inf],
                  'f0':[-np.inf, np.inf]
                  }
    
    paramsf={'freq':None,
            'mag':None,
            'phase':None,
            'p0':p0,
            'param_override':p_over,
            'plot':False,
            'line_sub':False,
            'param_domain':param_domain,
            'fit_range':fit_range
           }
    
    if line_sub==True:
        paramsf['line_sub']=True
    else:
        pass
    
    ind=np.argsort(val)
    vals=np.take(val, ind)
    
    for I, sort in enumerate(ind):
        phase=phase_S11[sort]
        mags=mags_S11[sort]
        freq=freq_S11[sort]
        
        if fit_check!=None:
            if I==int(fit_check-1):
                print('Fit for value %s=%.3f'%(val_name, vals[I]))
                paramsf['plot']=True
                paramsf['verbose']=True
            else:
                paramsf['plot']=False
                paramsf['verbose']=False
        else:
            pass
        
        if Qi_dom!=None:
            f0_est=freq[np.argmin(mags)]
            ki_dom=f0_est/Qi_dom
            param_domain['ki'][0]=min(ki_dom)
            param_domain['ki'][1]=max(ki_dom)
            paramsf['param_domain']=param_domain
        else:
            pass
        
        if params['savgol']!=None:
            paramsf['savgol']=params['savgol']
        else:
            pass
        
        paramsf['phase']=phase
        paramsf['mag']=mags
        paramsf['freq']=freq
        
        fit_params=gen_S11_fit(paramsf)
        
        ki=fit_params['ki']
        kc=fit_params['kc']
        f0=fit_params['f0']
        
        err_array[I,:]=np.array([fit_params['ki_err'], fit_params['kc_err'], fit_params['f0_err'], fit_params['gam_err']])
        fits_array[I,:]=np.array([fit_params['ki'], fit_params['kc'], fit_params['f0'], fit_params['gam']])
        Qi_array[I]=f0/(2*ki)
        Qi_err[I]=Qi_err_fcn(fit_params)
        if fit_check!=None:
            if I==int(fit_check-1):
                print('Fit error for Qi=%.2f'%(Qi_err[I]))
        else:
            pass
    if params['plot']==True:
        fig1, (ax1, ax2)=plt.subplots(2,1, figsize=(10,10))

        ax1.semilogy(vals, 100.*Qi_err/Qi_array,'*', label="Qi error")
        ax1.legend()
        ax1.set_xlabel('%s'%val_name)
        ax1.set_ylabel('% Qi Error')
        ax1.set_title('Qi Error vs %s'%val_name)
        ax1.grid()
        
        ax2.semilogy(vals, Qi_array, '.-g')
        ax2.set_xlabel('%s'%val_name)
        ax2.set_ylabel('Qi')
        ax2.set_title('Measured Qi vs Input Power')
        ax2.grid()
        #ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #plt.errorbar(val, Qi_array, yerr=Qi_err, fmt='o', markersize=5, capsize=5, markeredgewidth=1)

    else:
        pass
    
    output={'Qi':Qi_array, 
            'Qi_err':Qi_err,
            'err':err_array, 
            'fits':fits_array, 
            'sweep_val':vals, 
            'val_name':val_name,
            'freq':freq_S11, 
            'mags_data':mags_S11, 
            'phase_data':phase,
            'Qi_guess':Qi_guess
           }
    return output

def Ql_err_fcn(fit_params):
    k1=2*fit_params['kc']
    k2=2*fit_params['ki']
    f0=fit_params['f0']
    k1_err=2*fit_params['kc_err']
    k2_err=2*fit_params['ki_err']
    f0_err=fit_params['f0_err']
    dqdk=-f0/(2*(k1+k2)**2)
    dqdf=1/(2*(k1+k2))
    err=np.sqrt((dqdk*k1_err**2)**2+(dqdk*k2_err**2)**2+(dqdf*f0_err)**2)
    return err

def Qi_err_fcn(fit_params):
    ki=2*fit_params['ki']
    f0=fit_params['f0']
    ki_err=2*fit_params['ki_err']
    f0_err=2*fit_params['f0_err']
    err=np.sqrt((((-f0/(2*ki**2))*ki_err)**2)+((1/(2*ki))*f0_err)**2)
    return err

def power_fits(freq, mags, phase, f_range, **kwargs):
    
    Qi_guess=3E9

    f0=6692368954.740292
    
    if f_range==True:
        f_over=f0
    else:
        f_over=None

    p_over={'kc':8.8,
            'ki':None,
            'gam':None,
            'f0':f_over
           }

    p0={'Qi_guess':Qi_guess,
        'Qc_guess':None, 
        }


    param_domain={'kc':[0, 13],
                'ki':[0, np.inf],
                'gam':[-np.inf, np.inf],
                'f0':[-np.inf, np.inf],
                    }
    df=None

    params={'freq':freq,
            'mag':mags,
            'phase':phase,
            'p0':p0,
            'param_override':p_over,
            'plot':False,
            'line_sub':False,
            'param_domain':param_domain,
            'fit_range':df,
            'verbose':False,
            'savgol':[7,5], 
            'smooth':False
           }
    S11_fit_params=gen_S11_fit(params)
    #print(S11_fit_params)
    #print(freq[np.argmin(mag_S11)])
    
    return S11_fit_params

def power_sweep_unpack(cwd, params):
    data_files=[fname for fname in os.listdir(cwd) if fname.split('.')[0][-3::]=='dBm']
    P_vals=[]
    fit_vals=[]
    for data in data_files:
        # print(data)
        P_vals.append(float(data.split('_')[-1].split('.')[0].split('dBm')[0]))
        with File(cwd+data, 'r') as f:
            try:
                mags=np.array(f['mags'][0,:])
                freq=np.array(f['freq'][0,:])
                phase=np.array(f['phases'][0,:])
            except:
                mags=np.array(f['mags'])
                freq=np.array(f['readfreq_pts'][0,:])
                phase=np.array(f['phases'])
        try:
            params['freq']=freq
            params['mag']=mags
            params['phase']=phase
            f0=freq[np.argmin(mags)]
            params['param_override']['f0']=f0
            fit_vals.append(gen_S11_fit(params)) 
        except:
            pass 
    zip_sorted=sorted(zip(P_vals, fit_vals), key=lambda x: x[0])
    P_vals, fit_vals=map(list, zip(*zip_sorted))
    return P_vals, fit_vals