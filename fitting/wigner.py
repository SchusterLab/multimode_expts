from fitting.fit_display_classes import GeneralFitting

import numpy as np
from scipy.optimize import minimize
from math import pi, exp
from numpy import ones, zeros, diag, sqrt, array, dot, transpose, conjugate, real, exp as np_exp
from numpy.linalg import pinv, eig, norm
from scipy.linalg import expm
from scipy.special import genlaguerre, factorial
import qutip
from numpy.linalg import svd
from IPython.display import clear_output, display
import os
from scipy.optimize import fmin, check_grad, minimize
from numpy import sqrt, linspace, cos, sin, real, arange
import matplotlib.pyplot as plt
import qutip as qt

class WignerAnalysis(GeneralFitting):
    def __init__(self, data=None, readout_per_round=None, threshold=None, config=None, alphas=None, mode_state_num=5, station=None):
        if data is not None:
            super().__init__(data, readout_per_round, threshold, config, station)
        else:
            self.station = station
        self.alphas = alphas
        self.m = mode_state_num
        if self.m is not None:
            self.I = np.diag(ones(self.m), 0)
            self.Par = np.real(np.diag(np_exp(1j * pi * (np.arange(0, self.m))), 0))
            self.init_states()

    # ---------------------- For Gain to Alpha Conversion ----------------------
    def get_gain_to_alpha(self, allocated_counts, initial_guess=[0.0001], bounds=[(0, 0.01)], plot_guess=False, plot=True, scale=False): 
        expec_value = 2/np.pi * allocated_counts
        gain_to_alpha, result  = self.fit_gain_to_alpha(
            xdata=self.data['xpts'], 
            ydata=expec_value, 
            W_vacuum=self.W_vacuum, 
            initial_guess=initial_guess, 
            bounds=bounds,
            scale=scale
        )
        print('Gain to Alpha Conversion Factor:', gain_to_alpha)

        # plot the initial guess 
        if plot_guess:
            self.plot_wigner_fit(
                xdata=self.data['xpts'],
                ydata=expec_value,
                gain_to_alpha=initial_guess[0],
                scale=initial_guess[1] if scale and len(initial_guess) >1 else 1.0,
            W_vacuum=self.W_vacuum,
            title='Initial Guess: Gain to Alpha Conversion Factor: {:.8f}'.format(initial_guess[0])
            )

        if plot:
            self.plot_wigner_fit(
                xdata=self.data['xpts'], 
                ydata=expec_value, 
                gain_to_alpha=gain_to_alpha, 
                scale=result.x[1] if scale and len(result.x) > 1 else 1.0,
                W_vacuum=self.W_vacuum, 
                title=f'Gain to Alpha Conversion Factor: {gain_to_alpha:.8f}'
            )
        print('alpha = 1 requires gain of : {:.8f}'.format(1/gain_to_alpha))
        return gain_to_alpha, result, expec_value

    def fit_gain_to_alpha(self, xdata, ydata, W_vacuum, initial_guess=[0.0002], bounds=[(0, 0.001)], scale=False):
        
        # if scale=true we add a scaling factor to the Wigner function

        def objective(param):
            gain_to_alpha = param[0]
            scale_factor = param[1] if scale else 1.0
            new_alpha_list = gain_to_alpha * xdata
            ws = scale_factor * W_vacuum(new_alpha_list)
            return np.sum(np.abs(ws - ydata))
        if scale and len(initial_guess) < 2:
            initial_guess = initial_guess + [1.0]
            if len(bounds) < 2:
                bounds = bounds + [(0.1, 10.0)]
        result = minimize(objective, initial_guess, bounds=bounds)
        gain_to_alpha = result.x[0]
        return gain_to_alpha, result

    def plot_wigner_fit(self, xdata, ydata, gain_to_alpha, W_vacuum, scale=1.0, title=None):
        import matplotlib.pyplot as plt
        new_alpha_list = gain_to_alpha * xdata
        fig = plt.figure(figsize=(6,4))
        plt.plot(new_alpha_list, scale * W_vacuum(new_alpha_list), '-', label='Fit')
        plt.plot(new_alpha_list, ydata, 'o', label='Data')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$W(\alpha)$')
        plt.legend()
        if title is None:
            title = f'Conversion factor: {gain_to_alpha:.8f}'
        plt.title(title)
        plt.grid(True)
        plt.show()
        # filename = title.replace(' ', '_').replace(':', '') + '.png'
        # self.save_plot(fig, filename=filename)

    def W_vacuum(self, alpha):
        '''
        W_alpha = (2/pi) * exp(-2 * alpha**2)
        '''
        W_vaccum = (2/np.pi)*np.exp(-2 * np.abs(alpha)**2)
        return W_vaccum

    def W_vaccum_for_fitting(self, alpha):
        """
        W_vaccum_for_fitting is a modified version of W_vacuum that normalizes the Wigner function
        to have a maximum value of 1 and a minimum value of 0.

        $W_alpha = 4/np.pi * e^{-|\alpha|^2} * sinh(|\alpha|^2)}$

        Somehow 4/pi disappears? Some normalization happening to data I guess
        """
        W_vaccum =  np.exp(-1 * np.abs(alpha)**2) * np.sinh(np.abs(alpha)**2)
        return W_vaccum

    # ---------------------- Wigner Tomography Methods ----------------------

    ## Utilities for Wigner Tomography
    def init_states(self):
        self.vac = qutip.basis(self.m, 0)
        return None

    def D_qutip(self, beta):
        """
        Returns the displacement operator for a given beta using qutip.
        """
        return qutip.displace(self.m, beta)

    def D(self, beta):
        """
        Returns the displacement operator for a given beta using numpy.
        """
        M_p = np.diag(np.sqrt(np.arange(1, 100)), 1)
        M_m = np.diag(np.sqrt(np.arange(1, 100)), -1)
        return (expm(beta * M_m - np.conj(beta) * M_p))[:self.m, :self.m]

    def W_op(self, alpha, analytic=True):
        """
        Returns the Wigner operator for a given alpha.
        """
        if analytic:
            w = np.zeros((self.m, self.m), dtype=np.complex128)
            B = 4 * abs(alpha) ** 2
            pf = (2 / np.pi) * np.exp(-B / 2.0)
            for m in range(self.m):
                x = pf * real((-1) ** m * genlaguerre(m, 0)(B))
                w[m, m] = x
                for n in range(m + 1, self.m):
                    pf_nm = sqrt(factorial(m) / float(factorial(n)))
                    x = pf * pf_nm * (-1) ** m * 2 * (2 * alpha) ** (n - m - 1) * genlaguerre(m, n - m)(B)
                    if m > 0:
                        y = 8 * pf * pf_nm * (-1) ** (m - 1) * (2 * alpha) ** (n - m) * genlaguerre(m - 1, n - m + 1)(B)
                    w[m, n] = alpha * x
                    w[n, m] = (alpha * x).conj()
        else:
            w = 2 / np.pi * dot(self.D(-alpha), dot(self.Par, self.D(alpha)))
        return w

    def W(self, alpha, w_vec):
        """
        Computes the Wigner function for a given alpha and w_vec 
        """
        return np.trace(dot(self.W_op(alpha, analytic=False), self.rho_pinv(w_vec)))

    def curlyM(self, analytic=True):
        if analytic:
            curlyM, gcurlyMx, gcurlyMy = self.wigner_mat_and_grad(self.alphas, self.m, reshape=True)
        else:
            M_mat = array([self.W_op(alpha, analytic=False) for alpha in self.alphas])
            curlyM = array([transpose(m).flatten() for m in M_mat])
        return curlyM

    def rho_pinv(self, w_vec):
        self.x = w_vec
        curlyM_inv = np.linalg.pinv(self.curlyM(analytic=False))
        rho_vec = dot(curlyM_inv, self.x)
        rho = rho_vec.reshape(self.m, self.m)
        return rho

    def rho_pinv2(self, w_vec):
        self.x = w_vec
        curlyM_inv2 = dot(pinv(dot((transpose(self.curlyM())), self.curlyM())), (transpose(self.curlyM())))
        rho_vec = dot(curlyM_inv2, self.x)
        rho = rho_vec.reshape(self.m, self.m)
        return rho
    
    ## Now we tryu to find the optimizal wigner displacement generation

    def wigner_mat_and_grad(self, disps, FD, reshape=True):
        
        ND = len(disps)
        # seb: for some reason I need to apply a phase shift to disps
        disps = np.abs(disps) * np.exp(1j * (np.angle(disps) + np.pi))
        wig_tens = np.zeros((ND, FD, FD), dtype=np.complex128)
        grad_mat_r = np.zeros((ND, FD, FD), dtype=np.complex128)
        grad_mat_i = np.zeros((ND, FD, FD), dtype=np.complex128)
        B = 4 * abs(disps) ** 2
        pf = (2 / np.pi) * np.exp(-B / 2.0)
        for m in range(FD):
            x = pf * np.real((-1) ** m * genlaguerre(m, 0)(B))
            term_r = -4 * disps.real * x
            term_i = -4 * disps.imag * x
            if m > 0:
                y = 8 * pf * (-1) ** (m - 1) * genlaguerre(m - 1, 1)(B)
                term_r += disps.real * y
                term_i += disps.imag * y
            wig_tens[:, m, m] = x
            grad_mat_r[:, m, m] = term_r
            grad_mat_i[:, m, m] = term_i
            for n in range(m + 1, FD):
                pf_nm = sqrt(factorial(m) / float(factorial(n)))
                x = pf * pf_nm * (-1) ** m * 2 * (2 * disps) ** (n - m - 1) * genlaguerre(m, n - m)(B)
                term_r = ((n - m) - 4 * disps.real * disps) * x
                term_i = (1j * (n - m) - 4 * disps.imag * disps) * x
                if m > 0:
                    y = 8 * pf * pf_nm * (-1) ** (m - 1) * (2 * disps) ** (n - m) * genlaguerre(m - 1, n - m + 1)(B)
                    term_r += disps.real * y
                    term_i += disps.imag * y
                wig_tens[:, m, n] = disps * x
                wig_tens[:, n, m] = (disps * x).conj()
                grad_mat_r[:, m, n] = term_r
                grad_mat_r[:, n, m] = term_r.conjugate()
                grad_mat_i[:, m, n] = term_i
                grad_mat_i[:, n, m] = term_i.conjugate()
        if reshape:
            return (wig_tens.reshape((ND, FD ** 2)), grad_mat_r.reshape((ND, FD ** 2)), grad_mat_i.reshape((ND, FD ** 2)))
        else:
            return (wig_tens, grad_mat_r, grad_mat_i)

    

    def extracted_W(self, rho, alphaxs, alphays):
        return array([[np.trace(dot(self.W_op(alphax + 1j * alphay, analytic=True), rho))
                       for alphax in alphaxs] for alphay in alphays])
    

    # def extracted_W_fast(rho, alphaxs, alphays, wigner_obj):
    #     def calc(alpha):
    #         return np.trace(dot(wigner_obj.W_op(alpha, analytic=True), rho))

    #     alphas = [x + 1j * y for y in alphays for x in alphaxs]
    #     W_vals = Parallel(n_jobs=-1)(delayed(calc)(a) for a in alphas)
    #     return np.array(W_vals).reshape(len(alphays), len(alphaxs))

    def extracted_W_single(self, rho, alpha):
        return np.trace(dot(self.W_op(alpha, analytic=False), rho))

    def extracted_W_single_analytic(self, rho, disps, FD):
        W, gWx, gWy = self.wigner_mat_and_grad(disps, FD, reshape=False)
        return array([np.trace(dot(w, rho)) for w in W])

    def rho_pinv_trace_1(self, w_vec):
        self.x = w_vec
        i_vec = self.I.flatten()
        A = zeros([self.m ** 2 + 1, self.m ** 2 + 1], dtype=np.complex128)
        A[:self.m ** 2, :self.m ** 2] = dot(conjugate((transpose(self.curlyM()))), self.curlyM())
        A[self.m ** 2, self.m ** 2], A[self.m ** 2][:self.m ** 2], A.T[self.m ** 2][:self.m ** 2] = 0, i_vec, i_vec
        vec_b = ones([self.m ** 2 + 1], dtype=np.complex128)
        vec_b[:self.m ** 2] = dot(conjugate(transpose(self.curlyM())), w_vec)
        rho_vec = dot(pinv(A), vec_b)[:self.m ** 2]
        rho = rho_vec.reshape(self.m, self.m)
        # for some reason I need to conjugate the result
        return (rho)

    def rho_pinv_positive_sd(self, w_vec):
        rho = self.rho_pinv_trace_1(w_vec)
        l, v = eig(rho)
        ls_index = sorted(range(len(l)), key=lambda k: l[k], reverse=True)
        ls = sorted(l, reverse=True)
        vs = array([v.T.conj()[i] for i in ls_index])
        lps = self.positive_semidefinite_eigs(ls)
        rho_new = dot((array(vs).T.conj()), dot(diag(lps, 0), array(vs)))
        return (rho_new)
        # return (rho)

    def positive_semidefinite_eigs(self, vs):
        a = 0
        d = len(vs)
        vout = array(zeros(d))
        vout = vs
        for i in sorted(range(d), reverse=True):
            if (vs[i] + a / (i + 1) < 0):
                a = a + vs[i]
                vout[i] = 0
            else:
                break
        if i != d - 1:
            vout[:i + 1] = vs[:i + 1] + a / (i + 1)
        return (vout)
    
    ## Final Plotting and analysis methods
    
    def wigner_analysis_results(self, allocated_counts, initial_state, rotate=False):
        # print the keys in the data dictionary
        alpha_list = self.data['alpha']
        allocated_readout = 2 / np.pi * allocated_counts  # normalization
        initial_state = initial_state.unit()
        # accept a pure ket OR an already-mixed density matrix as the target
        rho_ideal = (initial_state if initial_state.isoper
                     else qutip.ket2dm(initial_state)).unit()
        alphas2 = np.arange(-np.sqrt(self.m) / np.sqrt(1) + 0.1, np.sqrt(self.m) / np.sqrt(1), 0.1)
        rho = self.rho_pinv_positive_sd(allocated_readout)
        P_ns = [np.array([rho[i][i] for i in range(self.m)])]


        # rotate: False/None -> no rotation; True/'optimal' -> maximize fidelity
        # over a cavity rotation exp(-i theta N); a numeric value -> apply that
        # fixed angle (e.g. a calibrated channel phase phi_ch).
        N = np.diag(np.arange(self.m))
        if rotate is True or (isinstance(rotate, str) and rotate.lower() == 'optimal'):
            theta = np.linspace(0, 2 * np.pi, 361)
            fid_vec = np.array([
                qutip.fidelity(
                    qutip.Qobj(expm(-1j * t * N) @ rho @ expm(1j * t * N)),
                    rho_ideal) ** 2
                for t in theta])
            theta_max = float(theta[np.argmax(fid_vec)])
        elif rotate is False or rotate is None:
            theta_max = 0.0
        else:
            theta_max = float(rotate)   # user-provided fixed angle

        R = expm(-1j * theta_max * N)
        rho_rotated = R @ rho @ R.conj().T
        fid = qutip.fidelity(qutip.Qobj(rho_rotated), rho_ideal) ** 2

        alpha_max = np.max(np.abs(alpha_list))
        x_vec = np.linspace(-alpha_max, alpha_max, 200)
        W_fit = qt.wigner(qt.Qobj(rho_rotated), x_vec, x_vec, g=2)  # rotation-aligned state
        W_ideal = qt.wigner(rho_ideal, x_vec, x_vec, g=2)


        return {
            'alpha_list': alpha_list,
            'allocated_readout': allocated_readout,
            'rho': rho,
            'rho_rotated': rho_rotated,
            'rho_ideal': rho_ideal,
            'P_ns': P_ns,
            'alphas2': alphas2,
            'fidelity': fid,
            'theta_max': theta_max,
            'wigner_analysis': self,
            'W_fit': W_fit,
            'W_ideal': W_ideal,
            'x_vec': x_vec,
        }

    def plot_wigner_reconstruction_results(self, results, initial_state=None, state_label = None):
        
        alpha_list = results['alpha_list']
        allocated_readout = results['allocated_readout']
        rho = results['rho']
        rho_ideal = results['rho_ideal']
        alphas2 = results['alphas2']
        w = results['wigner_analysis']
        mode_state_num = rho.shape[0]
        fidelity = results.get('fidelity', None)

        x_vec = results['x_vec']
        # why do I need to reverse the x_vec?
        W_fit = results['W_fit']
        W_ideal = results['W_ideal']
        vmin = -2 / np.pi
        vmax = 2 / np.pi
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].set_aspect('equal')
        ax[0, 1].set_aspect('equal')

        cs = ax[0,0].tricontourf(
            -np.real(alpha_list), -np.imag(alpha_list), allocated_readout,
            np.linspace(vmin, vmax, 30), cmap='RdBu_r', vmin=vmin, vmax=vmax
        )
        ax[0,0].plot(-np.real(alpha_list), -np.imag(alpha_list), 'g. ')
        cbar1 = fig.colorbar(cs, ax=ax[0, 0],  fraction=0.045, pad=0.04)
        # cbar1.set_label('')
        ticks = np.linspace(vmin, vmax, 5)
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels([r'$-2/\pi$', r'$-1/\pi$', '0', r'$1/\pi$', r'$2/\pi$'])
        ax[0, 0].set_title('Direct (reverse order)')

        c = ax[0, 1].pcolormesh(np.arange(mode_state_num), np.arange(mode_state_num), np.abs(rho), cmap='viridis', vmin=0, vmax=1)
        ax[0, 1].set_ylabel('n')
        ax[0, 1].set_xlabel('m')
        # colorbar for the pcolormesh
        cbar2 = fig.colorbar(ax[0, 1].collections[0], ax=ax[0, 1], fraction=0.045, pad=0.04)
        cbar2.set_label(r'$|\rho_{nm}|$')
        cbar2.set_ticks([0, 0.5, 1])
        cbar2.set_ticklabels(['0', '1/2', '1'])
        ax[0, 1].set_title('Density matrix fock basis')

        ax[1, 0].set_aspect('equal')
        ax[1, 0].pcolormesh(x_vec, x_vec, W_fit, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[1, 0].set_xlabel(r'Re($\alpha$)')
        ax[1, 0].set_ylabel(r'Im($\alpha$)')
        ax[1, 0].set_title('Wigner function fit')
        # colorbar for the Wigner function
        cbar3 = fig.colorbar(ax[1, 0].collections[0], ax=ax[1, 0], fraction=0.045, pad=0.04)
        cbar3.set_label('')
        cbar3.set_ticks(ticks)
        cbar3.set_ticklabels([r'$-2/\pi$', r'$-1/\pi$', '0', r'$1/\pi$', r'$2/\pi$'])

        if initial_state is not None:

            ax[1, 1].set_aspect('equal')
            ax[1, 1].pcolormesh(x_vec, x_vec, W_ideal, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax[1, 1].set_xlabel(r'Re($\alpha$)')
            ax[1, 1].set_ylabel(r'Im($\alpha$)')
            ax[1, 1].set_title('Wigner function ideal state')
            # colorbar for the Wigner function
            cbar4 = fig.colorbar(ax[1, 1].collections[0], ax=ax[1, 1], fraction=0.045, pad=0.04)
            cbar4.set_label('')
            cbar4.set_ticks(ticks)
            cbar4.set_ticklabels([r'$-2/\pi$', r'$-1/\pi$', '0', r'$1/\pi$', r'$2/\pi$'])

        fig.suptitle(f'Wigner Tomography Results\nFidelity: {fidelity:.4f}', fontsize=16)
        fig.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
        fig.tight_layout()

        return fig
    

class OptimalDisplacementGeneration():
    """
    Pick a set of displacements {alpha_i} that make Wigner-tomography
    reconstruction well-posed. The reconstruction inverts M rho_vec = w_vec where
    M is the (n_disps x FD^2) Wigner measurement matrix (row i = W_mn(alpha_i)).

    objective:
      'a-optimal' (default) -- min Tr((M^H M)^{-1}), the mean variance of the
                              reconstructed rho_mn under uniform measurement
                              noise. Right criterion for parity tomography.
      'd-optimal'           -- min -log det(M^H M); maximizes Fisher information
                              per shot. Robust to outlier rows.
      'condition'           -- min kappa(M) = sigma_max/sigma_min. Original
                              behavior. Only
                              cares about worst-case noise amplification; tends
                              to push points to the edge of the disk.
    """
    def __init__(self, FD=3, n_disps=None, objective='a-optimal'):
        # super().__init__()
        self.FD = FD
        if n_disps is None:
            self.n_disps = (FD**2 + 30) * 2
        else:
            self.n_disps = n_disps
        if objective not in ('a-optimal', 'd-optimal', 'condition'):
            raise ValueError(f"objective must be one of 'a-optimal','d-optimal','condition'; got {objective!r}")
        self.objective = objective
        self.best_cost = float('inf')
        self.f, self.ax = None, None

    def wigner_mat_and_grad(self, disps, FD):
        ND = len(disps)
        wig_tens = np.zeros((ND, FD, FD), dtype=np.complex128)
        grad_mat_r = np.zeros((ND, FD, FD), dtype=np.complex128)
        grad_mat_i = np.zeros((ND, FD, FD), dtype=np.complex128)

        B = 4 * np.abs(disps)**2
        pf = (2 / np.pi) * np.exp(-B/2)
        for m in range(FD):
            x = pf * np.real((-1) ** m * genlaguerre(m, 0)(B))
            term_r = -4 * disps.real * x
            term_i = -4 * disps.imag * x

            if m > 0:
                y = 8 * pf * (-1)**(m-1) * genlaguerre(m-1, 1)(B)
                term_r += disps.real * y
                term_i += disps.imag * y
            wig_tens[:, m, m] = x
            grad_mat_r[:, m, m] = term_r
            grad_mat_i[:, m, m] = term_i

            for n in range(m+1, FD):
                pf_nm = sqrt(factorial(m)/float(factorial(n)))
                x = pf * pf_nm * (-1)**m * 2 * (2*disps)**(n-m-1) * genlaguerre(m, n-m)(B)
                term_r = ((n-m)-4*disps.real*disps) * x
                term_i = (1j* (n-m)-4*disps.imag*disps) * x
                if m>0:
                    y = 8*pf*pf_nm*(-1)**(m-1)*(2*disps)**(n-m) * genlaguerre(m-1,n-m+1)(B)
                    term_r += disps.real * y
                    term_i += disps.imag * y
                wig_tens[:, m, n] = disps * x
                wig_tens[:, n, m] = np.conj(disps * x)
                grad_mat_r[:, m, n] = term_r
                grad_mat_r[:, n, m] = np.conj(term_r)
                grad_mat_i[:, m, n] = term_i
                grad_mat_i[:, n, m] = np.conj(term_i)

        return wig_tens.reshape((ND, FD**2)), grad_mat_r.reshape((ND, FD**2)), grad_mat_i.reshape((ND, FD**2))
    
    

    # def cost_and_grad(self, r_disps):
    #     N = len(r_disps)
    #     c_disps = r_disps[:N//2] + 1j*r_disps[N//2:]
    #     M, dM_rs, dM_is = self.wigner_mat_and_grad(c_disps, self.FD)

    #     if self.objective == 'condition':
    #         U, S, Vd = svd(M)
    #         NS = len(Vd)
    #         cn = S[0] / S[-1]
    #         dS_r = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_rs, Vd.conj().T).real
    #         dS_i = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_is, Vd.conj().T).real
    #         grad_cn_r = (dS_r[0]*S[-1]-S[0]*dS_r[-1]) / (S[-1]**2)
    #         grad_cn_i = (dS_i[0]*S[-1]-S[0]*dS_i[-1]) / (S[-1]**2)
    #         return cn, np.concatenate((grad_cn_r, grad_cn_i))

    #     # A-optimal / D-optimal share the same gradient form:
    #     #   d(cost)/dx_i = -2 Re( dM_rs[i,:] @ B @ M[i,:].conj() )
    #     # where B = A_inv^2 for A-optimal and B = A_inv for D-optimal,
    #     # with A = M^H M. Derived from d(A_inv) = -A_inv*dA*A_inv and
    #     # d(log det A) = Tr(A_inv*dA), then using the fact that row i of M
    #     # is the only row that depends on (x_i, y_i).
    #     A = M.conj().T @ M
    #     A_inv = np.linalg.inv(A)
    #     if self.objective == 'a-optimal':
    #         cost = float(np.real(np.trace(A_inv)))
    #         B = A_inv @ A_inv
    #     else:  # 'd-optimal'
    #         sign, logdet = np.linalg.slogdet(A)
    #         cost = -float(logdet)
    #         B = A_inv

    #     Z = B @ M.conj().T                                   # (FD^2, ND)
    #     grad_r = -2.0 * np.real(np.einsum('ik,ki->i', dM_rs, Z))
    #     grad_i = -2.0 * np.real(np.einsum('ik,ki->i', dM_is, Z))
    #     return cost, np.concatenate((grad_r, grad_i))

    def cost_and_grad(self, r_disps):
        N = len(r_disps)
        c_disps = r_disps[:N//2] + 1j*r_disps[N//2:]
        M, dM_rs, dM_is = self.wigner_mat_and_grad(c_disps, self.FD)

        # ==========================================
        # 1. COMPUTE BASE COST & GRADIENTS
        # ==========================================
        if self.objective == 'condition':
            U, S, Vd = svd(M)
            NS = len(Vd)
            cn = S[0] / S[-1]
            dS_r = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_rs, Vd.conj().T).real
            dS_i = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_is, Vd.conj().T).real
            
            cost = cn
            grad_r = (dS_r[0]*S[-1]-S[0]*dS_r[-1]) / (S[-1]**2)
            grad_i = (dS_i[0]*S[-1]-S[0]*dS_i[-1]) / (S[-1]**2)

        else:
            A = M.conj().T @ M
            A_inv = np.linalg.inv(A)
            
            if self.objective == 'a-optimal':
                cost = float(np.real(np.trace(A_inv)))
                B = A_inv @ A_inv
            else:  # 'd-optimal'
                sign, logdet = np.linalg.slogdet(A)
                cost = -float(logdet)
                B = A_inv

            Z = B @ M.conj().T                                   # (FD^2, ND)
            grad_r = -2.0 * np.real(np.einsum('ik,ki->i', dM_rs, Z))
            grad_i = -2.0 * np.real(np.einsum('ik,ki->i', dM_is, Z))

        # ==========================================
        # 2. APPLY "SOFT WALL" RADIAL PENALTY
        # ==========================================
        # Define the circular boundary (adjust the 1.5 factor if needed)
        max_radius = 1.5 * np.sqrt(self.FD)
        radii_sq = c_disps.real**2 + c_disps.imag**2
        
        # Find which specific points escaped the circle
        penalty_mask = radii_sq > max_radius**2
        
        if np.any(penalty_mask):
            # The penalty weight needs to be strong enough to overpower the objective
            # For A-optimal, the base cost might be small, so 1000.0 is usually dominant.
            lambda_pen = 1000.0  
            
            diff = radii_sq[penalty_mask] - max_radius**2
            
            # Add a quadratic penalty to the objective cost
            cost += lambda_pen * np.sum(diff**2)
            
            # Add the exact analytical derivative of the penalty to the gradients
            grad_r[penalty_mask] += 4 * lambda_pen * diff * c_disps.real[penalty_mask]
            grad_i[penalty_mask] += 4 * lambda_pen * diff * c_disps.imag[penalty_mask]

        return cost, np.concatenate((grad_r, grad_i))

    def wrap_cost(self, disps):
        import matplotlib.pyplot as plt
        cost, grad = self.cost_and_grad(disps)
        self.best_cost = min(cost, self.best_cost)
        if self.f is None or self.ax is None:
            self.f, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.clear()
        self.ax.plot(disps[:self.n_disps], disps[self.n_disps:], 'k.')
        label = {'condition': 'kappa', 'a-optimal': 'Tr(A^-1)',
                 'd-optimal': '-log det A'}[self.objective]
        self.ax.set_title('%s [%s] = %.3g' % (label, self.objective, cost))
        clear_output(wait=True)
        display(self.f)
        return cost, grad

    def optimize(self, save_dir=None):
        import matplotlib.pyplot as plt
        self.f, self.ax = plt.subplots(figsize=(5, 5))
        low, high = -np.sqrt(self.FD), np.sqrt(self.FD)
        # init_disps = np.random.normal(0, scale=2, size=(self.n_disps, 2))
        # init_disps = np.clip(init_disps, low, high)
        from scipy.stats import truncnorm

        low, high = -1.5*np.sqrt(self.FD), 1.5*np.sqrt(self.FD)
        # low, high = -1*np.sqrt(self.FD), 1*np.sqrt(self.FD)

        # parameters for truncnorm: (a, b) are in std units
        a, b = (low - 0) / 2, (high - 0) / 2

        # sample
        samples = truncnorm.rvs(a, b, loc=0, scale=1, size=(self.n_disps, 2))
        samples[0] = [0.0, 0.0] 
        init_disps = samples
        print(f'objective = {self.objective}, FD = {self.FD}, n_disps = {self.n_disps}')
        x0 = np.concatenate([init_disps[:, 0], init_disps[:, 1]])
        bounds = [(low, high)] * len(x0)
        ret = minimize(self.wrap_cost, x0, method='L-BFGS-B', jac=True, options=dict(ftol=1E-6),)
                        # bounds=bounds)
        new_disps = ret.x[:self.n_disps] + 1j*ret.x[self.n_disps:]
        new_disps = np.concatenate(([0], new_disps))
        save_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            base_name = "optimized_displacements"
            ext = ".npy"
            save_path = os.path.join(save_dir, base_name + ext)
            count = 1
            while os.path.exists(save_path):
                save_path = os.path.join(save_dir, f"{base_name}_{count}{ext}")
                count += 1
            np.save(save_path, new_disps)
            print(f"Displacements saved to {save_path}")
        return {"disps": new_disps, "path": save_path}

    # def optimize(self, save_dir=None):
    #     import matplotlib.pyplot as plt
    #     from scipy.stats import truncnorm
        
    #     self.f, self.ax = plt.subplots(figsize=(5, 5))
        
    #     # 1. Set physical boundaries to prevent vanishing gradients
    #     max_alpha = 1.5 * np.sqrt(self.FD)
    #     low, high = -max_alpha, max_alpha
        
    #     a, b = (low - 0) / 2, (high - 0) / 2
    #     samples = truncnorm.rvs(a, b, loc=0, scale=1, size=(self.n_disps, 2))
        
    #     # 2. Pin the first point exactly to the origin before optimizing
    #     samples[0] = [0.0, 0.0] 
        
    #     x0 = np.concatenate([samples[:, 0], samples[:, 1]])
        
    #     # 3. Apply bounds to the solver
    #     bounds = [(low, high)] * len(x0)
        
    #     print(f'objective = {self.objective}, FD = {self.FD}, n_disps = {self.n_disps}')
        
    #     ret = minimize(self.wrap_cost, x0, method='L-BFGS-B', jac=True, 
    #                     bounds=bounds, options=dict(ftol=1E-6))
                        
    #     new_disps = ret.x[:self.n_disps] + 1j*ret.x[self.n_disps:]
        
    #     # (Optional) You no longer need to manually prepend [0] here, 
    #     # as it was part of the optimization block.