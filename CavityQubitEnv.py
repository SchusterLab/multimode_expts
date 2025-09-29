import numpy as np
import gymnasium as gym
from gymnasium import spaces
import qutip as qt
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class CavityQubitEnv(gym.Env):
    """
    Environment for residual learning of a pi-pulse.
    The agent modifies L control points of I and Q over a total pulse duration.
    Reward is the final-state fidelity.
    """

    def __init__(self, L=10,
                 pulse_time=20.0,
                 amplitude_bounds=[0.3, 0.3],  # bounds on residual amplitude
                 system_params=None,
                 psi_target=None, 
                 psi_0=None, 
                 pulse_IQ =None,
                 nb_sample = None,
                 N_points = None
                 ):
        super().__init__()
        self.n_times = 200  # number of time steps for simulation

        if system_params is None:
            Kq = -0.2
            Kc = -1e-6
            chi = -300e-6
            cutoff_qubit = 2
            cutoff_cavity = 10
            self.system_params = {
                "Kq": Kq,
                "Kc": Kc,
                "chi": chi,
                "cutoff_qubit": cutoff_qubit,
                "cutoff_cavity": cutoff_cavity
            }
        else:
            self.system_params = system_params


        if psi_0 is not None:
            self.psi_0 = psi_0
        else:
            self.psi_0 = qt.fock(self.N, 0)
        if psi_target is not None:
            self.psi_target = psi_target
        else:
            # self.psi_target = (qt.fock(self.N, 1) + qt.fock(self.N, 0)).unit()
            self.psi_target = (qt.fock(self.N, 0)).unit()

        self.fid_list = np.array([])

        if pulse_IQ is not None:
            self.Iq = pulse_IQ['Iq']
            self.Qq = pulse_IQ['Qq']
            self.Ic = pulse_IQ['Ic']
            self.Qc = pulse_IQ['Qc']
            self.times = pulse_IQ['times']
            self.pulse_time = self.times[-1]
            if N_points is None:
                self.L = len(self.Ic)-2
            else: 
                self.L = N_points-2
                times = np.linspace(0, self.pulse_time, self.L+2)
                idx = [np.argmin(np.abs(self.times - t)) for t in times]
                self.Iq = self.Iq[idx]
                self.Qq = self.Qq[idx]
                self.Ic = self.Ic[idx]
                self.Qc = self.Qc[idx]
                self.times = times

        else:
            self.pulse_time = pulse_time
            self.L = L
            self.times = np.linspace(0, self.pulse_time, self.L+2)
            self.Ic = np.zeros(self.L+2)
            self.Qc = np.zeros(self.L+2)
            self.Iq = np.zeros(self.L+2)
            self.Qq = np.zeros(self.L+2)

        # create the H0 and H_drives 

        Nq = self.system_params["cutoff_qubit"]
        Nc = self.system_params["cutoff_cavity"]
        Kq = self.system_params["Kq"]
        Kc = self.system_params["Kc"]
        chi = self.system_params["chi"]

        q = qt.tensor(qt.destroy(Nq), qt.qeye(Nc))
        c = qt.tensor(qt.qeye(Nq), qt.destroy(Nc))
        self.Xq = (q + q.dag())*2*np.pi
        self.Pq = -1j*(q - q.dag())*2*np.pi
        self.Xc = (c + c.dag())*2*np.pi
        self.Pc = -1j*(c - c.dag())*2*np.pi
        H0 = Kq/2*(q.dag()**2)*(q**2)
        H0 += Kc/2*(c.dag()**2)*(c**2)
        H0 += chi*(q.dag()*q)*(c.dag()*c)
        H0 *= 2*np.pi
        self.H0 = H0
        
        self.wigner_sampling = False 
        if nb_sample is not None: 
            self.wigner_sampling = True 
            self.nb_sample = nb_sample

        self.N = self.system_params["cutoff_qubit"]*self.system_params["cutoff_cavity"]

        self.amplitude_bounds = amplitude_bounds  # Example bound on pulse amplitude
        # Action space: residuals on I/Q control points, bounded between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4*self.L,), dtype=np.float32)
        # Observation space: trivial (final reward only) or could include pulse info
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Allow psi_0 and psi_target to be passed dynamically
        if options is not None:
            self.psi_0 = options.get("psi_0", self.psi_0)
            self.psi_target = options.get("psi_target", self.psi_target)

        return np.array([0.0], dtype=np.float32), {}
    
    def generate_IQ_pulse(self, action):
        # Split action into residuals for I and Q
        residual_Iq = action[:self.L]
        residual_Qq = action[self.L:2*self.L]
        residual_Ic = action[2*self.L:3*self.L]
        residual_Qc = action[3*self.L:]
        self.last_action = action  # store for rendering

        Iq = np.zeros(self.L+2)
        Qq = np.zeros(self.L+2)
        Ic = np.zeros(self.L+2)
        Qc = np.zeros(self.L+2)

        # Compute actual pulse
        Iq[1:-1] = self.Iq[1:-1] + residual_Iq * self.amplitude_bounds[0]
        Qq[1:-1] = self.Qq[1:-1] + residual_Qq * self.amplitude_bounds[0]
        Ic[1:-1] = self.Ic[1:-1] + residual_Ic * self.amplitude_bounds[1]
        Qc[1:-1] = self.Qc[1:-1] + residual_Qc * self.amplitude_bounds[1]
        

        pulse_IQ = {
            'Iq': Iq,
            'Qq': Qq,
            'Ic': Ic,
            'Qc': Qc,
            'times': self.times
        }
        return pulse_IQ
    
    def interpolate_IQ_pulse(self, pulse_IQ):

        cs_Iq = CubicSpline(pulse_IQ['times'], pulse_IQ['Iq'])
        cs_Qq = CubicSpline(pulse_IQ['times'], pulse_IQ['Qq'])
        cs_Ic = CubicSpline(pulse_IQ['times'], pulse_IQ['Ic'])
        cs_Qc = CubicSpline(pulse_IQ['times'], pulse_IQ['Qc'])

        t_fine = np.linspace(0, self.pulse_time, self.n_times)
        Iq_fine = cs_Iq(t_fine)
        Qq_fine = cs_Qq(t_fine)
        Ic_fine = cs_Ic(t_fine)
        Qc_fine = cs_Qc(t_fine)

        pulse_IQ_fine = {
            'Iq': Iq_fine,
            'Qq': Qq_fine,
            'Ic': Ic_fine,
            'Qc': Qc_fine,
            'times': t_fine
        }
        return pulse_IQ_fine

    def step(self, action):
        # Split action into residuals for I and Q
        self.pulse_IQ = self.generate_IQ_pulse(action)
        # Run physics simulation
        reward, _ = self._simulate(self.pulse_IQ)

        # One-step episode
        terminated = True
        truncated = False
        info = {}
        obs = np.zeros(1, dtype=np.float32)
        return obs, reward, terminated, truncated, info
    
    def generate_alpha_distribution(self, psi, n_points, plot=False):
        """
        Given the psi we generate a set of alpha base on the state vector psi. 
        The distribution is given by the absolute value of the wigner function. 
        Since the wigner function is not trivial the points are generated using sample rejection.   
        """

        rho_cav = psi.ptrace(1)  # trace out the qubit
        alpha_max = np.sqrt(self.system_params["cutoff_cavity"])
        n_grid = 500
        x_vec = np.linspace(-alpha_max, alpha_max, n_grid)
        W_cav = qt.wigner(rho_cav, x_vec, x_vec)*2
        P_cav = np.abs(W_cav)
        P_cav /= np.max(P_cav)  # Normalize the Wigner function
        self.P_norm = np.sum(np.abs(W_cav))*(2*alpha_max/n_grid)**2

        alpha_points = np.zeros(n_points, dtype=np.complex128)
        i = 0
        while i < n_points:
            x = np.random.uniform(-alpha_max, alpha_max)
            y = np.random.uniform(-alpha_max, alpha_max)
            idx_x = int((x + alpha_max) / (2 * alpha_max) * (n_grid-1))
            idx_y = int((y + alpha_max) / (2 * alpha_max) * (n_grid-1))
            if np.random.uniform(0, 1) < P_cav[idx_x, idx_y]:
                alpha_points[i] = x + 1j * y
                i += 1

        # print('carefull I am manually setting alpha_points')
        # alpha_points = np.linspace(-alpha_max, alpha_max, 10)

        if plot:
            fig, ax = plt.subplots()
            vmin = -2/np.pi
            vmax = 2/np.pi
            # vmin = np.min(W_cav)
            # vmax = np.max(W_cav)
            ax.pcolormesh(x_vec, x_vec, W_cav, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.scatter(alpha_points.real, alpha_points.imag, color='black', s=10)
            ax.set_xlabel("Re(α)")
            ax.set_ylabel("Im(α)")
            cbar3 = fig.colorbar(ax.collections[0], ax=ax, fraction=0.045, pad=0.04)
            cbar3.set_label('')
            ticks = np.linspace(vmin, vmax, 5)
            cbar3.set_ticks(ticks)
            ax.set_aspect('equal')
            cbar3.set_ticklabels(['-2/π', '-1/π', '0', '1/π', '2/π'])
            ax.set_title("Sampled α points on Wigner function")
            fig.tight_layout()

        return alpha_points
        

    def _simulate(self, pulse_IQ):
        """
        Interpolate control points to fine time grid and run QuTiP simulation.
        """
        # L = len(pulse_real)
        t_fine = np.linspace(0, self.pulse_time, self.n_times)
        self.t_fine = t_fine


        cs_Iq = CubicSpline(pulse_IQ['times'], pulse_IQ['Iq']) #Iq
        cs_Qq = CubicSpline(pulse_IQ['times'], pulse_IQ['Qq']) #Qq
        cs_Ic = CubicSpline(pulse_IQ['times'], pulse_IQ['Ic']) #Ic
        cs_Qc = CubicSpline(pulse_IQ['times'], pulse_IQ['Qc']) #Qc


        # Define callable for Hamiltonian
        pulse_Iq = lambda t: float(cs_Iq(t))
        pulse_Qq = lambda t: float(cs_Qq(t)) 
        pulse_Ic = lambda t: float(cs_Ic(t))
        pulse_Qc = lambda t: float(cs_Qc(t)) 

        H = [self.H0, [self.Xq, pulse_Iq], [self.Pq, pulse_Qq], [self.Xc, pulse_Ic], [self.Pc, pulse_Qc]]

        psi0 = self.psi_0
        target = self.psi_target
        result = qt.sesolve(H, psi0, tlist=t_fine)
        final_state = result.states[-1]

        # Fidelity as reward
        if self.wigner_sampling:
            reward = 0

            self.alpha_sample = self.generate_alpha_distribution(
                psi=self.psi_target, n_points=self.nb_sample, plot=False
            )
            rho_cav = self.psi_target.ptrace(1)
            self.W_vec = np.array([qt.wigner(rho_cav, alpha.real, alpha.imag, g=2)[0][0] for alpha in self.alpha_sample])
            W_vec = self.W_vec
            rho_final = final_state.ptrace(1)
            meas = np.array([qt.wigner(rho_final, alpha.real, alpha.imag, g=2)[0][0] for alpha in self.alpha_sample])/2*np.pi
            P_vec = np.abs(W_vec)
            P_vec /= self.P_norm
            # print('alpha_sample', self.alpha_sample)
            # print('P_vec', P_vec)
            # print('P_sum', np.sum(P_vec))
            # print('W_vec', W_vec/2*np.pi)
            # print("meas", meas)


            sgn_W = W_vec/P_vec  # sign of Wigner function
            reward_list = meas*sgn_W
            reward = np.mean(reward_list)*2
            fid = qt.fidelity(final_state, target)**2
            self.fid_list = np.append(self.fid_list, [fid])
        else: 
            fid = qt.fidelity(final_state, target)**2  # Scale to [-1, 1]
            # reward = np.log(fid)
            reward = fid
            self.fid_list = np.append(self.fid_list, [fid])

        return reward, result.states

    def render(self):
        t_points_full = self.times
        t_fine = np.linspace(0, self.pulse_time, self.n_times)

        if hasattr(self, 'last_action'):

            residual_Iq = self.last_action[:self.L]
            residual_Qq = self.last_action[self.L:2*self.L]
            residual_Ic = self.last_action[2*self.L:3*self.L]
            residual_Qc = self.last_action[3*self.L:]
            Iq = np.zeros(self.L+2)
            Qq = np.zeros(self.L+2)
            Ic = np.zeros(self.L+2)
            Qc = np.zeros(self.L+2)

            # Compute actual pulse
            Iq[1:-1] = self.Iq[1:-1] + residual_Iq * self.amplitude_bounds[0]
            Qq[1:-1] = self.Qq[1:-1] + residual_Qq * self.amplitude_bounds[0]
            Ic[1:-1] = self.Ic[1:-1] + residual_Ic * self.amplitude_bounds[1]
            Qc[1:-1] = self.Qc[1:-1] + residual_Qc * self.amplitude_bounds[1]


        else:
            Iq = self.Iq
            Qq = self.Qq
            Ic = self.Ic
            Qc = self.Qc

        pulse_IQ = {
            'Iq': Iq,
            'Qq': Qq,
            'Ic': Ic,
            'Qc': Qc,
            'times': self.times
        }


        reward, result_states = self._simulate(pulse_IQ)
        Fid_vec =[qt.fidelity(state, self.psi_target)**2 for state in result_states]
        state_table = np.zeros((self.N, self.n_times))
        for n in range(self.N):
            state_table[n, :] = [qt.fidelity(state, qt.fock(self.N, n))**2 for state in result_states]

        
        
        cs_Iq = CubicSpline(t_points_full, Iq)
        cs_Qq = CubicSpline(t_points_full, Qq)
        cs_Ic = CubicSpline(t_points_full, Ic)
        cs_Qc = CubicSpline(t_points_full, Qc)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4))
        ax1.plot(t_fine, cs_Iq(t_fine), label="I_q")
        ax1.plot(t_fine, cs_Qq(t_fine), label="Q_q")
        ax1.plot(t_fine, cs_Ic(t_fine), label="I_c")
        ax1.plot(t_fine, cs_Qc(t_fine), label="Q_c")
        ax1.scatter(t_points_full, Iq, color='tab:blue')
        ax1.scatter(t_points_full, Qq, color='tab:orange')
        ax1.scatter(t_points_full, Ic, color='tab:green')
        ax1.scatter(t_points_full, Qc, color='tab:red')
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        ax2.plot(np.linspace(0, self.pulse_time, self.n_times), Fid_vec)
        # for n in range(self.N):
        #     ax2.plot(np.linspace(0, self.pulse_time, self.n_times), state_table[n, :], label=f"|{n}>")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Fidelity")
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
