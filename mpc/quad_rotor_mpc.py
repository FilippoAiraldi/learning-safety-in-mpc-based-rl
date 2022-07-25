from envs.quad_rotor_env import QuadRotorPars, QuadRotorEnv
from mpc.generic_mpc import GenericMPC
import numpy as np
import casadi as cs
from util import quad_form


class QuadRotorMPC(GenericMPC):
    '''An MPC controller specifically designed for the quadrotor dynamics.'''

    def __init__(self, Np: int, Nc: int = None, type: str = 'V') -> None:
        assert type in {'V', 'Q'}, \
            'MPC must be either V (state value func) or Q (action value func)'
        super().__init__(name=type)
        Nc = Np if Nc is None else Nc

        # create variables - states are softly constrained
        nx, nu = QuadRotorEnv.nx, QuadRotorEnv.nu
        x, _, _ = self.add_var('x', nx, Np + 1)
        u, _, _ = self.add_var('u', nu, Nc,
                               lb=QuadRotorEnv.u_bounds[:, 0, None],
                               ub=QuadRotorEnv.u_bounds[:, 1, None])
        slack, _, _ = self.add_var('slack', nx, Np, lb=0)

        # create model parameters
        for name in ('thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
                     'roll_d', 'roll_dd', 'roll_gain'):
            self.add_par(name, 1, 1)

        # constraint on initial conditions
        x0 = self.add_par('x0', QuadRotorEnv.nx, 1)  # initial conditions
        self.add_con('init_state', x[:, 0] - x0, 0, 0)

        # constraints on dynamics
        u_exp = cs.horzcat(u, cs.repmat(u[:, -1], 1, Np - Nc))
        A, B, e = self._get_dynamics_matrices()
        for k in range(Np):
            self.add_con(f'dyn_{k}',
                         x[:, k + 1] - (A @ x[:, k] + B @ u_exp[:, k] + e),
                         0, 0)

        # constraint on state (soft)
        backoff = self.add_par('backoff', 1, 1)  # constraint backoff parameter
        m = QuadRotorEnv.x_bounds[:, 0, None]
        M = QuadRotorEnv.x_bounds[:, 1, None]
        for k in range(1, Np + 1):
            # soft-backedoff minimum constraint: (1+back)*m - slack <= x
            self.add_con(f'state_min_{k}',
                         x[:, k] + slack[:, k - 1] - backoff * m, m, 0)
            # soft-backedoff maximum constraint: x <= (1-back)*M + slack
            self.add_con(f'state_max_{k}',
                         x[:, k] - slack[:, k - 1] + backoff * M, 0, M)

        # initial cost
        J = 0  # (no initial state cost not required since it is not economic)

        # stage cost
        xf = self.add_par('xf', nx, 1)
        w_L = self.add_par('w_L', nx, 1)  # weights for stage
        w_s = self.add_par('w_s', nx, 1)  # weights for slack
        J += sum(quad_form(w_L, x[:, k] - xf) + cs.dot(w_s, slack[:, k - 1]) 
                 for k in range(1, Np))

        # terminal cost
        w_V = self.add_par('w_V', nx, 1)  # weights for final
        w_s_f = self.add_par('w_s_f', nx, 1)  # weights for final slack
        J += quad_form(w_V, x[:, Np] - xf) + cs.dot(w_s_f, slack[:, Np - 1])

        # assign cost
        self.minimize(J)

        # case-specific modifications
        if type == 'Q':
            u0 = self.add_par('u0', nu, 1)
            self.add_con('init_action', u[:, 0] - u0, 0, 0)
        else:
            perturbation = self.add_par('perturbation', nu, 1)
            self.f += cs.dot(perturbation, u[:, 0])

    def _get_dynamics_matrices(self):
        T, g = QuadRotorPars.T, QuadRotorPars.g  # fixed
        Ad = cs.diag(cs.vertcat(self.pars['pitch_d'], self.pars['roll_d']))
        Add = cs.diag(cs.vertcat(self.pars['pitch_dd'], self.pars['roll_dd']))
        A = T * cs.vertcat(
            cs.horzcat(np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))),
            cs.horzcat(np.zeros((2, 6)), np.eye(2) * g,
                       np.zeros((2, 2))),
            np.zeros((1, 10)),
            cs.horzcat(np.zeros((2, 6)), -Ad, np.eye(2)),
            cs.horzcat(np.zeros((2, 6)), -Add, np.zeros((2, 2)))
        ) + np.eye(10)
        B = T * cs.vertcat(
            np.zeros((5, 3)),
            cs.horzcat(0, 0, self.pars['thrust_coeff']),
            np.zeros((2, 3)),
            cs.horzcat(self.pars['pitch_gain'], 0, 0),
            cs.horzcat(0, self.pars['roll_gain'], 0)
        )
        e = cs.vertcat(
            np.zeros((5, 1)),
            - T * g,
            np.zeros((4, 1))
        )
        return A, B, e
