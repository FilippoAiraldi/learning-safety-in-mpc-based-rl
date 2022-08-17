import casadi as cs
import numpy as np
from dataclasses import dataclass, field
from envs.quad_rotor_env import QuadRotorEnv
from mpc.generic_mpc import GenericMPC, Solution
from typing import Union
from util import quad_form


@dataclass(frozen=True)
class QuadRotorMPCConfig:
    '''
    Quadrotor MPC configuration, such as horizons and CasADi/IPOPT options.
    '''
    # horizons
    N: int = 20

    # solver options
    solver_opts: dict = field(default_factory=lambda: {
        'expand': True, 'print_time': False,
        'ipopt': {
            'tol': 1e-6,
            'barrier_tol_factor': 1,
            'sb': 'yes',
            # debug
            'print_level': 0,
            'print_user_options': 'no',
            'print_options_documentation': 'no'
        }})

    # NLP scaling
    # The scaling operation is x_scaled = Tx * x, and yields a scaled state
    # whose elements lay in comparable ranges
    scaling_x: np.ndarray = field(default_factory=lambda: np.array(
        [1e0, 1e0, 1e0, 1e1, 1e1, 1e1, 1e-1, 1e-1, 1e0, 1e0]))
    scaling_u: np.ndarray = field(default_factory=lambda: np.array(
        [1e0, 1e0, 1e1]))

    @property
    def Tx(self) -> np.ndarray:
        return np.diag(1 / 1 / self.scaling_x)

    @property
    def Tu(self) -> np.ndarray:
        return np.diag(1 / 1 / self.scaling_u)

    @property
    def Tx_inv(self) -> np.ndarray:
        return np.diag(self.scaling_x)

    @property
    def Tu_inv(self) -> np.ndarray:
        return np.diag(self.scaling_u)


class QuadRotorMPC(GenericMPC):
    '''An MPC controller specifically designed for the quadrotor dynamics.'''

    def __init__(
        self,
        env: QuadRotorEnv,
        config: Union[dict, QuadRotorMPCConfig] = None,
        type: str = 'V'
    ) -> None:
        '''
        Instantiates an MPC for the quad rotor env.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the MPC.
        config : dict, QuadRotorMPCConfig
            A set of configuration parameters for the MPC. If not given, the 
            default ones are used.
        type : 'Q' or 'V'
            Type of MPC to instantiate, either state value function or action 
            value function.
        '''
        assert type in {'V', 'Q'}, \
            'MPC must be either V (state value func) or Q (action value func)'
        super().__init__(name=type)
        if config is None:
            config = QuadRotorMPCConfig()
        elif isinstance(config, dict):
            keys = QuadRotorMPCConfig.__dataclass_fields__.keys()
            config = QuadRotorMPCConfig(
                **{k: config[k] for k in keys if k in config})
        self.config = config
        N = config.N

        # ======================= #
        # Variable and Parameters #
        # ======================= #

        # within x bounds, get which are redundant (lb=-inf, ub=+inf) and which
        # are not. Create slacks only for non-redundant constraints on x.
        lbx, ubx = env.config.x_bounds[:, 0], env.config.x_bounds[:, 1]
        not_red = ~(np.isneginf(lbx) & np.isposinf(ubx))
        not_red_idx = np.where(not_red)[0]
        lbx, ubx = lbx[not_red].reshape(-1, 1), ubx[not_red].reshape(-1, 1)

        # u bounds must be scaled before creating the variable
        lbu = config.Tu @ env.config.u_bounds[:, 0, None]
        ubu = config.Tu @ env.config.u_bounds[:, 1, None]

        # 1) create variables - states are softly constrained
        nx, nu, ns = env.nx, env.nu, not_red_idx.size
        x, _, _ = self.add_var('x', nx, N)
        u, _, _ = self.add_var('u', nu, N, lb=lbu, ub=ubu)
        slack, _, _ = self.add_var('slack', ns, N, lb=0)

        # scale the variables
        x = config.Tx_inv @ x
        u = config.Tu_inv @ u

        # 2) create model parameters
        for name in ('g', 'thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
                     'roll_d', 'roll_dd', 'roll_gain'):
            self.add_par(name, 1, 1)

        # =========== #
        # Constraints #
        # =========== #

        # 1) constraint on initial conditions
        x0 = self.add_par('x0', env.nx, 1)
        x_ = cs.horzcat(x0, x)

        # 2) constraints on dynamics
        A, B, e = self._get_dynamics_matrices(env)
        norm = cs.sum2(cs.fabs(cs.horzcat(A, B, e))) + 1 # cs.mmax
        self.add_con('dyn', x_[:, 1:] / norm,
                     '==', (A @ x_[:, :-1] + B @ u + e) / norm)

        # 3) constraint on state (soft, backed off, without infinity in g, and
        # removing redundant entries)
        # constraint backoff parameter and bounds
        backoff = self.add_par('backoff', 1, 1)

        # set the state constraints as
        #  - soft-backedoff minimum constraint: (1+back)*lb - slack <= x
        #  - soft-backedoff maximum constraint: x <= (1-back)*ub + slack
        norm = cs.sum2(cs.fabs((1 + backoff) * lbx)) + 1
        self.add_con('state_min', ((1 + backoff) * lbx - slack) / norm,
                     '<=', x[not_red_idx, :] / norm)
        norm = cs.sum2(cs.fabs((1 - backoff) * ubx)) + 1
        self.add_con('state_max', x[not_red_idx, :] / norm,
                     '<=', ((1 - backoff) * ubx + slack) / norm)

        # ========= #
        # Objective #
        # ========= #

        # 1) initial cost
        J = 0  # (no initial state cost not required since it is not economic)

        # 2) stage cost
        xf = self.add_par('xf', nx, 1)
        uf = cs.vertcat(0, 0, self.pars['g'])
        w_Lx = self.add_par('w_Lx', nx, 1)    # weights for stage state
        w_Lu = self.add_par('w_Lu', nu, 1)    # weights for stage control
        w_Ls = self.add_par('w_Ls', ns, 1)    # weights for stage slack
        J += sum((
            quad_form(w_Lx, x[:, k] - xf) +
            quad_form(w_Lu, u[:, k] - uf) +
            cs.dot(w_Ls, slack[:, k])) for k in range(N - 1))

        # 3) terminal cost
        w_Tx = self.add_par('w_Tx', nx, 1)  # weights for final state
        w_Tu = self.add_par('w_Tu', nu, 1)  # weights for final control
        w_Ts = self.add_par('w_Ts', ns, 1)  # weights for final slack
        J += quad_form(w_Tx, x[:, -1] - xf) + \
            quad_form(w_Tu, u[:, -1] - uf) + \
            cs.dot(w_Ts, slack[:, -1])

        # assign cost
        self.minimize(J)

        # ====== #
        # Others #
        # ====== #

        # case-specific modifications
        if type == 'Q':
            u0 = self.add_par('u0', nu, 1)
            self.add_con('init_action', u[:, 0], '==', u0)
        else:
            perturbation = self.add_par('perturbation', nu, 1)
            self.f += cs.dot(perturbation, u[:, 0])

        # initialize solver
        self.init_solver(config.solver_opts)

    def _get_dynamics_matrices(
        self, env: QuadRotorEnv
    ) -> tuple[cs.SX, cs.SX, cs.SX]:
        T = env.config.T  # NOTE: T is here fixed
        pars = self.pars
        Ad = cs.diag(cs.vertcat(pars['pitch_d'], pars['roll_d']))
        Add = cs.diag(cs.vertcat(pars['pitch_dd'], pars['roll_dd']))
        A = T * cs.blockcat([
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))],
            [np.zeros((2, 6)), np.eye(2) * pars['g'], np.zeros((2, 2))],
            [np.zeros((1, 10))],
            [np.zeros((2, 6)), -Ad, np.eye(2)],
            [np.zeros((2, 6)), -Add, np.zeros((2, 2))]
        ]) + np.eye(10)
        B = T * cs.blockcat([
            [np.zeros((5, 3))],
            [0, 0, pars['thrust_coeff']],
            [np.zeros((2, 3))],
            [pars['pitch_gain'], 0, 0],
            [0, pars['roll_gain'], 0]
        ])
        e = cs.vertcat(
            np.zeros((5, 1)),
            - T * pars['g'],
            np.zeros((4, 1))
        )
        return A, B, e

    def solve(
        self, pars: dict[str, np.ndarray], vals0: dict[str, np.ndarray] = None
    ) -> Solution:
        sol = super().solve(pars, vals0)
        # add unscaled variables and values to solution
        sol.vals['x_unscaled'] = self.config.Tx_inv @ sol.vals['x']
        sol.vars['x_unscaled'] = self.config.Tx_inv @ sol.vars['x']
        sol.vals['u_unscaled'] = self.config.Tu_inv @ sol.vals['u']
        sol.vars['u_unscaled'] = self.config.Tu_inv @ sol.vars['u']
        return sol
