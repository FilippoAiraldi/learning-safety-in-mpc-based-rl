import numpy as np
import casadi as cs
from dataclasses import dataclass, field
from util import quad_form
from envs.quad_rotor_env import QuadRotorEnv
from mpc.generic_mpc import GenericMPC


@dataclass(frozen=True)
class QuadRotorMPCConfig:
    '''
    Quadrotor MPC configuration, such as horizons and CasADi/IPOPT options.
    '''
    # horizons
    Np: int = 20
    Nc: int = None

    # solver options
    solver_opts: dict = field(default_factory=lambda: {
        'expand': True, 'print_time': False,
        'ipopt': {
            'tol': 1e-6,
            'barrier_tol_factor': 1,
            # debug
            'print_level': 0,
            'print_user_options': 'no',
            'print_options_documentation': 'no'
        }})

    def __post_init__(self) -> None:
        # overwrite Nc if None
        if self.Nc is None:
            self.__dict__['Nc'] = self.Np


class QuadRotorMPC(GenericMPC):
    '''An MPC controller specifically designed for the quadrotor dynamics.'''

    def __init__(
        self,
        env: QuadRotorEnv,
        config: dict | QuadRotorMPCConfig = None,
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
        Np, Nc = config.Np, config.Nc

        # ======================= #
        # Variable and Parameters #
        # ======================= #

        # within x bounds, get which are redundant (lb=-inf, ub=+inf) and which
        # are not. Create slacks only for non-redundant constraints on x.
        lb, ub = env.config.x_bounds[:, 0], env.config.x_bounds[:, 1]
        not_red = ~(np.isneginf(lb) & np.isposinf(ub))
        not_red_idx = np.where(not_red)[0]
        lb, ub = lb[not_red].reshape(-1, 1), ub[not_red].reshape(-1, 1)

        # 1) create variables - states are softly constrained
        nx, nu, ns = env.nx, env.nu, not_red_idx.size
        x, _, _ = self.add_var('x', nx, Np + 1)
        u, _, _ = self.add_var('u', nu, Nc,
                               lb=env.config.u_bounds[:, 0, None],
                               ub=env.config.u_bounds[:, 1, None])
        slack, _, _ = self.add_var('slack', ns, Np, lb=0)

        # 2) create model parameters
        for name in ('g', 'thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
                     'roll_d', 'roll_dd', 'roll_gain'):
            self.add_par(name, 1, 1)

        # =========== #
        # Constraints #
        # =========== #

        # 1) constraint on initial conditions
        x0 = self.add_par('x0', env.nx, 1)  # initial conditions
        self.add_con('init_state', x[:, 0], '==', x0)

        # 2) constraints on dynamics
        u_exp = cs.horzcat(u, cs.repmat(u[:, -1], 1, Np - Nc))
        A, B, e = self._get_dynamics_matrices(env)
        self.add_con('dyn', x[:, 1:], '==', A @ x[:, :-1] + B @ u_exp + e)        

        # 3) constraint on state (soft, backed off, without infinity in g, and
        # removing redundant entries)
        # constraint backoff parameter and bounds
        backoff = self.add_par('backoff', 1, 1)

        # set the state constraints as
        #  - soft-backedoff minimum constraint: (1+back)*lb - slack <= x
        #  - soft-backedoff maximum constraint: x <= (1-back)*ub + slack
        self.add_con('state_min', 
                     (1 + backoff) * lb - slack, '<=', x[not_red_idx, 1:])
        self.add_con('state_max', 
                     x[not_red_idx, 1:], '<=', (1 - backoff) * ub + slack)

        # ========= #
        # Objective #
        # ========= #

        # 1) initial cost
        J = 0  # (no initial state cost not required since it is not economic)

        # 2) stage cost
        xf = self.add_par('xf', nx, 1)
        w_L = self.add_par('w_L', nx, 1)  # weights for stage
        w_s = self.add_par('w_s', ns, 1)  # weights for slack
        J += sum(quad_form(w_L, x[:, k] - xf) + cs.dot(w_s, slack[:, k - 1])
                 for k in range(1, Np))

        # 3) terminal cost
        w_V = self.add_par('w_V', nx, 1)  # weights for final
        w_s_f = self.add_par('w_s_f', ns, 1)  # weights for final slack
        J += quad_form(w_V, x[:, Np] - xf) + cs.dot(w_s_f, slack[:, Np - 1])

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

    def _get_dynamics_matrices(self, env: QuadRotorEnv):
        T = env.config.T  # NOTE: T is here fixed
        pars = self.pars
        Ad = cs.diag(cs.vertcat(pars['pitch_d'], pars['roll_d']))
        Add = cs.diag(cs.vertcat(pars['pitch_dd'], pars['roll_dd']))
        A = T * cs.vertcat(
            cs.horzcat(np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))),
            cs.horzcat(
                np.zeros((2, 6)), np.eye(2) * pars['g'], np.zeros((2, 2))),
            np.zeros((1, 10)),
            cs.horzcat(np.zeros((2, 6)), -Ad, np.eye(2)),
            cs.horzcat(np.zeros((2, 6)), -Add, np.zeros((2, 2)))
        ) + np.eye(10)
        B = T * cs.vertcat(
            np.zeros((5, 3)),
            cs.horzcat(0, 0, pars['thrust_coeff']),
            np.zeros((2, 3)),
            cs.horzcat(pars['pitch_gain'], 0, 0),
            cs.horzcat(0, pars['roll_gain'], 0)
        )
        e = cs.vertcat(
            np.zeros((5, 1)),
            - T * pars['g'],
            np.zeros((4, 1))
        )
        return A, B, e
