import casadi as cs
import numpy as np
from dataclasses import dataclass, field
from envs.quad_rotor_env import QuadRotorEnv
from mpc.generic_mpc import GenericMPC
from typing import Union
from util.casadi import quad_form


@dataclass(frozen=True)
class QuadRotorMPCConfig:
    '''
    Quadrotor MPC configuration, such as horizons and CasADi/IPOPT options.
    '''
    # horizon
    N: int = 15

    # solver options
    solver_opts: dict = field(default_factory=lambda: {
        'expand': True, 'print_time': False,
        'ipopt': {
            'max_iter': 500,
            'tol': 1e-6,
            'barrier_tol_factor': 1,
            'sb': 'yes',
            # debug
            'print_level': 0,
            'print_user_options': 'no',
            'print_options_documentation': 'no'
        }})


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

        # 1) create variables - states are softly constrained
        nx, nu, ns = env.nx, env.nu, not_red_idx.size + env.nu
        x, _, _ = self.add_var('x', nx, N)
        u, _, _ = self.add_var('u', nu, N)
        s, _, _ = self.add_var('slack', ns, N, lb=0)

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
        A, B, e = env.get_dynamics(
            T=env.config.T,
            g=self.pars['g'],
            thrust_coeff=self.pars['thrust_coeff'],
            pitch_d=self.pars['pitch_d'],
            pitch_dd=self.pars['pitch_dd'],
            pitch_gain=self.pars['pitch_gain'],
            roll_d=self.pars['roll_d'],
            roll_dd=self.pars['roll_dd'],
            roll_gain=self.pars['roll_gain']
        )
        self.add_con('dyn', x_[:, 1:], '==', A @ x_[:, :-1] + B @ u + e)

        # 3) constraint on state (soft, backed off, without infinity in g, and
        # removing redundant entries)
        # constraint backoff parameter and bounds
        bo = self.add_par('backoff', 1, 1)

        # set the state constraints as
        #  - soft-backedoff minimum constraint: (1+back)*lb - slack <= x
        #  - soft-backedoff maximum constraint: x <= (1-back)*ub + slack
        s_x = s[:-env.nu, :]
        self.add_con('x_min', (1 + bo) * lbx - s_x, '<=', x[not_red_idx, :])
        self.add_con('x_max', x[not_red_idx, :], '<=', (1 - bo) * ubx + s_x)

        # 4) constraint on input (soft)
        s_u = s[-env.nu:, :]
        self.add_con('u_min', env.config.u_bounds[:, 0] - s_u, '<=', u)
        self.add_con('u_max', u, '<=', env.config.u_bounds[:, 1] + s_u)

        # ========= #
        # Objective #
        # ========= #

        # 1) initial cost
        J = 0  # (no initial state cost not required since it is not economic)

        # 2) stage cost
        xf = self.add_par('xf', nx, 1)
        uf = cs.vertcat(0, 0, self.pars['g'])
        w_x = self.add_par('w_x', nx, 1)    # weights for stage/final state
        w_u = self.add_par('w_u', nu, 1)    # weights for stage/final control
        w_s = self.add_par('w_s', ns, 1)    # weights for stage/final slack
        J += sum((
            quad_form(w_x, x[:, k] - xf) +
            quad_form(w_u, u[:, k] - uf) +
            cs.dot(w_s, s[:, k])) for k in range(N - 1))

        # 3) terminal cost
        J += quad_form(w_x, x[:, -1] - xf) + \
            quad_form(w_u, u[:, -1] - uf) + \
            cs.dot(w_s, s[:, -1])

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
