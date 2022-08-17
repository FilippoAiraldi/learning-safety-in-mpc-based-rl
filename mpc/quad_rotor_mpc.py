import casadi as cs
import cvxopt as cvx
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
    Np: int = 20
    Nc: int = None

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
        return np.diag(1 / self.scaling_x)

    @property
    def Tu(self) -> np.ndarray:
        return np.diag(1 / self.scaling_u)

    @property
    def Tx_inv(self) -> np.ndarray:
        return np.diag(self.scaling_x)

    @property
    def Tu_inv(self) -> np.ndarray:
        return np.diag(self.scaling_u)

    def __post_init__(self) -> None:
        # overwrite Nc if None
        if self.Nc is None:
            self.__dict__['Nc'] = self.Np


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
        Np, Nc = config.Np, config.Nc

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
        x, _, _ = self.add_var('x', nx, Np)
        u, _, _ = self.add_var('u', nu, Nc, lb=lbu, ub=ubu)
        slack, _, _ = self.add_var('slack', ns, Np, lb=0)

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
        x_exp = cs.horzcat(x0, x)

        # 2) constraints on dynamics
        u_exp = cs.horzcat(u, cs.repmat(u[:, -1], 1, Np - Nc))
        A, B, e = self._get_dynamics_matrices(env)
        self.add_con('dyn',
                     x_exp[:, 1:], '==', A @ x_exp[:, :-1] + B @ u_exp + e)

        # 3) constraint on state (soft, backed off, without infinity in g, and
        # removing redundant entries)
        # constraint backoff parameter and bounds
        backoff = self.add_par('backoff', 1, 1)

        # set the state constraints as
        #  - soft-backedoff minimum constraint: (1+back)*lb - slack <= x
        #  - soft-backedoff maximum constraint: x <= (1-back)*ub + slack
        self.add_con('state_min',
                     (1 + backoff) * lbx - slack, '<=', x[not_red_idx, :])
        self.add_con('state_max',
                     x[not_red_idx, :], '<=', (1 - backoff) * ubx + slack)

        # ========= #
        # Objective #
        # ========= #

        # 1) initial cost
        J = 0  # (no initial state cost not required since it is not economic)

        # 2) stage cost
        xf = self.add_par('xf', nx, 1)
        gamma = self.add_par('gamma', 1, 1)  # discount factor
        w_L = self.add_par('w_L', nx, 1)    # weights for stage
        w_s = self.add_par('w_s', ns, 1)    # weights for slack
        J += sum(gamma ** (k + 1) *
                 (quad_form(w_L, x[:, k] - xf) + cs.dot(w_s, slack[:, k]))
                 for k in range(Np - 1))

        # 3) terminal cost
        w_V = self.add_par('w_V', nx, 1)  # weights for final
        w_s_f = self.add_par('w_s_f', ns, 1)  # weights for final slack
        J += quad_form(w_V, x[:, -1] - xf) + cs.dot(w_s_f, slack[:, -1])

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

        #######################################################################

        # create cost matrices
        x_size, u_size, s_size = x_exp.numel(), u_exp.numel(), slack.numel()
        P = 2 * cs.diag(cs.vertcat(
            np.zeros(nx),
            *[gamma ** (k + 1) * w_L for k in range(Np - 1)],
            w_V,
            np.zeros(u_size),
            np.zeros(s_size)
        ))
        q = cs.vertcat(
            np.zeros(nx),
            *[gamma ** (k + 1) * (-2 * w_L * xf) for k in range(Np - 1)],
            -2 * w_V * xf,
            np.zeros(u_size),
            *[gamma ** (k + 1) * w_s for k in range(Np - 1)],
            w_s_f
        )
        self.cost_offset = sum(gamma ** (k + 1) * (xf.T @ cs.diag(w_L) @ xf)
                               for k in range(Np - 1)) + xf.T @ cs.diag(w_V) @ xf

        # create constraint matrices
        # dynamics
        Aeq = cs.blockcat([
            [np.eye(nx), np.zeros((nx, Np * (nx + nu)))],
            [
                cs.horzcat(cs.kron(np.eye(Np), -A), np.zeros((Np * nx, nx))) +
                cs.horzcat(np.zeros((Np * nx, nx)), np.eye(Np * nx)),
                cs.kron(np.eye(Np), -B)
            ]
        ])
        Aeq = cs.horzcat(Aeq, np.zeros((Aeq.shape[0], s_size)))
        beq = cs.vertcat(x0, *[e for _ in range(Np)])

        # state and input constraints
        special_eye = np.diag(not_red)[not_red, :].astype(int)
        C_ = cs.vertcat(special_eye, -special_eye, np.zeros((2 * nu + ns, nx)))
        D_ = cs.vertcat(np.zeros((2 * ns, nu)), np.eye(nu), -np.eye(nu),
                        np.zeros((ns, nu)))
        E_ = -cs.vertcat(np.eye(ns), np.eye(ns), np.zeros((2 * nu, ns)),
                         np.eye(ns))
        Aineq = cs.horzcat(
            np.zeros(((2 * (ns + nu) + ns) * Np, nx)),
            cs.kron(np.eye(Np), C_),
            cs.kron(np.eye(Np), D_),
            cs.kron(np.eye(Np), E_),
        )
        bineq = cs.kron(np.ones(Np), cs.vertcat(
            (1 - backoff) * ubx,
            -(1 + backoff) * lbx,
            ubu,
            -lbu,
            np.zeros(ns)
        ))
        if type == 'Q':
            Aeq = cs.vertcat(Aeq, cs.horzcat(
                np.zeros((nu, x_size)), np.eye(nu),
                np.zeros((nu, u_size - nu + s_size))
            ))
            beq = cs.vertcat(beq, u0)
        else:
            q[x_size:x_size + nu] = perturbation

        self.cvx = {'P': P, 'q': q, 'G': Aineq, 'h': bineq, 'A': Aeq, 'b': beq}

        # ### numeric tests ###
        # y = cs.vertcat(cs.vec(x_exp), cs.vec(u_exp), cs.vec(slack))
        # from mpc.generic_mpc import subsevalf
        # y_rnd = np.random.randn(*y.shape)
        # p_rnd = np.random.randn(*self.p.shape)
        # J2 = 0.5 * y.T @ P @ y + q.T @ y + cost_offset
        # J_val = float(subsevalf(J, [self.p, y], [p_rnd, y_rnd]))
        # J2_val = float(subsevalf(J2, [self.p, y], [p_rnd, y_rnd]))
        # print('Cost:', abs(J_val - J2_val))
        # dyn1 = subsevalf(cs.vec(self.cons['dyn']), [self.p, y], [p_rnd, y_rnd])
        # dyn2 = subsevalf(Aeq @ y - beq, [self.p, y], [p_rnd, y_rnd])[10:]
        # print('Dyn: ', np.abs(dyn1 - dyn2).max())
        # print('DONE')

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

        # solve also in cvx
        from mpc.generic_mpc import subsevalf
        x0 = cvx.matrix(np.array(cs.vertcat(
            pars['x0'],
            cs.vec(vals0['x']),
            cs.vec(vals0['u']),
            cs.vec(vals0['slack']))))
        P = subsevalf(self.cvx['P'], self.pars, pars)
        q = subsevalf(self.cvx['q'], self.pars, pars)
        G = subsevalf(self.cvx['G'], self.pars, pars)
        h = subsevalf(self.cvx['h'], self.pars, pars)
        A = subsevalf(self.cvx['A'], self.pars, pars)
        b = subsevalf(self.cvx['b'], self.pars, pars)
        sol2 = cvx.solvers.qp(
            P=cvx.matrix(P),
            q=cvx.matrix(q),
            G=cvx.matrix(G),
            h=cvx.matrix(h),
            A=cvx.matrix(A),
            b=cvx.matrix(b),
            initvals={'x': x0}
        )
        x = np.array(sol2['x'])
        sol2['f'] = 0.5 * x.T @ P @ x + q.T @ x + \
            subsevalf(self.cost_offset, self.pars, pars)
        return sol
