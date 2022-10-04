import casadi as cs
import logging
import numpy as np
import time
from agents.quad_rotor_base_agents import QuadRotorBaseLearningAgent
from dataclasses import dataclass, field
from envs import QuadRotorEnv
from itertools import chain
from mpc import Solution, QuadRotorMPCConfig
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dtrtri
from sklearn.gaussian_process import kernels
from typing import Optional, Union
from util.casadi import norm_ppf
from util.configurations import BaseConfig, init_config
from util.errors import MPCSolverError, UpdateError
from util.gp import MultitGaussianProcessRegressor, CasadiKernels
from util.math import NormalizationService, \
    cholesky_added_multiple_identities, constraint_violation
from util.rl import ReplayMemory


@dataclass
class QuadRotorLSTDQAgentConfig(BaseConfig):
    # initial learnable RL weights and their bounds
    init_pars: dict[str, tuple[float, tuple[float, float]]] = field(
        default_factory=lambda: {
            'g': (9.81, (1, 40)),
            'thrust_coeff': (2.0, (0.1, 4)),
            # 'w_x': (1e1, (1e-3, np.inf)),
            # 'w_u': (1e0, (1e-3, np.inf)),
            # 'w_s': (1e2, (1e-3, np.inf))
        })

    # fixed non-learnable weights
    fixed_pars: dict[str, float] = field(default_factory=lambda: {
        'pitch_d': 12,
        'pitch_dd': 7,
        'pitch_gain': 11,
        'roll_d': 10.5,
        'roll_dd': 8,
        'roll_gain': 9,
        'w_x': 1e1,
        'w_u': 1e0,
        'w_s': 1e2
    })

    # experience replay parameters
    replay_maxlen: float = 20
    replay_sample_size: float = 10
    replay_include_last: float = 5

    # RL algorithm parameters
    gamma: float = 1.0
    lr: float = 1e-1
    max_perc_update: float = np.inf

    # normalization ranges for stage costs (for other pars ranges are taken
    # from env)
    normalization_ranges: dict[str, np.ndarray] = field(
        default_factory=lambda: {
            'w_x': np.array([0, 1e2]),
            'w_u': np.array([0, 1e1]),
            'w_s': np.array([0, 1e3]),
        })


class QuadRotorLSTDQAgent(QuadRotorBaseLearningAgent):
    '''
    Least-Squares Temporal Difference-based Q-learning RL agent for the quad 
    rotor environment. The agent adapts its MPC parameters/weights by value 
    methods, averaging over batches of episodes via Least-Squares, with the 
    goal of improving performance/reducing cost of each episode.

    The RL update exploits a replay memory to spread out noise.
    '''

    config_cls: type = QuadRotorLSTDQAgentConfig

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: Union[dict, QuadRotorLSTDQAgentConfig] = None,
        mpc_config: Union[dict, QuadRotorMPCConfig] = None,
        seed: int = None
    ) -> None:
        '''
        Initializes a LSTDQ agent for the quad rotor env.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the Q-learning agent.
        agentname : str, optional
            Name of the Q-learning agent.
        agent_config : dict, QuadRotorQLearningAgentConfig
            A set of parameters for the quadrotor Q-learning agent. If not 
            given, the default ones are used.
        mpc_config : dict, QuadRotorMPCConfig
            A set of parameters for the agent's MPC. If not given, the default
            ones are used.
        seed : int, optional
            Seed for the random number generator.
        '''
        # create base agent
        agent_config = self._init_config(agent_config, env.normalization)
        fixed_pars, init_pars = agent_config.fixed_pars, agent_config.init_pars
        fixed_pars.update({
            'xf': env.config.xf,  # already normalized
            'perturbation': np.nan,
            'backoff': 0.05,
        })
        super().__init__(
            env,
            agentname=agentname,
            agent_config=agent_config,
            fixed_pars=fixed_pars,
            init_learnable_pars=init_pars,
            mpc_config=mpc_config,
            seed=seed
        )

        # in order to learn, Q learning must sometime perturb the action
        self.perturbation_chance = 0.7
        self.perturbation_strength = 5e-1

        # initialize the replay memory. Per each episode the memory saves the
        # gradient and Hessian of Q at each instant
        self.replay_memory = ReplayMemory[list[tuple[np.ndarray, ...]]](
            maxlen=self.config.replay_maxlen, seed=seed)
        self._episode_buffer: list[tuple[np.ndarray, ...]] = []

        # initialize symbols for derivatives to be used later. Also initialize
        # the QP solver used to compute updates
        self._init_derivative_symbols()
        self._init_qp_solver()

    def save_transition(
        self,
        cost: float,
        solQ: Solution,
        solV: Solution
    ) -> None:
        '''
        Schedules the current time-step data to be processed and saved into the
        experience replay memory.

        Parameters
        ----------
        cost : float
            Stage cost given by the environment at the current time step.
        solQ : mpc.Solution
            MPC solution of Q(s,a) where s and a are the current state and 
            action.
        solV : mpc.Solution
            MPC solution of V(s+) where s+ is the net state.
        '''
        # compute td error
        target = cost + self.config.gamma * solV.f
        td_err = target - solQ.f

        # compute numerical gradients w.r.t. params
        dQ = solQ.value(self.dQdtheta).reshape(-1, 1)
        d2Q = solQ.value(self.d2Qdtheta)

        # compute gradient and approximated hessian
        g = -td_err * dQ
        H = dQ @ dQ.T - td_err * d2Q

        # save to replay memory
        self._episode_buffer.append((g, H))

    def consolidate_episode_experience(self) -> None:
        '''
        At the end of an episode, computes the remaining operations and 
        saves results to the replay memory as arrays.
        '''
        if len(self._episode_buffer) == 0:
            return
        self.replay_memory.append(self._episode_buffer.copy())
        self._episode_buffer.clear()

    def update(self) -> np.ndarray:
        # sample the memory
        cfg: QuadRotorLSTDQAgentConfig = self.config
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # sum over the batch of samples and compute update direction p
        g, H = (np.mean(o, axis=0) for o in zip(*chain.from_iterable(sample)))
        R = cholesky_added_multiple_identities(H)
        p = cho_solve((R, True), g).flatten()

        # run QP solver and update weights
        theta = self.weights.values()
        pars = np.block([theta, p, cfg.lr])
        lb, ub = self._get_percentage_bounds(
            theta, self.weights.bounds(), cfg.max_perc_update)
        sol = self._solver(p=pars, lbx=lb, ubx=ub, x0=theta - cfg.lr * p)
        if not self._solver.stats()['success']:
            raise UpdateError(f'RL update failed in epoch {self._epoch_n}.')
        self.weights.update_values(sol['x'].full().flatten())
        return p

    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        return_info: bool = False
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ]:
        logger = logger or logging.getLogger('dummy')

        env, name, epoch_n = self.env, self.name, self._epoch_n
        returns = np.zeros(n_episodes)
        seeds = self._make_seed_list(seed, n_episodes)

        for e in range(n_episodes):
            state = env.reset(seed=seeds[e])
            self.reset()
            truncated, terminated, t = False, False, 0
            action = self.predict(state, deterministic=False)[0]

            while not (truncated or terminated):
                # compute Q(s, a)
                self.fixed_pars.update({'u0': action})
                solQ = self.solve_mpc('Q', state)

                # step the system
                state, r, truncated, terminated, _ = env.step(action)
                returns[e] += r

                # compute V(s+)
                action, _, solV = self.predict(state, deterministic=False)

                # save only successful transitions
                if solQ.success and solV.success:
                    self.save_transition(r, solQ, solV)
                else:
                    raise MPCSolverError(
                        f'{name}|{epoch_n}|{e}|{t}: MPC failed.')
                t += 1

            # when episode is done, consolidate its experience into memory
            self.consolidate_episode_experience()
            logger.debug(f'{name}|{epoch_n}|{e}: J={returns[e]:,.3f}')

        # when all m episodes are done, perform RL update and reduce
        # exploration strength and chance
        update_grad = self.update()
        self.perturbation_strength *= perturbation_decay
        self.perturbation_chance *= perturbation_decay

        # log training outcomes and return cumulative returns
        logger.debug(f'{self.name}|{epoch_n}: J_mean={returns.mean():,.3f}; '
                     f'||p||={np.linalg.norm(update_grad):.3e}; ' +
                     self.weights.values2str())
        return (
            (returns, update_grad, self.weights.values(as_dict=True))
            if return_info else
            returns
        )

    def _init_config(
        self,
        config: Optional[QuadRotorLSTDQAgentConfig],
        normalization: Optional[NormalizationService]
    ) -> QuadRotorLSTDQAgentConfig:
        '''
        Initializes the agent configuration and fixed and initial pars (does
        not save to self).
        '''
        C = init_config(config, self.config_cls)
        N = normalization
        if N is not None:
            N.register(C.normalization_ranges)
            for k, v in C.fixed_pars.items():
                C.fixed_pars[k] = N.normalize(k, v)
            for k, v in C.init_pars.items():
                C.init_pars[k] = (N.normalize(k, v[0]), N.normalize(k, v[1]))
        return C

    def _init_derivative_symbols(self) -> None:
        '''Computes symbolical derivatives needed for Q learning.'''
        theta = self.weights.symQ()
        lagr = self.Q.lagrangian
        d2Qdtheta, dQdtheta = cs.hessian(lagr, theta)
        self.dQdtheta = cs.simplify(dQdtheta)
        self.d2Qdtheta = cs.simplify(d2Qdtheta)

    def _init_qp_solver(self) -> None:
        n_theta = self.weights.n_theta

        # prepare symbols
        theta: cs.SX = cs.SX.sym('theta', n_theta, 1)
        theta_new: cs.SX = cs.SX.sym('theta+', n_theta, 1)
        dtheta = theta_new - theta
        p: cs.SX = cs.SX.sym('p', n_theta, 1)
        lr: cs.SX = cs.SX.sym('lr', n_theta, 1)

        # prepare solver
        qp = {
            'x': theta_new,
            'f': 0.5 * dtheta.T @ dtheta + (lr * p).T @ dtheta,
            'p': cs.vertcat(theta, p, lr)
        }
        opts = {'print_iter': False, 'print_header': False}
        self._solver = cs.qpsol(f'qpsol_{self.name}', 'qrqp', qp, opts)


@dataclass
class QuadRotorGPSafeLSTDQAgentConfig(QuadRotorLSTDQAgentConfig):
    alpha: float = 1e-10
    kernel_cls: type = kernels.RBF  # kernels.Matern
    average_violation: bool = True

    mu0: float = 0.0  # target constraint violation
    beta: float = 0.9   # probability of target violation satisfaction
    
    mu0_backtracking: float = 0.0
    beta_backtracking: float = 0.95
    max_backtracking_iter: int = 35 

    n_opti: int = 14  # number of multistart for nonlinear optimization

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_cls, str):
            return
        self.kernel_cls = getattr(kernels, self.kernel_cls)


class QuadRotorGPSafeLSTDQAgent(QuadRotorLSTDQAgent):
    '''GP-based safe variant of the LSTDQ agent.'''

    config_cls: type = QuadRotorGPSafeLSTDQAgentConfig

    def update(self) -> tuple[np.ndarray, tuple[float, float], float]:
        cfg: QuadRotorGPSafeLSTDQAgentConfig = self.config
        self._solver, gp_fit_time = self._init_qp_solver()

        # sample the memory
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # sum over the batch of samples and compute update direction p
        g, H = [sum(o) for o in zip(*chain.from_iterable(sample))]
        R = cholesky_added_multiple_identities(H)
        p = cho_solve((R, True), g).flatten()

        # run QP solver (backtrack on beta if necessary) and update weights
        theta = self.weights.values()
        candidates = np.linspace(theta, theta - cfg.lr * p, cfg.n_opti) + \
            self.np_random.normal(size=(cfg.n_opti, theta.size), scale=0.1)
        mu0, beta = cfg.mu0, cfg.beta
        pars = np.block([theta, p, cfg.lr, mu0, beta])
        lb, ub = self._get_percentage_bounds(
            theta, self.weights.bounds(), cfg.max_perc_update)
        for _ in range(cfg.max_backtracking_iter):
            # run the solver for each candidate
            best_sol = None
            for x0 in candidates:
                sol = self._solver(
                    p=pars, lbx=lb, ubx=ub, lbg=-np.inf, ubg=0, x0=x0)
                if self._solver.stats()['success'] and \
                        (best_sol is None or sol['f'] < best_sol['f']):
                    best_sol = sol

            # either apply the successful update or backtrack
            if best_sol is not None:
                self.weights.update_values(best_sol['x'].full().flatten())
                return p, (mu0, beta), gp_fit_time
            else:
                mu0 += cfg.mu0_backtracking
                beta *= cfg.beta_backtracking
                pars[-2:] = (mu0, beta)
        raise UpdateError(f'Update failed (beta={beta:.3f}, mu0={mu0:.2f}).')
        

    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        return_info: bool = False
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ]:
        logger = logger or logging.getLogger('dummy')

        env, name, epoch_n = self.env, self.name, self._epoch_n
        returns = np.zeros(n_episodes)
        seeds = self._make_seed_list(seed, n_episodes)
        violations: list[np.ndarray] = []

        for e in range(n_episodes):
            state = env.reset(seed=seeds[e])
            self.reset()
            truncated, terminated, t = False, False, 0
            action = self.predict(state, deterministic=False)[0]
            states, actions = [state], []

            while not (truncated or terminated):
                # compute Q(s, a)
                self.fixed_pars.update({'u0': action})
                solQ = self.solve_mpc('Q', state)

                # step the system
                state, r, truncated, terminated, _ = env.step(action)
                returns[e] += r
                states.append(state)
                actions.append(action)

                # compute V(s+)
                action, _, solV = self.predict(state, deterministic=False)

                # save only successful transitions
                if solQ.success and solV.success:
                    self.save_transition(r, solQ, solV)
                else:
                    raise MPCSolverError(
                        f'{name}|{epoch_n}|{e}|{t}: MPC failed.')
                t += 1

            # when episode is done, consolidate its experience into memory, and
            # compute its trajectories' constraint violations
            self.consolidate_episode_experience()
            logger.debug(f'{name}|{epoch_n}|{e}: J={returns[e]:,.3f}')
            violations.append(self._compute_violation(states, actions))

        # when all m episodes are done, append the violations for the GP to fit
        # and perform RL update and reduce exploration strength and chance
        theta = self.weights.values()
        if self.config.average_violation:
            self._gpr_dataset.append((theta, np.mean(violations, axis=0)))
        else:
            self._gpr_dataset.extend(((theta, v) for v in violations))
        update_grad, backtracked_gp_pars, gp_fit_time = self.update()
        self.backtracked_gp_pars_history.append(backtracked_gp_pars)
        self.perturbation_strength *= perturbation_decay
        self.perturbation_chance *= perturbation_decay

        # log training outcomes and return cumulative returns
        logger.debug(f'{self.name}|{epoch_n}: J_mean={returns.mean():,.3f}; '
                     f'||p||={np.linalg.norm(update_grad):.3e}; ' +
                     f'beta={backtracked_gp_pars[1] * 100:.1f}%' +
                     f'GP fit time={gp_fit_time:.2}s' +
                     self.weights.values2str())
        return (
            (returns, update_grad, self.weights.values(as_dict=True))
            if return_info else
            returns
        )

    def _compute_violation(
        self,
        states: list[np.ndarray],
        actions: list[np.ndarray],
    ) -> np.ndarray:
        x_bnd, u_bnd = self.env.config.x_bounds, self.env.config.u_bounds
        x, u = np.stack(states, axis=-1), np.stack(actions, axis=-1)

        # compute state and action constraints violations and apply 2
        # reductions: first merge lb and ub in a single constraint (since both
        # cannot be active at the same time) (max axis=1); then reduce each
        # trajectory's violations to scalar by picking max violation (max
        # axis=2)
        x_cv, u_cv = (
            cv.max(axis=(1, 2))
            for cv in constraint_violation((x, x_bnd), (u, u_bnd))
        )

        # egress only over finite data
        cv = np.concatenate((x_cv, u_cv))
        return cv[np.isfinite(cv)]

    def _init_qp_solver(self) -> Optional[tuple[cs.Function, float]]:
        if not hasattr(self, '_gpr'):
            # this is the first time the initilization gets called, so only the
            # GP regressor is initialized and the QP symbols with fixed size
            cfg: QuadRotorGPSafeLSTDQAgentConfig = self.config
            n_theta = self.weights.n_theta

            # create regressor
            kernel = (
                1**2 * cfg.kernel_cls(
                    length_scale=np.ones(n_theta),
                    length_scale_bounds=(1e-5, 1e6)) +
                kernels.WhiteKernel()
            )
            self._gpr = MultitGaussianProcessRegressor(
                kernel=kernel,
                alpha=cfg.alpha,
                n_restarts_optimizer=cfg.n_opti,
                random_state=self.seed
            )
            # if np.isnan(cfg.mu0):
            #     # initiliaze GP prior so that all thetas are safe at start
            #     prior_mu, prior_std = 0, np.sqrt(
            #         CasadiKernels.sklearn2func(kernel)(np.zeros(1), diag=True)
            #     ).item()
            #     cfg.mu0 = prior_mu + norm_ppf(cfg.beta) * prior_std

            # compute symbols that do not depend on GP
            theta: cs.SX = cs.SX.sym('theta', n_theta, 1)
            theta_new: cs.SX = cs.SX.sym('theta+', n_theta, 1)
            dtheta = theta_new - theta
            p: cs.SX = cs.SX.sym('p', n_theta, 1)
            lr: cs.SX = cs.SX.sym('lr', n_theta, 1)
            mu0: cs.SX = cs.SX.sym('mu0', 1, 1)
            beta: cs.SX = cs.SX.sym('beta', 1, 1)
            self._qp = {
                'theta+': theta_new,
                'mu0': mu0,
                'beta': beta,
                'f': 0.5 * dtheta.T @ dtheta + (lr * p).T @ dtheta,
                'p': cs.vertcat(theta, p, lr, mu0, beta),
                'g': None,
                'opts': {
                    'expand': True,  # False when using callback
                    'print_time': False,
                    'ipopt': {
                        'max_iter': 500,
                        'sb': 'yes',
                        # for debugging
                        'print_level': 0,
                        'print_user_options': 'no',
                        'print_options_documentation': 'no'
                    }
                }
            }

            # initialize storages
            self._gpr_dataset: list[tuple[np.ndarray, np.ndarray]] = []
            self.backtracked_gp_pars_history: list[tuple[float, float]] = []
        else:
            # regressor initialization (other branch) has been done, so we can
            # move to fitting the GP and creating the QP constraints
            theta, cv = (np.stack(o, axis=0) for o in zip(*self._gpr_dataset))
            start = time.perf_counter()
            self._gpr.fit(theta, cv)
            fit_time = time.perf_counter() - start
            for gpr in self._gpr.estimators_:
                gpr.kernel = gpr.kernel_

            # compute the symbolic posterior mean and std of the GP
            theta_new = self._qp['theta+'].T
            mean, var = [], []
            for gpr in self._gpr.estimators_:
                kernel_func = CasadiKernels.sklearn2func(gpr.kernel_)
                L_inv = dtrtri(gpr.L_, lower=True)[0]
                k = kernel_func(gpr.X_train_, theta_new)  # X_train_==theta
                V = L_inv @ k
                mean.append(k.T @ gpr.alpha_)
                var.append(kernel_func(theta_new, diag=True) - cs.sum1(V**2).T)
            mean, std = cs.vertcat(*mean), cs.sqrt(cs.vertcat(*var))

            # compute the constraint function for the new theta
            mu0, beta = self._qp['mu0'], self._qp['beta']
            self._qp['g'] = mean - mu0 + norm_ppf(beta) * std

            # create QP solver
            qp = {
                'x': theta_new,
                'p': self._qp['p'],
                'f': self._qp['f'],
                'g': self._qp['g']
            }
            solver = cs.nlpsol(
                f'QP_ipopt_{self.name}', 'ipopt', qp, self._qp['opts'])
            return solver, fit_time
