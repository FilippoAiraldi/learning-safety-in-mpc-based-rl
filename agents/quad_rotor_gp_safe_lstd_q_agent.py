import casadi as cs
import logging
import numpy as np
import time
from agents.quad_rotor_lstd_q_agent import (
    QuadRotorLSTDQAgent,
    QuadRotorLSTDQAgentConfig,
)
from agents.safety import (
    constraint_violation,
    MultitGaussianProcessRegressor,
    MultiGaussianProcessRegressorCallback
)
from dataclasses import dataclass
from itertools import chain
from mpc import MPCSolverError
from scipy.linalg import cho_solve
from sklearn.gaussian_process import kernels
from typing import Union
from util import cs_norm_ppf, cholesky_added_multiple_identities


@dataclass(frozen=True)
class QuadRotorGPSafeLSTDQAgentConfig(QuadRotorLSTDQAgentConfig):
    alpha: float = 0.0  # target constraint violation
    beta: float = 0.9   # probability of target violation satisfaction
    beta_backtracking: float = 0.9
    n_restarts_optimizer: int = 9


class QuadRotorGPSafeLSTDQAgent(QuadRotorLSTDQAgent):
    config_cls: type = QuadRotorGPSafeLSTDQAgentConfig

    def update(self) -> tuple[np.ndarray, float, float]:
        cfg: QuadRotorGPSafeLSTDQAgentConfig = self.config

        # fit the gp to the data
        theta, cv = (np.stack(o, axis=0) for o in zip(*self._gpr_data))
        start = time.perf_counter()
        self._gpr.fit(theta, cv)
        fit_time = time.perf_counter() - start
        alpha, beta, beta_bt = cfg.alpha, cfg.beta, cfg.beta_backtracking

        # sample the memory
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # sum over the batch of samples and compute update direction p
        g, H = [sum(o) for o in zip(*chain.from_iterable(sample))]
        R = cholesky_added_multiple_identities(H)
        p = cho_solve((R, True), g).flatten()
        c = cfg.lr * p

        # run QP solver (backtrack on beta if necessary) and update weights
        theta = self.weights.values()
        x0 = theta - c
        p = np.block([theta, c, alpha, beta])
        lb, ub = self._get_percentage_bounds(
            theta, self.weights.bounds(), cfg.max_perc_update)
        while True:
            sol = self._solver(lbx=lb, ubx=ub, lbg=-np.inf, ubg=0, x0=x0, p=p)
            if self._solver.stats()['success']:
                self.weights.update_values(sol['x'].full().flatten())
                break
            else:
                beta *= beta_bt
                if beta < 1 / 3:
                    raise RuntimeError('RL update failed.')
                p[-1] = beta

        # save the new kernel to warmstart next fitting
        for gpr in self._gpr.estimators_:
            gpr.kernel = gpr.kernel_
        return p, p[-1], fit_time

    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        raises: bool = True,
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
                    logger.warning(f'{name}|{epoch_n}|{e}|{t}: MPC failed.')
                    if raises:
                        raise MPCSolverError('MPC failed.')
                t += 1

            # when episode is done, consolidate its experience into memory, and
            # compute its trajectories' constraint violations
            self.consolidate_episode_experience()
            logger.debug(f'{name}|{epoch_n}|{e}: J={returns[e]:,.3f}')
            self._gpr_data.append((
                self.weights.values(),
                self._compute_constraint_violation(states, actions)
            ))

        # when all m episodes are done, perform RL update and reduce
        # exploration strength and chance
        update_grad, backtracked_beta, gp_fit_time = self.update()
        self.perturbation_strength *= perturbation_decay
        self.perturbation_chance *= perturbation_decay

        # log training outcomes and return cumulative returns
        logger.debug(f'{self.name}|{epoch_n}: J_mean={returns.mean():,.3f}; '
                     f'||p||={np.linalg.norm(update_grad):.3e}; ' +
                     f'beta={backtracked_beta * 100:.1f}%' +
                     f'GP fit time={gp_fit_time:.2}s' +
                     self.weights.values2str())
        return (
            (returns, update_grad, self.weights.values(as_dict=True))
            if return_info else
            returns
        )

    def _init_gpr(self, n: int, n_constraints: int) -> None:
        self._gpr_data: list[tuple[np.ndarray, np.ndarray]] = []
        self._gpr = MultitGaussianProcessRegressor(
            kernel=(
                1**2 * kernels.RBF(length_scale=np.ones(n)) +
                kernels.WhiteKernel()  # use only if not averaging
            ),
            alpha=1e-6,
            n_restarts_optimizer=self.config.n_restarts_optimizer
        )
        self._cs_gpr = MultiGaussianProcessRegressorCallback(
            gpr=self._gpr,
            n_theta=n,
            n_features=n_constraints,
            opts={'enable_fd': True}
        )

    def _init_qp_solver(self) -> None:
        # initialize GP constraints
        n = sum(self.weights.sizes())
        self._n_constraints = \
            (np.isfinite(self.env.config.x_bounds).sum(axis=1) > 0).sum() + \
            (np.isfinite(self.env.config.u_bounds).sum(axis=1) > 0).sum()
        self._init_gpr(n, self._n_constraints)

        # prepare symbols
        theta: cs.MX = cs.MX.sym('theta', n, 1)
        theta_new: cs.MX = cs.MX.sym('theta+', n, 1)
        c: cs.MX = cs.MX.sym('c', n, 1)
        alpha: cs.MX = cs.MX.sym('alpha', 1, 1)
        beta: cs.MX = cs.MX.sym('beta', 1, 1)

        # compute objective
        dtheta = theta_new - theta
        f = 0.5 * dtheta.T @ dtheta + c.T @ dtheta

        # compute GP safety constraints (in canonical form: g(theta) <= 0)
        gp_mean, gp_std = self._cs_gpr(theta)
        g = gp_mean - alpha + cs_norm_ppf(beta) * gp_std

        # prepare solver
        p = cs.vertcat(theta, c, alpha, beta)
        qp = {'x': theta_new, 'p': p, 'f': f, 'g': g}
        opts = {
            'expand': False,  # required (or just omit)
            'print_time': True,
            'ipopt': {
                'max_iter': 100,
                'sb': 'yes',
                # debug
                'print_level': 5,
                'print_user_options': 'no',
                'print_options_documentation': 'no'
            }
        }
        self._solver = cs.nlpsol(f'ipopt_{self.name}', 'ipopt', qp, opts)

    def _compute_constraint_violation(
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
        cv = cv[np.isfinite(cv)]
        assert self._n_constraints == cv.size, \
            'Constraint violation has invalid size.'
        return cv
