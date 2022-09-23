import casadi as cs
import logging
import numpy as np
import time
from agents.quad_rotor_lstd_q_agent import QuadRotorLSTDQAgent, \
    QuadRotorLSTDQAgentConfig
from dataclasses import dataclass
from itertools import chain
from mpc import MPCSolverError
from scipy.linalg import cho_solve
from scipy.linalg.lapack import dtrtri
from sklearn.gaussian_process import kernels
from typing import Optional, Union
from util.casadi import norm_ppf
from util.math import cholesky_added_multiple_identities, constraint_violation
from util.gp import MultitGaussianProcessRegressor, CasadiKernels


@dataclass(frozen=True)
class QuadRotorGPSafeLSTDQAgentConfig(QuadRotorLSTDQAgentConfig):
    mu0: float = 0.0    # target constraint violation
    beta: float = 0.9   # probability of target violation satisfaction
    beta_backtracking: float = 0.9
    n_restarts_optimizer: int = 9


class QuadRotorGPSafeLSTDQAgent(QuadRotorLSTDQAgent):
    config_cls: type = QuadRotorGPSafeLSTDQAgentConfig

    def update(self) -> tuple[np.ndarray, float, float]:
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
        x0 = theta - cfg.lr * p
        beta = cfg.beta
        pars = np.block([theta, p, cfg.lr, cfg.mu0, beta])
        lb, ub = self._get_percentage_bounds(
            theta, self.weights.bounds(), cfg.max_perc_update)
        while True:
            sol = self._solver(
                lbx=lb, ubx=ub, lbg=-np.inf, ubg=0, x0=x0, p=pars)
            if self._solver.stats()['success']:
                self.weights.update_values(sol['x'].full().flatten())
                break
            else:
                beta *= cfg.beta_backtracking
                if beta < 1 / 10:
                    raise RuntimeError('RL update failed.')
                p[-1] = beta
        return p, beta, gp_fit_time

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
            self._gpr_dataset.append((
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
        return cv[np.isfinite(cv)]

    def _init_qp_solver(self) -> Optional[tuple[cs.Function, float]]:
        if not hasattr(self, '_gpr'):
            # this is the first time the initilization gets called, so only the
            # GP regressor is initialized and the QP symbols with fixed size
            n_theta = sum(self.weights.sizes())

            self._gpr_dataset: list[tuple[np.ndarray, np.ndarray]] = []
            self._gpr = MultitGaussianProcessRegressor(
                kernel=(
                    1**2 * kernels.RBF(length_scale=np.ones(n_theta)) +
                    kernels.WhiteKernel()  # use only if not averaging
                ),
                alpha=1e-6,
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                random_state=self.seed
            )

            theta: cs.MX = cs.MX.sym('theta', n_theta, 1)
            theta_new: cs.MX = cs.MX.sym('theta+', n_theta, 1)
            dtheta = theta_new - theta
            p: cs.MX = cs.MX.sym('p', n_theta, 1)
            lr: cs.MX = cs.MX.sym('lr', 1, 1)
            mu0: cs.MX = cs.MX.sym('mu0', 1, 1)
            beta: cs.MX = cs.MX.sym('beta', 1, 1)

            self._qp = {
                'theta_new': theta_new,
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
                        # debug
                        'print_level': 0,
                        'print_user_options': 'no',
                        'print_options_documentation': 'no'
                    }
                }
            }
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
            theta_new = self._qp['theta_new'].T
            mean, std = [], []
            for gpr in self._gpr.estimators_:
                kernel_func = CasadiKernels.sklearn2func(gpr.kernel_)
                L_inv = dtrtri(gpr.L_, lower=True)[0]
                k = kernel_func(gpr.X_train_, theta_new)  # X_train_==theta
                V = L_inv @ k
                mean.append(k.T @ gpr.alpha_)
                std.append(kernel_func(theta_new, diag=True) - cs.sum1(V**2).T)
            mean, std = cs.vertcat(*mean), cs.sqrt(cs.vertcat(*std))

            # compute the constraint function for the new theta
            mu0, beta = self._qp['mu0'], self._qp['beta']
            g = mean - mu0 + norm_ppf(beta) * std
            self._qp['g'] = g

            # create QP solver
            solver = cs.nlpsol(
                f'QP_ipopt_{self.name}',
                'ipopt', {
                    'x': theta_new,
                    'f': self._qp['f'],
                    'p': self._qp['p'],
                    'g': g
                },
                self._qp['opts'])
            return solver, fit_time
