import casadi as cs
import logging
import numpy as np
from agents.quad_rotor_lstd_q_agent import QuadRotorLSTDQAgent
from agents.safety import (
    constraint_violation,
    MultitGaussianProcessRegressor,
    GaussianProcessRegressorConstraintCallback
)
from mpc import MPCSolverError
from sklearn.gaussian_process import kernels
from typing import Union


class QuadRotorSafeLSTDQAgent(QuadRotorLSTDQAgent):
    def update(self) -> np.ndarray:
        raise NotImplementedError('Launch new type of qp solver.')

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

    def _init_qp_solver(self) -> None:
        n = sum(self.weights.sizes())

        # prepare symbols
        theta: cs.SX = cs.SX.sym('theta', n, 1)
        theta_new: cs.SX = cs.SX.sym('theta+', n, 1)
        c: cs.SX = cs.SX.sym('c', n, 1)

        # compute objective
        dtheta = theta_new - theta
        f = 0.5 * dtheta.T @ dtheta + c.T @ dtheta

        # additional constraint modelled via the GP
        # create a GP as regressor of (theta, constraint violation) dataset
        self._gpr = MultiOutputGaussianProcessRegressor(
            kernel=(
                1**2 * kernels.RBF(length_scale=np.ones(n)) +
                kernels.WhiteKernel()
            ),
            alpha=1e-6,
            n_restarts_optimizer=9
        )
        self._gpr_data = []
        gpr = GaussianProcessRegressorCallback(
            'GPR', gpr=self._gpr, opts={'enable_fd': True})
        g = None

        # prepare solver
        qp = {'x': theta_new, 'p': cs.vertcat(theta, c), 'f': f, 'g': g}
        opts = {'print_iter': True, 'print_header': True}
        self._solver = cs.qpsol(f'qpsol_{self.name}', 'qrqp', qp, opts)

    def _compute_constraint_violation(
        self,
        states: list[np.ndarray],
        actions: list[np.ndarray],
    ) -> np.ndarray:
        x_bnd, u_bnd = self.env.config.x_bounds, self.env.config.u_bounds
        x, u = np.stack(states, axis=-1), np.stack(actions, axis=-1)

        # compute state and action constraints violations along trajectory and
        # pick maximum violation
        (x_cv_lb, x_cv_ub), (u_cv_lb, u_cv_ub) = [
            (cv_lb.max(axis=-1), cv_ub.max(axis=-1))
            for cv_lb, cv_ub in constraint_violation((x, x_bnd), (u, u_bnd))
        ]
        cv = np.concatenate((x_cv_lb, x_cv_ub, u_cv_lb, u_cv_ub))
        return cv[np.isfinite(cv)]  # only regress over finite data
