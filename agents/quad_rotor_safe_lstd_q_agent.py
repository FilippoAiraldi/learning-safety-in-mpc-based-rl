import casadi as cs
import logging
import numpy as np
import sklearn.gaussian_process as gp
from agents.quad_rotor_lstd_q_agent import QuadRotorLSTDQAgent
from mpc import MPCSolverError
from typing import Any, Union


# useful links:
# https://scikit-learn.org/stable/modules/gaussian_process.html
# http://gaussianprocess.org/gpml/chapters/RW.pdf
# https://web.casadi.org/blog/tensorflow/
# https://groups.google.com/g/casadi-users/c/gLJNzajFM6w


class GPRCallback(cs.Callback):
    def __init__(
        self,
        name: str,
        gpr: gp.GaussianProcessRegressor,
        opts: dict[str, Any] = None
    ) -> None:
        if opts is None:
            opts = {}
        self._gpr = gpr
        cs.Callback.__init__(self)
        self.construct(name, opts)

    def eval(self, arg):
        return self._gpr.predict(np.array(arg[0]))


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

            # additionally, save the trajectory outcome for the GP to learn
            # self.

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
        self._gpr = gp.GaussianProcessRegressor(
            kernel=(
                1**2 * gp.kernels.RBF(length_scale=np.ones(n)) +
                gp.kernels.WhiteKernel()
            ),
            alpha=0.1,
            n_restarts_optimizer=9
        )
        self._gpr_data = []
        gpr = GPRCallback('GPR', gpr=self._gpr, opts={'enable_fd': True})
        g = None

        # prepare solver
        qp = {'x': theta_new, 'p': cs.vertcat(theta, c), 'f': f, 'g': g}
        opts = {'print_iter': True, 'print_header': True}
        self._solver = cs.qpsol(f'qpsol_{self.name}', 'qrqp', qp, opts)
