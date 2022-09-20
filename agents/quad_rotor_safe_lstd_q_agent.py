import casadi as cs
import logging
import numpy as np
from agents.quad_rotor_lstd_q_agent import QuadRotorLSTDQAgent
from joblib import Parallel
from mpc import MPCSolverError
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted
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
        gpr: GaussianProcessRegressor,
        opts: dict[str, Any] = None
    ) -> None:
        if opts is None:
            opts = {}
        self._gpr = gpr
        cs.Callback.__init__(self)
        self.construct(name, opts)

    def eval(self, arg):
        return self._gpr.predict(np.array(arg[0]))


class MultiOutputGaussianProcessRegressor(MultiOutputRegressor):
    '''Custom multioutput regressor adapted to GP regression.'''

    def __init__(
        self,
        kernel: kernels.Kernel = None,
        *,
        alpha: float = 1e-10,
        optimizer: str = 'fmin_l_bfgs_b',
        n_restarts_optimizer: int = 0,
        normalize_y: bool = False,
        copy_X_train: bool = True,
        random_state: int = None,
        n_jobs: int = None
    ) -> None:
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state
        )
        super().__init__(estimator, n_jobs=n_jobs)

    def predict(self, X, return_std=False, return_cov=False):
        '''See `GaussianProcessRegressor.predict`.'''
        check_is_fitted(self)
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(
                X, return_std, return_cov
            ) for e in self.estimators_
        )
        if return_std or return_cov:
            return tuple(np.array(o).T for o in zip(*y))
        return np.asarray(y).T


def constraint_violation(
    *arrays_and_bounds: tuple[np.ndarray, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    '''Computes constraint violations.

    Parameters
    ----------
    arrays_and_bounds : *tuple[array_like, array_like]
        Any number of `(array, bounds)` for which to compute the constraint 
        violations.
        For each tuple, `bounds` is an array of shape `(N, 2)`, where `N` is 
        the number of features, and the first and second columns are lower and
        upper bounds, respectively. `array` is an array of shape `(N, ...)`.

    Returns
    -------
    cv : list[tuple[array_like, array_like]]
        For each tuple provided, returns a tuple of lower and upper bound 
        violations. 
    '''
    cv = []
    for a, bnd in arrays_and_bounds:
        lb, ub = bnd[:, 0], bnd[:, 1]
        g_lb = np.expand_dims(lb, tuple(range(1, a.ndim))) - a
        g_ub = a - np.expand_dims(ub, tuple(range(1, a.ndim)))
        cv.append((g_lb, g_ub))
    return cv


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
        self._gpr = MultiOutputGaussianProcessRegressor(
            kernel=(
                1**2 * kernels.RBF(length_scale=np.ones(n)) +
                kernels.WhiteKernel()
            ),
            alpha=1e-6,
            n_restarts_optimizer=9
        )
        self._gpr_data = []
        gpr = GPRCallback('GPR', gpr=self._gpr, opts={'enable_fd': True})
        g = None

        # prepare solver
        qp = {'x': theta_new, 'p': cs.vertcat(theta, c), 'f': f, 'g': g}
        opts = {'print_iter': True, 'print_header': True}
        self._solver = cs.qpsol(f'qpsol_{self.name}', 'qrqp', qp, opts)
