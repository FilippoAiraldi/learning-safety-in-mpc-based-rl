import casadi as cs
import logging
import numpy as np
import sklearn.gaussian_process as gp
from agents.quad_rotor_LSTD_Q_agent import QuadRotorLSTDQAgent
from typing import Any, Union


class QuadRotorGPSafeLSTDQAgent(QuadRotorLSTDQAgent):
    def __init__(self, *args, **kwargs) -> None:
        # create a GP as regressor of (theta, constraint violation) dataset
        self._gpr = gp.GaussianProcessRegressor(
            kernel=1**2 * gp.kernels.RBF(),
            n_restarts_optimizer=19
        )

        # instantiate super
        super().__init__(*args, **kwargs)

    def update(self) -> np.ndarray:
        raise NotImplementedError('Launch new type of qp solver.')

    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        raises: bool = True
    ) -> np.ndarray:
        raise NotImplementedError('Save GP data point after each episode.')

    def _init_qp_solver(self) -> None:
        raise NotImplementedError('Add GP as constraint.')

        # n = sum(self.weights.sizes())

        # # prepare symbols
        # theta: cs.SX = cs.SX.sym('theta', n, 1)
        # theta_new: cs.SX = cs.SX.sym('theta+', n, 1)
        # c: cs.SX = cs.SX.sym('c', n, 1)

        # # compute objective
        # dtheta = theta_new - theta
        # f = 0.5 * dtheta.T @ dtheta + c.T @ dtheta

        # # the qp solver for RL updates now has an additional constraint
        # # modelled via the GP
        # class GPRCallback(cs.Callback):
        #     def __init__(
        #         self,
        #         name: str,
        #         gpr: gp.GaussianProcessRegressor,
        #         opts: dict[str, Any] = None
        #     ) -> None:
        #         if opts is None:
        #             opts = {}
        #         self._gpr = gpr
        #         cs.Callback.__init__(self)
        #         self.construct(name, opts)

        #     def eval(self, arg):
        #         return self._gpr.predict(np.array(arg[0]))
