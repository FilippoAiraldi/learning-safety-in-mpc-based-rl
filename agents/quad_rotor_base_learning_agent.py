import logging
import numpy as np
from abc import ABC, abstractmethod
from agents.quad_rotor_base_agent import QuadRotorBaseAgent
from mpc import MPCSolverError, QuadRotorMPC
from mpc.wrappers import DifferentiableMPC
from typing import Union
from util.rl import RLParameter, RLParameterCollection


class UpdateError(RuntimeError):
    '''Exception class to raise an error when the agent's update fails.'''
    ...


class QuadRotorBaseLearningAgent(QuadRotorBaseAgent, ABC):
    '''
    Abstract base agent class that renders the two MPC function approximators
    `Q` and `V` differentiable, such that their parameters can be learnt. 
    '''

    def __init__(
        self,
        *args,
        init_learnable_pars: dict[str, tuple[np.ndarray, np.ndarray]],
        **kwargs,
    ) -> None:
        '''
        Instantiates a learning agent.

        Parameters
        ----------
        init_learnable_pars : dict[str, tuple[array_like, array_like]]
            Initial values and bounds for each learnable MPC parameter.
        *args, **kwargs
            See `envs.QuadRotorBaseAgent`.
        '''
        super().__init__(*args, **kwargs)
        self._V = DifferentiableMPC[QuadRotorMPC](self._V)
        self._Q = DifferentiableMPC[QuadRotorMPC](self._Q)
        self._init_learnable_pars(init_learnable_pars)
        self._init_learning_rate()
        self._epoch_n = None  # keeps track of epoch number just for logging

    @property
    def V(self) -> DifferentiableMPC[QuadRotorMPC]:
        return self._V

    @property
    def Q(self) -> DifferentiableMPC[QuadRotorMPC]:
        return self._Q

    @abstractmethod
    def update(self) -> np.ndarray:
        '''
        Updates the MPC function approximation's weights based on the 
        information stored in the replay memory.

        Returns
        -------
        gradient : array_like
            Gradient of the update.
        '''
        pass

    @abstractmethod
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
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_episodes : int
            Number of training episodes for the current epoch.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after this epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : logging.Logger, optional
            For logging purposes.
        return_info : bool, optional
            Whether to return additional information for this epoch update:
                - update gradient
                - agent's updated weights

        Returns
        -------
        returns : array_like
            An array of the returns for each episode of this epoch.
        gradient : array_like, optional
            Gradient of the update. Only returned if 'return_info=True'.   
        new_weights : dict[str, array_like], optional
            Agent's new set of weights after the update. Only returned if 
            'return_info=True'.
        '''
        pass

    def learn(
        self,
        n_epochs: int,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        throw_on_exception: bool = False,
        logger: logging.Logger = None,
        return_info: bool = True
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]
    ]:
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        n_episodes : int
            Number of training episodes per epoch.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after each epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : logging.Logger, optional
            For logging purposes.

        throw_on_exception : bool, optional
            When a training exception occurs, if `throw_on_exception=True`,
            then the exception is fired again and training fails. Otherwise; 
            the training is prematurely stopped and returned.
        return_info : bool, optional
            Whether to return additional information for each epoch update:
                - a list of update gradients
                - a list of agent's updated weights after each update

        Returns
        -------
        returns : array_like
            An array of the returns for each episode in each epoch.
        gradient : list[array_like], optional
            Gradients of each update. Only returned if 'return_info=True'.   
        new_weights : list[dict[str, array_like]], optional
            Agent's new set of weights after each update. Only returned if 
            'return_info=True'.
        '''
        logger = logger or logging.getLogger('dummy')
        results = []

        for e in range(n_epochs):
            self._epoch_n = e  # just for logging
            try:
                results.append(
                    self.learn_one_epoch(
                        n_episodes=n_episodes,
                        perturbation_decay=perturbation_decay,
                        seed=None if seed is None else seed + n_episodes * e,
                        logger=logger,
                        return_info=return_info)
                )
            except (MPCSolverError, UpdateError) as ex:
                if throw_on_exception:
                    raise ex
                logger.error(f'Suppressing agent \'{self.name}\': {ex}')
                break

        if not results:
            return (np.nan, [], []) if return_info else np.nan
        if not return_info:
            return np.stack(results, axis=0)
        returns, grads, weightss = list(zip(*results))
        return np.stack(returns, axis=0), grads, weightss

    def _init_learnable_pars(
        self, init_pars: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        '''Initializes the learnable parameters of the MPC.'''
        required_pars = sorted(set(self._Q.pars).intersection(
            self._V.pars).difference({'x0', 'xf'}).difference(self.fixed_pars))
        self.weights = RLParameterCollection(
            *(RLParameter(
                name, *init_pars[name], self.V.pars[name], self.Q.pars[name])
              for name in required_pars)
        )

    def _init_learning_rate(self) -> None:
        cfg = self.config
        if cfg is None or not hasattr(cfg, 'lr'):
            return
        lr = cfg.lr
        n_pars, n_theta = len(self.weights), self.weights.n_theta
        if np.isscalar(lr):
            lr = np.full((n_theta,), lr)
        else:
            lr = np.asarray(cfg.lr).squeeze()
            if lr.size == n_pars and lr.size != n_theta:
                lr = np.concatenate(
                    [np.full(p.size, r) for p, r in zip(self.weights, lr)])
        assert lr.shape == (n_theta,), 'Learning rate must have the same ' \
            'size as the learnable parameter vector.'
        cfg.__dict__['lr'] = lr

    def _merge_mpc_pars_callback(self) -> dict[str, np.ndarray]:
        return self.weights.values(as_dict=True)

    @staticmethod
    def _get_percentage_bounds(
        theta: np.ndarray,
        theta_bounds: np.ndarray,
        max_perc_update: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_delta = np.maximum(np.abs(max_perc_update * theta), 0.1)
        lb = np.maximum(theta_bounds[:, 0], theta - max_delta)
        ub = np.minimum(theta_bounds[:, 1], theta + max_delta)
        return lb, ub
