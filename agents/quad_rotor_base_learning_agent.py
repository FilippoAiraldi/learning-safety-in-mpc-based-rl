import logging
import numpy as np
from abc import ABC, abstractmethod
from agents.quad_rotor_base_agent import QuadRotorBaseAgent
from envs.quad_rotor_env import QuadRotorEnv
from mpc.quad_rotor_mpc import QuadRotorMPC, QuadRotorMPCConfig
from mpc.wrappers import DifferentiableMPC
from typing import Union


class QuadRotorBaseLearningAgent(QuadRotorBaseAgent, ABC):
    '''
    Abstract base agent class that renders the two MPC function approximators
    Q and V differentiable, such that their parameters can be learnt. 
    '''

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        init_pars: dict[str, np.ndarray] = None,
        fixed_pars: dict[str, np.ndarray] = None,
        mpc_config: Union[dict, QuadRotorMPCConfig] = None,
        seed: int = None
    ) -> None:
        super().__init__(
            env, agentname, init_pars, fixed_pars, mpc_config, seed)
        self._V = DifferentiableMPC[QuadRotorMPC](self._V)
        self._Q = DifferentiableMPC[QuadRotorMPC](self._Q)
        self._epoch_n = None  # keeps track of epoch number just for logging

    @property
    def V(self) -> DifferentiableMPC[QuadRotorMPC]:
        '''Gets the V action-value function approximation MPC scheme.'''
        return self._V

    @property
    def Q(self) -> DifferentiableMPC[QuadRotorMPC]:
        '''Gets the Q action-value function approximation MPC scheme.'''
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
        raises: bool = True,
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
        raises : bool, optional
            Whether to raise an exception when the MPC solver fails.
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
        n_train_epochs: int,
        n_train_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        raises: bool = True,
        return_info: bool = True
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]
    ]:
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_train_epochs : int
            Number of training epochs.
        n_train_episodes : int
            Number of training episodes per epoch.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after each epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : logging.Logger, optional
            For logging purposes.
        raises : bool, optional
            Whether to raise an exception when the MPC solver fails.
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

        for e in range(n_train_epochs):
            self._epoch_n = e  # just for logging

            results.append(
                self.learn_one_epoch(
                    n_episodes=n_train_episodes,
                    perturbation_decay=perturbation_decay,
                    seed=None if seed is None else seed + n_train_episodes * e,
                    logger=logger,
                    raises=raises,
                    return_info=return_info)
            )

        if not return_info:
            return np.stack(results, axis=0)
        returns, grads, weightss = list(zip(*results))
        return np.stack(returns, axis=0), grads, weightss

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

    @staticmethod
    def _make_seed_list(seed: Union[int, list[int]], n: int) -> list[int]:
        if seed is None:
            return [None] * n
        if isinstance(seed, int):
            return [seed + i for i in range(n)]
        assert len(seed) == n, 'Seed sequence with invalid length.'
        return seed
