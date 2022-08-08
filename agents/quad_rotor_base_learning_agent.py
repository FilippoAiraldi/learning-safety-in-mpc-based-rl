import numpy as np
from abc import ABC, abstractmethod
from agents.quad_rotor_base_agent import QuadRotorBaseAgent
from envs.quad_rotor_env import QuadRotorEnv
from mpc.generic_mpc import Solution
from mpc.quad_rotor_mpc import QuadRotorMPC, QuadRotorMPCConfig
from mpc.wrappers import DifferentiableMPC


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
        mpc_config: dict | QuadRotorMPCConfig = None,
        seed: int = None
    ) -> None:
        super().__init__(
            env, agentname, init_pars, fixed_pars, mpc_config, seed)
        self._V = DifferentiableMPC[QuadRotorMPC](self._V)
        self._Q = DifferentiableMPC[QuadRotorMPC](self._Q)

    @property
    def V(self) -> DifferentiableMPC[QuadRotorMPC]:
        '''Gets the V action-value function approximation MPC scheme.'''
        return self._V

    @property
    def Q(self) -> DifferentiableMPC[QuadRotorMPC]:
        '''Gets the Q action-value function approximation MPC scheme.'''
        return self._Q

    @abstractmethod
    def save_transition(self, s: np.ndarray, a: np.ndarray, L: float,
                        solution: Solution) -> None:
        '''
        Schedules the current time-step data to be processed and saved into the
        experience replay memory.

        Parameters 
        ----------
        s, a, L : array_like, array_like, float
            The SAR tuple which stands for
                - State for which the MPC V(s) is run
                - Action taken as a result of computing V(s) (not necessarily the 
                  optimal one, e.g., due to exploration).
                - Reward/cost incurred subsequently.
        solution : Solution
            Solution object of V(s), where s is the current state.
        '''
        pass

    @abstractmethod
    def consolidate_episode_experience(self) -> None:
        '''
        At the end of an episode, computes the remaining operations and 
        saves results to the replay memory as arrays.
        '''
        pass

    @abstractmethod
    def update(self) -> np.ndarray:
        '''Updates the MPC function approximation's weights based on the 
        information stored in the replay memory.

        Returns
        -------
        gradient : array_like
            Gradient of the update.
        '''
        pass
