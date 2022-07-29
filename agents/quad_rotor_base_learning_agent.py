from abc import ABC
import numpy as np
from agents.quad_rotor_base_agent import QuadRotorBaseAgent
from envs.quad_rotor_env import QuadRotorEnv
from mpc.quad_rotor_mpc import QuadRotorMPCConfig
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
        self._V = DifferentiableMPC(self._V)
        self._Q = DifferentiableMPC(self._Q)

    @property
    def V(self) -> DifferentiableMPC:
        '''Gets the V action-value function approximation MPC scheme.'''
        return self._V

    @property
    def Q(self) -> DifferentiableMPC:
        '''Gets the Q action-value function approximation MPC scheme.'''
        return self._Q
