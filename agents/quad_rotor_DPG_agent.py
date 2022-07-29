import numpy as np
import casadi as cs
from envs.quad_rotor_env import QuadRotorEnvConfig, QuadRotorEnv
from mpc.quad_rotor_mpc import QuadRotorMPCConfig
from agents.quad_rotor_base_agent import QuadRotorBaseAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass


@dataclass(frozen=True)
class QuadRotorDPGAgentConfig:
    # initial RL pars
    # model
    init_g: float = 9.81
    init_thrust_coeff: float = 1.0
    init_pitch_d: float = 14
    init_pitch_dd: float = 10
    init_pitch_gain: float = 14
    init_roll_d: float = 6
    init_roll_dd: float = 7
    init_roll_gain: float = 9
    # cost
    init_w_L: np.ndarray = 1
    init_w_V: np.ndarray = 1
    init_w_s: np.ndarray = 1e2
    init_w_s_f: np.ndarray = 1e2
    init_xf: np.ndarray = \
        QuadRotorEnvConfig.__dataclass_fields__['xf'].default_factory()
    # others
    init_backoff: float = 0.05

    # experience replay parameters
    replay_maxlen: float = 30 * 10  # more or less 10 episodes
    replay_sample_size: float = 0.5  # sample from 5 out of 10 episodes
    replay_include_last: float = 0.2  # include in the sample the last 2 episodes

    @property
    def init_pars(self) -> dict[str, float | np.ndarray]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


@dataclass(frozen=True)
class Derivatives:
    dRdtheta: cs.SX
    dRdy: cs.SX
    dydu0: cs.SX


class QuadRotorDPGAgent(QuadRotorBaseAgent):
    '''
    Deterministic Policy Gradient based RL agent for the quad rotor 
    environment. The agent adapts its MPC parameters/weights by policy gradient
    methods, with the goal of improving performance/reducing cost of each
    episode.

    The policy gradient-based RL update exploits a replay memory to spread out 
    the gradient noise.
    '''

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: dict | QuadRotorDPGAgentConfig = None,
        mpc_config: dict | QuadRotorMPCConfig = None,
    ) -> None:
        '''
        Initializes a Deterministic-Policy-Gradient agent for the quad rotor 
        env.

        Parameters
        ----------
        config : dict, QuadRotorDPGAgentConfig
            A set of parameters for the quadrotor agent. If not given, the 
            default ones are used.
        *args, **kwargs
            See QuadRotorBaseAgent.
        '''
        if agent_config is None:
            agent_config = QuadRotorDPGAgentConfig()
        elif isinstance(agent_config, dict):
            keys = QuadRotorDPGAgentConfig.__dataclass_fields__.keys()
            agent_config = QuadRotorDPGAgentConfig(
                **{k: agent_config[k] for k in keys if k in agent_config})
        self.config = agent_config

        # initialize base class
        super().__init__(env, agentname=agentname,
                         init_pars=self.config.init_pars,
                         fixed_pars={'perturbation': np.nan},
                         mpc_config=mpc_config)

        # initialize the replay memory
        self.replay_memory = ReplayMemory(maxlen=agent_config.replay_maxlen)

        # compute the symbolical derivatives needed to perform the DPG updates
        # gather some variables
        theta = self.weights.symV()  # vector of MPC parameters
        R, y = self.V.kkt_matrix    # KKT matrix and primal-dual variables

        # compute the derivative of the policy (pi) w.r.t. the mpc pars (theta)
        dRdtheta = cs.simplify(cs.jacobian(R, theta)).T
        dRdy = cs.simplify(cs.jacobian(R, y)).T
        dydu0 = cs.simplify(cs.jacobian(y, self.V.vars['u'][:, 0]))
        self.derivatives = Derivatives(dRdtheta, dRdy, dydu0)
        # dpidtheta = -dRdtheta @ cs.inv_minor(dRdy) @ dydu0 CANNOT BE DONE SYMBOLICALLY

        # import time
        # t0 = time.perf_counter()
        # tf = time.perf_counter() - t0
