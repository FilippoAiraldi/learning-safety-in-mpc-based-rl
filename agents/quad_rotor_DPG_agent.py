import numpy as np
import casadi as cs
from envs import QuadRotorEnvConfig, QuadRotorEnv
from mpc import Solution, QuadRotorMPCConfig
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass
from scipy.linalg import lstsq


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
    replay_maxlen: float = 20  # 20 episodes
    replay_sample_size: float = 10  # sample from 10 out of 20 episodes
    replay_include_last: float = 5  # include in the sample the last 5 episodes

    @property
    def init_pars(self) -> dict[str, float | np.ndarray]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


class QuadRotorDPGAgent(QuadRotorBaseLearningAgent):
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
        seed: int = None
    ) -> None:
        '''
        Initializes a Deterministic-Policy-Gradient agent for the quad rotor 
        env.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the DPG agent.
        agentname : str, optional
            Name of the DPG agent.
        agent_config : dict, QuadRotorDPGAgentConfig
            A set of parameters for the quadrotor DPG agent. If not given, the 
            default ones are used.
        mpc_config : dict, QuadRotorMPCConfig
            A set of parameters for the agent's MPC. If not given, the default 
            ones are used.
        seed : int, optional
            Seed for the random number generator.
        '''
        if agent_config is None:
            agent_config = QuadRotorDPGAgentConfig()
        elif isinstance(agent_config, dict):
            keys = QuadRotorDPGAgentConfig.__dataclass_fields__.keys()
            agent_config = QuadRotorDPGAgentConfig(
                **{k: agent_config[k] for k in keys if k in agent_config})
        self.config = agent_config
        super().__init__(env, agentname=agentname,
                         init_pars=self.config.init_pars,
                         fixed_pars={'perturbation': np.nan},
                         mpc_config=mpc_config, seed=seed)

        # initialize the replay memory
        self.replay_memory = ReplayMemory[np.ndarray](
            maxlen=agent_config.replay_maxlen, seed=seed)

        # compute the symbolical derivatives needed to perform the DPG updates
        # gather some variables
        theta = self.weights.symV()
        R, y, _, g_ineq_all = self.V.kkt_conditions

        # compute the derivative of the policy (pi) w.r.t. the mpc pars (theta)
        dRdtheta = cs.simplify(cs.jacobian(R, theta)).T
        dRdy = cs.simplify(cs.jacobian(R, y)).T
        dydu0 = cs.simplify(cs.jacobian(y, self.V.vars['u'][:, 0]))
        self._sym: dict[str, cs.SX] = {
            'g_ineq_all': g_ineq_all,
            'dRdy': dRdy,
            'dydu0': dydu0,
            'dRdtheta': dRdtheta
        }
        # NOTE: cannot go further with symbolic computations as they get too
        # heavy due to matrix inversion of dRdy, which tends to be quite big.

        # initialize small internal buffer of solutions, one per each timestep
        self._buffer: list[Solution] = []

    def save_transition(self, solV: Solution) -> None:
        '''Saves the solution at the current time-step.'''
        self._buffer.append(solV)

    def consolidate_episode_experience(self) -> None:
        # extract the symbols
        g_ineq_all = self._sym['g_ineq_all']
        dRdy = self._sym['dRdy']
        dydu0 = self._sym['dydu0']
        dRdtheta = self._sym['dRdtheta']

        # pre-compute constants
        offset = self.V.nx + self.V.ng_eq
        dLdw_and_g_eq_idx = np.arange(offset)

        # for each solution, make the computations
        # TODO: check if parallelizing this loop helps
        dpidthetas = []
        for sol in self._buffer:
            # get active constraints only
            idx2 = np.where(np.concatenate((
                [True] * offset,
                np.isclose(sol.value(g_ineq_all), 0)
            )))[0]
            idx = np.concatenate((
                dLdw_and_g_eq_idx,
                offset + np.where(np.isclose(sol.value(g_ineq_all), 0))[0]
            ))
            # TODO: check these are equal
            assert (idx == idx2).all()

            # compute the derivative of the policy w.r.t. the mpc weights theta
            dRdy = sol.value(dRdy[idx, idx])
            dydu0 = sol.value(dydu0[idx, :])
            dRdtheta = sol.value(dRdtheta[:, idx])
            dpidthetas.append(
                -dRdtheta @ lstsq(dRdy, dydu0, lapack_driver='gelsy')[0])

        # save to replay memory
        # TODO: compact dpidthetas to an array
        # TODO: save also the state in which the solution was computed

        # finally, clear the temp buffer
        self._buffer.clear()

# #
# # q0 = np.linalg.solve(dRdys[-1], dydu0s[-1])
# #
# q1 = lstsq(dRdys[-1], dydu0s[-1], lapack_driver='gelsy')[0]
# #
# U2, S2, V2T = np.linalg.svd(dRdys[-1])
# y2 = np.linalg.solve(np.diag(S2), U2.T @ dydu0s[-1])
# q2 = np.linalg.solve(V2T, y2)
# #
