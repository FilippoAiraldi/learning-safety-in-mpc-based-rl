import numpy as np
import casadi as cs
from threading import Thread
from queue import Queue
from itertools import pairwise
from dataclasses import dataclass
from scipy.linalg import lstsq
from envs import QuadRotorEnvConfig, QuadRotorEnv
from mpc import Solution, QuadRotorMPCConfig
from util import monomial_powers, cs_prod
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory


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

    # RL parameters
    gamma: float = 0.97
    lr: float = 1e0

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

        # during learning, DPG must always perturb the action in order to learn
        self.perturbation_chance = 1.0

        # initialize the replay memory. It saves one object per episode
        #       sum_k=1^K { dpidtheta(s_k) * dpidtheta(s_k).T w }
        # which represents the strenght of the update from one episode.
        self.replay_memory = ReplayMemory[np.ndarray](
            maxlen=agent_config.replay_maxlen, seed=seed)

        # initialize symbols for derivatives to be used later and worker to
        # compute these numerically. Also initialize the QP solver used to
        # compute updates
        self._init_symbols()
        self._init_worker()
        self._init_qp_solver()

    def save_transition(self, sar: tuple[np.ndarray, np.ndarray, float],
                        solution: Solution) -> None:
        if not self._worker.is_alive():
            self._worker.start()
        self._work_queue.put((*sar, solution))

    def consolidate_episode_experience(self) -> None:
        # wait for the current episode's transitions to be fully saved
        self._work_queue.join()
        buffer = self._episode_buffer

        # compute episode's weights v via least-squares
        A, b = 0, 0
        for (Phi, _, L, _), (Phi_n, _, _, _) in pairwise(buffer):
            A += Phi @ (Phi - self.config.gamma * Phi_n).T
            b += Phi * L
        v = lstsq(A, b, lapack_driver='gelsy')[0]

        # compute episode's weights w via least-squares
        A, b = 0, 0
        for (Phi, Psi, L, _), (Phi_n, _, _, _) in pairwise(buffer):
            A += Psi @ Psi.T
            b += (L + (self.config.gamma * Phi_n - Phi).T @ v) * Psi
        w = lstsq(A, b, lapack_driver='gelsy')[0]

        # compute episode's update
        update = sum(
            dpidtheta @ dpidtheta.T @ w for _, _, _, dpidtheta in buffer)

        # save this episode's update to memory and clear buffer
        self.replay_memory.append(update.flatten())
        self._episode_buffer.clear()

    def update(self) -> None:
        # sample the replay memory
        sample = list(self.replay_memory.sample(
            self.config.replay_sample_size, self.config.replay_include_last))
        c = self.config.lr * np.mean(sample, axis=0)

        # perform update
        theta = self.weights.values()
        bounds = self.weights.bounds()
        sol = self._solver(lbx=bounds[:, 0], ubx=bounds[:, 1], x0=theta - c,
                           p=np.concatenate((theta, c)))
        theta_new: np.ndarray = sol['x'].full().flatten()

        # update weights
        self.weights.update_values(theta_new)

    def _init_symbols(self) -> None:
        '''Computes symbolical derivatives needed for DPG updates.'''
        # gather some variables
        theta = self.weights.symV()
        R, y, _, self._g_ineq_all = self.V.kkt_conditions

        # compute the derivative of the policy (pi) w.r.t. the mpc pars (theta)
        self._dRdtheta = cs.simplify(cs.jacobian(R, theta)).T
        self._dRdy = cs.simplify(cs.jacobian(R, y)).T
        self._dydu0 = cs.simplify(cs.jacobian(y, self.V.vars['u'][:, 0]))
        # NOTE: cannot go further with symbolic computations as they get too
        # heavy due to matrix inversion of dRdy, which tends to be quite big.

        # compute baseline function approximating the value function with
        # monomials as basis
        x: cs.SX = cs.SX.sym('x', self.env.nx, 1)
        y: cs.SX = cs.vertcat(
            1, x, *(cs_prod(x**p) for p in monomial_powers(x.size1(), 2)))
        self._Phi = cs.Function('Phi', [x], [y], ['s'], ['Phi(s)'])

    def _init_worker(self) -> None:
        '''initialize worker thread tasked with computing the derivatives per
        each transition (computationally heavy). Results for the current
        episode are then saved in the buffer.'''
        self._worker = Thread(daemon=True, target=self._do_work)
        self._work_queue = \
            Queue[tuple[np.ndarray, np.ndarray, float, Solution]]()
        self._episode_buffer: \
            list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []

        # pre-compute some constants for the worker
        self._offset = self.V.nx + self.V.ng_eq
        self._dLdw_and_g_eq_idx = np.arange(self._offset)

    def _do_work(self) -> None:
        '''Actual method executed by the worker thread.'''
        while True:
            s, a, L, sol = self._work_queue.get()

            # compute derivative of policy pi w.r.t. the MPC parameters theta
            # get active constraints only
            idx = np.concatenate((
                self._dLdw_and_g_eq_idx,
                np.where(np.isclose(sol.value(self._g_ineq_all), 0))[0] +
                self._offset
            ))

            # compute the derivative of the policy w.r.t. the mpc weights theta
            dRdy = sol.value(self._dRdy[idx, idx])
            dydu0 = sol.value(self._dydu0[idx, :])
            dRdtheta = sol.value(self._dRdtheta[:, idx])
            q = lstsq(dRdy, dydu0, lapack_driver='gelsy')[0]
            dpidtheta = -dRdtheta @ q
            assert (dRdy @ q - dydu0).max() <= 1e-10, 'Linear solver failed.'
            # NOTE: Other methods to solving the linear system are
            # 1. q = np.linalg.solve(dRdy, dydu0)
            # 2. U, S, VT = np.linalg.svd(dRdy)
            #    y = np.linalg.solve(np.diag(S), U.T @ dydu)
            #    q = np.linalg.solve(VT, y)

            # compute Phi
            Phi = self._Phi(s).full()

            # compute Psi
            u_opt = sol.value(sol.vals['u'][:, 0])  # i.e., V(s), pi(s)
            Psi = (dpidtheta @ (a - u_opt)).reshape(-1, 1)

            # save in temporary buffer
            self._episode_buffer.append((Phi, Psi, L, dpidtheta))
            self._work_queue.task_done()

    def _init_qp_solver(self) -> None:
        n = sum(self.weights.sizes())

        # prepare symbols
        theta: cs.SX = cs.SX.sym('theta', n, 1)
        theta_new: cs.SX = cs.SX.sym('theta+', n, 1)
        c: cs.SX = cs.SX.sym('c', n, 1)

        # compute objective
        dtheta = theta_new - theta
        f = 0.5 * dtheta.T @ dtheta + c.T @ dtheta

        # prepare solver
        qp = {'x': theta_new, 'p': cs.vertcat(theta, c), 'f': f}
        opts = {'print_iter': False, 'print_header': False}
        self._solver = cs.qpsol(f'qpsol_{self.name}', 'qrqp', qp, opts)
