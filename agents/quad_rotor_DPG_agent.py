import numpy as np
import casadi as cs
from threading import Thread
from queue import Queue
from dataclasses import dataclass
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
    gamma: float = 1  # it is episodic...
    lr: float = 1e-6

    @property
    def init_pars(self) -> dict[str, float | np.ndarray]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


transp = lambda o: np.transpose(o, axes=(0, 2, 1))


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

        # initialize the replay memory. Per each episode the memory saves an
        # array of Phi(s), Psi(s,a), L(s,a), dpidtheta(s) and weights v.
        self.replay_memory = ReplayMemory[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]](
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

        # consolidate Phi(s), Psi(s,a), L(s,a), dpidtheta(s) into arrays
        Phi, Psi, L, dpidtheta = tuple(
            np.stack(o, axis=0) for o in zip(*self._episode_buffer))

        # compute this episode's weights v via LSTD
        A = (
            Phi[:-1] @ transp(Phi[:-1] - self.config.gamma * Phi[1:])
        ).sum(axis=0)
        b = (Phi[:-1] * L[:-1]).sum(axis=0)
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(A, b, rcond=None)[0]
        v = v.reshape(1, -1, 1)

        # save this episode to memory and clear buffer
        self.replay_memory.append((Phi, Psi, L, dpidtheta, v))
        self._episode_buffer.clear()

    def update(self) -> np.ndarray:
        # sample the replay memory
        sample = list(self.replay_memory.sample(
            self.config.replay_sample_size, self.config.replay_include_last))
        m = len(sample)

        # average weights over m episodes
        v = sum(v for _, _, _, _, v in sample) / m

        # compute weights w via LSTD and averaging over m episodes
        w = 0
        for Phi, Psi, L, _, _ in sample:
            A = (Psi[:-1] @ transp(Psi[:-1])).sum(axis=0)
            b = (
                (L[:-1] +
                 transp(self.config.gamma * Phi[1:] - Phi[:-1]) @ v) * Psi[:-1]
            ).sum(axis=0)
            w += np.linalg.solve(A, b)
        w = w.reshape(1, -1, 1) / m

        # compute episode's update
        dJdtheta = 1 / m * sum((dpidtheta @ transp(dpidtheta) @ w).sum(axis=0)
                               for _, _, _, dpidtheta, _ in sample).flatten()

        # perform update
        c = self.config.lr * dJdtheta
        theta = self.weights.values()
        bounds = self.weights.bounds()
        sol = self._solver(lbx=bounds[:, 0], ubx=bounds[:, 1], x0=theta - c,
                           p=np.concatenate((theta, c)))
        theta_new: np.ndarray = sol['x'].full().flatten()

        # update weights
        self.weights.update_values(theta_new)
        return dJdtheta

    def _init_symbols(self) -> None:
        '''Computes symbolical derivatives needed for DPG updates.'''
        # gather some variables
        theta = self.weights.symV()
        R, _, y = self.V.kkt_conditions

        # compute the derivative of the policy (pi) w.r.t. the mpc pars (theta)
        self._dRdtheta = cs.simplify(cs.jacobian(R, theta)).T
        self._dRdy = cs.simplify(cs.jacobian(R, y)).T
        self._dydu0 = cs.DM(cs.jacobian(y, self.V.vars['u'][:, 0])).full()

        # compute baseline function approximating the value function with
        # monomials as basis
        x: cs.SX = cs.SX.sym('x', self.env.nx, 1)
        y: cs.SX = cs.vertcat(
            1,
            x,
            *(cs_prod(x**p) for p in monomial_powers(x.size1(), 2)))
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

    def _do_work(self) -> None:
        '''Actual method executed by the worker thread.'''
        while True:
            s, a, L, sol = self._work_queue.get()

            # compute the derivative of the policy w.r.t. the mpc weights theta
            dRdy = sol.value(self._dRdy)
            dRdtheta = sol.value(self._dRdtheta)
            q = np.linalg.solve(dRdy, self._dydu0)
            dpidtheta = -dRdtheta @ q
            assert (dRdy @ q - self._dydu0).max() <= 1e-10, \
                'Linear solver failed.'

            # compute Phi
            Phi = self._Phi(s).full()

            # compute Psi
            u_opt = sol.value(sol.vals['u'][:, 0])  # i.e., V(s), pi(s)
            Psi = (dpidtheta @ (a - u_opt)).reshape(-1, 1)

            # reshape L into an array
            L = np.array(L).reshape(1, 1)

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
