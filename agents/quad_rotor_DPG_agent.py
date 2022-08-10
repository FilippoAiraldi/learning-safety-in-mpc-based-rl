import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass
from envs import QuadRotorEnvConfig, QuadRotorEnv
from mpc import Solution, QuadRotorMPCConfig
from util import monomial_powers, cs_prod


@dataclass(frozen=True)
class QuadRotorDPGAgentConfig:
    # initial RL pars
    # model
    init_g: float = 9.81
    init_thrust_coeff: float = 1.0
    init_pitch_d: float = 14
    init_pitch_dd: float = 10
    init_pitch_gain: float = 14
    init_roll_d: float = 6  # 40
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
    lr: float = 1e-6
    clip_grad_norm: float = 1e6

    @property
    def init_pars(self) -> dict[str, float | np.ndarray]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


trsp = lambda o: np.transpose(o, axes=(0, 2, 1))


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
        self.replay_memory = ReplayMemory[tuple[np.ndarray, ...]](
            maxlen=agent_config.replay_maxlen, seed=seed)

        # initialize symbols for derivatives to be used later and worker to
        # compute these numerically. Also initialize the QP solver used to
        # compute updates
        self._init_symbols()
        self._init_work()
        self._init_qp_solver()

    def save_transition(
        self, sars: tuple[np.ndarray, ...], solution: Solution
    ) -> None:
        self._episode_buffer.append((*sars, solution))

    def consolidate_episode_experience(self) -> None:
        if len(self._episode_buffer) == 0:
            return
            
        S, A, L, S_next, sols = tuple(
            np.stack(o, axis=0) for o in zip(*self._episode_buffer))
        K = sols.size
        L = L.reshape(-1, 1, 1)

        # compute Phi (value function approximation basis functions)
        Phi = self._Phi(S.T).full().T.reshape(K, -1, 1)
        Phi_next = self._Phi(S_next.T).full().T.reshape(K, -1, 1)

        # compute Psi
        #
        dRdy, dRdtheta, U_opt = [], [], []
        for sol in sols:
            dRdy.append(sol.value(self._dRdy))
            dRdtheta.append(sol.value(self._dRdtheta))
            U_opt.append(sol.vals['u'][:, 0])
        dRdy = np.stack(dRdy, axis=0)
        dRdtheta = np.stack(dRdtheta, axis=0)
        q = np.linalg.solve(dRdy, np.tile(self._dydu0, (sols.size, 1, 1)))
        dpidtheta = -dRdtheta @ q
        #
        U_opt = np.stack(U_opt, axis=0)
        Psi = dpidtheta @ (A - U_opt).reshape(K, -1, 1)

        # compute this episode's weights v via LSTD
        A = (Phi @ trsp(Phi - self.config.gamma * Phi_next)).sum(axis=0)
        b = (Phi * L).sum(axis=0)
        try:
            v = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(A, b, rcond=None)[0]

        # save this episode to memory and clear buffer
        self.replay_memory.append((Phi, Phi_next, Psi, L, dpidtheta, v))
        self._episode_buffer.clear()

    def update(self) -> np.ndarray:
        # sample the memory. Each item in the sample comes from one episode
        cfg = self.config
        sample = list(self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last))
        m = len(sample)

        # average weights over m episodes
        v = sum(v for _, _, _, _, _, v in sample) / m

        # compute weights w via LSTD and averaging over m episodes
        w = 0
        for Phi, Phi_next, Psi, L, _, _ in sample:
            A = (Psi @ trsp(Psi)).sum(axis=0)
            b = ((L + trsp(cfg.gamma * Phi - Phi_next) @ v) * Psi).sum(axis=0)
            w += np.linalg.solve(A, b)
        w /= m

        # compute episode's update
        dJdtheta = sum((dpidtheta @ trsp(dpidtheta) @ w).sum(axis=0)
                       for _, _, _, _, dpidtheta, _ in sample).flatten() / m

        # clip gradient if requested
        if cfg.clip_grad_norm is None:
            c = cfg.lr * dJdtheta
        else:
            clip_coef = min(
                cfg.clip_grad_norm / (np.linalg.norm(dJdtheta) + 1e-6), 1.0)
            c = (cfg.lr * clip_coef) * dJdtheta

        # run QP solver
        theta = self.weights.values()
        bounds = self.weights.bounds()
        sol = self._solver(lbx=bounds[:, 0], ubx=bounds[:, 1], x0=theta - c,
                           p=np.concatenate((theta, c)))
        assert self._solver.stats()['success'], 'RL update failed.'
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

    def _init_work(self) -> None:
        self._episode_buffer: \
            list[tuple[np.ndarray, np.ndarray, float, np.ndarray, Solution]] \
            = []

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
