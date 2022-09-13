import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass
from envs import QuadRotorEnv
from logging import Logger
from mpc import Solution, QuadRotorMPCConfig
from scipy.linalg import lstsq
from typing import Union
from util import monomial_powers, cs_prod


@dataclass(frozen=True)
class QuadRotorLSTDDPGAgentConfig:
    # initial RL pars
    # model
    # NOTE: initial values were made closer to real in an attempt to check if
    # learning happens. Remember to reset them to more difficult values at some
    # point
    init_g: float = 9.81
    init_thrust_coeff: float = 1.2
    init_pitch_d: float = 12
    init_pitch_dd: float = 9
    init_pitch_gain: float = 11
    init_roll_d: float = 8
    init_roll_dd: float = 7
    init_roll_gain: float = 9
    # cost
    init_w_Lx: np.ndarray = 1e1
    init_w_Lu: np.ndarray = 1e0
    init_w_Ls: np.ndarray = 1e2
    init_w_Tx: np.ndarray = 1e1
    init_w_Tu: np.ndarray = 1e0
    init_w_Ts: np.ndarray = 1e2

    # experience replay parameters
    replay_maxlen: float = 20  # 20 episodes
    replay_sample_size: float = 10  # sample from 10 out of 20 episodes
    replay_include_last: float = 5  # include in the sample the last 5 episodes

    # RL parameters
    lr: float = 1e-1
    max_perc_update: float = 1 / 4
    clip_grad_norm: float = None

    @property
    def init_pars(self) -> dict[str, Union[float, np.ndarray]]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


class QuadRotorLSTDDPGAgent(QuadRotorBaseLearningAgent):
    '''
    Least-Squares Temporal Difference-based Deterministic Policy Gradient RL
    agent for the quad rotor environment. The agent adapts its MPC
    parameters/weights by policy gradient methods, averaging over batches of
    episodes via Least-Squares, with the goal of improving performance/reducing
    cost of each episode.

    The policy gradient-based RL update exploits a replay memory to spread out
    the gradient noise.
    '''

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: Union[dict, QuadRotorLSTDDPGAgentConfig] = None,
        mpc_config: Union[dict, QuadRotorMPCConfig] = None,
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
            agent_config = QuadRotorLSTDDPGAgentConfig()
        elif isinstance(agent_config, dict):
            keys = QuadRotorLSTDDPGAgentConfig.__dataclass_fields__.keys()
            agent_config = QuadRotorLSTDDPGAgentConfig(
                **{k: agent_config[k] for k in keys if k in agent_config})
        self.config = agent_config
        super().__init__(env, agentname=agentname,
                         init_pars=self.config.init_pars,
                         fixed_pars={'perturbation': np.nan},
                         mpc_config=mpc_config, seed=seed)

        # during learning, DPG must always perturb the action in order to learn
        self.perturbation_chance = 1.0
        self.perturbation_strength = 1e-1

        # initialize the replay memory. Per each episode the memory saves an
        # array of Phi(s), Psi(s,a), L(s,a), dpidtheta(s) and weights v. Also
        # initialize the episode buffer which temporarily stores values before
        # batch-processing them into the replay memory
        self.replay_memory = ReplayMemory[tuple[np.ndarray, ...]](
            maxlen=agent_config.replay_maxlen, seed=seed)
        self._episode_buffer: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray,
                  Solution]] = []

        # initialize symbols for derivatives to be used later. Also initialize
        # the QP solver used to compute updates
        self._init_symbols()
        self._init_qp_solver()

    def save_transition(
        self,
        state: np.ndarray,
        action_taken: np.ndarray,
        optimal_action: np.ndarray,
        cost: np.ndarray,
        new_state: np.ndarray,
        solution: Solution
    ) -> None:
        item = (state, action_taken, optimal_action, cost, new_state, solution)
        self._episode_buffer.append(item)

    def consolidate_episode_experience(self) -> None:
        if len(self._episode_buffer) == 0:
            return

        # stack everything in arrays and compute derivatives
        S, L, S_next, E = [], [], [], []
        dRdy, dRdtheta = [], []
        for s, a, a_opt, r, s_next, sol in self._episode_buffer:
            S.append(s)
            L.append(r)
            S_next.append(s_next)
            E.append(a - a_opt)
            dRdy.append(sol.value(self._dRdy))
            dRdtheta.append(sol.value(self._dRdtheta))
        K = len(S)
        S = np.stack(S, axis=0)
        L = np.stack(L, axis=0).reshape(K, 1)
        S_next = np.stack(S_next, axis=0)
        E = np.stack(E, axis=0)
        dRdy = np.stack(dRdy, axis=0)
        dRdtheta = np.stack(dRdtheta, axis=0)

        # normalize reward
        # L = (L - L.mean()) / (L.std() + 1e-10)

        # compute Phi (value function approximation basis functions)
        Phi = self._Phi(S.T).full().T
        Phi_next = self._Phi(S_next.T).full().T

        # compute Psi
        q = np.linalg.solve(dRdy, np.tile(self._dydu0, (K, 1, 1)))
        dpidtheta = -dRdtheta @ q
        Psi = (dpidtheta @ E.reshape(K, -1, 1)).squeeze()

        # compute this episode's weights v via LSTD
        v = lstsq(Phi - Phi_next, L,
                  lapack_driver='gelsy')[0]

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
            A = Psi.T @ Psi
            b = Psi.T @ (L + (Phi_next - Phi) @ v)
            w += lstsq(A, b, lapack_driver='gelsy')[0]
        w /= m

        # compute episode's update
        dJdtheta = sum(
            (dpidth @ np.transpose(dpidth, axes=(0, 2, 1)) @ w).sum(axis=0)
            for _, _, _, _, dpidth, _ in sample).flatten() / m

        # clip gradient if requested
        if cfg.clip_grad_norm is None:
            c = cfg.lr * dJdtheta
        else:
            clip_coef = min(
                cfg.clip_grad_norm / (np.linalg.norm(dJdtheta) + 1e-6), 1.0)
            c = (cfg.lr * clip_coef) * dJdtheta

        # compute bounds on parameter update
        theta = self.weights.values()
        bounds = self.weights.bounds()
        max_delta = np.maximum(np.abs(cfg.max_perc_update * theta), 0.1)
        lb = np.maximum(bounds[:, 0], theta - max_delta)
        ub = np.minimum(bounds[:, 1], theta + max_delta)

        # run QP solver
        sol = self._solver(lbx=lb, ubx=ub, x0=theta - c,
                           p=np.concatenate((theta, c)))
        assert self._solver.stats()['success'], 'RL update failed.'
        theta_new: np.ndarray = sol['x'].full().flatten()

        # update weights
        self.weights.update_values(theta_new)
        return dJdtheta

    def learn(
        self,
        n_train_sessions: int,
        n_train_episodes: int,
        eval_env: QuadRotorEnv,
        n_eval_episodes: int,
        perturbation_decay: float = 0.9,
        seed: int = None,
        logger: Logger = None
    ) -> None:
        # simulate m episodes for each session
        env, cnt = self.env, 0
        for s in range(n_train_sessions):
            for e in range(n_train_episodes):
                state = env.reset(seed=None if seed is None else (seed + cnt))
                self.reset()
                done, t = False, 0
                while not done:
                    action = self.predict(state, deterministic=False)[0]
                    action_opt, _, sol = self.predict(
                        state, deterministic=True)
                    # action, _, sol = self.predict(
                    #     state, deterministic=False, perturb_gradient=False)
                    # action_opt = sol.vals['u'][:, 0]
                    new_state, r, done, _ = env.step(action)

                    # save only successful transitions
                    if sol.success:
                        self.save_transition(
                            state, action, action_opt, r, new_state, sol)
                    else:
                        logger.warning(f'{self.name}|{s}|{e}|{t}: MPC failed'
                                       f' - {sol.status}.')
                        # The solver can still reach maximum iteration and not
                        # converge to a good solution. If that happens, in the
                        # safe variant break the episode and label the
                        # parameters unsafe.
                        raise NotImplementedError()
                    state = new_state
                    t += 1

                # when the episode is done, consolidate its experience into memory
                self.consolidate_episode_experience()
                cnt += 1

            # when all m episodes are done, perform RL update and reduce
            # exploration strength
            update_grad = self.update()
            self.perturbation_strength *= perturbation_decay

            # at the end of each session, evaluate the policy
            costs = self.eval(eval_env, n_eval_episodes, seed=seed + cnt)
            cnt += n_eval_episodes

            # log evaluation outcomes
            if logger is not None:
                logger.debug(
                    f'{self.name}|{s}: J_mean={costs.mean():,.3f} '
                    f'||dJ||={np.linalg.norm(update_grad):.3e}; ' +
                    self.weights.values2str())

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
        # mean = np.array([-200, 200, 100, -100, 50, 50, -1, -1, -10, -5])
        # std = np.array([300, 200, 100, 75, 50, 25, 5, 5, 25, 30])
        # x_norm = (x - mean) / std
        y: cs.SX = cs.vertcat(*(cs_prod(x**p) for i in range(1, 4)
                                for p in monomial_powers(self.env.nx, i)))
        self._Phi = cs.Function('Phi', [x], [y], ['s'], ['Phi(s)'])

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