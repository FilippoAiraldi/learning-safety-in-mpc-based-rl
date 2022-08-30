import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from agents.rl_parameters import RLParameter, RLParameterCollection
from dataclasses import dataclass
from envs import QuadRotorEnv, QuadRotorEnvConfig
from mpc import Solution
from scipy.linalg import lstsq
from typing import Union
from util import monomial_powers, cs_prod


@dataclass(frozen=True)
class TestLSTDDPGAgentConfig:
    # experience replay parameters
    replay_maxlen: float = 20  # 20 episodes
    replay_sample_size: float = 10  # sample from 10 out of 20 episodes
    replay_include_last: float = 5  # include in the sample the last 5 episodes

    # RL parameters
    lr: float = 1e-9
    max_perc_update: float = np.inf
    clip_grad_norm: float = None

    @property
    def init_pars(self) -> dict[str, Union[float, np.ndarray]]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


class TestLSTDDPGAgent(QuadRotorBaseLearningAgent):
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
        agent_config: Union[dict, TestLSTDDPGAgentConfig] = None,
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
            agent_config = TestLSTDDPGAgentConfig()
        elif isinstance(agent_config, dict):
            keys = TestLSTDDPGAgentConfig.__dataclass_fields__.keys()
            agent_config = TestLSTDDPGAgentConfig(
                **{k: agent_config[k] for k in keys if k in agent_config})
        self.config = agent_config
        super().__init__(env, agentname=agentname,
                         init_pars=None,
                         fixed_pars={'perturbation': np.nan},
                         seed=seed)

        # during learning, DPG must always perturb the action in order to learn
        self.perturbation_chance = 1.0

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
        S, L, S_next = [], [], []
        E = []  # exploration
        for s, a, a_opt, r, s_next, _ in self._episode_buffer:
            S.append(s)
            L.append(r)
            S_next.append(s_next)
            E.append(a - a_opt)
        K = len(S)
        S = np.stack(S, axis=0)
        L = np.stack(L, axis=0).reshape(K, 1)
        S_next = np.stack(S_next, axis=0)
        E = np.stack(E, axis=0)

        # normalize state and reward
        mean, std = S.mean(), S.std()
        S_norm = (S - mean) / (std + 1e-10)
        S_next_norm = (S_next - mean) / (std + 1e-10)
        # L = (L - L.mean()) / (L.std() + 1e-10)

        # compute Phi (value function approximation basis functions)
        Phi = self._Phi(S_norm.T).full().T
        Phi_next = self._Phi(S_next_norm.T).full().T

        # compute Psi
        # #
        # import torch
        # u_bnd = torch.from_numpy(self.env.config.u_bounds)
        # Phi_t = torch.from_numpy(self._Phi(S.T).full().T)

        # def pi_(theta):
        #     A, b = theta[:-3], theta[-3:]
        #     Am = A.reshape((-1, 3)).T
        #     pi = Phi_t @ Am.T + b
        #     pi = (torch.tanh(pi) + 1) / 2 * torch.diff(u_bnd).squeeze() + \
        #         u_bnd[:, 0]
        #     # a = self._pi(S.T, A.detach().numpy(), b.detach().numpy()).full().T
        #     return pi

        # dpidtheta0 = np.transpose(
        #     torch.autograd.functional.jacobian(
        #         pi_, torch.from_numpy(self.weights.values())).numpy().squeeze(),
        #     axes=(0, 2, 1)
        #     )
        # #
        A, b = self.weights['A'].value, self.weights['b'].value
        dpidtheta = np.transpose(
            self._dpidtheta(S.T, A, b).full().reshape(b.shape[0], K, -1),
            axes=(1, 2, 0))
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
        lb, ub = bounds[:, 0], bounds[:, 1]
        if np.isfinite(cfg.max_perc_update):
            max_delta = np.maximum(np.abs(cfg.max_perc_update * theta), 0.1)
            lb = np.maximum(lb, theta - max_delta)
            ub = np.minimum(ub, theta + max_delta)
        

        # run QP solver
        sol = self._solver(lbx=lb, ubx=ub, x0=theta - c,
                           p=np.concatenate((theta, c)))
        assert self._solver.stats()['success'], 'RL update failed.'
        theta_new: np.ndarray = sol['x'].full().flatten()

        # update weights
        self.weights.update_values(theta_new)
        return dJdtheta

    def predict(
        self, state: np.ndarray = None, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, Solution]:
        # run the policy function
        a = self._pi(
            state, self.weights['A'].value, self.weights['b'].value
        ).full().squeeze()

        if deterministic or self.np_random.random() > self.perturbation_chance:
            return a, None, None

        # set std to a % of the action range
        u_bnd = self.env.config.u_bounds
        rng = self.np_random.normal(
            scale=self.perturbation_strength * np.diff(u_bnd).flatten(),
            size=a.shape)

        # directly perturb the action
        a_noisy = np.clip(a + rng, u_bnd[:, 0], u_bnd[:, 1])

        return a_noisy, None, None

    def learn(
        self,
        n_train_sessions: int,
        n_train_episodes: int,
        eval_env: QuadRotorEnv,
        n_eval_episodes: int,
        perturbation_decay: float = 0.97,
        seed: int = None,
        logger: Logger = None
    ) -> None:
        # simulate m episodes for each session
        env, cnt = self.env, 0
        for s in range(n_train_sessions):
            for e in range(n_train_episodes):
                state = env.reset(seed=None if seed is None else (seed + cnt))
                done = False
                while not done:
                    action_opt = self.predict(state, deterministic=True)[0]
                    action = self.predict(state, deterministic=False)[0]
                    # action = env.np_random.uniform(
                    #     low=env.config.u_bounds[:, 0],
                    #     high=env.config.u_bounds[:, 1])
                    new_state, r, done, _ = env.step(action)
                    self.save_transition(
                        state, action, action_opt, r, new_state, None)
                    state = new_state

                # when the episode is done, consolidate its experience into memory
                self.consolidate_episode_experience()
                cnt += 1

            # when all m episodes are done, perform RL update and reduce
            # exploration strength
            update_grad = self.update()
            self.perturbation_strength *= perturbation_decay

            # at the end of each session, evaluate the policy
            returns = self.eval(eval_env, n_eval_episodes, seed=seed + cnt)
            cnt += n_eval_episodes

            # log evaluation outcomes
            if logger is not None:
                logger.debug(
                    f'{self.name}|{s}|{e}: J_mean={returns.mean():,.3f} '
                    f'||dJ||={np.linalg.norm(update_grad):.3e}; ' +
                    self.weights.values2str())

    def _init_symbols(self) -> None:
        '''Computes symbolical derivatives needed for DPG updates.'''
        # compute baseline function approximating the value function with
        # monomials as basis
        x: cs.SX = cs.SX.sym('x', self.env.nx, 1)
        y: cs.SX = cs.vertcat(
            1,
            x,
            *(cs_prod(x**p) for p in monomial_powers(x.size1(), 2)))
        self._Phi = cs.Function('Phi', [x], [y], ['s'], ['Phi(s)'])

        # remove MPCs and re-create weights for the policy
        na, nx = QuadRotorEnv.nu, self._Phi.size1_out(0)
        g = QuadRotorEnvConfig.__dataclass_fields__['g'].default
        A, b = cs.SX.sym('A', na * nx, 1), cs.SX.sym('b', na, 1)
        self.weights = RLParameterCollection(
            RLParameter(
                'A',
                self.np_random.normal(size=na * nx) * 1e-3,
                [-np.inf, np.inf], A, A),
            RLParameter(
                'b',
                np.hstack((self.np_random.normal(size=na - 1) * 1e-3, g)),
                [-np.inf, np.inf], b, b)
        )
        self._A, self._b = A, b

        # compute derivative of the policy w.r.t. its weights
        pi = A.reshape((na, nx)) @ y + b
        u_bnd = self.env.config.u_bounds
        # a, b, act = 0, 1, cs_sigmoid
        lb, ub, act = -1, 1, cs.tanh
        pi = (act(pi) - lb) / (ub - lb) * np.diff(u_bnd) + u_bnd[:, 0]

        self._pi = cs.Function(
            'pi', [x, A, b], [pi], ['s', 'A', 'b'], ['pi(s)'])
        theta = self.weights.symV()
        dpidtheta = cs.jacobian(pi, theta)
        self._dpidtheta = cs.Function(
            'dpidtheta', [x, A, b], [dpidtheta],
            ['s', 'A', 'b'], ['dpidtheta(s)'])

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
