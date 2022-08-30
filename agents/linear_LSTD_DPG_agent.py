import casadi as cs
import numpy as np
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from agents.rl_parameters import RLParameter, RLParameterCollection
from dataclasses import dataclass
from envs import QuadRotorEnv
from logging import Logger
from scipy.linalg import lstsq
from typing import Union, Tuple
from util import monomial_powers, cs_prod


@dataclass(frozen=True)
class LinearLSTDDPGAgentConfig:
    # experience replay parameters
    replay_maxlen: float = 20  # 20 episodes
    replay_sample_size: float = 10  # sample from 10 out of 20 episodes
    replay_include_last: float = 5  # include in the sample the last 5 episodes

    # RL parameters
    gamma: float = 1.0
    lr: float = 1e-9
    clip_grad_norm: float = None

    @property
    def init_pars(self) -> dict[str, Union[float, np.ndarray]]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


class LinearLSTDDPGAgent(QuadRotorBaseLearningAgent):
    '''
    Least-Squares Temporal Difference-based Deterministic Policy Gradient RL 
    agent for the quad rotor environment. The agent adapts its linear policy 
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
        agent_config: Union[dict, LinearLSTDDPGAgentConfig] = None,
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
        seed : int, optional
            Seed for the random number generator.
        '''
        if agent_config is None:
            agent_config = LinearLSTDDPGAgentConfig()
        elif isinstance(agent_config, dict):
            keys = LinearLSTDDPGAgentConfig.__dataclass_fields__.keys()
            agent_config = LinearLSTDDPGAgentConfig(
                **{k: agent_config[k] for k in keys if k in agent_config})
        self.config = agent_config
        super().__init__(env, agentname=agentname,
                         init_pars=None,
                         fixed_pars={'perturbation': np.nan},
                         seed=seed)

        # during learning, DPG must always perturb the action in order to learn
        self.perturbation_chance = 1.0
        self.perturbation_strength = 0.5

        # initialize the replay memory. Per each episode the memory saves an
        # array of Phi(s), Psi(s,a), L(s,a), dpidtheta(s) and weights v. Also
        # initialize the episode buffer which temporarily stores values before
        # batch-processing them into the replay memory
        self.replay_memory = ReplayMemory[tuple[np.ndarray, ...]](
            maxlen=agent_config.replay_maxlen, seed=seed)
        self._episode_buffer: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []

        # initialize symbols for derivatives to be used later
        self._init_symbols()

    def save_transition(
        self,
        state: np.ndarray,
        action_taken: np.ndarray,
        optimal_action: np.ndarray,
        cost: np.ndarray,
        new_state: np.ndarray,
    ) -> None:
        item = (state, action_taken, optimal_action, cost, new_state)
        self._episode_buffer.append(item)

    # compute Psi with PyTorch
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK']='True'
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

    def consolidate_episode_experience(self) -> None:
        if len(self._episode_buffer) == 0:
            return

        # stack everything in arrays and compute derivatives
        S, L, S_next, E = [], [], [], []
        K = len(self._episode_buffer)
        for s, a, a_opt, r, s_next in self._episode_buffer:
            S.append(s)
            L.append(r)
            S_next.append(s_next)
            E.append(a - a_opt)
        S = np.stack(S, axis=0)
        L = np.stack(L, axis=0).reshape(K, 1)
        S_next = np.stack(S_next, axis=0)
        E = np.stack(E, axis=0)

        # compute Phi (value function approximation basis functions)
        Phi = self._Phi(S.T).full().T
        Phi_next = self._Phi(S_next.T).full().T

        # compute Psi
        A, b = self.weights['A'].value, self.weights['b'].value
        dpidtheta = np.transpose(
            self._dpidtheta(S.T, A, b).full().reshape(b.shape[0], K, -1),
            axes=(1, 2, 0))
        Psi = (dpidtheta @ E.reshape(K, -1, 1)).squeeze()

        # compute this episode's weights v via LSTD
        try:
            v = np.linalg.solve(
                Phi.T @ (Phi - self.config.gamma * Phi_next),
                Phi.T @ L)
        except Exception:
            v = lstsq(Phi.T @ (Phi - self.config.gamma * Phi_next),
                      Phi.T @ L, lapack_driver='gelsy')[0]

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
            b = Psi.T @ (L + (self.config.gamma * Phi_next - Phi) @ v)
            try:
                w += np.linalg.solve(A, b)
            except Exception:
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

        # compute the new parameters and update the weights
        theta = self.weights.values()
        theta_new = theta - c
        self.weights.update_values(theta_new)
        return dJdtheta

    def predict(
        self, state: np.ndarray = None, deterministic: bool = False
    ) -> Tuple[np.ndarray, None, None]:
        a = self._pi(
            state, self.weights['A'].value, self.weights['b'].value
        ).full().squeeze()
        assert np.isfinite(a).all()

        if deterministic or self.np_random.random() > self.perturbation_chance:
            return a, None, None

        u_bnd = self.env.config.u_bounds
        rng = self.np_random.normal(
            scale=self.perturbation_strength * np.diff(u_bnd).flatten(),
            size=a.shape)
        return a + rng, None, None

    def learn(
        self,
        n_train_sessions: int,
        n_train_episodes: int,
        eval_env: QuadRotorEnv,
        n_eval_episodes: int,
        perturbation_decay: float = 0.8,
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
                        state, action, action_opt, r, new_state)
                    state = new_state

                # when episode is done, consolidate its experience into memory
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
        mean = np.array([-200, 200, 100, -100, 50, 50, -1, -1, -10, -5])
        std = np.array([300, 200, 100, 75, 50, 25, 5, 5, 25, 30])
        x_norm = (x - mean) / std

        y: cs.SX = cs.vertcat(*(cs_prod(x_norm**p) for i in range(1, 4)
                                for p in monomial_powers(self.env.nx, i)))
        self._Phi = cs.Function('Phi', [x], [y], ['s'], ['Phi(s)'])

        # re-create weights for the policy
        na, nx = self.env.nu, self._Phi.size1_out(0)
        u_bnd = self.env.config.u_bounds
        A, b = cs.SX.sym('A', na * nx, 1), cs.SX.sym('b', na, 1)
        self.weights = RLParameterCollection(
            RLParameter(
                'A',
                self.np_random.normal(size=na * nx) * 1e-1,
                [-np.inf, np.inf], A, A),
            RLParameter(
                'b',
                self.np_random.normal(size=na) * 1e-1,
                [-np.inf, np.inf], b, b)
        )
        self._A, self._b = A, b

        # compute derivative of the policy w.r.t. its weights
        pi = A.reshape((na, nx)) @ y + b
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
