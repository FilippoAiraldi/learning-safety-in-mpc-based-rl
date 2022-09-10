import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass
from envs import QuadRotorEnv
from logging import Logger
from mpc import Solution, QuadRotorMPCConfig
from scipy.linalg import cho_solve
from typing import Union
from util import cholesky_added_multiple_identities


@dataclass(frozen=True)
class QuadRotorLSTDQAgentConfig:
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
    replay_maxlen: float = 2000
    replay_sample_size: float = 500
    replay_include_last: float = 100

    # RL parameters
    gamma: float = 1.0
    lr: float = 1e-1
    max_perc_update: float = np.inf
    clip_grad_norm: float = None

    @property
    def init_pars(self) -> dict[str, Union[float, np.ndarray]]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


class QuadRotorLSTDQAgent(QuadRotorBaseLearningAgent):
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
        agent_config: Union[dict, QuadRotorLSTDQAgentConfig] = None,
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
        agent_config : dict, QuadRotorQLearningAgentConfig
            A set of parameters for the quadrotor DPG agent. If not given, the
            default ones are used.
        mpc_config : dict, QuadRotorMPCConfig
            A set of parameters for the agent's MPC. If not given, the default
            ones are used.
        seed : int, optional
            Seed for the random number generator.
        '''
        if agent_config is None:
            agent_config = QuadRotorLSTDQAgentConfig()
        elif isinstance(agent_config, dict):
            keys = QuadRotorLSTDQAgentConfig.__dataclass_fields__.keys()
            agent_config = QuadRotorLSTDQAgentConfig(
                **{k: agent_config[k] for k in keys if k in agent_config})
        self.config = agent_config
        super().__init__(env, agentname=agentname,
                         init_pars=self.config.init_pars,
                         fixed_pars={'perturbation': np.nan},
                         mpc_config=mpc_config, seed=seed)

        # during learning, DPG must always perturb the action in order to learn
        self.perturbation_chance = 0.5
        self.perturbation_strength = 1e-1

        # initialize the replay memory. Per each episode the memory saves an
        # array of Phi(s), Psi(s,a), L(s,a), dpidtheta(s) and weights v. Also
        # initialize the episode buffer which temporarily stores values before
        # batch-processing them into the replay memory
        self.replay_memory = ReplayMemory[tuple[np.ndarray, ...]](
            maxlen=agent_config.replay_maxlen, seed=seed)

        # initialize symbols for derivatives to be used later. Also initialize
        # the QP solver used to compute updates
        self._init_symbols()
        self._init_qp_solver()

    def save_transition(
        self,
        cost: float,
        solQ: Solution,
        solV: Solution
    ) -> None:
        # compute td error
        target = cost + self.config.gamma * solV.f
        td_err = target - solQ.f

        # compute numerical gradients w.r.t. params
        dQ = solQ.value(self.dQdtheta).reshape(-1, 1)
        d2Q = solQ.value(self.d2Qdtheta)

        # compute gradient and approximated hessian
        g = -td_err * dQ
        H = dQ @ dQ.T - td_err * d2Q

        # save to replay memory
        self.replay_memory.append((g, H))

    def consolidate_episode_experience(self) -> None:
        pass

    def update(self) -> np.ndarray:
        # sample the memory
        cfg = self.config
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # sum over the batch of samples and compute update direction p
        g, H = [sum(o) for o in zip(*sample)]
        R = cholesky_added_multiple_identities(H)
        p = cho_solve((R, True), g).flatten()

        # clip gradient if requested
        if cfg.clip_grad_norm is None:
            c = cfg.lr * p
        else:
            clip_coef = min(
                cfg.clip_grad_norm / (np.linalg.norm(p) + 1e-6), 1.0)
            c = (cfg.lr * clip_coef) * p

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
        return p

    def learn(
        self,
        n_train_sessions: int,
        n_train_episodes: int,
        perturbation_decay: float = 0.9,
        seed: int = None,
        logger: Logger = None,
        **kwargs
    ) -> None:
        # simulate m episodes for each session
        env, cnt = self.env, 0
        for s in range(n_train_sessions):
            for e in range(n_train_episodes):
                state = env.reset(seed=None if seed is None else (seed + cnt))
                self.reset()
                action = self.predict(state, deterministic=False)[0]
                done, t = False, 0
                while not done:
                    # compute Q(s, a)
                    self.fixed_pars.update({'u0': action})
                    solQ = self.solve_mpc('Q', state)

                    # step the system
                    state, r, done, _ = env.step(action)

                    # compute V(s+)
                    action, _, solV = self.predict(state, deterministic=False)

                    # save only successful transitions
                    if solQ.success and solV.success:
                        self.save_transition(r, solQ, solV)
                    else:
                        logger.warning(f'{self.name}|{s}|{e}|{t}: MPC failed.')
                        # The solver can still reach maximum iteration and not
                        # converge to a good solution. If that happens, in the
                        # safe variant break the episode and label the
                        # parameters unsafe.
                        raise NotImplementedError()
                    t += 1

                if logger is not None:
                    logger.debug(
                        f'{self.name}|{s}|{e}: J={env.cum_rewards[-1]:,.3f}')

                # does nothing
                self.consolidate_episode_experience()
                cnt += 1

            # when all m episodes are done, perform RL update and reduce
            # exploration strength
            update_grad = self.update()
            self.perturbation_strength *= perturbation_decay
            self.perturbation_chance *= perturbation_decay

            # log evaluation outcomes
            J_mean = np.mean(
                [env.cum_rewards[i] for i in range(-n_train_episodes, 0)])
            if logger is not None:
                logger.debug(
                    f'{self.name}|{s}: J_mean={J_mean:.3f}; '
                    f'||p||={np.linalg.norm(update_grad):.3e}; ' +
                    self.weights.values2str())

    def _init_symbols(self) -> None:
        '''Computes symbolical derivatives needed for Q learning.'''
        theta = self.weights.symQ()
        lagr = self.Q.lagrangian
        d2Qdtheta, dQdtheta = cs.hessian(lagr, theta)
        self.dQdtheta = cs.simplify(dQdtheta)
        self.d2Qdtheta = cs.simplify(d2Qdtheta)

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
