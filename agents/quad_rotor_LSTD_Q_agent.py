import logging
import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass
from envs import QuadRotorEnv
from itertools import chain
from mpc import Solution, MPCSolverError, QuadRotorMPCConfig
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
    init_thrust_coeff: float = 2.0
    # cost
    init_w_x: np.ndarray = 1e1
    init_w_u: np.ndarray = 1e0
    init_w_s: np.ndarray = 1e2

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
    Least-Squares Temporal Difference-based Q-learning RL agent for the quad 
    rotor environment. The agent adapts its MPC parameters/weights by value 
    methods, averaging over batches of episodes via Least-Squares, with the 
    goal of improving performance/reducing cost of each episode.

    The RL update exploits a replay memory to spread out noise.
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
        Initializes a LSTDQ agent for the quad rotor env.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the Q-learning agent.
        agentname : str, optional
            Name of the Q-learning agent.
        agent_config : dict, QuadRotorQLearningAgentConfig
            A set of parameters for the quadrotor Q-learning agent. If not 
            given, the default ones are used.
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
        super().__init__(
            env,
            agentname=agentname,
            init_pars=self.config.init_pars,
            fixed_pars={
                'pitch_d': 12,
                'pitch_dd': 7,
                'pitch_gain': 11,
                'roll_d': 10.5,
                'roll_dd': 8,
                'roll_gain': 9,
                'perturbation': np.nan
            },
            mpc_config=mpc_config,
            seed=seed
        )

        # during learning, DPG must always perturb the action in order to learn
        self.perturbation_chance = 0.7
        self.perturbation_strength = 5e-1

        # initialize the replay memory. Per each episode the memory saves the
        # gradient and Hessian of Q at each instant
        self.replay_memory = ReplayMemory[list[tuple[np.ndarray, ...]]](
            maxlen=agent_config.replay_maxlen, seed=seed)
        self._episode_buffer: list[tuple[np.ndarray, ...]] = []

        # initialize symbols for derivatives to be used later. Also initialize
        # the QP solver used to compute updates
        self._init_symbols()
        self._init_qp_solver()

        # initialize others
        self._epoch_n = None  # keeps track of epoch number just for logging

    def save_transition(
        self,
        cost: float,
        solQ: Solution,
        solV: Solution
    ) -> None:
        '''
        Schedules the current time-step data to be processed and saved into the
        experience replay memory.

        Parameters
        ----------
        cost : float
            Stage cost given by the environment at the current time step.
        solQ : mpc.Solution
            MPC solution of Q(s,a) where s and a are the current state and 
            action.
        solV : mpc.Solution
            MPC solution of V(s+) where s+ is the net state.
        '''
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
        self._episode_buffer.append((g, H))

    def consolidate_episode_experience(self) -> None:
        '''
        At the end of an episode, computes the remaining operations and 
        saves results to the replay memory as arrays.
        '''
        if len(self._episode_buffer) == 0:
            return
        self.replay_memory.append(self._episode_buffer.copy())
        self._episode_buffer.clear()

    def update(self) -> np.ndarray:
        '''
        Updates the MPC function approximation's weights based on the 
        information stored in the replay memory.

        Returns
        -------
        gradient : array_like
            Gradient of the update.
        '''
        # sample the memory
        cfg = self.config
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # sum over the batch of samples and compute update direction p
        g, H = [sum(o) for o in zip(*chain.from_iterable(sample))]
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

    def learn_one_epoch(
        self,
        n_episodes: int,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        raises: bool = True
    ) -> np.ndarray:
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_episodes : int
            Number of training episodes for the current epoch.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after each epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : Logger, optional
            For logging purposes.
        raises : bool, optional
            Whether to raise an exception when the MPC solver fails.
        '''
        logger = logger or logging.getLogger('dummy')

        env, name, epoch_n = self.env, self.name, self._epoch_n
        returns = np.zeros(n_episodes)
        seeds = self._prepare_seed(seed, n_episodes)

        for e in range(n_episodes):
            state = env.reset(seed=seeds[e])
            self.reset()
            done, t = False, 0
            action = self.predict(state, deterministic=False)[0]

            while not done:
                # compute Q(s, a)
                self.fixed_pars.update({'u0': action})
                solQ = self.solve_mpc('Q', state)

                # step the system
                state, r, done, _ = env.step(action)
                returns[e] += r

                # compute V(s+)
                action, _, solV = self.predict(state, deterministic=False)

                # save only successful transitions
                if solQ.success and solV.success:
                    self.save_transition(r, solQ, solV)
                else:
                    logger.warning(f'{name}|{epoch_n}|{e}|{t}: MPC failed.')
                    if raises:
                        raise MPCSolverError('MPC failed.')
                t += 1

            # when episode is done, consolidate its experience into memory
            self.consolidate_episode_experience()
            logger.debug(f'{name}|{epoch_n}|{e}: J={returns[e]:,.3f}')

        # when all m episodes are done, perform RL update
        update_grad = self.update()

        # log training outcomes and return cumulative returns
        logger.debug(f'{self.name}|{epoch_n}: J_mean={returns.mean():,.3f}; '
                     f'||p||={np.linalg.norm(update_grad):.3e}; ' +
                     self.weights.values2str())
        return returns

    def learn(
        self,
        n_train_epochs: int,
        n_train_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        raises: bool = True
    ) -> np.ndarray:
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_train_epochs : int
            Number of training sessions/epochs.
        n_train_episodes : int
            Number of training episodes per session.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after each epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : Logger, optional
            For logging purposes.
        raises : bool, optional
            Whether to raise an exception when the MPC solver fails.
        '''
        logger = logger or logging.getLogger('dummy')
        returns, cnt = [], 0
        for self._epoch_n in range(n_train_epochs):
            # let this epoch run
            returns.append(self.learn_one_epoch(
                n_episodes=n_train_episodes,
                seed=None if seed is None else seed + cnt,
                logger=logger,
                raises=raises
            ))
            cnt += n_train_episodes

            # when the epoch is done, reduce exploration
            self.perturbation_strength *= perturbation_decay
            self.perturbation_chance *= perturbation_decay
        return np.stack(returns, axis=0)

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

    def _prepare_seed(self, seed: Union[int, list[int]], n: int) -> list[int]:
        if seed is None:
            return [None] * n
        if isinstance(seed, int):
            return [seed + i for i in range(n)]
        assert len(seed) == n, 'Seed sequence with invalid length.'
        return seed
