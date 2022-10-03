import logging
import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import \
    QuadRotorBaseLearningAgent, UpdateError
from dataclasses import dataclass
from envs import QuadRotorEnv
from itertools import chain
from mpc import Solution, MPCSolverError, QuadRotorMPCConfig
from scipy.linalg import cho_solve
from typing import Union
from util.configurations import BaseConfig, init_config
from util.math import cholesky_added_multiple_identities
from util.rl import ReplayMemory


@dataclass
class QuadRotorLSTDQAgentConfig(BaseConfig):
    # initial learnable RL weights and their bounds
    init_g: float = (9.81, (1, 40))
    init_thrust_coeff: float = (2.0, (0.1, 4))  # safe=0.7
    # init_w_x: np.ndarray = (1e1, (1e-3, np.inf))
    # init_w_u: np.ndarray = (1e0, (1e-3, np.inf))
    # init_w_s: np.ndarray = (1e2, (1e-3, np.inf))
    # fixed non-learnable weights
    fixed_pitch_d: float = 12
    fixed_pitch_dd: float = 7
    fixed_pitch_gain: float = 11
    fixed_roll_d: float = 10.5
    fixed_roll_dd: float = 8
    fixed_roll_gain: float = 9
    fixed_w_x: np.ndarray = 1e1
    fixed_w_u: np.ndarray = 1e0
    fixed_w_s: np.ndarray = 1e2

    # experience replay parameters
    replay_maxlen: float = 20
    replay_sample_size: float = 10
    replay_include_last: float = 5

    # RL algorithm parameters
    gamma: float = 1.0
    lr: float = 1e-1
    max_perc_update: float = np.inf


class QuadRotorLSTDQAgent(QuadRotorBaseLearningAgent):
    '''
    Least-Squares Temporal Difference-based Q-learning RL agent for the quad 
    rotor environment. The agent adapts its MPC parameters/weights by value 
    methods, averaging over batches of episodes via Least-Squares, with the 
    goal of improving performance/reducing cost of each episode.

    The RL update exploits a replay memory to spread out noise.
    '''

    config_cls: type = QuadRotorLSTDQAgentConfig

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
        # initialize fixed and learnable parameters. If the environment is
        # normalized, then apply normalization here
        agent_config = init_config(agent_config, self.config_cls)
        init_pars = agent_config.get_group('init')
        fixed_pars = agent_config.get_group('fixed')
        if env.normalized:
            fixed_pars = {
                name: (env.normalize(name, par)
                       if env.can_be_normalized(name) else par)
                for name, par in fixed_pars.items()
            }
            init_pars = {
                name: (
                    (env.normalize(name, par), env.normalize(name, bnd))
                    if env.can_be_normalized(name) else
                    (par, bnd)
                )
                for name, (par, bnd) in init_pars.items()
            }
        fixed_pars['perturbation'] = np.nan

        # create base agent
        super().__init__(
            env,
            agentname=agentname,
            agent_config=agent_config,
            fixed_pars=fixed_pars,
            init_learnable_pars=init_pars,
            mpc_config=mpc_config,
            seed=seed
        )

        # in order to learn, Q learning must sometime perturb the action
        self.perturbation_chance = 0.7
        self.perturbation_strength = 5e-1

        # initialize the replay memory. Per each episode the memory saves the
        # gradient and Hessian of Q at each instant
        self.replay_memory = ReplayMemory[list[tuple[np.ndarray, ...]]](
            maxlen=self.config.replay_maxlen, seed=seed)
        self._episode_buffer: list[tuple[np.ndarray, ...]] = []

        # initialize symbols for derivatives to be used later. Also initialize
        # the QP solver used to compute updates
        self._init_derivative_symbols()
        self._init_qp_solver()

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
        # sample the memory
        cfg: QuadRotorLSTDQAgentConfig = self.config
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # sum over the batch of samples and compute update direction p
        g, H = [sum(o) for o in zip(*chain.from_iterable(sample))]
        R = cholesky_added_multiple_identities(H)
        p = cho_solve((R, True), g).flatten()

        # run QP solver and update weights
        theta = self.weights.values()
        pars = np.block([theta, p, cfg.lr])
        lb, ub = self._get_percentage_bounds(
            theta, self.weights.bounds(), cfg.max_perc_update)
        sol = self._solver(p=pars, lbx=lb, ubx=ub, x0=theta - cfg.lr * p)
        if not self._solver.stats()['success']:
            raise UpdateError(f'RL update failed in epoch {self._epoch_n}.')
        self.weights.update_values(sol['x'].full().flatten())
        return p

    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        return_info: bool = False
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ]:
        logger = logger or logging.getLogger('dummy')

        env, name, epoch_n = self.env, self.name, self._epoch_n
        returns = np.zeros(n_episodes)
        seeds = self._make_seed_list(seed, n_episodes)

        for e in range(n_episodes):
            state = env.reset(seed=seeds[e])
            self.reset()
            truncated, terminated, t = False, False, 0
            action = self.predict(state, deterministic=False)[0]

            while not (truncated or terminated):
                # compute Q(s, a)
                self.fixed_pars.update({'u0': action})
                solQ = self.solve_mpc('Q', state)

                # step the system
                state, r, truncated, terminated, _ = env.step(action)
                returns[e] += r

                # compute V(s+)
                action, _, solV = self.predict(state, deterministic=False)

                # save only successful transitions
                if solQ.success and solV.success:
                    self.save_transition(r, solQ, solV)
                else:
                    raise MPCSolverError(
                        f'{name}|{epoch_n}|{e}|{t}: MPC failed.')
                t += 1

            # when episode is done, consolidate its experience into memory
            self.consolidate_episode_experience()
            logger.debug(f'{name}|{epoch_n}|{e}: J={returns[e]:,.3f}')

        # when all m episodes are done, perform RL update and reduce
        # exploration strength and chance
        update_grad = self.update()
        self.perturbation_strength *= perturbation_decay
        self.perturbation_chance *= perturbation_decay

        # log training outcomes and return cumulative returns
        logger.debug(f'{self.name}|{epoch_n}: J_mean={returns.mean():,.3f}; '
                     f'||p||={np.linalg.norm(update_grad):.3e}; ' +
                     self.weights.values2str())
        return (
            (returns, update_grad, self.weights.values(as_dict=True))
            if return_info else
            returns
        )

    def _init_derivative_symbols(self) -> None:
        '''Computes symbolical derivatives needed for Q learning.'''
        theta = self.weights.symQ()
        lagr = self.Q.lagrangian
        d2Qdtheta, dQdtheta = cs.hessian(lagr, theta)
        self.dQdtheta = cs.simplify(dQdtheta)
        self.d2Qdtheta = cs.simplify(d2Qdtheta)

    def _init_qp_solver(self) -> None:
        n_theta = self.weights.n_theta

        # prepare symbols
        theta: cs.SX = cs.SX.sym('theta', n_theta, 1)
        theta_new: cs.SX = cs.SX.sym('theta+', n_theta, 1)
        dtheta = theta_new - theta
        p: cs.SX = cs.SX.sym('p', n_theta, 1)
        lr: cs.SX = cs.SX.sym('lr', n_theta, 1)

        # prepare solver
        qp = {
            'x': theta_new,
            'f': 0.5 * dtheta.T @ dtheta + (lr * p).T @ dtheta,
            'p': cs.vertcat(theta, p, lr)
        }
        opts = {'print_iter': False, 'print_header': False}
        self._solver = cs.qpsol(f'qpsol_{self.name}', 'qrqp', qp, opts)
