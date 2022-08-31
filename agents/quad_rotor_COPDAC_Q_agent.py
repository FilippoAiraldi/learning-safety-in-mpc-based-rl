import numpy as np
import casadi as cs
from agents.quad_rotor_base_learning_agent import QuadRotorBaseLearningAgent
from agents.replay_memory import ReplayMemory
from dataclasses import dataclass
from envs import QuadRotorEnv
from logging import Logger
from mpc import Solution, QuadRotorMPCConfig
from typing import Union
from util import monomial_powers, cs_prod


@dataclass(frozen=True)
class QuadRotorCOPDACQAgentConfig:
    # initial RL pars
    # model
    # NOTE: initial values were made closer to real in an attempt to check if
    # learning happens. Remember to reset them to more difficult values at some
    # point
    init_g: float = 10.0
    init_thrust_coeff: float = 1.2
    init_pitch_d: float = 12
    init_pitch_dd: float = 8
    init_pitch_gain: float = 12
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
    gamma: float = 1.0
    lr_theta: float = 1e-3
    lr_w: float = 5e-1
    lr_v: float = 5e-1
    max_perc_update: float = np.inf
    clip_grad_norm: float = None

    @property
    def init_pars(self) -> dict[str, Union[float, np.ndarray]]:
        '''Groups the initial RL parameters into a dictionary.'''
        return {
            name.removeprefix('init_'): val
            for name, val in self.__dict__.items() if name.startswith('init_')
        }


class QuadRotorCOPDACQAgent(QuadRotorBaseLearningAgent):
    '''
    Compatible Off-Policy Deterministic Actor-Critic agent for the 
    quad rotor environment. The agent adapts its MPC actor parameters/weights 
    by  policy gradient methods and the critic by Q-learning, with the goal of 
    improving performance/reducing cost of each episode.

    The policy gradient-based RL update exploits a replay memory to spread out
    the gradient noise.
    '''

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: Union[dict, QuadRotorCOPDACQAgentConfig] = None,
        mpc_config: Union[dict, QuadRotorMPCConfig] = None,
        seed: int = None
    ) -> None:
        '''
        Initializes a COPDAC-Q agent for the quad rotor env.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the DPG agent.
        agentname : str, optional
            Name of the DPG agent.
        agent_config : dict, QuadRotorCOPDACQAgentConfig
            A set of parameters for the quadrotor DPG agent. If not given, the
            default ones are used.
        mpc_config : dict, QuadRotorMPCConfig
            A set of parameters for the agent's MPC. If not given, the default
            ones are used.
        seed : int, optional
            Seed for the random number generator.
        '''
        if agent_config is None:
            agent_config = QuadRotorCOPDACQAgentConfig()
        elif isinstance(agent_config, dict):
            keys = QuadRotorCOPDACQAgentConfig.__dataclass_fields__.keys()
            agent_config = QuadRotorCOPDACQAgentConfig(
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
        self._init_weights()
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

        # compute Phi (value function approximation basis functions)
        Phi = self._Phi(S.T).full().T
        Phi_next = self._Phi(S_next.T).full().T

        # compute Psi
        q = np.linalg.solve(dRdy, np.tile(self._dydu0, (K, 1, 1)))
        dpidtheta = -dRdtheta @ q
        Psi = (dpidtheta @ E.reshape(K, -1, 1)).squeeze()

        # compute the td error
        td_error = \
            L + (self.config.gamma * Phi_next - Phi) @ self._v - Psi @ self._w

        # save this episode to memory and clear buffer
        self.replay_memory.extend(zip(Phi, Psi, dpidtheta, td_error))
        self._episode_buffer.clear()

    def update(self) -> np.ndarray:
        # sample the memory. Each item in the sample comes from one episode
        cfg = self.config
        sample = self.replay_memory.sample(
            cfg.replay_sample_size, cfg.replay_include_last)

        # average over the batch of samples
        Phi, Psi, dpidtheta, td_error = [
            np.mean(o, axis=0) for o in zip(*sample)]

        # compute the policy gradient
        dJdtheta = (dpidtheta @ (dpidtheta.T @ self._w)).flatten()

        # clip gradient if requested
        if cfg.clip_grad_norm is None:
            c = cfg.lr_theta * dJdtheta
        else:
            clip_coef = min(
                cfg.clip_grad_norm / (np.linalg.norm(dJdtheta) + 1e-6), 1.0)
            c = (cfg.lr_theta * clip_coef) * dJdtheta

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

        # update other incremental weights
        self._w += cfg.lr_w * td_error * Psi.reshape(-1, 1)
        self._v += cfg.lr_v * td_error * Phi.reshape(-1, 1)

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
                    f'{self.name}|{s}: J={costs.mean():,.3f} '
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

    def _init_weights(self) -> None:
        # initialize COPDAC-Q weights
        self._w = self.np_random.random(size=(sum(self.weights.sizes()), 1))
        self._v = self.np_random.random(size=(self._Phi.size1_out(0), 1))

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
