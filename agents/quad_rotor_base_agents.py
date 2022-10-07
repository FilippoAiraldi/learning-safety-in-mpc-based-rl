from abc import ABC, abstractmethod
from itertools import count
import logging
from typing import Any, Optional, Union
import numpy as np
from gym import Env
from gym.utils.seeding import np_random
from envs import QuadRotorEnv
from mpc import QuadRotorMPC, QuadRotorMPCConfig, Solution
from mpc.wrappers import DifferentiableMPC
from util.configurations import init_config
from util.errors import MPCSolverError, UpdateError
from util.math import NormalizationService
from util.rl import RLParameter, RLParameterCollection


class QuadRotorBaseAgent(ABC):
    '''
    Abstract base agent class that contains the two MPC function approximators
    `Q` and `V`.
    '''
    _ids = count(0)

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        agent_config: Union[dict[str, Any], Any] = None,
        fixed_pars: dict[str, np.ndarray] = None,
        mpc_config: Union[dict, QuadRotorMPCConfig] = None,
        seed: int = None
    ) -> None:
        '''
        Instantiates an agent.

        Parameters
        ----------
        env : QuadRotorEnv
            Environment for which to create the agent.
        agentname : str, optional
            Name of the agent.
        agent_config : dict, ConfigType, optional
            A set of parameters for the quadrotor agent. If not given, the 
            default ones are used.
        fixed_pars : dict[str, np.ndarray], optional
            A dictionary containing MPC parameters that are fixed.
        mpc_config : dict, QuadRotorMPCConfig, optional
            A set of parameters for the agent's MPC. If not given, the default
            ones are used.
        seed : int, optional
            Seed for the random number generator.
        '''
        super().__init__()
        self.id = next(self._ids)
        self.name = f'Agent{self.id}' if agentname is None else agentname
        self.env = env
        self._config = init_config(agent_config, self.config_cls) \
            if hasattr(self, 'config_cls') else None
        self.fixed_pars = {} if fixed_pars is None else fixed_pars

        # set RNG and disturbances
        self.seed = seed
        self.np_random, _ = np_random(seed)
        self.perturbation_chance = 0.2
        self.perturbation_strength = 0.05

        # initialize MPCs
        self.last_solution: Solution = None
        self._Q = QuadRotorMPC(env, config=mpc_config, type='Q')
        self._V = QuadRotorMPC(env, config=mpc_config, type='V')

    @property
    def normalized(self) -> bool:
        '''Returns whether the agent's env is normalized.'''
        return self.env.normalization is not None

    @property
    def normalization(self) -> Optional[NormalizationService]:
        '''Returns the agent's env normalization.'''
        return self.env.normalization

    @property
    def V(self) -> QuadRotorMPC:
        '''Gets the `V` action-value function approximation MPC scheme.'''
        return self._V

    @property
    def Q(self) -> QuadRotorMPC:
        '''Gets the `Q` action-value function approximation MPC scheme.'''
        return self._Q

    @property
    def config(self) -> Any:
        '''Gets this agent's configuration.'''
        return self._config

    def reset(self) -> None:
        '''Resets internal variables of the agent.'''
        # reset MPC last solution
        self.last_solution = None

        # reset MPC failure count
        self._Q.failures = 0
        self._V.failures = 0

    def solve_mpc(
        self,
        type: str,
        state: np.ndarray = None,
        sol0: dict[str, np.ndarray] = None,
    ) -> Solution:
        '''
        Solves the MPC optimization problem embedded in the agent.

        Parameters
        ----------
        type : 'Q' or 'V'
            Type of MPC function approximation to run.
        state : array_like, optional
            Environment's state for which to solve the MPC problem. If not
            given, the current state of the environment is used.
        sol0 : dict[str, array_like]
            Last numerical solution of the MPC used to warmstart. If not given,
            a heuristic is used.

        Returns
        -------
        sol : Solution
            Solution object containing values and information of the solution.
        '''
        mpc: QuadRotorMPC = getattr(self, type)

        # if the state which to solve the MPC for is not provided, use current
        if state is None:
            state = self.env.x

        # merge all parameters in a single dict
        pars = self.fixed_pars | {'x0': state}
        pars |= self._merge_mpc_pars_callback()

        # if provided, use vals0 to warmstart the MPC. If not provided,
        # use the last_sol field. If the latter is not available yet,
        # just use some default values
        if sol0 is None:
            if self.last_solution is None:
                g = float(pars.get('g', 0))
                sol0 = {
                    'x': np.tile(state, (mpc.vars['x'].shape[1], 1)).T,
                    'u': np.tile([0, 0, g], (mpc.vars['u'].shape[1], 1)).T,
                    'slack': 0
                }
            else:
                sol0 = self.last_solution.vals

        # call the MPC
        self.last_solution = mpc.solve(pars, sol0)
        return self.last_solution

    def predict(
        self,
        state: np.ndarray = None,
        deterministic: bool = False,
        perturb_gradient: bool = True,
        **solve_mpc_kwargs
    ) -> tuple[np.ndarray, np.ndarray, Solution]:
        '''
        Computes the optimal action for the given state by solving the MPC
        scheme `V` and predicts the next state.

        Parameters
        ----------
        state : array_like, optional
            Environment's state for which to solve the MPC problem V.
        deterministic : bool, optional
            Whether the computed optimal action should be modified by some
            noise (either summed or in the objective gradient).
        perturb_gradient : bool, optional
            Whether to perturb the MPC objective's gradient (if 'perturbation'
            parameter is present), or to directly perturb the optimal action.
        solve_mpc_kwargs
            See BaseMPCAgent.solve_mpc.

        Returns
        -------
        u_opt : np.ndarray
            The optimal action to take in the current state.
        next_state : np.ndarray
            The predicted next state of the environment.
        sol : Solution
            Solution object containing values and information of the solution.
        '''
        perturbation_in_dict = 'perturbation' in self.fixed_pars
        if perturbation_in_dict:
            self.fixed_pars['perturbation'] = 0

        if deterministic or self.np_random.random() > self.perturbation_chance:
            # just solve the V scheme without noise
            sol = self.solve_mpc(type='V', state=state, **solve_mpc_kwargs)
            u_opt = sol.vals['u'][:, 0]
        else:
            # set std to a % of the action range
            u_bnd = self.env.config.u_bounds
            rng = self.np_random.normal(
                scale=self.perturbation_strength * np.diff(u_bnd).flatten(),
                size=self.V.vars['u'].shape[0])

            # if there is the parameter to do so, perturb gradient
            if perturb_gradient:
                assert perturbation_in_dict, \
                    'No parameter \'perturbation\' found to perturb gradient.'
                self.fixed_pars['perturbation'] = rng

            sol = self.solve_mpc(type='V', state=state, **solve_mpc_kwargs)
            u_opt = sol.vals['u'][:, 0]

            # otherwise, directly perturb the action
            if not perturb_gradient:
                u_opt = np.clip(u_opt + rng, u_bnd[:, 0], u_bnd[:, 1])

        x_next = sol.vals['x'][:, 0]
        return u_opt, x_next, sol

    def eval(
        self,
        env: Env,
        n_eval_episodes: int,
        deterministic: bool = True,
        seed: int = None,
    ) -> np.ndarray:
        '''
        Evaluates the given environment.

        Parameters
        ----------
        env : gym.Env
            The environment to evaluate.
        n_eval_episodes : int
            Number of episodes over which to evaluate.
        deterministic : bool, optional
            Whether to use deterministic or stochastic actions.
        seed : int or list[int], optional
            RNG seed.

        Returns
        -------
        returns : array_like
            An array of the accumulated rewards/costs for each episode
        '''
        returns = np.zeros(n_eval_episodes)
        seeds = self._make_seed_list(seed, n_eval_episodes)

        for e in range(n_eval_episodes):
            state = env.reset(seed=seeds[e])
            self.reset()
            truncated, terminated = False, False

            while not (truncated or terminated):
                action = self.predict(state, deterministic=deterministic)[0]
                state, r, truncated, terminated, _ = env.step(action)
                returns[e] += r

        return returns

    def _merge_mpc_pars_callback(self) -> dict[str, np.ndarray]:
        '''
        Callback to allow the merging of additional MPC parameters from 
        inheriting classes.
        '''
        return {}

    def __str__(self) -> str:
        '''Returns the agent name.'''
        return f'<{type(self).__name__}: {self.name}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the Agent.'''
        return str(self)

    @staticmethod
    def _make_seed_list(
        seed: Optional[Union[int, list[int]]], n: int
    ) -> list[int]:
        '''Given a seed, possibly None, converts it into a list of length n,
        where each seed is different or None'''
        if seed is None:
            return [None] * n
        if isinstance(seed, int):
            return [seed + i for i in range(n)]
        assert len(seed) == n, 'Seed sequence with invalid length.'
        return seed


class QuadRotorBaseLearningAgent(QuadRotorBaseAgent, ABC):
    '''
    Abstract base agent class that renders the two MPC function approximators
    `Q` and `V` differentiable, such that their parameters can be learnt. 
    '''

    def __init__(
        self,
        *args,
        init_learnable_pars: dict[str, tuple[np.ndarray, np.ndarray]],
        **kwargs,
    ) -> None:
        '''
        Instantiates a learning agent.

        Parameters
        ----------
        init_learnable_pars : dict[str, tuple[array_like, array_like]]
            Initial values and bounds for each learnable MPC parameter.
        *args, **kwargs
            See `agents.QuadRotorBaseAgent`.
        '''
        super().__init__(*args, **kwargs)
        self._V = DifferentiableMPC[QuadRotorMPC](self._V)
        self._Q = DifferentiableMPC[QuadRotorMPC](self._Q)
        self._init_learnable_pars(init_learnable_pars)
        self._init_learning_rate()
        self._epoch_n = None  # keeps track of epoch number just for logging

    @property
    def V(self) -> DifferentiableMPC[QuadRotorMPC]:
        return self._V

    @property
    def Q(self) -> DifferentiableMPC[QuadRotorMPC]:
        return self._Q

    @abstractmethod
    def update(self) -> np.ndarray:
        '''
        Updates the MPC function approximation's weights based on the 
        information stored in the replay memory.

        Returns
        -------
        gradient : array_like
            Gradient of the update.
        '''
        pass

    @abstractmethod
    def learn_one_epoch(
        self,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        return_info: bool = True
    ) -> Union[
        np.ndarray,
        tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ]:
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_episodes : int
            Number of training episodes for the current epoch.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after this epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : logging.Logger, optional
            For logging purposes.
        return_info : bool, optional
            Whether to return additional information for this epoch update:
                - update gradient
                - agent's updated weights

        Returns
        -------
        returns : array_like
            An array of the returns for each episode of this epoch.
        gradient : array_like, optional
            Gradient of the update. Only returned if `return_info=True`.
        new_weights : dict[str, array_like], optional
            Agent's new set of weights after the update. Only returned if 
            `return_info=True`.
        '''
        pass

    def learn(
        self,
        n_epochs: int,
        n_episodes: int,
        perturbation_decay: float = 0.75,
        seed: Union[int, list[int]] = None,
        logger: logging.Logger = None,
        throw_on_exception: bool = False,
        return_info: bool = True
    ) -> Union[
        tuple[bool, np.ndarray],
        tuple[bool, np.ndarray, list[np.ndarray], list[dict[str, np.ndarray]]]
    ]:
        '''
        Trains the agent on its environment.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        n_episodes : int
            Number of training episodes per epoch.
        perturbation_decay : float, optional
            Decay factor of the exploration perturbation, after each epoch.
        seed : int or list[int], optional
            RNG seed.
        logger : logging.Logger, optional
            For logging purposes.
        throw_on_exception : bool, optional
            When a training exception occurs, if `throw_on_exception=True`,
            then the exception is fired again and training fails. Otherwise; 
            the training is prematurely stopped and returned.
        return_info : bool, optional
            Whether to return additional information for each epoch update:
                - a list of update gradients
                - a list of agent's updated weights after each update

        Returns
        -------
        success : bool
            True if the training was successfull.
        returns : array_like
            An array of the returns for each episode in each epoch.
        gradient : list[array_like], optional
            Gradients of each update. Only returned if `return_info=True`.
        new_weights : list[dict[str, array_like]], optional
            Agent's new set of weights after each update. Only returned if 
            `return_info=True`.
        '''
        logger = logger or logging.getLogger('dummy')
        ok = True
        results = []

        for e in range(n_epochs):
            self._epoch_n = e  # just for logging
            try:
                results.append(
                    self.learn_one_epoch(
                        n_episodes=n_episodes,
                        perturbation_decay=perturbation_decay,
                        seed=None if seed is None else seed + n_episodes * e,
                        logger=logger,
                        return_info=return_info)
                )
            except (MPCSolverError, UpdateError) as ex:
                if throw_on_exception:
                    raise ex
                ok = False
                logger.error(f'Suppressing agent \'{self.name}\': {ex}')
                break

        if not results:
            return (ok, np.nan, [], []) if return_info else (ok, np.nan)

        if not return_info:
            return ok, np.stack(results, axis=0)

        returns, grads, weightss = list(zip(*results))
        return ok, np.stack(returns, axis=0), grads, weightss

    def _init_learnable_pars(
        self, init_pars: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        '''Initializes the learnable parameters of the MPC.'''
        required_pars = sorted(set(self._Q.pars).intersection(
            self._V.pars).difference({'x0', 'xf'}).difference(self.fixed_pars))
        self.weights = RLParameterCollection(
            *(RLParameter(
                name, *init_pars[name], self.V.pars[name], self.Q.pars[name])
              for name in required_pars)
        )

    def _init_learning_rate(self) -> None:
        cfg = self.config
        if cfg is None or not hasattr(cfg, 'lr'):
            return
        n_pars, n_theta = len(self.weights), self.weights.n_theta
        lr = np.asarray(cfg.lr).squeeze()
        if lr.ndim == 0:
            lr = np.full((n_theta,), lr)
        elif lr.size == n_pars and lr.size != n_theta:
            lr = np.concatenate(
                [np.full(p.size, r) for p, r in zip(self.weights, lr)])
        assert lr.shape == (n_theta,), 'Learning rate must have the same ' \
            'size as the learnable parameter vector.'
        cfg.lr = lr

    def _merge_mpc_pars_callback(self) -> dict[str, np.ndarray]:
        return self.weights.values(as_dict=True)

    @staticmethod
    def _get_percentage_bounds(
        theta: np.ndarray,
        theta_bounds: np.ndarray,
        max_perc_update: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_delta = np.maximum(np.abs(max_perc_update * theta), 0.1)
        lb = np.maximum(theta_bounds[:, 0], theta - max_delta)
        ub = np.minimum(theta_bounds[:, 1], theta + max_delta)
        return lb, ub
