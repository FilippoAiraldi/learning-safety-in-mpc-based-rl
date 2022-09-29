import numpy as np
from abc import ABC
from envs import QuadRotorEnv
from gym import Env
from gym.utils.seeding import np_random
from itertools import count
from mpc import QuadRotorMPC, QuadRotorMPCConfig, Solution
from typing import Any, Optional, Union
from util.configurations import init_config


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
        agent_config: Union[dict, Any] = None,
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
        agent_config : dict, ConfigType
            A set of parameters for the quadrotor agent. If not given, the 
            default ones are used.
        fixed_pars : dict[str, np.ndarray]
            A dictionary containing MPC parameters that are fixed.
        mpc_config : dict, QuadRotorMPCConfig
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

        # initialize default MPC parameters
        self.fixed_pars = {} if fixed_pars is None else fixed_pars
        default_pars = [
            ('xf', env.config.xf),
            ('backoff', 0.05)
        ]
        for p, default in default_pars:
            if p not in self.fixed_pars:
                self.fixed_pars[p] = default

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
        observation_mean_std: tuple[np.ndarray, np.ndarray] = (0.0, 1.0),
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
        observation_mean_std : tuple[array_like, array_like], optional
            Mean and std for normalization of the observations before calling 
            the agent.
        seed : int or list[int], optional
            RNG seed.

        Returns
        -------
        returns : array_like
            An array of the accumulated rewards/costs for each episode
        '''
        returns = np.zeros(n_eval_episodes)
        mean, std = observation_mean_std
        seeds = self._make_seed_list(seed, n_eval_episodes)

        for e in range(n_eval_episodes):
            state = env.reset(seed=seeds[e])
            self.reset()
            truncated, terminated = False, False

            while not (truncated or terminated):
                state = (state - mean) / (std + 1e-8)
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
