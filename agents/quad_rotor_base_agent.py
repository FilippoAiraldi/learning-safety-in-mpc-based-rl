import numpy as np
from abc import ABC
from itertools import count
from gym.utils.seeding import np_random
from mpc import QuadRotorMPC, QuadRotorMPCConfig, Solution
from envs import QuadRotorEnv
from agents import RLParameter, RLParameterCollection


class QuadRotorBaseAgent(ABC):
    '''
    Abstract base agent class that contains the two MPC function approximators 
    Q and V. 
    '''
    _ids = count(0)

    def __init__(
        self,
        env: QuadRotorEnv,
        agentname: str = None,
        init_pars: dict[str, np.ndarray] = None,
        fixed_pars: dict[str, np.ndarray] = None,
        mpc_config: dict | QuadRotorMPCConfig = None,
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
        init_pars : dict[str, np.ndarray]
            A dictionary containing for each RL parameter the corresponding 
            initial value.
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

        # save some stuff
        self.env = env
        self.fixed_pars = {} if fixed_pars is None else fixed_pars
        self.np_random, _ = np_random(seed)
        self.perturbation_chance = 0.2
        self.perturbation_strength = 0.1
        self.last_solution: Solution = None

        # initialize MPCs
        self._Q = QuadRotorMPC(env, config=mpc_config, type='Q')
        self._V = QuadRotorMPC(env, config=mpc_config, type='V')

        # initialize learnable weights/parameters
        self.init_mpc_parameters(init_pars=init_pars)

    @property
    def V(self) -> QuadRotorMPC:
        '''Gets the V action-value function approximation MPC scheme.'''
        return self._V

    @property
    def Q(self) -> QuadRotorMPC:
        '''Gets the Q action-value function approximation MPC scheme.'''
        return self._Q

    def solve_mpc(
        self,
        type: str,
        state: np.ndarray = None,
        sol0: Solution = None,
    ) -> tuple[np.ndarray, Solution]:
        '''
        Solves the MPC optimization problem embedded in the agent.

        Parameters
        ----------
        type : 'Q' or 'V'
            Type of MPC function approximation to run.
        state : array_like, optional
            Environment's state for which to solve the MPC problem. If not 
            given, the current state of the environment is used.
        sol0 : Solution
            Last numerical solution of the MPC used to warmstart. If not given,
            a heuristic is used.
        '''
        mpc: QuadRotorMPC = getattr(self, type)

        # if the state which to solve the MPC for is not provided, use current
        if state is None:
            state = self.env.x

        # merge all parameters in a single dict
        pars = (self.weights.values(as_dict=True) |
                self.fixed_pars | {'x0': state})

        # if provided, use vals0 to warmstart the MPC. If not provided,
        # use the last_sol field. If the latter is not available yet,
        # just use some default values
        if sol0 is None:
            if self.last_solution is None:
                sol0 = {
                    'x': np.tile(state,
                                 (mpc.vars['x'].shape[1], 1)).T,
                    'u': np.tile([0, 0, self.weights['g'].value.item()],
                                 (mpc.vars['u'].shape[1], 1)).T,
                    'slack': 0
                }
            else:
                sol0 = self.last_solution.vals

        # call the MPC
        self.last_solution = mpc.solve(pars, sol0)

        # get the optimal action
        u_opt = self.last_solution.vals['u'][:, 0]
        return u_opt, self.last_solution

    def predict(
        self,
        state: np.ndarray = None,
        deterministic: bool = False,
        perturb_gradient: bool = True,
        **solve_mpc_kwargs
    ) -> tuple[np.ndarray, np.ndarray, Solution]:
        '''
        Computes the optimal action for the given state by solving the MPC 
        scheme V and predicts the next state.

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
        solution : Solution
            Solution object containing some information on the solution.
        '''
        perturbation_in_dict = 'perturbation' in self.fixed_pars
        if perturbation_in_dict:
            self.fixed_pars['perturbation'] = 0

        if deterministic or self.np_random.random() > self.perturbation_chance:
            # just solve the V scheme without noise
            u, sol = self.solve_mpc(type='V', state=state, **solve_mpc_kwargs)
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

            u, sol = self.solve_mpc(type='V', state=state, **solve_mpc_kwargs)

            # otherwise, directly perturb the action
            if not perturb_gradient:
                u = np.clip(u + rng, u_bnd[:, 0], u_bnd[:, 1])

        x_next = sol.vals['x'][:, 0]
        return u, x_next, sol

    def init_mpc_parameters(
            self, init_pars: dict[str, np.ndarray] = None) -> None:
        '''
        Initializes the learnable parameters of the MPC.

        Parameters
        ----------
        init_pars : dict[str, array_like]
            A dict containing, for each learnable parameter in the MPC scheme, 
            its initial value.
        '''
        # learnable parameters are:
        #   - model pars: 'thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
        #                 'roll_d', 'roll_dd', 'roll_gain'
        #   - cost pars: 'w_L', 'w_V', 'w_s', 'w_s_f', 'xf'
        #   - constraint pars: 'backoff'
        # NOTE: all these parameters must be column vectors. Cannot deal with
        # multidimensional matrices!
        if init_pars is None:
            init_pars = {}

        # create initial values (nan if not provided), bounds and references to
        # symbols
        names_and_bnds = [
            # model
            ('g', (1, 40)),
            ('thrust_coeff', (0.1, 4)),
            ('pitch_d', (1, 40)),
            ('pitch_dd', (1, 40)),
            ('pitch_gain', (1, 40)),
            ('roll_d', (1, 40)),
            ('roll_dd', (1, 40)),
            ('roll_gain', (1, 40)),
            # cost
            ('w_L', (1e-3, np.inf)),
            ('w_V', (1e-3, np.inf)),
            ('w_s', (1e-3, np.inf)),
            ('w_s_f', (1e-3, np.inf)),
            ('xf', self.env.config.x_bounds),
            # others
            ('backoff', (0, 0.2))
        ]
        self.weights = RLParameterCollection(
            RLParameter(name, init_pars.get(name, np.nan), bnd,
                        self.V.pars[name], self.Q.pars[name])
            for name, bnd in names_and_bnds
        )

    def __str__(self) -> str:
        '''Returns the agent name.'''
        return f'<{type(self).__name__}: {self.name}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the Agent.'''
        return str(self)
