from abc import ABC
from itertools import count
import numpy as np
import casadi as cs
from mpc import QuadRotorMPC, QuadRotorMPCPars, Solution
from envs import QuadRotorEnv


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
        mpc_pars: dict | QuadRotorMPCPars = None,
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
        mpc_pars : dict, QuadRotorPars
            A set of parameters for the agent's MPC. If not given, the default 
            ones are used.
        '''
        super().__init__()
        self.id = next(self._ids)
        self.name = f'Agent{self.id}' if agentname is None else agentname
        self.env = env

        # initialize MPCs
        self.Q = QuadRotorMPC(env, pars=mpc_pars, type='Q')
        self.V = QuadRotorMPC(env, pars=mpc_pars, type='V')
        self.last_solution: Solution = None

        # initialize learnable weights/parameters
        self._init_weights(init_pars=init_pars)

    @property
    def w(self) -> np.ndarray:
        '''Gets the numerical weights in vector form.'''
        return np.hstack(list(self.weights['value'].values()))

    @property
    def bound(self) -> np.ndarray:
        '''Gets the numerical bounds for the weights in matrix form.'''
        return np.vstack(list(self.weights['bound'].values()))

    def w_sym(self, type: str) -> cs.SX:
        '''Gets the symbolical weights in vector form for either Q or V.'''
        n = 'symQ' if type == 'Q' else 'symV'
        return cs.vertcat(*self.weights[n].values())

    def solve_mpc(
        self,
        type: str,
        state: np.ndarray,
        pars: dict[str, np.ndarray],
        rlpars: dict[str, np.ndarray] = None,
        sol0: dict[str, np.ndarray] = None
    ) -> tuple[np.ndarray, Solution]:
        # if not provided, use the agent's latest weights
        if rlpars is None:
            rlpars = self.weights['value']

        # if provided, use vals0 to warmstart the MPC. If not provided,
        # use the last_sol field. If the latter is not available yet,
        # just use some default values
        if sol0 is None:
            if self.last_sol is None:
                sol0 = {
                    'x': np.tile(state.reshape(-1, 1), (1, self.Q.Np)),
                    'u': 0,
                    'slack': 0
                }
            else:
                sol0 = self.last_sol

        # merge RL pars and common pars
        pars |= rlpars
        pars['x0'] = state

        # call the MPC
        mpc: QuadRotorMPC = getattr(self, type)
        self.last_solution = mpc.solve(pars, sol0)
        return self.last_solution['u'][:, 0], self.last_solution

    def _init_weights(self, init_pars: dict[str, np.ndarray] = None) -> None:
        # learnable parameters are:
        #   - model pars: 'thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
        #                 'roll_d', 'roll_dd', 'roll_gain'
        #   - cost pars: 'w_L', 'w_V', 'w_s', 'w_s_f', 'xf'
        #   - constraint pars: 'backoff'
        # NOTE: all these parameters must be column vectors. Cannot deal with
        # multidimensional matrices!
        if init_pars is None:
            init_pars = {}

        # create bounds
        bounds = {}
        names_and_lb = [
            [('thrust_coeff', 'pitch_d', 'pitch_dd', 'pitch_gain',
              'roll_d', 'roll_dd', 'roll_gain'), 1e-1],
            [('w_L', 'w_V', 'w_s', 'w_s_f'), 1e-6],
            [('xf',), -np.inf],
            [('backoff',), 0]
        ]
        for names, lb in names_and_lb:
            for name in names:
                p = self.Q.pars[name]
                assert p.is_column(), \
                    f'Invalid parameter {name} shape; must be a column vector.'
                bounds[name] = np.broadcast_to([lb, np.inf], (p.shape[0], 2))

        # create initial values and references to symbols
        values, symQ, symV = {}, {}, {}
        for name in bounds:
            symQ[name] = self.Q.pars[name]
            symV[name] = self.V.pars[name]
            values[name] = np.broadcast_to(init_pars.get(name, np.nan),
                                           symV[name].shape[0])

        self.weights = {
            'bound': bounds,
            'value': values,
            'symQ': symQ,
            'symV': symV
        }

    def __str__(self):
        '''Returns the agent name.'''
        return f'<{type(self).__name__}: {self.name}>'

    def __repr__(self):
        '''Returns the string representation of the Agent.'''
        return str(self)