from gym import spaces
import numpy as np
from envs.base_env import BaseEnv
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuadRotorEnvPars:
    '''
    Quadrotor environments parameters. The model parameters must be 
    nonnegative, whereas the disturbance parameter 'winds' is a dictionary with 
    each wind's altitude and strength. Also the bounds on state and action are 
    included, as well as the numerical tolerance.
    '''

    # model parameters
    T: float = 0.2
    g: float = 9.81
    thrust_coeff: float = 1.4
    pitch_d: float = 10
    pitch_dd: float = 8
    pitch_gain: float = 10
    roll_d: float = 10
    roll_dd: float = 8
    roll_gain: float = 10

    # disturbance parameters
    winds: dict[float, float] = field(
        default_factory=lambda: {-10: 0.95, 0: -0.35, 10: 0.3})

    # simulation
    x0: np.ndarray = field(default_factory=lambda: np.array(
        [0, 0, 20, 0, 0, 0, 0, 0, 0, 0]))
    xf: np.ndarray = field(default_factory=lambda: np.array(
        [20, 20, -20, 0, 0, 0, np.deg2rad(10), np.deg2rad(10), 0, 0]))

    # constraints
    soft_state_constraints: bool = True
    x_bounds: np.ndarray = field(
        default_factory=lambda: np.array([[-25, 25],
                                          [-25, 25],
                                          [-25, 25],
                                          [-np.inf, np.inf],
                                          [-np.inf, np.inf],
                                          [-np.inf, np.inf],
                                          [np.deg2rad(-30), np.deg2rad(30)],
                                          [np.deg2rad(-30), np.deg2rad(30)],
                                          [-np.inf, np.inf],
                                          [-np.inf, np.inf]]))
    u_bounds: np.ndarray = field(
        default_factory=lambda: np.array([[-np.pi, np.pi],
                                          [-np.pi, np.pi],
                                          [0, 2 * 9.81]]))

    # others
    num_tol: float = 1e1


class QuadRotorEnv(BaseEnv):
    '''
    ### Description
    The QuadRotorEnv consists in control problem whose goal is to drive a
    quadrotor to a specific terminal location. Altitude-dependent winds 
    stochastically disturb the dynamics. The model is most accurate when the 
    pitch and roll do not exceed 30° in both directions. However, these 
    constraints can be momentarily violated, so no hard limit is imposed within
    the environment itself. On the contrary, the control authority is limited.

    ### Observation Space
    The observation is a `ndarray` with shape `(10,)` where the elements 
    correspond to the following states of the quadrotor:
    | Num | Observation                          | Min  | Max | Unit          |
    |-----|--------------------------------------|------|-----|---------------|
    | 0   | position along the x-axis            | -25  | 25  | position (m)  |
    | 1   | position along the y-axis            | -25  | 25  | position (m)  |
    | 2   | position along the z-axis (altitude) | -25  | 25  | position (m)  |
    | 3   | velocity along the x-axis            | -Inf | Inf | speed (m/s)   |
    | 4   | velocity along the y-axis            | -Inf | Inf | speed (m/s)   |
    | 5   | velocity along the z-axis            | -Inf | Inf | speed (m/s)   |
    | 6   | pitch                                | -30° | 30° | angle (rad)   |
    | 7   | roll                                 | -30° | 30° | angle (rad)   |
    | 8   | pitch rate                           | -Inf | Inf | speed (rad/s) |
    | 8   | roll rate                            | -Inf | Inf | speed (rad/s) |
    The constraints can be made soft with the appropriate flag. In this case, 
    the observation space becomes unbounded.

    ### Action Space
    There are 3 continuous deterministic actions:
    | Num | Action                               | Min  | Max | Unit          |
    |-----|--------------------------------------|------|-----|---------------|
    | 0   | desired pitch                        | -pi  | pi  | angle (rad)   |
    | 1   | desired roll                         | -pi  | pi  | angle (rad)   |
    | 2   | desired vertical acceleration        | 0    | 2*g | acc. (m/s^2)  |

    ### Transition Dynamics:
    Given an action, the quadrotor follows the following transition dynamics:
        x+ = A*x + B*u + C*phi(s[2])*w + e
    where x and x+ are the current and next state, u is the control action, phi
    is a nonlinear term modulating wind disturbance strength and w is a 
    uniformly distributed random variable. A, B, C and e are system elements.

    ### Cost:
    ...

    ### Initial And Final State
    The initial and final states are constant across resets, unless manually 
    specified.

    ### Episode End
    The episode ends if the state approaches the final one within the specified
    tolerance.

    ### Arguments
    ```
    gym.make('QuadRotor')
    ```
    '''
    spec: dict = None
    nx: int = 10
    nu: int = 3

    def __init__(
        self,
        pars: dict | QuadRotorEnvPars = None,
    ) -> None:
        '''
        This environment simulates a 10-state quadrotor system with limited 
        input authority whose goal is to reach a target position, from an 
        initial position.

        Parameters
        ----------
        pars : dict, QuadRotorPars
            A set of parameters for the quadrotor model and disturbances. If 
            not given, the default ones are used.
        '''
        super().__init__()
        if pars is None:
            pars = QuadRotorEnvPars()
        elif isinstance(pars, dict):
            keys = QuadRotorEnvPars.__dataclass_fields__.keys()
            pars = QuadRotorEnvPars(**{k: pars[k] for k in keys if k in pars})
        self.pars = pars

        # create dynamics matrices
        self.nw = len(pars.winds)
        self._A = pars.T * np.block([
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))],
            [np.zeros((2, 6)), np.eye(2) * pars.g, np.zeros((2, 2))],
            [np.zeros((1, 10))],
            [np.zeros((2, 6)), -np.diag((pars.pitch_d, pars.roll_d)),
             np.eye(2)],
            [np.zeros((2, 6)), -np.diag((pars.pitch_dd, pars.roll_dd)),
             np.zeros((2, 2))]
        ]) + np.eye(10)
        self._B = pars.T * np.block([
            [np.zeros((5, 3))],
            [0, 0, pars.thrust_coeff],
            [np.zeros((2, 3))],
            [pars.pitch_gain, 0, 0],
            [0, pars.roll_gain, 0]
        ])
        self._C = pars.T * np.vstack((
            list(pars.winds.values()),
            np.zeros((9, self.nw))
        ))
        self._e = np.vstack((
            np.zeros((5, 1)), - pars.T * pars.g, np.zeros((4, 1))))

        # create spaces
        self.observation_space = spaces.Box(
            low=(-np.inf 
                 if pars.soft_state_constraints else 
                 pars.x_bounds[:, 0]),
            high=(np.inf
                  if pars.soft_state_constraints else 
                  pars.x_bounds[:, 1]),
            shape=(self.nx,),
            dtype=np.float64)
        self.action_space = spaces.Box(
            low=pars.u_bounds[:, 0],
            high=pars.u_bounds[:, 1],
            shape=(self.nu,),
            dtype=np.float64)

    @property
    def A(self) -> np.ndarray:
        '''Returns the dynamics A matrix.'''
        return self._A.copy()

    @property
    def B(self) -> np.ndarray:
        '''Returns the dynamics B matrix.'''
        return self._B.copy()

    @property
    def C(self) -> np.ndarray:
        '''Returns the dynamics C matrix.'''
        return self._C.copy()

    @property
    def e(self) -> np.ndarray:
        '''Returns the dynamics e vector.'''
        return self._e.copy()

    @property
    def error(self) -> float:
        '''Error of the current state to the final position.'''
        return np.linalg.norm(self._x - self.pars.xf)

    @property
    def x(self) -> np.ndarray:
        '''Gets the current state of the quadrotor.'''
        return self._x

    @x.setter
    def x(self, val: np.ndarray) -> None:
        '''Sets the current state of the quadrotor.'''
        assert self.observation_space.contains(val), f'Invalid state {val}.'
        self._x = val

    def phi(self, alt: float | np.ndarray) -> np.ndarray:
        '''
        Computes the wind disturbance's radial basis functions at the given 
        altitude.

        # Parameters
        ----------
        altitude : array_like
            Altitude value at which to evaluate the functions.

        Returns
        -------
        y : array_like
            The value of the functions.
        '''
        if isinstance(alt, np.ndarray):
            alt = np.squeeze(alt)
            assert alt.ndim == 1, 'Altitudes must be a vector'

        return np.vstack([
            np.exp(-np.square(alt - h) / 50) for h in self.pars.winds.keys()
        ])

    def reset(
        self,
        seed: int = None,
        x0: np.ndarray = None,
        xf: np.ndarray = None
    ) -> np.ndarray:
        '''
        Resets the quadrotor environment. 

        # Parameters
        ----------
        seed : int, optional
            Random number generator seed.
        x0, xf : array_like, optional
            Sets the initial and terminal states of the simulation. If not 
            passed, the conditions are chosen from default.

        Returns
        -------
        x : array_like
            The value of the functions.
        '''
        super().reset(seed=seed)
        self.observation_space.seed(seed=seed)
        self.action_space.seed(seed=seed)
        if x0 is None:
            x0 = self.pars.x0
        if xf is None:
            xf = self.pars.xf
        assert (self.observation_space.contains(x0) and
                self.observation_space.contains(xf)), \
                    'Invalid initial or final state.'
        self.x = x0
        self.pars.__dict__['x0'] = x0
        self.pars.__dict__['xf'] = xf
        return x0

    def step(self, u: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        '''
        Steps the quadrotor environment. 

        # Parameters
        ----------
        u : array_like
            Action to apply to the quadrotor.

        Returns
        -------
        new_state, cost, done, info : array_like, float, bool, dict
            A tuple containing the new state of the quadrotor, the 
            instantenuous cost of taking this action in this state, the 
            termination flag and a dictionary of information.
        '''
        u = np.squeeze(u)  # in case a row or col was passed
        assert self.action_space.contains(u), 'Invalid action.'

        # compute new state: x+ = A*x + B*u + C*phi(s[2])*w + e
        self.x = (
            self._A @ self.x.reshape((-1, 1)) +
            self._B @ u.reshape((-1, 1)) +
            self._C @ self.phi(self.x[2]) * self.np_random.random() +
            self._e
        ).flatten()
        assert self.observation_space.contains(self.x), 'Invalid state.'

        # compute cost
        # TODO: squared norm to desired position (no need to normalize)
        # + input usage (we can normalize here)
        # + constraint violation
        cost = np.nan

        # check if done
        done = self.error <= self.pars.num_tol
        #
        return self.x, cost, done, {}

    def render(self):
        raise NotImplementedError('Render method unavailable.')