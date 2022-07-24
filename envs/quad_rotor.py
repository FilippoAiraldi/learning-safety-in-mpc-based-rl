import gym
from gym import spaces
import numpy as np
from envs.base_env import BaseEnv
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuadRotorPars:
    '''
    Quadrotor environments parameters. The model parameters must be 
    nonnegative, whereas the disturbance parameter 'winds' is a dictionary with 
    each wind's altitude and strength.
    '''
    # model pars
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

    def __post_init__(self) -> None:
        assert all(
            p >= 0 for p in self.__dict__.values()
            if not isinstance(p, dict)), 'Parameters must be nonnegative'


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
    | 0   | position along the x-axis            | -Inf | Inf | position (m)  |
    | 1   | position along the y-axis            | -Inf | Inf | position (m)  |
    | 2   | position along the z-axis (altitude) | -Inf | Inf | position (m)  |
    | 3   | velocity along the x-axis            | -Inf | Inf | speed (m/s)   |
    | 4   | velocity along the y-axis            | -Inf | Inf | speed (m/s)   |
    | 5   | velocity along the z-axis            | -Inf | Inf | speed (m/s)   |
    | 6   | pitch                                | -Inf | Inf | angle (rad)   |
    | 7   | roll                                 | -Inf | Inf | angle (rad)   |
    | 8   | pitch rate                           | -Inf | Inf | speed (rad/s) |
    | 8   | roll rate                            | -Inf | Inf | speed (rad/s) |

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

    def __init__(
        self,
        pars: dict | QuadRotorPars = None,
        tol: float = 1e1
    ) -> None:
        '''
        This environment simulates a 10-state quadrotor system with limited 
        input authority whose goal is to reach a target position, from an 
        initial position.

        Parameters
        ----------
        pars : dict, QuadRotorPars
            A set of parmeters for the quadrotor model and disturbances. If not
            given, the default ones are used.
        tol : float
            Numerical tolerance used to check episode termination.
        '''
        super().__init__()
        if pars is None:
            pars = QuadRotorPars()
        elif isinstance(pars, dict):
            pars = QuadRotorPars(**pars)
        self.tol = tol

        # create dynamics matrices
        self.pars = pars
        self.nx, self.nu = 10, 3
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
            low=-np.inf,
            high=np.inf,
            shape=(self.nx,),
            dtype=np.float64)
        self.action_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, 0]),
            high=np.array([np.pi, np.pi, 2 * pars.g]),
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
        return np.linalg.norm(self._x - self._xf)

    @property
    def x(self) -> np.ndarray:
        '''Gets the current state of the quadrotor.'''
        return self._x

    @x.setter
    def x(self, val: np.ndarray) -> None:
        '''Sets the current state of the quadrotor.'''
        assert self.observation_space.contains(val), f'Invalid state {val}.'
        self._x = val

    @property
    def x0(self) -> np.ndarray:
        '''Gets the initial state of the episode.'''
        return self._x0

    @property
    def xf(self) -> np.ndarray:
        '''Gets the termination state of the episode.'''
        return self._xf

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
        x0 : array_like, optional
            Sets the initial state of the simulation. If not passed, the 
            initial condition is automatically and deterministically chosen.
        xf : array_like, optional
            Sets the terminal state of the simulation. If not passed, the 
            terminal condition is automatically and deterministically chosen.

        Returns
        -------
        x : array_like
            The value of the functions.
        '''
        super().reset(seed=seed)
        self.observation_space.seed(seed=seed)
        self.action_space.seed(seed=seed)
        if x0 is None:
            x0 = np.zeros(self.nx)
            x0[2] = 50  # altitude
        if xf is None:
            xf = np.zeros(self.nx)
            xf[:2] = 20  # x, y
            xf[2] = -20  # altitude
            xf[6:8] = np.deg2rad(10)  # pitch, roll
        self.x, self._x0, self._xf = x0, x0, xf
        assert (self.observation_space.contains(self._x0) and
                self.observation_space.contains(self._xf)), \
            'Invalid initial or final state.'
        return self.x

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
        # TODO: check how Cai did it in his naval mission application
        # and remember to penalizde constraint violation
        cost = np.nan

        # check if done
        done = self.error <= self.tol
        #
        return self.x, cost, done, {}

    def render(self):
        raise NotImplementedError('Render method unavailable.')
