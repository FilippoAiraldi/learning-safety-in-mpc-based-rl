import numpy as np
from dataclasses import dataclass, field
from envs.base_env import BaseEnv
from gym import spaces
from typing import Union


@dataclass(frozen=True)
class QuadRotorEnvConfig:
    '''
    Quadrotor environments configuration parameters. The model parameters must 
    be nonnegative, whereas the disturbance parameter 'winds' is a dictionary 
    with each gust's altitude and strength. Also the bounds on state and action
    are included, as well as the numerical tolerance.
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
        default_factory=lambda: {1.5: 0.8, 2.5: 0.5, 3: -0.6})

    # simulation
    x0: np.ndarray = field(default_factory=lambda: np.array(
        [0, 0, 3.5, 0, 0, 0, np.deg2rad(10), np.deg2rad(-10), 0, 0]))
    xf: np.ndarray = field(default_factory=lambda: np.array(
        [3, 3, 0.2, 0, 0, 0, 0, 0, 0, 0]))

    # constraints
    soft_state_constraints: bool = True
    x_bounds: np.ndarray = field(
        default_factory=lambda: np.array([[-0.5, 3.5],
                                          [-0.5, 3.5],
                                          [-0.175, 4],
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

    # termination conditions
    termination_N: int = 5
    termination_error: float = 0.5


class QuadRotorEnv(BaseEnv[np.ndarray, np.ndarray]):
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
    | Num | Observation                          | Min    | Max | Unit          |
    |-----|--------------------------------------|--------|-----|---------------|
    | 0   | position along the x-axis            | -0.5   | 3.5 | position (m)  |
    | 1   | position along the y-axis            | -0.5   | 3.5 | position (m)  |
    | 2   | position along the z-axis (altitude) | -0.175 | 4   | position (m)  |
    | 3   | velocity along the x-axis            | -Inf   | Inf | speed (m/s)   |
    | 4   | velocity along the y-axis            | -Inf   | Inf | speed (m/s)   |
    | 5   | velocity along the z-axis            | -Inf   | Inf | speed (m/s)   |
    | 6   | pitch                                | -30°   | 30° | angle (rad)   |
    | 7   | roll                                 | -30°   | 30° | angle (rad)   |
    | 8   | pitch rate                           | -Inf   | Inf | speed (rad/s) |
    | 8   | roll rate                            | -Inf   | Inf | speed (rad/s) |
    The constraints can be changed, as well as made soft with the appropriate 
    flag. In this case, the observation space becomes unbounded.

    ### Action Space
    There are 3 continuous deterministic actions:
    | Num | Action                               | Min  | Max | Unit          |
    |-----|--------------------------------------|------|-----|---------------|
    | 0   | desired pitch                        | -pi  | pi  | angle (rad)   |
    | 1   | desired roll                         | -pi  | pi  | angle (rad)   |
    | 2   | desired vertical acceleration        | 0    | 2*g | acc. (m/s^2)  |
    Again, these constraints can be changed.

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
    error and stays within this error bound for the specified amount of steps.

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
        config: Union[dict, QuadRotorEnvConfig] = None,
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
        if config is None:
            config = QuadRotorEnvConfig()
        elif isinstance(config, dict):
            keys = QuadRotorEnvConfig.__dataclass_fields__.keys()
            config = QuadRotorEnvConfig(
                **{k: config[k] for k in keys if k in config})
        self.config = config

        # create dynamics matrices
        self.nw = len(config.winds)
        self._A = config.T * np.block([
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))],
            [np.zeros((2, 6)), np.eye(2) * config.g, np.zeros((2, 2))],
            [np.zeros((1, 10))],
            [np.zeros((2, 6)), -np.diag((config.pitch_d, config.roll_d)),
             np.eye(2)],
            [np.zeros((2, 6)), -np.diag((config.pitch_dd, config.roll_dd)),
             np.zeros((2, 2))]
        ]) + np.eye(10)
        self._B = config.T * np.block([
            [np.zeros((5, 3))],
            [0, 0, config.thrust_coeff],
            [np.zeros((2, 3))],
            [config.pitch_gain, 0, 0],
            [0, config.roll_gain, 0]
        ])
        self._C = config.T * np.vstack((
            list(config.winds.values()),
            np.zeros((9, self.nw))
        ))
        self._e = np.vstack((
            np.zeros((5, 1)), - config.T * config.g, np.zeros((4, 1))))

        # create spaces
        self.observation_space = spaces.Box(
            low=(-np.inf
                 if config.soft_state_constraints else
                 config.x_bounds[:, 0]),
            high=(np.inf
                  if config.soft_state_constraints else
                  config.x_bounds[:, 1]),
            shape=(self.nx,),
            dtype=np.float64)
        self.action_space = spaces.Box(
            low=config.u_bounds[:, 0],
            high=config.u_bounds[:, 1],
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
    def x(self) -> np.ndarray:
        '''Gets the current state of the quadrotor.'''
        return self._x.copy()

    @x.setter
    def x(self, val: np.ndarray) -> None:
        '''Sets the current state of the quadrotor.'''
        assert self.observation_space.contains(val), f'Invalid state {val}.'
        self._x = val.copy()

    def error(self, x: np.ndarray) -> float:
        '''Error of the given state w.r.t. the final position.'''
        # return np.linalg.norm(x - self.config.xf)
        # give more weight to pitch and roll
        return np.sqrt(np.inner(np.square(x - self.config.xf),
                                [1, 1, 1, 1, 1, 1, 1e1, 1e1, 1, 1]))

    def phi(self, alt: Union[float, np.ndarray]) -> np.ndarray:
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
            alt = alt.squeeze()
            assert alt.ndim == 1, 'Altitudes must be a vector'

        return np.vstack([
            np.exp(-np.square(alt - h)) for h in self.config.winds
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
            x0 = self.config.x0
        if xf is None:
            xf = self.config.xf
        assert (self.observation_space.contains(x0) and
                self.observation_space.contains(xf)), \
            'Invalid initial or final state.'
        self.x = x0
        self.config.__dict__['x0'] = x0
        self.config.__dict__['xf'] = xf
        self._n_within_termination = 0
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
        u = u.squeeze()  # in case a row or col was passed
        assert self.action_space.contains(u), 'Invalid action.'

        # compute new state: x+ = A*x + B*u + C*phi(s[2])*w + e
        self.x = (
            self._A @ self.x.reshape((-1, 1)) +
            self._B @ u.reshape((-1, 1)) +
            self._C @ self.phi(self.x[2]) * self.np_random.random() +
            self._e
        ).flatten()

        # compute cost
        error = self.error(self.x)
        cost = float(
            error +
            2 * np.linalg.norm(u) +
            1e2 * (
                np.maximum(0, self.config.x_bounds[:, 0] - self.x) +
                np.maximum(0, self.x - self.config.x_bounds[:, 1])
            ).sum()
        )

        # check if done
        within_bounds = ((self.config.x_bounds[:, 0] <= self._x) &
                         (self._x <= self.config.x_bounds[:, 1])).all()
        if error <= self.config.termination_error and within_bounds:
            self._n_within_termination += 1
        else:
            self._n_within_termination = 0
        done = self._n_within_termination >= self.config.termination_N
        #
        return self.x, cost, done, {'error': error}

    def render(self):
        raise NotImplementedError('Render method unavailable.')
