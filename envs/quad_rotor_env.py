import casadi as cs
import numpy as np
from dataclasses import dataclass, field
from envs.base_env import BaseEnv
from gym import spaces
from typing import Union
from util.configurations import init_config


@dataclass(frozen=True)
class QuadRotorEnvConfig:
    '''
    Quadrotor environments configuration parameters. The model parameters must 
    be nonnegative, whereas the disturbance parameter 'winds' is a dictionary 
    with each gust's altitude and strength. Also the bounds on state and action
    are included.
    '''
    # model parameters
    T: float = 0.1
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
        default_factory=lambda: {1: 1.45, 2: 1.1, 3: 1.25})

    # simulation
    x0: np.ndarray = field(default_factory=lambda: np.array(
        [0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0]))
    xf: np.ndarray = field(default_factory=lambda: np.array(
        [3, 3, 0.2, 0, 0, 0, 0, 0, 0, 0]))

    # constraints
    soft_constraints: bool = True
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
    | N | Observation                          | Min    | Max | Unit          |
    |---|--------------------------------------|--------|-----|---------------|
    | 0 | position along the x-axis            | -0.5   | 3.5 | position (m)  |
    | 1 | position along the y-axis            | -0.5   | 3.5 | position (m)  |
    | 2 | position along the z-axis (altitude) | -0.175 | 4   | position (m)  |
    | 3 | velocity along the x-axis            | -Inf   | Inf | speed (m/s)   |
    | 4 | velocity along the y-axis            | -Inf   | Inf | speed (m/s)   |
    | 5 | velocity along the z-axis            | -Inf   | Inf | speed (m/s)   |
    | 6 | pitch                                | -30°   | 30° | angle (rad)   |
    | 7 | roll                                 | -30°   | 30° | angle (rad)   |
    | 8 | pitch rate                           | -Inf   | Inf | speed (rad/s) |
    | 8 | roll rate                            | -Inf   | Inf | speed (rad/s) |
    The constraints can be changed, as well as made soft with the appropriate 
    flag. In this case, the observation space becomes unbounded.

    ### Action Space
    There are 3 continuous deterministic actions:
    | N | Action                               | Min    | Max | Unit          |
    |---|--------------------------------------|--------|-----|---------------|
    | 0 | desired pitch                        | -pi    | pi  | angle (rad)   |
    | 1 | desired roll                         | -pi    | pi  | angle (rad)   |
    | 2 | desired vertical acceleration        | 0      | 2*g | acc. (m/s^2)  |
    Again, these constraints can be changed and made soft.

    ### Transition Dynamics:
    Given an action, the quadrotor follows the following transition dynamics:
    ```
    x+ = A*x + B*u + C*phi(s[2])*w + e
    ```
    where `x` and `x+` are the current and next state, `u` is the control 
    action, `phi` is a nonlinear term modulating wind disturbance strength and 
    `w` is a uniformly distributed random variable. `A`, `B`, `C` and `e` are 
    system elements.

    ### Cost:
    The cost consists of three source: positional error (proportional to the 
    distance of the quadrotor to the final position), control action usage (
    proportional to the magnitude of the control action), and constraint 
    violation (proportional to violations of both state and action bounds).

    ### Initial And Final State
    The initial and final states are constant across resets, unless manually 
    specified.

    ### Episode End
    The episode ends if the state approaches the final one within the specified
    error and stays within this error bound for the specified amount of steps.
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
        config = init_config(config, QuadRotorEnvConfig)
        self._config = config

        # create dynamics matrices
        self._A, self._B, self._C, self._e = self.get_dynamics(
            g=config.g,
            thrust_coeff=config.thrust_coeff,
            pitch_d=config.pitch_d,
            pitch_dd=config.pitch_dd,
            pitch_gain=config.pitch_gain,
            roll_d=config.roll_d,
            roll_dd=config.roll_dd,
            roll_gain=config.roll_gain,
            winds=config.winds
        )

        # create spaces
        if config.soft_constraints:
            low_x, high_w = -np.inf, np.inf
            low_u, high_u = -np.inf, np.inf
        else:
            low_x, high_w = config.x_bounds[:, 0], config.x_bounds[:, 1]
            low_u, high_u = config.u_bounds[:, 0], config.u_bounds[:, 0]
        self.observation_space = spaces.Box(low=low_x,
                                            high=high_w,
                                            shape=(self.nx,),
                                            dtype=np.float64)
        self.action_space = spaces.Box(low=low_u,
                                       high=high_u,
                                       shape=(self.nu,),
                                       dtype=np.float64)

    @property
    def config(self) -> QuadRotorEnvConfig:
        '''Returns a reference to the environment's configuration.'''
        return self._config

    @property
    def A(self) -> np.ndarray:
        '''Returns a copy of the dynamics `A` matrix.'''
        return self._A.copy()

    @property
    def B(self) -> np.ndarray:
        '''Returns a copy of the dynamics `B` matrix.'''
        return self._B.copy()

    @property
    def C(self) -> np.ndarray:
        '''Returns a copy of the dynamics `C` matrix.'''
        return self._C.copy()

    @property
    def e(self) -> np.ndarray:
        '''Returns a copy of the dynamics `e` vector.'''
        return self._e.copy()

    @property
    def x(self) -> np.ndarray:
        '''Gets a copy of the current state of the quadrotor.'''
        return self._x.copy()

    @x.setter
    def x(self, val: np.ndarray) -> None:
        '''Sets the current state of the quadrotor.'''
        assert self.observation_space.contains(val), f'Invalid state {val}.'
        self._x = val.copy()

    def position_error(self, x: np.ndarray) -> float:
        '''Error of the given state w.r.t. the final position.'''
        return np.square((x - self.config.xf)).sum(axis=-1)

    def control_usage(self, u: np.ndarray) -> float:
        '''Error of the given action related to its norm.'''
        return np.square((u - np.array([0, 0, self.config.g]))).sum()

    def constraint_violations(self, x: np.ndarray, u: np.ndarray) -> float:
        '''Error of the given state and action w.r.t. constraint violations.'''
        return (
            1e2 * np.maximum(0, self.config.x_bounds[:, 0] - x).sum() +
            1e2 * np.maximum(0, x - self.config.x_bounds[:, 1]).sum() +
            3e2 * np.maximum(0, self.config.u_bounds[:, 0] - u).sum() +
            3e2 * np.maximum(0, u - self.config.u_bounds[:, 1]).sum()
        )

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

    def step(
            self, u: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        '''
        Steps the quadrotor environment. 

        # Parameters
        ----------
        u : array_like
            Action to apply to the quadrotor.

        Returns
        -------
        new_state, cost, truncated, terminated, info : 
                                                array_like, float, bool, dict
            A tuple containing the new state of the quadrotor, the 
            instantenuous cost of taking this action in this state, the 
            truncation/termination flags and a dictionary of information.
        '''
        u = u.squeeze()  # in case a row or col was passed
        assert self.action_space.contains(u), 'Invalid action.'

        # compute noise disturbance
        wind = self._C @ self.phi(self.x[2]) * self.np_random.uniform(
            low=[0, 0, -1, 0, 0, 0, -1, -1, 0, 0],
            high=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0]).reshape(self.nx, 1)

        # compute new state: x+ = A*x + B*u + e + C*phi(s[2])*w
        self.x = (
            self._A @ self.x.reshape((-1, 1)) +
            self._B @ u.reshape((-1, 1)) +
            self._e +
            wind
        ).flatten()

        # compute cost
        error = self.position_error(self.x)
        usage = self.control_usage(u)
        violations = self.constraint_violations(self.x, u)
        cost = float(error + usage + violations)

        # check if terminated
        within_bounds = ((self.config.x_bounds[:, 0] <= self._x) &
                         (self._x <= self.config.x_bounds[:, 1])).all()
        if error <= self.config.termination_error and within_bounds:
            self._n_within_termination += 1
        else:
            self._n_within_termination = 0
        terminated = self._n_within_termination >= self.config.termination_N
        #
        return self.x, cost, terminated, False, {'error': error}

    def render(self):
        raise NotImplementedError('Render method unavailable.')

    def get_dynamics(
        self,
        g: Union[float, cs.SX],
        thrust_coeff: Union[float, cs.SX],
        pitch_d: Union[float, cs.SX],
        pitch_dd: Union[float, cs.SX],
        pitch_gain: Union[float, cs.SX],
        roll_d: Union[float, cs.SX],
        roll_dd: Union[float, cs.SX],
        roll_gain: Union[float, cs.SX],
        winds: dict[float, float] = None
    ) -> Union[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[cs.SX, cs.SX, cs.SX]
    ]:
        '''
        If arguments are all numerical, returns the matrices of the system's 
        dynamics `A`, `B`, `C` and `e`; otherwise, returns the `A`, `B` and `e`
        in symbolical form.
        '''
        T = self.config.T
        is_cs = any(isinstance(o, (cs.SX, cs.MX, cs.DM))
                    for o in [g, thrust_coeff, pitch_d, pitch_dd, pitch_gain,
                              roll_d, roll_dd, roll_gain])
        if is_cs:
            diag = lambda o: cs.diag(cs.vertcat(*o))
            block = cs.blockcat
        else:
            diag = np.diag
            block = np.block
            assert winds is not None, 'Winds are required to compute matrix C.'
            nw = len(winds)
            wind_mag = np.array(list(winds.values()))

        A = T * block([
            [np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))],
            [np.zeros((2, 6)), np.eye(2) * g, np.zeros((2, 2))],
            [np.zeros((1, 10))],
            [np.zeros((2, 6)), -diag((pitch_d, roll_d)),
             np.eye(2)],
            [np.zeros((2, 6)), -diag((pitch_dd, roll_dd)),
             np.zeros((2, 2))]
        ]) + np.eye(10)
        B = T * block([
            [np.zeros((5, 3))],
            [0, 0, thrust_coeff],
            [np.zeros((2, 3))],
            [pitch_gain, 0, 0],
            [0, roll_gain, 0]
        ])
        if not is_cs:
            C = T * block([
                [wind_mag],
                [wind_mag],
                [wind_mag],
                [np.zeros((3, nw))],
                [wind_mag],
                [wind_mag],
                [np.zeros((2, nw))]
            ])
        e = block([
            [np.zeros((5, 1))],
            [- T * g],
            [np.zeros((4, 1))]
        ])
        return (A, B, e) if is_cs else (A, B, C, e)
