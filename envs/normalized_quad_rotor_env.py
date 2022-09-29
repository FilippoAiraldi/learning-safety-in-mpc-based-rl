import casadi as cs
import numpy as np
from copy import deepcopy
from envs.quad_rotor_env import QuadRotorEnv, QuadRotorEnvConfig
from typing import Union
from util.configurations import init_config


class NormalizedQuadRotorEnv(QuadRotorEnv):
    ranges: dict[str, np.ndarray] = {
        # model parameters
        'g': np.array([0, 20]),
        'thrust_coeff': np.array([0, 4]),
        'pitch_d': np.array([0, 200]),
        'pitch_dd': np.array([0, 200]),
        'pitch_gain': np.array([0, 20]),
        'roll_d': np.array([0, 20]),
        'roll_dd': np.array([0, 20]),
        'roll_gain': np.array([0, 20]),
        # system states
        'x': np.array([[-1, 5],
                       [-1, 5],
                       [-1, 5],
                       [-4, 4],
                       [-4, 4],
                       [-4, 4],
                       [np.deg2rad(-30), np.deg2rad(30)],
                       [np.deg2rad(-30), np.deg2rad(30)],
                       [-3, 3],
                       [-3, 3]]),
        # system control actions
        'u': np.array([[-np.pi, np.pi],
                       [-np.pi, np.pi],
                       [0, 20]]),
    }

    def __init__(
        self,
        config: Union[dict, QuadRotorEnvConfig] = None
    ) -> None:
        # precompute scaling matrices
        rx, ru = self.ranges['x'], self.ranges['u']
        self._Tx = np.diag(1 / (rx[:, 1] - rx[:, 0]))
        self._Tx_inv = np.diag(rx[:, 1] - rx[:, 0])
        self._Mx = - (self._Tx @ rx[:, 0]).reshape(-1, 1)
        self._Tu = np.diag(1 / (ru[:, 1] - ru[:, 0]))
        self._Tu_inv = np.diag(ru[:, 1] - ru[:, 0])
        self._Mu = - (self._Tu @ ru[:, 0]).reshape(-1, 1)

        # normalize the configuration model parameters and bounds
        c = deepcopy(init_config(config, QuadRotorEnvConfig))
        for k in c.__dataclass_fields__.keys():
            if k not in self.ranges:
                continue
            r = self.ranges[k]
            c.__dict__[k] = (c.__dict__[k] - r[0]) / (r[1] - r[0])
        c.__dict__['x_bounds'] = self._normalize('x', c.x_bounds)
        c.__dict__['u_bounds'] = self._normalize('u', c.u_bounds)
        c.__dict__['x0'] = self._normalize('x', c.x0)
        c.__dict__['xf'] = self._normalize('x', c.xf)
        c.__dict__['termination_error'] = \
            c.termination_error / np.linalg.norm(self._Tx**2) / self.nx

        # let the base class do the rest
        super().__init__(c)

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
        is_cs = any(isinstance(o, (cs.SX, cs.MX, cs.DM))
                    for o in [g, thrust_coeff, pitch_d, pitch_dd, pitch_gain,
                              roll_d, roll_dd, roll_gain])
        o = super().get_dynamics(
            g=self._denormalize('g', g),
            thrust_coeff=self._denormalize('thrust_coeff', thrust_coeff),
            pitch_d=self._denormalize('pitch_d', pitch_d),
            pitch_dd=self._denormalize('pitch_dd', pitch_dd),
            pitch_gain=self._denormalize('pitch_gain', pitch_gain),
            roll_d=self._denormalize('roll_d', roll_d),
            roll_dd=self._denormalize('roll_dd', roll_dd),
            roll_gain=self._denormalize('roll_gain', roll_gain),
            winds=winds
        )
        if is_cs:
            A, B, e = o
        else:
            A, B, C, e = o

        Tx, Tx_inv, Mx = self._Tx, self._Tx_inv, self._Mx
        Tu_inv, Mu = self._Tu_inv, self._Mu

        As = Tx @ A @ Tx_inv
        Bs = Tx @ B @ Tu_inv
        if not is_cs:
            Cs = Tx @ C
        es = (
            Tx @ e +
            (np.eye(Tx.shape[0]) - Tx @ A @ Tx_inv) @ Mx -
            Tx @ B @ Tu_inv @ Mu
        )
        return (As, Bs, es) if is_cs else (As, Bs, Cs, es)

    def position_error(self, x: np.ndarray) -> float:
        return super().position_error(x)

    def control_usage(self, u: np.ndarray) -> float:
        return super().control_usage(u)

    def constraint_violations(self, x: np.ndarray, u: np.ndarray) -> float:
        return super().constraint_violations(x, u)

    def _normalize(
        self, name: str, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        r = self.ranges[name]
        if r.ndim == 1:
            return (x - r[0]) / (r[1] - r[0])
        dims = tuple(range(1, x.ndim))
        min_ = np.expand_dims(r[:, 0], dims)
        div_ = np.expand_dims(r[:, 1] - r[:, 0], dims)
        return (x - min_) / div_

    def _denormalize(
        self, name: str, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        r = self.ranges[name]
        if r.ndim == 1:
            return (r[1] - r[0]) * x + r[0]
        dims = tuple(range(1, x.ndim))
        min_ = np.expand_dims(r[:, 0], dims)
        div_ = np.expand_dims(r[:, 1] - r[:, 0], dims)
        return div_ * x + min_
