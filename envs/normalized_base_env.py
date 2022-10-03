import numpy as np
from abc import ABC
from envs.base_env import BaseEnv
from typing import Union


class NormalizedBaseEnv(BaseEnv, ABC):
    '''Base environment with utilities for normalization.'''

    normalized: bool = True
    ranges: dict[str, np.ndarray] = {}

    def normalize(
        self, name: str, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        '''Normalizes the value `x` according to the ranges of `name`.'''
        r = self.ranges[name]
        if r.ndim == 1:
            return (x - r[0]) / (r[1] - r[0])
        if isinstance(x, np.ndarray) and x.shape[-1] != r.shape[0]:
            raise ValueError('Input with invalid dimensions: '
                             'normalization would alter shape.')
        return (x - r[:, 0]) / (r[:, 1] - r[:, 0])

    def denormalize(
        self, name: str, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        '''Denormalizes the value `x` according to the ranges of `name`.'''
        r = self.ranges[name]
        if r.ndim == 1:
            return (r[1] - r[0]) * x + r[0]
        if isinstance(x, np.ndarray) and x.shape[-1] != r.shape[0]:
            raise ValueError('Input with invalid dimensions: '
                             'denormalization would alter shape.')
        return (r[:, 1] - r[:, 0]) * x + r[:, 0]

    def can_be_normalized(self, name: str) -> bool:
        '''Whether variable `name` can be normalized.'''
        return name in self.ranges
