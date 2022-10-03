import gym
import numpy as np
from abc import ABC
from gym.wrappers import (
    TimeLimit, OrderEnforcing, NormalizeObservation, NormalizeReward)
from envs.wrappers import RecordData, ClipActionIfClose
from typing import Optional, TypeVar, Type, Union


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
SuperEnvType = TypeVar('SuperEnvType')


class BaseEnv(gym.Env[ObsType, ActType], ABC):
    @classmethod
    def get_wrapped(
        cls: Type[SuperEnvType],
        max_episode_steps: Optional[int] = 50,
        normalize_observation: Optional[bool] = False,
        normalize_reward: Optional[bool] = False,
        record_data: Optional[bool] = True,
        deque_size: Optional[int] = None,
        clip_action: Optional[bool] = False,
        enforce_order: Optional[bool] = True,
        *env_args, **env_kwargs
    ) -> SuperEnvType:
        '''
        Returns the environment properly encapsulated in some useful wrappers. 
        Passing `None` to an argument disables the corresponding wrapper, aside
        from `OrderEnforcing` and .

        The wrappers are (from in to outward)
            - `OrderEnforcing`
            - `ClipActionIfClose`
            - `RecordData`
            - `NormalizeReward`
            - `NormalizeObservation`
            - `TimeLimit`

        Parameters
        ---------
        max_episode_steps : int, optional
            Maximum number of steps per episode (see `TimeLimit`).
        record_data : bool, optional
            Whether to wrap the env for data recording (see `RecordData`).
        deque_size : int, optional
            Maximum number of episodic data saved (see `RecordData`).
        normalize_observation : bool, optional
            Whether to apply observation normalization (see 
            `NormalizeObservation`).
        normalize_reward : bool, optional
            Whether to apply return normalization (see `NormalizeReward`).
        clip_action : bool, optional
            Whether to clip actions that violates the action space
            (see `ClipActionIfClose`).
        enforce_order : bool, optional
            Whether to apply order enforcing or not (see `OrderEnforcing`).

        Returns
        -------
        env : wrapped gym.Env
            The environment wrapped in wrappers.
        '''
        # wrap the environment. The last wrapper is the first to be executed,
        # so put the data-recording close to the env, after possible
        # modifications by outer wrappers
        # NOTE: RecordData must be done after ClipActionIfClose. TimeLimit must
        # be done after RecordData
        env = cls(*env_args, **env_kwargs)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if normalize_observation is not None and normalize_observation:
            env = NormalizeObservation(env)
        if normalize_reward is not None and normalize_reward:
            env = NormalizeReward(env)
        if record_data is not None and record_data:
            env = RecordData(env, deque_size=deque_size)
        if clip_action is not None and clip_action and (\
            env.action_space.bounded_below.any() or \
                env.action_space.bounded_above.any()):
            env = ClipActionIfClose(env)
        if enforce_order is not None and enforce_order:
            env = OrderEnforcing(env)
        return env

    def __str__(self) -> str:
        '''Returns the environment string.'''
        return f'<{type(self).__name__}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the environment.'''
        return str(self)


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
