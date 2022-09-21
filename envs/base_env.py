import gym
from abc import ABC
from gym.wrappers import (
    TimeLimit, OrderEnforcing, NormalizeObservation, NormalizeReward)
from envs.wrappers import RecordData, ClipActionIfClose
from typing import TypeVar, Type


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
SuperEnvType = TypeVar('SuperEnvType')


class BaseEnv(gym.Env[ObsType, ActType], ABC):
    @classmethod
    def get_wrapped(
        cls: Type[SuperEnvType],
        max_episode_steps: int = 50,
        record_data: bool = True,
        deque_size: int = None,
        normalize_observation: bool = False,
        normalize_reward: bool = False,
        *env_args, **env_kwargs
    ) -> SuperEnvType:
        '''
        Returns the environment properly encapsulated in some useful wrappers.
        The wrappers are (from in to outward)
            - `OrderEnforcing`
            - `ClipActionIfClose` (optional)
            - `RecordData`
            - `NormalizeObservation` (optional)
            - `NormalizeReward` (optional)
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
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if normalize_observation:
            env = NormalizeObservation(env)
        if normalize_reward:
            env = NormalizeReward(env)
        if record_data:
            env = RecordData(env, deque_size=deque_size)
        if env.action_space.bounded_below.any() or \
                env.action_space.bounded_above.any():
            env = ClipActionIfClose(env)
        env = OrderEnforcing(env)
        return env

    def __str__(self) -> str:
        '''Returns the environment string.'''
        return f'<{type(self).__name__}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the environment.'''
        return str(self)
