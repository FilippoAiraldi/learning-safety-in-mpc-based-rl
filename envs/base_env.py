import gym
from abc import ABC
from gym.wrappers import TimeLimit, OrderEnforcing
from envs.wrappers import RecordData, ClipActionIfClose
from typing import TypeVar


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class BaseEnv(gym.Env[ObsType, ActType], ABC):
    @classmethod
    def get_wrapped(
        cls,
        max_episode_steps: int = 50,
        deque_size: int = None,
        *env_args, **env_kwargs
    ):
        '''
        Returns the environment properly encapsulated in some useful wrappers.
        The wrappers are (from in to outward)
            - OrderEnforcing
            - TimeLimit
            - RecordData

        Parameters
        ---------
        max_episode_steps : int, optional
            Maximum number of steps per episode (see TimeLimit).
        deque_size : int, optional
            Maximum number of espiodic data saved (see RecordData).

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
        env = RecordData(env, deque_size=deque_size)
        env = ClipActionIfClose(env)
        env = OrderEnforcing(env)
        return env

    def __str__(self) -> str:
        '''Returns the environment string.'''
        return f'<{type(self).__name__}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the environment.'''
        return str(self)
