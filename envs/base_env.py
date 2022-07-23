import gym
from abc import ABC
from gym.wrappers import TimeLimit, OrderEnforcing
from wrappers.record_data import RecordData


class BaseEnv(gym.Env, ABC):
    @classmethod
    def get_wrapped(
        cls,
        max_episode_steps: int = 50,
        deque_size: int = None,
        *env_args, **env_kwargs
    ) -> gym.Env:
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
        return (
            RecordData(
                TimeLimit(
                    OrderEnforcing(cls(*env_args, **env_kwargs)),
                    max_episode_steps=max_episode_steps),
                deque_size=deque_size))

    def __str__(self):
        '''Returns the wrapper name and the unwrapped environment string.'''
        return f'<{type(self).__name__}>'

    def __repr__(self):
        '''Returns the string representation of the wrapper.'''
        return str(self)