import gym
from abc import ABC
from gym.wrappers import TimeLimit, OrderEnforcing
from envs.wrappers import RecordData, ClipActionIfClose


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
        # wrap the environment. The last wrapper is the first to be executed,
        # so put the data-recording close to the env, after possible
        # modifications by outer wrappers
        return (
            OrderEnforcing(
                TimeLimit(
                    ClipActionIfClose(
                        RecordData(
                            cls(*env_args, **env_kwargs),
                            deque_size=deque_size
                        )
                    ),
                    max_episode_steps=max_episode_steps))
        )

    def __str__(self):
        '''Returns the wrapper name and the unwrapped environment string.'''
        return f'<{type(self).__name__}>'

    def __repr__(self):
        '''Returns the string representation of the wrapper.'''
        return str(self)
