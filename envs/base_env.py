from abc import ABC
from typing import Type, TypeVar
import gym
from gym.wrappers import TimeLimit, OrderEnforcing, NormalizeReward
from envs.wrappers import RecordData


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
SuperEnvType = TypeVar('SuperEnvType')


class BaseEnv(gym.Env[ObsType, ActType], ABC):
    '''Base abstract class for gym environments.'''

    @classmethod
    def get_wrapped(
        cls: Type[SuperEnvType],
        max_episode_steps: int = 50,
        record_data: bool = True,
        normalize_reward: tuple[bool, float] = (False,),
        deque_size: int = None,
        enforce_order: bool = True,
        **env_kwargs,
    ) -> SuperEnvType:
        '''
        Returns the environment properly encapsulated in some useful wrappers. 
        Passing `None` to an argument disables the corresponding wrapper, aside
        from `OrderEnforcing` and .

        The wrappers are (from in to outward)
            - `OrderEnforcing`
            - `RecordData`
            - `NormalizeReward`
            - `TimeLimit`

        Parameters
        ---------
        max_episode_steps : int, optional
            Maximum number of steps per episode (see `TimeLimit`).
        record_data : bool, optional
            Whether to wrap the env for data recording (see `RecordData`).
        deque_size : int, optional
            Maximum number of episodic data saved (see `RecordData`).
        normalize_reward : tuple[bool, gamma], optional
            Whether to apply reward normalization or not 
            (see `NormalizeReward`). `gamma` is the discount factor.
        enforce_order : bool, optional
            Whether to apply order enforcing or not (see `OrderEnforcing`).
        env_kwargs : dict
            Additional arguments passed to the env constructor.

        Returns
        -------
        env : wrapped gym.Env
            The environment wrapped in wrappers.
        '''
        # wrap the environment. The last wrapper is the first to be executed,
        # so put the data-recording close to the env, after possible
        # modifications by outer wrappers
        env = cls(**env_kwargs)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if record_data:
            env = RecordData(env, deque_size=deque_size)
        if normalize_reward[0]:
            env = NormalizeReward(env, gamma=normalize_reward[1])
        if enforce_order:
            env = OrderEnforcing(env)
        return env

    def __str__(self) -> str:
        '''Returns the environment string.'''
        return f'<{type(self).__name__}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the environment.'''
        return str(self)
