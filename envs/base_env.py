from abc import ABC
from typing import Optional, TypeVar, Type
import gym
from gym.wrappers import TimeLimit, OrderEnforcing, NormalizeReward
from envs.wrappers import RecordData, ClipActionIfClose


ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')
SuperEnvType = TypeVar('SuperEnvType')


class BaseEnv(gym.Env[ObsType, ActType], ABC):
    '''Base abstract class for gym environments.'''

    @classmethod
    def get_wrapped(
        cls: Type[SuperEnvType],
        max_episode_steps: Optional[int] = 50,
        record_data: Optional[bool] = True,
        deque_size: Optional[int] = None,
        clip_action: Optional[bool] = False,
        enforce_order: Optional[bool] = True,
        **env_kwargs,
    ) -> SuperEnvType:
        '''
        Returns the environment properly encapsulated in some useful wrappers. 
        Passing `None` to an argument disables the corresponding wrapper, aside
        from `OrderEnforcing` and .

        The wrappers are (from in to outward)
            - `OrderEnforcing`
            - `ClipActionIfClose`
            - `RecordData`
            - `TimeLimit`

        Parameters
        ---------
        max_episode_steps : int, optional
            Maximum number of steps per episode (see `TimeLimit`).
        record_data : bool, optional
            Whether to wrap the env for data recording (see `RecordData`).
        deque_size : int, optional
            Maximum number of episodic data saved (see `RecordData`).
        clip_action : bool, optional
            Whether to clip actions that violates the action space
            (see `ClipActionIfClose`).
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
        # NOTE: RecordData must be done after ClipActionIfClose. TimeLimit must
        # be done after RecordData
        env = cls(**env_kwargs)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        if record_data is not None and record_data:
            env = RecordData(env, deque_size=deque_size)
        if clip_action is not None and clip_action and (
            env.action_space.bounded_below.any() or
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
