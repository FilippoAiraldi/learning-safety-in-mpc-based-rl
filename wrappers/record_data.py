import gym
import numpy as np
from collections import deque


class RecordData(gym.Wrapper):
    '''
    This wrapper records all the observations, actions and costs/rewards at 
    each time step coming to and from the environment.
    '''

    def __init__(
        self,
        env: gym.Env,
        deque_size: int = None,
        as_numpy: bool = True
    ) -> None:
        '''
        This wrapper will keep track of observations, actions and rewards.

        Parameters
        ----------
        env : gym.Env 
            The environment to apply the wrapper to.
        deque_size : int, optional
            The maximum size of the historical data. By default, None.
        as_numpy : bool, optional
            Whether to save the data at the end of the episode as an array. By
            default, true.
        '''
        super().__init__(env)
        self.observations_history = deque(maxlen=deque_size)
        self.actions_history = deque(maxlen=deque_size)
        self.rewards_history = deque(maxlen=deque_size)
        self.current_observations = []
        self.current_actions = []
        self.current_rewards = []
        self.as_numpy = as_numpy

    def reset(self, *args, **kwargs) -> np.ndarray:
        '''Resets the environment and resets the current lists.'''
        observation = super().reset(*args, **kwargs)
        self.current_observations.clear()
        self.current_actions.clear()
        self.current_rewards.clear()
        self.current_observations.append(observation)

    def step(self, action):
        '''Steps through the environment, recording the episode data.'''
        observation, reward, done, info = super().step(action)

        # append to current data
        self.current_observations.append(observation)
        self.current_actions.append(action)
        self.current_rewards.append(reward)

        # if episode is done, save the current data to history
        if done:
            o = self.current_observations
            a = self.current_actions
            r = self.current_rewards
            if self.as_numpy:
                o = np.stack(o, axis=-1)
                a = np.stack(o, axis=-1)
                r = np.stack(o, axis=-1)
            self.observations_history.append(o)
            self.actions_history.append(a)
            self.rewards_history.append(r)
        return observation, reward, done, info


