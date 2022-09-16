import gym
import numpy as np
import time
from collections import deque


class RecordData(gym.Wrapper):
    '''
    At each step, this wrapper records
        - observations
        - actions 
        - costs/rewards (also cumulative)
        - episode length
        - episode execution time
    from the environment.
    '''

    def __init__(
        self,
        env: gym.Env,
        deque_size: int = None,
        as_numpy: bool = True
    ) -> None:
        '''
        This wrapper will keep track of observations, actions and rewards as 
        well as episode length and execution time.

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
        self.as_numpy = as_numpy
        # long-term storages
        self.observations = deque(maxlen=deque_size)
        self.actions = deque(maxlen=deque_size)
        self.rewards = deque(maxlen=deque_size)
        self.cum_rewards = deque(maxlen=deque_size)
        self.episode_lengths = deque(maxlen=deque_size)
        self.exec_times = deque(maxlen=deque_size)
        # current-episode-storages
        self.t0 = None
        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_cum_reward = None
        self.ep_length = None

    def reset(self, *args, **kwargs) -> np.ndarray:
        '''Resets the environment and resets the current data accumulators.'''
        observation = self.env.reset(*args, **kwargs)
        self._clear_ep_data()
        self.ep_observations.append(observation)
        return observation

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        '''Steps through the environment, accumulating the episode data.'''
        obs, reward, terminated, truncated, info = self.env.step(action)

        # accumulate data
        self.ep_observations.append(obs)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        self.ep_cum_reward += reward
        self.ep_length += 1

        # if episode is done, save the current data to history
        if terminated or truncated:
            # stack as numpy
            o = self.ep_observations
            a = self.ep_actions
            r = self.ep_rewards
            if self.as_numpy:
                o = np.stack(o, axis=0)
                a = np.stack(a, axis=0)
                r = np.stack(r, axis=0)

            # append data
            self.observations.append(o)
            self.actions.append(a)
            self.rewards.append(r)
            self.cum_rewards.append(self.ep_cum_reward)
            self.episode_lengths.append(self.ep_length)
            self.exec_times.append(round(time.perf_counter() - self.t0, 6))

            # clear this episode's data
            self._clear_ep_data()

        return obs, reward, terminated, truncated, info

    def _clear_ep_data(self) -> None:
        self.ep_observations.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()
        self.ep_cum_reward = 0
        self.ep_length = 0
        self.t0 = time.perf_counter()
