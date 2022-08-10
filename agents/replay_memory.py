import numpy as np
from collections import deque
from itertools import chain
from gym.utils.seeding import np_random
from typing import Iterable, TypeVar, Optional, Union


T = TypeVar('T')


class ReplayMemory(deque[T]):
    '''
    Container class for RL traning to save and sample experience transitions. 
    The class inherits from collections.deque, adding a couple of simple 
    functionalities to it.
    '''

    def __init__(
        self,
        *args,
        maxlen: Optional[int],
        seed: int = None
    ) -> None:
        '''
        Instantiate a replay memory.

        Parameters
        ----------
        maxlen : int
            Maximum length/capacity of the memory. Can be None if unlimited.
        args : ...
            Args passed to collections.deque.__init__.
        seed : int, optional
            Seed for the random number generator.
        '''
        super().__init__(*args, maxlen=maxlen)
        self.np_random, _ = np_random(seed)

    def sample(
        self,
        n: Union[int, float],
        include_last_n: Union[int, float]
    ) -> Iterable[T]:
        '''
        Samples the memory and yields the sampled elements.

        Parameters
        n : int or float
            Size of the sample to draw from memory, either as size or 
            percentage of the maximum capacity (if not None).
        include_last_n : int or float
            Size or percentage of the sample dedicated to including the last 
            elements added to the memory.

        Returns
        -------
        sample : iterable
            An iterable sample is yielded.
        '''
        length = len(self)

        # convert percentages to int
        if isinstance(n, float):
            n = int(self.maxlen * n)
        n = np.clip(n, 0, length)
        if isinstance(include_last_n, float):
            include_last_n = int(n * include_last_n)
        include_last_n = np.clip(include_last_n, 0, n)

        # get last n indices and the sampled indices from the remaining
        last_n = range(length - include_last_n, length)
        sampled = self.np_random.choice(
            range(length - include_last_n), n - include_last_n, replace=False)

        # yield the sample
        yield from (self[i] for i in chain(last_n, sampled))
