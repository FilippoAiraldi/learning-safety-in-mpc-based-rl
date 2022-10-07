from collections import deque
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Iterator, Optional, Sequence, Union, TypeVar
from gym.utils.seeding import np_random
import casadi as cs
import numpy as np


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
            Args passed to `collections.deque.__init__`.
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
        n = np.clip(n, min(1, length), length)
        if isinstance(include_last_n, float):
            include_last_n = int(n * include_last_n)
        include_last_n = np.clip(include_last_n, 0, n)

        # get last n indices and the sampled indices from the remaining
        last_n = range(length - include_last_n, length)
        sampled = self.np_random.choice(
            range(length - include_last_n), n - include_last_n, replace=False)

        # yield the sample
        yield from (self[i] for i in chain(last_n, sampled))


@dataclass
class RLParameter:
    '''An RL parameter class for compactly managing information and the value 
    of a learnable parameter.'''
    name: str
    value: np.ndarray
    bounds: np.ndarray
    symV: cs.SX
    symQ: cs.SX

    @property
    def size(self) -> int:
        return self.symV.shape[0]  # since rl pars are all column vectors

    def __post_init__(self) -> None:
        shape = self.symV.shape
        assert shape == self.symQ.shape, \
            f'Parameter {self.name} has different shapes in ' \
            f'Q ({self.symQ.shape}) and V ({self.symV.shape}).'
        assert self.symV.is_column(), \
            f'Parameter {self.name} must be a column vector.'
        self.bounds = np.broadcast_to(self.bounds, (shape[0], 2))
        self.update_value(self.value)

    def update_value(self, new_val: np.ndarray) -> None:
        '''Updates the parameter's current value to the new one.'''
        new_val = np.broadcast_to(new_val, self.bounds.shape[0])
        assert ((
            (self.bounds[:, 0] <= new_val) |
            np.isclose(new_val, self.bounds[:, 0])
        ).all() and (
            (new_val <= self.bounds[:, 1]) |
            np.isclose(new_val, self.bounds[:, 1])
        ).all()), 'Parameter value outside bounds.'
        self.value = np.clip(
            new_val, self.bounds[:, 0], self.bounds[:, 1])


class RLParameterCollection(Sequence[RLParameter]):
    '''Collection of learnable RL parameters, which can be accessed by string
    as a dictionary or by index as a list.'''

    def __init__(self, *parameters: RLParameter) -> None:
        '''Instantiate the collection from another iterable, if provided.'''
        self._list: list[RLParameter] = []
        self._dict: dict[str, RLParameter] = {}
        for parameter in parameters:
            self._list.append(parameter)
            self._dict[parameter.name] = parameter

    @property
    def as_list(self) -> dict[str, RLParameter]:
        '''Returns a view of the collection as `list`.'''
        return self._list

    @property
    def as_dict(self) -> dict[str, RLParameter]:
        '''Returns a view of the collection as `dict`.'''
        return self._dict

    @property
    def n_theta(self) -> int:
        '''Returns the length of the parameters vector theta.'''
        return sum(self.sizes())

    def names(self) -> Sequence[str]:
        '''Returns the names of the parameters in the collection.'''
        return self._dict.keys()

    def values(
        self, as_dict: bool = False
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        '''Returns the values of the parameters in the collection concatenated
        into a single array, by default. Otherwise, if `as_dict=True`, a `dict`
        with each value is returned.'''
        if as_dict:
            return {name: p.value for name, p in self.items()}
        return np.concatenate([p.value for p in self._list])

    def bounds(
        self, as_dict: bool = False
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        '''Returns the bounds of the parameters in the collection concatenated
        into a single array, by default. Otherwise, if `as_dict=True`, a `dict`
        with each bound is returned.'''
        if as_dict:
            return {name: p.bounds for name, p in self.items()}
        return np.row_stack([p.bounds for p in self._list])

    def symV(self, as_dict: bool = False) -> Union[cs.SX, dict[str, cs.SX]]:
        '''Returns the symbols of the parameters in the collection concatenated
        into a single array, by default. Otherwise, if `as_dict=True`, a `dict`
        with each symbolical V variable is returned.'''
        if as_dict:
            return {name: p.symV for name, p in self.items()}
        return cs.vertcat(*(p.symV for p in self._list))

    def symQ(self, as_dict: bool = False) -> Union[cs.SX, dict[str, cs.SX]]:
        '''Returns the symbols of the parameters in the collection concatenated
        into a single array, by default. Otherwise, if `as_dict=True`, a `dict`
        with each symbolical Q variable is returned.'''
        if as_dict:
            return {name: p.symQ for name, p in self.items()}
        return cs.vertcat(*(p.symQ for p in self._list))

    def sizes(self, as_dict: bool = False) -> Union[list[int], dict[str, int]]:
        '''Returns the size of each parameter.'''
        if as_dict:
            return {p.name: p.size for p in self._list}
        return [p.size for p in self._list]

    def update_values(
        self,
        new_vals: Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]
    ) -> None:
        '''Updates the values of each parameter in the collection.'''
        if isinstance(new_vals, np.ndarray):
            new_vals = np.split(new_vals, np.cumsum(self.sizes())[:-1])
            for p, val in zip(self._list, new_vals):
                p.update_value(val)
        elif isinstance(new_vals, list):
            for p, val in zip(self._list, new_vals):
                p.update_value(val)
        elif isinstance(new_vals, dict):
            for n in self._dict.keys():
                self._dict[n].update_value(new_vals[n])

    def values2str(self, summarize_arrays: bool = True, **kwargs) -> str:
        '''Creates a string with all the values.'''
        if 'precision' not in kwargs:
            kwargs['precision'] = 3
        if 'separator' not in kwargs:
            kwargs['separator'] = ','
        prc = kwargs['precision']

        def par2str(p: RLParameter) -> str:
            if p.value.size == 1:
                return f'{p.name}={p.value.item():.{prc}f}'
            if summarize_arrays:
                mean = p.value.mean().item()
                min_ = p.value.min().item()
                max_ = p.value.max().item()
                return \
                    f'{p.name}={mean:.{prc}f} [{min_:.{prc}f}, {max_:.{prc}f}]'
            return np.array2string(p.value, **kwargs)

        return '; '.join(par2str(p) for p in self._list)

    def items(self) -> Iterable[tuple[str, RLParameter]]:
        return self._dict.items()

    def __getitem__(
        self,
        index: Union[str, Iterable[str], int, slice, Iterable[int]]
    ) -> Union[RLParameter, list[RLParameter]]:
        if isinstance(index, str):
            return self._dict[index]
        if isinstance(index, (int, slice)):
            return self._list[index]
        if isinstance(index, Iterable):
            return [self._list[i] for i in index]

    def __iter__(self) -> Iterator[RLParameter]:
        return iter(self._list)

    def __next__(self) -> RLParameter:
        return next(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __str__(self) -> str:
        return f'<{type(self).__name__} (L={len(self._list)}, ' \
               f'nw={sum(self.sizes())}): {self.values2str()}>'

    def __repr__(self) -> str:
        return str(self)
