import numpy as np
from collections import UserDict, Mapping
from itertools import combinations
from scipy.special import comb
from typing import Any, Iterable, Iterator, Union


def nchoosek(n: Union[int, Iterable[Any]],
             k: int) -> Union[int, Iterable[tuple[Any]]]:
    '''Returns the binomial coefficient, i.e.,  the number of combinations of n
    items taken k at a time. If n is an iterable, then it returns an iterable 
    containing all possible combinations of the elements of n taken k at a 
    time.'''
    return comb(n, k, exact=True) if isinstance(n, int) else combinations(n, k)


def monomial_powers(d: int, k: int) -> np.ndarray:
    '''Computes the powers of degree k contained in a d-dimensional 
    monomial.'''
    # Thanks to: https://stackoverflow.com/questions/40513896/compute-all-d-dimensional-monomials-of-degree-less-than-k
    m = nchoosek(k + d - 1, d - 1)
    dividers = np.column_stack((
        np.zeros((m, 1), dtype=int),
        np.row_stack(list(nchoosek(np.arange(1, k + d), d - 1))),
        np.full((m, 1), k + d, dtype=int)
    ))
    return np.flipud(np.diff(dividers, axis=1) - 1)


def cholesky_added_multiple_identities(
    A: np.ndarray, beta: float = 1e-3, maxiter: int = 1000
) -> np.ndarray:
    '''Lower Cholesky factorization with added multiple of the identity matrix.
    (Algorithm 3.3 from Nocedal&Wright)'''
    a_min = np.diag(A).min()
    tau = 0 if a_min > 0 else -a_min + beta

    I = np.eye(A.shape[0])
    for _ in range(maxiter):
        try:
            return np.linalg.cholesky(A + tau * I)
        except np.linalg.LinAlgError:
            tau = max(1.1 * tau, beta)
    raise ValueError('Maximum iterations reached.')


def constraint_violation(
    *arrays_and_bounds: tuple[np.ndarray, np.ndarray]
) -> Iterator[np.ndarray]:
    '''Computes constraint violations.

    Parameters
    ----------
    arrays_and_bounds : *tuple[array_like, array_like]
        Any number of `(array, bounds)` for which to compute the constraint 
        violations.
        For each tuple, `bounds` is an array of shape `(N, 2)`, where `N` is 
        the number of features, and the first and second columns are lower and
        upper bounds, respectively. `array` is an array of shape `(N, ...)`.
        A violation is defined as `x <= bound`.

    Returns
    -------
    cv : iterator[array_like]
        For each tuple provided, yields an array of lower and upper bound 
        violations of shape `(N, 2, ...)`. 
    '''
    for a, bnd in arrays_and_bounds:
        a = np.asarray(a)
        lb = np.expand_dims(bnd[:, 0], tuple(range(1, a.ndim)))
        ub = np.expand_dims(bnd[:, 1], tuple(range(1, a.ndim)))
        yield np.stack((lb - a, a - ub), axis=1)


def jaggedstack(
    arrays,
    axis: int = 0,
    out: np.ndarray = None,
    constant_values: Union[float, np.ndarray] = np.nan
) -> np.ndarray:
    '''
    Joins a sequence of arrays with different shapes along a new axis. To do 
    so, each array is padded with `constant_values` (see `numpy.pad`) to the 
    right to even out the shapes. Then, `numpy.stack` is called.

    Parameters
    ----------
    arrays, axis, out
        See `numpy.stack`.
    constant_values
        See `numpy.pad`.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.
    '''
    arrays: list[np.ndarray] = [np.asanyarray(a) for a in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')
    maxndim = max(map(lambda a: a.ndim, arrays))
    newarrays = []
    maxshape = None
    for a in arrays:
        if a.ndim < maxndim:
            a = np.expand_dims(a, tuple(range(a.ndim, maxndim)))
        maxshape = \
            a.shape if maxshape is None else np.maximum(maxshape, a.shape)
        newarrays.append(a)
    newarrays = [
        np.pad(
            a,
            [(0, d_max - d) for d, d_max in zip(a.shape, maxshape)],
            mode='constant',
            constant_values=constant_values
        )
        for a in newarrays
    ]
    return np.stack(newarrays, axis, out)


def logmean(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=np._NoValue,
    *,
    where=np._NoValue,
    nanmean: bool = False
) -> np.ndarray:
    '''
    Computes the arithmetic mean of the exponents of the elements along the 
    specified axis. To do so, first the log of elements in `a` is taken, then 
    the mean of the exponents is computed, and finally the elements are raised
    to the base.

    Parameters
    ----------
    a, axis, dtype, out, keepdims, where
        See `numpy.mean`.
    nanmean : bool, optional
        Whether to compute the mean with `numpy.mean` or `numpy.nanmean`. By 
        default `False`.

    Returns
    -------
    np.ndarray
        The mean in the logarithmic space of the array.
    '''
    _mean = np.nanmean if nanmean else np.mean
    return np.exp(_mean(
        np.log(a),
        axis=axis,
        dtype=dtype,
        out=out, keepdims=keepdims, where=where
    ))


class NormalizationService(UserDict[str, np.ndarray]):
    '''Shared service for normalizing quantities.'''

    def normalize(
        self, name: str, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        '''Normalizes the value `x` according to the ranges of `name`.'''
        r = self.data[name]
        if r.ndim == 1:
            return (x - r[0]) / (r[1] - r[0])
        if isinstance(x, np.ndarray) and x.shape[-1] != r.shape[0]:
            raise ValueError('Input with invalid dimensions: '
                             'normalization would alter shape.')
        return (x - r[:, 0]) / (r[:, 1] - r[:, 0])

    def denormalize(
        self, name: str, x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        '''Denormalizes the value `x` according to the ranges of `name`.'''
        r = self.data[name]
        if r.ndim == 1:
            return (r[1] - r[0]) * x + r[0]
        if isinstance(x, np.ndarray) and x.shape[-1] != r.shape[0]:
            raise ValueError('Input with invalid dimensions: '
                             'denormalization would alter shape.')
        return (r[:, 1] - r[:, 0]) * x + r[:, 0]

    def can_be_normalized(self, name: str) -> bool:
        '''Whether variable `name` can be normalized.'''
        return name in self.data

    def update(self, other: Any = None, **kwargs: np.ndarray) -> None:
        '''Updates but throws if duplicate keys occur.'''
        # thanks to  https://stackoverflow.com/a/30242574/19648688
        if other is not None:
            for k, v in other.items() if isinstance(other, Mapping) else other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def register(self, other: Any = None, **kwargs: np.ndarray) -> None:
        '''Updates the normalization ranges. Raises if duplicates occur.'''
        return self.update(other, **kwargs)

    def __setitem__(self, name: str, range: np.ndarray) -> None:
        if name in self.data:
            raise KeyError(f'\'{name}\' already registered for normalization.')
        self.data[name] = range
