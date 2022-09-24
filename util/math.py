import numpy as np
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
    Join a sequence of arrays with different shapes along a new axis. To do 
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
    arrays = [np.asanyarray(arr) for arr in arrays]
    if not arrays:
        raise ValueError('need at least one array to stack')
    maxshape = arrays[0].shape
    for a in arrays:
        maxshape = np.maximum(maxshape, a.shape)
    arrays = [
        np.pad(
            a,
            [(0, d_max - d) for d, d_max in zip(a.shape, maxshape)],
            mode='constant',
            constant_values=constant_values
        )
        for a in arrays
    ]
    return np.stack(arrays, axis, out)
