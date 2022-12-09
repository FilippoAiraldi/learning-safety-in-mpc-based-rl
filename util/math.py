from typing import Iterator, Union

import numpy as np


def cholesky_added_multiple_identities(
    A: np.ndarray, beta: float = 1e-3, maxiter: int = 1000
) -> np.ndarray:
    """Lower Cholesky factorization with added multiple of the identity matrix.
    (Algorithm 3.3 from Nocedal&Wright)"""
    a_min = np.diag(A).min()
    tau = 0 if a_min > 0 else -a_min + beta

    I = np.eye(A.shape[0])
    for _ in range(maxiter):
        try:
            return np.linalg.cholesky(A + tau * I)
        except np.linalg.LinAlgError:
            tau = max(1.05 * tau, beta)
    raise ValueError("Maximum iterations reached.")


def constraint_violation(
    *arrays_and_bounds: tuple[np.ndarray, np.ndarray]
) -> Iterator[np.ndarray]:
    """Computes constraint violations.

    Parameters
    ----------
    arrays_and_bounds : *tuple[array_like, array_like]
        Any number of `(array, bounds)` for which to compute the constraint violations.
        For each tuple, `bounds` is an array of shape `(N, 2)`, where `N` is the number
        of features, and the first and second columns are lower and upper bounds,
        respectively. `array` is an array of shape `(N, ...)`. A violation is defined
        as `x <= bound`.

    Returns
    -------
    cv : iterator[array_like]
        For each tuple provided, yields an array of lower and upper bound violations of
        shape `(N, 2, ...)`.
    """
    for a, bnd in arrays_and_bounds:
        a = np.asarray(a)
        lb = np.expand_dims(bnd[:, 0], tuple(range(1, a.ndim)))
        ub = np.expand_dims(bnd[:, 1], tuple(range(1, a.ndim)))
        yield np.stack((lb - a, a - ub), axis=1)


def jaggedstack(
    arrays,
    axis: int = 0,
    out: np.ndarray = None,
    constant_values: Union[float, np.ndarray] = np.nan,
) -> np.ndarray:
    """
    Joins a sequence of arrays with different shapes along a new axis. To do so, each
    array is padded with `constant_values` (see `numpy.pad`) to the right to even out
    the shapes. Then, `numpy.stack` is called.

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
    """
    arrays: list[np.ndarray] = [np.asanyarray(a) for a in arrays]
    if not arrays:
        raise ValueError("need at least one array to stack")
    maxndim = max(map(lambda a: a.ndim, arrays))
    newarrays = []
    maxshape = None
    for a in arrays:
        if a.ndim < maxndim:
            a = np.expand_dims(a, tuple(range(a.ndim, maxndim)))
        maxshape = a.shape if maxshape is None else np.maximum(maxshape, a.shape)
        newarrays.append(a)
    newarrays = [
        np.pad(
            a,
            [(0, d_max - d) for d, d_max in zip(a.shape, maxshape)],
            mode="constant",
            constant_values=constant_values,
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
    nanmean: bool = False,
) -> np.ndarray:
    """
    Computes the arithmetic mean of the exponents of the elements along the specified
    axis. To do so, first the log of elements in `a` is taken, then the mean of the
    exponents is computed, and finally the elements are raised to the base.

    Parameters
    ----------
    a, axis, dtype, out, keepdims, where
        See `numpy.mean`.
    nanmean : bool, optional
        Whether to compute the mean with `numpy.mean` or `numpy.nanmean`. By default
        `False`.

    Returns
    -------
    np.ndarray
        The mean in the logarithmic space of the array.
    """
    _mean = np.nanmean if nanmean else np.mean
    return np.exp(
        _mean(
            np.log(a), axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
        )
    )
