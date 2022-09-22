import numpy as np
from itertools import combinations
from scipy.special import comb
from typing import Any, Iterable, Union


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
