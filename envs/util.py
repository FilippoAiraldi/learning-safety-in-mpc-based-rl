import numpy as np
import casadi as cs


def expK(
    x: np.ndarray | cs.SX | cs.MX | cs.DM, 
    c: np.ndarray, 
    l: float
) -> np.ndarray | cs.SX | cs.MX | cs.DM:
    '''
    Exponential Radial Basis Function.

    Parameters
    ----------
    x : array_like
        Values at which to evaluate the function.
    c : array_like
        Center of the function. Size must be compatible with x.
    l : float
        Length factor of the function.

    Returns
    -------
    y : scalar
        The value of the function, computed as y=exp(-|x-c|^2 / 2l).
    '''
    x_ = x - c
    return np.exp(-(x_.T @ x_ / (2 * l)))


def phi_pos(
    x: np.ndarray | cs.SX | cs.MX | cs.DM,
    C: tuple[np.ndarray],
    l: float = 15
) -> np.ndarray:
    '''TODO'''
    phi = tuple(expK(x, c, l) for c in C)
    return cs.vertcat(*phi) if is_casadi_type(x) else np.vstack(phi)


def is_casadi_type(array_like, recursive: bool = False) -> bool:
    '''
    Returns a boolean of whether an object is a CasADi data type or not. If the
    recursive flag is True, iterates recursively.

    Parameters
    ----------
    array_like : array_like
        The object to evaluate. Either from numpy or casadi.

    recursive : bool
        If the object is a list or tuple, recursively iterate through every 
        subelement. If any of the subelements are a CasADi type, return True. 
        Otherwise, return False.

    Returns
    ----------
    is_casadi : bool
        A boolean if the object is a CasADi data type.
    '''
    if recursive and isinstance(array_like, (list, tuple)):
        for element in array_like:
            if is_casadi_type(element, recursive=True):
                return True

    return (
        isinstance(array_like, cs.MX) or
        isinstance(array_like, cs.DM) or
        isinstance(array_like, cs.SX)
    )
