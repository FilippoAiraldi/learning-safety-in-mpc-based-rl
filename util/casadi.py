import casadi as cs
import math
from typing import Union


def prod(
    x: Union[cs.SX, cs.MX, cs.DM], axis: int = 0
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''CasADi version of `numpy.prod`.'''
    if x.is_vector():
        return cs.det(cs.diag(x))
    sum_ = cs.sum1 if axis == 0 else cs.sum2
    return cs.exp(sum_(cs.log(x))) if axis == 0 else cs.exp(sum_(cs.log(x)))


def quad_form(
    A: Union[cs.SX, cs.MX, cs.DM], x: Union[cs.SX, cs.MX, cs.DM]
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''Calculates quadratic form `x.T*A*x`.'''
    if A.is_vector():
        A = cs.diag(A)
    return cs.bilin(A, x, x)


def norm_cdf(
    x: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = 0,
    scale: Union[cs.SX, cs.MX, cs.DM] = 1
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''CasADi version of `scipy.stats.norm.cdf`.'''
    return 0.5 * (1 + cs.erf((x - loc) / math.sqrt(2) / scale))


def norm_ppf(
    p: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = 0,
    scale: Union[cs.SX, cs.MX, cs.DM] = 1
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''CasADi version of `scipy.stats.norm.ppf`.'''
    return math.sqrt(2) * scale * cs.erfinv(2 * p - 1) + loc
