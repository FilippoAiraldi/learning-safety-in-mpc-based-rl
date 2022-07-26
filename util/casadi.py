import math
from typing import Any, Union

import casadi as cs


def is_casadi_object(obj: Any) -> bool:
    """
    Checks if the object belongs to the casadi module.
    Thanks to https://stackoverflow.com/a/52783240/19648688
    """
    if not hasattr(obj, "__module__"):
        return False
    module: str = obj.__module__.split(".")[0]
    return module == cs.__name__


def quad_form(
    A: Union[cs.SX, cs.MX, cs.DM], x: Union[cs.SX, cs.MX, cs.DM]
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Calculates quadratic form `x.T*A*x`."""
    if A.is_vector():
        A = cs.diag(A)
    return cs.bilin(A, x, x)


def norm_ppf(
    p: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = 0,
    scale: Union[cs.SX, cs.MX, cs.DM] = 1,
) -> Union[cs.SX, cs.MX, cs.DM]:
    """CasADi version of `scipy.stats.norm.ppf`."""
    return math.sqrt(2) * scale * cs.erfinv(2 * p - 1) + loc
