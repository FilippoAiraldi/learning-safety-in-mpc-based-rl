import warnings
import casadi as cs
import numpy as _np
from itertools import count
from functools import partial
from dataclasses import dataclass
from mpc.mpc_debug import MPCDebug


# NOTE: np.flatten and cs.vec operate on row- and column-wise, respectively!


@dataclass(frozen=True)
class Solution:
    '''A class containing information on the solution of an MPC run.'''
    f: float
    vals: dict[str, _np.ndarray]
    msg: str
    success: bool
    _get_value: partial

    def value(self, x: cs.SX) -> _np.ndarray:
        '''Gets the value of the expression.'''
        return self._get_value(x)


class GenericMPC:
    '''
    The generic MPC class is a controller that solves an optimization problem 
    to yield the (possibly, sub-) optimal action, according to the prediction 
    model, in the current state.

    This is a generic class in the sense that it does not solve one problem in 
    particular, but only offers the general methods to do so.
    '''
    _ids = count(0)

    def __init__(self, name: str = None) -> None:
        '''Creates an MPC controller instance with a given name.'''
        self.id = next(self._ids)
        self.name = f'MPC{self.id}' if name is None else name

        # initialize empty class parameters
        self.f: cs.SX = None  # objective
        self.vars: dict[str, cs.SX] = {}
        self.pars: dict[str, cs.SX] = {}
        self.cons: dict[str, cs.SX] = {}
        self.p = cs.SX()
        self.x, self.lbx, self.ubx = cs.SX(), _np.array([]), _np.array([])
        self.lam_lbx, self.lam_ubx = cs.SX(), cs.SX()
        self.g, self.lbg, self.ubg = cs.SX(), _np.array([]), _np.array([])
        self.lam_g = cs.SX()
        self.Ig_eq, self.Ig_ineq = set(), set()

        # initialize solver
        self.solver: cs.Function = None
        self.opts: dict = None

        # others
        self.failures = 0
        self.debug = MPCDebug()

    @property
    def nx(self) -> int:
        '''Number of variables in the MPC problem.'''
        return self.x.shape[0]

    @property
    def np(self) -> int:
        '''Number of parameters in the MPC problem.'''
        return self.p.shape[0]

    @property
    def ng(self) -> int:
        '''Number of constraints in the MPC problem.'''
        return self.g.shape[0]

    @property
    def ng_eq(self) -> int:
        '''Number of equality constraints in the MPC problem.'''
        return len(self.Ig_eq)

    @property
    def ng_ineq(self) -> int:
        '''Number of inequality constraints in the MPC problem.'''
        return len(self.Ig_ineq)

    @property
    def g_eq(self) -> tuple[cs.SX, cs.SX]:
        '''Vector of equality constraints and their multipliers.'''
        inds = tuple(self.Ig_eq)
        try:
            return self.g[inds], self.lam_g[inds]
        except Exception:
            inds = _np.array(inds)
            return self.g[inds], self.lam_g[inds]

    @property
    def g_ineq(self) -> tuple[cs.SX, cs.SX]:
        '''Vector of inequality constraints and their multipliers.'''
        inds = tuple(self.Ig_ineq)
        try:
            return self.g[inds], self.lam_g[inds]
        except Exception:
            inds = _np.array(inds)
            return self.g[inds], self.lam_g[inds]

    def add_par(self, name: str, *dims: int) -> cs.SX:
        '''
        Adds a parameter to the MPC problem.

        Parameters
        ----------
        name : str
            Name of the new parameter. Must not be already in use.
        dims : int...
            Dimensions of the new parameter.

        Returns
        -------
        par : casadi.SX
            The symbol of the new parameter.
        '''
        assert name not in self.pars, f'Parameter {name} already exists.'
        par = cs.SX.sym(name, *dims)
        self.pars[name] = par
        self.p = cs.vertcat(self.p, cs.vec(par))
        return par

    def add_var(
        self, name: str,
        *dims: int,
        lb: _np.ndarray = -_np.inf, ub: _np.ndarray = _np.inf
    ) -> tuple[cs.SX, cs.SX, cs.SX]:
        '''
        Adds a variable to the MPC problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        dims : int...
            Dimensions of the new variable.
        lb, ub: array_like, optional
            Lower and upper bounds of the new variable. By default, unbounded.
            If provided, their dimension must be broadcastable.

        Returns
        -------
        var : casadi.SX
            The symbol of the new variable.
        lam_lb : casadi.SX
            The symbol corresponding to the new variable lower bound 
            constraint's multipliers.
        lam_ub : casadi.SX
            Same as above, for upper bound.
        '''
        assert name not in self.vars, f'Variable {name} already exists.'
        lb, ub = _np.broadcast_to(lb, dims), _np.broadcast_to(ub, dims)
        assert _np.all(lb < ub), 'Improper variable bounds.'

        var = cs.SX.sym(name, *dims)
        self.vars[name] = var
        self.x = cs.vertcat(self.x, cs.vec(var))
        self.lbx = _np.concatenate((self.lbx, cs.vec(lb).full().flatten()))
        self.ubx = _np.concatenate((self.ubx, cs.vec(ub).full().flatten()))
        self.debug._register('x', name, dims)

        # create also the multiplier associated to the variable
        lam_lb = cs.SX.sym(f'lam_lb_{name}', *dims)
        self.lam_lbx = cs.vertcat(self.lam_lbx, cs.vec(lam_lb))
        lam_ub = cs.SX.sym(f'lam_ub_{name}', *dims)
        self.lam_ubx = cs.vertcat(self.lam_ubx, cs.vec(lam_ub))
        return var, lam_lb, lam_ub

    def add_con(
        self, name: str, g: cs.SX, lb: _np.ndarray, ub: _np.ndarray
    ) -> cs.SX:
        '''
        Adds a constraint to the MPC problem.

        Parameters
        ----------
        name : str
            Name of the new constraint. Must not be already in use.
        g : casadi.SX
            Symbolic expression of the new constraint.
        lb, ub: array_like
            Lower and upper bounds of the new constraint. Must be constant and 
            broadcastable to the size of the constraint expression.

        Returns
        -------
        lam_g : casadi.SX
            The symbol corresponding to the new constraint's multipliers.
        '''
        assert name not in self.cons, f'Constraint {name} already exists.'
        dims = g.shape
        lb, ub = _np.broadcast_to(lb, dims), _np.broadcast_to(ub, dims)
        assert _np.all(lb <= ub), 'Improper variable bounds.'

        # warn if any redundant constraints, i.e., lb=-inf, ub=+inf
        lb = cs.vec(lb).full().flatten()
        ub = cs.vec(ub).full().flatten()
        redundant = _np.isneginf(lb) & _np.isposinf(ub)
        if redundant.any():
            warnings.warn(f'Found {redundant.sum()} redundant entries in '
                          f'constraint \'{name}\'.')

        # save to internal structures
        self.cons[name] = g
        self.g = cs.vertcat(self.g, cs.vec(g))
        self.lbg = _np.concatenate((self.lbg, lb))
        self.ubg = _np.concatenate((self.ubg, ub))
        self.debug._register('g', name, dims)

        # save indices of this constraint to either eq. or ineq. set
        ng, L = self.ng, g.numel()
        (self.Ig_eq if _np.all(lb == ub) else self.Ig_ineq).update(
            range(ng - L, ng))

        # create also the multiplier associated to the constraint
        lam = cs.SX.sym(f'lam_g_{name}', *dims)
        self.lam_g = cs.vertcat(self.lam_g, cs.vec(lam))
        return lam

    def minimize(self, objective: cs.SX) -> None:
        '''Sets the objective function to be minimized.'''
        self.f = objective

    def init_solver(self, opts: dict) -> None:
        '''Initializes the IPOPT solver for this MPC with the given options.'''
        nlp = {'x': self.x, 'p': self.p, 'g': self.g, 'f': self.f}
        self.solver = cs.nlpsol(f'nlpsol_{self.name}', 'ipopt', nlp, opts)
        self.opts = opts

    def solve(
        self, pars: dict[str, _np.ndarray],
        vals0: dict[str, _np.ndarray] = None
    ) -> Solution:
        '''
        Solves the MPC optimization problem.

        Parameters
        ----------
        pars : dict[str, array_like]
            Dictionary containing, for each parameter in the problem, the 
            corresponding numerical value.
        vals0 : dict[str, array_like], optional
            Dictionary containing, for each variable in the problem, the 
            corresponding initial guess.

        Returns
        -------
        sol : Solution
            A solution object containing all the information.
        '''
        assert self.solver is not None, 'Solver uninitialized.'

        # convert to nlp format and solve
        p = subsevalf(self.p, self.pars, pars)
        kwargs = {
            'p': p,
            'lbx': self.lbx,
            'ubx': self.ubx,
            'lbg': self.lbg,
            'ubg': self.ubg,
        }
        if vals0 is not None:
            kwargs['x0'] = _np.clip(
                subsevalf(self.x, self.vars, vals0), self.lbx, self.ubx)
        sol: dict[str, cs.DM] = self.solver(**kwargs)

        # get return status
        status = self.solver.stats()['return_status']
        success = status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
        self.failures += int(not success)

        # build info
        lam_lbx_ = -_np.minimum(sol['lam_x'], 0)
        lam_ubx_ = _np.maximum(sol['lam_x'], 0)
        S = cs.vertcat(self.p, self.x, self.lam_g, self.lam_lbx, self.lam_ubx)
        D = cs.vertcat(p, sol['x'], sol['lam_g'], lam_lbx_, lam_ubx_)
        get_value = partial(subsevalf, old=S, new=D)

        # build vals
        vals = {name: get_value(var) for name, var in self.vars.items()}

        # build solution
        return Solution(f=float(sol['f']), vals=vals, msg=status,
                        success=success, _get_value=get_value)

    def __str__(self) -> str:
        '''Returns the MPC name and a short description.'''
        msg = 'not initialized' if self.solver is None else 'initialized'
        return f'{type(self).__name__} {{\n' \
               f'  name: {self.name}\n' \
               f'  #variables: {len(self.vars)} (nx = {self.nx})\n' \
               f'  #parameters: {len(self.pars)} (np = {self.np})\n' \
               f'  #constraints: {len(self.cons)} (ng = {self.ng})\n' \
               f'  CasADi solver {msg}.\n}}'

    def __repr__(self) -> str:
        '''Returns the string representation of the MPC instance.'''
        return str(self)


def subsevalf(
    expr: cs.SX,
    old: cs.SX | dict[str, cs.SX] | list[cs.SX] | tuple[cs.SX],
    new: cs.SX | dict[str, cs.SX] | list[cs.SX] | tuple[cs.SX],
    eval: bool = True
) -> cs.SX | _np.ndarray:
    '''
    Substitute in the expression the old variable with
    the new one, evaluating the expression if required.

    Parameters
    ----------
    expr : casadi.SX
        Expression for substitution and, possibly, evaluation.
    old : casadi.SX (or collection of)
        Old variable to be substituted.
    new : numpy array or casadi.SX (or collection of)
        New variable that substitutes the old one.
    eval : bool, optional
        Evaluates also the new expression. By default, true.

    Returns
    -------
    new_expr : casadi.SX | np.ndarray
        New expression after substitution and, possibly, evaluation.
    '''
    if isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])
    elif isinstance(old, (tuple, list)):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        expr = cs.substitute(expr, old, new)

    if eval:
        expr = _np.squeeze(cs.DM(expr).full())  # faster
        # expr = _np.squeeze(_np.array(cs.evalf(expr)))
    return expr
