import casadi as cs
import numpy as _np
from dataclasses import dataclass
from functools import partial
from itertools import count
from mpc.mpc_debug import MPCDebug
from typing import Any, Union


# NOTE: np.flatten and cs.vec operate on row- and column-wise, respectively!


@dataclass(frozen=True)
class Solution:
    '''A class containing information on the solution of an MPC run.'''
    f: float
    vars: dict[str, cs.SX]
    vals: dict[str, _np.ndarray]
    stats: dict[str, Any]
    get_value: partial

    @property
    def status(self) -> str:
        '''Gets the status of the solver at this solution.'''
        return self.stats['return_status']

    @property
    def success(self) -> bool:
        '''Gets whether the MPC was run successfully.'''
        # return self.status in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
        return self.stats['success']

    def value(self, x: cs.SX) -> _np.ndarray:
        '''Gets the value of the expression.'''
        return self.get_value(x)


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
        self.h, self.lbh, self.ubh = cs.SX(), _np.array([]), _np.array([])
        self.lam_h = cs.SX()

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
        '''Number of equality constraints in the MPC problem.'''
        return self.g.shape[0]

    @property
    def nh(self) -> int:
        '''Number of inequality constraints in the MPC problem.'''
        return self.h.shape[0]

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
        self, name: str, expr1: cs.SX, op: str, expr2: cs.SX
    ) -> tuple[cs.SX, cs.SX]:
        '''
        Adds a constraint to the MPC problem, e.g., 'expr1 <= expr2'.

        Parameters
        ----------
        name : str
            Name of the new constraint. Must not be already in use.
        expr1 : casadi.SX
            Symbolic expression of the left-most term of the constraint.
        op: str, {'=', '==', '>', '>=', '<=', '<='}
            Operator relating the two expressions.
        expr2 : casadi.SX
            Symbolic expression of the right-most term of the constraint.

        Returns
        -------
        expr : casadi.SX
            The constraint expression in canonical form, i.e., h(x,u) = 0 or 
            g(x,u) <= 0.
        lam : casadi.SX
            The symbol corresponding to the new constraint's multipliers.
        '''
        assert name not in self.cons, f'Constraint {name} already exists.'
        expr = expr1 - expr2
        dims = expr.shape

        # create bounds
        if op in {'=', '=='}:
            is_eq = True
            lb, ub = _np.zeros(dims), _np.zeros(dims)
        elif op in {'<', '<='}:
            is_eq = False
            lb, ub = _np.full(dims, -_np.inf), _np.zeros(dims)
        elif op in {'>', '>='}:
            is_eq = False
            expr = -expr
            lb, ub = _np.full(dims, -_np.inf), _np.zeros(dims)
        else:
            raise ValueError(f'Unrecognized operator {op}.')
        expr = cs.simplify(expr)
        lb, ub = cs.vec(lb).full().flatten(), cs.vec(ub).full().flatten()

        # save to internal structures
        self.cons[name] = expr
        group = 'g' if is_eq else 'h'
        setattr(self, group,
                cs.vertcat(getattr(self, group), cs.vec(expr)))
        setattr(self, f'lb{group}',
                _np.concatenate((getattr(self, f'lb{group}'), lb)))
        setattr(self, f'ub{group}',
                _np.concatenate((getattr(self, f'ub{group}'), ub)))
        self.debug._register(group, name, dims)

        # create also the multiplier associated to the constraint
        lam = cs.SX.sym(f'lam_{group}_{name}', *dims)
        setattr(self, f'lam_{group}',
                cs.vertcat(getattr(self, f'lam_{group}'), cs.vec(lam)))
        return expr, lam

    def minimize(self, objective: cs.SX) -> None:
        '''Sets the objective function to be minimized.'''
        self.f = objective

    def init_solver(self, opts: dict) -> None:
        '''Initializes the IPOPT solver for this MPC with the given options.'''
        g = cs.vertcat(self.g, self.h)
        nlp = {'x': self.x, 'p': self.p, 'g': g, 'f': self.f}
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
            'lbg': _np.concatenate((self.lbg, self.lbh)),
            'ubg': _np.concatenate((self.ubg, self.ubh)),
        }
        if vals0 is not None:
            kwargs['x0'] = _np.clip(
                subsevalf(self.x, self.vars, vals0), self.lbx, self.ubx)
        sol: dict[str, cs.DM] = self.solver(**kwargs)

        # extract lam_x
        lam_lbx = -_np.minimum(sol['lam_x'], 0)
        lam_ubx = _np.maximum(sol['lam_x'], 0)

        # extract lam_g and lam_h
        lam_g = sol['lam_g'][:self.ng, :]
        lam_h = sol['lam_g'][self.ng:, :]

        # build info
        S = cs.vertcat(self.p, self.x, self.lam_g, self.lam_h, self.lam_lbx,
                       self.lam_ubx)
        D = cs.vertcat(p, sol['x'], lam_g, lam_h, lam_lbx, lam_ubx)
        get_value = partial(subsevalf, old=S, new=D)

        # build vals
        vals = {name: get_value(var) for name, var in self.vars.items()}

        # build solution
        sol_ = Solution(f=float(sol['f']), vars=self.vars.copy(), vals=vals,
                        get_value=get_value, stats=self.solver.stats().copy())
        self.failures += int(not sol_.success)
        return sol_

    def __str__(self) -> str:
        '''Returns the MPC name and a short description.'''
        msg = 'not initialized' if self.solver is None else 'initialized'
        C = len(self.cons)
        return f'{type(self).__name__} {{\n' \
               f'  name: {self.name}\n' \
               f'  #variables: {len(self.vars)} (nx={self.nx})\n' \
               f'  #parameters: {len(self.pars)} (np={self.np})\n' \
               f'  #constraints: {C} (ng={self.ng}, nh={self.nh})\n' \
               f'  CasADi solver {msg}.\n}}'

    def __repr__(self) -> str:
        '''Returns the string representation of the MPC instance.'''
        return str(self)


def subsevalf(
    expr: cs.SX,
    old: Union[cs.SX, dict[str, cs.SX], list[cs.SX], tuple[cs.SX]],
    new: Union[cs.SX, dict[str, cs.SX], list[cs.SX], tuple[cs.SX]],
    eval: bool = True
) -> Union[cs.SX, _np.ndarray]:
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
        expr = cs.DM(expr).full().squeeze()  # faster
        # expr = _np.squeeze(_np.array(cs.evalf(expr)))
    return expr
