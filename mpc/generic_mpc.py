import casadi as cs
import numpy as np
from itertools import count

# NOTE: np.flatten and cs.vec operate on row- and column-wise, respectively!


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
        self._init_fields()

        # initialize empty class parameters
        self.f = None  # objective
        self.vars, self.pars = {}, {}
        self.p = cs.SX()
        self.x, self.lbx, self.ubx = cs.SX(), cs.SX(), cs.SX()
        self.lam_lbx, self.lam_ubx = cs.SX(), cs.SX()
        self.g, self.lbg, self.ubg = cs.SX(), cs.SX(), cs.SX()
        self.lam_g = cs.SX(), cs.SX()
        self.Ig_eq, self.Ig_ineq = set(), set()

        # initialize solver
        self.solver, self.opts = None, None

        # others
        self.failures = 0

    @property
    def nx(self) -> int:
        '''Number of variables in the MPC problem.'''
        return self.x.shape[0]

    @property
    def ng(self) -> int:
        '''Number of constraints in the MPC problem.'''
        return self.g.shape[0]

    @property
    def lagrangian(self) -> cs.SX:
        '''Lagrangian of the MPC problem.'''
        return (self.f +
                cs.dot(self.lam_g, self.g) +
                cs.dot(self.lam_lbx.T, self.lbx - self.x) +
                cs.dot(self.lam_ubx.T, self.x - self.ubx))

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
        lb: np.ndarray = -np.inf, ub: np.ndarray = np.inf
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
        lb, ub = np.broadcast_to(lb, dims), np.broadcast_to(ub, dims)
        assert np.all(lb <= ub), 'Improper variable bounds.'

        var = cs.SX.sym(name, *dims)
        self.vars[name] = var
        self.x = cs.vertcat(self.x, cs.vec(var))
        self.lbx = np.array(cs.vertcat(self.lbx, cs.vec(lb)))
        self.ubx = np.array(cs.vertcat(self.ubx, cs.vec(ub)))

        # create also the multiplier associated to the variable
        lam_lb = cs.SX.sym(f'lam_lb_{name}', *dims)
        self.lam_lbx = cs.vertcat(self.lam_lbx, lam_lb)
        lam_ub = cs.SX.sym(f'lam_ub{name}', *dims)
        self.lam_ubx = cs.vertcat(self.lam_ubx, lam_ub)
        return var, lam_lb, lam_ub

    def add_con(
        self, name: str, g: cs.SX, lb: np.ndarray, ub: np.ndarray
    ) -> cs.SX:
        '''
        Adds a constraint to the MPC problem.

        Parameters
        ----------
        name : str
            Name of the new constraint.
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
        dims = g.shape
        lb, ub = np.broadcast_to(lb, dims), np.broadcast_to(ub, dims)
        assert np.all(lb <= ub), 'Improper variable bounds.'

        self.g = cs.vertcat(self.g, cs.vec(g))
        self.lbg = np.array(cs.vertcat(self.lbg, cs.vec(lb)))
        self.ubg = np.array(cs.vertcat(self.ubg, cs.vec(ub)))

        # save indices of this constraint to either eq. or ineq. set
        ng, L = self.ng, g.numel()
        (self.Ig_eq if np.all(lb == ub) else self.Ig_ineq).update(
            range(ng - L, ng))

        # create also the multiplier associated to the constraint
        lam = cs.SX.sym(f'lam_g_{name}', *dims)
        self.lam_g = cs.vertcat(self.lam_g, lam)
        return lam

    def minimize(self, objective: cs.SX) -> None:
        '''Sets the objective function to be minimized.'''
        self.f = objective

    def init_solver(self, opts: ...) -> None:
        '''Initializes the IPOPT solver for this MPC with the given options.'''
        nlp = {'x': self.x, 'p': self.p, 'g': self.g, 'f': self.f}
        self.solver = cs.nlpsol(f'nlpsol_{self.name}', 'ipopt', nlp, opts)
        self.opts = opts

    def solve():
        # TODO: write solve method
        pass


def subsevalf(
    expr: cs.SX, old: cs.SX, new: cs.SX, eval: bool = True
    ) -> cs.SX | np.ndarray:
    '''
    Substitute in the expression the old variable with
    the new one, evaluating the expression if required.
    
    Parameters
    ----------
    expr : casadi.SX
        Expression for substitution and, possibly, evaluation.
    old : casadi.SX, or collection of
        Old variable to be substituted.
    new : casadi.SX, or collection of
        New variable that substitutes the old one.
    eval : bool, optional
        Evaluates also the new expression. By default, true.

    Returns
    -------
    new_expr : casadi.SX | np.ndarray
        New expression after substitution and, possibly, evaluation.
    '''
    if isinstance(old, dict):
        old = tuple(old.values())
        new = tuple(new.values())

    if not isinstance(old, (tuple, list)):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        expr = cs.substitute(expr, old, new)

    if eval:
        expr = np.array(cs.evalf(expr))
    return expr
