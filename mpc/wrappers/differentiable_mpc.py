import casadi as cs
import numpy as np
from typing import Any, Generic, TypeVar


MPCType = TypeVar('MPCType')


class DifferentiableMPC(Generic[MPCType]):
    def __init__(self, mpc: MPCType) -> None:
        '''
        Wraps an MPC controller to allow computing its symbolic derivatives.

        Parameters
        ----------
        mpc : GenericMPC or subclasses
            The MPC instance to wrap.
        '''
        self._mpc = mpc

    @property
    def mpc(self) -> MPCType:
        return self._mpc

    @property
    def lagrangian(self) -> cs.SX:
        '''Lagrangian of the MPC problem.'''
        return (self.f +
                cs.dot(self.lam_g, self.g) +
                cs.dot(self.lam_lbx, self.lbx - self.x) +
                cs.dot(self.lam_ubx, self.x - self.ubx))

    @property
    def kkt_matrix(self) -> tuple[cs.SX, cs.SX]:
        '''
        Gets the KKT matrix defined as 
                [          dLdw         ]
            K = [          G_eq         ],
                [ diag(lam_ineq)*H_ineq ]
        where w = [x, u, slacks] are the primal decision variables. Also
        returns the collection y of primal-dual variables defined as
                [   w    ]
            y = [ lam_eq ]
                [lam_ineq].
        '''
        # compute derivative of lagrangian - use w to discern 'x' the state
        # from 'x' the primal variable of the MPC
        dLdw = cs.simplify(cs.jacobian(self.lagrangian, self.x).T)

        # get equality constraints (G_eq = 0)
        g_eq, lam_g_eq = self.g_eq

        # get inequality constraints (H_ineq <= 0)
        g_ineq, lam_g_ineq = self.g_ineq
        g_lbx = self.lbx - self.x
        g_ubx = self.x - self.ubx

        # by using one list we ensure that the order is the same in R and y
        items = [
            (dLdw, self.x, False),
            (g_eq, lam_g_eq, False),        # G
            (g_ineq, lam_g_ineq, True),     # |
            (g_lbx, self.lam_lbx, True),    # | diag(lam)*H
            (g_ubx, self.lam_ubx, True),    # |
        ]

        # build the matrix
        R = cs.vertcat(*(o[0] * o[1] if o[2] else o[0] for o in items))
        # R = cs.vertcat(dLdw,
        #                g_eq,                 # G
        #                g_ineq * lam_g_ineq,  # |
        #                g_lbx * self.lam_lbx, # | diag(lam)*H
        #                g_ubx * self.lam_ubx) # |

        # build the collection of primal-dual variables
        y = cs.vertcat(*(o[1] for o in items))

        return R, y

    def __getattr__(self, name) -> Any:
        '''Reroutes attributes to the wrapped MPC instance.'''
        return getattr(self.mpc, name)

    def __str__(self) -> str:
        '''Returns the wrapper name and the unwrapped MPC string.'''
        return f'<{type(self).__name__}: {self.mpc}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the wrapper.'''
        return str(self)
