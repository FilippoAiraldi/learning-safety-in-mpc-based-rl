import casadi as cs
import numpy as np
from mpc.generic_mpc import GenericMPC
from typing import Any, Generic, TypeVar


MPCType = TypeVar('MPCType', bound=GenericMPC)


class DifferentiableMPC(Generic[MPCType]):
    def __init__(
        self,
        mpc: MPCType,
        reduce_redundant_x_bounds: bool = True
    ) -> None:
        '''
        Wraps an MPC controller to allow computing its symbolic derivatives.

        Parameters
        ----------
        mpc : GenericMPC or subclasses
            The MPC instance to wrap.
        reduce_redundant_x_bounds : bool, optional
            Whether the redundant (i.e., lb=-inf, ub=+inf) on the decision 
            variables should be removed automatically.
        '''
        self._mpc = mpc
        self.reduce_redundant_x_bounds = reduce_redundant_x_bounds

    @property
    def mpc(self) -> MPCType:
        return self._mpc

    @property
    def _non_redundant_x_bound_indices(self) -> np.ndarray:
        '''
        Gets the indices of lbx and ubx which are not redundant, i.e., 
                idx = { i : not ( lbx[i]=-inf and ubx[i]=+inf ) },
        if not disabled. Otherwise, simply returns all the indices
        '''
        return (
            np.where((self._mpc.lbx != -np.inf) | (self._mpc.ubx != np.inf))[0]
            if self.reduce_redundant_x_bounds else
            np.arange(self._mpc.nx)
        )

    @property
    def lagrangian(self) -> cs.SX:
        '''Lagrangian of the MPC problem.'''
        idx = self._non_redundant_x_bound_indices
        h_lbx = self._mpc.lbx[idx, None] - self._mpc.x[idx]
        h_ubx = self._mpc.x[idx] - self._mpc.ubx[idx, None]
        return (self._mpc.f +
                cs.dot(self._mpc.lam_g, self._mpc.g) +
                cs.dot(self._mpc.lam_h, self._mpc.h) +
                cs.dot(self._mpc.lam_lbx[idx], h_lbx) +
                cs.dot(self._mpc.lam_ubx[idx], h_ubx))

    @property
    def kkt_conditions(self) -> tuple[cs.SX, cs.SX, cs.SX]:
        '''
        Gets:
            1-2) the KKT matrix defined as
                    [            dLdw             ]
                K = [            G_eq             ],
                    [ diag(lam_ineq)*H_ineq + tau ]
               where w = [x, u, slacks] are the primal decision variables, and 
               tau is the IPOPT barrier parameter.

            3) the collection y of primal-dual variables defined as
                    [    w     ]
                y = [ lam_G_eq ]
                    [lam_H_ineq].
        '''
        # compute derivative of lagrangian - use w to discern 'x' the state
        # from 'x' the primal variable of the MPC
        dLdw = cs.simplify(cs.jacobian(self.lagrangian, self._mpc.x).T)

        # get non redundant inequalities on x
        idx = self._non_redundant_x_bound_indices
        h_lbx = self._mpc.lbx[idx, None] - self._mpc.x[idx]
        h_ubx = self._mpc.x[idx] - self._mpc.ubx[idx, None]
        h_lam_lbx = self._mpc.lam_lbx[idx]
        h_lam_ubx = self._mpc.lam_ubx[idx]

        # include barrier function parameter
        tau: cs.SX = cs.SX.sym('tau', 1, 1)

        # build KKT conditions and collection of primal-dual variables
        R: cs.SX = cs.vertcat(
            dLdw,
            self._mpc.g,
            self._mpc.lam_h * self._mpc.h + tau,
            h_lam_lbx * h_lbx,  # is tau required here?
            h_lam_ubx * h_ubx,  # is tau required here?
        )
        y: cs.SX = cs.vertcat(self._mpc.x, self._mpc.lam_g,
                              self._mpc.lam_h, h_lam_lbx, h_lam_ubx)
        return R, tau, y

    def __getattr__(self, name) -> Any:
        '''Reroutes attributes to the wrapped MPC instance.'''
        return getattr(self.mpc, name)

    def __str__(self) -> str:
        '''Returns the wrapper name and the unwrapped MPC string.'''
        return f'<{type(self).__name__}: {self.mpc}>'

    def __repr__(self) -> str:
        '''Returns the string representation of the wrapper.'''
        return str(self)
