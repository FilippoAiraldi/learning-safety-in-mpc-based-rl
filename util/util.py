import numpy as np
import casadi as cs
from envs import QuadRotorEnv


def quadrotor_dynamics(
    env: QuadRotorEnv
) -> tuple[cs.Function, dict[str, cs.SX], dict[str, cs.SX]]:
    '''
    Gets the symbolic quadrotor dynamics.

    Parameters
    ----------
    env : QuadRotorEnv
        A quadrotor environment from which to extract the symbolic dynamics.

    Returns
    -------
    f : casadi.Function
        A function computing the discrete-time dynamics
    vars, pars : dict[str, casadi.SX]
        Dictionaries containing variables and parameters involved in the 
        dynamics.
    '''
    # create system variables - without noise!
    vars_ = {
        'x': cs.SX.sym('x', env.nx, 1),
        'u': cs.SX.sym('u', env.nu, 1)
    }

    # create system parameters
    pars = {}
    for par, val in env.pars.__dict__.items():
        if isinstance(val, (int, float)):
            pars[par] = cs.SX.sym(par, 1, 1)
        elif isinstance(val, np.ndarray):
            pars[par] = cs.SX.sym(par, *val.shape)
        # else do nothing for dict, set, etc.

    # create matrices
    Ad = cs.vertcat(
        cs.horzcat(pars['pitch_d'], 0),
        cs.horzcat(0, pars['roll_d']))
    Add = cs.vertcat(
        cs.horzcat(pars['pitch_dd'], 0),
        cs.horzcat(0, pars['roll_dd']))
    A = pars['T'] * cs.vertcat(
        cs.horzcat(np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))),
        cs.horzcat(np.zeros((2, 6)), np.eye(2) * pars['g'], np.zeros((2, 2))),
        np.zeros((1, 10)),
        cs.horzcat(np.zeros((2, 6)), -Ad, np.eye(2)),
        cs.horzcat(np.zeros((2, 6)), -Add, np.zeros((2, 2)))
    ) + np.eye(10)
    B = pars['T'] * cs.vertcat(
        np.zeros((5, 3)),
        cs.horzcat(0, 0, pars['thrust_coeff']),
        np.zeros((2, 3)),
        cs.horzcat(pars['pitch_gain'], 0, 0),
        cs.horzcat(0, pars['roll_gain'], 0)
    )
    e = cs.vertcat(
        np.zeros((5, 1)),
        - pars['T'] * pars['g'],
        np.zeros((4, 1))
    )

    # compute dynamics function
    x_new = A @ vars_['x'] + B @ vars_['u'] + e
    f = cs.Function(
        'f',
        list(vars_.values()) + list(pars.values()), (x_new,),
        list(vars_.keys()) + list(pars.keys()), ('x+',)
    )

    return f, vars_, pars
