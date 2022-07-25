import numpy as np
from wrappers import RecordData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import PercentFormatter


def plot_trajectory(env: RecordData, i: int):
    '''Plots the i-th trajecetory of the recorded data.'''
    x = env.observations_history[i][0]
    y = env.observations_history[i][1]
    z = env.observations_history[i][2]
    x0, xf = env.x0, env.xf
    fig = plt.figure()

    # 3d trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    # ax.plot3D(x, y, z)
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, cmap='RdBu_r', norm=plt.Normalize(0, 100))
    lc.set_array(np.linspace(0, 100, x.size - 1))
    line = ax.add_collection3d(lc)
    ax.scatter(x0[0], x0[1], x0[2], marker='o', color='k')
    ax.scatter(xf[0], xf[1], xf[2], marker='x', color='k')
    fig.colorbar(line, ax=ax, location='left', format=PercentFormatter())
    ax.set_xlim(min(x.min(), ax.get_xlim()[0]), max(x.max(), ax.get_xlim()[1]))
    ax.set_ylim(min(y.min(), ax.get_ylim()[0]), max(y.max(), ax.get_ylim()[1]))
    ax.set_zlim(min(z.min(), ax.get_zlim()[0]), max(z.max(), ax.get_zlim()[1]))
    ax.set_xlabel('Pos: x [m]')
    ax.set_ylabel('Pos: y [m]')
    ax.set_zlabel('Altitude [m]')

    # y-z trajectory
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(y, z)
    ax.plot(x0[1], x0[2], marker='o', color='k')
    ax.plot(xf[1], xf[2], marker='x', color='k')
    ax.set_xlabel('Pos: y [m]')
    ax.set_ylabel('Altitude [m]')

    # x-y trajectory
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, y)
    ax.plot(x0[0], x0[1], marker='o', color='k')
    ax.plot(xf[0], xf[1], marker='x', color='k')
    ax.set_xlabel('Pos: x [m]')
    ax.set_ylabel('Pos: y [m]')

    # x-z trajectory
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x, z)
    ax.plot(x0[0], x0[2], marker='o', color='k')
    ax.plot(xf[0], xf[2], marker='x', color='k')
    ax.set_xlabel('Pos: x [m]')
    ax.set_ylabel('Altitude [m]')

    fig.subplots_adjust(wspace=0.3, hspace=0.3)


def show():
    return plt.show()

###############################################################################
# import numpy as np
# import casadi as cs


# def expK(
#     x: np.ndarray | cs.SX | cs.MX | cs.DM,
#     c: np.ndarray,
#     l: float
# ) -> np.ndarray | cs.SX | cs.MX | cs.DM:
#     '''
#     Exponential Radial Basis Function.

#     Parameters
#     ----------
#     x : array_like
#         Values at which to evaluate the function.
#     c : array_like
#         Center of the function. Size must be compatible with x.
#     l : float
#         Length factor of the function.

#     Returns
#     -------
#     y : scalar
#         The value of the function, computed as y=exp(-|x-c|^2 / 2l).
#     '''
#     x_ = x - c
#     return np.exp(-(x_.T @ x_ / (2 * l)))


# def phi_pos(
#     x: np.ndarray | cs.SX | cs.MX | cs.DM,
#     C: tuple[np.ndarray],
#     l: float = 15
# ) -> np.ndarray:
#     '''TODO'''
#     phi = tuple(expK(x, c, l) for c in C)
#     return cs.vertcat(*phi) if is_casadi_type(x) else np.vstack(phi)


# def is_casadi_type(array_like, recursive: bool = False) -> bool:
#     '''
#     Returns a boolean of whether an object is a CasADi data type or not. If the
#     recursive flag is True, iterates recursively.

#     Parameters
#     ----------
#     array_like : array_like
#         The object to evaluate. Either from numpy or casadi.

#     recursive : bool
#         If the object is a list or tuple, recursively iterate through every
#         subelement. If any of the subelements are a CasADi type, return True.
#         Otherwise, return False.

#     Returns
#     ----------
#     is_casadi : bool
#         A boolean if the object is a CasADi data type.
#     '''
#     if recursive and isinstance(array_like, (list, tuple)):
#         for element in array_like:
#             if is_casadi_type(element, recursive=True):
#                 return True

#     return (
#         isinstance(array_like, cs.MX) or
#         isinstance(array_like, cs.DM) or
#         isinstance(array_like, cs.SX)
#     )
