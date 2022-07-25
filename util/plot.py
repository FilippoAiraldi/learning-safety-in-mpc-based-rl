import numpy as np
from wrappers import RecordData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.ticker import PercentFormatter
from itertools import product


def plot_trajectory(env: RecordData, i: int):
    '''Plots the i-th trajecetory of the recorded data.'''
    x = env.observations_history[i][0]
    y = env.observations_history[i][1]
    z = env.observations_history[i][2]
    x0, xf = env.x0, env.xf
    fig = plt.figure(figsize=(12, 7))
    patch_kwargs = {
        'facecolor': 'k', 'alpha': 0.1, 'edgecolor': 'k', 'linewidth': 2}

    # 3d trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(
        segments, cmap='coolwarm', norm=plt.Normalize(0, 100))
    lc.set_array(np.linspace(0, 100, x.size - 1))
    line = ax.add_collection3d(lc)
    ax.scatter(x0[0], x0[1], x0[2], marker='o', color='k')
    ax.scatter(xf[0], xf[1], xf[2], marker='x', color='k')
    fig.colorbar(line, ax=ax, location='left', format=PercentFormatter())
    #
    Z = np.array(list(product(*env.x_bounds[:3, :])))
    verts = [[Z[0], Z[4], Z[6], Z[2]],
             [Z[1], Z[5], Z[7], Z[3]],
             [Z[0], Z[4], Z[5], Z[1]],
             [Z[6], Z[2], Z[3], Z[7]],
             [Z[4], Z[6], Z[7], Z[5]],
             [Z[1], Z[3], Z[2], Z[0]]]
    # many thanks to: https://www.pythonpool.com/matplotlib-draw-rectangle/
    ax.add_collection3d(Poly3DCollection(verts, **patch_kwargs))
    #
    ax.set_xlabel('Pos: x [m]')
    ax.set_ylabel('Pos: y [m]')
    ax.set_zlabel('Altitude [m]')
    ax.set_xlim(min(x.min(), env.x_bounds[0, 0], ax.get_xlim()[0]),
                max(x.max(), env.x_bounds[0, 1], ax.get_xlim()[1]))
    ax.set_ylim(min(y.min(), env.x_bounds[1, 0], ax.get_ylim()[0]),
                max(y.max(), env.x_bounds[1, 1], ax.get_ylim()[1]))
    ax.set_zlim(min(z.min(), env.x_bounds[2, 0], ax.get_zlim()[0]),
                max(z.max(), env.x_bounds[2, 1], ax.get_zlim()[1]))

    # y-z trajectory
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(y, z)
    ax.plot(x0[1], x0[2], marker='o', color='k')
    ax.plot(xf[1], xf[2], marker='x', color='k')
    ax.add_patch(plt.Rectangle(env.x_bounds[(1, 2), 0],
                               *np.diff(env.x_bounds[(1, 2), :]), 
                               **patch_kwargs))
    ax.set_xlabel('Pos: y [m]')
    ax.set_ylabel('Altitude [m]')

    # x-y trajectory
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x, y)
    ax.plot(x0[0], x0[1], marker='o', color='k')
    ax.plot(xf[0], xf[1], marker='x', color='k')
    ax.add_patch(plt.Rectangle(env.x_bounds[(0, 1), 0],
                               *np.diff(env.x_bounds[(0, 1), :]), 
                               **patch_kwargs))
    ax.set_xlabel('Pos: x [m]')
    ax.set_ylabel('Pos: y [m]')

    # x-z trajectory
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x, z)
    ax.plot(x0[0], x0[2], marker='o', color='k')
    ax.plot(xf[0], xf[2], marker='x', color='k')
    ax.add_patch(plt.Rectangle(env.x_bounds[(0, 2), 0],
                               *np.diff(env.x_bounds[(0, 2), :]), 
                               **patch_kwargs))
    ax.set_xlabel('Pos: x [m]')
    ax.set_ylabel('Altitude [m]')

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show(block=False)
