import numpy as np
from envs.wrappers import RecordData
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.ticker import PercentFormatter
from itertools import product


def plot_trajectory(env: RecordData, i: int):
    '''Plots the i-th trajecetory of the recorded data.'''
    x0, xf = env.x0, env.xf
    data = env.observations_history[i][:3, :]  # x, y, z

    # prepare stuff for plotting
    labels = ('Pos: x [m]', 'Pos: y [m]', 'Altitude [m]')
    linecollection_kwargs = {'cmap': 'coolwarm', 'norm': plt.Normalize(0, 100)}
    patch_kwargs = {
        'facecolor': 'k', 'alpha': 0.1, 'edgecolor': 'k', 'linewidth': 2}
    order = ((0, 1, 2), (1, 2), (0, 1), (0, 2))

    # plot
    fig = plt.figure(figsize=(12, 7))
    for ax_num, inds in enumerate(order, start=1):
        inds = np.array(inds)
        points = np.expand_dims(data[inds, :].T, axis=1)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if len(inds) == 3:
            # 3D plot
            ax = fig.add_subplot(2, 2, ax_num, projection='3d')

            # plot multicolored 3D line
            lc = Line3DCollection(segments, **linecollection_kwargs)
            lc.set_array(np.linspace(0, 100, segments.shape[0]))
            line = ax.add_collection3d(lc)
            fig.colorbar(line, ax=ax, location='left',
                         format=PercentFormatter())

            # initial and termination state
            ax.scatter(*x0[:3], marker='o', color='k')
            ax.scatter(*xf[:3], marker='x', color='k')

            # plot state constraints
            # many thanks to: https://www.pythonpool.com/matplotlib-draw-rectangle/
            Z = np.array(list(product(*env.x_bounds[:3, :])))
            verts = [[Z[0], Z[4], Z[6], Z[2]],
                     [Z[1], Z[5], Z[7], Z[3]],
                     [Z[0], Z[4], Z[5], Z[1]],
                     [Z[6], Z[2], Z[3], Z[7]],
                     [Z[4], Z[6], Z[7], Z[5]],
                     [Z[1], Z[3], Z[2], Z[0]]]
            ax.add_collection3d(Poly3DCollection(verts, **patch_kwargs))

        else:
            # 2D plot
            ax = fig.add_subplot(2, 2, ax_num)

            # plot multicolored 2D line
            lc = LineCollection(segments, **linecollection_kwargs)
            lc.set_array(np.linspace(0, 100, segments.shape[0]))
            line = ax.add_collection(lc)

            # initial and termination state
            ax.plot(*x0[inds], marker='o', color='k')
            ax.plot(*xf[inds], marker='x', color='k')

            # plot state constraints
            ax.add_patch(plt.Rectangle(env.x_bounds[inds, 0],
                                       *np.diff(env.x_bounds[inds, :]),
                                       **patch_kwargs))

        # add labels and impose limits
        for axisname, ind in zip(('x', 'y', 'z'), inds):
            getattr(ax, f'set_{axisname}label')(labels[ind])
            lim = getattr(ax, f'get_{axisname}lim')()
            getattr(ax, f'set_{axisname}lim')(
                min(data[ind].min(), env.x_bounds[ind, 0], lim[0]),
                max(data[ind].max(), env.x_bounds[ind, 1], lim[1])
            )

    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show(block=False)
