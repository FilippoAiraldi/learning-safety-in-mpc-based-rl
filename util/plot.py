import numpy as np
from envs.wrappers import RecordData
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from itertools import product


def plot_trajectory_3d(env: RecordData, traj_num: int) -> None:
    '''Plots the i-th trajecetory of the recorded data in 3D.'''
    x0, xf = env.config.x0, env.config.xf
    data = env.observations[traj_num]

    # prepare stuff for plotting
    labels = {0: 'Pos: x [$m$]', 1: 'Pos: y [$m$]', 2: 'Altitude [$m$]',
              6: 'Pitch [$rad$]', 7: 'Roll [$rad$]'}
    linecollection_kwargs = {'cmap': 'coolwarm', 'norm': plt.Normalize(0, 100)}
    patch_kwargs = {
        'facecolor': 'k', 'alpha': 0.1, 'edgecolor': 'k', 'linewidth': 2}

    # plot
    fig = plt.figure(constrained_layout=True)
    G = gridspec.GridSpec(2, 3, figure=fig)
    axes = [
        ((0, 1, 2), fig.add_subplot(G[0, :2], projection='3d')),
        ((1, 2), fig.add_subplot(G[0, 2])),
        ((0, 1), fig.add_subplot(G[1, 0])),
        ((0, 2), fig.add_subplot(G[1, 1])),
        ((6, 7), fig.add_subplot(G[1, 2]))
    ]
    for inds, ax in axes:
        inds = np.array(inds)
        points = np.expand_dims(data[inds, :].T, axis=1)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if len(inds) == 3:
            # plot multicolored 3D line
            lc = Line3DCollection(segments, **linecollection_kwargs)
            lc.set_array(np.linspace(0, 100, segments.shape[0]))
            ax.add_collection3d(lc)

            # initial and termination state
            ax.scatter(*x0[:3], marker='o', color='k')
            ax.scatter(*xf[:3], marker='x', color='k')

            # plot state constraints
            # many thanks to: https://www.pythonpool.com/matplotlib-draw-rectangle/
            Z = np.array(list(product(*env.config.x_bounds[:3, :])))
            verts = [[Z[0], Z[4], Z[6], Z[2]],
                     [Z[1], Z[5], Z[7], Z[3]],
                     [Z[0], Z[4], Z[5], Z[1]],
                     [Z[6], Z[2], Z[3], Z[7]],
                     [Z[4], Z[6], Z[7], Z[5]],
                     [Z[1], Z[3], Z[2], Z[0]]]
            ax.add_collection3d(Poly3DCollection(verts, **patch_kwargs))

        else:
            # plot multicolored 2D line
            lc = LineCollection(segments, **linecollection_kwargs)
            lc.set_array(np.linspace(0, 100, segments.shape[0]))
            ax.add_collection(lc)

            # initial and termination state
            ax.plot(*x0[inds], marker='o', color='k')
            ax.plot(*xf[inds], marker='x', color='k')

            # plot state constraints
            ax.add_patch(plt.Rectangle(env.config.x_bounds[inds, 0],
                                       *np.diff(env.config.x_bounds[inds, :]),
                                       **patch_kwargs))

        # add labels and impose limits
        for axisname, ind in zip(('x', 'y', 'z'), inds):
            getattr(ax, f'set_{axisname}label')(labels[ind])
            lim = getattr(ax, f'get_{axisname}lim')()
            getattr(ax, f'set_{axisname}lim')(
                min(data[ind].min(), env.config.x_bounds[ind, 0], lim[0]),
                max(data[ind].max(), env.config.x_bounds[ind, 1], lim[1])
            )
    plt.show(block=False)


def plot_trajectory_in_time(env: RecordData, traj_num: int) -> None:
    '''Plots the i-th trajecetory of the recorded data.'''

    # prepare for plotting
    xf = env.config.xf
    X = env.observations[traj_num]
    U = env.actions[traj_num]
    R = env.rewards[traj_num]
    t = np.arange(X.shape[1]) * env.config.T  # time
    error = np.linalg.norm(X - xf.reshape(-1, 1), axis=0)
    items = [
        [X[:3].T, ('x', 'y', 'z'), 'Position [$m$]', xf[:3]],
        [X[3:6].T, ('x', 'y', 'z'), 'Speed [$m/s$]', xf[3:6]],
        [X[6:8].T, ('pitch', 'roll'), 'Angle [$rad$]', xf[6:8]],
        [X[8:].T, ('pitch', 'roll'), 'Angular Speed [$rad/s$]', xf[8:]],
        [U[:2].T, ('desired pitch', 'desired roll'), 'Angle [$rad$]', None],
        [U[-1].T, ('desired z acc.',), 'Acceleration [$m/s^2$]', None],
        [error, None, 'Error', env.config.tol],
        [R, None, 'Reward', None],
    ]

    # create figure and grid
    fig = plt.figure(constrained_layout=True)
    G = gridspec.GridSpec(4, 2, figure=fig)

    # do plot
    ax = None
    for i, (x, lgds, ylbl, asymptot) in enumerate(items):
        # create axis
        ax = fig.add_subplot(G[np.unravel_index(i, (G.nrows, G.ncols))],
                             sharex=ax)

        # plot data
        L = x.shape[0]
        lines = ax.plot(t[:L], x)
        if asymptot is not None:
            ax.hlines(asymptot, xmin=t[0], xmax=t[L - 1], linestyles='dashed',
                      colors=[l.get_color() for l in lines], linewidths=0.8)

        # embellish
        if lgds is not None:
            ax.legend(lgds)
        ax.set_ylabel(ylbl)
        if i >= 6:
            ax.set_xlabel('Time [s]')

    ax.set_xlim([t[0], t[-1]])
    plt.show(block=False)
