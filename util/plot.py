import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from agents.wrappers import RecordLearningData
from envs.wrappers import RecordData
from itertools import product
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from util.util import MATLAB_COLORS


LINEWIDTHS = (0.05, 1.5)


def _set_axes3d_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(np.diff(x_limits))
    x_middle = np.mean(x_limits)
    y_range = abs(np.diff(y_limits))
    y_middle = np.mean(y_limits)
    z_range = abs(np.diff(z_limits))
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max(x_range, y_range, z_range)
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_trajectory_3d(env: RecordData, traj_num: int) -> Figure:
    '''Plots the i-th trajecetory of the recorded data in 3D.'''
    x0, xf = env.config.x0, env.config.xf
    data = env.observations[traj_num].T

    # prepare stuff for plotting
    labels = {0: 'Pos: x [$m$]', 1: 'Pos: y [$m$]', 2: 'Altitude [$m$]',
              6: 'Pitch [$rad$]', 7: 'Roll [$rad$]'}
    linecollection_kwargs = {'cmap': 'coolwarm', 'norm': plt.Normalize(0, 100)}
    patch_kwargs = {'alpha': 0.1, 'edgecolor': 'k', 'linewidth': 2}

    # plot
    fig = plt.figure(constrained_layout=True)
    G = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1e-1])
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
            trajectory = ax.add_collection3d(lc)

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
            ax.add_patch(plt.Rectangle(
                env.config.x_bounds[inds, 0],
                *np.diff(env.config.x_bounds[inds, :]).flatten(),
                **patch_kwargs))

        # embellish
        for axisname, ind in zip(('x', 'y', 'z'), inds):
            # add labels
            getattr(ax, f'set_{axisname}label')(labels[ind])

            # impose data limits
            lim = getattr(ax, f'get_{axisname}lim')()
            getattr(ax, f'set_{axisname}lim')(
                min(data[ind].min(), env.config.x_bounds[ind, 0], lim[0]),
                max(data[ind].max(), env.config.x_bounds[ind, 1], lim[1])
            )

        # make axis equal
        if len(inds) == 3:
            _set_axes3d_equal(ax)
        else:
            ax.axis('equal')

    # add colorbar
    cax = fig.add_subplot(G[:, -1])
    fig.colorbar(trajectory, cax=cax, format=PercentFormatter(),
                 label='% of trajectory')
    return fig


def plot_trajectory_in_time(env: RecordData, traj_num: int) -> Figure:
    '''Plots the i-th trajecetory of the recorded data.'''

    # prepare for plotting
    xf = env.config.xf
    x_bnd, u_bnd = env.config.x_bounds, env.config.u_bounds
    X = env.observations[traj_num]
    U = env.actions[traj_num]
    R = env.rewards[traj_num]
    t = np.arange(X.shape[0]) * env.config.T  # time
    error = env.position_error(X)
    items = [
        [X[:, :3], ('x', 'y', 'z'), 'Position [$m$]', xf[:3], x_bnd[:3]],
        [X[:, 3:6], ('x', 'y', 'z'), 'Speed [$m/s$]', xf[3:6], x_bnd[3:6]],
        [X[:, 6:8], ('pitch', 'roll'), 'Angle [$rad$]', xf[6:8], x_bnd[6:8]],
        [X[:, 8:], ('pitch', 'roll'), 'Angular Speed [$rad/s$]', xf[8:], x_bnd[8:]],
        [U[:, :2], ('desired pitch', 'desired roll'), 'Angle [$rad$]', None, u_bnd[:2]],
        [U[:, -1], ('desired z acc.',), 'Acceleration [$m/s^2$]', None, u_bnd[-1]],
        [error, None, 'Termination error', None, env.config.termination_error],
        [R, (f'J = {R.sum():.1f}',), 'Reward', None, None],
    ]

    # create figure and grid
    fig = plt.figure(constrained_layout=True)
    G = gridspec.GridSpec(4, 2, figure=fig)

    # do plot
    ax = None
    for i, (x, lgds, ylbl, asymptot, bnds) in enumerate(items):
        # create axis
        ax = fig.add_subplot(G[np.unravel_index(i, (G.nrows, G.ncols))],
                             sharex=ax)

        # plot data
        L = x.shape[0]
        lines = ax.plot(t[:L], x)
        if asymptot is not None:
            ax.hlines(asymptot, xmin=t[0], xmax=t[L - 1], linestyles='dotted',
                      colors=[l.get_color() for l in lines], linewidths=0.7)
        if bnds is not None:
            ax.hlines(bnds, xmin=t[0], xmax=t[L - 1], linestyles='dashed',
                      colors='k', linewidths=0.7)

        # embellish
        if lgds is not None:
            ax.legend(lgds)
        ax.set_ylabel(ylbl)
        if i >= 6:
            ax.set_xlabel('Time [s]')

    ax.set_xlim([t[0], t[-1]])
    return fig


def plot_performance_and_unsafe_episodes(
    envs: list[RecordData],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    '''
    Plots the performance in each environment and the average performance, 
    as well as the number of unsafe episodes.
    '''
    Nenv, Nep = len(envs), len(envs[0].cum_rewards)
    episodes = np.arange(Nep) + 1

    # compute rewards and mean reward
    rewards: np.ndarray = np.stack([env.cum_rewards for env in envs])
    mean_reward: np.ndarray = rewards.mean(axis=0)

    # compute number of unsafe episodes
    unsafes = np.empty((Nenv, Nep))
    for i, env in enumerate(envs):
        x_bnd, u_bnd = env.config.x_bounds, env.config.u_bounds
        for j, (obs, acts) in enumerate(zip(env.observations, env.actions)):
            # # count unsafe
            # item = not (
            #     ((obs >= x_bnd[:, 0]) & (obs <= x_bnd[:, 1])).all() and
            #     ((acts >= u_bnd[:, 0]) & (acts <= u_bnd[:, 1])).all()
            # )

            # constraint violation
            # item = max(
            #     np.maximum(0, x_bnd[:, 0] - obs).max(),
            #     np.maximum(0, obs - x_bnd[:, 1]).max(),
            #     np.maximum(0, u_bnd[:, 0] - acts).max(),
            #     np.maximum(0, acts - u_bnd[:, 1]).max()
            # )
            item = (
                np.maximum(0, x_bnd[:, 0] - obs).sum() +
                np.maximum(0, obs - x_bnd[:, 1]).sum() +
                np.maximum(0, u_bnd[:, 0] - acts).sum() +
                np.maximum(0, acts - u_bnd[:, 1]).sum()
            )

            # assign to data
            unsafes[i, j] = item
    mean_unsafe: np.ndarray = unsafes.mean(axis=0)

    # create figure and grid
    if fig is None:
        fig = plt.figure(constrained_layout=True)
        G = gridspec.GridSpec(1, 2, figure=fig)
        axs = (fig.add_subplot(G[0, 0]), fig.add_subplot(G[0, 1]))
        axs[1].sharex(axs[0])
    else:
        axs = fig.axes

    # plot performance
    color = color or MATLAB_COLORS[0]
    axs[0].plot(episodes, rewards.T, linewidth=LINEWIDTHS[0], color=color)
    axs[0].plot(
        episodes, mean_reward, linewidth=LINEWIDTHS[1],
        color=color, label=label)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Cumulative cost')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # plot number of unsafe episodes
    color = color or MATLAB_COLORS[0]
    axs[1].plot(episodes, unsafes.T, linewidth=LINEWIDTHS[0], color=color)
    axs[1].plot(episodes, mean_unsafe,
                linewidth=LINEWIDTHS[1], color=color, label=label)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Constraint violation')
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def plot_learned_weights(
    agents: list[RecordLearningData],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    Nupdates = len(agents[0].update_gradient)
    weightnames = agents[0].weights_history.keys()
    Nweights = len(weightnames)
    updates = np.arange(Nupdates + 1) + 1

    # create figure and grid
    ncols = int(np.floor(np.sqrt(Nweights + 1)))
    nrows = ncols if ncols**2 >= Nweights else (ncols + 1)
    if fig is None:
        fig = plt.figure(constrained_layout=True)
        G = gridspec.GridSpec(nrows, ncols, figure=fig)
        axs = [fig.add_subplot(G[i, j])
               for i, j in product(range(nrows), range(ncols))]
        for i in range(1, len(axs)):
            axs[i].sharex(axs[0])
    else:
        axs = fig.axes

    # plot update gradient norm history
    color = color or MATLAB_COLORS[0]
    norms: np.ndarray = np.linalg.norm(
        np.stack([agent.update_gradient for agent in agents]), axis=-1)
    log10_mean_norm = 10**(np.log10(norms).mean(axis=0))
    axs[0].semilogy(updates[:-1], norms.T,
                    linewidth=LINEWIDTHS[0], color=color)
    axs[0].semilogy(updates[:-1], log10_mean_norm,
                    linewidth=LINEWIDTHS[1], color=color, label=label)
    axs[0].set_xlabel('Update')
    axs[0].set_ylabel('||p||')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # plot each weight's history
    for i, (ax, name) in enumerate(zip(axs[1:], weightnames), start=1):
        # get history and average it
        weights: np.ndarray = np.stack(
            [np.squeeze(agent.weights_history[name]) for agent in agents])
        lbl = f'Parameter \'{name}\''
        if weights.ndim > 2:
            weights = weights.mean(axis=-1)
            lbl += ' (mean)'
        mean_weight = weights.mean(axis=0)  # average over agents

        # plot
        ax.plot(updates, weights.T, linewidth=LINEWIDTHS[0], color=color)
        ax.plot(updates, mean_weight,
                linewidth=LINEWIDTHS[1], color=color, label=label)
        ax.set_xlabel('Update')
        ax.set_ylabel(lbl)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i in range(i + 1, len(axs)):
        axs[i].set_axis_off()
    return fig
