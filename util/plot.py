import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from agents.safety import constraint_violation as cv_
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


def trajectory_3d(env: RecordData, traj_num: int) -> Figure:
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


def trajectory_time(env: RecordData, traj_num: int) -> Figure:
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


def performance(
    envs: list[RecordData],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    '''
    Plots the performance in each environment and the average performance.
    '''
    if fig is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    else:
        ax = fig.axes[0]

    episodes = np.arange(len(envs[0].cum_rewards)) + 1
    rewards: np.ndarray = np.stack([env.cum_rewards for env in envs])
    mean_reward: np.ndarray = rewards.mean(axis=0)

    color = color or MATLAB_COLORS[0]
    ax.plot(episodes, rewards.T, linewidth=LINEWIDTHS[0], color=color)
    ax.plot(
        episodes, mean_reward, linewidth=LINEWIDTHS[1],
        color=color, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative cost')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return fig


def constraint_violation(
    envs: list[RecordData],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    '''
    Plots the constraint violations in each environment and the average 
    violation.
    '''
    x_bnd, u_bnd = envs[0].config.x_bounds, envs[0].config.u_bounds
    N = (np.isfinite(x_bnd).sum(axis=1) > 0).sum() + \
        (np.isfinite(u_bnd).sum(axis=1) > 0).sum()
    ncols = int(np.round(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    if fig is None:
        fig = plt.figure(constrained_layout=True)
        G = gridspec.GridSpec(nrows, ncols, figure=fig)
        axs = [fig.add_subplot(G[i, j])
               for i, j in product(range(nrows), range(ncols))]
        for i in range(1, len(axs)):
            axs[i].sharex(axs[0])
    else:
        axs = fig.axes

    max_ep_len = max(max(env.episode_lengths) for env in envs)
    observations, actions = [], []
    for env in envs:
        obs, act = [], []
        for ep in range(len(env.episode_lengths)):
            x = env.observations[ep][1:, :]
            x = np.pad(x, ((0, max_ep_len - x.shape[0]), (0, 0)),
                       constant_values=np.nan)
            u = env.actions[ep]
            u = np.pad(u, ((0, max_ep_len - u.shape[0]), (0, 0)),
                       constant_values=np.nan)
            obs.append(x)
            act.append(u)
        observations.append(obs)
        actions.append(act)
    observations, actions = [
        np.transpose(np.array(a), (3, 2, 1, 0))
        for a in (observations, actions)  # [nx(nu), max_ep_len, Nep, Nenv]
    ]
    episodes = np.arange(observations.shape[2]) + 1

    # reduce each trajectory's violations to scalar by picking max
    cv_obs, cv_act = (
        (np.nanmax(cv_lb, axis=1), np.nanmax(cv_ub, axis=1))
        for cv_lb, cv_ub in cv_((observations, x_bnd), (actions, u_bnd))
    )

    axs = iter(axs)
    for n, (cv_lb, cv_ub), bnd in [('x', cv_obs, x_bnd), ('u', cv_act, u_bnd)]:
        for i in range(bnd.shape[0]):
            if not np.isfinite(bnd[i]).any():
                continue
            ax = next(axs)
            ax.plot(
                episodes, cv_lb[i],
                linewidth=LINEWIDTHS[0], color=color, linestyle='--')
            ax.plot(
                episodes, cv_lb[i].mean(axis=-1),
                linewidth=LINEWIDTHS[1], color=color, linestyle='--',
                label=label)
            ax.plot(
                episodes, cv_ub[i],
                linewidth=LINEWIDTHS[0], color=color, linestyle='-')
            ax.plot(
                episodes, cv_ub[i].mean(axis=-1),
                linewidth=LINEWIDTHS[1], color=color, linestyle='-',
                label=label)

            ax.set_xlabel('Episode')
            ax.set_ylabel(f'Violation of ${n}_{i}$')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axs:
        ax.set_axis_off()
    return fig


def learned_weights(
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
    ncols = int(np.round(np.sqrt(Nweights + 1)))
    nrows = int(np.ceil((Nweights + 1) / ncols))
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
    axs = iter(axs)
    ax = next(axs)
    color = color or MATLAB_COLORS[0]
    norms: np.ndarray = np.linalg.norm(
        np.stack([agent.update_gradient for agent in agents]), axis=-1)
    log10_mean_norm = 10**(np.log10(norms).mean(axis=0))
    ax.semilogy(updates[:-1], norms.T,
                linewidth=LINEWIDTHS[0], color=color)
    ax.semilogy(updates[:-1], log10_mean_norm,
                linewidth=LINEWIDTHS[1], color=color, label=label)
    ax.set_xlabel('Update')
    ax.set_ylabel('||p||')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # plot each weight's history
    for ax, name in zip(axs, weightnames):
        # get history and average it
        weights: np.ndarray = np.stack(
            [np.squeeze(agent.weights_history[name]) for agent in agents])
        lbl = f'Parameter ${name}$'
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
    for ax in axs:
        ax.set_axis_off()
    return fig
