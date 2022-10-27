from itertools import chain, product
from typing import Iterable, Union
import casadi as cs
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from agents import \
    QuadRotorPKAgent, QuadRotorLSTDQAgent, QuadRotorGPSafeLSTDQAgent
from agents.wrappers import RecordLearningData
from util.math import \
    constraint_violation as cv_, jaggedstack as jstack, logmean


AGENTTYPE = Union[
    QuadRotorPKAgent,
    RecordLearningData[Union[QuadRotorLSTDQAgent, QuadRotorGPSafeLSTDQAgent]]
]
PAPERMODE = False
SMALL_ALPHA = {False: 0.5, True: 0.25}
SMALLER_LW_FACTOR = {False: 40, True: 50}
MATLAB_COLORS = [
    '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'
]


def _plot_population(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    y_mean: np.ndarray = None,
    use_median: bool = False,
    y_std: np.ndarray = None,
    color: str = None,
    linestyle: str = None,
    label: str = None,
    xlabel: str = None,
    ylabel: str = None,
    method: str = 'plot',
    legendloc: str = 'upper right',
    **kwargs
) -> None:
    if y_mean is None and y is not None:
        y_mean = (np.nanmedian if use_median else np.nanmean)(y, axis=0)
    lw = mpl.rcParams['lines.linewidth']
    lw_small = lw / SMALLER_LW_FACTOR[PAPERMODE]
    func = getattr(ax, method)
    if method == 'errorbar':
        y_std = y_std if y_std is not None else np.nanstd(y, axis=0)
        func(x, y_mean, yerr=y_std, color=color, lw=lw,
             linestyle=linestyle, label=label, errorevery=x.size // 10,
             capsize=5, **kwargs)
    elif method == 'fill_between':
        y_std = y_std if y_std is not None else np.nanstd(y, axis=0)
        func(x, y_mean - y_std, y_mean + y_std, alpha=SMALL_ALPHA[PAPERMODE], 
             color=color)
        ax.plot(x, y_mean, lw=lw, color=color, linestyle=linestyle,
                label=label, **kwargs)
    else:
        if y is not None:
            func(x, y.T, lw=lw_small, color=color, linestyle=linestyle,
                 alpha=SMALL_ALPHA[PAPERMODE], **kwargs)
        if y_mean is not None:
            func(x, y_mean, lw=lw, color=color, linestyle=linestyle,
                 label=label, **kwargs)

    if xlabel is not None and len(ax.get_xlabel()) == 0:
        ax.set_xlabel(xlabel)
    if ylabel is not None and len(ax.get_ylabel()) == 0:
        ax.set_ylabel(ylabel)
    if label is not None:
        ax.legend(loc=legendloc)
    if not isinstance(ax.xaxis.get_major_locator(), MaxNLocator):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _set_empty_axis_off(axs: Iterable[Axes]) -> None:
    (ax.set_axis_off() for ax in axs if len(ax.get_lines()) == 0)


def _save2tikz(*figs: Figure) -> None:
    '''
    Saves the figure to a tikz file. 
    See https://pypi.org/project/tikzplotlib/.
    '''
    import tikzplotlib
    for fig in figs:
        tikzplotlib.save(
            f'figure_{fig.number}.tex',
            figure=fig,
            extra_axis_parameters={r'tick scale binop=\times'}
        )


def spy(H: Union[cs.SX, cs.MX, cs.DM, np.ndarray], **spy_kwargs) -> Figure:
    '''See `matplotlib.pyplot.spy`.'''
    # try convert to numerical; if it fails, then use symbolic method from cs
    try:
        H = np.array(H)
    except Exception:
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            H.sparsity().spy()
        out = f.getvalue()
        H = np.array([list(line) for line in
                      out.replace('.', '0').replace('*', '1').splitlines()],
                     dtype=int)

    # plot looks nicer
    if 'markersize' not in spy_kwargs:
        spy_kwargs['markersize'] = 1

    # do plotting
    fig, ax = plt.subplots(1, 1)
    ax.spy(H, **spy_kwargs)
    nz = np.count_nonzero(H)
    ax.set_xlabel(f'nz = {nz} ({nz / H.size * 100:.2f}%)')
    return fig


def set_mpl_defaults(
        matlab_colors: bool = False, papermode: bool = False) -> None:
    '''Sets the default options for Matplotlib.'''
    global PAPERMODE
    PAPERMODE = papermode
    np.set_printoptions(precision=4)
    mpl.style.use('bmh')  # 'seaborn-darkgrid'
    mpl.rcParams['lines.solid_capstyle'] = 'round'
    if papermode:
        mpl.rcParams['lines.linewidth'] = 5
    else:
        mpl.rcParams['savefig.dpi'] = 600
        mpl.rcParams['lines.linewidth'] = 3
    if matlab_colors:
        mpl.rcParams['axes.prop_cycle'] = cycler('color', MATLAB_COLORS)


def performance(
    agents: list[AGENTTYPE],
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

    rewards: np.ndarray = jstack([a.env.cum_rewards for a in agents])
    episodes = np.arange(rewards.shape[1]) + 1

    _plot_population(
        ax, episodes, rewards, use_median=False,
        color=color, label=label, xlabel='Episode',
        ylabel='Cumulative cost')
    return fig


def constraint_violation(
    agents: list[AGENTTYPE],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    '''
    Plots the constraint violations in each environment and the average 
    violation.
    '''
    x_bnd, u_bnd = agents[0].env.config.x_bounds, agents[0].env.config.u_bounds
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

    # [nx(nu), max_ep_len, Nep, Nenv]
    observations = np.transpose(jstack(
        [jstack(a.env.observations) for a in agents]))
    actions = np.transpose(jstack([jstack(a.env.actions) for a in agents]))
    episodes = np.arange(observations.shape[2]) + 1

    # apply 2 reductions: first merge lb and ub in a single constraint (since
    # both cannot be active at the same time) (max axis=1); then reduce each
    # trajectory's violations to scalar by picking max violation (max axis=2)
    cv_obs, cv_act = (np.nanmax(cv, axis=(1, 2))
                      for cv in cv_((observations, x_bnd), (actions, u_bnd)))

    # proceed to plot
    axs = iter(axs)
    for n, cv, bnd in [('x', cv_obs, x_bnd), ('u', cv_act, u_bnd)]:
        for i in range(bnd.shape[0]):
            if not np.isfinite(bnd[i]).any():
                continue
            ax = next(axs)
            _plot_population(ax, episodes, cv[i].T, color=color, label=label,
                             xlabel='Episode', ylabel=f'${n}_{i}$ violation')
    _set_empty_axis_off(axs)
    return fig


def learned_weights(
    agents: list[AGENTTYPE],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    '''Plots the learning curves of the MPC parameters.'''
    if any(not isinstance(a, RecordLearningData) for a in agents):
        return

    weightnames = sorted(agents[0].weights_history.keys())
    Nweights = len(weightnames)

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
    norms = np.sqrt(np.square(np.ma.masked_invalid(
        jstack([agent.update_gradient for agent in agents]))).sum(axis=-1)
    )
    mean_norm = logmean(norms, axis=0)
    updates = np.arange(norms.shape[1] + 1)
    _plot_population(ax, updates[1:], norms, y_mean=mean_norm, color=color,
                     label=label, xlabel='Update', ylabel='||p||',
                     method='semilogy')

    # plot each weight's history
    for ax, name in zip(axs, weightnames):
        # get history and average it
        weights = jstack(
            [np.squeeze(agent.weights_history[name]) for agent in agents])
        lbl = f'Parameter ${name}$'
        if weights.ndim > 2:
            weights = np.nanmean(weights, axis=-1)
            lbl += ' (mean)'

        # plot
        _plot_population(ax, updates, weights, color=color, label=label,
                         xlabel='Update', ylabel=lbl)
    _set_empty_axis_off(axs)
    return fig


def safety(
    agents: list[AGENTTYPE],
    fig: Figure = None,
    color: str = None,
    label: str = None
) -> Figure:
    '''Plots safety related quantities for the simulation.'''
    if fig is None:
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
    else:
        axs = fig.axes

    axs = iter(axs)
    # [nx(nu), max_ep_len, Nep, Nenv]
    observations = np.transpose(jstack(
        [jstack(a.env.observations) for a in agents]))
    actions = np.transpose(jstack(
        [jstack(a.env.actions) for a in agents]))
    episodes = np.arange(observations.shape[2]) + 1

    # compute constraint violations and apply 2 reductions: first merge lb
    # and ub in a single constraint (since both cannot be active at the
    # same time) (max axis=1); then reduce each trajectory's violations to
    # scalar by picking max violation (max axis=2)
    x_bnd, u_bnd = agents[0].env.config.x_bounds, agents[0].env.config.u_bounds
    cv_obs, cv_act = (
        np.nanmax(cv, axis=(1, 2))
        for cv in cv_((observations, x_bnd), (actions, u_bnd))
    )

    # plot cumulative number of unsafe episodes
    cv_all = np.concatenate((cv_obs, cv_act), axis=0)
    cnt = np.nancumsum((np.nanmax(cv_all, axis=0) > 0.0), axis=0)
    _plot_population(next(axs), episodes, cnt.T, color=color, label=label,
                     xlabel='Episode', ylabel='Number of unsafe episodes',
                     legendloc='upper left')

    attr = 'backtracked_betas'
    if agents is not None and all(
            a is not None and hasattr(a, attr) for a in agents):
        betas = jstack([getattr(a, attr) for a in agents])
        updates = np.arange(betas.shape[1]) + 1

        ax = next(axs)
        ax.set_axis_on()
        _plot_population(ax, updates, betas, color=color, label=r'$\beta$',
                         xlabel='Update', ylabel='GP parameters')
    _set_empty_axis_off(axs)
    return fig


def paperplots(
    agents: dict[str, list[AGENTTYPE]],
    save2tikz: bool = False
) -> list[Figure]:
    '''
    Produces and saves to .tex the plots for the paper. For more details and 
    comments on the inner workings, see standard visualization functions above.
    '''
    lstdq = agents['lstdq']
    safe_lstdq = agents['safe-lstdq']
    safe_lstdq_prior = agents['safe-lstdq-prior']
    learning_agents = (lstdq, safe_lstdq, safe_lstdq_prior)
    pk = agents['pk']
    colors = [c['color'] for c in mpl.rcParams['axes.prop_cycle']]
    labels = ('LSTD Q', 'Safe LSTD Q', 'Safe LSTD Q (prior knowledge)')
    use_median = False

    def figure1() -> Figure:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        baseline = np.nanmean(
            list(chain.from_iterable(a.env.cum_rewards for a in pk)))
        performances = [jstack([a.env.cum_rewards for a in agents])
                        for agents in learning_agents]
        episodes = np.arange(performances[0].shape[1]) + 1
        L = 25
        for performance, clr, lbl in zip(performances, colors, labels):
            _plot_population(
                ax, episodes[:L], performance[:, :L], use_median=use_median,
                color=clr, label=lbl, method='fill_between',
                xlabel='Learning episode', ylabel=r'$J(\pi_\theta)$')
        ax.axhline(y=baseline, color='k', lw=1, ls='--')
        ax.set_xlim(episodes[0], episodes[L])
        ax.set_ylim(0, 30000)
        return fig

    def figure2() -> Figure:
        fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)

        altitude_violations = []
        unsafe_episodes = []
        for agents in learning_agents:
            observations = np.transpose(
                jstack([jstack(a.env.observations) for a in agents]))
            actions = np.transpose(
                jstack([jstack(a.env.actions) for a in agents]))
            x_bnd = agents[0].env.config.x_bounds
            u_bnd = agents[0].env.config.u_bounds
            cv_obs, cv_act = (
                np.nanmax(cv, axis=(1, 2))
                for cv in cv_((observations, x_bnd), (actions, u_bnd))
            )
            altitude_violations.append(cv_obs[2].T)
            max_cv = np.concatenate((cv_obs, cv_act), axis=0).max(axis=0)
            unsafe_episodes.append(np.nancumsum(max_cv > 0.0, axis=0).T)

        UE_avg = [eps[:, -1].mean() for eps in unsafe_episodes]
        percs = [(UE_avg[0] - ue) / UE_avg[0] * 100 for ue in UE_avg[1:]]
        print('Unsafe episodes average reduction =',
              ', '.join(f'{p:.2f}%' for p in percs))

        episodes = np.arange(unsafe_episodes[0].shape[1]) + 1
        for ue, av, clr, lbl in zip(
                unsafe_episodes, altitude_violations, colors, labels):
            _plot_population(
                axs[0], episodes, ue, use_median=use_median,
                color=clr, label=lbl, method='fill_between',
                ylabel=r'\# of unsafe episodes', legendloc='upper left')
            _plot_population(
                axs[1], episodes, av, use_median=use_median,
                color=clr, label=lbl, method='fill_between',
                xlabel='Learning episode', ylabel='Altitude constraint')
        axs[0].set_xlim(episodes[0], episodes[-1])
        axs[1].set_ylim(-1, 5)
        return fig

    def figure3() -> Figure:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        for agents, clr, lbl in zip(
                learning_agents[1:], colors[1:], labels[1:]):
            betas = jstack([a.backtracked_betas for a in agents])
            episodes = np.arange(betas.shape[1]) + 1
            _plot_population(
                ax, episodes, betas, use_median=use_median,
                color=clr, label=lbl, legendloc='lower right',
                method='fill_between',
                xlabel='Learning episode', ylabel=r'$\beta$')
        ax.set_xlim(episodes[0], episodes[-1])
        ax.set_ylim(0.43, 1.1)
        return fig

    figs = [fcn() for k, fcn in locals().items()
            if k.startswith('figure') and callable(fcn)]
    if save2tikz:
        _save2tikz(*figs)
    return figs
