import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from agents.wrappers import RecordLearningData
from envs.wrappers import RecordData
from typing import Any
from util import io, plot


if __name__ == '__main__':
    plot.set_mpl_defaults()
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # parse arguments
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument(
        'filenames', type=str, nargs='*', help='Data to be visualized.')
    parser.add_argument(
        '-p', '--plots', type=int, nargs='*', help='Enables the i-th plot.')
    args = parser.parse_args()

    # prepare args and plot functions
    if len(args.filenames) == 0:
        args.filenames = [
            ('results/lstdq_baseline', 'LSTD Q'),
            ('results/gp_safe_lstdq', 'GPsafe LSTD Q'),
            # ('results/pk_baseline', 'PK'),
        ]
    funcs = [
        # plot.performance,
        # plot.constraint_violation,
        # plot.learned_weights,
        plot.safety
    ]
    figs = [None] * len(funcs)
    colors = mpl.rcParams['axes.prop_cycle']
    if args.plots is None:
        args.plots = range(len(funcs))

    # plot each data
    for n, (filename, color) in enumerate(zip(args.filenames, colors)):
        # if label is not passed, set it to None
        filename, label = \
            filename if isinstance(filename, tuple) else (filename, f'p{n}')

        # load data
        results = io.load_results(filename)
        data: dict[str, Any] = results.pop('data')
        envs: list[RecordData] = data.get('envs')
        agents: list[RecordLearningData] = data.get('agents')

        # print summary
        print(f'{n}) {filename}\n',
              *(f' -{k}: {v}\n' for k, v in results.items()))

        # plot
        for i, func in enumerate(funcs):
            if i in args.plots:
                figs[i] = func(
                    envs=envs,
                    agents=agents,
                    fig=figs[i],
                    color=color['color'],
                    label=label
                )
    plt.show()


# from scipy.io import loadmat
# axs = figs[0].axes
# data = loadmat('results/lstdq_1_baseline_old.mat')
# rewards, unsafes = data['rewards'], data['unsafes']
# color = next(colors)
# eps = list(range(1, 101))
# axs[0].plot(eps, rewards.T, linewidth=0.05, color=color)
# axs[0].plot(eps, rewards.mean(axis=0), linewidth=1.5,
#             color=color, label='LSTD Q (old)')
# axs[1].plot(eps, unsafes.T, linewidth=0.05, color=color)
# axs[1].plot(eps, unsafes.mean(axis=0), linewidth=1.5,
#             color=color, label='LSTD Q (old)')
# axs[0].legend()
# axs[1].legend()
# axs[0].set_ylim(top=2500)
# axs[1].set_ylim(top=10)


# # shorten plot
# target_epochs = 13
# ep_per_epoch = args.train_episodes
# for datum in data:
#     for name in ['observations', 'actions', 'rewards', 'cum_rewards',
#                  'episode_lengths', 'exec_times']:
#         o = list(getattr(datum['env'], name))
#         setattr(datum['env'], name, o[:target_epochs * ep_per_epoch])
#     datum['agent'].update_gradient = \
#         datum['agent'].update_gradient[:target_epochs]
#     datum['agent'].update_gradient_norm = \
#         datum['agent'].update_gradient_norm[:target_epochs]
#     datum['agent'].update_gradient_norm
#     for k, v in datum['agent'].weights_history.items():
#         datum['agent'].weights_history[k] = v[:target_epochs + 1]
