import argparse
import matplotlib.pyplot as plt
import util
from agents.wrappers import RecordLearningData
from itertools import cycle
from envs.wrappers import RecordData
from typing import Any
from util import plot


if __name__ == '__main__':
    util.set_np_mpl_defaults()

    # parse arguments
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument(
        'filenames', type=str, nargs='*',
        help='The pickle data to be visualized.')
    parser.add_argument(
        '-p', '--plots', type=int, nargs='*', help='Enables the i-th plot.')
    args = parser.parse_args()

    if len(args.filenames) == 0:
        args.filenames = [
            'R_20220919_193856.pkl',
        ]
    if args.plots is None:
        args.plots = range(2)

    figs, colors = [None, None], cycle(plot.MATLAB_COLORS)
    for filename, color in zip(args.filenames, colors):
        # load data
        results = util.load_results(filename)
        data: dict[str, Any] = results.pop('data')
        envs: list[RecordData] = data.get('envs')
        agents: list[RecordLearningData] = data.get('agents')

        # print summary
        print(filename, *(f'\n - {k}: {v}' for k, v in results.items()))

        # plot
        if 0 in args.plots and envs is not None:
            figs[0] = plot.plot_performance_and_unsafe_episodes(
                envs, fig=figs[0], color=color)
        if 1 in args.plots and agents is not None:
            figs[1] = plot.plot_learned_weights(
                agents, fig=figs[1], color=color)
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


# # shorten results
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
