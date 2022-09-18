import argparse
import matplotlib.pyplot as plt
import util
from agents.wrappers import RecordLearningData
from itertools import cycle
from envs.wrappers import RecordData
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
            'results/lstdq_1_baseline.pkl',
            'results/lstdq_1_baseline_longer.pkl'
        ]
    if len(args.plots) == 0:
        args.plots = range(2)

    figs, colors = [None, None], cycle(plot.MATLAB_COLORS)
    for filename, color in zip(args.filenames, colors):
        # load data
        data = util.load_results(filename)
        sim_args = data['args']
        agent_config = data['agent_config']
        data = data['data']
        envs: list[RecordData] = [d['env'] for d in data]
        agents: list[RecordLearningData] = [d['agent'] for d in data]

        # print summary
        print(filename, '\n - args:', sim_args,
              '\n - agent config:', agent_config, '\n')

        # plot
        if 0 in args.plots:
            figs[0] = plot.plot_performance_and_unsafe_episodes(
                envs, fig=figs[0], color=color)
        if 1 in args.plots and all(a is not None for a in agents):
            figs[1] = plot.plot_learned_weights(
                agents, fig=figs[1], color=color)

        # util.plot.plot_trajectory_3d(env, 0)
        # util.plot.plot_trajectory_in_time(env, 0)

    plt.show()

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
