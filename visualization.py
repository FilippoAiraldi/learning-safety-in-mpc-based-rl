import argparse
import matplotlib.pyplot as plt
import util
from agents.wrappers import RecordLearningData
from envs.wrappers import RecordData
from util import plot


if __name__ == '__main__':
    util.set_np_mpl_defaults()

    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument(
        '-fn', '--filenames', type=str, default=None, nargs='+',
        help='The pickle data to be visualized.')
    parser.add_argument(
        '-dp', '--disable_plots', type=int, default=None, nargs='+',
        help='Disables the i-th plot.')
    args = parser.parse_args()

    if args.filenames is None:
        args.filenames = ('results/lstdq_5ep.pkl',)
    if args.disable_plots is None:
        args.disable_plots = ()

    for filename in args.filenames:
        # load data
        data = util.load_results(filename)
        sim_args = data['args']
        agent_config = data['agent_config']
        data = data['data']
        envs: list[RecordData] = [d['env'] for d in data]
        agents: list[RecordLearningData] = [d['agent'] for d in data]

        print(filename, '\n - args:', sim_args,
              '\n - agent config:', agent_config, '\n')

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

        # plot
        if 0 not in args.disable_plots:
            plot.plot_performance_and_unsafe_episodes(envs).suptitle(filename)
        if 1 not in args.disable_plots:
            plot.plot_learned_weights(agents).suptitle(filename)
        # util.plot.plot_trajectory_3d(env, 0)
        # util.plot.plot_trajectory_in_time(env, 0)

    plt.show()
