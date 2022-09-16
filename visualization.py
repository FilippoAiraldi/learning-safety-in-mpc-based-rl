import argparse
import matplotlib.pyplot as plt
import util
from agents.wrappers import RecordLearningData
from envs.wrappers import RecordData
from util import plot


if __name__ == '__main__':
    util.set_np_mpl_defaults()

    # parse arguments
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('-fn', '--filename', type=str, default=None,
                        help='The pickle data to be visualized')
    args = parser.parse_args()
    if args.filename is None:
        args.filename = 'results/5ep.pkl'

    # load data
    data = util.load_results(args.filename)
    args = data['args']
    agent_config = data['agent_config']
    data = data['data']
    envs: list[RecordData] = [d['env'] for d in data]
    agents: list[RecordLearningData] = [d['agent'] for d in data]

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
    plot.plot_performance_and_unsafe_episodes(envs)
    plot.plot_learned_weights(agents)
    plt.show()
    # util.plot.plot_trajectory_3d(env, 0)
    # util.plot.plot_trajectory_in_time(env, 0)
