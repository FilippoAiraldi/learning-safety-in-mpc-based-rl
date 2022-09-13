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
        args.filename = 'results/5_episodes_per_session.pkl'

    # load data
    data = util.load_results(args.filename)
    args = data['args']
    agent_config = data['agent_config']
    data = data['data']
    envs: list[RecordData] = [d['env'] for d in data]
    # eval_envs: list[RecordData] = [d['eval_env'] for d in data]
    agents: list[RecordLearningData] = [d['agent'] for d in data]

    # plot
    plot.plot_performance_and_unsafe_episodes(envs)
    plot.plot_learned_weights(agents)
    plt.show()
    # util.plot.plot_trajectory_3d(env, 0)
    # util.plot.plot_trajectory_in_time(env, 0)
