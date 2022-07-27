import argparse
import util
from envs.wrappers import RecordData
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('-fn', '--filename', type=str, required=True, 
                        help='The pickle data to be visualized')
    args = parser.parse_args()

    # load data
    data = util.load_results(args.filename)
    env: RecordData = data['env']

    # plot
    util.plot.plot_trajectory_3d(env, 0)
    util.plot.plot_trajectory_in_time(env, 0)
    plt.show()

