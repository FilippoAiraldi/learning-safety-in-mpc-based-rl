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
            ('results/lstdq_no_w_safe_50agents', '50a'),
            ('results/lstdq_no_w_safe_200agents', '200a'),
            # ('results/pk_backoff_010backoff', 'PK'),
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
