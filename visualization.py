import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from util import io, plot


def visualization(args):
    '''Runs standard visualization.'''
    # prepare args and plot functions
    funcs = [
        plot.performance,
        plot.constraint_violation,
        plot.learned_weights,
        plot.safety
    ]
    figs = [None] * len(funcs)
    colors = mpl.rcParams['axes.prop_cycle']
    if args.plots is None:
        args.plots = range(len(funcs))
    args.plots = set(args.plots)

    # plot each data
    for n, (filename, color) in enumerate(zip(args.filenames, colors)):
        # load data
        results = io.load_results(filename)
        agents = results.pop('agents')

        # print summary
        print(f'p{n}) {filename}\n',
              *(f' -{k}: {v}\n' for k, v in results.items()))

        # plot
        for i, func in enumerate(funcs):
            if i in args.plots:
                figs[i] = func(
                    agents=agents,
                    fig=figs[i],
                    color=color['color'],
                    label=f'p{n}'
                )
    plt.show()


def paper() -> None:
    '''Produces and saves to .tex the plots for the paper.'''
    filenames = {
        'lstdq': 'sim/lstdq',
        'safe-lstdq': 'sim/safe-lstdq',
        'safe-lstdq-prior': 'sim/safe-lstdq-prior',
        'pk': 'sim/pk'
    }
    agents = {n: io.load_results(fn)['agents'] for n, fn in filenames.items()}
    plot.paperplots(agents, save2tikz=True)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualization script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    group = parser.add_argument_group('Visualization')
    group.add_argument('filenames', type=str, nargs='*',
                       help='Data to be visualized.')
    group.add_argument('-p', '--plots', type=int, nargs='*',
                       help='Enables the i-th plot.')
    group = parser.add_argument_group('Paper')
    group.add_argument('-pm', '--papermode', action='store_true',
                       help='Switches to paper plots (hardcoded).')
    args = parser.parse_args()

    plot.set_mpl_defaults(papermode=args.papermode)

    if args.papermode:
        paper()
    else:
        visualization(args)
