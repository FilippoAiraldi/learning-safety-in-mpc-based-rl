import casadi as cs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from cycler import cycler


def quad_form(A: cs.SX | cs.DM, x: cs.SX | cs.DM) -> cs.SX | cs.DM:
    '''Calculates quadratic form x^T A x.'''
    if A.is_vector():
        A = cs.diag(A)
    return cs.bilin(A, x, x)


def spy(H: cs.SX | cs.DM | np.ndarray, **original_kwargs):
    '''See Matplotlib.pyplot.spy.'''
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
    plt.spy(H, **original_kwargs)
    nz = np.count_nonzero(H)
    plt.xlabel(f'nz = {nz} ({nz / H.size * 100:.2f}%)')
    plt.show(block=False)


def set_np_mpl_defaults() -> None:
    '''Sets the default options for Numpy and Matplotlib.'''
    # numpy defaults
    np.set_printoptions(precision=3, suppress=False)

    # matplotlib defaults
    mpl.style.use('seaborn-darkgrid')
    mpl.rcParams['axes.prop_cycle'] = cycler(
        'color',
        ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#A2142F'])
    mpl.rcParams['lines.linewidth'] = 1
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.tick_params(which='both', direction='in')
    # ax.legend(frameon=False)


def save_results(filename: str = None, **data) -> str:
    '''
    Saves results to pickle.

    Parameters
    ----------
    filename : str, optional
        The name of the file to save to.
    **data : dict
        Any data to be saved to the pickle file.

    Returns
    -------
    filename : str
        The complete name of the file where the data was written to.
    '''
    if filename is None:
        filename = 'R'
    filename = f'{filename}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return filename


def load_results(filename: str) -> dict:
    '''
    Loads results from pickle.

    Parameters
    ----------
    filename : str, optional
        The name of the file to load.

    Returns
    -------
    data : dict
        The saved data in the shape of a dictionary.
    '''
    with open(filename, 'rb') as f:
        return pickle.load(f)
