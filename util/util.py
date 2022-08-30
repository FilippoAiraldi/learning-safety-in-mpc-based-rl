import casadi as cs
import cloudpickle
import contextlib
import gym
import joblib
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from datetime import datetime
from itertools import combinations
from scipy.special import comb
from tqdm import tqdm
from typing import Any, Iterable, Union, Type


def cs_prod(x: Union[cs.SX, cs.DM], axis: int = 0) -> Union[cs.SX, cs.DM]:
    '''I couldn't find an equivalent of np.prod in CasADi...'''
    if x.is_vector():
        return cs.det(cs.diag(x))
    sum_ = cs.sum1 if axis == 0 else cs.sum2
    return cs.exp(sum_(cs.log(x))) if axis == 0 else cs.exp(sum_(cs.log(x)))


def cs_sigmoid(x: Union[cs.SX, cs.DM, cs.MX]) -> Union[cs.SX, cs.DM, cs.MX]:
    '''Computes sigmoid.'''
    e = cs.exp(x)
    return e / (e + 1)


def nchoosek(
    n: Union[int, Iterable[Any]], k: int) -> Union[int, Iterable[tuple[Any]]]:
    '''Returns the binomial coefficient, i.e.,  the number of combinations of n
    items taken k at a time. If n is an iterable, then it returns an iterable 
    containing all possible combinations of the elements of n taken k at a 
    time.'''
    return comb(n, k, exact=True) if isinstance(n, int) else combinations(n, k)


def monomial_powers(d: int, k: int) -> np.ndarray:
    '''Computes the powers of degree k contained in a d-dimensional 
    monomial.'''
    # Thanks to: https://stackoverflow.com/questions/40513896/compute-all-d-dimensional-monomials-of-degree-less-than-k
    m = nchoosek(k + d - 1, d - 1)
    dividers = np.column_stack((
        np.zeros((m, 1), dtype=int),
        np.row_stack(list(nchoosek(np.arange(1, k + d), d - 1))),
        np.full((m, 1), k + d, dtype=int)
    ))
    return np.flipud(np.diff(dividers, axis=1) - 1)


def quad_form(
    A: Union[cs.SX, cs.DM], x: Union[cs.SX, cs.DM]) -> Union[cs.SX, cs.DM]:
    '''Calculates quadratic form x^T A x.'''
    if A.is_vector():
        A = cs.diag(A)
    return cs.bilin(A, x, x)


def spy(H: Union[cs.SX, cs.DM, np.ndarray], **original_kwargs) -> None:
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

    # plot looks nicer
    if 'markersize' not in original_kwargs:
        original_kwargs['markersize'] = 1

    # do plotting
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
        ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE',
         '#A2142F'])
    mpl.rcParams['lines.linewidth'] = 1
    # mpl.rcParams['font.family'] = 'serif'
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['pgf.rcfonts'] = False


def get_run_name(prefix: str = None) -> str:
    '''Gets the name of the run.'''
    if prefix is None:
        prefix = 'R'
    return f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


def save_results(filename: str, **data) -> str:
    '''
    Saves results to pickle.

    Parameters
    ----------
    filename : str
        The name of the file to save to.
    **data : dict
        Any data to be saved to the pickle file.

    Returns
    -------
    filename : str
        The complete name of the file where the data was written to.
    '''
    if not filename.endswith('.pkl'):
        filename = f'{filename}.pkl'
    with open(filename, 'wb') as f:
        cloudpickle.dump(data, f)
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
        data = cloudpickle.load(f)
    if isinstance(data, dict) and len(data.keys()) == 1:
        data = data[next(iter(data.keys()))]
    return data


def create_logger(run_name: str, to_file: bool = True) -> logging.Logger:
    '''Returns a logger that logs to both output and to a file.'''
    # create logger
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler
    if to_file:
        fh = logging.FileHandler(f'{run_name}.txt')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    '''Context manager to patch joblib to report into tqdm progress bar
    given as argument.'''
    # thanks to: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def is_env_wrapped(env: gym.Env, wrapper_type: Type[gym.Wrapper]) -> bool:
    '''Checks if the environment is wrapped with the specified type.'''
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper_type):
            return True
        env = env.env
    return False
