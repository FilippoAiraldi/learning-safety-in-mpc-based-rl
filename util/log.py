import contextlib
import joblib
import logging
from datetime import datetime
from tqdm import tqdm


def get_run_name(prefix: str = None) -> str:
    '''Gets the name of the run.'''
    if prefix is None:
        prefix = 'R'
    return f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


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
