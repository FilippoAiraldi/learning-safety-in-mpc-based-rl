import re
import unicodedata
from datetime import datetime
from typing import Any
import pickle


def is_pickleable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception as ex:
        return False


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
    if not filename.endswith('.pkl'):
        filename = f'{filename}.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and len(data.keys()) == 1:
        data = data[next(iter(data.keys()))]
    return data


def slugify(value: str, allow_unicode: bool = False) -> str:
    '''
    Converts a string to a valid filename. Taken from 
    https://github.com/django/django/blob/master/django/utils/text.py. Converts
    to ASCII if `allow_unicode=False.`; converts spaces or repeated dashes to 
    single dashes; removes characters that aren't alphanumerics, underscores, 
    or hyphens; converts to lowercase; strips leading and trailing whitespace, 
    dashes, and underscores.
    '''
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize(
            'NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def get_runname(candidate: str = None) -> str:
    '''
    Gets the name of the run from an optional candidates that is a valid 
    filename.
    '''
    if candidate is None or not candidate:
        return f'R_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    return slugify(candidate)
