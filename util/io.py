import cloudpickle


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
