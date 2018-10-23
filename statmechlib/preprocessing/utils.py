import numpy as np

def normalize_histogram(hist, columns='all'):
    """
    Normalizes supplied histograms on specified columns.

    Parameters
    ----------
    hist: ndarray of floats
          unnormalized histogram
    columns: list of int
          columns to be normalized. Defaults to all columns

    Returns
    -------
    norm_hist: ndarray of floats
               Normalized histogram
    """


    norm_hist = np.empty_like(hist)

    if columns == 'all':
        columns = range(hist.shape[1])

    if col in columns:
        norm_hist[:,col] = hist[:,col]/np.sum(hist[:,col]) # normalize histogram

    return norm_hist


