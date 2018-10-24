import numpy as np
#import pandas as pd

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

def map_histograms(hist, mapfunc):
    """
    Performs histogram transformation from one domain to another.
    Mostly used for coarse graining based on symmetry

    Parameters
    ----------
    hist: ndarray of floats
          original histogram
    mapfunc: function or dict
          maps histogram bins (first column) from old to new

    Returns
    -------
    new_hist: ndarray of floats
    """

    new_hist = np.empty_like(hist)


    return new_hist

