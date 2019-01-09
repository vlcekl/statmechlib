from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy as np

eos_params = {
    'W':{'l':0.274, 'r_wse':1.584, 'eta':5.69, 'dE':8.9}
}

def universal_eos(x, system):
    """
    Universal equation of state for a given system.

    Parameters
    ----------
    x: float
       lattice expansion/compression parameter
    system: str
       system (element) id

    Returns
    -------
    end: float
         Energy of the crystal lattice for a given x
    """

    syst = eos_params[system]
    l = syst['l']
    r_wse = syst['r_wse']
    eta = syst['eta']
    dE = syst['dE']

    a = (x - 1.0)*r_wse/l
    ene = np.exp(-a)
    ene *= -1.0 - a - 0.05*a**3
    ene *= dE

    return ene


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

