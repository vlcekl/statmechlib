"""
Collection of lattice gas potential functions
"""

import numpy as np

def sd_hist(params, stats, targets):
    """Statistical distance between histograms collected from different
    sources

    Parameters
    ----------
    params : list of lists and floats
             lattice model interaction parameters
    stats  : list of lists and floats
    targets: list of lists and floats
             target histograms

    Returns
    -------
    sd2: float
         squared statistical distance between model and target histograms
    """

    # apply bounds on parametes
    params = np.where(params < -1.0, -1.0, params)
    params = np.where(params >  1.0,  1.0, params)

    # nearest and next nearest interactions between unlike particles
    pp = np.array([0.0, params[0], 0.0, 0.0, params[1], 0.0])

    sd2 = 0.0
    for targ, stat in zip(targets, stats):

        beta = 1.0/np.mean(stat['temp'])
        hru = stat['interaction_stats'] # interaction histogram statistics
        u_ref = stat['energy'] # reference system energy

        uuu = beta*np.sum(hru*pp - u_ref), axis=1)
        uuu -= np.mean(uuu)

        eee = np.exp(-uuu)
        fx = 1/np.sum(eee)

        # statistical distance for surface configuration histograms
        hrs = stat['config_stats']
        grs = targ['config_stats']
        w = targ.get('weight', 1.0)
        sd2 += w*np.arccos(np.sum(np.sqrt(np.sum(hrs*eee, axis=1)*fx*grs[:])))**2

    return sd2

