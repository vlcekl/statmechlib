"""
Collection of lattice gas potential functions
"""

import numpy as np

def sd2(params, stats, targets):
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

    # apply bounds on parameters
    params = np.where(params < -2.0, -2.0, params)
    params = np.where(params >  2.0,  2.0, params)

    # nearest and next nearest interactions between unlike particles
    par = np.array([0.0, params[0], 0.0, 0.0, params[1], 0.0])
    #par = np.array([0.0, params[0], params[1], 0.0, params[2], 0.0])

    sd2 = 0.0
    for key in targets.keys():
        targ = targets[key]
        stat = stats[key]

        beta = 1.0/np.mean(stat['temp'])
        hru = stat['interaction_stats'] # interaction histogram statistics
        u_ref = stat['energy'] # reference system energy

        uuu = beta*(np.sum(hru*par, axis=1) - u_ref)
        #print(u_ref[10], np.sum(hru*par, axis=1)[10])
        uuu -= np.mean(uuu)

        # histogram reweighting factor
        eee = np.exp(-uuu)
        eee /= np.sum(eee)

        # statistical distance for surface configuration histograms
        phst = targ['config_stats']     # target histogram
        rhst = stat['config_stats']     # reference histograms for each configuration
        qhst = np.sum(rhst.T*eee, axis=1) # rescaled average reference histogram
        w = targ.get('weight', 1.0)     # weight of the target data set

        # Statistical distance contribution
        sd2 += w*np.arccos(np.sum(np.sqrt(qhst*phst)))**2
        #print(key, w, np.arccos(np.sum(np.sqrt(qhst*phst)))**2)

    #print('---', sd2)
    return sd2

