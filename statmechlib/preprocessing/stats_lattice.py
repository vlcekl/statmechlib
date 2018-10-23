import numpy as np

def sd_hist(p, q, grs, grp, hrs, hrp, hru):
    """Statistical distance between histograms (surface, profile)"""

    # apply bounds on parametes
    p = np.where(p < -1.0, -1.0, p)
    p = np.where(p >  1.0,  1.0, p)

    # nearest and next nearest interactions between unlike particles
    pp = np.array([0.0, p[0], 0.0, 0.0, p[1], 0.0])
    qq = np.array([0.0, q[0], 0.0, 0.0, q[1], 0.0])

    # energy diference: bulk(1,2), surface(1,2), surface(1,1)
    uuu = beta*np.sum(hru*(pp - qq), axis=1)
    uave = np.sum(uuu)*fn
    uuu -= uave
    eee = np.exp(-uuu)
    fx = 1/np.sum(eee)

    # statistical distance for surface configuration histogram
    dloss  = np.arccos(np.sum(np.sqrt(np.sum(hrs*eee, axis=1)*fx*grs[:])))**2

    fx = -np.log(fn/fx)
    eee = np.exp(0.5*(fx - uuu))
    db = -2.0*np.log(np.sum(eee)*fn)
    ge = (fx + uave)/beta

    return dloss


def get_stats_latt(hrs, hrx):

    hrs = np.array(hrs).T
    hrs = hrs*float(hrs.shape[1])/np.sum(hrs)
    hrx = np.array(hrx)

    print('hrs shape', hrs.shape)
    print('hrx shape', hrx.shape)

    return hrs, hrx

