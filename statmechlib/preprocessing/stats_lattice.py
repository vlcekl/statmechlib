from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy as np

def get_stats_latt(hrs, hrx):

    hrs = np.array(hrs).T
    hrs = hrs*float(hrs.shape[1])/np.sum(hrs)
    hrx = np.array(hrx)

    print('hrs shape', hrs.shape)
    print('hrx shape', hrx.shape)

    return hrs, hrx

