from __future__ import print_function #, unicode_literals
from __future__ import absolute_import, division
try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import numpy as np
import copy

eos_params = {
    'W':{'l':0.274, 'r_wse':1.584, 'eta':5.69, 'dE':8.9}
}

def universal_eos(x, system='W'):
    """
    Universal equation of state for a given system.

    Parameters
    ----------
    x: float
       lattice expansion/compression parameter
    system: str
       system (element) id (default 'W')

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


def select_nodes(stats_input, p_index, m_index):
    """
    Select only configuration statistics from stats (spline nodes) that are given in index.
    Parameters
    ----------
    stats_input: dict
               statistics for all nodes
    p_index: list
             list of spline knots used for pair interactions
    m_index: list
             list of spline knots used for density function in manybody interactions
    """
    
    stats_select = copy.deepcopy(stats_input)
    
    for key, stats in stats_select.items():
        if type(stats) == dict and 'energy' in stats.keys():
            for i, conf in enumerate(stats['energy']):
                #new_conf = np.empty((3, sum(index)), dtype=float)
                #new_conf[0] = conf[0][index]
                #new_conf[1] = conf[1][index]
                #new_conf[2] = conf[2][index]
                new_conf =  [c[p_index] for c in conf[0:3]]
                new_conf.append(conf[3][m_index])
                stats['energy'][i] = new_conf
                
    stats_select['hyperparams']['pair'] = list(np.array(stats_select['hyperparams']['pair'])[p_index])
    stats_select['hyperparams']['edens'] = list(np.array(stats_select['hyperparams']['edens'])[m_index])

    return stats_select

def insert_zero_params(params_dict, stats):
    
    params_out = {'hyperparams':{'pair':[], 'edens':[]},
                  'params':{'pair':[], 'edens':[]}}

    for key in ['pair', 'edens']:
        for i, s_knot in enumerate(stats['hyperparams'][key]):
            for p_knot, p_val in zip(params_dict['hyperparams'][key], params_dict['params'][key]):
                if abs(s_knot - p_knot) < 1e-6:
                    params_out['hyperparams'][key].append(p_knot)
                    params_out['params'][key].append(p_val)
                    break
            else:
                params_out['hyperparams'][key].append(s_knot)
                params_out['params'][key].append(0.0)

    return params_out

def to_param_dict(params_list, hp):
    """
    Convert from params list to parameter dictionary suitable for saving to a pickle

    Parameters
    ----------
    params_list: list of floats
                List of parameters used as input and output for optimization.
    hp: list or dict
        If list, gives number of parameters of each kind (embed, pair, edens, lrcorr)
        If dict, assumes dictionary of hyperparameters and infers corresponding numbers of parameters

    Returns
    -------
    params_dict: dict
               Dictionary of parameters (+hyperparameters if available)
    """

    params_dict = {}

    if isinstance(hp, dict):
        params_dict.update({'hyperparams':hp})

        hp = [2] + [len(hp[k]) for k in ['pair','edens']]

        # check if there are parameters for long range corrections
        nlr = len(params_list) - sum(hp)
        if nlr < 0 or nlr > 1:
            raise ValueError("Number of parameters does not match number of spline knots.")

        hp += [nlr]

    assert sum(hp) == len(params_list), 'Number of parameters does not match'

    p_dict = {}
    p_dict['embed']  = params_list[0:hp[0]]
    p_dict['pair']   = params_list[hp[0]:sum(hp[0:2])]
    p_dict['edens']  = params_list[sum(hp[0:2]):sum(hp[0:3])]
    p_dict['lrcorr'] = params_list[sum(hp[0:3]):sum(hp)]    

    params_dict.update({'params':p_dict})
    
    return params_dict

def rescale_manybody_params(params_dict):
    """
    Rescale edens and embed parameters so that the last edens==1.0. Then delete the last edens.
    This will help eliminate the last edens parameter from optimization (because it is colinear)
    """
    
    scale = params_dict['edens'][-1]
    params_out = {}
    params_out['edens'] = [p/scale for p in params_dict['edens']][:-1]
    params_out['embed'] = [params_dict['embed'][1]*scale**0.5, params_dict['embed'][1]*scale**2]
    params_out['pair'] = params_dict['pair']

    return params_out

def to_param_list(params_dict):
    """
    Convert from parameters dictionary to list usable as optimization input
    Input params_dict is a sub-dictionary stored in params_list[i]['params']
    """
    
    params_list  = list(params_dict['embed'])
    params_list += list(params_dict['pair'])
    params_list += list(params_dict['edens'])
    
    return params_list

def find_index(select_list, full_list):
    knots = []
    for sel in select_list:
        for i, elem in enumerate(full_list):
            if abs(sel - elem) < 1e-9:
                knots.append(i)
                break
    
    assert len(knots) == len(select_list), "Knots and select_list lengths do not match"
    
    return knots
