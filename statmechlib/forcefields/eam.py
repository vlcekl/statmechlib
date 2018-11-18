"""
Collection of EAM functions
"""

import numpy as np
from .potentials import f_spline3

# Functional form for the embedding potential and its scaled form
f_embed = lambda d, a: a[0]*d**0.5 + a[1]*d + a[2]*d**2
f_embed_s = lambda d, a: f_embed(d/S, a) - C*d/S 

# Density function and its scaled form
f_dens = lambda x, dens_a, dens_r: f_spline3(x, dens_a, dens_r)
f_dens_s = lambda x, dens_a, dens_r: f_spline3(x, dens_a, dens_r)*S

def u_equi(r, pair_a, pair_r, dens_a, dens_r):
    """
    Equilibrium (normal, adjustable) part of the pair potential
    based on cubic splines.
    """

    # cubic spline pair potential
    u = f_spline3(x, pair_a, pair_r)

    # gauge transformation into regular form
    u += 2*C*f_spline3(x, dens_a, dens_r)

    return u

# Define the core parts of the potential (kept constant)
def u_core(r, za=74, zb=74):
    """
    Repulsive potential of the atomic cores. Default atomic numbers for W

    Parameters
    ----------
    r: float
       atom pair distance
    za, zb: floats
       atomic numbers of the two atoms

    Returns
    ------
    u: float
       pair energy at pair distance r
    """

    qe_sq = 14.3992 # squared electron charge  
    rs = 0.4683766/(za**(2/3) + zb**(2/3))**0.5
    x = r/rs

    u = 0.0
    if r > 0.0:
        u += 0.1818*np.exp(-3.2*x)
        u += 0.5099*np.exp(-0.9423*x)
        u += 0.2802*np.exp(-0.4029*x)
        u += 0.02817*np.exp(-0.2016*x)
        u *= za*zb*qe_sq/r

    return u

def f_pair(r, param_a, param_r, za=78, zb=78, ri=1.0, ro=2.0):
    """
    Overall EAM pair potential combining inner, transition, and outer parts.
    The inner part is fixed, while the outer part is based on supplied spline
    function (cubic by default). Transition part ensures smooth change from
    inner to outer.

    Parameters
    ----------
    r: float
       pair distance
    param_a, param_r: lists of floats
                      Parameters of the cubic spline for 
    za, zb: floats
            atomic numbers of the two atoms (default to W-W)
    ri, ro: floats
            inner and outer boundary of the transition region

    Returns
    -------
    u: float
       value of the pair potential at pair distance r
    """

    if r < ri:
        u = u_core(r, za, zb)

    elif r < ro:
        x = (ro + ri - 2*r)/(ro - ri)
        eta = 3/16*x**5 - 5/8*x**3 + 15/16*x + 1/2

        unucl = u_core(r, za, zb)
        uequi = u_equi(r, pair_a, pair_r, dens_a, dens_r)

        u = uequi + eta*(unucl - uequi)

    else:
        u = u_equi(r, pair_a, pair_r, dens_a, dens_r)

    return u

def utot_EAM(params, ustats, hparams=[-1]):
    """
    Calculates configurational energy from EAM sufficient statistics and model parameters

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    ustats : list of lists and floats
             Sufficient statistics for a trajectory of configurations
    hparams: list of ints
             hyperparameters - distance cutoff of the density function

    Returns
    -------
    u_total: float
             total configurational energy (sum of pair and manybody interactions) for trajectory of configurations
    """

    n_sample = len(ustats)

    if not hparams:
        hp = -1
    else:
        hp = hparams[0]

    # pair interactions from array of spline coefficeints and corresponding statistic
    u_pair = np.array([sum([a*s for a, s in zip(params[2:], ustats[i][2,:])]) for i in range(n_sample)])

    # manybody interactions from embedding function parameters and corresponding statistics
    u_many = np.array([params[0]*ustats[i][0, hp] + params[1]*ustats[i][1, hp] for i in range(n_sample)])

    u_total = 0.5*u_pair + u_many
    #print(u_pair, u_many, u_total)

    return u_total


def ftot_EAM(params, fstats):
    """
    Calculates configurational energy from EAM sufficient statistics and model parameters

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    fstats : list of lists and floats
             Sufficient statistics

    Returns
    -------
    f_total: float
             total configurational energy (sum of pair and manybody interactions)
    """

    # number of samples and atoms
    n_sample = len(fstats)
    
    
    # cycle over samples
    f_total = []
    for i in range(n_sample):

        # pair interactions from array of spline coefficeints and corresponding statistic
        f_pair = sum([p*s for p, s in zip(params[2:], fstats[i][2,:])]) 

        # manybody interactions from embedding function parameters and corresponding statistics
        f_many = params[0]*fstats[i][0,-1] + params[1]*fstats[i][1, -1]
        
        n_atom = fstats[i][0,0].shape[0]
        # Create a 6N + 1 array of 0, f, and -f
        fx = np.zeros((6*n_atom + 1), dtype=float)
        fx[1:3*n_atom+1] = 0.5*f_pair.flatten() + f_many.flatten()
        fx[3*n_atom+1:] = -fx[1:3*n_atom+1]
        #print('natom', n_atom, type(f_pair), type(f_many), f_pair.shape, f_many.shape)
 
        f_total.append(fx)
        
    return np.array(f_total)


def sd2_loss(params, stats, targets, utot_func, ftot_func=None, hparams=None, dl=0.05, verbose=0):
    """
    Calculates squared statistical distance loss function for configurational energies and forces.

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    stats  : list of lists and floats
             Sufficient statistics
    targets: list of lists and floats
             target energies and forces
    utot_func: function
               takes parameters and statistics and returns configurational energies
    ftot_func: function
               takes parameters and statistics and returns atomic forces
    dl: float
        coordinate perturbation magnitude for energy calculation from forces: du = f*dl

    Returns
    -------
    sd2, sd2f: float
               Squared statistical distances between model and target (energy and force-based)
    """

    # apply bounds on parametes
    #p = np.where(p < -1.0, -1.0, p)
    #p = np.where(p >  1.0,  1.0, p)

    # cycle over target system trajectories and statistics to determine SD
    sd2 = sd2f = 0.0
    for targ, stat in zip(targets, stats):

        beta = np.mean(targ['beta']) # system inverse temperature
        u_targ = np.array(targ['energy']) # target energies
        u_stat = stat['energy'] # energy statistics
        n_sample = len(u_targ)
        w = targ.get('weight', 1.0)

        # energy diference array for a given target trajectory
        uuu = beta*(utot_func(params, u_stat, hparams) - u_targ) # array(n_sample)
        uuu -= np.mean(uuu)
        eee = np.exp(-uuu)
        #print('uuu', uuu)
        #print('utarg', u_targ)
        #print('ustat', utot_func(params, u_stat))
        #print('eee', eee)
        
        #print('sd2', utot_EAM(params, u_stat)[0], u_targ[0])

        # are we using forces?
        if (not ftot_func) and ('forcesx' not in targ):

            # energy-based free energy difference and statistical distance
            ge = -np.log(np.mean(eee))   # free energy difference (shifted)
            cb = np.mean(np.exp(-0.5*(uuu - ge))) # Bhattacharyya coefficient
            sd2 += w*np.arccos(cb)**2              # statistical distance

        else:

            betad = beta*dl  # beta * dl
            f_targ = targ['forces'] # target forces (n_sample, 1+6N) (0, 3Nf, -3Nf)
            f_stat = stat['forces'] # force statistics (n_sample, npars, 3N)
            #print('ftarg', f_targ[-1].shape)
            #print('fstat', f_stat[-1].shape)

            eeh = np.exp(-0.5*uuu)
            fff = ftot_func(params, f_stat) # n_sample *(6N + 1) force contributions
            #print('s', fff[-1][:])
            #print('t', f_targ[-1][:])
            #print('ee', np.mean(np.exp(betad*fff[-1])), eee[-1]*np.mean(np.exp(betad*fff[-1]))) 
            #print('eall', eee[:])

            # target and model force terms
            fpave = np.mean([np.mean(np.exp(betad*f_targ[i])) for i in range(n_sample)])
            fqave = np.mean([eee[i]*np.mean(np.exp(betad*fff[i])) for i in range(n_sample)])
            fhave = np.mean([eeh[i]*np.mean(np.exp(0.5*betad*(fff[i]+f_targ[i]))) for i in range(n_sample)])
            
            # force-based free energy difference and statistical distance
            gef = -np.log(fqave/fpave)
            cb = fhave/(fqave*fpave)**0.5
            if cb > 1: cb = 1
            sd2f += w*np.arccos(cb)**2

            #print('fff', fpave, fqave, fhave, gef, cb, betad)
    
    if verbose > 0:
        print('loss', sd2+sd2f, sd2, sd2f)
    
    return sd2 + sd2f


def udif_print(params, stats, targets, utot_func, hparams=None):
    """
    Calculates squared statistical distance loss function for configurational energies and forces.

    Parameters
    ----------
    params : list of lists and floats
             EAM interaction parameters (spline coefficients array and embedding function parameters)
    stats  : list of lists and floats
             Sufficient statistics
    targets: list of lists and floats
             target energies and forces
    utot_func: function
               takes parameters and statistics and returns configurational energies

    Returns
    -------
    opti_out, targ_out: lists of floats
            model and target configurational energies
    """
    
    opti_out = []
    targ_out = []
    # cycle over target system trajectories and statistics to determine SD
    for targ, stat in zip(targets, stats):

        u_targ = np.array(targ['energy']) # target energies
        u_stat = stat['energy'] # energy statistics

        opti_out.append(list(utot_func(params, u_stat, hparams)))
        targ_out.append(list(u_targ))
    
    return opti_out, targ_out
