import numpy as np
from itertools import combinations_with_replacement
from collections import defaultdict

def get_stats_Mie(rr, rx, ms, atom_type=None, rcut=None):
    """
    Takes atom pair distances and calculates sufficeint statistics needed
    for the parameterization 1/r^m type potentials (Lennard-Jones, Mie).
 
    Parameters
    ----------
    rr : 2D numpy array of floats
         set of pair distances
    rx : numpy array
         set of pair distance coordinates
    ms : python list of integers 
         exponents of 1/r^m
    atom_type: 1D numpy array of integers (particle types)


    Returns
    -------
    ar, a1, a2 : numpy arrays (len(ms))
                 atom energy-related statistics
                 el_density**0.5, el_density, el_density**2
    br, b1, b2 : numpy arrays (len(ms), natoms, 3 coordinates)
                 atom force-related statistics (gradients of energy)
                 grad(el_density**0.5), grad(el_density), grad(el_density**2)
    """

    # set rcut to max if None
    if rcut == None:
        rcut = rr.max()

    # Create 2D matrix of inverse distances (eliminate rr=0 and distances beyond cutoff)
    rinv = np.where(rr > 0.01 and rr <= rcut, 1/rr, 0.0)

    # Create a list of 2D matrices with 1/r^m values
    rpow = [rinv**m for m in ms]

    # if needed, set the atom number types
    if atom_type == None:
        atom_type = np.ones((rr.shape[0]), dtype=int)

    # particle type mesh grids
    mti, mtj = np.meshgrid(atom_type, atom_type)

    # cycle over all combinations of pairs of particle types
    u_stats = defaultdict(list)
    for ti, tj in combinations_with_replacement(set(atom_type)):
        for rpmat in rpow:
            u_stats[(ti, tj)].append(np.sum(np.where(mti==ti and mtj==tj, rpmat, 0.0)))
            #f_stats[(ti, tj)].append(np.sum(np.where(mti==ti and mtj==tj, rpmat, 0.0)))

    # 
    f_stats = defaultdict(list)
    for ti, tj in combinations_with_replacement(set(atom_type)):
        for rpmat in rpow:
            pass
            #f_stats[(ti, tj)].append(np.sum(np.where(mti==ti and mtj==tj, rpmat, 0.0)))

    return a1, ar, a2, b1, br, b2
