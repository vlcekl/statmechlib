import numpy as np
from itertools import combinations_with_replacement
from collections import defaultdict
from .pair_dist import pair_dist, pair_dist_cutoff

def get_stats_Mie(xyz, box, atom_type=None, ms=[12,6], rcut=None):
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
        rcut = 0.5*min(box[0,0], box[1,1], box[2,2])

    # get pair distances (absolute and Cartesian components)
    rr, rx = pair_dist_cutoff(xyz, box, rcut)

    # if needed, set the atom number types
    if type(atom_type) == None:
        atom_type1 = np.ones((rr.shape[0]), dtype=int)
        atom_type2 = np.ones((rr.shape[1]), dtype=int)
    else:
        atom_type1 = atom_type
        # how many times the original box was replicated?
        nreplicas = rr.shape[1]//rr.shape[0] 
        atom_type2 = np.concatenate([atom_type]*nreplicas)

    # particle type mesh grids
    mti, mtj = np.meshgrid(atom_type1, atom_type2, indexing='ij')
    ii, jj = np.meshgrid(range(rr.shape[0]), range(rr.shape[1]), indexing='ij')

    assert mti.shape == rr.shape, f"stats_mie: Shapes of mti {mti.shape} and rr {rr.shape} are incompatible"
    assert ii.shape == rr.shape, f"stats_mie: Shapes of ii {ii.shape} and rr {rr.shape} are incompatible"

    # Create 2D matrix of inverse distances (eliminate rr=0 and distances beyond cutoff)
    with np.errstate(divide='ignore'):
        rinv = np.where(np.logical_and(rr > 0.01, rr <= rcut), 1.0/rr, 0.0)

    # Create lists of 2D matrices with 1/r^m and 1/r^(m+2) values for energy
    # ane force calculations, respectively
    rpow_u = [rinv**m for m in ms]
    rpow_f = [rinv**(m+2) for m in ms]

    print('rcut:', rcut, mti.shape, mtj.shape, rpow_u[0].shape)

    # cycle over all combinations of pairs of particle types
    u_stats = defaultdict(list)
    f_stats = defaultdict(list)
    for ti, tj in combinations_with_replacement(set(atom_type1), 2):
        for m, rmat_u, rmat_f in zip(ms, rpow_u, rpow_f):
            rmat_u = np.where(jj > ii, rmat_u, 0.0)
            u_stats[(ti, tj)].append(np.sum(np.where(np.logical_and(mti==ti, mtj==tj), rmat_u, 0.0)))
            f_stats[(ti, tj)].append(np.sum(np.where(np.logical_and(mti==ti, mtj==tj), rmat_f, 0.0), axis=1))

    return dict(u_stats), dict(f_stats)

