"""
Collection of EAM functions
"""

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
