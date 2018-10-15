"""
Collection of simple potential functions used in different force fields
"""

# Cubic spline function for pair potentials and electronic density
f_spline3 = lambda r, aa, kk: sum([a*(rk - r)**3 for a, rk in zip(aa, kk) if r < rk and r > 0.01])

# Lennard-Jones potential in the 12-6 form
f_12_6 = lambda r, A, C: A/r**12 - C/r**6
