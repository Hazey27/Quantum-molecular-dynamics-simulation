# Hazel Dellario
# 2024/03/30
# Molecular dynamics simulation of H2 using QP, QUBO, and QAOA

import numpy as np
from matplotlib import pyplot as plt
from docplex.mp.model import Model
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver

# Constants
a0_per_m = 1.889726124565062e+10
me_per_amu = 1.822888484770040e+3
eh_per_cm_1 = 4.556335256391438e-6
eh_per_ev = 3.6749308136649e-2


# Hydrogen specific constants
"""Hydrogen molecule (H2). All values are in atomic units."""
equilibrium = 74e-12 * a0_per_m
mu = 1.00782503207 * me_per_amu / 2  # Reduced mass
freq = 4342 * eh_per_cm_1
dissociation_energy = 4.52 * eh_per_ev
force_const = mu * freq ** 2
harmonic_period = np.pi * np.sqrt(2) / freq
morse_a = np.sqrt(force_const / 2 / dissociation_energy)
# Hydrogen-related functions
def get_force_harmonic(r):
    return -2 * force_const * (r - equilibrium)

def harmonic_trajectory(initial_position, initial_speed, t):
    w = freq
    return equilibrium + (initial_position - equilibrium) * np.cos(np.sqrt(2) * w * t) + initial_speed / np.sqrt(2) / w * np.sin(np.sqrt(2) * w * t)

def get_potential_morse(r):  # V(r)
    De = dissociation_energy
    a = morse_a
    r0 = equilibrium
    pot = De * (np.exp(-2 * a * (r - r0)) - 2 * np.exp(-a * (r - r0)))
    return pot

def get_force_morse(r):  # F(r)
    re = equilibrium
    De = dissociation_energy
    a = morse_a
    force = 2 * a * De * (np.exp(-2 * a * (r - re)) - np.exp(-a * (r - re)))
    return force

def morse_trajectory_v0(initial_position, t):  # r(t) ?
    """Returns morse trajectory at time t with specified initial position and 0 initial speed."""
    De = dissociation_energy
    # mu = mu
    a = morse_a
    re = equilibrium
    r0 = initial_position

    c1 = np.exp(a * re)
    c2 = np.exp(a * r0)
    c3 = -De * c1 / c2 * (2 - c1 / c2)
    c4 = De + c2 * c3 / c1
    tau = np.exp(np.sqrt(2 * c3 / mu, dtype=complex) * a * t)

    trajectory = np.log(c1 ** 2 * tau * (c3 * De + (De - c4 / tau) ** 2) / (2 * c1 * c3 * c4)) / a
    return trajectory


# Let's define the Hamiltonian first
H_matrix = np.empty(2, object)  # Hamiltonian is 1x2 matrix with f_1 and f_2 in it
H_matrix[0] = lambda t, r, p: p / mu  # dr/dt
H_matrix[1] = lambda t, r, p: get_force_morse(r)  # dp/dt

# TODO: code vector d, vector q, and matrix Q

# cost function
def cost_function(dy_dx, f_n, grid):  # grid for # of grid points to sum over
    # dy/dx where y_vector = {dr/dt, dp/dt}
    # f_n = {p/mu, F(r)}
    # grid = grid points (OF POTENTIAL SURFACE?)
    eq_i = 0  # value representing [(dy_n/dx)_i - f_n,i] ** 2 for specific n and i values
    for i in len(grid):  # for each point in grid
        individual_term = 0  # term for specific i and n
        for n in len(dy_dx):  # for each equation in Hamiltonian
            individual_term = (dy_dx[n][i] - f_n[n][i]) ** 2
        eq_i += individual_term  # summing term for each equation
    # eq_i is sum of all terms for each function at every grid point
    return eq_i
    # TODO investigate how epsilon(y_vector), then eq in terms of dy/dx; need to differentiate?


# TODO can use sympy? cause that would be so nice

def approximate_dy_dx(y_n, delta_x, i):  # delta_x is time step
    approximate_dy_dx_n_i = (y_n[i + 1] - y_n[i]) / delta_x
    return approximate_dy_dx_n_i  # approimate (dy/dx)_n,i value

def calc_functional_for_segment(x, y_vector, f_n, f_n_0):
    term = f_n_0  # value of f_n when r = r_0 (1.3-1.423) and p = p_0 (0.0)
    for k in len(f_n):  # len(f_n) represents N (upper bound of summation in this case)
        term += f_n[k] * y_vector[k]  # represents adding sum from i to N of f_n,k(x) * y_k(x)
    return term  # represents f_n(x, y_vector) aka functional for each segment

# TODO turn into matrix somehow :sob:
# TODO figure out whats happening with d
# TODO make grid (np.linspace(0, end, num_vals))


