import os

import numpy as np

import pandas

import matplotlib.pyplot as plt

import paired_down_qde as qde


class Constants:
    a0_per_m = 1.889726124565062e+10
    me_per_amu = 1.822888484770040e+3
    eh_per_cm_1 = 4.556335256391438e-6
    eh_per_ev = 3.6749308136649e-2


class Hydrogen:
    """Hydrogen molecule (H2). All values are in atomic units."""
    equilibrium = 74e-12 * Constants.a0_per_m
    mu = 1.00782503207 * Constants.me_per_amu / 2  # Reduced mass
    freq = 4342 * Constants.eh_per_cm_1
    dissociation_energy = 4.52 * Constants.eh_per_ev
    force_const = mu * freq ** 2

    @staticmethod
    def get_harmonic_period():
        return np.pi * np.sqrt(2) / Hydrogen.freq

    @staticmethod
    def get_force_harmonic(r):
        return -2 * Hydrogen.force_const * (r - Hydrogen.equilibrium)

    @staticmethod
    def harmonic_trajectory(initial_position, initial_speed, t):
        w = Hydrogen.freq
        return Hydrogen.equilibrium + (initial_position - Hydrogen.equilibrium) * np.cos(np.sqrt(2) * w * t) + initial_speed / np.sqrt(2) / w * np.sin(np.sqrt(2) * w * t)

    @staticmethod
    def get_morse_a():
        return np.sqrt(Hydrogen.force_const / 2 / Hydrogen.dissociation_energy)

    @staticmethod
    def get_potential_morse(r):
        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        r0 = Hydrogen.equilibrium
        pot = De * (np.exp(-2 * a * (r - r0)) - 2 * np.exp(-a * (r - r0)))
        return pot

    @staticmethod
    def get_force_morse(r):
        re = Hydrogen.equilibrium
        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        force = 2 * a * De * (np.exp(-2 * a * (r - re)) - np.exp(-a * (r - re)))
        return force

    @staticmethod
    def morse_trajectory_v0(initial_position, t):
        """Returns morse trajectory at time t with specified initial position and 0 initial speed."""
        De = Hydrogen.dissociation_energy
        mu = Hydrogen.mu
        a = Hydrogen.get_morse_a()
        re = Hydrogen.equilibrium
        r0 = initial_position

        c1 = np.exp(a * re)
        c2 = np.exp(a * r0)
        c3 = -De * c1 / c2 * (2 - c1 / c2)
        c4 = De + c2 * c3 / c1
        tau = np.exp(np.sqrt(2 * c3 / mu, dtype=complex) * a * t)

        trajectory = np.log(c1 ** 2 * tau * (c3 * De + (De - c4 / tau) ** 2) / (2 * c1 * c3 * c4)) / a
        return trajectory


def get_problem(problem_id, **kwargs):
    """This function returns description of a given problem, which consists from: grid, system_terms, boundary_condition and solution (optionally).
    grid - an array that defines what grid is used for time propagation
    system_terms - an array where each element is a function of time and values of all unknown functions (y-vector).
    boundary_condition - an array that defines the values of all unknown functions at time t=0
    solution - an optional function that describes analytical form of the solution (if known) for comparison with the results of propagation.
    kwargs: N, time_max, initial_position."""
    # if problem_id == 0:
    time_max = kwargs['time_max']
    N = kwargs['N']
    initial_position = kwargs['initial_position']
    grid = np.linspace(0, time_max, N)

    system_terms = np.empty(2, dtype=object)
    system_terms[0] = lambda t, r, p: p / Hydrogen.mu
    system_terms[1] = lambda t, r, p: Hydrogen.get_force_morse(r)

    boundary_condition = np.array([initial_position, 0])
    solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)
    # elif problem_id == 1:
    #     # Same as 0, but with scaled momentum
    #     time_max = kwargs['time_max']
    #     N = kwargs['N']
    #     initial_position = kwargs['initial_position']
    #     grid = np.linspace(0, time_max, N)
    #
    #     system_terms = np.empty(2, dtype=object)
    #     system_terms[0] = lambda t, r, p: p / Hydrogen.mu * 20
    #     system_terms[1] = lambda t, r, p: Hydrogen.get_force_morse(r) / 20
    #
    #     boundary_condition = np.array([initial_position, 0])
    #     solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)
    return grid, system_terms, boundary_condition, solution


def get_analytical_solution(problem_id=0, N=1000, time_max=400, initial_position=1.3, **kwargs):
    grid, _, _, solution = get_problem(problem_id, N=N, time_max=time_max, initial_position=initial_position, **kwargs)
    solution_vals = solution(grid)
    if max(abs(np.imag(solution_vals))) < 1e-10:
        solution_vals = np.real(solution_vals)
    return grid, solution_vals


def get_solver(method, **kwargs):
    """Returns solver corresponding to requested method."""
    if method == 'qp':
        return qde.QPSolver()
    else:
        if method == 'qbsolv':
            sampler = qde.QBSolvWrapper(kwargs['num_repeats'])
        elif method == 'dwave':
            sampler = qde.DWaveSamplerWrapper(kwargs['num_reads'], kwargs['use_greedy'])
        else:
            raise Exception('Unknown solver')
        return qde.QUBOSolver(kwargs['bits_integer'], kwargs['bits_decimal'], sampler)


def get_solution(problem_id, N=100, time_max=400, initial_position=1.3, points_per_step=1, equations_per_step=1, max_attempts=1,
                 max_error=1e-10, method='qp', num_repeats=100, num_reads=10000, use_greedy=False, bits_integer=6, bits_decimal=15):
    grid, system_terms, boundary_condition, _ = get_problem(problem_id, N=N, time_max=time_max, initial_position=initial_position)
    solver = get_solver(method, num_repeats=num_repeats, num_reads=num_reads, use_greedy=use_greedy, bits_integer=bits_integer, bits_decimal=bits_decimal)
    solution, errors = qde.solve_ode(system_terms, grid, boundary_condition, points_per_step, equations_per_step, solver, max_attempts, max_error)
    return grid, solution, errors


def save_makedirs(dir_path, solution):
    """Saves given array in a file at given path, creating directories if necessary."""
    os.makedirs(dir_path, exist_ok=True)
    np.savetxt(dir_path + '/solution.txt', solution)


def main():
    _, solution, error = get_solution(problem_id=1,   # problem_id = 0 default; problem_id = 1 uses scaled momentum
                                      N=200,  # increases num of data points from 100 (default) to specified value
                                      time_max=400,  # increases amount of time simulation runs for; diverges as time increase
                                      initial_position=1.3,  # optimal bond length of H2 in Bohr is 1.423; flips grpah on x axis compared to 1.39 Bohr (default val) [flip happens going from 1.39 to 1.4]
                                      max_attempts=1,  # doesn't seem to change graph from 1 (default) to 10 (used in paper?)
                                      use_greedy=False  # also doesn't seem to change graph from False (default) to True (better performance in paper [ONLY APPLICABLE IN QUANTUM?])
                                      )
    solution_df = pandas.DataFrame(data=np.transpose(solution),  # putting data in long form (2 cols, 100 rows)
                                   index=list(range(len(solution[0]))),  # setting length of data frame to the amount of data given
                                   columns=['r', 'p'])  # r in Bohr (coordinate), p in a.u. (momentum)
    return solution_df




if __name__ == '__main__':
    soln = main()
    # _, analytical_solution = get_analytical_solution(problem_id=1, N=200, time_max=400)

    soln_df = pandas.DataFrame(data=soln,  # putting data in long form (2 cols, 100 rows)
                               index=list(range(len(soln))),  # setting length of data frame to the amount of data given
                               columns=['r', 'p'])
    # print(soln)

    save_makedirs(os.getcwd() + "/solution", soln)  # saves data frame in solutions.txt in working directory

    # GRAPHING
    plt.plot(soln_df['r'], soln_df['p'])  # plots r (coord) vs p (momentum) connected by time
    plt.show()

