# Hazel Dellario
# 2024/03/29
# Using Igor Gaidai's code and qiskit warm start qaoa

import numpy as np

from utils_general import print_progress_bar

from docplex.mp.model import Model

from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler

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


def calculate_term_coefficients(system_terms, approximation_point, sampling_steps, grid):
    """Linearly approximates system RHS in the vicinity of a given point.
    Args:
        system_terms (numpy.ndarray): 1D array of functions that define rhs of each ODE in the system. Each function is linearly approximated within a given job.
        approximation_point (numpy.ndarray): 1D array that specifies coordinate around which linear approximation is made.
        sampling_steps (numpy.ndarray): 1D array of steps along each coordinate where additional points are sampled for linear fitting.
        grid (numpy.ndarray): 1D array of x values for which the system terms are evaluated.
    Returns:
        funcs (numpy.ndarray): 3D array with values of approximated rhs terms at all points of this job. 1st dim - equations, 2nd dim - terms, 3rd dim - points.
    """
    funcs = np.zeros((len(system_terms), 1 + len(system_terms), len(grid)))
    if len(grid) == 1:
        # Only shifts need to be calculated
        for eq_ind in range(funcs.shape[0]):
            funcs[eq_ind, 0, 0] = system_terms[eq_ind](grid[0], *approximation_point)
    else:
        fitting_matrix = np.zeros((funcs.shape[1], funcs.shape[1]))
        for row_ind in range(funcs.shape[1]):
            next_point = approximation_point.copy()
            if row_ind > 0:
                next_point[row_ind - 1] += sampling_steps[row_ind - 1]
            fitting_matrix[row_ind, :] = [1, *next_point]

        for eq_ind in range(funcs.shape[0]):
            for point_ind in range(funcs.shape[2]):
                fitting_vector = np.zeros(funcs.shape[1])
                for row_ind in range(funcs.shape[1]):
                    next_point = approximation_point.copy()
                    if row_ind > 0:
                        next_point[row_ind - 1] += sampling_steps[row_ind - 1]
                    fitting_vector[row_ind] = system_terms[eq_ind](grid[point_ind], *next_point)
                funcs[eq_ind, :, point_ind] = np.linalg.solve(fitting_matrix, fitting_vector)
    return funcs


def add_symmetric(H, ind1, ind2, value):  # keeps Q a semi-definite matrix?
    """Splits specified value between the two off-diagonals of H.
    Args:
        H (numpy.ndarray): Matrix to which value is added.
        ind1 (int): first index of position in H where value is added.
        ind2 (int): second index of position in H where value is added.
        value (float): value to add.
    """
    H[ind1, ind2] += value / 2
    H[ind2, ind1] += value / 2


def add_point_terms_qp(H, d, point_ind, eq_ind_start, eq_ind_end, funcs_i, dx, known_points=None):
    """Adds functional terms for a given point to H and d.
    Args:
        H (numpy.ndarray): Current quadratic minimization matrix to which quadratic terms of specified point are added.
        d (numpy.ndarray): Current quadratic minimization vector to which linear terms of specified point are added.
        point_ind (int): Local point index within the current job.
        eq_ind_start (int): Index of the first considered equation.
        eq_ind_end (int): Index of the last considered equation (exclusive).
        funcs_i (numpy.ndarray): 2D array with values of approximated rhs terms at the current point. Equations are along rows, terms along columns.
        dx (float): Grid step.
        known_points (numpy.ndarray): When adding terms for the last known point, this is 1D array of the values of each function at that point, otherwise not needed.
    Returns:
        energy_shift (float): Constant part of minimization functional.
    """
    energy_shift = 0
    get_unknown_ind = lambda point, eq: (point - 1) * (eq_ind_end - eq_ind_start) + (eq - eq_ind_start)
    for eq_ind in range(eq_ind_start, eq_ind_end):
        next_unknown_ind = get_unknown_ind(point_ind + 1, eq_ind)
        H[next_unknown_ind, next_unknown_ind] += 1 / dx ** 2
        d[next_unknown_ind] += -2 * funcs_i[eq_ind, 0] / dx
        energy_shift += funcs_i[eq_ind, 0] ** 2

        if point_ind == 0:
            # Current point is known
            assert known_points is not None, 'known_points have to be supplied for 0th point in each job'
            d[next_unknown_ind] += -2 * known_points[eq_ind] / dx ** 2
            energy_shift += (known_points[eq_ind] / dx) ** 2
            energy_shift += 2 * known_points[eq_ind] * funcs_i[eq_ind, 0] / dx
            for term_ind in range(1, funcs_i.shape[1]):
                d[next_unknown_ind] += -2 * funcs_i[eq_ind, term_ind] * known_points[term_ind - 1] / dx
                energy_shift += 2 * funcs_i[eq_ind, term_ind] * known_points[term_ind - 1] * known_points[eq_ind] / dx
                energy_shift += 2 * funcs_i[eq_ind, 0] * funcs_i[eq_ind, term_ind] * known_points[term_ind - 1]
                for term_ind2 in range(1, funcs_i.shape[1]):
                    energy_shift += funcs_i[eq_ind, term_ind] * funcs_i[eq_ind, term_ind2] * known_points[term_ind - 1] * known_points[term_ind2 - 1]
        else:
            unknown_ind = get_unknown_ind(point_ind, eq_ind)
            add_symmetric(H, unknown_ind, next_unknown_ind, -2 / dx ** 2)
            H[unknown_ind, unknown_ind] += 1 / dx ** 2
            d[unknown_ind] += 2 * funcs_i[eq_ind, 0] / dx
            for term_ind in range(1, funcs_i.shape[1]):
                term_unknown_ind = get_unknown_ind(point_ind, term_ind - 1)
                add_symmetric(H, term_unknown_ind, next_unknown_ind, -2 * funcs_i[eq_ind, term_ind] / dx)
                add_symmetric(H, term_unknown_ind, unknown_ind, 2 * funcs_i[eq_ind, term_ind] / dx)
                d[term_unknown_ind] += 2 * funcs_i[eq_ind, 0] * funcs_i[eq_ind, term_ind]
                for term_ind2 in range(1, funcs_i.shape[1]):
                    term_unknown_ind2 = get_unknown_ind(point_ind, term_ind2 - 1)
                    add_symmetric(H, term_unknown_ind, term_unknown_ind2, funcs_i[eq_ind, term_ind] * funcs_i[eq_ind, term_ind2])

    return energy_shift


def build_qp_matrices(funcs, dx, known_points, eq_ind_start, eq_ind_end):
    """Builds matrices H and d that define quadratic minimization problem corresponding to a given system of differential equations.
    Args:
        funcs (numpy.ndarray): 3D array with values of approximated rhs terms at all points of this job. 1st dim - equations, 2nd dim - terms, 3rd dim - points.
        dx (float): Grid step.
        known_points (numpy.ndarray): 1D array of known points for each function at 0th point (boundary condition).
        eq_ind_start (int): Index of the first considered equation.
        eq_ind_end (int): Index of the last considered equation (exclusive).
    Returns:
        H (numpy.ndarray): Quadratic minimization matrix.
        d (numpy.ndarray): Quadratic minimization vector.
        energy_shift (float): Constant part of minimization functional.
    """
    unknowns = (eq_ind_end - eq_ind_start) * funcs.shape[2]
    H = np.zeros((unknowns, unknowns))
    d = np.zeros(unknowns)
    energy_shift = 0
    for point_ind in range(funcs.shape[2]):
        energy_shift += add_point_terms_qp(H, d, point_ind, eq_ind_start, eq_ind_end, funcs[:, :, point_ind], dx, known_points)
    return H, d, energy_shift


def solve_ode(system_terms, grid, boundary_condition, points_per_step, equations_per_step, solver, max_attempts, max_error):
    """Solves a given ODE system, defined by system_terms and known_points, by formulating it as a QP problem.
    Args:
        system_terms (numpy.ndarray): 1D array of functions that define rhs of each ODE in the system. Each function is linearly approximated within a given job.
        grid (numpy.ndarray): 1D Array of equidistant grid points.
        boundary_condition (numpy.ndarray): 1D array of initial values for each function in the system.
        points_per_step (int): Number of points to vary per job.
        equations_per_step (int): Number of equations to vary per job.
        solver (Solver): Solver to solve QP problem.
        max_attempts (int): Maximum number of times each problem can be solved (restarts can find a better solution for some solvers).
        max_error (float): Maximum error that does not trigger restart.
    Returns:
        solution (numpy.ndarray): 2D array with solution for all functions at all grid points.
        errors (numpy.ndarray): 1D array with errors for each job.
    """
    print(f'Solving ODE... Solver={type(solver).__name__}; N={len(grid)}.')
    solution = np.zeros((len(system_terms), len(grid)))
    solution[:, 0] = boundary_condition
    dx = grid[1] - grid[0]
    point_ind = 0
    errors = []
    working_grid = grid[:-1]
    while point_ind < len(working_grid):
        if point_ind == 0:
            sampling_steps = np.zeros(len(system_terms))
            funcs = calculate_term_coefficients(system_terms, solution[:, point_ind], sampling_steps, working_grid[point_ind: point_ind + 1])
        else:
            sampling_steps = solution[:, point_ind] - solution[:, point_ind - 1]
            sampling_steps[abs(sampling_steps) < 1e-10] = 1e-10  # Ensure non-zero steps
            funcs = calculate_term_coefficients(system_terms, solution[:, point_ind], sampling_steps, working_grid[point_ind: point_ind + points_per_step])

        solution_points = np.zeros(solution.shape[0])
        eq_ind = 0
        while eq_ind < len(solution_points):
            H, d, energy_shift = build_qp_matrices(funcs, dx, solution[:, point_ind], eq_ind, eq_ind + equations_per_step)
            lowest_error = np.inf
            for attempt in range(max_attempts):
                job_label = f'Point ind: {point_ind}; Eq. {eq_ind}; Attempt {attempt + 1}'
                trial_points = solver.solve(H, d, job_label)
                trial_error = np.dot(np.matmul(trial_points, H), trial_points) + np.dot(trial_points, d) + energy_shift
                if trial_error < lowest_error:
                    lowest_error = trial_error
                    solution_points[eq_ind: eq_ind + equations_per_step] = trial_points
                if trial_error < max_error:
                    break
            errors.append(lowest_error)
            eq_ind += equations_per_step

        solution_points_shaped = np.reshape(solution_points, (len(system_terms), funcs.shape[2]), order='F')
        solution[:, point_ind + 1: point_ind + funcs.shape[2] + 1] = solution_points_shaped
        point_ind += funcs.shape[2]
        print_progress_bar(point_ind, len(working_grid))

    return solution, np.array(errors)


def get_problem(**kwargs):
    """This function returns description of a given problem, which consists from: grid, system_terms, boundary_condition and solution (optionally).
    grid - an array that defines what grid is used for time propagation
    system_terms - an array where each element is a function of time and values of all unknown functions (y-vector).
    boundary_condition - an array that defines the values of all unknown functions at time t=0
    solution - an optional function that describes analytical form of the solution (if known) for comparison with the results of propagation.
    kwargs: N, time_max, initial_position."""
    time_max = kwargs['time_max']
    N = kwargs['N']
    initial_position = kwargs['initial_position']
    grid = np.linspace(0, time_max, N)

    system_terms = np.empty(2, dtype=object)
    system_terms[0] = lambda t, r, p: p / Hydrogen.mu
    system_terms[1] = lambda t, r, p: Hydrogen.get_force_morse(r)

    boundary_condition = np.array([initial_position, 0])
    # solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)  # an optional function that describes analytical form of the solution (if known) for comparison with the results of propagation
    return grid, system_terms, boundary_condition  # , solution


def get_solver(method, **kwargs):  # where we call function to create QP matrix
    """Returns solver corresponding to requested method."""
    # if method == 'qp':
    #     return qde.QPSolver()
    # else:
    #     if method == 'qbsolv':
    #         sampler = qde.QBSolvWrapper(kwargs['num_repeats'])
    # elif method == 'dwave':
    #     sampler = qde.DWaveSamplerWrapper(kwargs['num_reads'], kwargs['use_greedy'])
    if method == "qiskit":
        sampler = Sampler()
    else:
        raise Exception('Unknown solver')
    return qde.QUBOSolver(kwargs['bits_integer'], kwargs['bits_decimal'], sampler)


def get_solution(N=100, time_max=400, initial_position=1.3, points_per_step=1, equations_per_step=1, max_attempts=1,
                 max_error=1e-10, method='qp', num_repeats=100, num_reads=10000, use_greedy=False, bits_integer=6, bits_decimal=15):
    grid, system_terms, boundary_condition, _ = get_problem(N=N, time_max=time_max, initial_position=initial_position)
    solver = get_solver(method, num_repeats=num_repeats, num_reads=num_reads, use_greedy=use_greedy, bits_integer=bits_integer, bits_decimal=bits_decimal)
    solution, errors = solve_ode(system_terms, grid, boundary_condition, points_per_step, equations_per_step, solver, max_attempts, max_error)
    return grid, solution, errors



