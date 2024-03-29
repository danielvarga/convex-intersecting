import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import sympy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull
import gurobipy
from gurobipy import GRB

import sys


n = 4


def create_stripe():
    return np.array([0] + sorted(np.random.uniform(size=n - 2)) + [1])


def create_config():
    x = create_stripe()
    y = create_stripe()

    xy = np.stack([x, y], axis=0)
    R = 2
    z = np.random.uniform(size=(n, n), low=-R, high=R)

    # z = np.sin(x[None, :] + y[:, None])

    z[0, 0] = z[n - 1, n - 1] = 0
    z[0, n - 1] = z[n - 1, 0] = 1
    return xy, z


def create_lp(xy, z):
    x, y = xy[0, :], xy[1, :]
    barycentric = cp.Variable((n, n), "barycentric")
    delta = cp.Variable()
    x_t = cp.Variable(name="x_t")
    z_t = cp.Variable(name="z_t")
    constraints = [barycentric >= 0]
    for j in range(n):
        y_j = y[j]
        z_j = z[:, j]
        xz_j = np.stack([x, z_j], axis=-1)
        barycentric_j = barycentric[:, j]
        convex_comb_j = barycentric_j @ xz_j
        constraint_x = convex_comb_j[0] == x_t
        constraint_z = convex_comb_j[1] - delta * y_j == z_t
        constraints += [constraint_x, constraint_z, cp.sum(barycentric_j) == 1]
    lp = cp.Problem(cp.Maximize(cp.min(barycentric)), constraints)
    lp.solve(solver="GUROBI")
    # print("minimal barycentric", lp.value)

    delta = delta.value ; x_t = x_t.value ; z_t = z_t.value
    print("the verification part has not been ported 4->n")
    return x_t, z_t, delta

    if x_t is not None:
        alpha = barycentric.value
        print("alpha", alpha)
        print("alpha sums", alpha.sum(axis=0), alpha.sum(axis=1))
        alpha_z = z.T @ alpha
        convex_comb_z = np.diag(alpha_z)
        assert np.allclose(x @ alpha - x_t, 0)
        assert np.allclose(convex_comb_z - delta * y - z_t, 0)
        assert np.allclose(alpha.sum(axis=0) - 1, 0)

        # in this format it's trivial to dualize the system:
        big_A = np.zeros((12, 19))
        for j in range(4):
            big_A[j, j * 4: (j+1) * 4] = x
            big_A[j, -2] = -1
            big_A[j + 4, j * 4: (j+1) * 4] = z[:, j]
            big_A[j + 4, -3] = - y[j]
            big_A[j + 4, -1] = -1
            big_A[j + 8, j * 4: (j+1) * 4] = 1
        big_x = np.array(alpha.T.flatten().tolist() + [delta, x_t, z_t])
        big_b = np.zeros(12)
        big_b[-4:] = 1
        assert np.allclose(big_A @ big_x - big_b, 0)

        # that's just an awkward way to verify that alpha is nonnegative:
        big_C = np.zeros((16, 19))
        big_C[:16, :16] = - np.eye(16)
        # assert np.all(big_C @ big_x <= 0)

    # all None if infeasible
    return x_t, z_t, delta


# https://chat.openai.com/share/42a336d9-6853-4963-8bdc-239b55c84e24
# this is supposed to do the same as create_lp(xy, z),
# but with all-nonnegative variables, and an easier to dualize format.
def create_lp_matrix(xy, z):
    assert n == 4, "not yet ported 4->n"
    x, y = xy[0, :], xy[1, :]
    dtype = z.dtype if isinstance(z, np.ndarray) else object
    big_A = np.zeros((12, 22), dtype=dtype)
    # 22 = 16 alphas + 3 positive part + 3 negative part.
    for j in range(4):
        big_A[j, j * 4: (j+1) * 4] = x
        big_A[j + 4, j * 4: (j+1) * 4] = z[:, j]
        big_A[j + 4, -3-3] = - y[j] # delta_p
        big_A[j, -2-3] = -1 # x_t_p
        big_A[j + 4, -1-3] = -1 # z_t_p
        big_A[j + 4, -3] = y[j] # delta_m
        big_A[j, -2] = 1 # x_t_m
        big_A[j + 4, -1] = 1 # z_t_m
        big_A[j + 8, j * 4: (j+1) * 4] = 1
    big_b = np.zeros(12, dtype=dtype)
    big_b[-4:] = 1

    return big_A, big_b
    # next three lines just documentation
    big_x = cp.Variable(22, name="big_x")
    constraints = [big_x >= 0, big_A @ big_x == big_b]
    lp = cp.Problem(cp.Maximize(cp.min(big_x[:16])), constraints)


# should be functionally completely identical to create_lp(xy, z)
def create_primal_lp_via_matrix(xy, z):
    assert n == 4, "not yet ported 4->n"
    big_A, big_b = create_lp_matrix(xy, z)
    big_x = cp.Variable(22, name="big_x")
    constraints = [big_x >= 0, big_A @ big_x == big_b]
    lp = cp.Problem(cp.Maximize(cp.min(big_x[:16])), constraints)
    lp.solve(solver="GUROBI")
    big_x_np = big_x.value
    if big_x_np is None:
        return None, None, None
    alphas = big_x_np[:16].reshape((4, 4)).T
    delta = big_x_np[-3-3] - big_x_np[-3]
    x_t = big_x_np[-2-3] - big_x_np[-2]
    z_t = big_x_np[-1-3] - big_x_np[-1]
    return x_t, z_t, delta


def create_dual_lp_via_matrix(xy, z):
    assert n == 4, "not yet ported 4->n"
    big_A, big_b = create_lp_matrix(xy, z)
    big_y = cp.Variable(12, name="big_y")
    constraints = [big_A.T @ big_y >= 0, cp.norm(big_y, 1) <= 1]
    lp = cp.Problem(cp.Minimize(big_b @ big_y), constraints)
    lp.solve(solver="GUROBI")
    return lp.value, big_y.value


def verify_farkas_lemma():
    for _ in range(1000):
        xy, z = create_config()
        x_t, z_t, delta = create_lp(xy, z)
        x_t_prime, z_t_prime, delta_prime = create_primal_lp_via_matrix(xy, z)
        assert (x_t is not None) == (x_t_prime is not None)

        dual_objective_value, big_y = create_dual_lp_via_matrix(xy, z)
        primal_solvable = x_t is not None
        dual_solvable = dual_objective_value is not None and dual_objective_value < 0 and not np.isclose(dual_objective_value, 0)
        assert primal_solvable != dual_solvable, (primal_solvable, dual_solvable)


def create_combined_duals_lp_matrix(xy, z):
    big_A, big_b = create_lp_matrix(xy, z)
    xy_trans = xy[::-1, :]
    z_trans = z.T
    big_A_prime, big_b_prime = create_lp_matrix(xy_trans, z_trans)
    big_y = cp.Variable(3 * n, name="big_y")
    big_y_prime = cp.Variable(3 * n, name="big_y_prime")
    slack = cp.Variable(name="slack")
    constraints =  [big_A.T @ big_y >= 0,
                    cp.norm(big_y, 1) <= 1,
                    big_A_prime.T @ big_y_prime >= 0,
                    cp.norm(big_y_prime, 1) <= 1,
                    big_b @ big_y <= slack,
                    big_b_prime @ big_y_prime <= slack
    ]
    lp = cp.Problem(cp.Minimize(slack), constraints)
    lp.solve(solver="GUROBI")
    return lp.value, big_y.value, big_y_prime.value


def test_create_primal_lp_via_matrix():
    xy, z = create_config()
    x_t, z_t, delta = create_lp(xy, z)
    print(x_t, z_t, delta)
    print("========")
    x_t, z_t, delta = create_primal_lp_via_matrix(xy, z)
    print(x_t, z_t, delta)


def verify(xy, z):
    x_t, z_t, delta = create_lp(xy, z)
    x_good = x_t is not None
    xy_trans = xy[::-1, :]
    z_trans = z.T
    x_t_2, z_t_2, delta_2 = create_lp(xy_trans, z_trans)
    y_good = x_t_2 is not None
    is_counterexample = not x_good and not y_good
    if is_counterexample:
        print("COUNTEREXAMPLE")
        print(repr(xy), repr(z))
    return int(x_good) + int(y_good)


# verify(xy, z) ; exit()

# verify_farkas_lemma() ; exit()


def search_counterexample_via_combined_duals():
    for i in range(10000):
        xy, z = create_config()
        slack, big_y, big_y_prime = create_combined_duals_lp_matrix(xy, z)
        if not np.isclose(slack, 0):
            print("COUNTEREXAMPLE")
            print(xy, z)
            exit()
        if i % 100 == 0:
            print(i)


# search_counterexample_via_combined_duals() ; exit()


def create_symbolic_combined_duals(xy, z, u, v):
    big_A, big_b = create_lp_matrix(xy, z)
    xy_trans = xy[::-1, :]
    z_trans = z.T
    big_A_prime, big_b_prime = create_lp_matrix(xy_trans, z_trans)

    '''
    constraints =  [big_A.T @ u,
                    big_A_prime.T @ v,
                    - big_b @ u,
                    - big_b_prime @ v
    ]
    '''

    u = np.array([uu for uu in u])
    v = np.array([vv for vv in v])

    c_primal = np.array([row @ u for row in big_A.T])
    c_dual = np.array([row @ v for row in big_A_prime.T])

    constraints =  [c_primal, c_dual,
                    - big_b @ u,
                    - big_b_prime @ v
    ]
    return constraints


def create_symbolic_combined_duals_sympy():
    x = sympy.symbols(list(f'x_{i+1}' for i in range(4)))
    y = sympy.symbols(list(f'y_{i+1}' for i in range(4)))
    xy = np.array([x, y], dtype=object)
    z = np.array([[sympy.symbols(list(f'z_{i+1}{j+1}' for j in range(4)))] for i in range(4)], dtype=object)
    z = z.squeeze()

    u = np.array([[sympy.symbols(list(f'u_{i+1}{j+1}' for j in range(n)))] for i in range(3)], dtype=object).flatten()
    v = np.array([[sympy.symbols(list(f'v_{i+1}{j+1}' for j in range(n)))] for i in range(3)], dtype=object).flatten()

    constraints = create_symbolic_combined_duals(xy, z, u, v)

    return xy, z, constraints


def vis_solution(ax, xy, z, x_t, z_t, delta, transpose=False):
    x, y = xy[0, :], xy[1, :]
    facecolors = "blue" if transpose else "cyan"
    for j in range(n):
        y_j = y[j]
        z_j = z[:, j]
        xz_j = np.stack([x, z_j], axis=-1)
        xyz_j = np.stack([x, y_j * np.ones(n), z_j], axis=-1)
        if transpose:
            xyz_j = xyz_j[:, [1, 0, 2]]
        ax.add_collection3d(Poly3DCollection([xyz_j], facecolors=facecolors, linewidths=1, edgecolors='b', alpha=.25))

    if x_t is not None:
        y_a = -1
        y_b = 2
        verts = np.array([[x_t, y_a, z_t + y_a * delta], [x_t, y_b, z_t + y_b * delta]])
        if transpose:
            verts = verts[:, [1, 0, 2]]
        ax.add_collection3d(Poly3DCollection([verts], facecolors='cyan', linewidths=3, edgecolors='r', alpha=.25))


def vis_solution_combined(ax, xy, z):
    x_t, z_t, delta = create_lp(xy, z)

    vis_solution(ax, xy, z, x_t, z_t, delta)

    xy_trans = xy[::-1, :]
    z_trans = z.T
    x_t_2, z_t_2, delta_2 = create_lp(xy_trans, z_trans)
    vis_solution(ax, xy_trans, z_trans, x_t_2, z_t_2, delta_2, transpose=True)


def vis_solution_2d(ax, xy, z, x_t, z_t, delta):
    assert x_t is not None, "no solution, cannot visualize it"
    x, y = xy[0, :], xy[1, :]
    points = np.array([[[x[i], y[j], z[i][j]] for j in range(n)] for i in range(n)])
    projected_x = points[:, :, 0] - x_t
    projected_y = points[:, :, 1] # not really part of the projection to xz, but kept around.
    projected_z = points[:, :, 2] - delta * points[:, :, 1] - z_t
    projected = np.stack([projected_x, projected_y, projected_z], axis=-1)
    colors = ['red', 'green', 'blue', 'yellow']
    for j, color in enumerate(colors):
        xyz_j = projected[:, j, :]
        xz_j = xyz_j[:, [0, 2]]
        points = xz_j
        hull = ConvexHull(points)
        ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'darkblue', edgecolor=color, alpha=0.3)  # Fill the convex hull with a transparent color

    plt.axhline(0, color='black')
    plt.axvline(0, color='black')



def mini_main():
    xy = np.stack([np.linspace(0, 1, n), np.linspace(0, 1, n)], axis=0)
    np.random.seed(3)
    z = np.random.uniform(size=(n, n))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis_solution_combined(ax, xy, z)
    plt.show()


def subs_saddle(constraint, z):
    return constraint.subs(z[0, 0], 0).subs(z[-1, -1], 0).subs(z[0, -1], 1).subs(z[-1, 0], 0)


def combined_dual_to_gurobi():
    m = gurobipy.Model("combined_dual")
    m.setParam('NonConvex', 2)
    m.setParam('Seed', 44)
    m.setParam('FeasibilityTol', 1e-9)

    x = m.addMVar(n - 2, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    y = m.addMVar(n - 2, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="y")
    xy = np.zeros((2, n), dtype=object)
    xy[0, 1: n-1] = np.array([xx for xx in x])
    xy[1, 1: n-1] = np.array([yy for yy in y])
    xy[:, n-1] = 1

    z = m.addMVar((n, n), vtype=GRB.CONTINUOUS, lb=-2, ub=+2, name="z")

    u = m.addMVar(3 * n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="u")
    v = m.addMVar(3 * n, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="v")

    constraints = create_symbolic_combined_duals(xy, z, u, v)

    slack = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="slack")

    nonnegativity_constraints = constraints[0].tolist() + constraints[1].tolist()
    negativity_constraints = [ - constraints[2], - constraints[3] ]

    m.addConstr(z[0, 0] == 0)
    m.addConstr(z[n-1, n-1] == 0)
    m.addConstr(z[0, n-1] == 1)
    m.addConstr(z[n-1, 0] == 1)

    for constraint in nonnegativity_constraints:
        m.addConstr(constraint >= 0)
    
    for constraint in negativity_constraints:
        m.addConstr(constraint <= slack)

    # we only care about negative solutions, how much negative does not matter.
    # gurobi settles on optimality faster if we accept not-so-good solutions.
    m.addConstr(slack >= -0.01)

    epsilon = 0
    m.addConstr(x[0] >= epsilon)
    m.addConstr(x[1] <= 1 - epsilon)
    m.addConstr(x[0] + epsilon <= x[1])

    m.addConstr(y[0] >= epsilon)
    m.addConstr(y[1] <= 1 - epsilon)
    m.addConstr(y[0] + epsilon <= y[1])

    m.setObjective(slack, GRB.MINIMIZE)

    m.optimize()
    if m.status == GRB.OPTIMAL:
        print("slack", slack.X)
    else:
        print("could not solve")
        return

    x = x.X
    y = y.X
    z = z.X
    xy = np.zeros((2, 4))
    xy[0, 1:3] = np.array([xx for xx in x])
    xy[1, 1:3] = np.array([yy for yy in y])
    xy[:, 3] = 1
    print(repr(xy))
    print(repr(z))
    solvable = verify(xy, z)
    print("solvable", solvable)
    slack, big_y, big_y_prime = create_combined_duals_lp_matrix(xy, z)
    print("dual slack", slack)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis_solution_combined(ax, xy, z)
    plt.show()

    if not np.isclose(slack, 0):
        print("COUNTEREXAMPLE")
        print(xy, z)
        exit()


# combined_dual_to_gurobi() ; exit()


def dump_combined_dual():
    xy, z, constraints = create_symbolic_combined_duals_sympy()

    equality_constraints = []
    for block in (0, 1):
        constraint_block = constraints[block]
        for line in (-1, -2, -3):
            if constraint_block[line] + constraint_block[line-3] == 0:
                print("equality identified", constraint_block[line], file=sys.stderr)
                equality_constraints.append(constraint_block[line])
    assert len(equality_constraints) == 6
    constraints[0] = constraints[0][:-6]
    constraints[1] = constraints[1][:-6]

    equality_constraints = np.array(equality_constraints, dtype=object)
    equality_constraints = equality_constraints[[1, 4, 0, 3, 2, 5]]

    nonnegativity_constraints = constraints[0].tolist() + constraints[1].tolist()
    negativity_constraints = [ - constraints[2], - constraints[3] ]
    print("""\\documentclass{article}
\\usepackage[margin=1in]{geometry}
\\usepackage{amsfonts}
\\begin{document}
""")

    print("$\\exists x, y \in \mathbb{R}^{4}, \\exists Z \in \mathbb{R}^{4 \\times 4},$")
    print("$\\exists U, V \in \mathbb{R}^{3 \\times 4}:$")
    print()

    for constraint in nonnegativity_constraints:
        print("$" + sympy.latex(constraint) + " \\geq 0,$")
        # print("$" + sympy.latex(subs_saddle(constraint, z)) + " \\geq 0,$")
        print()
    for constraint in equality_constraints:
        print("$" + sympy.latex(constraint) + " = 0,$")
        print()
    for constraint in negativity_constraints:
        print("$" + sympy.latex(constraint) + " = -1,$")
        print()

    print("\\end{document}")


dump_combined_dual() ; exit()

create_symbolic_combined_duals() ; exit()



def update(xy, z):
    xy = xy.copy()
    z = z.copy()
    xy += np.random.normal(size=xy.shape, scale=0.1)
    xy = np.clip(xy, 0, 1)
    z += np.random.normal(size=z.shape, scale=0.1)
    return xy, z


seed = int(sys.argv[1])
np.random.seed(seed)


direction_dict = {0: 0, 1: 0 , 2: 0}
xy, z = create_config()

stack = [(xy, z)]
for i in range(1000):
    directions = verify(xy, z)
    direction_dict[directions] += 1
    if directions == 1:
        stack.append((xy, z))
    if i % 100 == 0:
        print(i, len(stack), direction_dict)
    if i % 100 == 0:
        xy, z = stack[np.random.randint(len(stack))]
    xy, z = update(xy, z)

exit()

direction_dict = {0: 0, 1: 0 , 2: 0}
for i in range(10000):
    xy, z = create_config()
    directions = verify(xy, z)
    direction_dict[directions] += 1
    if i % 100 == 0:
        print(i, direction_dict)

exit()

xy1, z1 = create_config()
xy2, z2 = create_config()

for i, lmbda in enumerate(np.linspace(-1, 1, 100)):
    xy = xy1 * lmbda + xy2 * (1 - lmbda)
    z = z1 * lmbda + z2 * (1 - lmbda)

    print(lmbda, int(x_good), int(y_good))
    if not x_good and not y_good:
        assert False, "whoa, counterexample?"

exit()



xy, z = create_config()

x_t, z_t, delta = create_lp(xy, z)
print("x_t", x_t, "z_t", z_t, "delta", delta)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_solution(ax, xy, z, x_t, z_t, delta)

xy_trans = xy[::-1, :]
z_trans = z.T
x_t_2, z_t_2, delta_2 = create_lp(xy_trans, z_trans)
vis_solution(ax, xy_trans, z_trans, x_t_2, z_t_2, delta_2, transpose=True)

plt.show()


if x_t is not None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vis_solution_2d(ax, xy, z, x_t, z_t, delta)
    plt.xlim(-1, 1) ; plt.ylim(-1, 1)
    plt.show()

if x_t_2 is not None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vis_solution_2d(ax, xy_trans, z_trans, x_t_2, z_t_2, delta_2)
    plt.xlim(-1, 1) ; plt.ylim(-1, 1)
    plt.show()
