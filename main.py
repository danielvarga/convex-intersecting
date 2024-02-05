import numpy as np
import matplotlib.pyplot as plt
import sympy
import cvxpy as cp
import sys


def create_stripe():
    return np.array([0] + sorted(np.random.uniform(size=2)) + [1])


def create_config():
    x = create_stripe()
    y = create_stripe()

    xy = np.stack([x, y], axis=0)
    R = 2
    z = np.random.uniform(size=(4, 4), low=-R, high=R)

    z = np.sin(x[None, :] + y[:, None])

    z[0, 0] = z[3, 3] = 0
    z[0, 3] = z[3, 0] = 1
    return xy, z


def create_counterexample():
    r = 1 / 3
    x = np.array([0, r, 1 - r, 1])
    y = x
    xy = np.stack([x, y], axis=0)
    z = np.ones((4, 4))

    z = np.sin(0.9 * x[None, :] + 0.7 * y[:, None] + 0.5)

    z[0, 0] = z[3, 3] = 0
    z[0, 3] = z[3, 0] = 1

    print(z)

    return xy, z


def create_counterexample_2():
    r = 1 / 3
    x = np.array([0, r, 1 - r, 1])
    y = x
    xy = np.stack([x, y], axis=0)
    z = np.ones((4, 4))

    z = np.sin(0.9 * x[None, :] + 0.7 * y[:, None] + 0.5)

    z[0, 0] = z[3, 3] = 0
    z[0, 3] = z[3, 0] = 1

    z = np.around(z, decimals=1)
    z[2, 0] = z[0, 2]

    print(z)

    return xy, z


def create_lp(xy, z):
    x, y = xy[0, :], xy[1, :]
    barycentric = cp.Variable((4, 4), "barycentric")
    delta = cp.Variable()
    x_t = cp.Variable()
    z_t = cp.Variable()
    constraints = [barycentric >= 0]
    for j in range(4):
        y_j = y[j]
        z_j = z[:, j] # TODO is z[j, :] more logical?
        xz_j = np.stack([x, z_j], axis=-1)
        barycentric_j = barycentric[:, j]
        convex_comb_j = barycentric_j @ xz_j
        constraint_x = convex_comb_j[0] == x_t
        constraint_z = convex_comb_j[1] - delta * y_j == z_t
        constraints += [constraint_x, constraint_z, cp.sum(barycentric_j) == 1]
    lp = cp.Problem(cp.Minimize(0), constraints)
    lp.solve()
    # all none if infeasible
    return x_t.value, z_t.value, delta.value


def vis_solution(ax, xy, z, x_t, z_t, delta, transpose=False):
    x, y = xy[0, :], xy[1, :]
    facecolors = "blue" if transpose else "cyan"
    for j in range(4):
        y_j = y[j]
        z_j = z[:, j] # TODO is z[j, :] more logical?
        xz_j = np.stack([x, z_j], axis=-1)
        xyz_j = np.stack([x, y_j * np.ones(4), z_j], axis=-1)
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


seed = int(sys.argv[1])
np.random.seed(seed)

xy, z = create_config()
xy, z = create_counterexample()
# xy, z = create_counterexample_2()

x_t, z_t, delta = create_lp(xy, z)
print("x_t", x_t, "z_t", z_t, "delta", delta)


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_solution(ax, xy, z, x_t, z_t, delta)

xy_trans = xy[::-1, :]
z_trans = z.T
x_t_2, z_t_2, delta_2 = create_lp(xy, z)
vis_solution(ax, xy_trans, z_trans, x_t_2, z_t_2, delta_2, transpose=True)

'''
if x_t_2 is not None:
    y_a = -1
    y_b = 2
    verts = [[y_a, x_t_2 + y_a * delta_2, z_t_2], [y_b, x_t_2 + y_b * delta_2, z_t_2]]
    ax.add_collection3d(Poly3DCollection([verts], facecolors='cyan', linewidths=3, edgecolors='r', alpha=.25))
'''

plt.show()

exit()



def log(*s):
    print(*s, file=sys.stderr)


# variables with a universal quantor:
x = sympy.symbols(list(f'x_{i}' for i in range(4)))
y = sympy.symbols(list(f'y_{i}' for i in range(4)))
z = [sympy.symbols(list(f'z_{i}{j}' for j in range(4))) for i in range(4)]

all_vars = x + y + z
eqs = [x[0] - 0, x[3] - 1, y[0] - 0, y[3] - 1]
positives = [x[1], 1 - x[2], x[2] - x[1]] + [y[1], 1 - y[2], y[2] - y[1]]
eqs += [z[0][0] - 0, z[3][3] - 0, z[0][3] - 1, z[3][0] - 1]
log(eqs)

points = [[(x[i], y[j], z[i][j]) for j in range(4)] for i in range(4)]
log("points", points)

# variables with an existential quantor:
delta, x_t, z_t = sympy.symbols('delta x_t z_t')
alpha = [sympy.symbols(list(f'alpha_{i}{j}' for j in range(4))) for i in range(4)]
alpha = np.array(alpha, dtype=object)

projected_points_raw = [[(x[i] + delta * y[j], z[i][j]) for j in range(4)] for i in range(4)]
projected_points_raw = np.array(projected_points_raw, dtype=object)


def p(x,y):
    return np.array([x, y], dtype=object)

projected_points = [[p(x[i] + delta * y[j] - x_t, z[i][j] - z_t) for j in range(4)] for i in range(4)]
projected_points = np.array(projected_points, dtype=object)
log("projected_points", repr(projected_points))


alpha_positives = [alpha[i][j] for j in range(4) for i in range(4)]
alpha_eqs = [sum(alpha[i][j] for j in range(4)) - 1 for i in range(4)]
log("alpha_eqs", alpha_eqs)

convex_eqs = []
for i in range(4):
    prod = alpha[i, :, None] * projected_points_raw[i, :]
    comb = prod.sum(axis=0)
    for k in range(2):
        expected_result = x_t if k == 0 else z_t
        convex_eqs.append(comb[k] - expected_result)

for convex_eq in convex_eqs:
    log(convex_eq)


print("""\\documentclass{article}
\\begin{document}""")

print("$\\forall x_0,\dots, x_3, y_0,\dots,y_3, z_{00},\dots,z_{33},$")
print()
print("$" + ", ".join(sympy.latex(eq) + " = 0" for eq in eqs) + ",$")
print("$" + ", ".join(sympy.latex(eq) + " \\geq 0" for eq in positives) + ":$")
print()
print("$\\exists \\delta, x_t, z_t$")
print("$\\exists \\alpha_{00},\dots,\\alpha_{33}:$")
print("$" + ", ".join(sympy.latex(eq) + " = 0" for eq in alpha_eqs) + ",$")
print("$" + ", ".join(sympy.latex(eq) + " \\geq 0" for eq in alpha_positives) + ",$")
print()
for convex_eq in convex_eqs:
    print("$" + sympy.latex(convex_eq) + " = 0,$")
    print()


print("\\end{document}")
