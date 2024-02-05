import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from scipy.spatial import ConvexHull
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


def create_lp(xy, z):
    x, y = xy[0, :], xy[1, :]
    barycentric = cp.Variable((4, 4), "barycentric")
    delta = cp.Variable()
    x_t = cp.Variable()
    z_t = cp.Variable()
    constraints = [barycentric >= 0]
    for j in range(4):
        y_j = y[j]
        z_j = z[:, j]
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
        z_j = z[:, j]
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


def vis_solution_2d(ax, xy, z, x_t, z_t, delta):
    assert x_t is not None, "no solution, cannot visualize it"
    x, y = xy[0, :], xy[1, :]
    points = np.array([[[x[i], y[j], z[i][j]] for j in range(4)] for i in range(4)])
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



seed = int(sys.argv[1])
np.random.seed(seed)

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
