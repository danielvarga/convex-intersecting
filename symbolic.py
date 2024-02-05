import numpy as np
import matplotlib.pyplot as plt
import sympy
import sys


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
# TODO alpha_ij now corresponds to z_ij, but this is achieved in a super confusing, ad hoc way:
alpha = [sympy.symbols(list(f'alpha_{j}{i}' for j in range(4))) for i in range(4)]
alpha = np.array(alpha, dtype=object)

projected_points_raw = [[(x[i], z[i][j] + delta * y[j]) for j in range(4)] for i in range(4)]
projected_points_raw = np.array(projected_points_raw, dtype=object)
projected_points_raw = np.transpose(projected_points_raw, (1, 0, 2))

xz = [[[x[i], z[i][j]] for j in range(4)] for i in range(4)]
xz = np.array(xz, dtype=object)
xz = np.transpose(xz, (1, 0, 2))


def p(x,y):
    return np.array([x, y], dtype=object)

projected_points = [[p(x[i] - x_t, z[i][j] + delta * y[j] - z_t) for j in range(4)] for i in range(4)]
projected_points = np.array(projected_points, dtype=object)
projected_points = np.transpose(projected_points, (1, 0, 2))
log("projected_points", projected_points.shape, repr(projected_points))


alpha_positives = [alpha[i][j] for j in range(4) for i in range(4)]
alpha_eqs = [sum(alpha[i][j] for j in range(4)) - 1 for i in range(4)]
log("alpha_eqs", alpha_eqs)

convex_eqs = []
for i in range(4):
    prod = alpha[i, :, None] * xz[i, :]
    comb = prod.sum(axis=0)
    comb[1] -= delta * y[i]
    for k in range(2):
        expected_result = x_t if k == 0 else z_t
        convex_eq = comb[k] - expected_result
        # convex_eq = convex_eq.subs(x[0], 0).subs(x[-1], 1).subs(y[0], 0).subs(y[-1], 1)
        convex_eqs.append(convex_eq)

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
