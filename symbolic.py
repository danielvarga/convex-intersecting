assert False, "this sympy code has not been updated after noticing the formulation issue"


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
