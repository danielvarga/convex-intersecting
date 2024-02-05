import numpy as np
import matplotlib.pyplot as plt
import sympy
import sys


def log(*s):
    print(*s, file=sys.stderr)


# variables with a universal quantor:
x = sympy.symbols(list(f'x_{i+1}' for i in range(4)))
y = sympy.symbols(list(f'y_{i+1}' for i in range(4)))
z = [sympy.symbols(list(f'z_{i+1}{j+1}' for j in range(4))) for i in range(4)]

all_vars = x + y + z
eqs = [x[0] - 0, x[3] - 1, y[0] - 0, y[3] - 1]
positives = [x[1], 1 - x[2], x[2] - x[1]] + [y[1], 1 - y[2], y[2] - y[1]]
eqs += [z[0][0] - 0, z[3][3] - 0, z[0][3] - 1, z[3][0] - 1]
log(eqs)

points = [[(x[i], y[j], z[i][j]) for j in range(4)] for i in range(4)]
log("points", points)

# variables with an existential quantor:
delta, x_t, z_t = sympy.symbols('delta x_t z_t')
alpha = [sympy.symbols(list(f'alpha_{i+1}{j+1}' for j in range(4))) for i in range(4)]
alpha = np.array(alpha, dtype=object)

xz = [[[x[i], z[i][j]] for j in range(4)] for i in range(4)]
xz = np.array(xz, dtype=object)
xz = np.transpose(xz, (1, 0, 2))

alpha_positives = [alpha[i][j] for j in range(4) for i in range(4)]
alpha_eqs = [sum(alpha[i][j] for i in range(4)) - 1 for j in range(4)]
log("alpha_eqs", alpha_eqs)

convex_eqs = []
for j in range(4):
    prod = alpha[:, j, None] * xz[j, :]
    comb = prod.sum(axis=0)
    comb[1] -= delta * y[j]
    for k in range(2):
        expected_result = x_t if k == 0 else z_t
        convex_eq = comb[k] - expected_result
        convex_eqs.append(convex_eq)

for convex_eq in convex_eqs:
    log(convex_eq)


print("""\\documentclass{article}
\\begin{document}""")

print("$\\forall x_1,\dots, x_4, y_1,\dots,y_4, z_{11},\dots,z_{44},$")
print()
print("$" + ", ".join(sympy.latex(eq) + " = 0" for eq in eqs) + ",$")
print("$" + ", ".join(sympy.latex(eq) + " \\geq 0" for eq in positives) + ":$")
print()
print("$\\exists \\delta, x_t, z_t$")
print("$\\exists \\alpha_{11},\dots,\\alpha_{44}:$")
print("$" + ", ".join(sympy.latex(eq) + " = 0" for eq in alpha_eqs) + ",$")
print("$" + ", ".join(sympy.latex(eq) + " \\geq 0" for eq in alpha_positives) + ",$")
print()
for convex_eq in convex_eqs:
    print("$" + sympy.latex(convex_eq) + " = 0,$")
    print()


print("\\end{document}")
