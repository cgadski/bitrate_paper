# %%
import sympy

tau, eta, C = sympy.symbols('tau eta C')
P_0 = eta - (1 - tau)**2 * C / 2
P_1 = 1 - tau**2 * C / 2
solutions = sympy.solve(P_0 - P_1, tau)
for s in solutions:
    sympy.pprint(sympy.simplify(s))
# %%
import matplotlib.pyplot as plt
ex = sympy.simplify(P_0.subs(tau, solutions[0]))
# sympy.plot(ex.subs(eta, 1/4))
# plt.show()
sympy.solve(ex == 0, C)

# %%
opt_tau = sympy.solve(P_0 - P_1, tau)
sympy.solve(opt_tau[0] - 1, C)

# %%
# so C must be greater than 2 (1 - \eta)
