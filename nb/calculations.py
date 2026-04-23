# %%
import sympy

tau, eta, C, eps = sympy.symbols('tau eta C eps')
P_0 = eta - (1 - tau)**2 * C / 2
P_1 = 1 - tau**2 * C / 2
solutions = sympy.solve(P_0 - P_1, tau)
for s in solutions:
    sympy.pprint(sympy.simplify(s))

# %%
sympy.pprint(P_0.subs(tau, 1 / (1 + sympy.sqrt(eta))).\
    subs(C, (1 + eps * C))


# %%
opt_tau = sympy.solve(P_0 - P_1, tau)
sympy.solve(opt_tau[0] - 1, C)

# %%
# so C must be greater than 2 (1 - \eta)
