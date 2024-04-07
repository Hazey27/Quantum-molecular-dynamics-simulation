# Hazel Dellario
# 2024.03.23
# Attempting to solve differential equation as laid out in this paper: https://www.nature.com/articles/s41598-022-21163-x

import sympy


# System of equations:
#   dr/dt = p/mu
#   dp/dt = F(r)
# where r is the function r(t)
# r is the distance between two covalently bonded hydrogen atoms and p is the momentum of the bond energy (?)
# this code uses the Hamiltonian (above) and the Morse equations (below) to estimate the dynamics of H2 at a quantum level


t = sympy.Symbol('t')
p = sympy.Function('p')(t)

a, D_e, r_e, r_0, mu = sympy.symbols('a D_e r_e r_0 mu')

c1 = sympy.exp(a*r_e)
c2 = sympy.exp(a*r_0)
c3 = -1 * D_e * (c1 / c2) * (2 - (c1 / c2))
c4 = D_e + ((c2 * c3) / c1)
tau = sympy.exp(sympy.sqrt((2 * c3) / mu) * a * t)

r = sympy.Function('r')(t)
f = sympy.Function('f')(r)

r_t = (1 / a) * sympy.log((c1**2) * tau * ((c3 * D_e + (D_e - (c4 / tau)) ** 2) / 2 * c1 * c3 * c4))
f = 2 * a * D_e * (sympy.exp(-2*a*(r - r_e)) - sympy.exp(-a * (r - r_e)))

# ANALYTICAL SOLUTION DOES NOT WORK :sob:
# diffeq = sympy.Eq(sympy.diff(p, t), f.subs({r: r_t}))
# p_t = sympy.dsolve(diffeq, p)
#
# print(p_t)


# TIME TO DO THIS AS A QAOA BITCHES


