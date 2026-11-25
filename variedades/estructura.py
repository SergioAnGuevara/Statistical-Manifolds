"""Cálculo simbólico de la estructura estadística de un modelo paramétrico.

"""
from itertools import product

import sympy as sp


def esperanza(f, densidad, x, soporte):

    if isinstance(soporte, (list, set)):
        return sp.simplify(sum((f * densidad).subs(x, v) for v in soporte))
    return sp.simplify(sp.integrate(sp.expand(f * densidad), (x, *soporte)))


def estructura_estadistica(densidad, x, params, soporte):
    """Devuelve (g, T): métrica de Fisher (Matriz n×n) y tensor de
    Amari–Chentsov (dict {(i, j, k): expr}, simétrico)."""
    n = len(params)
    ell = sp.log(densidad)
    score = [sp.diff(ell, t) for t in params]
    E = lambda f: esperanza(f, densidad, x, soporte)

    g = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            g[i, j] = g[j, i] = E(score[i] * score[j])

    T = {}
    for i, j, k in product(range(n), repeat=3):
        clave = tuple(sorted((i, j, k)))
        if clave not in T:
            T[clave] = E(score[i] * score[j] * score[k])
    T = {(i, j, k): T[tuple(sorted((i, j, k)))]
         for i, j, k in product(range(n), repeat=3)}
    return g, T


def christoffel_alfa(g, T, params, alpha):
    """Símbolos de la alpha conexion """
    n = len(params)
    gi = g.inv()
    bajo = {}
    for i, j, k in product(range(n), repeat=3):
        lc = sp.Rational(1, 2) * (sp.diff(g[j, k], params[i])
                                  + sp.diff(g[i, k], params[j])
                                  - sp.diff(g[i, j], params[k]))
        bajo[(i, j, k)] = sp.simplify(lc - alpha / 2 * T[(i, j, k)])
    alto = {(k, i, j): sp.simplify(sum(gi[k, l] * bajo[(i, j, l)] for l in range(n)))
            for k, i, j in product(range(n), repeat=3)}
    return bajo, alto


def christoffel_alfa_esperanza(densidad, x, params, soporte, alpha):

    n = len(params)
    ell = sp.log(densidad)
    s = [sp.diff(ell, t) for t in params]
    return {(i, j, k): esperanza((sp.diff(ell, params[i], params[j])
                                  + (1 - alpha) / 2 * s[i] * s[j]) * s[k],
                                 densidad, x, soporte)
            for i, j, k in product(range(n), repeat=3)}


def curvatura_gaussiana(g, params):
    """Curvatura gaussiana de la conexión de Levi-Civita (dimensión 2):
    K = R_122^l g_l1 / det g, con R_ijk^l dado por la fórmula en componentes."""
    _, G = christoffel_alfa(g, {k: 0 for k in product(range(2), repeat=3)},
                            params, 0)

    def R(i, j, k, l):
        return (sp.diff(G[(l, j, k)], params[i]) - sp.diff(G[(l, i, k)], params[j])
                + sum(G[(m, j, k)] * G[(l, i, m)] - G[(m, i, k)] * G[(l, j, m)]
                      for m in range(2)))

    return sp.simplify(sum(R(0, 1, 1, l) * g[l, 0] for l in range(2)) / g.det())
