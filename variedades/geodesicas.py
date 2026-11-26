"""Geodesicas de una conexión afín y distancia de Fisher–Rao."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


def _sistema(gamma_alto):
    """gamma_alto(θ) -> arreglo G[k, i, j] = Γ^k_ij(θ)."""
    def f(t, y):
        n = y.size // 2
        th, v = y[:n], y[n:]
        acc = -np.einsum("kij,i,j->k", gamma_alto(th), v, v)
        return np.concatenate([v, acc])
    return f


def geodesica_inicial(gamma_alto, theta0, v0, t_final=1.0, n_puntos=400,
                      dominio=None):
    """Problema de valor inicial. `dominio(θ)` > 0 dentro del dominio de las
    coordenadas; si se anula, la integración se detiene."""
    eventos = None
    if dominio is not None:
        def salida(t, y):
            return dominio(y[: y.size // 2])
        salida.terminal = True
        eventos = salida
    t = np.linspace(0.0, t_final, n_puntos)
    sol = solve_ivp(_sistema(gamma_alto), (0.0, t_final),
                    np.concatenate([theta0, v0]), t_eval=t, events=eventos,
                    rtol=1e-11, atol=1e-12)
    n = len(theta0)
    return sol.t, sol.y[:n].T, sol.y[n:].T, sol.status


def geodesica_frontera(gamma_alto, theta_a, theta_b, v_inicial=None,
                       dominio=None, n_puntos=400):
    """Método de disparo: geodésica con θ(0) = θ_A y θ(1) = θ_B.

    """
    theta_a, theta_b = np.asarray(theta_a, float), np.asarray(theta_b, float)
    recta = theta_b - theta_a
    perp = np.array([-recta[1], recta[0]]) if recta.size == 2 else np.zeros_like(recta)
    candidatos = [] if v_inicial is None else [np.asarray(v_inicial, float)]
    candidatos += [recta + c * perp for c in (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0)]

    def residuo(v):
        _, th, _, estado = geodesica_inicial(gamma_alto, theta_a, v, 1.0, 2, dominio)
        if estado != 0:          # la curva salió del dominio antes de t = 1
            return 1e3 * np.ones_like(v)
        return th[-1] - theta_b

    for v0 in candidatos:
        res = root(residuo, v0, method="hybr", tol=1e-12)
        if res.success and np.max(np.abs(residuo(res.x))) < 1e-9:
            t, th, v, _ = geodesica_inicial(gamma_alto, theta_a, res.x, 1.0,
                                            n_puntos, dominio)
            return t, th, v
    raise RuntimeError("El método de disparo no convergió.")


def longitud(metrica, t, th, v):
    """Longitud ∫ sqrt(g_ij θ'^i θ'^j) dt de una curva muestreada."""
    rapidez = np.array([np.sqrt(vi @ metrica(xi) @ vi) for xi, vi in zip(th, v)])
    return np.trapezoid(rapidez, t)
