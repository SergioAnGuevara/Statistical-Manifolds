
import numpy as np


def metrica_fisher(theta):
    """ecuación (3.3))."""
    _, s = theta
    return np.diag([1.0, 2.0]) / s**2


def distancia_fisher_rao(m1, s1, m2, s2):
    return np.sqrt(2.0) * np.arccosh(
        1.0 + ((m1 - m2) ** 2 / 2.0 + (s1 - s2) ** 2) / (2.0 * s1 * s2))


def distancia_euclidiana(m1, s1, m2, s2):
    return np.hypot(m1 - m2, s1 - s2)


def emv(muestra):
    """Estimador de máxima verosimilitud."""
    muestra = np.asarray(muestra, float)
    return muestra.mean(), muestra.std(ddof=0)


def e_geodesica(theta_a, theta_b, t):
    """Geodésica de alpha = 1: segmento de recta en los parámetros naturales (Proposición 3.4), escrita en coordenadas (mu, sigma)."""
    def a_natural(m, s):
        return np.array([m / s**2, -1.0 / (2.0 * s**2)])
    na, nb = a_natural(*theta_a), a_natural(*theta_b)
    n = na[None, :] + np.asarray(t)[:, None] * (nb - na)[None, :]
    s = np.sqrt(-1.0 / (2.0 * n[:, 1]))
    return np.column_stack([n[:, 0] * s**2, s])
