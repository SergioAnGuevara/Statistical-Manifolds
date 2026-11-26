
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import font_manager

from variedades.estructura import (christoffel_alfa, christoffel_alfa_esperanza,
                                   curvatura_gaussiana, estructura_estadistica)
from variedades.geodesicas import geodesica_frontera, longitud
from variedades import normal

SALIDA = Path(__file__).parent / "salida"
SALIDA.mkdir(exist_ok=True)
SEMILLA = 2026
informe = []


def anotar(texto=""):
    print(texto)
    informe.append(texto)


# fuente
for f in font_manager.findSystemFonts():
    if "texgyrepagella" in f.lower():
        font_manager.fontManager.addfont(f)
if any("Pagella" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams.update({"font.family": "TeX Gyre Pagella",
                         "mathtext.fontset": "custom",
                         "mathtext.rm": "TeX Gyre Pagella",
                         "mathtext.it": "TeX Gyre Pagella:italic",
                         "mathtext.cal": "TeX Gyre Pagella"})
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
COLORES = {0: "#1f3b73", 1: "#c0392b", -1: "#2e8b57"}


x = sp.Symbol("x", real=True)
mu = sp.Symbol("mu", real=True)
sigma = sp.Symbol("sigma", positive=True)
alpha = sp.Symbol("alpha", real=True)
params = [mu, sigma]
densidad = sp.exp(-(x - mu) ** 2 / (2 * sigma**2)) / (sp.sqrt(2 * sp.pi) * sigma)

g, T = estructura_estadistica(densidad, x, params, (-sp.oo, sp.oo))
gam_bajo, gam_alto = christoffel_alfa(g, T, params, alpha)
gam_esp = christoffel_alfa_esperanza(densidad, x, params, (-sp.oo, sp.oo), alpha)
K = curvatura_gaussiana(g, params)

anotar("== 4.2 Estructura estadística de la familia normal ==")
anotar(f"g = {g.tolist()}")
anotar(f"T no nulos = { {k: v for k, v in T.items() if v != 0} }")
anotar("Gamma^(alpha)_{ij,k} (i,j,k en {1=mu, 2=sigma}):")
for (i, j, k), v in sorted(gam_bajo.items()):
    if i <= j:
        anotar(f"  Gamma_{{{i+1}{j+1},{k+1}}} = {sp.factor(v)}")
coinciden = all(sp.simplify(gam_bajo[c] - gam_esp[c]) == 0 for c in gam_bajo)
anotar(f"Coincide con la fórmula en esperanzas (Prop. 3.3): {coinciden}")
anotar(f"Curvatura gaussiana K = {K}")

gamma_num = {}
for a in (0, 1, -1):
    G = sp.lambdify((mu, sigma), [[[gam_alto[(k, i, j)].subs(alpha, a)
                                    for j in range(2)] for i in range(2)]
                                  for k in range(2)], "numpy")
    gamma_num[a] = lambda th, G=G: np.array(G(*th), dtype=float)
dentro = lambda th: th[1]          # σ > 0


# 4.3  Geodesicas y distancia de Fisher–Rao

anotar("\n== 4.3 Geodésicas y validación ==")
pares_fig1 = [((-2.0, 1.0), (2.0, 1.0)), ((-1.0, 0.5), (1.5, 2.0))]
fig, ejes = plt.subplots(1, 2, figsize=(6.3, 2.6))
for eje, (A, B), letra in zip(ejes, pares_fig1, "ab"):
    for a, nombre in ((0, r"$\alpha=0$ (Levi-Civita)"),
                      (1, r"$\alpha=1$ (exponencial)"),
                      (-1, r"$\alpha=-1$ (mezcla)")):
        t, th, v = geodesica_frontera(gamma_num[a], A, B, dominio=dentro)
        L = longitud(normal.metrica_fisher, t, th, v)
        eje.plot(th[:, 0], th[:, 1], color=COLORES[a], lw=1.6, label=nombre)
        extra = ""
        if a == 0:
            d = normal.distancia_fisher_rao(*A, *B)
            extra = f" | fórmula cerrada = {d:.10f} | error = {abs(L - d):.1e}"
        if a == 1:
            err = np.max(np.abs(th - normal.e_geodesica(A, B, t)))
            extra = f" | desv. máx. respecto a la recta en param. naturales = {err:.1e}"
        anotar(f"({letra}) A={A}, B={B}, alpha={a:+d}: longitud = {L:.10f}{extra}")
    eje.plot(*zip(A, B), "ko", ms=4, zorder=5)
    eje.annotate("A", A, textcoords="offset points", xytext=(-9, -10))
    eje.annotate("B", B, textcoords="offset points", xytext=(4, -10))
    eje.set_xlabel(r"$\mu$")
    eje.set_ylabel(r"$\sigma$")
    eje.set_ylim(bottom=0)
    eje.set_title(f"({letra})", loc="left")
ejes[0].legend(frameon=False, fontsize=8, loc="lower center")
fig.savefig(SALIDA / "fig-alfa-geodesicas.pdf")
plt.close(fig)


# 4.4  Comparacion de distribuciones

anotar("\n== 4.4 Ejemplo de la introducción ==")
verdaderas = {"A": (0.0, 1.0), "B": (1.0, 1.0), "C": (0.0, 0.1), "D": (1.0, 0.1)}
for p, q in (("A", "B"), ("C", "D")):
    P, Q = verdaderas[p], verdaderas[q]
    t, th, v = geodesica_frontera(gamma_num[0], P, Q, dominio=dentro)
    anotar(f"{p}-{q}: euclidiana = {normal.distancia_euclidiana(*P, *Q):.4f} | "
           f"Fisher-Rao cerrada = {normal.distancia_fisher_rao(*P, *Q):.4f} | "
           f"disparo = {longitud(normal.metrica_fisher, t, th, v):.4f}")

anotar("\n== 4.4 Muestras simuladas (n = 100, semilla %d) ==" % SEMILLA)
rng = np.random.default_rng(SEMILLA)
ajustes = {k: normal.emv(rng.normal(m, s, size=100)) for k, (m, s) in verdaderas.items()}
for k, (m, s) in ajustes.items():
    anotar(f"{k}: verdadera {verdaderas[k]} -> EMV (mu, sigma) = ({m:.3f}, {s:.3f})")
anotar("Par | euclidiana | Fisher-Rao")
nombres = list(ajustes)
for i in range(4):
    for j in range(i + 1, 4):
        P, Q = ajustes[nombres[i]], ajustes[nombres[j]]
        anotar(f"{nombres[i]}-{nombres[j]} | {normal.distancia_euclidiana(*P, *Q):.3f} | "
               f"{normal.distancia_fisher_rao(*P, *Q):.3f}")

fig, eje = plt.subplots(figsize=(4.2, 2.8))
for i in range(4):
    for j in range(i + 1, 4):
        P, Q = ajustes[nombres[i]], ajustes[nombres[j]]
        t, th, v = geodesica_frontera(gamma_num[0], P, Q, dominio=dentro)
        eje.plot(th[:, 0], th[:, 1], color=COLORES[0], lw=1.0, alpha=0.75)
for k, (m, s) in ajustes.items():
    eje.plot(m, s, "o", color=COLORES[1], ms=5, zorder=5)
    eje.annotate(k, (m, s), textcoords="offset points", xytext=(5, 3))
eje.set_xlabel(r"$\mu$")
eje.set_ylabel(r"$\sigma$")
eje.set_ylim(bottom=0)
fig.savefig(SALIDA / "fig-muestras.pdf")
plt.close(fig)

(SALIDA / "resultados.txt").write_text("\n".join(informe), encoding="utf-8")
print(f"\nFiguras y resultados guardados en {SALIDA}/")
