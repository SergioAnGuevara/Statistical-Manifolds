# código 

Código de la aplicación computacional (capítulo 4) del documento


## Contenido



| `variedades/estructura.py` | Cálculo simbólico (SymPy) de la métrica de Fisher, el tensor de Amari–Chentsov, los símbolos de las α-conexiones y la curvatura gaussiana, a partir de una densidad. : (3.1), (3.2), (2.4), Christoffel y curvatura del cap. 1 .

| `variedades/geodesicas.py` | Integración numérica de la ecuación geodésica y método de disparo para unir dos puntos; longitud de curvas. | (2.5) .

| `variedades/normal.py` | Familia normal: métrica, distancia de Fisher–Rao en forma cerrada, estimador de máxima verosimilitud y e-geodésica exacta. | (3.3), Prop. 3.4 .


| `aplicacion.py` | Ejecuta la aplicación completa y genera las figuras y los resultados. | Cap. 4 |

## Uso

```bash
pip install -r requirements.txt
python aplicacion.py
```

El script escribe en `salida/`:

- `fig-alfa-geodesicas.pdf`:  (figura 4.1).
- `fig-muestras.pdf`:  (figura 4.2).
- `resultados.txt`: todos los valores numéricos citados en el texto.

Las muestras simuladas usan la semilla 2026.

