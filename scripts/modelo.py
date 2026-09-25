"""
Estancioncita Aurea — núcleo común de modelos.

Este módulo lo usan tanto el entrenamiento (entrenar.py) como la
actualización periódica (actualizar.py). La página web (index.html)
implementa en JavaScript exactamente las mismas funciones; tests/paridad.py
comprueba que ambas implementaciones dan el mismo resultado.

Convenciones
------------
* Serie horaria en hora local de Bogotá (UTC-5, sin horario de verano).
* En la hora H:  T, HR, P  = valor instantáneo en H
                 G         = irradiancia media de la hora (H-1, H]
                 lluvia    = acumulado (mm) en (H-1, H]
* Objetivo lluvia : acumulado en (H, H+3h] >= 0,2 mm
* Objetivo FV     : potencia de 1 kWp en H+1 con el modelo físico
"""
import math

NOCT = 45.0          # °C, temperatura nominal de operación de la celda
GAMMA = -0.004       # 1/°C, coeficiente de temperatura de potencia
P_NOM_KW = 1.0       # kWp del sistema de referencia
UMBRAL_LLUVIA_MM = 0.2
HORIZONTE_LLUVIA_H = 3

FEAT_LLUVIA = ['T', 'RH', 'P', 'G', 'dP1', 'dP3', 'dRH1', 'dT1', 'hs', 'hc', 'ms', 'mc', 'rain_now']
FEAT_FV = ['PV', 'G', 'T', 'RH', 'dP1', 'dG1', 'hs', 'hc', 'ms', 'mc']


def pv_kw(G, T):
    """Potencia de un sistema de 1 kWp (modelo NOCT + coeficiente de temperatura)."""
    if G is None or T is None:
        return None
    t_celda = T + (NOCT - 20.0) / 800.0 * G
    return max(0.0, P_NOM_KW * (G / 1000.0) * (1.0 + GAMMA * (t_celda - 25.0)))


def features(serie, i):
    """Variables del modelo en el índice i de una serie horaria.

    serie: lista de dicts {'hora': int, 'mes': int, 'T','RH','P','G','rain'} (None si falta).
    Devuelve dict o None si falta algún dato necesario.
    """
    if i < 3:
        return None
    a, b1, b3 = serie[i], serie[i - 1], serie[i - 3]
    need = [a['T'], a['RH'], a['P'], a['G'], a['rain'], b1['P'], b1['RH'], b1['T'], b1['G'], b3['P']]
    if any(v is None for v in need):
        return None
    h, m = a['hora'], a['mes']
    return {
        'T': a['T'], 'RH': a['RH'], 'P': a['P'], 'G': a['G'],
        'dP1': a['P'] - b1['P'], 'dP3': a['P'] - b3['P'],
        'dRH1': a['RH'] - b1['RH'], 'dT1': a['T'] - b1['T'], 'dG1': a['G'] - b1['G'],
        'hs': math.sin(2 * math.pi * h / 24), 'hc': math.cos(2 * math.pi * h / 24),
        'ms': math.sin(2 * math.pi * m / 12), 'mc': math.cos(2 * math.pi * m / 12),
        'rain_now': 1.0 if a['rain'] >= UMBRAL_LLUVIA_MM else 0.0,
        'PV': pv_kw(a['G'], a['T']),
    }


# ---------------- Lluvia: regresión logística con actualización en línea ----------------
def _z(m, x):
    return [(x[k] - mu) / sd for k, mu, sd in zip(m['features'], m['mu'], m['sd'])]


def prob_lluvia(m, x):
    z = _z(m, x)
    s = m['b'] + sum(w * zi for w, zi in zip(m['w'], z))
    return 1.0 / (1.0 + math.exp(-s))


def contribuciones_lluvia(m, x):
    z = _z(m, x)
    return {k: w * zi for k, w, zi in zip(m['features'], m['w'], z)}


def actualizar_lluvia(m, x, y):
    """Un paso de descenso de gradiente con regularización hacia el modelo base.

    w <- w - eta * [ c_y (p - y) z + lambda (w - w0) ]
    c_y es el peso de clase usado en el entrenamiento (clases balanceadas), para que la
    calibración de las probabilidades y el umbral sigan siendo coherentes.
    Así el modelo se adapta a la estación sin olvidar lo aprendido en 32 meses.
    """
    eta, lam = m['eta'], m['lambda']
    z = _z(m, x)
    p = prob_lluvia(m, x)
    g = (p - y) * m['peso_clase'][int(y)]
    m['w'] = [w - eta * (g * zi + lam * (w - w0)) for w, zi, w0 in zip(m['w'], z, m['w0'])]
    m['b'] = m['b'] - eta * (g + lam * (m['b'] - m['b0']))
    m['actualizaciones'] = m.get('actualizaciones', 0) + 1
    return p


# ---------------- FV: Gradient Boosting exportado + corrección RLS ----------------
def _arbol(t, x):
    n = 0
    f, th, l, r, v = t['f'], t['t'], t['l'], t['r'], t['v']
    while l[n] != -1:
        n = l[n] if x[f[n]] <= th[n] else r[n]
    return v[n]


def pv_pronostico(m, x):
    """Pronóstico de potencia (kW) en H+1."""
    xs = [x[k] for k in m['features']]
    s = m['init'] + m['lr'] * sum(_arbol(t, xs) for t in m['trees'])
    return max(0.0, s)


def pv_corregido(c, yhat):
    a, b = c['theta']
    return max(0.0, a + b * yhat)


def actualizar_fv(c, yhat, y):
    """Mínimos cuadrados recursivos con olvido: y ≈ a + b·ŷ (ajuste a la estación real)."""
    lam = c['olvido']
    phi = [1.0, yhat]
    P = c['P']
    Pphi = [P[0][0] * phi[0] + P[0][1] * phi[1], P[1][0] * phi[0] + P[1][1] * phi[1]]
    den = lam + phi[0] * Pphi[0] + phi[1] * Pphi[1]
    k = [Pphi[0] / den, Pphi[1] / den]
    err = y - (c['theta'][0] * phi[0] + c['theta'][1] * phi[1])
    th = [c['theta'][0] + k[0] * err, c['theta'][1] + k[1] * err]
    th[0] = min(0.2, max(-0.2, th[0]))
    th[1] = min(1.5, max(0.5, th[1]))
    c['theta'] = th
    c['P'] = [[(P[i][j] - k[i] * Pphi[j]) / lam for j in range(2)] for i in range(2)]
    c['actualizaciones'] = c.get('actualizaciones', 0) + 1


def banda_lluvia(p):
    if p >= 0.66:
        return 'PROBABLE'
    if p >= 0.35:
        return 'POSIBLE'
    return 'IMPROBABLE'


def riesgo_helada(T, hora_local):
    noche = hora_local >= 21 or hora_local <= 7
    if T < 0:
        return 'alto'
    if noche and T < 5:
        return 'medio'
    return 'bajo'
