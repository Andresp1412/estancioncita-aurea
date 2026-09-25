"""
Prueba de extremo a extremo del ciclo de retroalimentación.

Genera registros con el MISMO formato de la API de AmbientWeather (cada 5 min)
a partir de 10 días del reanálisis (marzo 2026) y los entrega a actualizar.py
en lotes de 6 h, como si fueran ejecuciones sucesivas de GitHub Actions.
Se ejecuta en una copia temporal del repositorio: no toca data/.

    python tests/simular_estacion.py
"""
import json, os, shutil, subprocess, sys, tempfile
import pandas as pd, numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp(prefix='aurea_sim_')
shutil.copytree(RAIZ, os.path.join(tmp, 'repo'), ignore=shutil.ignore_patterns('.git', 'node_modules'))
R = os.path.join(tmp, 'repo')
for f in ('observaciones.csv', 'predicciones.csv', 'estado_modelo.json', 'estado.json'):
    p = os.path.join(R, 'data', f)
    if os.path.exists(p):
        os.remove(p)

e = pd.read_csv(os.path.join(RAIZ, 'data', 'era5_duitama.csv'), index_col=0, parse_dates=True).loc['2026-03-01':'2026-03-10 23:00']
idx5 = pd.date_range(e.index[0], e.index[-1], freq='5min')
X = e[['T', 'RH', 'P']].reindex(idx5).interpolate()
G = e['G'].reindex(idx5).bfill()                              # G(H) es la media de (H-1,H] → constante en ese intervalo
PRh = e['PR'].reindex(idx5).bfill().fillna(0) / 12           # lluvia de (H-1,H] repartida en sus 12 pasos de 5 min
tot = PRh.cumsum()
regs = []
for t in idx5:
    utc = (t + pd.Timedelta(hours=5)).tz_localize('UTC')
    Tf = X.loc[t, 'T'] * 9 / 5 + 32
    regs.append({'dateutc': int(utc.value // 10**6), 'date': utc.isoformat(), 'tempf': round(Tf, 1), 'humidity': int(round(X.loc[t, 'RH'])),
                 'baromrelin': round(X.loc[t, 'P'] / 33.8639, 3), 'solarradiation': round(float(G.loc[t]), 2),
                 'hourlyrainin': round(float(PRh.loc[t]) * 12 / 25.4, 3), 'totalrainin': round(float(tot.loc[t]) / 25.4, 4),
                 'dailyrainin': 0, 'windspeedmph': 1.0, 'windgustmph': 2.0, 'winddir': 90, 'uv': 1, 'tempinf': round(Tf + 5, 1)})

lote = 72  # 6 h
for k in range(0, len(regs), lote):
    fx = os.path.join(tmp, 'lote.json'); json.dump(regs[k:k + lote], open(fx, 'w'))
    out = subprocess.run([sys.executable, os.path.join(R, 'scripts', 'actualizar.py'), '--fixture', fx], capture_output=True, text=True)
    if out.returncode:
        print(out.stderr); sys.exit(1)
print(out.stdout.strip())
est = json.load(open(os.path.join(R, 'data', 'estado.json')))
v = est['verificacion']['total']
print(json.dumps({'verificacion': v, 'modelo': est['modelo'], 'actual': {k: est['prediccion_actual'].get(k) for k in ('hora', 'p_lluvia', 'banda', 'pv_corr')}}, indent=1, ensure_ascii=False))
assert v['lluvia']['n'] > 200 and est['modelo']['lluvia']['actualizaciones'] == v['lluvia']['n'], 'cada verificación debe actualizar el modelo'
# Paridad: la predicción emitida por la tubería debe coincidir con la calculada
# directamente sobre el reanálisis horario (mismas variables, mismo modelo base).
sys.path.insert(0, os.path.join(RAIZ, 'scripts')); import modelo as M
import csv
preds = list(csv.DictReader(open(os.path.join(R, 'data', 'predicciones.csv'))))
ml0 = json.load(open(os.path.join(RAIZ, 'data', 'modelo_lluvia.json'))); mf = json.load(open(os.path.join(RAIZ, 'data', 'modelo_fv.json')))
serie = [{'hora': t.hour, 'mes': t.month, 'T': r.T, 'RH': r.RH, 'P': r.P, 'G': r.G, 'rain': r.PR} for t, r in zip(e.index, e.itertuples())]
pos = {t.isoformat(): i for i, t in enumerate(e.index)}
dif_pv = []
for p in preds:
    i = pos[p['hora'][:19]]
    x = M.features(serie, i); dif_pv.append(abs(M.pv_pronostico(mf, x) - float(p['pv_hat'])))
print('diferencia pv_hat tubería vs directo: mediana', round(float(np.median(dif_pv)), 5), 'máx', round(max(dif_pv), 5), 'kW en', len(dif_pv), 'horas')
# la API redondea (°F a 0,1; HR entera; inHg a 0,001) → pequeñas diferencias en los cortes de los árboles
assert np.median(dif_pv) < 1e-3 and max(dif_pv) < 0.02
# la primera predicción se emite con el modelo base sin actualizar
p0 = preds[0]; x0 = M.features(serie, pos[p0['hora'][:19]])
print('p_lluvia primera hora tubería / directo:', p0['p_lluvia'], round(M.prob_lluvia(ml0, x0), 4))
assert abs(float(p0['p_lluvia']) - M.prob_lluvia(ml0, x0)) < 0.01   # redondeo de la API
print('OK · carpeta de simulación:', R)
