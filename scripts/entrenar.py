"""
Entrena y exporta los modelos de la web a partir del reanálisis histórico.

Uso:
    python scripts/entrenar.py            # usa data/era5_duitama.csv
    python scripts/entrenar.py --descargar  # vuelve a descargar de Open-Meteo

Salidas (carpeta data/):
    modelo_lluvia.json   regresión logística (base + estado actualizable)
    modelo_fv.json       Gradient Boosting exportado como árboles
    correccion_fv.json   estado inicial de la corrección en línea (RLS)
    validacion.json      métricas en el conjunto de prueba (ene–ago 2026)
    replay_2026.json     serie horaria de prueba para la reproducción en la web
"""
import argparse, json, math, os, sys, urllib.request
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
import modelo as M

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(RAIZ, 'data')
LAT, LON = 5.8277, -73.0251
URL = ('https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
       '&start_date=2024-01-01&end_date=2026-08-31'
       '&hourly=temperature_2m,relative_humidity_2m,pressure_msl,shortwave_radiation,precipitation'
       '&timezone=America%2FBogota')
CORTE = pd.Timestamp('2026-01-01')


def descargar():
    with urllib.request.urlopen(URL.format(lat=LAT, lon=LON), timeout=60) as r:
        j = json.load(r)['hourly']
    df = pd.DataFrame({'T': j['temperature_2m'], 'RH': j['relative_humidity_2m'], 'P': j['pressure_msl'],
                       'G': j['shortwave_radiation'], 'PR': j['precipitation']}, index=pd.to_datetime(j['time']))
    df.index.name = 'time'
    df.to_csv(os.path.join(D, 'era5_duitama.csv'))
    return df


def serie_desde_df(df):
    return [{'hora': t.hour, 'mes': t.month, 'T': float(r.T), 'RH': float(r.RH), 'P': float(r.P),
             'G': float(r.G), 'rain': float(r.PR)} for t, r in zip(df.index, df.itertuples())]


def objetivos(serie):
    n = len(serie); y_ll = [None] * n; y_fv = [None] * n
    for i in range(n):
        if i + M.HORIZONTE_LLUVIA_H < n:
            acc = sum(serie[i + k]['rain'] for k in range(1, M.HORIZONTE_LLUVIA_H + 1))
            y_ll[i] = 1 if acc >= M.UMBRAL_LLUVIA_MM else 0
        if i + 1 < n:
            y_fv[i] = M.pv_kw(serie[i + 1]['G'], serie[i + 1]['T'])
    return y_ll, y_fv


def exportar_gb(gb, feats):
    trees = []
    for est in gb.estimators_[:, 0]:
        t = est.tree_
        trees.append({'f': t.feature.tolist(), 't': [round(float(v), 6) for v in t.threshold],
                      'l': t.children_left.tolist(), 'r': t.children_right.tolist(),
                      'v': [round(float(v), 6) for v in t.value[:, 0, 0]]})
    init = float(gb.init_.constant_[0][0]) if hasattr(gb.init_, 'constant_') else float(gb._raw_predict_init(np.zeros((1, len(feats))))[0][0])
    return {'features': feats, 'init': round(init, 6), 'lr': gb.learning_rate, 'trees': trees}


def clasif(y, yhat):
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1); fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0); tn = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'vp': tp, 'fp': fp, 'fn': fn, 'vn': tn, 'exactitud': (tp + tn) / len(y), 'precision': prec,
            'sensibilidad': rec, 'f1': f1, 'csi': tp / (tp + fp + fn) if tp + fp + fn else 0.0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--descargar', action='store_true'); a = ap.parse_args()
    df = descargar() if a.descargar else pd.read_csv(os.path.join(D, 'era5_duitama.csv'), index_col=0, parse_dates=True)
    df = df[['T', 'RH', 'P', 'G', 'PR']]
    serie = serie_desde_df(df); y_ll, y_fv = objetivos(serie)
    X = [M.features(serie, i) for i in range(len(serie))]
    idx = [i for i in range(len(serie)) if X[i] is not None and y_ll[i] is not None and y_fv[i] is not None]
    tr = [i for i in idx if df.index[i] < CORTE]; te = [i for i in idx if df.index[i] >= CORTE]

    # ---------- Lluvia ----------
    A = lambda ii, F: np.array([[X[i][k] for k in F] for i in ii])
    sc = StandardScaler().fit(A(tr, M.FEAT_LLUVIA))
    lr = LogisticRegression(max_iter=3000, class_weight='balanced').fit(sc.transform(A(tr, M.FEAT_LLUVIA)), [y_ll[i] for i in tr])
    ml = {'version': '1.0', 'entrenado_con': 'Reanálisis ERA5 (Open-Meteo) 2024-01-01 a 2025-12-31, 5,8277 N; -73,0251 O',
          'features': M.FEAT_LLUVIA, 'mu': [round(float(v), 6) for v in sc.mean_], 'sd': [round(float(v), 6) for v in sc.scale_],
          'w0': [round(float(v), 6) for v in lr.coef_[0]], 'b0': round(float(lr.intercept_[0]), 6)}
    prev = float(np.mean([y_ll[i] for i in tr]))
    ml['w'] = list(ml['w0']); ml['b'] = ml['b0']; ml['eta'] = 0.01; ml['lambda'] = 0.05; ml['actualizaciones'] = 0
    ml['peso_clase'] = [round(1 / (2 * (1 - prev)), 6), round(1 / (2 * prev), 6)]   # [clase 0, clase 1], igual que class_weight='balanced'
    ptr = [M.prob_lluvia(ml, X[i]) for i in tr]; ytr = [y_ll[i] for i in tr]
    ths = [round(0.05 + 0.01 * k, 2) for k in range(91)]
    ml['umbral'] = max(ths, key=lambda t: f1_score(ytr, [1 if p >= t else 0 for p in ptr]))

    # ---------- FV ----------
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(A(tr, M.FEAT_FV), [y_fv[i] for i in tr])
    mf = exportar_gb(gb, M.FEAT_FV); mf['version'] = '1.0'; mf['entrenado_con'] = ml['entrenado_con']
    Xf = A(te, M.FEAT_FV); chk = [M.pv_pronostico(mf, X[i]) for i in te[:200]]
    assert max(abs(a - max(0, b)) for a, b in zip(chk, gb.predict(Xf[:200]))) < 1e-4, 'exportación GB no coincide'
    cf = {'theta': [0.0, 1.0], 'P': [[0.05, 0.0], [0.0, 0.5]], 'olvido': 0.995, 'actualizaciones': 0}

    # ---------- Validación (con las MISMAS funciones que usa la web) ----------
    yte = [y_ll[i] for i in te]
    p_lr = [M.prob_lluvia(ml, X[i]) for i in te]
    p_web = [min(100, max(0, (X[i]['RH'] - 40) / 60 * 60 + max(0, -X[i]['dP1']) * 8)) / 100 for i in te]
    persist = [int(X[i]['rain_now']) for i in te]
    V = {'periodo_prueba': [str(df.index[te[0]]), str(df.index[te[-1]])], 'n_horas': len(te),
         'prevalencia_lluvia': sum(yte) / len(yte), 'umbral_modelo': ml['umbral'], 'lluvia': {}, 'fv': {}}
    V['lluvia']['Regla anterior de la web (≥ 25 %)'] = {**clasif(yte, [1 if p >= .25 else 0 for p in p_web]), 'auc': roc_auc_score(yte, p_web)}
    V['lluvia']['Persistencia (llueve ahora)'] = clasif(yte, persist)
    V['lluvia']['Regresión logística (modelo de la web)'] = {**clasif(yte, [1 if p >= ml['umbral'] else 0 for p in p_lr]), 'auc': roc_auc_score(yte, p_lr)}
    dia = [i for i in te if 6 <= (df.index[i] + pd.Timedelta(hours=1)).hour < 18]
    yv = np.array([y_fv[i] for i in dia])
    trd = [i for i in tr]
    clim = pd.Series([y_fv[i] for i in trd], index=[(df.index[i] + pd.Timedelta(hours=1)).month * 100 + (df.index[i] + pd.Timedelta(hours=1)).hour for i in trd]).groupby(level=0).mean()
    pc = np.array([clim.get((df.index[i] + pd.Timedelta(hours=1)).month * 100 + (df.index[i] + pd.Timedelta(hours=1)).hour, 0.0) for i in dia])
    pg = np.array([M.pv_pronostico(mf, X[i]) for i in dia]); pp = np.array([X[i]['PV'] for i in dia])
    reg = lambda p: {'rmse': float(np.sqrt(np.mean((yv - p) ** 2))), 'mae': float(np.mean(np.abs(yv - p))),
                     'r2': float(1 - np.sum((yv - p) ** 2) / np.sum((yv - yv.mean()) ** 2))}
    V['fv']['Persistencia'] = reg(pp); V['fv']['Climatología mes-hora'] = reg(pc); V['fv']['Gradient Boosting (modelo de la web)'] = reg(pg)
    V['fv']['n_horas_diurnas'] = len(dia)
    V['fv']['mejora_vs_persistencia_pct'] = 100 * (1 - V['fv']['Gradient Boosting (modelo de la web)']['rmse'] / V['fv']['Persistencia']['rmse'])
    V['fv']['mejora_vs_climatologia_pct'] = 100 * (1 - V['fv']['Gradient Boosting (modelo de la web)']['rmse'] / V['fv']['Climatología mes-hora']['rmse'])
    V['heladas'] = {'dias': int(len(df['T'].resample('D').min())), 'tmin_absoluta': float(df['T'].min()),
                    'horas_bajo_0': int((df['T'] < 0).sum()), 'horas_bajo_5': int((df['T'] < 5).sum())}
    # Misma prueba, pero dejando que el modelo aprenda de cada hora verificada (flujo cronológico)
    import copy
    mo = copy.deepcopy(ml); p_on = {}
    for i in range(te[0], te[-1] + 1):
        if X[i] is not None:
            p_on[i] = M.prob_lluvia(mo, X[i])
        j = i - M.HORIZONTE_LLUVIA_H
        if j in p_on and y_ll[j] is not None and X[j] is not None and j >= te[0]:
            M.actualizar_lluvia(mo, X[j], y_ll[j])
    pon = [p_on[i] for i in te]
    V['lluvia']['Regresión logística + aprendizaje en línea'] = {**clasif(yte, [1 if p >= ml['umbral'] else 0 for p in pon]), 'auc': roc_auc_score(yte, pon)}
    V['fuente'] = URL.format(lat=LAT, lon=LON)

    # ---------- Serie para reproducción en la web ----------
    ini = te[0] - 3
    sub = df.iloc[ini:]
    R = {'t0': str(sub.index[0]), 'paso_h': 1, 'n': len(sub),
         'T': [round(float(v), 1) for v in sub['T']], 'RH': [int(round(v)) for v in sub['RH']],
         'P': [round(float(v), 1) for v in sub['P']], 'G': [int(round(v)) for v in sub['G']],
         'PR': [round(float(v), 2) for v in sub['PR']]}

    for nombre, obj in [('modelo_lluvia.json', ml), ('modelo_fv.json', mf), ('validacion.json', V), ('replay_2026.json', R)]:
        with open(os.path.join(D, nombre), 'w') as f:
            json.dump(obj, f, ensure_ascii=False, separators=(',', ':') if nombre in ('modelo_fv.json', 'replay_2026.json') else None, indent=None if nombre in ('modelo_fv.json', 'replay_2026.json') else 1)
    if not os.path.exists(os.path.join(D, 'correccion_fv.json')):
        json.dump(cf, open(os.path.join(D, 'correccion_fv.json'), 'w'), indent=1)
    print(json.dumps({k: V[k] for k in ('lluvia', 'fv', 'heladas')}, indent=1, ensure_ascii=False))
    print('umbral', ml['umbral'])


if __name__ == '__main__':
    main()
