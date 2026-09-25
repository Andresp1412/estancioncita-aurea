"""
Estancioncita Aurea — ciclo de retroalimentación.

Lo ejecuta GitHub Actions cada 15 minutos (.github/workflows/actualizar.yml).
Pasos:
  1. Descarga los últimos registros de la WS-2902 (API AmbientWeather, llaves en Secrets).
  2. Control de calidad de cada lectura.
  3. Construye la serie horaria (hora local de Bogotá).
  4. Recorre las horas nuevas EN ORDEN CRONOLÓGICO:
       a. emite la predicción de la hora H (lluvia 3 h y potencia FV a 1 h);
       b. verifica la predicción de lluvia emitida en H-3 y la de FV emitida en H-1;
       c. con cada verificación actualiza los modelos (aprendizaje en línea).
     Como solo se usan datos <= H, no hay fuga de información aunque se procese un atraso.
  5. Escribe data/estado.json, que es lo único que lee la página.

Uso local:
  python scripts/actualizar.py --fixture tests/api_fixture_2026-08-26.json
  AMBIENT_API_KEY=... AMBIENT_APP_KEY=... AMBIENT_MAC=... python scripts/actualizar.py
"""
import argparse, csv, json, math, os, sys, time, urllib.parse, urllib.request
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
import modelo as M

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(RAIZ, 'data')
BOG = timezone(timedelta(hours=-5))          # Colombia: UTC-5 todo el año
OBS = os.path.join(D, 'observaciones.csv')
PRED = os.path.join(D, 'predicciones.csv')
CAMPOS_OBS = ['dateutc', 'T', 'RH', 'P', 'G', 'rain_rate', 'rain_total', 'rain_day', 'wind', 'gust', 'wind_dir', 'uv', 'Tin', 'qc']
CAMPOS_PRED = ['hora', 'version', 'p_lluvia', 'banda', 'pv_hat', 'pv_corr', 'pv_persist', 'y_lluvia', 'pv_real', 'estado_lluvia', 'estado_fv']
TOL_MIN = 10          # tolerancia para el valor instantáneo en la hora exacta
MIN_LECT_HORA = 6     # lecturas válidas mínimas para promediar la radiación de una hora


# ---------------------------------------------------------------- utilidades
def f(x, nd=2):
    return None if x is None else round(float(x), nd)


def leer_json(nombre, defecto=None):
    p = os.path.join(D, nombre)
    if not os.path.exists(p):
        return defecto
    with open(p) as fh:
        return json.load(fh)


def escribir_json(nombre, obj, compacto=False):
    with open(os.path.join(D, nombre), 'w') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=None if compacto else 1, separators=(',', ':') if compacto else None)


def leer_csv(ruta):
    if not os.path.exists(ruta):
        return []
    with open(ruta, newline='') as fh:
        return list(csv.DictReader(fh))


def escribir_csv(ruta, campos, filas):
    with open(ruta, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        for r in filas:
            w.writerow({k: ('' if r.get(k) is None else r[k]) for k in campos})


def num(v):
    return None if v in (None, '') else float(v)


# ---------------------------------------------------------------- 1. ingesta
def descargar(limit=288, end_date=None):
    try:
        key, app, mac = os.environ['AMBIENT_API_KEY'], os.environ['AMBIENT_APP_KEY'], os.environ['AMBIENT_MAC']
    except KeyError as falta:
        sys.exit(f'Falta la variable {falta}. Configúrala en Settings → Secrets and variables → Actions del repositorio.')
    q = {'apiKey': key, 'applicationKey': app, 'limit': limit}
    if end_date:
        q['endDate'] = end_date
    url = f'https://api.ambientweather.net/v1/devices/{urllib.parse.quote(mac)}?' + urllib.parse.urlencode(q)
    for intento in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:                      # 429 por límite de 1 req/s u otro error transitorio
            print('reintento API:', e)
            time.sleep(2 + 2 * intento)
    raise RuntimeError('No se pudo consultar la API de AmbientWeather')


def convertir(r):
    g = lambda k: r.get(k)
    c = lambda v, fn: None if v is None else fn(v)
    return {
        'dateutc': int(g('dateutc')),
        'T': c(g('tempf'), lambda v: (v - 32) * 5 / 9),
        'RH': c(g('humidity'), float),
        'P': c(g('baromrelin'), lambda v: v * 33.8639),
        'G': c(g('solarradiation'), float),
        'rain_rate': c(g('hourlyrainin'), lambda v: v * 25.4),
        'rain_total': c(g('totalrainin') if g('totalrainin') is not None else g('eventrainin'), lambda v: v * 25.4),
        'rain_day': c(g('dailyrainin'), lambda v: v * 25.4),
        'wind': c(g('windspeedmph'), lambda v: v * 1.609344),
        'gust': c(g('windgustmph'), lambda v: v * 1.609344),
        'wind_dir': g('winddir'),
        'uv': g('uv'),
        'Tin': c(g('tempinf'), lambda v: (v - 32) * 5 / 9),
    }


# ---------------------------------------------------------------- 2. control de calidad
def control_calidad(obs):
    """Marca cada lectura. Devuelve la misma lista con el campo 'qc' (banderas separadas por '|')."""
    prev = None
    for o in obs:
        flags = []
        rangos = [('T', -15, 40), ('RH', 1, 100), ('P', 900, 1100), ('G', 0, 1600)]
        if any(o[k] is None or not (lo <= o[k] <= hi) for k, lo, hi in rangos):
            flags.append('fuera_de_rango')
        hora = datetime.fromtimestamp(o['dateutc'] / 1000, BOG).hour
        if o['Tin'] is not None and o['T'] is not None and o['G'] is not None:
            if abs(o['T'] - o['Tin']) < 0.6 and 8 <= hora <= 16 and o['G'] < 20:
                flags.append('posible_interior')
        if (o['rain_rate'] or 0) > 5 and o['RH'] is not None and o['RH'] < 60:
            flags.append('lluvia_sin_humedad')
        if prev and o['rain_total'] is not None and prev['rain_total'] is not None and o['rain_total'] < prev['rain_total'] - 1e-6:
            flags.append('reinicio_contador')
        if prev and o['dateutc'] - prev['dateutc'] > 30 * 60 * 1000:
            flags.append('hueco_previo')
        o['qc'] = '|'.join(flags)
        prev = o
    return obs


def valida(o):
    return not any(b in o['qc'] for b in ('fuera_de_rango', 'posible_interior'))


# ---------------------------------------------------------------- 3. serie horaria
def serie_horaria(obs):
    """Serie horaria (hora local). Solo incluye horas cerradas: H <= última lectura - 10 min."""
    if not obs:
        return []
    ts = [o['dateutc'] for o in obs]
    tol = TOL_MIN * 60 * 1000
    t0 = datetime.fromtimestamp(ts[0] / 1000, BOG).replace(minute=0, second=0, microsecond=0)
    t1 = datetime.fromtimestamp((ts[-1] - tol) / 1000, BOG).replace(minute=0, second=0, microsecond=0)
    serie, H = [], t0
    while H <= t1:
        hms = int(H.timestamp() * 1000); ini = hms - 3600 * 1000
        # valor instantáneo: lectura válida más cercana a H dentro de ±10 min
        cand = [obs[k] for k in range(bisect_left(ts, hms - tol), bisect_right(ts, hms + tol)) if valida(obs[k])]
        inst = min(cand, key=lambda o: abs(o['dateutc'] - hms)) if cand else None
        v0, v1 = bisect_right(ts, ini), bisect_right(ts, hms)
        ventana = obs[v0:v1]
        vg = [o['G'] for o in ventana if valida(o) and o['G'] is not None]
        G = sum(vg) / len(vg) if len(vg) >= MIN_LECT_HORA else None
        # lluvia de la hora: suma de incrementos del contador total (un descenso = reinicio del contador)
        rain = None
        previa = obs[v0 - 1] if v0 > 0 and ini - ts[v0 - 1] <= 15 * 60 * 1000 else None
        if previa and ventana and hms - ventana[-1]['dateutc'] <= 15 * 60 * 1000:
            seq = [previa] + ventana
            if all(o['rain_total'] is not None for o in seq) and not any('lluvia_sin_humedad' in o['qc'] or not valida(o) for o in ventana):
                rain = 0.0
                for x, y in zip(seq, seq[1:]):
                    d = y['rain_total'] - x['rain_total']
                    rain += d if d >= 0 else y['rain_total']
        serie.append({'H': H, 'hora': H.hour, 'mes': H.month,
                      'T': inst['T'] if inst else None, 'RH': inst['RH'] if inst else None,
                      'P': inst['P'] if inst else None, 'G': G, 'rain': rain})
        H += timedelta(hours=1)
    return serie


# ---------------------------------------------------------------- 4. predicción, verificación, aprendizaje
def procesar(serie, preds, ml, cf, mf, estado_modelo):
    idx = {s['H'].isoformat(): i for i, s in enumerate(serie)}
    por_hora = {p['hora']: p for p in preds}
    ultima = estado_modelo.get('ultima_hora_procesada')
    nuevas = 0
    for i, s in enumerate(serie):
        hiso = s['H'].isoformat()
        if ultima and hiso <= ultima:
            continue
        # a. predicción en H
        x = M.features(serie, i)
        if x is not None and hiso not in por_hora:
            p = M.prob_lluvia(ml, x); yhat = M.pv_pronostico(mf, x)
            por_hora[hiso] = {'hora': hiso, 'version': f"{ml['version']}+{ml.get('actualizaciones', 0)}",
                              'p_lluvia': f(p, 4), 'banda': M.banda_lluvia(p), 'pv_hat': f(yhat, 4),
                              'pv_corr': f(M.pv_corregido(cf, yhat), 4), 'pv_persist': f(x['PV'], 4),
                              'estado_lluvia': 'pendiente', 'estado_fv': 'pendiente'}
            nuevas += 1
        # b. verificar lluvia emitida en H-3
        h3 = (s['H'] - timedelta(hours=M.HORIZONTE_LLUVIA_H)).isoformat()
        pr = por_hora.get(h3)
        if pr and pr['estado_lluvia'] == 'pendiente':
            lluvias = [serie[i - k]['rain'] for k in range(0, M.HORIZONTE_LLUVIA_H)] if i >= 2 else [None]
            if all(v is not None for v in lluvias):
                y = 1 if sum(lluvias) >= M.UMBRAL_LLUVIA_MM else 0
                x3 = M.features(serie, idx[h3])
                M.actualizar_lluvia(ml, x3, y)                      # c. aprendizaje en línea
                pr['y_lluvia'] = y; pr['estado_lluvia'] = 'verificada'
            else:
                pr['estado_lluvia'] = 'sin_datos'
        # b'. verificar FV emitida en H-1
        h1 = (s['H'] - timedelta(hours=1)).isoformat()
        pf = por_hora.get(h1)
        if pf and pf['estado_fv'] == 'pendiente':
            if s['G'] is not None and s['T'] is not None:
                real = M.pv_kw(s['G'], s['T']); pf['pv_real'] = f(real, 4); pf['estado_fv'] = 'verificada'
                if 6 <= s['hora'] < 18:
                    M.actualizar_fv(cf, float(pf['pv_hat']), real)  # c'. corrección en línea
            else:
                pf['estado_fv'] = 'sin_datos'
        estado_modelo['ultima_hora_procesada'] = hiso
    filas = sorted(por_hora.values(), key=lambda r: r['hora'])
    return filas, nuevas


def resumen_verificacion(preds, umbral, desde=None):
    ll = [p for p in preds if p.get('estado_lluvia') == 'verificada' and (not desde or p['hora'] >= desde)]
    y = [int(float(p['y_lluvia'])) for p in ll]; pr = [float(p['p_lluvia']) for p in ll]
    yh = [1 if v >= umbral else 0 for v in pr]
    vp = sum(1 for a, b in zip(y, yh) if a and b); fp = sum(1 for a, b in zip(y, yh) if not a and b)
    fn = sum(1 for a, b in zip(y, yh) if a and not b); vn = sum(1 for a, b in zip(y, yh) if not a and not b)
    prec = vp / (vp + fp) if vp + fp else None; rec = vp / (vp + fn) if vp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    brier = sum((a - b) ** 2 for a, b in zip(pr, y)) / len(y) if y else None
    fv = [p for p in preds if p.get('estado_fv') == 'verificada' and (not desde or p['hora'] >= desde)
          and 6 <= int(p['hora'][11:13]) + 1 < 18]
    err = [float(p['pv_corr']) - float(p['pv_real']) for p in fv]
    errp = [float(p['pv_persist']) - float(p['pv_real']) for p in fv]
    rm = lambda e: math.sqrt(sum(v * v for v in e) / len(e)) if e else None
    return {'lluvia': {'n': len(y), 'vp': vp, 'fp': fp, 'fn': fn, 'vn': vn, 'precision': f(prec, 3), 'sensibilidad': f(rec, 3),
                       'f1': f(f1, 3), 'brier': f(brier, 4)},
            'fv': {'n': len(fv), 'mae': f(sum(abs(v) for v in err) / len(err), 4) if err else None, 'rmse': f(rm(err), 4),
                   'rmse_persistencia': f(rm(errp), 4)}}


# ---------------------------------------------------------------- 5. estado para la web
def construir_estado(obs, serie, preds, ml, cf, mf, nuevas_obs, fuente):
    ahora = datetime.now(timezone.utc)
    u = obs[-1] if obs else None
    ult = None
    if u:
        ult = {k: f(u[k], 2) for k in ('T', 'RH', 'P', 'G', 'rain_rate', 'rain_day', 'wind', 'gust', 'uv', 'Tin')}
        ult.update({'t_utc': datetime.fromtimestamp(u['dateutc'] / 1000, timezone.utc).isoformat(), 'wind_dir': u['wind_dir'],
                    'qc': u['qc'].split('|') if u['qc'] else [], 'pv_kw': f(M.pv_kw(u['G'], u['T']) if u['G'] is not None and u['T'] is not None else None, 4)})
    lim = (u['dateutc'] - 24 * 3600 * 1000) if u else 0
    s24 = [o for o in obs if o['dateutc'] >= lim]
    serie24 = {'t': [o['dateutc'] for o in s24], 'T': [f(o['T'], 1) for o in s24], 'RH': [f(o['RH'], 0) for o in s24],
               'P': [f(o['P'], 1) for o in s24], 'G': [f(o['G'], 0) for o in s24], 'rain': [f(o['rain_rate'], 1) for o in s24],
               'wind': [f(o['wind'], 1) for o in s24], 'uv': [o['uv'] for o in s24], 'ok': [1 if valida(o) else 0 for o in s24]}
    # predicción vigente = la de la última hora de la serie
    actual = None
    if serie:
        i = len(serie) - 1; x = M.features(serie, i)
        if x is not None:
            p = M.prob_lluvia(ml, x); yhat = M.pv_pronostico(mf, x)
            actual = {'hora': serie[i]['H'].isoformat(), 'p_lluvia': f(p, 4), 'banda': M.banda_lluvia(p),
                      'pv_hat': f(yhat, 4), 'pv_corr': f(M.pv_corregido(cf, yhat), 4),
                      'features': {k: f(v, 4) for k, v in x.items()}}
        else:
            faltan = sum(1 for s in serie[-4:] if any(s[k] is None for k in ('T', 'RH', 'P', 'G', 'rain')))
            actual = {'hora': serie[i]['H'].isoformat(), 'motivo': f'Se requieren 4 horas consecutivas de datos válidos; faltan datos en {faltan} de las últimas 4.'}
    lim24 = (datetime.fromtimestamp(u['dateutc'] / 1000, BOG) - timedelta(hours=24)).isoformat() if u else None
    hace7 = (datetime.fromtimestamp(u['dateutc'] / 1000, BOG) - timedelta(days=7)).isoformat() if u else None
    qc24 = {'lecturas': len(s24), 'validas': sum(1 for o in s24 if valida(o))}
    for b in ('fuera_de_rango', 'posible_interior', 'lluvia_sin_humedad', 'reinicio_contador', 'hueco_previo'):
        qc24[b] = sum(1 for o in s24 if b in o['qc'])
    recientes = [p for p in preds if p.get('estado_lluvia') == 'verificada' or p.get('estado_fv') == 'verificada'][-24:]
    return {
        'generado_utc': ahora.isoformat(), 'fuente': fuente, 'lecturas_nuevas': nuevas_obs,
        'estacion': {'nombre': 'Estancioncita Aurea · WS-2902', 'lat': 5.8277, 'lon': -73.0251, 'alt_m': 2590},
        'ultima_lectura': ult, 'serie_24h': serie24, 'qc_24h': qc24,
        'prediccion_actual': actual,
        'verificacion': {'total': resumen_verificacion(preds, ml['umbral']), 'ultimos_7_dias': resumen_verificacion(preds, ml['umbral'], hace7),
                         'recientes': recientes, 'pendientes': sum(1 for p in preds if p.get('estado_lluvia') == 'pendiente')},
        'modelo': {'lluvia': {'version': ml['version'], 'actualizaciones': ml.get('actualizaciones', 0), 'umbral': ml['umbral'],
                              'deriva': f(math.sqrt(sum((a - b) ** 2 for a, b in zip(ml['w'], ml['w0']))), 4)},
                   'fv': {'version': mf['version'], 'correccion_a': f(cf['theta'][0], 4), 'correccion_b': f(cf['theta'][1], 4),
                          'actualizaciones': cf.get('actualizaciones', 0)}},
        'total_observaciones': len(obs), 'total_predicciones': len(preds),
    }


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', help='JSON con registros de la API (pruebas sin llaves)')
    ap.add_argument('--backfill-dias', type=int, default=0, help='descarga hacia atrás N días (primera vez)')
    a = ap.parse_args()

    if a.fixture:
        crudos = json.load(open(a.fixture)); fuente = 'fixture:' + os.path.basename(a.fixture)
    else:
        crudos = descargar(288); fuente = 'api.ambientweather.net'
        if a.backfill_dias and crudos:
            fin = min(r['dateutc'] for r in crudos)
            for _ in range(a.backfill_dias):
                time.sleep(1.1)
                lote = descargar(288, datetime.fromtimestamp(fin / 1000, timezone.utc).isoformat())
                if not lote:
                    break
                crudos += lote; fin = min(r['dateutc'] for r in lote)

    previas = {int(r['dateutc']): {k: (num(r[k]) if k not in ('dateutc', 'qc') else r[k]) for k in CAMPOS_OBS} for r in leer_csv(OBS)}
    for r in previas.values():
        r['dateutc'] = int(r['dateutc'])
    n0 = len(previas)
    for r in crudos:
        if r.get('dateutc'):
            previas[int(r['dateutc'])] = convertir(r)
    obs = control_calidad(sorted(previas.values(), key=lambda o: o['dateutc']))
    nuevas_obs = len(obs) - n0

    ml = leer_json('modelo_lluvia.json'); mf = leer_json('modelo_fv.json'); cf = leer_json('correccion_fv.json')
    em = leer_json('estado_modelo.json', {})
    preds = leer_csv(PRED)
    serie = serie_horaria(obs)
    preds, nuevas = procesar(serie, preds, ml, cf, mf, em)

    escribir_csv(OBS, CAMPOS_OBS, [{**o, **{k: f(o[k], 3) for k in CAMPOS_OBS if k not in ('dateutc', 'qc', 'wind_dir', 'uv')}} for o in obs])
    escribir_csv(PRED, CAMPOS_PRED, preds)
    escribir_json('modelo_lluvia.json', ml); escribir_json('correccion_fv.json', cf); escribir_json('estado_modelo.json', em)
    estado = construir_estado(obs, serie, preds, ml, cf, mf, nuevas_obs, fuente)
    escribir_json('estado.json', estado)
    print(f"obs={len(obs)} (+{nuevas_obs}) predicciones={len(preds)} (+{nuevas}) "
          f"verif_lluvia={estado['verificacion']['total']['lluvia']['n']} act_modelo={ml.get('actualizaciones', 0)}")


if __name__ == '__main__':
    main()
