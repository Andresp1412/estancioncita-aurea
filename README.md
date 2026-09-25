# Estancioncita Aurea — v2 con retroalimentación

Pronóstico de lluvia a 3 h y de potencia fotovoltaica a 1 h con la estación Ambient Weather WS-2902 (Duitama, Boyacá). Cada pronóstico se verifica cuando se cumple su horizonte y el resultado ajusta el modelo (aprendizaje en línea).

## Qué cambió frente a la versión anterior

| Falencia detectada | Solución en v2 |
|---|---|
| Llaves de la API visibles en el HTML | La página ya no llama a la API. Las llaves van en GitHub Secrets y solo las usa la Action. |
| "IA" que en realidad era una regla fija | Regresión logística entrenada con 32 meses de datos reales (lluvia) y Gradient Boosting de 300 árboles (FV), evaluados en la propia página. |
| Métricas escritas a mano y datos simulados | `data/validacion.json` se genera con `scripts/entrenar.py`; ya no se muestra ningún dato simulado. |
| Datos de instalación usados como si fueran clima | Control de calidad por lectura: rangos, sensor bajo techo, lluvia con humedad < 60 %, reinicio de contadores, huecos. |
| Sin verificación ni aprendizaje | Cada hora: pronóstico → verificación → actualización del modelo. Todo queda en `data/predicciones.csv`. |
| ΔP de 15 min, textos con la hora de la lectura, zona horaria fija, sparklines por índice | ΔP de 1 h y 3 h; textos con la hora actual de Bogotá; `Intl` con `America/Bogota`; sparklines con eje de tiempo y huecos. |
| "Datos en vivo" con datos de hace 30 días | El título y el estado cambian según la antigüedad de la lectura (en vivo / reciente / última lectura registrada). |

## Puesta en marcha (una sola vez)

1. **Rotar las llaves.** Las anteriores quedaron publicadas en el historial del repositorio. En ambientweather.net → Account → API Keys, borra la API key y la Application key viejas y crea unas nuevas.
2. **Guardar los Secrets.** En GitHub: *Settings → Secrets and variables → Actions → New repository secret*:
   - `AMBIENT_API_KEY` = la nueva API key
   - `AMBIENT_APP_KEY` = la nueva Application key
   - `AMBIENT_MAC` = `8C:4F:00:4F:8B:95`
3. **Subir estos archivos** al repositorio `estancioncita-aurea` y reemplazar el `index.html` anterior.
4. **Activar la Action.** En la pestaña *Actions* habilita los workflows y ejecuta *Actualizar datos y modelos → Run workflow* (en `backfill_dias` puedes poner, por ejemplo, 7 para traer la última semana).
5. GitHub Pages sigue igual (rama `main`, carpeta raíz).

Desde ahí la Action corre cada 15 min. Solo publica un commit cuando llegan lecturas nuevas.

## Estructura

```
index.html                  página (misma interfaz, sin llaves)
data/estado.json            lo único que la página necesita en tiempo real
data/observaciones.csv      todas las lecturas + banderas de calidad
data/predicciones.csv       cada pronóstico y su verificación
data/modelo_lluvia.json     regresión logística (base w0 + estado actual w)
data/modelo_fv.json         Gradient Boosting exportado (árboles)
data/correccion_fv.json     corrección en línea a + b·ŷ (mínimos cuadrados recursivos)
data/validacion.json        métricas en datos históricos reales (ene–ago 2026)
data/replay_2026.json       serie horaria para la reproducción de la pestaña Validación
data/era5_duitama.csv       reanálisis 2024–2026 usado para entrenar
scripts/modelo.py           núcleo común (misma lógica que el JavaScript de la página)
scripts/actualizar.py       ciclo de retroalimentación (solo librería estándar de Python)
scripts/entrenar.py         reentrenamiento y validación (requiere pandas y scikit-learn)
src/                        fuentes de index.html (head.html, extra.css, app.js)
tests/                      pruebas de extremo a extremo y de paridad Python ↔ JavaScript
```

## Cómo verificar

```bash
python scripts/actualizar.py --fixture tests/api_fixture_2026-08-26.json  # sin llaves: registros reales del 26-ago-2026
python tests/simular_estacion.py      # 10 días en formato API, 40 ejecuciones seguidas; verifica el ciclo completo y la paridad
python tests/construir.py && python -m http.server 8765
node tests/paridad_navegador.js http://localhost:8765/_local.html   # la reproducción en JS reproduce exactamente validacion.json
```

`scripts/entrenar.py --descargar` vuelve a bajar el reanálisis de Open-Meteo y regenera modelos y métricas con semilla fija.

## Definiciones

- Hora H (local): T, HR y P instantáneas en H (±10 min); G = media de (H−1, H] con ≥ 6 lecturas; lluvia = incremento del contador total en (H−1, H].
- Lluvia: evento = acumulado ≥ 0,2 mm en (H, H+3]. Umbral de alerta elegido en entrenamiento (máx. F1).
- FV: potencia de 1 kWp en H+1 con T_celda = T + (45 − 20)/800·G y P = G/1000·[1 − 0,004 (T_celda − 25)].
- Aprendizaje en línea: w ← w − η [c_y (p − y) z + λ (w − w₀)], con η = 0,01 y λ = 0,05.
