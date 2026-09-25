// Comprueba que la reproducción de la página (JavaScript) da EXACTAMENTE las métricas
// que calculó Python en data/validacion.json.
// Uso: servir la carpeta del repo (python -m http.server 8765) y ejecutar
//      node tests/paridad_navegador.js http://localhost:8765/_local.html
const { chromium } = require('playwright');
const url = process.argv[2] || 'http://localhost:8765/index.html';
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto(url); await p.waitForTimeout(1500);
  const V = await p.evaluate(() => fetch('data/validacion.json').then(r => r.json()));
  await p.getByRole('button', { name: 'Validación', exact: true }).click(); await p.waitForTimeout(2500);
  await p.evaluate(() => { const s = document.querySelector('input[type=range]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(s, s.max); s.dispatchEvent(new Event('input', { bubbles: true })); });
  const leer = () => p.evaluate(() => [...document.querySelectorAll('.conf div')].map(d => d.textContent).filter(t => /^\d+$/.test(t)).map(Number));
  const on = await leer();
  await p.getByRole('button', { name: 'Modelo base' }).click(); await p.waitForTimeout(300);
  const base = await leer();
  const ref = k => { const m = V.lluvia[k]; return [m.vp, m.fp, m.fn, m.vn]; };
  const ok1 = JSON.stringify(on) === JSON.stringify(ref('Regresión logística + aprendizaje en línea'));
  const ok2 = JSON.stringify(base) === JSON.stringify(ref('Regresión logística (modelo de la web)'));
  console.log('aprendizaje en línea', on, ok1 ? 'OK' : 'DIFERENTE'); console.log('modelo base', base, ok2 ? 'OK' : 'DIFERENTE');
  await b.close(); process.exit(ok1 && ok2 ? 0 : 1);
})();
