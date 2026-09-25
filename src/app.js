const { useState, useEffect, useMemo, useRef, Fragment } = React;
const e = React.createElement;

/* =========================================================================
   Estancioncita Aurea v2
   - La página NO contiene llaves de API. Lee data/estado.json, que publica
     cada 15 min la GitHub Action (scripts/actualizar.py).
   - Los modelos (data/modelo_lluvia.json, data/modelo_fv.json) se evalúan
     aquí con las mismas fórmulas que scripts/modelo.py (ver tests/).
   ========================================================================= */
const DATA = 'data/';
const TZ = 'America/Bogota';
const REFRESCO_MS = 5 * 60 * 1000;

/* ------------------------------------------------------------------ utilidades */
function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
function isNum(v){ return typeof v === 'number' && isFinite(v); }
function fmt(v, d=1){ return isNum(v) ? v.toLocaleString('es-CO', {minimumFractionDigits:d, maximumFractionDigits:d}) : '—'; }
function pct(v, d=0){ return isNum(v) ? fmt(v*100, d)+' %' : '—'; }
function horaBogota(date){ return Number(new Intl.DateTimeFormat('en-US',{timeZone:TZ, hour:'numeric', hourCycle:'h23'}).format(date)); }
function fmtFecha(date, conHora=true){
  return new Intl.DateTimeFormat('es-CO', conHora ? {timeZone:TZ, day:'2-digit', month:'short', hour:'numeric', minute:'2-digit'} : {timeZone:TZ, day:'2-digit', month:'short'}).format(date);
}
function fmtHora(date){ return new Intl.DateTimeFormat('es-CO',{timeZone:TZ, hour:'numeric', minute:'2-digit'}).format(date); }
function relTime(date, nowMs){
  const min = Math.floor((nowMs - date.getTime())/60000);
  if(min < 1) return 'justo ahora';
  if(min < 60) return `hace ${min} min`;
  const h = Math.floor(min/60);
  if(h < 48) return `hace ${h} h`;
  return `hace ${Math.floor(h/24)} d`;
}
function getVar(name){ return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'; }
async function cargarJSON(nombre){
  const r = await fetch(DATA + nombre + '?t=' + Date.now(), {cache:'no-store'});
  if(!r.ok) throw new Error(nombre + ': HTTP ' + r.status);
  return r.json();
}

/* =========================================================================
   Modelos — equivalentes exactos de scripts/modelo.py
   ========================================================================= */
const NOCT = 45, GAMMA = -0.004, P_NOM = 1.0, UMBRAL_MM = 0.2, HORIZONTE = 3;
function pvKw(G, T){
  if(!isNum(G) || !isNum(T)) return null;
  const tc = T + (NOCT-20)/800*G;
  return Math.max(0, P_NOM*(G/1000)*(1 + GAMMA*(tc-25)));
}
function features(serie, i){
  if(i < 3) return null;
  const a = serie[i], b1 = serie[i-1], b3 = serie[i-3];
  const need = [a.T,a.RH,a.P,a.G,a.rain,b1.P,b1.RH,b1.T,b1.G,b3.P];
  if(need.some(v => !isNum(v))) return null;
  const h = a.hora, m = a.mes, TAU = 2*Math.PI;
  return { T:a.T, RH:a.RH, P:a.P, G:a.G, dP1:a.P-b1.P, dP3:a.P-b3.P, dRH1:a.RH-b1.RH, dT1:a.T-b1.T, dG1:a.G-b1.G,
    hs:Math.sin(TAU*h/24), hc:Math.cos(TAU*h/24), ms:Math.sin(TAU*m/12), mc:Math.cos(TAU*m/12),
    rain_now: a.rain >= UMBRAL_MM ? 1 : 0, PV: pvKw(a.G, a.T) };
}
function zLluvia(m, x){ return m.features.map((k,i) => (x[k]-m.mu[i])/m.sd[i]); }
function probLluvia(m, x){
  const z = zLluvia(m, x); let s = m.b;
  for(let i=0;i<z.length;i++) s += m.w[i]*z[i];
  return 1/(1+Math.exp(-s));
}
function contribuciones(m, x){
  const z = zLluvia(m, x);
  return m.features.map((k,i) => ({k, v: m.w[i]*z[i]})).sort((a,b) => Math.abs(b.v)-Math.abs(a.v));
}
function actualizarLluvia(m, x, y){        // descenso de gradiente regularizado hacia el modelo base
  const z = zLluvia(m, x), p = probLluvia(m, x), g = (p - y)*m.peso_clase[y];
  m.w = m.w.map((w,i) => w - m.eta*(g*z[i] + m.lambda*(w - m.w0[i])));
  m.b = m.b - m.eta*(g + m.lambda*(m.b - m.b0));
}
function pvPronostico(m, x){
  const xs = m.features.map(k => x[k]); let s = 0;
  for(const t of m.trees){
    let n = 0;
    while(t.l[n] !== -1) n = xs[t.f[n]] <= t.t[n] ? t.l[n] : t.r[n];
    s += t.v[n];
  }
  return Math.max(0, m.init + m.lr*s);
}
function bandaLluvia(p){ return p >= 0.66 ? 'PROBABLE' : p >= 0.35 ? 'POSIBLE' : 'IMPROBABLE'; }
function riesgoHelada(T, hora){
  const noche = hora >= 21 || hora <= 7;
  if(T < 0) return 'alto';
  if(noche && T < 5) return 'medio';
  return 'bajo';
}
const NOMBRES = { T:'Temperatura', RH:'Humedad relativa', P:'Presión', G:'Radiación', dP1:'Cambio de presión 1 h', dP3:'Cambio de presión 3 h',
  dRH1:'Cambio de humedad 1 h', dT1:'Cambio de temperatura 1 h', hs:'Hora del día (seno)', hc:'Hora del día (coseno)', ms:'Mes (seno)', mc:'Mes (coseno)', rain_now:'Llueve ahora' };

/* =========================================================================
   Textos de apoyo a la decisión (usan la HORA ACTUAL, no la de la lectura)
   ========================================================================= */
function textoVentanaLluvia(p, horaEmision){
  const h1 = (horaEmision+1)%24, h3 = (horaEmision+HORIZONTE)%24;
  const franja = `${h1}:00–${h3}:00`;
  if(p >= 0.66) return `El modelo espera lluvia de al menos 0,2 mm entre las <strong>${franja}</strong>.`;
  if(p >= 0.35) return `Hay señales de posible lluvia entre las <strong>${franja}</strong>; conviene estar pendiente.`;
  return `No se esperan lluvias apreciables entre las <strong>${franja}</strong>.`;
}
function textoVentanaHelada(horaAhora, riesgo){
  if(horaAhora>=18 || horaAhora<3){
    return riesgo==='bajo'
      ? 'Con la lectura actual el riesgo es bajo, pero el mayor enfriamiento de la noche llega entre las <strong>3:00 a.m. y las 6:00 a.m.</strong>'
      : 'La ventana de <strong>mayor riesgo</strong> será esta madrugada, entre las 3:00 a.m. y las 6:00 a.m.';
  }
  if(horaAhora>=3 && horaAhora<7) return 'Estás dentro de la franja de <strong>mayor riesgo de helada</strong> (3:00 a.m.–6:00 a.m.).';
  return 'De día el riesgo de helada es mínimo; la próxima evaluación importante es la madrugada siguiente (3:00 a.m.–6:00 a.m.).';
}
function recoRiego(banda, riesgoH){
  if(banda === 'PROBABLE') return 'Es probable que llueva en las próximas horas: puedes esperar antes de regar y ahorrar agua.';
  if(riesgoH && riesgoH !== 'bajo') return 'Si vas a regar, hazlo temprano; el riego por aspersión durante la helada misma (no antes) protege por calor de congelación.';
  if(banda === 'POSIBLE') return 'Posible lluvia: si el cultivo no está en déficit hídrico, espera la próxima actualización antes de regar.';
  return 'Sin señales de lluvia: riega según el plan normal del cultivo.';
}
function recoCultivo(banda, riesgoH){
  if(riesgoH !== 'bajo') return 'Cubre plántulas y cultivos sensibles (hortalizas, papa recién sembrada) con plástico o agrotela antes del atardecer.';
  if(banda === 'PROBABLE') return 'Si hay cultivos recién fumigados o abonados, la lluvia puede lavar el producto.';
  return 'No se esperan condiciones extremas; monitoreo normal.';
}
function recoEnergia(pvAhora, pvProx){
  const ref = Math.max(pvAhora||0, pvProx||0);
  return ref > 0.4
    ? 'Buen momento para tareas que consuman energía (bombeo de agua, carga de baterías) si tienes paneles solares.'
    : 'Generación baja; si dependes de paneles, prioriza el consumo entre las 10 a.m. y las 2 p.m.';
}

/* =========================================================================
   Íconos (SVG en línea)
   ========================================================================= */
function RainIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.6,strokeLinecap:'round',strokeLinejoin:'round',...p},
  e('path',{d:'M7 15.5a4 4 0 0 1 .7-7.94 5 5 0 0 1 9.6.44A3.5 3.5 0 0 1 17 15.5H7Z'}), e('path',{d:'M8 18.5v1.2M12 18.5v2M16 18.5v1.2'})); }
function FrostIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.6,strokeLinecap:'round',strokeLinejoin:'round',...p},
  e('path',{d:'M12 2v20M4.5 6.5l15 11M19.5 6.5l-15 11'}), e('path',{d:'M12 2 9.8 4.2M12 2l2.2 2.2M12 22l-2.2-2.2M12 22l2.2-2.2'})); }
function SunIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.6,strokeLinecap:'round',strokeLinejoin:'round',...p},
  e('circle',{cx:12,cy:12,r:4.2}), e('path',{d:'M12 2.5v2.4M12 19.1v2.4M4.6 4.6l1.7 1.7M17.7 17.7l1.7 1.7M2.5 12h2.4M19.1 12h2.4M4.6 19.4l1.7-1.7M17.7 6.3l1.7-1.7'})); }
function MoonIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'currentColor',...p}, e('path',{d:'M20 14.2A8.5 8.5 0 1 1 9.8 4a7 7 0 0 0 10.2 10.2Z'})); }
function RefreshIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.8,strokeLinecap:'round',strokeLinejoin:'round',...p},
  e('path',{d:'M20 11A8 8 0 0 0 6 5.3M4 4v5h5'}), e('path',{d:'M4 13a8 8 0 0 0 14 5.7M20 20v-5h-5'})); }
function BulbIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.6,strokeLinecap:'round',strokeLinejoin:'round',...p},
  e('path',{d:'M12 3a7 7 0 0 0-4 12.7V18a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-2.3A7 7 0 0 0 12 3Z'}), e('path',{d:'M10 22h4'})); }
function LoopIcon(p){ return e('svg',{viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.6,strokeLinecap:'round',strokeLinejoin:'round',...p},
  e('path',{d:'M4 12a8 8 0 0 1 13.7-5.6L20 9'}), e('path',{d:'M20 4v5h-5'}), e('path',{d:'M20 12a8 8 0 0 1-13.7 5.6L4 15'}), e('path',{d:'M4 20v-5h5'})); }

/* =========================================================================
   Sparkline con eje de tiempo real y huecos donde no hay datos válidos
   ========================================================================= */
function Sparkline({t, data, ok, color}){
  const w = 100, h = 28;
  const pts = [];
  for(let i=0;i<data.length;i++) if(isNum(data[i]) && (!ok || ok[i])) pts.push([t[i], data[i]]);
  if(pts.length < 2) return e('div',{className:'spark'});
  const t0 = t[0], t1 = t[t.length-1] || t0+1, span = (t1-t0) || 1;
  const vals = pts.map(p=>p[1]), min = Math.min(...vals), max = Math.max(...vals), range = (max-min) || 1;
  const X = tt => ((tt-t0)/span*w), Y = v => (h - 2 - (v-min)/range*(h-4));
  const segs = []; let cur = [];
  for(let i=0;i<pts.length;i++){
    if(i>0 && pts[i][0]-pts[i-1][0] > 30*60*1000){ segs.push(cur); cur = []; }   // hueco > 30 min
    cur.push(`${X(pts[i][0]).toFixed(2)},${Y(pts[i][1]).toFixed(2)}`);
  }
  segs.push(cur);
  return e('svg',{className:'spark',viewBox:`0 0 ${w} ${h}`,preserveAspectRatio:'none', role:'img', 'aria-label':'Tendencia últimas 24 h'},
    segs.filter(s=>s.length>1).map((s,i)=> e('polyline',{key:i, points:s.join(' '), fill:'none', stroke:color, strokeWidth:1.8, strokeLinecap:'round', strokeLinejoin:'round', vectorEffect:'non-scaling-stroke'})));
}

/* =========================================================================
   Chart.js — envoltorio genérico (una sola escala Y por gráfica)
   ========================================================================= */
function ChartBox({config, height=230}){
  const ref = useRef(null), chart = useRef(null);
  useEffect(()=>{
    chart.current = new Chart(ref.current, config);
    return ()=>{ chart.current && chart.current.destroy(); };
  }, []);
  useEffect(()=>{
    if(!chart.current) return;
    chart.current.data = config.data;
    if(config.options && config.options.scales) chart.current.options.scales = config.options.scales;
    chart.current.update('none');
  }, [config]);
  return e('div',{className:'chart-wrap', style:{height}}, e('canvas',{ref}));
}
function baseOptions(yTitle, extra={}){
  return {
    responsive:true, maintainAspectRatio:false, animation:false, resizeDelay:100,
    interaction:{mode:'index', intersect:false},
    scales:{
      x:{ ticks:{color:getVar('--ink-dim'), maxTicksLimit:8, maxRotation:0}, grid:{display:false} },
      y:{ ticks:{color:getVar('--ink-dim')}, grid:{color:getVar('--border')}, title:{display:!!yTitle, text:yTitle, color:getVar('--ink-dim')}, ...(extra.y||{}) },
    },
    plugins:{ legend:{position:'bottom', labels:{color:getVar('--ink'), boxWidth:12, font:{size:11}}},
      tooltip:{ callbacks: extra.tooltip || {} } },
  };
}

/* =========================================================================
   Vista "Hoy"
   ========================================================================= */
function VerdictCard({tone, Icon, badgeText, badgeClass, pregunta, respuesta, meterPct, ventanaHtml, nota, recoLabel, reco, detalle, expanded, onToggle}){
  return e('div',{className:`vcard vcard--${tone}`},
    e('div',{className:'vcard-head'}, e('span',{className:'vcard-icon'}, e(Icon,null)), e('span',{className:`badge badge--${badgeClass}`}, badgeText)),
    e('div',{className:'vcard-q'}, pregunta),
    e('div',{className:'vcard-a'}, respuesta),
    e('div',{className:'meter'}, e('i',{style:{width:clamp(meterPct,0,100).toFixed(0)+'%', background:`var(--${tone})`}})),
    e('div',{className:'vcard-window', dangerouslySetInnerHTML:{__html:ventanaHtml}}),
    nota && e('div',{className:'fresh-note'}, nota),
    e('div',{className:'vcard-reco'}, e('strong',null, recoLabel+': '), reco),
    detalle && e('button',{className:'vcard-toggle', onClick:onToggle}, expanded ? 'Ocultar cálculo ▲' : 'Ver cómo se calculó ▾'),
    detalle && expanded && e('div',{className:'vcard-detail'}, detalle)
  );
}

function SensorStrip({estado}){
  const u = estado.ultima_lectura || {}, s = estado.serie_24h || {t:[]};
  const pv24 = (s.G||[]).map((g,i)=> pvKw(g, s.T[i]));
  const tiles = [
    {k:'T', label:'Temperatura', v:u.T, d:1, unit:'°C', data:s.T, color:'var(--brand)'},
    {k:'RH', label:'Humedad', v:u.RH, d:0, unit:'%', data:s.RH, color:'var(--rain)'},
    {k:'P', label:'Presión', v:u.P, d:1, unit:'hPa', data:s.P, color:'var(--ink-dim)'},
    {k:'G', label:'Radiación', v:u.G, d:0, unit:'W/m²', data:s.G, color:'var(--sun)'},
    {k:'PV', label:'Generación FV', v:u.pv_kw, d:2, unit:'kW', data:pv24, color:'var(--ok)'},
    {k:'R', label:'Lluvia', v:u.rain_rate, d:1, unit:'mm/h', data:s.rain, color:'var(--rain)'},
    {k:'W', label:'Viento', v:u.wind, d:1, unit:'km/h', data:s.wind, color:'var(--ink-dim)'},
    {k:'UV', label:'Índice UV', v:u.uv, d:0, unit:'', data:s.uv, color:'var(--sun)'},
  ];
  return e('div',{className:'sensor-grid'}, tiles.map(t =>
    e('div',{className:'sensor-tile', key:t.k},
      e('span',{className:'label'}, t.label),
      e('div',{className:'value tabular'+(isNum(t.v)?'':' na')}, fmt(t.v, t.d), t.unit && e('small',null, t.unit)),
      e(Sparkline,{t:s.t, data:t.data||[], ok: t.k==='R'||t.k==='W'||t.k==='UV' ? null : s.ok, color:t.color})
    )));
}

function QCLine({qc}){
  if(!qc || !qc.lecturas) return null;
  const alertas = [
    ['posible_interior','posible sensor bajo techo'], ['lluvia_sin_humedad','lluvia inconsistente con la humedad'],
    ['fuera_de_rango','valores fuera de rango'], ['reinicio_contador','reinicio del contador de lluvia'], ['hueco_previo','huecos de transmisión']
  ].filter(([k]) => qc[k] > 0).map(([k,t]) => `${t} (${qc[k]})`);
  const ok = alertas.length === 0;
  return e('div',{className:'qc-line'},
    e('span',{className:'badge '+(ok?'badge--ok':'badge--warn')}, ok ? 'CALIDAD OK' : 'CONTROL DE CALIDAD'),
    e('span',null, ok ? `${qc.validas} de ${qc.lecturas} lecturas de las últimas 24 h pasaron todas las pruebas.`
      : `Lecturas válidas para el modelo: ${qc.validas} de ${qc.lecturas}. Alertas: ${alertas.join(' · ')}. Las lecturas marcadas no se usan para pronosticar ni para reentrenar.`));
}

function HoyView({estado, modelos, now, expandedCard, onToggleCard, irA}){
  const u = estado.ultima_lectura;
  if(!u) return e('div',{className:'panel'}, 'Todavía no hay lecturas de la estación.');
  const tLect = new Date(u.t_utc), edadMin = (now - tLect)/60000;
  const fresca = edadMin <= 120, enVivo = edadMin <= 20;
  const horaAhora = horaBogota(now);
  const pa = estado.prediccion_actual || {};
  const tPred = pa.hora ? new Date(pa.hora) : null;
  const predVigente = isNum(pa.p_lluvia) && tPred && (now - tPred) <= HORIZONTE*3600*1000;
  const notaVieja = !fresca ? `Basado en la última lectura registrada (${fmtFecha(tLect)}). Se actualizará solo cuando la estación vuelva a transmitir.` : null;

  // ---- Lluvia
  let lluvia;
  if(isNum(pa.p_lluvia)){
    const p = pa.p_lluvia, b = pa.banda || bandaLluvia(p);
    const contrib = modelos.lluvia && pa.features ? contribuciones(modelos.lluvia, pa.features).slice(0,4) : [];
    lluvia = { badge: predVigente ? {t:b, c: b==='PROBABLE'?'info':b==='POSIBLE'?'warn':'ok'} : {t:'VENCIDO', c:'warn'},
      resp: b==='PROBABLE' ? 'Sí, es probable' : b==='POSIBLE' ? 'Posible, hay que estar pendiente' : 'No es probable',
      meter: p*100, ventana: textoVentanaLluvia(p, tPred ? horaBogota(tPred) : horaAhora) + ` <span class="mono muted-note">(${fmt(p*100,0)} %)</span>`,
      nota: predVigente ? null : `Pronóstico emitido a las ${fmtFecha(tPred)}; su horizonte de 3 h ya pasó.`,
      reco: recoRiego(b, null), bandaActual: predVigente ? b : null,
      detalle: e(Fragment,null,
        e('div',{className:'big'}, fmt(p*100,1)+' %'),
        e('p',{style:{margin:'6px 0 0'}}, `Regresión logística entrenada con 32 meses de datos históricos de la ubicación y ajustada en línea con cada hora verificada (versión ${modelos.lluvia ? modelos.lluvia.version : '?'} + ${estado.modelo ? estado.modelo.lluvia.actualizaciones : 0} actualizaciones). Variables que más pesaron:`),
        e('ul',{className:'contrib'}, contrib.map(c => e('li',{key:c.k}, e('span',null, NOMBRES[c.k]||c.k), e('span',{className:c.v>0?'up':'down'}, (c.v>0?'+':'')+fmt(c.v,2))))),
        e('code',{className:'formula'}, 'p = 1 / (1 + e^−(b + Σ wᵢ·zᵢ)),  zᵢ = (xᵢ − μᵢ)/σᵢ'))
    };
  } else {
    lluvia = { badge:{t:'EN ESPERA', c:'warn'}, resp:'Pronóstico en espera', meter:0,
      ventana: pa.motivo || 'Aún no hay suficientes datos para pronosticar.', nota:null,
      reco:'Cuando la estación complete 4 horas continuas de lecturas válidas, el modelo emitirá el pronóstico de las próximas 3 horas.', bandaActual:null, detalle:null };
  }

  // ---- Helada (regla física sobre la última lectura válida)
  const riesgo = fresca && isNum(u.T) ? riesgoHelada(u.T, horaAhora) : null;
  const heladaBadge = riesgo==='alto' ? {t:'ALERTA', c:'danger'} : riesgo==='medio' ? {t:'VIGILAR', c:'warn'} : riesgo==='bajo' ? {t:'TRANQUILO', c:'ok'} : {t:'SIN LECTURA', c:'warn'};
  const heladaResp = riesgo==='alto' ? 'Sí, alerta' : riesgo==='medio' ? 'Posible, vigilar' : riesgo==='bajo' ? 'Sin riesgo por ahora' : 'Sin lectura reciente';

  // ---- FV
  const pvAhora = fresca ? u.pv_kw : null, pvProx = predVigente || (isNum(pa.pv_corr) && fresca) ? pa.pv_corr : null;
  const fvBadge = !fresca ? {t:'SIN LECTURA', c:'warn'} : (pvAhora||0) > 0.6 ? {t:'GENERANDO BIEN', c:'warn'} : (pvAhora||0) > 0.05 ? {t:'GENERANDO', c:'info'} : {t:'SIN SOL', c:'ok'};

  const ver = estado.verificacion || {}, vt = (ver.total||{}), ml = (estado.modelo||{}).lluvia || {};
  return e(Fragment, null,
    e('div',{className:'panel'},
      e('h2',{className:'panel-title'}, enVivo ? 'Datos en vivo' : fresca ? 'Datos recientes' : 'Última lectura registrada'),
      e('p',{className:'panel-sub'}, `Estación WS-2902 · lectura del ${fmtFecha(tLect)} (${relTime(tLect, now.getTime())})`,
        estado.generado_utc ? ` · servidor revisó ${relTime(new Date(estado.generado_utc), now.getTime())}` : ''),
      e(SensorStrip,{estado}),
      e(QCLine,{qc:estado.qc_24h})
    ),
    e('div',{className:'verdict-grid'},
      e(VerdictCard,{tone:'rain', Icon:RainIcon, badgeText:lluvia.badge.t, badgeClass:lluvia.badge.c, pregunta:'¿Va a llover en las próximas 3 h?',
        respuesta:lluvia.resp, meterPct:lluvia.meter, ventanaHtml:lluvia.ventana, nota:lluvia.nota || (isNum(pa.p_lluvia)?null:notaVieja),
        recoLabel:'Riego', reco:lluvia.reco, detalle:lluvia.detalle, expanded:expandedCard==='rain', onToggle:()=>onToggleCard('rain')}),
      e(VerdictCard,{tone:'frost', Icon:FrostIcon, badgeText:heladaBadge.t, badgeClass:heladaBadge.c, pregunta:'¿Hay riesgo de helada?',
        respuesta:heladaResp, meterPct:{alto:95, medio:55, bajo:12}[riesgo]||0, ventanaHtml:textoVentanaHelada(horaAhora, riesgo||'bajo'), nota:notaVieja,
        recoLabel:'Cultivos', reco: riesgo ? recoCultivo(lluvia.bandaActual, riesgo) : 'Sin una lectura reciente de temperatura no se puede evaluar la helada de esta noche.',
        detalle: e(Fragment,null, e('div',{className:'big'}, riesgo ? riesgo.toUpperCase() : '—'),
          e('p',{style:{margin:'6px 0 0'}}, 'Regla física: alerta si la temperatura está bajo 0 °C, o bajo 5 °C entre 9 p. m. y 7 a. m., cuando el enfriamiento radiativo nocturno puede dañar cultivos. El reanálisis regional no registra heladas en la zona (mín. 5,9 °C en 974 días), por eso este módulo depende de la medición local.'),
          e('code',{className:'formula'}, 'alto: T < 0 °C · medio: T < 5 °C y de noche')),
        expanded:expandedCard==='frost', onToggle:()=>onToggleCard('frost')}),
      e(VerdictCard,{tone:'sun', Icon:SunIcon, badgeText:fvBadge.t, badgeClass:fvBadge.c, pregunta:'¿Cuánta energía solar hay?',
        respuesta: fresca ? `${fmt(pvAhora,2)} kW ahora` : 'Sin lectura reciente', meterPct:(pvAhora||0)*100,
        ventanaHtml: isNum(pvProx) ? `Pronóstico para la próxima hora: <strong>${fmt(pvProx,2)} kW</strong> en un sistema de 1 kWp.` : 'El pronóstico a 1 h aparece cuando hay 4 horas continuas de datos válidos.',
        nota:notaVieja, recoLabel:'Energía', reco:recoEnergia(pvAhora, pvProx),
        detalle: e(Fragment,null, e('div',{className:'big'}, isNum(pvProx) ? fmt(pvProx,2)+' kW en 1 h' : fmt(pvAhora,2)+' kW'),
          e('p',{style:{margin:'6px 0 0'}}, `Ahora: modelo físico con la irradiancia y temperatura medidas. En 1 h: Gradient Boosting (300 árboles) corregido en línea con la estación: ŷ = ${fmt((estado.modelo||{}).fv ? estado.modelo.fv.correccion_a : 0,3)} + ${fmt((estado.modelo||{}).fv ? estado.modelo.fv.correccion_b : 1,3)}·ŷ_GB.`),
          e('code',{className:'formula'}, 'P = 1 kWp·(G/1000)·[1 − 0,004·(T_celda − 25)] · T_celda = T + 25/800·G')),
        expanded:expandedCard==='sun', onToggle:()=>onToggleCard('sun')})
    ),
    e('div',{className:'panel'},
      e('div',{className:'row-head'},
        e('div',null, e('h2',{className:'panel-title'}, 'Aprendizaje continuo'),
          e('p',{className:'panel-sub'}, 'Cada hora el sistema compara lo que pronosticó con lo que midió la estación y ajusta los modelos con ese resultado.')),
        e('button',{className:'btn-link', onClick:()=>irA('validacion')}, 'Ver validación →')),
      e('div',{className:'stat-row'},
        e('div',{className:'stat'}, e('span',{className:'label'},'Pronósticos verificados'), e('span',{className:'num'}, (vt.lluvia||{}).n ?? 0), e('span',{className:'sub'}, `${ver.pendientes||0} pendientes de verificar`)),
        e('div',{className:'stat'}, e('span',{className:'label'},'F1 lluvia (estación)'), e('span',{className:'num'}, isNum((vt.lluvia||{}).f1) ? fmt(vt.lluvia.f1,3) : '—'), e('span',{className:'sub'}, 'histórico real: '+(modelos.validacion ? fmt(modelos.validacion.lluvia['Regresión logística + aprendizaje en línea'].f1,3) : '—'))),
        e('div',{className:'stat'}, e('span',{className:'label'},'Error FV a 1 h (estación)'), e('span',{className:'num'}, isNum((vt.fv||{}).rmse) ? fmt(vt.fv.rmse,3)+' kW' : '—'), e('span',{className:'sub'}, isNum((vt.fv||{}).rmse_persistencia) ? 'persistencia: '+fmt(vt.fv.rmse_persistencia,3)+' kW' : 'RMSE diurno')),
        e('div',{className:'stat'}, e('span',{className:'label'},'Actualizaciones del modelo'), e('span',{className:'num'}, ml.actualizaciones ?? 0), e('span',{className:'sub'}, 'distancia al modelo base: '+fmt(ml.deriva,3)))
      )
    ),
    e('div',{className:'callout'}, e(BulbIcon,null),
      e('div',null, e('h4',null,'¿Sabías que…?'),
        e('p',null,'En el altiplano de Boyacá las heladas por enfriamiento radiativo son más frecuentes en temporada seca (diciembre–febrero y junio–agosto), cuando el cielo despejado deja escapar el calor del suelo durante la noche. Una técnica usada en fincas de la región es el ', e('strong',null,'riego por aspersión durante el evento de helada'), ': el calor que libera el agua al congelarse mantiene el tejido de la planta cerca de 0 °C. Requiere equipo y cronometraje adecuados; valídalo con un asistente técnico agropecuario antes de aplicarla.')))
  );
}

/* =========================================================================
   Vista "Validación"
   ========================================================================= */
function TablaMetricas({filas, cols, mejor}){
  return e('div',{className:'table-wrap'}, e('table',null,
    e('thead',null, e('tr',null, cols.map((c,i)=> e('th',{key:i}, c[0])))),
    e('tbody',null, filas.map(([nombre, m]) => e('tr',{key:nombre, className: nombre===mejor ? 'best':''},
      e('td',null, nombre), cols.slice(1).map((c,i) => e('td',{key:i, className:'mono'}, isNum(m[c[1]]) ? fmt(m[c[1]], c[2] ?? 3) : '—')))))));
}

function ValidacionHistorica({V}){
  const [mod, setMod] = useState('lluvia');
  const ll = Object.entries(V.lluvia), fv = Object.entries(V.fv).filter(([k,v]) => typeof v === 'object');
  const cfgLl = useMemo(()=>({type:'bar', data:{labels:ll.map(x=>x[0]), datasets:[{label:'F1', data:ll.map(x=>x[1].f1),
      backgroundColor:ll.map(x=> x[0].startsWith('Regresión') ? getVar('--ok') : 'rgba(107,99,84,0.35)'), borderRadius:4, barThickness:18}]},
    options:{...baseOptions(''), indexAxis:'y', plugins:{legend:{display:false}}, scales:{x:{min:0,max:1, grid:{color:getVar('--border')}, ticks:{color:getVar('--ink-dim')}, title:{display:true,text:'F1 (0–1, mayor es mejor)',color:getVar('--ink-dim')}}, y:{grid:{display:false}, ticks:{color:getVar('--ink')}}}}}),[V]);
  const cfgFv = useMemo(()=>({type:'bar', data:{labels:fv.map(x=>x[0]), datasets:[{label:'RMSE (kW)', data:fv.map(x=>x[1].rmse),
      backgroundColor:fv.map(x=> x[0].startsWith('Gradient') ? getVar('--ok') : 'rgba(107,99,84,0.35)'), borderRadius:4, barThickness:18}]},
    options:{...baseOptions(''), indexAxis:'y', plugins:{legend:{display:false}}, scales:{x:{min:0, grid:{color:getVar('--border')}, ticks:{color:getVar('--ink-dim')}, title:{display:true,text:'RMSE diurno en kW (menor es mejor)',color:getVar('--ink-dim')}}, y:{grid:{display:false}, ticks:{color:getVar('--ink')}}}}}),[V]);
  const H = V.heladas;
  return e('div',{className:'panel'},
    e('div',{className:'metrics-head'},
      e('div',null, e('h2',{className:'panel-title'}, 'Validación con datos históricos reales', e('span',{className:'tag'},'REAL')),
        e('p',{className:'panel-sub', style:{margin:0}}, `Reanálisis ERA5 (Open-Meteo) en las coordenadas de la estación. Entrenamiento 2024–2025; prueba ${V.n_horas.toLocaleString('es-CO')} h de ene–ago 2026 que el modelo nunca vio.`)),
      e('div',{className:'segmented'}, [['lluvia','Lluvia 3 h'],['fv','Fotovoltaico 1 h'],['heladas','Heladas']].map(([k,t]) =>
        e('button',{key:k, className:mod===k?'active':'', onClick:()=>setMod(k)}, t)))),
    mod==='lluvia' && e(Fragment,null,
      e('p',{className:'metrics-note'}, e('strong',null,'Evento: '), `≥ 0,2 mm en las 3 h siguientes (prevalencia ${pct(V.prevalencia_lluvia,1)}). Umbral de alerta ${fmt(V.umbral_modelo,2)}, elegido solo con datos de entrenamiento.`),
      e(TablaMetricas,{filas:ll, mejor:'Regresión logística + aprendizaje en línea', cols:[['Método'],['F1','f1'],['CSI','csi'],['Precisión','precision'],['Sensibilidad','sensibilidad'],['AUC','auc']]}),
      e('div',{style:{marginTop:12}}, e(ChartBox,{config:cfgLl, height:200}))),
    mod==='fv' && e(Fragment,null,
      e('p',{className:'metrics-note'}, e('strong',null,'Mejora: '), `${fmt(V.fv.mejora_vs_persistencia_pct,1)} % menos error que la persistencia y ${fmt(V.fv.mejora_vs_climatologia_pct,1)} % menos que la climatología, en ${V.fv.n_horas_diurnas.toLocaleString('es-CO')} h diurnas.`),
      e(TablaMetricas,{filas:fv, mejor:'Gradient Boosting (modelo de la web)', cols:[['Método'],['RMSE (kW)','rmse'],['MAE (kW)','mae'],['R²','r2']]}),
      e('div',{style:{marginTop:12}}, e(ChartBox,{config:cfgFv, height:170}))),
    mod==='heladas' && e('div',{className:'empty'},
      `En ${H.dias} días de reanálisis la temperatura mínima fue ${fmt(H.tmin_absoluta,1)} °C: ${H.horas_bajo_0} horas bajo 0 °C y ${H.horas_bajo_5} bajo 5 °C. La escala regional no resuelve las heladas, que son un fenómeno local de superficie. Por eso este módulo usa una regla física y su validación depende de las mediciones de la propia estación.`),
    e('p',{className:'muted-note', style:{marginTop:12}}, 'Reproducible: scripts/entrenar.py descarga los mismos datos y genera data/validacion.json. ', e('a',{href:V.fuente, target:'_blank', rel:'noopener', style:{color:'var(--brand-ink)'}}, 'Fuente de datos'))
  );
}

function prepararReplay(R, ml, mf){
  const [y0,mo0,d0,h0] = R.t0.match(/\d+/g).map(Number);
  const base = Date.UTC(y0, mo0-1, d0, h0);
  const serie = R.T.map((_,i) => { const d = new Date(base + i*3600e3);
    return {ms: base + i*3600e3, hora:d.getUTCHours(), mes:d.getUTCMonth()+1, T:R.T[i], RH:R.RH[i], P:R.P[i], G:R.G[i], rain:R.PR[i]}; });
  const mOn = JSON.parse(JSON.stringify(ml));
  const n = serie.length, out = [];
  const X = new Array(n).fill(null), pOn = new Array(n).fill(null);
  for(let i=0;i<n;i++){
    X[i] = features(serie, i);
    if(X[i]) pOn[i] = probLluvia(mOn, X[i]);
    const j = i - HORIZONTE;                       // en la hora i ya se conoce la lluvia de (j, j+3]
    if(j >= 3 && X[j]){
      const y = (serie[j+1].rain + serie[j+2].rain + serie[j+3].rain) >= UMBRAL_MM ? 1 : 0;
      actualizarLluvia(mOn, X[j], y);
    }
  }
  for(let i=3;i<n-HORIZONTE;i++){
    if(!X[i]) continue;
    const y = (serie[i+1].rain + serie[i+2].rain + serie[i+3].rain) >= UMBRAL_MM ? 1 : 0;
    const hSig = serie[i+1].hora;
    out.push({ i, ms: serie[i].ms, hora: serie[i].hora, pBase: probLluvia(ml, X[i]), pOn: pOn[i], y,
      mm: serie[i+1].rain + serie[i+2].rain + serie[i+3].rain, rainNow: serie[i].rain,
      pvHat: pvPronostico(mf, X[i]), pvReal: pvKw(serie[i+1].G, serie[i+1].T), pvPers: X[i].PV, dia: hSig >= 6 && hSig < 18 });
  }
  // acumulados para consultar cualquier posición del deslizador en O(1)
  const acc = {base:[], on:[]}; let a = {base:[0,0,0,0], on:[0,0,0,0]}, ae = 0, aep = 0, nd = 0; const fvAcc = [];
  for(const r of out){
    for(const k of ['base','on']){
      const yh = (k==='base' ? r.pBase : r.pOn) >= ml.umbral ? 1 : 0, c = a[k].slice();
      if(yh && r.y) c[0]++; else if(yh && !r.y) c[1]++; else if(!yh && r.y) c[2]++; else c[3]++;
      a[k] = c; acc[k].push(c);
    }
    if(r.dia){ ae += (r.pvHat - r.pvReal)**2; aep += (r.pvPers - r.pvReal)**2; nd++; }
    fvAcc.push({rmse: nd ? Math.sqrt(ae/nd) : null, rmseP: nd ? Math.sqrt(aep/nd) : null, n: nd});
  }
  return {out, acc, fvAcc};
}
function f1De(c){ const [tp,fp,fn] = c; return (2*tp+fp+fn) ? 2*tp/(2*tp+fp+fn) : null; }
const fechaReplay = ms => new Intl.DateTimeFormat('es-CO',{timeZone:'UTC', weekday:'short', day:'2-digit', month:'short', hour:'numeric', minute:'2-digit'}).format(new Date(ms));

function Reproduccion({modelos}){
  const [R, setR] = useState(null), [err, setErr] = useState(null);
  const [k, setK] = useState(71), [play, setPlay] = useState(false), [vel, setVel] = useState(1), [modo, setModo] = useState('on');
  useEffect(()=>{ cargarJSON('replay_2026.json').then(setR).catch(x=>setErr(String(x))); },[]);
  const P = useMemo(()=> R && modelos.lluvia && modelos.fv ? prepararReplay(R, modelos.lluvia, modelos.fv) : null, [R, modelos]);
  useEffect(()=>{
    if(!play || !P) return;
    const id = setInterval(()=> setK(v => { const nv = v + vel; if(nv >= P.out.length-1){ setPlay(false); return P.out.length-1; } return nv; }), 120);
    return ()=> clearInterval(id);
  }, [play, vel, P]);
  const win = useMemo(()=>{
    if(!P) return null;
    const a = Math.max(0, k-71), sl = P.out.slice(a, k+1);
    const labels = sl.map(r => { const d = new Date(r.ms); return `${d.getUTCDate()}/${d.getUTCMonth()+1} ${d.getUTCHours()}h`; });
    const pk = modo==='on' ? 'pOn' : 'pBase';
    return {
      lluvia:{type:'bar', data:{labels, datasets:[
        {type:'line', label:'Probabilidad pronosticada (%)', data:sl.map(r=>+(r[pk]*100).toFixed(1)), borderColor:getVar('--ok'), borderWidth:2, pointRadius:0, tension:.25, order:1},
        {type:'line', label:`Umbral de alerta (${fmt(modelos.lluvia.umbral*100,0)} %)`, data:sl.map(()=>modelos.lluvia.umbral*100), borderColor:getVar('--ink-dim'), borderWidth:1, borderDash:[4,4], pointRadius:0, order:2},
        {type:'bar', label:'Llovió en las 3 h siguientes', data:sl.map(r=> r.y ? 100 : null), backgroundColor:'rgba(150,101,11,0.18)', barPercentage:1, categoryPercentage:1, order:3},
      ]}, options:{...baseOptions('%', {y:{min:0, max:100}})}},
      fv:{type:'line', data:{labels, datasets:[
        {label:'Real (t + 1 h)', data:sl.map(r=>+r.pvReal.toFixed(3)), borderColor:getVar('--ink'), borderWidth:2, pointRadius:0, tension:.2},
        {label:'Pronóstico del modelo', data:sl.map(r=>+r.pvHat.toFixed(3)), borderColor:getVar('--ok'), borderWidth:2, pointRadius:0, tension:.2},
        {label:'Persistencia', data:sl.map(r=>+r.pvPers.toFixed(3)), borderColor:getVar('--brand'), borderWidth:1.5, borderDash:[5,3], pointRadius:0, tension:.2},
      ]}, options:{...baseOptions('kW', {y:{min:0}})}},
    };
  }, [P, k, modo]);
  if(err) return e('div',{className:'panel'}, e('div',{className:'err-box'}, 'No se pudo cargar la serie de reproducción: '+err));
  if(!P) return e('div',{className:'panel'}, 'Cargando reproducción…');
  const r = P.out[k], pk = modo==='on' ? 'pOn' : 'pBase', p = r[pk], yh = p >= modelos.lluvia.umbral ? 1 : 0;
  const c = P.acc[modo][k], f1 = f1De(c), fv = P.fvAcc[k];
  const acierto = yh === r.y;
  return e('div',{className:'panel'},
    e('h2',{className:'panel-title'}, 'Reproducción hora a hora', e('span',{className:'tag'},'DATOS REALES 2026')),
    e('p',{className:'panel-sub'}, 'El mismo código que usa la pestaña "Hoy" recorre, hora por hora, enero–agosto de 2026: pronostica, espera a ver qué pasó, lo verifica y (en modo "aprendizaje en línea") se corrige. Al llegar al final los acumulados coinciden con la tabla de validación.'),
    e('div',{className:'replay-controls'},
      e('button',{className:'btn-refresh', onClick:()=>{ if(k>=P.out.length-1) setK(0); setPlay(!play); }}, play ? '⏸ Pausar' : '▶ Reproducir'),
      e('select',{value:vel, onChange:ev=>setVel(+ev.target.value), 'aria-label':'Velocidad'},
        e('option',{value:1},'≈ 8 h por segundo'), e('option',{value:3},'≈ 1 día por segundo'), e('option',{value:24},'≈ 8 días por segundo')),
      e('input',{type:'range', min:0, max:P.out.length-1, value:k, onChange:ev=>{ setPlay(false); setK(+ev.target.value); }, 'aria-label':'Hora de la reproducción'}),
      e('span',{className:'replay-time'}, fechaReplay(r.ms)),
      e('div',{className:'segmented'},
        e('button',{className:modo==='base'?'active':'', onClick:()=>setModo('base')}, 'Modelo base'),
        e('button',{className:modo==='on'?'active':'', onClick:()=>setModo('on')}, 'Con aprendizaje en línea'))),
    e('div',{className:'replay-grid'},
      e('div',{className:'mini'}, e('span',{className:'q'}, 'Pronóstico de lluvia · 3 h'), e('span',{className:'a'}, `${fmt(p*100,0)} % · ${bandaLluvia(p)}`),
        e('span',{className:'s'}, yh ? 'El sistema emite alerta de lluvia.' : 'Sin alerta de lluvia.')),
      e('div',{className:'mini'}, e('span',{className:'q'}, 'Lo que pasó después'), e('span',{className:'a'}, r.y ? `Llovió ${fmt(r.mm,1)} mm` : 'No llovió'),
        e('span',{className:'s'}, acierto ? e('span',{className:'hit'}, '✓ Acierto') : e('span',{className:'miss'}, yh ? '✗ Falsa alarma' : '✗ Lluvia no detectada'))),
      e('div',{className:'mini'}, e('span',{className:'q'}, 'Potencia FV en 1 h (1 kWp)'), e('span',{className:'a'}, `${fmt(r.pvHat,2)} kW pronosticado`),
        e('span',{className:'s'}, `Real: ${fmt(r.pvReal,2)} kW · error ${fmt(Math.abs(r.pvHat-r.pvReal),2)} kW`))),
    e('div',{className:'charts-grid'},
      e('div',{className:'chart-card'}, e('h4',null,'Lluvia: probabilidad vs. lo ocurrido (últimas 72 h)'), e(ChartBox,{config:win.lluvia})),
      e('div',{className:'chart-card'}, e('h4',null,'Potencia FV: pronóstico vs. real (últimas 72 h)'), e(ChartBox,{config:win.fv}))),
    e('div',{className:'two-col', style:{marginTop:14}},
      e('div',null, e('h4',{style:{fontFamily:'var(--font-body)', fontSize:13, margin:'0 0 8px'}}, `Acumulado lluvia · ${(k+1).toLocaleString('es-CO')} h verificadas · F1 = ${fmt(f1,3)}`),
        e('div',{className:'conf'},
          e('div',{className:'h'}), e('div',{className:'h'},'Llovió'), e('div',{className:'h'},'No llovió'),
          e('div',{className:'h'},'Alerta'), e('div',{className:'good'}, c[0]), e('div',{className:'bad'}, c[1]),
          e('div',{className:'h'},'Sin alerta'), e('div',{className:'bad'}, c[2]), e('div',{className:'good'}, c[3]))),
      e('div',null, e('h4',{style:{fontFamily:'var(--font-body)', fontSize:13, margin:'0 0 8px'}}, `Acumulado FV · ${fv.n.toLocaleString('es-CO')} h diurnas`),
        e('div',{className:'stat-row', style:{gridTemplateColumns:'1fr 1fr'}},
          e('div',{className:'stat'}, e('span',{className:'label'},'RMSE modelo'), e('span',{className:'num'}, isNum(fv.rmse)?fmt(fv.rmse,3)+' kW':'—')),
          e('div',{className:'stat'}, e('span',{className:'label'},'RMSE persistencia'), e('span',{className:'num'}, isNum(fv.rmseP)?fmt(fv.rmseP,3)+' kW':'—')))))
  );
}

function VerificacionEstacion({estado}){
  const v = estado.verificacion || {}, t = v.total || {}, l = t.lluvia || {}, fv = t.fv || {}, rec = (v.recientes||[]).slice().reverse();
  return e('div',{className:'panel'},
    e('h2',{className:'panel-title'}, 'Verificación en vivo con la estación', e('span',{className:'tag tag--ref'},'WS-2902')),
    e('p',{className:'panel-sub'}, 'Pronósticos emitidos por el servidor con los datos de la estación y comparados después con lo que midió. Estas cifras crecen solas a medida que la estación transmite.'),
    (l.n||0) === 0 && (fv.n||0) === 0
      ? e('div',{className:'empty'}, `Aún no hay pronósticos verificados con datos de la estación. Se necesitan 4 horas continuas de lecturas válidas para emitir el primero y 3 horas más para verificarlo. Pendientes: ${v.pendientes||0}.`)
      : e(Fragment,null,
        e('div',{className:'stat-row'},
          e('div',{className:'stat'}, e('span',{className:'label'},'Lluvia verificadas'), e('span',{className:'num'}, l.n), e('span',{className:'sub'}, `VP ${l.vp} · FP ${l.fp} · FN ${l.fn} · VN ${l.vn}`)),
          e('div',{className:'stat'}, e('span',{className:'label'},'F1 / Brier'), e('span',{className:'num'}, isNum(l.f1)?fmt(l.f1,3):'—'), e('span',{className:'sub'}, 'Brier: '+(isNum(l.brier)?fmt(l.brier,3):'—'))),
          e('div',{className:'stat'}, e('span',{className:'label'},'FV verificadas (día)'), e('span',{className:'num'}, fv.n), e('span',{className:'sub'}, 'RMSE '+(isNum(fv.rmse)?fmt(fv.rmse,3)+' kW':'—'))),
          e('div',{className:'stat'}, e('span',{className:'label'},'Persistencia FV'), e('span',{className:'num'}, isNum(fv.rmse_persistencia)?fmt(fv.rmse_persistencia,3):'—'), e('span',{className:'sub'}, 'RMSE de referencia (kW)'))),
        e('div',{className:'table-wrap', style:{marginTop:14}}, e('table',null,
          e('thead',null, e('tr',null, ['Hora','Prob. lluvia','¿Llovió?','Resultado','FV pronóst.','FV real'].map(h=> e('th',{key:h},h)))),
          e('tbody',null, rec.map(p => {
            const pl = parseFloat(p.p_lluvia), y = p.y_lluvia === '' || p.y_lluvia == null ? null : Number(p.y_lluvia);
            const yh = pl >= ((estado.modelo||{}).lluvia||{}).umbral ? 1 : 0;
            return e('tr',{key:p.hora}, e('td',null, fmtFecha(new Date(p.hora))), e('td',{className:'mono'}, pct(pl,0)),
              e('td',null, y===null ? '—' : y ? 'Sí' : 'No'),
              e('td',null, y===null ? 'pendiente' : (yh===y ? e('span',{className:'hit'},'✓') : e('span',{className:'miss'},'✗'))),
              e('td',{className:'mono'}, (p.pv_corr !== '' && p.pv_corr != null) ? fmt(parseFloat(p.pv_corr),2) : '—'), e('td',{className:'mono'}, (p.pv_real !== '' && p.pv_real != null) ? fmt(parseFloat(p.pv_real),2) : '—'));
          }))))));
}

function ValidacionView({estado, modelos}){
  return e(Fragment,null, e(ValidacionHistorica,{V:modelos.validacion}), e(Reproduccion,{modelos}), e(VerificacionEstacion,{estado}));
}

/* =========================================================================
   Vista "Metodología"
   ========================================================================= */
function MetodologiaView({modelos}){
  const ml = modelos.lluvia;
  return e(Fragment,null,
    e('div',{className:'panel'},
      e('h2',{className:'panel-title'}, 'Cómo funciona'),
      e('p',{className:'panel-sub'}, 'Arquitectura sin servidor propio: GitHub Actions hace el trabajo pesado cada 15 minutos y la página solo lee archivos públicos. Las llaves de la API viven en los Secrets del repositorio, nunca en esta página.'),
      e('div',{className:'flow'},
        e('div',{className:'step'}, e('b',null,'1 · Estación'), 'La WS-2902 envía una lectura cada 5 min a AmbientWeather.'),
        e('div',{className:'step'}, e('b',null,'2 · Ingesta'), 'La Action descarga las últimas 24 h por la API REST y convierte a unidades SI.'),
        e('div',{className:'step'}, e('b',null,'3 · Calidad'), 'Rangos físicos, sensor bajo techo, lluvia sin humedad, reinicios y huecos.'),
        e('div',{className:'step'}, e('b',null,'4 · Pronóstico'), 'Serie horaria → variables → lluvia a 3 h y potencia FV a 1 h.'),
        e('div',{className:'step'}, e('b',null,'5 · Retroalimentación'), 'Cada pronóstico se verifica al cumplirse su horizonte y ajusta el modelo.'))),
    e('div',{className:'panel'},
      e('h2',{className:'panel-title'}, 'Modelos y fórmulas'),
      e('div',{className:'two-col'},
        e('div',null, e('h4',{style:{fontFamily:'var(--font-body)', fontSize:13, margin:'0 0 6px'}}, 'Lluvia (próximas 3 h, ≥ 0,2 mm)'),
          e('div',{className:'formula-block'}, 'p = σ(b + Σ wᵢ·zᵢ)', e('br'), 'zᵢ = (xᵢ − μᵢ)/σᵢ', e('br'), 'aprendizaje en línea:', e('br'), 'w ← w − η[c_y(p − y)z + λ(w − w₀)]', e('br'), `η = ${ml?ml.eta:'—'} · λ = ${ml?ml.lambda:'—'} · umbral = ${ml?fmt(ml.umbral,2):'—'}`),
          e('p',{className:'muted-note', style:{marginTop:6}}, 'Variables: ' + (ml ? ml.features.map(k=>NOMBRES[k]||k).join(', ') : '—') + '. El término λ mantiene el modelo cerca del aprendido en 32 meses; c_y es el peso de clase del entrenamiento.')),
        e('div',null, e('h4',{style:{fontFamily:'var(--font-body)', fontSize:13, margin:'0 0 6px'}}, 'Potencia fotovoltaica (1 kWp)'),
          e('div',{className:'formula-block'}, 'T_celda = T + (NOCT − 20)/800 · G,  NOCT = 45 °C', e('br'), 'P = 1 kWp · (G/1000) · [1 − 0,004 (T_celda − 25)]', e('br'), 'P(t+1) = a + b · GB(x),  300 árboles, profundidad 4', e('br'), '(a, b) por mínimos cuadrados recursivos, olvido 0,995'),
          e('p',{className:'muted-note', style:{marginTop:6}}, 'Variables: potencia actual, radiación, temperatura, humedad, cambios de presión y de radiación en 1 h, hora y mes.')))),
    e('div',{className:'panel'},
      e('h2',{className:'panel-title'}, 'Archivos abiertos para verificar'),
      e('ul',{className:'method-list'},
        e('li',null, e('a',{href:DATA+'estado.json', target:'_blank'},'data/estado.json'), ' — última lectura, pronóstico vigente, verificación y estado del modelo.'),
        e('li',null, e('a',{href:DATA+'observaciones.csv', target:'_blank'},'data/observaciones.csv'), ' — todas las lecturas con sus banderas de calidad.'),
        e('li',null, e('a',{href:DATA+'predicciones.csv', target:'_blank'},'data/predicciones.csv'), ' — cada pronóstico emitido y su verificación.'),
        e('li',null, e('a',{href:DATA+'validacion.json', target:'_blank'},'data/validacion.json'), ' — métricas en datos históricos reales.'),
        e('li',null, e('a',{href:DATA+'modelo_lluvia.json', target:'_blank'},'data/modelo_lluvia.json'), ' y ', e('a',{href:DATA+'modelo_fv.json', target:'_blank'},'modelo_fv.json'), ' — parámetros completos de los modelos.'),
        e('li',null, 'scripts/entrenar.py, scripts/actualizar.py, scripts/modelo.py y tests/ en el repositorio del proyecto.'))));
}

/* =========================================================================
   Cabecera y pie
   ========================================================================= */
function Header({tab, setTab, estado, now, onRefresh, refreshing}){
  const hourNow = horaBogota(now), esDia = hourNow >= 6 && hourNow < 18;
  const fechaTxt = new Intl.DateTimeFormat('es-CO',{timeZone:TZ, weekday:'long', day:'numeric', month:'long'}).format(now);
  const u = estado && estado.ultima_lectura, t = u ? new Date(u.t_utc) : null, edad = t ? (now - t)/60000 : Infinity;
  const estadoTxt = !t ? 'Sin datos' : edad <= 20 ? `En vivo · ${relTime(t, now.getTime())}` : edad <= 120 ? `Reciente · ${relTime(t, now.getTime())}` : `Sin transmisión · última ${relTime(t, now.getTime())}`;
  return e('header',{className:'app-header'},
    e('div',{className:'header-inner'},
      e('div',{className:'brand'},
        e('div',{className:'brand-mark'}, e(SunIcon,null)),
        e('div',null, e('h1',null,'Estancioncita Aurea'),
          e('p',{className:'loc'}, e(esDia?SunIcon:MoonIcon, {className:'daynight', style:{color: esDia?'var(--brand)':'var(--ink-dim)'}}),
            'Duitama, Boyacá (2.590 msnm) · ', fechaTxt, ' · ', fmtHora(now)))),
      e('div',{className:'header-right'},
        e('span',{className:'status-pill'}, e('span',{className:`dot ${edad<=120?'on':'off'}`}), estadoTxt),
        e('button',{className:'btn-refresh'+(refreshing?' spin':''), onClick:onRefresh, disabled:refreshing}, e(RefreshIcon,null), refreshing?'Actualizando…':'Actualizar'),
        e('div',{className:'segmented'},
          [['hoy','Hoy'],['validacion','Validación'],['metodo','Metodología']].map(([k,t]) => e('button',{key:k, className:tab===k?'active':'', onClick:()=>setTab(k)}, t))))));
}
function Footer(){
  return e('footer',{className:'app-footer'},
    e('div',{className:'footer-inner'},
      e('div',null, e('strong',null,'Proyecto GRIDSE / Semillero AUREA'), ' — UPTC Seccional Duitama. Predicción de lluvia y generación fotovoltaica con la estación Ambient Weather WS-2902, con verificación y aprendizaje continuo.'),
      e('details',null, e('summary',null,'Origen de los datos y alcance de las recomendaciones'),
        e('p',{style:{margin:'0 0 6px'}}, '"Hoy" y "Verificación en vivo" usan solo mediciones de la estación. "Validación con datos históricos" y "Reproducción" usan el reanálisis ERA5 en las coordenadas de la estación y están rotulados como tales. No se muestra ningún dato simulado.'),
        e('p',{style:{margin:0}}, 'Las recomendaciones son apoyo a la decisión, no un pronóstico oficial ni asesoría técnica personalizada.'))));
}

/* =========================================================================
   App
   ========================================================================= */
function App(){
  const [estado, setEstado] = useState(null), [modelos, setModelos] = useState({});
  const [error, setError] = useState(null), [tab, setTab] = useState('hoy');
  const [expandedCard, setExpandedCard] = useState(null), [now, setNow] = useState(new Date()), [refreshing, setRefreshing] = useState(false);

  async function cargarEstado(){
    setRefreshing(true);
    try{ setEstado(await cargarJSON('estado.json')); setError(null); }
    catch(err){ setError(String(err)); }
    finally{ setRefreshing(false); setNow(new Date()); }
  }
  async function cargarModelos(){
    try{
      const [lluvia, fv, validacion] = await Promise.all([cargarJSON('modelo_lluvia.json'), cargarJSON('modelo_fv.json'), cargarJSON('validacion.json')]);
      setModelos({lluvia, fv, validacion});
    }catch(err){ setError(String(err)); }
  }
  useEffect(()=>{
    cargarEstado(); cargarModelos();
    const a = setInterval(cargarEstado, REFRESCO_MS), b = setInterval(()=>setNow(new Date()), 30*1000);
    const c = setInterval(cargarModelos, 30*60*1000);          // el modelo cambia con cada verificación
    return ()=>{ clearInterval(a); clearInterval(b); clearInterval(c); };
  },[]);

  let cuerpo;
  if(error && !estado) cuerpo = e('div',{className:'err-box'}, e('strong',null,'No se pudieron cargar los datos. '), error,
    '. Si abriste el archivo directamente desde el disco, sírvelo con un servidor local (por ejemplo: python -m http.server) o ábrelo desde GitHub Pages.');
  else if(!estado) cuerpo = e('div',{className:'panel'}, 'Cargando datos de la estación…');
  else if(tab==='hoy') cuerpo = e(HoyView,{estado, modelos, now, expandedCard, onToggleCard:id=>setExpandedCard(p=>p===id?null:id), irA:setTab});
  else if(tab==='validacion') cuerpo = modelos.lluvia ? e(ValidacionView,{estado, modelos}) : e('div',{className:'panel'},'Cargando modelos…');
  else cuerpo = e(MetodologiaView,{modelos});

  return e(Fragment, null,
    e(Header,{tab, setTab, estado, now, onRefresh:cargarEstado, refreshing}),
    e('main',{className:'shell'}, cuerpo),
    e(Footer,null));
}

ReactDOM.createRoot(document.getElementById('root')).render(e(App));
