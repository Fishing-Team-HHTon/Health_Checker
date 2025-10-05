// Helpers
function $(sel){ return document.querySelector(sel); }
function fmtTime(ms){
  const s = ms/1000;
  const m = Math.floor(s/60);
  const r = s - m*60;
  return `${String(m).padStart(2,'0')}:${r.toFixed(1).padStart(4,'0')}`;
}
function download(filename, content, mime='text/plain'){
  const blob = new Blob([content], {type: mime + ';charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}
function qp(name, dflt=''){ return new URLSearchParams(location.search).get(name) || dflt; }

(function init(){
  const isMeasurement = !!document.getElementById('chart');
  if (isMeasurement) initMeasurement();
})();

function initMeasurement(){
  // теперь берём тип из ?type или из data-type на <body>
  const defaultType = (document.body && document.body.dataset && document.body.dataset.type) || 'ecg';
  let currentType = (qp('type') || defaultType).toLowerCase(); // ecg|emg|ppg|resp
  const humanMap = {ecg:'ЭКГ', emg:'ЭМГ', ppg:'ФПГ', resp:'Дыхание'};
  const dataAttrs = (document.body && document.body.dataset) || {};

  function sanitizeBase(base){
    if (!base) return '';
    return base.toString().trim().replace(/\/$/, '');
  }

  function resolveHttpBase(){
    const explicit = sanitizeBase(window.__BACKEND_HTTP_BASE__ || window.BACKEND_HTTP_BASE || dataAttrs.backendHttpBase);
    if (explicit) return explicit;

    const proto = window.location && window.location.protocol;
    const host = window.location && window.location.host;
    if (proto && host){
      return `${proto}//${host}`.replace(/\/$/, '');
    }
    return 'http://127.0.0.1:8000';
  }

  function resolveWsBase(httpBase){
    const explicit = sanitizeBase(window.__BACKEND_WS_BASE__ || window.BACKEND_WS_BASE || dataAttrs.backendWsBase);
    if (explicit) return explicit;

    if (httpBase.startsWith('https://')){
      return `wss://${httpBase.slice('https://'.length)}`;
    }
    if (httpBase.startsWith('http://')){
      return `ws://${httpBase.slice('http://'.length)}`;
    }

    const proto = window.location && window.location.protocol;
    const host = window.location && window.location.host;
    if (proto && host){
      const wsProto = proto === 'https:' ? 'wss' : 'ws';
      return `${wsProto}://${host}`;
    }
    return 'ws://127.0.0.1:8000';
  }

  const backendHttpBase = resolveHttpBase();
  const backendWsBase = resolveWsBase(backendHttpBase);

  const titleEl = $('#measure-title');
  const subtitleEl = $('#subtitle');

  function humanName(mode){
    return humanMap[mode] || 'Измерение';
  }

  function updateTypeContext(mode){
    const label = humanName(mode);
    document.title = `BioSignals — ${label}`;
    if (titleEl) titleEl.textContent = label;
    if (document.body && document.body.dataset) document.body.dataset.type = mode;
    return label;
  }

  function setStatus(message, tone = 'info'){
    if (!subtitleEl) return;
    subtitleEl.textContent = message;
    subtitleEl.classList.remove('status-ok', 'status-error');
    if (tone === 'error') subtitleEl.classList.add('status-error');
    else if (tone === 'ok') subtitleEl.classList.add('status-ok');
  }

  let human = updateTypeContext(currentType);
  setStatus('Синхронизация режима…');

  // Chart
  const chart = new TimeSeriesChart($('#chart'), {capacity: 1000, lineWidth: 2, smooth: true, grid: true, autoY: true});

  // Options UI
  const optSmooth = $('#opt-smooth');
  const optGrid   = $('#opt-grid');
  const optWidth  = $('#opt-width');
  const optAutoY  = $('#opt-autoY');
  const optYmin   = $('#opt-ymin');
  const optYmax   = $('#opt-ymax');
  const manualY   = $('#manualY');
  const optCap    = $('#opt-cap');
  const optFs     = $('#opt-fs');

  function applyOpts(){
    chart.setOptions({
      smooth: optSmooth.checked,
      grid: optGrid.checked,
      lineWidth: optWidth.value,
      autoY: optAutoY.checked,
      ymin: optYmin.value,
      ymax: optYmax.value,
      capacity: optCap.value
    });
    manualY.classList.toggle('hidden', optAutoY.checked);
  }
  [optSmooth,optGrid,optWidth,optAutoY,optYmin,optYmax,optCap].forEach(el => el.addEventListener('input', applyOpts));
  applyOpts();

  async function setBackendMode(modeType){
    const normalized = (modeType || '').toString().trim().toLowerCase();
    if (!normalized){
      console.warn('setBackendMode: empty mode, skip');
      return false;
    }
    const url = `${backendHttpBase}/api/mode`;
    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: normalized})
      });
      if (!resp.ok){
        let detail = '';
        try { detail = await resp.text(); } catch (_) {}
        console.error(`setBackendMode: backend responded with ${resp.status}`, detail);
        return false;
      }
      return true;
    } catch (err){
      console.error('setBackendMode: request failed', err);
      return false;
    }
  }

  window.setBackendMode = setBackendMode;

  // Data source (симуляция). Частота задается optFs (Гц).
  let simTimer = null;
  function stopSim(){
    if (simTimer){
      clearInterval(simTimer);
      simTimer = null;
    }
  }
  function startSim(){
    stopSim();
    const fs = Math.max(1, Math.min(1000, +optFs.value|0));
    const dt = 1000/fs;
    let phase = 0;
    simTimer = setInterval(()=>{
      const t = Date.now();

      let v = 0;
      if (currentType === 'ecg'){
        phase += 0.03;
        v = 0.2*Math.sin(phase) + 0.05*Math.sin(phase*3) + (Math.random()-0.5)*0.02;
        if (Math.random()<0.02) v += 1 + Math.random()*0.3; // R-пик
      } else if (currentType === 'emg'){
        v = (Math.random()*2-1)*0.5;            // шумоподобный
      } else if (currentType === 'ppg'){
        phase += 0.05;
        v = 0.6 + 0.35*Math.max(0, Math.sin(phase)) + (Math.random()-0.5)*0.02;
      } else { // resp
        phase += 0.02;
        v = 0.5 + 0.4*Math.sin(phase) + (Math.random()-0.5)*0.01;
      }

      const point = {t, v};
      chart.push(point);
      rawPush(point);
    }, dt);
  }
  optFs.addEventListener('change', ()=>{ if (simTimer) startSim(); });

  // Raw data window
  const rawBox = $('#raw');
  const rawBuf = [];
  function rawPush(p){
    rawBuf.push(`${new Date(p.t).toLocaleTimeString()}  ${p.v.toFixed(4)}`);
    if (rawBuf.length>50) rawBuf.splice(0, rawBuf.length-50);
    rawBox.textContent = rawBuf.join('\n');
    rawBox.scrollTop = rawBox.scrollHeight;
  }
  function resetRaw(){
    rawBuf.length = 0;
    rawBox.textContent = '';
  }

  // Real data stream
  let ws = null;
  let reconnectTimer = null;
  let reconnectDelay = 1000;
  let connectEpoch = 0;

  function scheduleReconnect(){
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(()=>{
      reconnectTimer = null;
      connectStream().catch(err => console.error('Reconnect failed', err));
    }, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
  }

  async function connectStream(){
    const epoch = ++connectEpoch;

    if (reconnectTimer){
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    if (ws){
      try { ws.close(); } catch (e) {}
      ws = null;
    }

    const signalType = currentType;
    const label = humanName(signalType);

    setStatus('Синхронизация режима…');

    let modeOk = false;
    try {
      modeOk = await setBackendMode(signalType);
    } catch (err){
      console.error('connectStream: setBackendMode failed', err);
    }

    if (epoch !== connectEpoch){
      return;
    }

    if (!modeOk){
      setStatus('Не удалось обновить режим бэкенда — ожидаем поток…', 'error');
    } else {
      setStatus('Подключение к потоку…');
    }

    let socket;
    try {
      socket = new WebSocket(`${backendWsBase}/ws/${signalType}`);
    } catch (err) {
      if (epoch !== connectEpoch) return;
      console.warn('WebSocket init failed, fallback to simulation', err);
      setStatus('Не удалось открыть поток — включена симуляция', 'error');
      startSim();
      scheduleReconnect();
      return;
    }

    if (epoch !== connectEpoch){
      try { socket.close(); } catch (_) {}
      return;
    }

    ws = socket;

    socket.addEventListener('open', () => {
      if (epoch !== connectEpoch){
        try { socket.close(); } catch (_) {}
        return;
      }
      if (reconnectTimer){
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      reconnectDelay = 1000;
      stopSim();
      setStatus(`Поток ${label} активен`, 'ok');
    });

    socket.addEventListener('message', (event) => {
      if (epoch !== connectEpoch || currentType !== signalType) return;

      let payload = event.data;
      if (payload && typeof payload === 'string'){
        try {
          payload = JSON.parse(payload);
        } catch (e){
          console.warn('Invalid JSON payload', e);
          return;
        }
      }

      if (!payload || payload.type !== 'batch' || !Array.isArray(payload.samples)) return;
      if (payload.mode && payload.mode !== signalType) return;

      payload.samples.forEach(sample => {
        const v = Number(sample);
        if (!Number.isFinite(v)) return;
        const point = { t: Date.now(), v };
        chart.push(point);
        rawPush(point);
      });
    });

    socket.addEventListener('error', (err) => {
      if (epoch !== connectEpoch) return;
      console.error('WebSocket error', err);
    });

    socket.addEventListener('close', () => {
      if (epoch !== connectEpoch) return;
      ws = null;
      setStatus('Поток недоступен — включена симуляция', 'error');
      startSim();
      scheduleReconnect();
    });
  }

  connectStream().catch(err => console.error('Initial stream connection failed', err));

  async function changeSignal(nextType){
    const normalized = (nextType || '').toString().trim().toLowerCase();
    if (!normalized || normalized === currentType) return;

    currentType = normalized;
    human = updateTypeContext(currentType);
    resetRaw();
    chart.data.length = 0;
    chart.draw();
    reconnectDelay = 1000;
    if (reconnectTimer){
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    stopSim();
    setStatus('Переподключение к новому сигналу…');

    try {
      await connectStream();
    } catch (err){
      console.error('Failed to switch signal', err);
    }
  }

  window.changeMeasurementSignal = changeSignal;

  // Recording
  const btnRec = $('#btn-record');
  const durInput = $('#rec-duration');
  const prog = $('#rec-progress');
  const timer = $('#rec-timer');
  const btnDl = $('#btn-download');
  const selFmt = $('#dl-format');
  const btnAn  = $('#btn-analyze');
  const analysis = $('#analysis');

  let recording = false;
  let recStart = 0;
  let recTimer = null;
  let recorded = []; // {t, v}

  function setRecording(on){
    recording = on;
    btnRec.textContent = on ? 'Остановить' : 'Начать запись';
    btnRec.classList.toggle('primary', !on);
    btnDl.disabled = on || recorded.length===0;
    prog.value = 0; prog.max = 100;
    timer.textContent = '00:00.0';
    analysis.textContent = '';
  }

  btnRec.addEventListener('click', ()=>{
    if (!recording){
      recorded = [];
      const ms = Math.max(1, +durInput.value) * 1000;
      recStart = performance.now();
      setRecording(true);
      const stopAt = recStart + ms;
      const cap = () => {
        if (!recording) return;
        const last = chart.data[chart.data.length-1];
        if (last) recorded.push({...last});
        const now = performance.now();
        const elapsed = now - recStart;
        prog.max = ms; prog.value = Math.min(ms, elapsed);
        timer.textContent = fmtTime(elapsed);
        if (now >= stopAt){
          setRecording(false);
          clearInterval(recTimer);
          btnDl.disabled = recorded.length===0;
          timer.textContent = fmtTime(ms);
        }
      };
      recTimer = setInterval(cap, 1000/25);
    } else {
      setRecording(false);
      clearInterval(recTimer);
      btnDl.disabled = recorded.length===0;
    }
  });

  // Download
  btnDl.addEventListener('click', ()=>{
    if (!recorded.length) return;
    const typeName = human.toLowerCase();
    const ts = new Date().toISOString().replace(/[:.]/g,'-');
    const fmt = selFmt.value;
    if (fmt === 'csv'){
      const header = 'timestamp,value\n';
      const body = recorded.map(p => `${new Date(p.t).toISOString()},${p.v}`).join('\n');
      download(`${typeName}_${ts}.csv`, header+body, 'text/csv');
    } else if (fmt === 'json'){
      const json = JSON.stringify(recorded.map(p => ({timestamp:new Date(p.t).toISOString(), value:p.v})), null, 2);
      download(`${typeName}_${ts}.json`, json, 'application/json');
    } else {
      const txt = recorded.map(p => `${new Date(p.t).toISOString()} ${p.v}`).join('\n');
      download(`${typeName}_${ts}.txt`, txt, 'text/plain');
    }
  });

  // Analysis (min/max/avg/RMS)
  btnAn.addEventListener('click', ()=>{
    if (!chart.data.length){ analysis.textContent = 'Нет данных.'; return; }
    const arr = chart.data.map(p=>p.v);
    const n = arr.length;
    const sum = arr.reduce((a,b)=>a+b,0);
    const avg = sum/n;
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const rms = Math.sqrt(arr.reduce((a,b)=>a+b*b,0)/n);
    analysis.innerHTML =
      `<div class="panel" style="padding:10px;margin-top:8px">
        <div><b>Статистика окна:</b></div>
        <div>min: ${min.toFixed(4)} | max: ${max.toFixed(4)} | avg: ${avg.toFixed(4)} | RMS: ${rms.toFixed(4)}</div>
      </div>`;
  });

  // Для реального источника: вместо startSim() читайте поток и вызывайте chart.push({t:Date.now(), v:value})
}
