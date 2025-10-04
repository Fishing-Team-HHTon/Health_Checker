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
  const type = (qp('type') || defaultType).toLowerCase(); // ecg|emg|ppg|resp
  const humanMap = {ecg:'ЭКГ', emg:'ЭМГ', ppg:'ФПГ', resp:'Дыхание'};
  const human = humanMap[type] || 'Измерение';

  document.title = `BioSignals — ${human}`;
  $('#measure-title').textContent = human;
  $('#subtitle').textContent = 'value 1';

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

  // Data source (симуляция). Частота задается optFs (Гц).
  let simTimer = null;
  function startSim(){
    if (simTimer) clearInterval(simTimer);
    const fs = Math.max(1, Math.min(1000, +optFs.value|0));
    const dt = 1000/fs;
    let phase = 0;
    simTimer = setInterval(()=>{
      const t = Date.now();

      let v = 0;
      if (type === 'ecg'){
        phase += 0.03;
        v = 0.2*Math.sin(phase) + 0.05*Math.sin(phase*3) + (Math.random()-0.5)*0.02;
        if (Math.random()<0.02) v += 1 + Math.random()*0.3; // R-пик
      } else if (type === 'emg'){
        v = (Math.random()*2-1)*0.5;            // шумоподобный
      } else if (type === 'ppg'){
        phase += 0.05;
        v = 0.6 + 0.35*Math.max(0, Math.sin(phase)) + (Math.random()-0.5)*0.02;
      } else { // resp
        phase += 0.02;
        v = 0.5 + 0.4*Math.sin(phase) + (Math.random()-0.5)*0.01;
      }

      chart.push({t, v});
      rawPush({t, v});
    }, dt);
  }
  startSim();
  optFs.addEventListener('change', startSim);

  // Raw data window
  const rawBox = $('#raw');
  const rawBuf = [];
  function rawPush(p){
    rawBuf.push(`${new Date(p.t).toLocaleTimeString()}  ${p.v.toFixed(4)}`);
    if (rawBuf.length>50) rawBuf.splice(0, rawBuf.length-50);
    rawBox.textContent = rawBuf.join('\n');
    rawBox.scrollTop = rawBox.scrollHeight;
  }

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
