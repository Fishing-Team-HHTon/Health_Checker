/* Minimal timeseries chart on canvas (white/blue/blue theme) */
class TimeSeriesChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = [];           // [{t: ms, v: number}]
    this.capacity = opts.capacity || 1000;
    this.lineWidth = opts.lineWidth || 2;
    this.color = opts.color || '#0d6efd';
    this.grid = opts.grid ?? true;
    this.smooth = opts.smooth ?? true;
    this.autoY = opts.autoY ?? true;
    this.ymin = 0;
    this.ymax = 1;
    this.margin = {l: 48, r: 12, t: 8, b: 26};

    const resize = () => this._resize();
    this._resize();
    window.addEventListener('resize', resize);
    this._cleanup = () => window.removeEventListener('resize', resize);
  }

  destroy(){ this._cleanup?.(); }

  setOptions(o){
    if ('capacity' in o) this.capacity = Math.max(10, +o.capacity);
    if ('lineWidth' in o) this.lineWidth = +o.lineWidth;
    if ('grid' in o) this.grid = !!o.grid;
    if ('smooth' in o) this.smooth = !!o.smooth;
    if ('autoY' in o) this.autoY = !!o.autoY;
    if ('ymin' in o) this.ymin = +o.ymin;
    if ('ymax' in o) this.ymax = +o.ymax;
    this.draw();
  }

  push(point){                 // point: {t: ms, v: number}
    this.data.push(point);
    if (this.data.length > this.capacity) this.data.splice(0, this.data.length - this.capacity);
    this.draw();
  }

  _resize(){
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(100, Math.floor(rect.width * dpr));
    this.canvas.height = Math.max(100, Math.floor(rect.height * dpr));
    this.ctx.setTransform(dpr,0,0,dpr,0,0);
    this.draw();
  }

  _calcYRange(){
    if (!this.data.length) return {min:0, max:1};
    if (this.autoY) {
      let min = +Infinity, max = -Infinity;
      for (const p of this.data){ if (isFinite(p.v)){ if (p.v<min) min=p.v; if (p.v>max) max=p.v; } }
      if (min===max){ min-=1; max+=1; }
      // small padding
      const pad = (max-min)*0.1 || 1;
      return {min:min-pad, max:max+pad};
    }
    return {min:this.ymin, max:this.ymax};
  }

  draw(){
    const ctx = this.ctx;
    const {width:W, height:H} = this.canvas.getBoundingClientRect();
    ctx.clearRect(0,0,W,H);

    // frame
    const m = this.margin;
    const plotW = W - m.l - m.r;
    const plotH = H - m.t - m.b;

    // bg
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0,0,W,H);

    // axes & grid
    const yr = this._calcYRange();
    const x0 = m.l, y0 = m.t, x1 = m.l + plotW, y1 = m.t + plotH;

    if (this.grid){
      ctx.strokeStyle = '#e6eef7';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const vTicks = 6, hTicks = 10;
      for (let i=0;i<=vTicks;i++){
        const y = y0 + (plotH * i / vTicks);
        ctx.moveTo(x0, y); ctx.lineTo(x1, y);
      }
      for (let i=0;i<=hTicks;i++){
        const x = x0 + (plotW * i / hTicks);
        ctx.moveTo(x, y0); ctx.lineTo(x, y1);
      }
      ctx.stroke();

      // y labels
      ctx.fillStyle = '#6b7b8d';
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let i=0;i<=vTicks;i++){
        const val = yr.max - (yr.max-yr.min)*i/vTicks;
        const y = y0 + (plotH * i / vTicks);
        ctx.fillText(val.toFixed(0), x0-6, y);
      }
    }

    if (this.data.length < 2) return;

    // mapping
    const tmin = this.data[0].t, tmax = this.data[this.data.length-1].t || (tmin+1);
    const tx = t => x0 + (plotW * (t - tmin) / (tmax - tmin || 1));
    const ty = v => y0 + plotH - (plotH * (v - yr.min) / (yr.max - yr.min || 1));

    // path
    ctx.lineWidth = this.lineWidth;
    ctx.strokeStyle = this.color;
    ctx.beginPath();
    const pts = this.data.map(p => ({x: tx(p.t), y: ty(p.v)}));

    if (this.smooth && pts.length>2){
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i=1;i<pts.length-1;i++){
        const xc = (pts[i].x + pts[i+1].x)/2;
        const yc = (pts[i].y + pts[i+1].y)/2;
        ctx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
      }
      ctx.quadraticCurveTo(pts[pts.length-1].x, pts[pts.length-1].y,
                           pts[pts.length-1].x, pts[pts.length-1].y);
    } else {
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.stroke();
  }
}

window.TimeSeriesChart = TimeSeriesChart;
