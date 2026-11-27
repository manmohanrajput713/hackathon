'use client';
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { fmt, saveSvg } from './lib';

export const COLORS = ['#007953', '#aa7551', '#2f5d9b', '#8a4fa3', '#b23b3b', '#4b8a8c', '#c08a00', '#5d6b2f'];

/**
 * Line chart. rows: [{[x]: number|ISO string, ...}], series: [{key, label, color?, dashed?, dots?}].
 * bands: [{from, to, label}] shaded x ranges; markers: [{x, label}] vertical lines; points: [{x, y, label}] highlights;
 * cursor: controlled row index (replay); threshold: horizontal line.
 */
export function LineChart({rows, x, series, height = 230, xLabel, unit, threshold, bands = [], markers = [], points = [], cursor, onCursor, title, yLabel}) {
  const [hover, setHover] = useState(null);
  const [menu, setMenu] = useState(false);
  const ref = useRef(null), observer = useRef(null);
  const [width, setWidth] = useState(1000);
  const clipId = 'reveal' + useId().replace(/[^w]/g, '');
  // Callback ref: follow whichever element is mounted (empty state or chart) so text stays legible at any width.
  const box = useCallback(node => {
    observer.current?.disconnect();
    if (!node || typeof ResizeObserver === 'undefined') return;
    observer.current = new ResizeObserver(([e]) => setWidth(Math.max(320, Math.round(e.contentRect.width))));
    observer.current.observe(node);
  }, []);
  if (!rows?.length) return <div className="chartEmpty" ref={box}>No data to plot yet.</div>;
  const isTime = typeof rows[0][x] === 'string';
  const toX = v => isTime ? Date.parse(v) : Number(v);
  const xs = rows.map(r => toX(r[x]));
  const values = rows.flatMap(r => series.map(s => r[s.key])).filter(v => v != null && Number.isFinite(v));
  points.forEach(p => p.y != null && values.push(p.y));
  if (threshold != null) values.push(threshold);
  if (!values.length) return <div className="chartEmpty" ref={box}>No numeric values to plot.</div>;
  let min = Math.min(...values), max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.06;
  min = min >= 0 ? Math.max(0, min - pad) : min - pad; max += pad;
  const W = width, H = height, L = 60, R = 14, T = yLabel ? 32 : 14, B = 30;  // room above the plot for the y-axis title
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const px = v => L + ((v - x0) / ((x1 - x0) || 1)) * (W - L - R);
  const py = v => T + (1 - (v - min) / (max - min)) * (H - T - B);
  const label = v => isTime ? new Date(v).toLocaleDateString(undefined, {year: '2-digit', month: 'short', day: 'numeric'}) : fmt(v);
  function move(e) {
    const box = ref.current.getBoundingClientRect();
    const vx = x0 + ((e.clientX - box.left) / box.width * W - L) / (W - L - R) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < xs.length; i++) if (Math.abs(xs[i] - vx) < Math.abs(xs[best] - vx)) best = i;
    setHover(best);
  }
  // During a replay the curves are drawn only up to the cursor, so the graph unfolds as it plays.
  const last = cursor != null ? Math.min(cursor, rows.length - 1) : rows.length - 1;
  const path = key => {
    let d = '', pen = false;
    rows.forEach((r, i) => {
      if (i > last) return;
      const v = r[key];
      if (v == null || !Number.isFinite(v)) { pen = false; return; }
      d += (pen ? 'L' : 'M') + px(xs[i]).toFixed(1) + ',' + py(v).toFixed(1);
      pen = true;
    });
    return d;
  };
  const active = cursor ?? hover;
  const h = active != null ? rows[Math.min(active, rows.length - 1)] : null;
  const hx = h ? xs[Math.min(active, rows.length - 1)] : null;
  const name = (title || series.map(s => s.label).join(' ')).replace(/[^\w\- ]+/g, '').slice(0, 60) || 'chart';
  let lastLabel = -Infinity;
  const markerLabel = mx => { if (mx - lastLabel < 90) return false; lastLabel = mx; return true; };
  return <div className="lineChart" ref={box}>
    <div className="chartTop">
      <div className="legend">{series.map((s, i) => <span key={s.key}>{s.dots ? <b className="dot" style={{borderColor: s.color || COLORS[i]}}/> : <i style={{borderTopColor: s.color || COLORS[i], borderTopStyle: s.dashed ? 'dashed' : 'solid'}}/>}{s.label}</span>)}
        {threshold != null && <span><i className="thr"/>Alert threshold</span>}
        {bands.map(b => <span key={b.label}><i className="band"/>{b.label}</span>)}</div>
      <div className="chartMenu"><button className="iconBtn small" aria-label="Export chart" onClick={() => setMenu(!menu)}>⤓</button>
        {menu && <div className="menuPop" onMouseLeave={() => setMenu(false)}>
          <button onClick={() => { saveSvg(ref.current, name, true); setMenu(false); }}>Download PNG</button>
          <button onClick={() => { saveSvg(ref.current, name, false); setMenu(false); }}>Download SVG</button></div>}</div>
    </div>
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="chart" onMouseMove={move} onMouseLeave={() => setHover(null)}
      onClick={() => onCursor && hover != null && onCursor(hover)} role="img" aria-label={series.map(s => s.label).join(', ')}>
      {bands.map(b => { const a = px(toX(b.from)), z = px(toX(b.to)); return <g key={b.label}><rect x={Math.min(a, z)} y={T} width={Math.abs(z - a)} height={H - T - B} fill="#2f5d9b" opacity="0.07"/>
        <text x={Math.min(a, z) + 6} y={T + 14} className="bandLabel">{b.label}</text></g>; })}
      {[0, 1, 2, 3, 4].map(i => { const v = min + (max - min) * i / 4; return <g key={i}><line x1={L} x2={W - R} y1={py(v)} y2={py(v)} stroke="var(--grid)"/><text x={L - 8} y={py(v) + 4} textAnchor="end">{fmt(v, Math.abs(max) < 10 ? 3 : 1)}</text></g>; })}
      {yLabel && <text x={L - 8} y={14} className="axisLabel">{yLabel}</text>}
      {threshold != null && <line x1={L} x2={W - R} y1={py(threshold)} y2={py(threshold)} stroke="#c0392b" strokeDasharray="4 4"/>}
      {markers.map((m, i) => { const mx = px(toX(m.x)); return mx >= L && mx <= W - R && <g key={i}><line x1={mx} x2={mx} y1={T} y2={H - B} stroke={m.color || '#8a4fa3'} strokeDasharray="2 3"/>
        {markerLabel(mx) && <text x={mx + 3} y={H - B - 4} className="markerLabel" fill={m.color || '#8a4fa3'}>{m.label}</text>}<title>{m.label}</title></g>; })}
      <clipPath id={clipId}><rect className="chartReveal" x={L - 4} y={0} width={W - L - R + 8} height={H}/></clipPath>
      <g clipPath={`url(#${clipId})`}>
      {series.map((s, i) => s.dots ?
        <g key={s.key}>{rows.map((r, j) => j <= last && r[s.key] != null && <circle key={j} cx={px(xs[j])} cy={py(r[s.key])} r="4" fill="#fff" stroke={s.color || COLORS[i]} strokeWidth="2"/>)}</g> :
        <path key={s.key} d={path(s.key)} fill="none" stroke={s.color || COLORS[i]} strokeWidth={s.dashed ? 1.8 : 2.4} strokeDasharray={s.dashed ? '6 4' : undefined} vectorEffect="non-scaling-stroke"/>)}
      </g>
      {points.map((p, i) => p.y != null && <g key={i}><circle cx={px(toX(p.x))} cy={py(p.y)} r="6" fill={p.color || '#c0392b'} stroke="#fff" strokeWidth="2"/>
        {p.label && <text x={px(toX(p.x)) + 9} y={py(p.y) - 8} className="pointLabel">{p.label}</text>}</g>)}
      {h && <><line x1={px(hx)} x2={px(hx)} y1={T} y2={H - B} stroke="#849890" strokeDasharray="3 4"/>
        {series.map((s, i) => h[s.key] != null && <circle key={s.key} cx={px(hx)} cy={py(h[s.key])} r="4" fill={s.color || COLORS[i]}/>)}</>}
      <text x={L} y={H - 8}>{label(x0)}</text><text x={W - R} y={H - 8} textAnchor="end">{label(x1)}</text>
      {xLabel && <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle">{xLabel}</text>}
    </svg>
    <div className="readout">{h ? <><b>{isTime ? h[x] : `${fmt(h[x])} ${xLabel || ''}`}</b>{series.map(s => <span key={s.key}>{s.label}: <b>{fmt(h[s.key], 3)}</b>{unit ? ' ' + unit : ''}</span>)}</> : <span>Hover the chart to read values.</span>}</div>
  </div>;
}

/** Play a cursor through a series: returns [index, controls element]. */
export function useReplay(length) {
  const [index, setIndex] = useState(null), [playing, setPlaying] = useState(false), [speed, setSpeed] = useState(1);
  useEffect(() => {
    if (!playing) return;
    const step = Math.max(1, Math.round(length / 200));
    const timer = setInterval(() => setIndex(i => { const n = (i ?? -1) + step * speed; if (n >= length - 1) { setPlaying(false); return length - 1; } return n; }), 60);
    return () => clearInterval(timer);
  }, [playing, length, speed]);
  const controls = <div className="replay">
    <button className="secondary small" onClick={() => { if (index == null || index >= length - 1) setIndex(0); setPlaying(!playing); }}>{playing ? '❚❚ Pause' : '▶ Replay'}</button>
    <input type="range" min="0" max={Math.max(0, length - 1)} value={index ?? length - 1} onChange={e => { setPlaying(false); setIndex(Number(e.target.value)); }} aria-label="Replay position"/>
    <select value={speed} onChange={e => setSpeed(Number(e.target.value))} aria-label="Replay speed"><option value={1}>1×</option><option value={4}>4×</option><option value={16}>16×</option></select>
    {index != null && <button className="link small" onClick={() => { setPlaying(false); setIndex(null); }}>Reset</button>}
  </div>;
  return [index, controls];
}

export function Modal({title, onClose, children, wide}) {
  useEffect(() => {
    const key = e => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [onClose]);
  return <div className="overlay" onMouseDown={e => e.target === e.currentTarget && onClose()}>
    <div className={'modal' + (wide ? ' wide' : '')} role="dialog" aria-modal="true" aria-label={title}>
      <div className="modalHead"><h2>{title}</h2><button className="iconBtn" onClick={onClose} aria-label="Close">×</button></div>
      {children}
    </div>
  </div>;
}

export function Confirm({title, message, action = 'Delete', onConfirm, onClose}) {
  const [busy, setBusy] = useState(false);
  return <Modal title={title} onClose={onClose}>
    <p>{message}</p>
    <div className="formActions"><button className="secondary" onClick={onClose}>Cancel</button>
      <button className="danger" disabled={busy} onClick={async () => { setBusy(true); try { await onConfirm(); onClose(); } finally { setBusy(false); } }}>{busy ? 'Working…' : action}</button></div>
  </Modal>;
}

export function Status({value}) {
  return <span className={'status ' + (value || 'none')}>{{alert: 'Alert', no_alert: 'Normal', unavailable: 'No score'}[value] || 'Not run'}</span>;
}

export function Empty({title, children, action}) {
  return <div className="empty"><h3>{title}</h3><p>{children}</p>{action}</div>;
}

export function Tabs({tabs, active, onChange, small}) {
  return <div className={'tabs' + (small ? ' sub' : '')} role="tablist">{tabs.map(t => <button key={t.id} role="tab" aria-selected={active === t.id} onClick={() => onChange(t.id)}>
    {t.label}{t.count != null && <span className="count">{t.count}</span>}</button>)}</div>;
}

export function Meter({label, used, capacity, unit, binding}) {
  const share = capacity ? Math.min(1, used / capacity) : 0;
  return <div className="meter">
    <div className="meterHead"><span>{label}{binding && <span className="tag bind">Binding</span>}</span><span><b>{fmt(used, 0)}</b> / {capacity == null ? 'no limit' : fmt(capacity, 0)} {unit}</span></div>
    <div className="meterBar"><i style={{width: `${share * 100}%`}} className={share > 0.98 ? 'full' : share > 0.85 ? 'high' : ''}/></div>
  </div>;
}

export function KV({items}) {
  return <dl className="kv">{items.filter(Boolean).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v ?? '—'}</dd></div>)}</dl>;
}

export function Stale({reasons}) {
  if (!reasons?.length) return null;
  return <div className="warning"><b>Needs review:</b> this run's inputs changed after it ran. Its stored inputs are kept as they were.<ul>{reasons.map(r => <li key={r}>{r}</li>)}</ul></div>;
}
