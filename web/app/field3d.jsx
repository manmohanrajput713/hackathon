'use client';
import { useEffect, useRef, useState } from 'react';
import { fmt } from './lib';
import { WellSymbol } from './canvas';
import { startConnectivity3D } from './flow3d';

const DEG = Math.PI / 180;
const HOME = {yaw: -32, pitch: 28, zoom: 1};

function niceStep(span, n = 5) {
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  return [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;
}
const setDepth = w => +(w.settings?.tubing_depth_m ?? w.settings?.pump_depth_m ?? w.settings?.depth_m) || null;
/** Tubing (gas-lift) or pump depth from the well's settings; schema default otherwise. */
export const depthOf = w => setDepth(w) || 2048;
const reducedMotion = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
const sum = a => a.reduce((x, y) => x + y, 0);
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => { const A = hex(a), B = hex(b), k = Math.max(0, Math.min(1, t)); return '#' + A.map((v, i) => Math.round(v + (B[i] - v) * k).toString(16).padStart(2, '0')).join(''); };

/** Light blue for a little water, dark navy for a lot (t in 0..1). */
export const waterColor = t => mix('#a8d4f5', '#082a5e', t);
const OIL = '#8a5a1c', INJ_WATER = '#2f86d6';

/**
 * Injector → producer water channels from a Deep-CRM history match. gains[j][i] is the share of injector i's water
 * that reaches producer j; what is left of each injector's water goes elsewhere (aquifer, unswept rock).
 * Also: the share of the field's injected water arriving at each producer.
 */
export function crmFlows(o, wells) {
  if (!o?.gains || !o.injectors || !o.producers) return null;
  const byKey = new Map();
  wells.forEach(w => { byKey.set(w.code, w); byKey.set(w.name, w); });
  const meanInj = o.injectors.map((_, i) => o.injection?.length ? sum(o.injection.map(r => +r[i] || 0)) / o.injection.length : 1);
  const total = sum(meanInj) || 1;
  const pair = o.tau_selection === 'per-pair';
  const channels = [], injectors = {};
  o.injectors.forEach((q, i) => { const w = byKey.get(q); if (w) injectors[w.id] = {w, share: meanInj[i] / total, out: 0}; });
  o.producers.forEach((p, j) => o.injectors.forEach((q, i) => {
    const f = +o.gains[j]?.[i] || 0, prod = byKey.get(p), inj = byKey.get(q);
    if (inj && injectors[inj.id]) injectors[inj.id].out += f;
    if (prod && inj && f >= 0.01) channels.push({inj, prod, f, tau: pair ? o.tau?.[j]?.[i] : o.tau?.[j]});
  }));
  const best = {};
  channels.forEach(c => { if (!best[c.inj.id] || c.f > best[c.inj.id].f) best[c.inj.id] = c; });
  channels.forEach(c => { c.main = best[c.inj.id] === c; });
  const arriving = {};
  o.producers.forEach((p, j) => { const w = byKey.get(p); if (w) arriving[w.id] = sum(o.injectors.map((_, i) => (+o.gains[j]?.[i] || 0) * meanInj[i])) / total; });
  return {channels, injectors, arriving, maxF: Math.max(0.05, ...channels.map(c => c.f)), maxArr: Math.max(0.05, ...Object.values(arriving))};
}

/**
 * The field in 3D as a geological cut-away: strata from the surface to the basement, the reservoir (oil zone over a
 * water leg), every well as a steel pipe with its fluid moving in it, and the waterflood moving through the rock
 * from injectors to producers (particles on a canvas, weighted by the history-matched connectivity).
 */
export function Field3D({wells, size, selected, alertOf, onSelect, onMenu, onGround, onHover, viewRef, flows, showFlow = true, onChannel}) {
  const [cam, setCam] = useState({yaw: -95, pitch: 82, zoom: 0.8});
  const drag = useRef(null), anim = useRef(0), canvas = useRef(null), gl = useRef(null), scene = useRef(null);
  const W = size.w, H = size.h;

  function flyTo(to, ms = 900) {
    cancelAnimationFrame(anim.current);
    if (reducedMotion()) { setCam(to); return; }
    const from = {...camRef.current}, t0 = performance.now();
    const step = now => {
      const p = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - p, 3);
      setCam({yaw: from.yaw + (to.yaw - from.yaw) * e, pitch: from.pitch + (to.pitch - from.pitch) * e, zoom: from.zoom + (to.zoom - from.zoom) * e});
      if (p < 1) anim.current = requestAnimationFrame(step);
    };
    anim.current = requestAnimationFrame(step);
  }
  const camRef = useRef(cam); camRef.current = cam;
  useEffect(() => { flyTo(HOME, 1300); return () => cancelAnimationFrame(anim.current); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  if (viewRef) viewRef.current = {reset: () => flyTo(HOME), top: () => flyTo({yaw: 0, pitch: 89, zoom: 1}), side: () => flyTo({yaw: 0, pitch: 4, zoom: 1})};

  // ---- World: strata, reservoir and wells ----
  const xs = wells.map(w => w.easting), ys = wells.map(w => w.northing);
  const [x0, x1, y0, y1] = wells.length ? [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)] : [0, 6000, 0, 6000];
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const known = wells.map(setDepth).filter(Boolean);
  const injDepth = known.length ? Math.max(...known) : 2048;
  const depth = w => setDepth(w) || (w.type === 'injector' ? injDepth : 2048);
  const maxD = Math.max(800, ...wells.map(depth));
  const half = Math.max(x1 - x0, y1 - y0, wells.length > 1 ? 1 : 3000, maxD * 1.15) * 0.68;
  const resTop = -maxD * 0.93, resBot = Math.min(-maxD * 1.07, resTop - 90);
  const thick = resTop - resBot, owc = resTop - thick * 0.62, baseZ = resBot - thick * 0.8, Dt = -resTop;
  const LAYERS = [
    {name: 'Soil', z0: 0, z1: -Dt * 0.05, fill: '#9a8560', pat: 'pt-dots'},
    {name: 'Clay & shale', z0: -Dt * 0.05, z1: -Dt * 0.33, fill: '#b19c78', pat: 'pt-shale'},
    {name: 'Sandstone', z0: -Dt * 0.33, z1: -Dt * 0.6, fill: '#ccb88f', pat: 'pt-sand'},
    {name: 'Limestone', z0: -Dt * 0.6, z1: -Dt * 0.86, fill: '#cdc4ad', pat: 'pt-brick'},
    {name: 'Cap rock (seal)', z0: -Dt * 0.86, z1: resTop, fill: '#6a635a', pat: 'pt-shale'},
    {name: 'Oil zone', z0: resTop, z1: owc, fill: '#a8762f', top: '#c99a58', pat: 'pt-sand', res: 'oil'},
    {name: 'Water leg', z0: owc, z1: resBot, fill: '#4b7ea6', pat: 'pt-sand', res: 'water'},
    {name: 'Basement', z0: resBot, z1: baseZ, fill: '#7b756b', pat: 'pt-brick'},
  ];
  const tz = baseZ * 0.47;
  const pinH = half * 0.13;

  // ---- Camera ----
  const yaw = cam.yaw * DEG, pitch = cam.pitch * DEG;
  const c = Math.cos(yaw), s = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const R0 = half * 7, R = R0 / cam.zoom;
  const f = R0 * Math.min(W, H * 1.2) * 0.25 / half;
  const oy = -H * 0.02;
  const P = (x, y, z) => {
    const X = x - cx, Y = y - cy, Z = z - tz;
    const xr = X * c - Y * s, yr = X * s + Y * c;
    const up = Z * cp + yr * sp, d = R + yr * cp - Z * sp;
    const k = f / Math.max(d, R * 0.05);
    return [W / 2 + xr * k, H / 2 + oy - up * k, d];
  };
  const toGround = (sx, sy) => {
    const u = sx - W / 2, v = H / 2 + oy - sy, Z = -tz;
    const den = v * cp - f * sp;
    if (Math.abs(den) < 1e-9) return null;
    const yr = (f * Z * cp - v * (R - Z * sp)) / den;
    const d = R + yr * cp - Z * sp;
    if (d <= 0) return null;
    const xr = u * d / f;
    const X = xr * c + yr * s, Y = -xr * s + yr * c;
    return Math.abs(X) <= half && Math.abs(Y) <= half ? [cx + X, cy + Y] : null;
  };
  const poly = pts => pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const area = pts => pts.reduce((a, p, i) => { const q = pts[(i + 1) % pts.length]; return a + p[0] * q[1] - q[0] * p[1]; }, 0);

  /** Faces of an axis-aligned box (outward winding); `front` = facing the camera. */
  function faces(zTop, zBot) {
    const k = [[-half, -half], [half, -half], [half, half], [-half, half]].map(([a, b]) => [cx + a, cy + b]);
    const top = k.map(([x, y]) => P(x, y, zTop)), bot = k.map(([x, y]) => P(x, y, zBot));
    const list = [{pts: top, kind: 'top'}, {pts: [...bot].reverse(), kind: 'bottom'}];
    for (let i = 0; i < 4; i++) { const j = (i + 1) % 4; list.push({pts: [bot[i], bot[j], top[j], top[i]], kind: 'side', shade: i % 2}); }
    return list.map(fc => ({...fc, front: area(fc.pts) < 0}));
  }
  const strata = LAYERS.map(L => ({...L, faces: faces(L.z0, L.z1)}));
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => [cx + a * half, cy + b * half]);
  const byX = corners.map(q => ({q, p: P(q[0], q[1], 0)})).sort((a, b) => a.p[0] - b.p[0]);
  const [ex, ey] = byX[0].q, [rx, ry] = byX[3].q;                  // depth scale on the left, layer names on the right
  const gStep = niceStep(half * 2, 6);
  const gridLines = [];
  for (let g = Math.ceil((cx - half) / gStep) * gStep; g <= cx + half; g += gStep) gridLines.push([P(g, cy - half, 0), P(g, cy + half, 0)]);
  for (let g = Math.ceil((cy - half) / gStep) * gStep; g <= cy + half; g += gStep) gridLines.push([P(cx - half, g, 0), P(cx + half, g, 0)]);
  const dStep = niceStep(maxD, 4);
  const ticks = []; for (let d = 0; d <= -baseZ; d += dStep) ticks.push(d);

  const bores = wells.map(w => {
    const D = depth(w), head = P(w.easting, w.northing, 0), toe = P(w.easting, w.northing, -D);
    return {w, D, head, toe, top: P(w.easting, w.northing, pinH), d: head[2]};
  }).sort((a, b) => b.d - a.d);

  // Hand the particle loop what it needs; it reads this ref every frame.
  const focus = selected && flows?.channels.some(x => x.inj.id === selected || x.prod.id === selected) ? selected : null;
  scene.current = {P, W, H, on: showFlow, flows, focus, half, resTop, resBot, owc, thick, cam: {c, s, cp, sp, R, f, oy, cx, cy, tz},
    producers: wells.filter(w => w.role === 'producer' || (w.type && w.type !== 'injector')).map(w => ({w, z: Math.max(resBot, Math.min(-depth(w), resTop))})),
    zOf: w => Math.max(resBot + thick * 0.1, Math.min(-depth(w), resTop - thick * 0.1)),
    key: (showFlow ? 1 : 0) + '|' + wells.map(w => w.id + w.easting + w.northing + w.type).join() + '|' + (flows?.channels.map(x => x.inj.id + x.prod.id + x.f.toFixed(3)).join() || '')};
  useEffect(() => startFlowParticles(canvas.current, scene), []);
  useEffect(() => startConnectivity3D(gl.current, scene), []);

  // ---- Pointer (on the wrapper, so the canvas and both SVG layers share it) ----
  const localXY = e => { const r = e.currentTarget.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  function down(e) {
    if (e.button !== 0) return;
    cancelAnimationFrame(anim.current);
    drag.current = {x: e.clientX, y: e.clientY, cam: {...cam}, moved: false, onWell: !!e.target.closest('.wellNode, .channel')};
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function move(e) {
    const g = drag.current;
    if (g) {
      const dx = e.clientX - g.x, dy = e.clientY - g.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) g.moved = true;
      if (g.moved) setCam({...g.cam, yaw: g.cam.yaw - dx * 0.35, pitch: Math.max(3, Math.min(89, g.cam.pitch + dy * 0.3))});
      return;
    }
    const [sx, sy] = localXY(e);
    onHover?.(toGround(sx, sy));
  }
  function up(e) {
    const g = drag.current; drag.current = null;
    if (!g || g.moved || g.onWell) return;
    const [sx, sy] = localXY(e);
    const at = toGround(sx, sy);
    if (at) onGround?.(at, sx, sy);
  }

  // Streamtubes: each injector → producer channel is a bundle of streamlines through the reservoir. More water
  // (a larger share) gives more streamlines and a wider bundle; the label spot avoids labels already placed.
  const tagSpots = [];
  const zOf = w => Math.max(resBot + thick * 0.1, Math.min(-depth(w), resTop - thick * 0.1));
  const STEPS = 28;
  const TUBE = 18 * Math.max(0.7, Math.min(1.8, cam.zoom));        // tube diameter on screen, px
  /** A streamtube from injector to producer: its centre line through the reservoir and lines offset across it.
   *  Offsets run along the screen normal that points down, so the liquid settles at the bottom of the tube. */
  function tubeGeom(ch) {
    const zA = zOf(ch.inj), zB = zOf(ch.prod);
    const centre = Array.from({length: STEPS + 1}, (_, k) => {
      const u = k / STEPS;
      return P(ch.inj.easting + (ch.prod.easting - ch.inj.easting) * u, ch.inj.northing + (ch.prod.northing - ch.inj.northing) * u,
        zA + (zB - zA) * u - thick * 0.18 * Math.sin(Math.PI * u));
    });
    const normals = centre.map((_, k) => {
      const a = centre[Math.max(0, k - 1)], b = centre[Math.min(STEPS, k + 1)];
      let nx = -(b[1] - a[1]), ny = b[0] - a[0];
      const L = Math.hypot(nx, ny) || 1; nx /= L; ny /= L;
      if (ny < -0.15 || (Math.abs(ny) <= 0.15 && nx < 0)) { nx = -nx; ny = -ny; }
      return [nx, ny];
    });
    const offset = d => centre.map((p, k) => [p[0] + normals[k][0] * d, p[1] + normals[k][1] * d]);
    const free = p => tagSpots.every(q => Math.abs(q[0] - p[0]) > 42 || Math.abs(q[1] - p[1]) > 24);
    const mid = [14, 11, 17, 8, 20, 6, 22].map(k => offset(-TUBE * 0.9)[k]).find(free) || offset(-TUBE * 0.9)[14];
    tagSpots.push(mid);
    return {centre, offset, mid, depthKey: centre.reduce((s, p) => s + p[2], 0) / centre.length};
  }
  const path = pts => 'M' + pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join('L');

  /** A well as a steel pipe with its fluid moving in it: water down injectors, oil/water up producers, gas bubbles above a gas-lift valve. */
  function pipe({w, D}) {
    const sel = selected === w.id, injector = w.type === 'injector';
    const head = P(w.easting, w.northing, 0), toe = P(w.easting, w.northing, -D);
    const share = flows?.arriving?.[w.id];
    const liquid = injector ? INJ_WATER : share != null ? mix(OIL, '#0b2f6b', 0.25 + 0.75 * share / flows.maxArr) : OIL;
    const [a, b] = injector ? [head, toe] : [toe, head];             // flow direction
    const L = (p, q, props) => <line x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} strokeLinecap="round" {...props}/>;
    const pump = w.type === 'esp' ? [P(w.easting, w.northing, -D * 0.9), P(w.easting, w.northing, -D * 0.97)] : null;
    const off = [1.6, 0];
    return <g key={'pipe' + w.id} className={'pipe' + (sel ? ' sel' : '')}>
      {sel && L(head, toe, {stroke: liquid, strokeWidth: 16, opacity: 0.22, filter: 'url(#g3-glow)'})}
      {L(head, toe, {stroke: '#26302c', strokeWidth: sel ? 10 : 9})}
      {L(head, toe, {stroke: '#a9b5af', strokeWidth: sel ? 8 : 7})}
      {L([head[0] - off[0], head[1]], [toe[0] - off[0], toe[1]], {stroke: 'rgba(255,255,255,.55)', strokeWidth: 1})}
      {L(head, toe, {stroke: liquid, strokeWidth: sel ? 4.6 : 4})}
      {showFlow && <>
        {L(a, b, {stroke: 'rgba(0,0,0,.28)', strokeWidth: sel ? 4.6 : 4, className: 'pipeDense', style: {animationDuration: injector ? '1.1s' : '0.9s'}})}
        {L(a, b, {stroke: 'rgba(255,255,255,.45)', strokeWidth: 1.6, className: 'pipeStreak', style: {animationDuration: injector ? '1.6s' : '1.25s'}})}
        {w.type === 'gas_lift' && <>
          {L(toe, head, {stroke: 'rgba(255,255,255,.92)', strokeWidth: 3.2, className: 'bubbles'})}
          {L(toe, head, {stroke: 'rgba(235,245,255,.8)', strokeWidth: 2, className: 'bubbles2'})}
        </>}
      </>}
      {pump && <>{L(pump[0], pump[1], {stroke: '#4a2a63', strokeWidth: 11})}{L(pump[0], pump[1], {stroke: '#8d5bb5', strokeWidth: 8})}</>}
      {w.type === 'gas_lift' && <circle cx={toe[0]} cy={toe[1]} r="5" fill="#bf7426" stroke="#fff" strokeWidth="1.5"/>}
      {[-0.012, 0, 0.012].map(k => { const p = P(w.easting, w.northing, -D * (1 + k));
        return <line key={k} x1={p[0] - 9} y1={p[1]} x2={p[0] + 9} y2={p[1]} stroke={injector ? INJ_WATER : '#3a2a10'} strokeWidth="1.6" opacity=".8"/>; })}
      {sel && <text x={toe[0] + 14} y={toe[1] + 4} className="t3 depthLabel">{fmt(D, 0)} m</text>}
    </g>;
  }

  const layerLabels = strata.map(L => { const a = P(rx, ry, L.z0), b = P(rx, ry, L.z1); return {L, x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, h: Math.abs(b[1] - a[1])}; });
  const dragProps = {onPointerDown: down, onPointerMove: move, onPointerUp: up, onPointerLeave: () => onHover?.(null),
    onWheel: e => setCam(v => ({...v, zoom: Math.max(0.6, Math.min(2.6, v.zoom * (e.deltaY > 0 ? 0.92 : 1.08)))})), onDoubleClick: () => flyTo(HOME)};

  return <div className="field3d" style={{width: W, height: H}} {...dragProps} role="img" aria-label="Field in 3D: drag to rotate, scroll to zoom">
    {/* Layer 1: the rock. Back walls and floor of every stratum (seen from inside the cut-away), then see-through fronts. */}
    <svg className="geo" width={W} height={H}>
      <defs>
        <pattern id="pt-sand" width="7" height="7" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r=".9" fill="rgba(50,32,8,.32)"/><circle cx="5" cy="4.6" r=".7" fill="rgba(255,255,255,.3)"/><circle cx="4.2" cy="1" r=".45" fill="rgba(50,32,8,.25)"/></pattern>
        <pattern id="pt-dots" width="9" height="9" patternUnits="userSpaceOnUse"><circle cx="2" cy="3" r="1.1" fill="rgba(40,30,10,.3)"/><circle cx="6.5" cy="7" r=".8" fill="rgba(40,30,10,.25)"/></pattern>
        <pattern id="pt-shale" width="16" height="6" patternUnits="userSpaceOnUse"><path d="M0 1.5h9M5 4.5h10" stroke="rgba(30,22,12,.3)" strokeWidth=".8"/></pattern>
        <pattern id="pt-brick" width="18" height="9" patternUnits="userSpaceOnUse"><path d="M0 .5h18M0 5h18M5 .5v4.5M14 5v4" stroke="rgba(45,38,28,.24)" strokeWidth=".8" fill="none"/></pattern>
        <linearGradient id="g3-sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#eef6f1"/><stop offset="1" stopColor="#dce9e2"/></linearGradient>
      </defs>
      {strata.flatMap((L, li) => L.faces.filter(fc => !fc.front && (fc.kind === 'side' || (fc.kind === 'bottom' && li === strata.length - 1))).map((fc, i) =>
        <g key={'w' + li + i}>
          <polygon points={poly(fc.pts)} fill={L.fill}/>
          <polygon points={poly(fc.pts)} fill={`url(#${L.pat})`}/>
          {(fc.shade || fc.kind === 'bottom') && <polygon points={poly(fc.pts)} fill={fc.kind === 'bottom' ? 'rgba(0,0,0,.1)' : 'rgba(0,0,0,.12)'}/>}
        </g>))}
      {/* Reservoir volume: oil zone and water leg get see-through tops and fronts. */}
      {strata.filter(L => L.res).map(L => L.faces.filter(fc => fc.front).map((fc, i) =>
        <polygon key={L.res + i} points={poly(fc.pts)} fill={fc.kind === 'top' && L.top ? L.top : L.fill} fillOpacity={fc.kind === 'top' ? (L.res === 'oil' ? 0.82 : 0.25) : 0.22}
          stroke={L.res === 'oil' ? '#6b4310' : '#2c5778'} strokeOpacity=".45" strokeWidth="1"/>))}
      {strata.filter(L => L.res === 'oil').map(L => L.faces.filter(fc => fc.front && fc.kind === 'top').map((fc, i) => <polygon key={'tex' + i} points={poly(fc.pts)} fill="url(#pt-sand)" opacity=".7"/>))}
      {/* Front edges of every stratum, so the block reads as layered rock without hiding the wells. */}
      {strata.map((L, li) => L.faces.filter(fc => fc.front && fc.kind === 'side').map((fc, i) =>
        <g key={'f' + li + i}>
          <polygon points={poly(fc.pts)} fill={L.fill} fillOpacity={L.res ? 0 : 0.07}/>
          <line x1={fc.pts[0][0]} y1={fc.pts[0][1]} x2={fc.pts[1][0]} y2={fc.pts[1][1]} stroke="rgba(60,45,25,.35)" strokeWidth="1"/>
        </g>))}
      {/* Surface */}
      {(() => { const t = faces(0, -1).find(fc => fc.kind === 'top'); return <polygon points={poly(t.pts)} fill="url(#g3-sky)" fillOpacity=".38" stroke="#9fb5a9" strokeWidth="1.2"/>; })()}
      {gridLines.map(([a, b], i) => <line key={'g' + i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className="grid3"/>)}
      {(() => { const a = P(cx + half * 0.82, cy + half * 0.62, 0), b = P(cx + half * 0.82, cy + half * 0.82, 0);
        return <g className="compass"><line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}/><circle cx={b[0]} cy={b[1]} r="2.5"/><text x={b[0]} y={b[1] - 7} textAnchor="middle" className="t3">N</text></g>; })()}
      <g className="depthScale">
        {(() => { const a = P(ex, ey, 0), b = P(ex, ey, baseZ); return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]}/>; })()}
        {ticks.map(d => { const p = P(ex, ey, -d); return <g key={d}><line x1={p[0] - 5} x2={p[0]} y1={p[1]} y2={p[1]}/><text x={p[0] < 70 ? p[0] + 6 : p[0] - 8} y={p[1] + 4} textAnchor={p[0] < 70 ? 'start' : 'end'} className="t3">{d === 0 ? 'Surface' : `${fmt(d, 0)} m`}</text></g>; })}
      </g>
      {layerLabels.filter(l => l.h > 11).map(({L, x, y}) => <text key={L.name} x={x + 10} y={y + 4} className={'t3 layerName' + (L.res ? ' res' : '')}>{L.name}{L.res === 'oil' ? ` · ${fmt(-L.z0, 0)} m` : ''}</text>)}
    </svg>

    {/* Layer 2: the waterflood moving through the rock. */}
    <canvas ref={canvas} className="flowCanvas" style={{width: W, height: H}}/>

    {/* Layer 3: injector → producer streamtubes in real 3D (WebGL, flow3d.js). */}
    <canvas ref={gl} className="glCanvas" style={{width: W, height: H}}/>

    {/* Layer 4: pipes, labels, channel hit areas and well heads. */}
    <svg className="wellsLayer" width={W} height={H}>
      <defs><filter id="g3-glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3"/></filter></defs>
      {bores.map(pipe)}
      {showFlow && flows?.channels.length > 0 && (() => {
        const items = [...flows.channels].sort((x, y) => y.f - x.f).map(ch => {
          const share = ch.f / flows.maxF, t = Math.pow(share, 1.6);
          const faded = focus && ch.inj.id !== focus && ch.prod.id !== focus;
          return {ch, share, t, col: waterColor(0.25 + 0.75 * t), label: `${Math.round(ch.f * 100)}%`, cls: 'channel' + (ch.main ? ' main' : '') + (faded ? ' faded' : ''), ...tubeGeom(ch)};
        });
        const pick = ch => ({onClick: e => { e.stopPropagation(); onChannel?.(ch); }});
        const W = TUBE;
        return <g className="channels">
          {/* the tubes themselves are drawn in WebGL; here only a hit area so a tube can be clicked */}
          {items.map(({ch, label, cls, centre}) => <g key={'c' + ch.inj.id + ch.prod.id} className={cls + ' streamtube'} {...pick(ch)}>
            <title>{`${label} of ${ch.inj.name}'s injected water reaches ${ch.prod.name}`}</title>
            <path d={path(centre)} stroke="transparent" strokeWidth={W + 10} fill="none" strokeLinecap="round"/>
          </g>)}
          {items.map(({ch, col, label, cls, mid}) => <g key={'t' + ch.inj.id + ch.prod.id} className={cls + ' tagOnly'} {...pick(ch)}>
            <title>{`${label} of ${ch.inj.name}'s injected water reaches ${ch.prod.name}`}</title>
            <g transform={`translate(${mid[0].toFixed(1)},${mid[1].toFixed(1)})`} className="channelTag">
              <rect x={-19} y={-10} width="38" height="20" rx="10" fill={ch.main ? col : '#fff'} stroke={col} strokeWidth="1.5"/>
              <text y="4" textAnchor="middle" fill={ch.main ? '#fff' : col}>{label}</text>
            </g>
          </g>)}
        </g>;
      })()}
      {bores.map(({w, head, top}) => {
        const sel = selected === w.id;
        return <g key={'p' + w.id} className="wellNode" onClick={e => { e.stopPropagation(); onSelect(w); }} onContextMenu={e => onMenu(e, w)}>
          <ellipse cx={head[0]} cy={head[1]} rx="12" ry={Math.max(2.5, 12 * sp)} className="pinShadow"/>
          <rect x={head[0] - 6} y={head[1] - 5} width="12" height="5" rx="1.5" className="wellheadBase"/>
          <line x1={head[0]} y1={head[1] - 4} x2={top[0]} y2={top[1]} className="pinStalk"/>
          <WellSymbol type={w.type} x={top[0]} y={top[1]} selected={sel} alert={alertOf(w)}/>
          <text x={top[0]} y={top[1] - 20} textAnchor="middle" className="wellLabel">{w.name}</text>
          {showFlow && flows?.arriving?.[w.id] > 0.005 && <text x={top[0]} y={top[1] - 34} textAnchor="middle" className="waterShare"
            style={{fill: waterColor(0.35 + 0.65 * flows.arriving[w.id] / flows.maxArr)}}>{Math.round(flows.arriving[w.id] * 100)}% of injected water</text>}
          <title>{`${w.name} · depth ${fmt(depth(w), 0)} m${setDepth(w) ? '' : ' (assumed)'} · right-click for options`}</title>
        </g>;
      })}
    </svg>
  </div>;
}

/**
 * Particles for the reservoir flow, drawn on a canvas every frame from the latest scene (camera and data).
 * Water leaves each injector split by its connectivity: a particle heads for producer j with probability f_ij, so
 * most of the water visibly travels to the best-connected producer; the remainder spreads out and fades (water that
 * leaves the pattern). Oil drains toward every producer. Flood-front rings expand from the injectors.
 */
function startFlowParticles(cv, sceneRef) {
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const still = reducedMotion();
  let raf = 0, last = performance.now(), key = '', parts = [];
  const rnd = () => Math.random(), gauss = () => (rnd() + rnd() + rnd() - 1.5) / 1.5;

  function build(S) {
    const out = [];
    const injs = S.flows ? Object.values(S.flows.injectors) : [];
    const per = injs.length ? Math.min(150, Math.round(420 / injs.length)) : 0;
    for (const I of injs) {
      // Water that reaches a producer is drawn inside the streamtubes (SVG); here only the part that leaves the pattern.
      const lost = Math.max(0, 1 - I.out);
      for (let k = 0; k < Math.round(per * lost * 0.5); k++) out.push({kind: 'lost', w: I.w, a: rnd() * Math.PI * 2, reach: S.half * (0.18 + 0.3 * rnd()), u: rnd(), zr: rnd(), v: 0.06 + 0.05 * rnd()});
    }
    for (const {w} of S.producers) for (let k = 0; k < 26; k++) out.push({kind: 'oil', w, a: rnd() * Math.PI * 2, rad: S.half * (0.1 + 0.32 * rnd()), u: rnd(), zr: rnd(), v: 0.05 + 0.06 * rnd()});
    return out;
  }
  function frame(now) {
    const S = sceneRef.current;
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(S.W * dpr) || cv.height !== Math.round(S.H * dpr)) { cv.width = Math.round(S.W * dpr); cv.height = Math.round(S.H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, S.W, S.H);
    if (S.on) {
      if (S.key !== key) { key = S.key; parts = build(S); }
      const {P, focus} = S;
      const zIn = (zr, zMid) => Math.max(S.resBot + S.thick * 0.06, Math.min(S.resTop - S.thick * 0.06, zMid + (zr - 0.5) * S.thick * 0.7));
      ctx.lineCap = 'round';
      for (const p of parts) {
        if (!still) { p.u += p.v * dt; if (p.u > 1) { p.u -= 1; p.r = gauss(); p.zr = rnd(); if (p.a != null) p.a = rnd() * Math.PI * 2; } }
        let a, b, col, alpha, width;
        if (p.kind === 'lost') {
          const z = zIn(p.zr, S.zOf(p.w)), at = u => P(p.w.easting + Math.cos(p.a) * p.reach * u, p.w.northing + Math.sin(p.a) * p.reach * u, z);
          a = at(Math.max(0, p.u - 0.06)); b = at(p.u);
          col = '#5aa0e0'; width = 1.3; alpha = 0.5 * (1 - p.u) * Math.min(1, p.u * 6) * (focus && focus !== p.w.id ? 0.3 : 1);
        } else {
          const z = zIn(p.zr, S.zOf(p.w)), at = u => P(p.w.easting + Math.cos(p.a) * p.rad * (1 - u), p.w.northing + Math.sin(p.a) * p.rad * (1 - u), z);
          a = at(Math.max(0, p.u - 0.05)); b = at(p.u);
          col = '#4a2a06'; width = 1.8; alpha = 0.75 * Math.min(1, p.u * 5, (1 - p.u) * 10) * (focus && focus !== p.w.id ? 0.3 : 1);
        }
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = col; ctx.lineWidth = width;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);
  return () => cancelAnimationFrame(raf);
}
