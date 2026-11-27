'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, fmt, qs } from './lib';
import { useCtx, useRun, latest } from './ctx';
import { Modal } from './components';
import { CsvTable } from './table';
import { Field3D, crmFlows } from './field3d';

function niceStep(span) {
  const raw = span / 6, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  return [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) || mag * 10;
}

/** The well symbol: producer ●, ESP producer (E), gas-lift producer (G), injector ▼. */
export function WellSymbol({type, x, y, selected, alert}) {
  const ring = selected && <circle cx={x} cy={y} r="15" fill="none" stroke="#c27b2c" strokeWidth="2.5"/>;
  if (type === 'injector') return <g className="sym3d">{ring}<path d={`M${x - 9},${y - 7}L${x + 9},${y - 7}L${x},${y + 9}Z`} fill="url(#ws-inj)" stroke="#fff" strokeWidth="1.5" strokeLinejoin="round"/></g>;
  return <g className="sym3d">{ring}
    <circle cx={x} cy={y} r="9" fill="url(#ws-ball)" stroke="#fff" strokeWidth="1.5"/>
    <ellipse cx={x - 2.6} cy={y - 3.4} rx="3.6" ry="2.2" fill="#fff" opacity=".38" pointerEvents="none"/>
    {type === 'esp' && <><rect x={x - 13} y={y - 13} width="26" height="26" rx="4" fill="none" stroke="#8a4fa3" strokeWidth="2.5"/><text x={x} y={y + 4} textAnchor="middle" className="sym">E</text></>}
    {type === 'gas_lift' && <><circle cx={x} cy={y} r="13" fill="none" stroke="#c27b2c" strokeWidth="2.5" strokeDasharray="4 3"/><text x={x} y={y + 4} textAnchor="middle" className="sym">G</text></>}
    {alert && <circle cx={x + 10} cy={y - 10} r="4.5" fill="#d9541e" stroke="#fff" strokeWidth="1.5"/>}
  </g>;
}

/** Shared gradients for the well symbols: a lit ball for producers, a shaded wedge for injectors. */
function SymbolDefs() {
  return <svg width="0" height="0" style={{position: 'absolute'}} aria-hidden="true"><defs>
    <radialGradient id="ws-ball" cx="38%" cy="32%" r="70%"><stop offset="0" stopColor="#5fcf96"/><stop offset=".55" stopColor="#1b7a4f"/><stop offset="1" stopColor="#0b432b"/></radialGradient>
    <linearGradient id="ws-inj" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#6d96d1"/><stop offset="1" stopColor="#1f416f"/></linearGradient>
  </defs></svg>;
}

const VIEW_KEY = 'field-console:mapView';
const readView = () => { try { return localStorage.getItem(VIEW_KEY) || '3d'; } catch { return '3d'; } };

export function FieldCanvas() {
  const {wells, field, schema, set, go, route, mode, notify, reloadWells, loadFields, runs, select} = useCtx();
  const box = useRef(null), svg = useRef(null);
  const [size, setSize] = useState({w: 900, h: 600});
  const [menu, setMenu] = useState(null), [adding, setAdding] = useState(null), [placing, setPlacing] = useState(null), [cursor, setCursor] = useState(null);
  const [column, setColumn] = useState(null), [removing, setRemoving] = useState(null);
  const [view, setViewState] = useState('2d'), view3d = useRef(null);
  useEffect(() => { setViewState(readView()); }, []);
  const setView = v => { setViewState(v); setMenu(null); setAdding(null); try { localStorage.setItem(VIEW_KEY, v); } catch {} };
  const edit = mode === 'edit';
  useEffect(() => {
    if (!box.current || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setSize({w: Math.max(400, e.contentRect.width), h: Math.max(360, e.contentRect.height)}));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const close = e => e.key === 'Escape' && (setMenu(null), setPlacing(null));
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, []);
  // World extent: placed wells with a margin, kept square so distances are true.
  const placed = wells.filter(w => w.easting != null);
  let [x0, x1, y0, y1] = placed.length ? [Math.min(...placed.map(w => w.easting)), Math.max(...placed.map(w => w.easting)), Math.min(...placed.map(w => w.northing)), Math.max(...placed.map(w => w.northing))] : [0, 6000, 0, 6000];
  const span = Math.max(x1 - x0, y1 - y0, placed.length > 1 ? 1 : 6000) * 1.3;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const scale = Math.min(size.w, size.h) / span;
  const wx0 = cx - size.w / 2 / scale, wy1 = cy + size.h / 2 / scale;
  const px = e => (e - wx0) * scale, py = n => (wy1 - n) * scale;
  const toWorld = (sx, sy) => [wx0 + sx / scale, wy1 - sy / scale];
  const step = niceStep(span);
  const local = e => { const r = box.current.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  const alertOf = w => w.results?.esp_risk?.status === 'alert' || w.results?.valve_risk?.status === 'alert';

  async function update(w, patch, message) {
    try { await api(`/wells/${w.id}`, {method: 'PUT', body: patch}); await reloadWells(); notify(message); }
    catch (e) { notify(e.message, 'error'); }
  }
  function onBackground(e) {
    const [sx, sy] = local(e);
    atPoint(toWorld(sx, sy), sx, sy);
  }
  function atPoint(world, sx, sy) {
    setMenu(null);
    const [ex, ny] = world.map(v => Math.round(v));
    if (placing) { const w = placing; setPlacing(null); update(w, {easting: ex, northing: ny}, `${w.name} placed at ${ex}, ${ny}`); return; }
    if (edit) setAdding({easting: ex, northing: ny, sx, sy});
  }
  function wellMenu(e, w) {
    e.preventDefault(); e.stopPropagation();
    const [sx, sy] = local(e);
    const items = [];
    if (w.role === 'producer' && edit) {
      if (w.type === 'producer') items.push(['Add ESP', () => update(w, {type: 'esp'}, `ESP added to ${w.name}`)], ['Add gas lift', () => update(w, {type: 'gas_lift'}, `Gas lift added to ${w.name}`)]);
      if (w.type === 'esp') items.push(['Switch to gas lift', () => update(w, {type: 'gas_lift'}, `${w.name} switched to gas lift`)], ['Remove ESP', () => update(w, {type: 'producer'}, `ESP removed from ${w.name}`)]);
      if (w.type === 'gas_lift') items.push(['Switch to ESP', () => update(w, {type: 'esp'}, `${w.name} switched to ESP`)], ['Remove gas lift', () => update(w, {type: 'producer'}, `Gas lift removed from ${w.name}`)]);
      items.push(null);
    }
    if (edit) items.push([`CRM ${w.role === 'producer' ? 'production' : 'injection'} rates`, () => setColumn(w)]);
    if (w.type === 'esp' || w.type === 'gas_lift') items.push(['Flow meter', () => { set({well: w.id}); go('vfm', null, {well: w.id}); }]);
    if (w.type === 'esp' || w.type === 'gas_lift') items.push(['Well Models', () => go('models', null, {well: w.id})], ['Failure', () => go('failure', w.type === 'esp' ? 'esp' : 'gas_lift', {well: w.id})]);
    if (edit) items.push(null, ['Delete well…', () => setRemoving(w), 'danger']);
    set({well: w.id});
    setMenu({sx, sy, title: `${w.name} · ${schema.wellTypes[w.type].label}`, items});
  }
  const unplaced = wells.filter(w => w.easting == null);
  const counts = {producers: wells.filter(w => w.role === 'producer').length, injectors: wells.filter(w => w.role === 'injector').length};
  const grid = [];
  for (let g = Math.ceil(wx0 / step) * step; g <= wx0 + size.w / scale; g += step) grid.push(['x', g]);
  for (let g = Math.ceil((wy1 - size.h / scale) / step) * step; g <= wy1; g += step) grid.push(['y', g]);
  const is3d = view === '3d';
  const [showFlow, setShowFlow] = useState(true);
  const crmRun = is3d ? latest(runs, 'crm_match') : null;
  const crm = useRun(crmRun?.id);
  const flows = crm?.outputs ? crmFlows(crm.outputs, placed) : null;
  const channelInfo = ch => select({title: `${ch.inj.name} → ${ch.prod.name}`, kind: 'pair',
    items: [['Share of injected water', `${fmt(ch.f * 100, 1)}%`], ch.tau != null && ['Response time τ', fmt(ch.tau, 1)], ['From run', crm.name]].filter(Boolean),
    text: `${fmt(ch.f * 100, 1)}% of ${ch.inj.name}'s injected water reaches ${ch.prod.name}${ch.main ? `: the main channel from ${ch.inj.name}` : ''}. Thicker, darker and faster streaks carry more water.`,
    links: [['Open connectivity', () => go('crm', 'connectivity', {run: crm.id})]]});
  return <div className={'canvasWrap' + (placing ? ' placing' : '') + (is3d ? ' is3d' : '')} ref={box} onContextMenu={e => e.preventDefault()}>
    <SymbolDefs/>
    <div className="mapHeader">
      <div className="mapTitle">
        <b>{field || 'No field'}</b>
        <span>{counts.producers} producer{counts.producers === 1 ? '' : 's'} · {counts.injectors} injector{counts.injectors === 1 ? '' : 's'}{unplaced.length ? ` · ${unplaced.length} not placed` : ''}</span>
      </div>
      <div className="seg" role="group" aria-label="Map view">
        <button aria-pressed={!is3d} onClick={() => setView('2d')}>Map</button>
        <button aria-pressed={is3d} onClick={() => setView('3d')}>3D</button>
      </div>
    </div>
    {is3d ? <Field3D wells={placed} size={size} selected={route.well} alertOf={alertOf} viewRef={view3d} flows={flows} showFlow={showFlow} onChannel={channelInfo}
      onSelect={w => { setMenu(null); set({well: w.id}); }} onMenu={wellMenu} onHover={setCursor}
      onGround={(at, sx, sy) => atPoint(at, sx, sy)}/> : <svg ref={svg} className="fieldCanvas" width={size.w} height={size.h} onClick={onBackground}
      onMouseMove={e => { const [sx, sy] = local(e); setCursor(toWorld(sx, sy)); }} onMouseLeave={() => setCursor(null)} role="img" aria-label={`Field ${field || '(no field)'} canvas`}>
      {grid.map(([axis, g]) => axis === 'x'
        ? <g key={'x' + g}><line x1={px(g)} x2={px(g)} y1="0" y2={size.h} className="gridLine"/><text x={px(g) + 3} y={size.h - 6} className="gridText">{fmt(g, 0)}</text></g>
        : <g key={'y' + g}><line x1="0" x2={size.w} y1={py(g)} y2={py(g)} className="gridLine"/><text x="6" y={py(g) + 13} className="gridText">{fmt(g, 0)}</text></g>)}
      {placed.map(w => <g key={w.id} className="wellNode" onClick={e => { e.stopPropagation(); setMenu(null); set({well: w.id}); }} onContextMenu={e => wellMenu(e, w)}>
        <WellSymbol type={w.type} x={px(w.easting)} y={py(w.northing)} selected={route.well === w.id} alert={alertOf(w)}/>
        <text x={px(w.easting)} y={py(w.northing) + 26} textAnchor="middle" className="wellLabel">{w.name}</text>
        <title>{`${w.name} (${w.code}) · ${schema.wellTypes[w.type].label} · ${fmt(w.easting, 0)}, ${fmt(w.northing, 0)} · right-click for options`}</title></g>)}
      {adding && <circle cx={adding.sx} cy={adding.sy} r="8" fill="none" stroke="#c27b2c" strokeWidth="2" strokeDasharray="3 2"/>}
    </svg>}
    {is3d && <div className="viewCtl" role="group" aria-label="3D view">
      <button className="secondary small" aria-pressed={showFlow} onClick={() => setShowFlow(v => !v)} title="Animate the flow in the wells and the injector-to-producer channels">Flow</button>
      <button className="secondary small" onClick={() => view3d.current?.top()}>Top</button>
      <button className="secondary small" onClick={() => view3d.current?.side()}>Side</button>
      <button className="secondary small" onClick={() => view3d.current?.reset()}>Reset</button>
    </div>}
    <div className="canvasLegend">
      <span><svg width="22" height="22"><WellSymbol type="producer" x={11} y={11}/></svg>Producer</span>
      <span><svg width="30" height="30"><WellSymbol type="esp" x={15} y={15}/></svg>ESP</span>
      <span><svg width="30" height="30"><WellSymbol type="gas_lift" x={15} y={15}/></svg>Gas lift</span>
      <span><svg width="22" height="22"><WellSymbol type="injector" x={11} y={11}/></svg>Injector</span>
      {is3d && showFlow && flows?.channels.length > 0 && <span className="waterKey">Water share<i/>low → high</span>}
      <span className="legendHint">{cursor ? `E ${fmt(cursor[0], 0)} · N ${fmt(cursor[1], 0)}` : is3d && !placing ? 'Drag to rotate · scroll to zoom · right-click a well for options' : edit ? (placing ? `Click the map to place ${placing.name}` : 'Click empty space to add a well · right-click a well for options') : 'Review mode · read only'}</span>
    </div>
    {unplaced.length > 0 && <div className="tray"><div className="trayTitle">Not on the map<span className="count">{unplaced.length}</span></div>{unplaced.map(w => <div key={w.id} className="trayItem">
      <span>{w.name}</span>{edit && <button className="link small" onClick={() => { setPlacing(w); setMenu(null); }}>{placing?.id === w.id ? 'Click the map…' : 'Place'}</button>}</div>)}</div>}
    {menu && <div className="ctxMenu" style={{left: Math.min(menu.sx, size.w - 220), top: Math.max(8, Math.min(menu.sy, size.h - 44 - menu.items.length * 34))}} onClick={e => e.stopPropagation()}>
      <div className="ctxTitle">{menu.title}</div>
      {menu.items.map((it, i) => it ? <button key={i} className={it[2] === 'danger' ? 'dangerItem' : undefined} onClick={() => { setMenu(null); it[1](); }}>{it[0]}</button> : <hr key={i}/>)}
      {!menu.items.length && <div className="muted small">No options in review mode.</div>}
    </div>}
    {adding && <AddWell at={adding} onClose={() => setAdding(null)} onDone={async () => { setAdding(null); await reloadWells(); loadFields(); }}/>}
    {column && <CrmColumn well={column} onClose={() => setColumn(null)}/>}
    {removing && <DeleteWell well={removing} onClose={() => setRemoving(null)}/>}
  </div>;
}

/** Confirm and delete a well, its datasets and latest results. Warns when the Deep-CRM dataset still has a column for it. */
function DeleteWell({well, onClose}) {
  const {field, route, set, notify, reloadWells, loadFields} = useCtx();
  const [inCrm, setInCrm] = useState(false), [busy, setBusy] = useState(false);
  useEffect(() => {
    api('/crm' + qs({field})).then(r => {
      const ids = [...(r.data?.producers || []), ...(r.data?.injectors || [])].map(s => String(s).toLowerCase());
      setInCrm(ids.includes(well.code.toLowerCase()) || ids.includes(well.name.toLowerCase()));
    }).catch(() => {});
  }, [field, well]);
  const rows = Object.values(well.counts || {}).reduce((n, c) => n + (c.rows || 0), 0);
  async function remove() {
    setBusy(true);
    try {
      await api(`/wells/${well.id}`, {method: 'DELETE'});
      if (route.well === well.id) set({well: null});
      await reloadWells(); loadFields();
      notify(`${well.name} deleted`);
      onClose();
    } catch (e) { notify(e.message, 'error'); setBusy(false); }
  }
  return <Modal title={`Delete ${well.name}?`} onClose={onClose}>
    <p>This removes <b>{well.name}</b>{well.code !== well.name ? ` (${well.code})` : ''}, its well settings{rows ? `, ${fmt(rows, 0)} data rows` : ''} and its latest results. It cannot be undone.
      Saved runs that used it are kept and marked “needs review”.</p>
    {inCrm && <div className="warning">{well.name} is a column in this field’s Deep-CRM dataset. The dataset keeps the column, so its water channels in 3D and
      the geometry options of the history match stop working until you save a dataset without it (Deep-CRM → Data).</div>}
    <div className="formActions"><button className="secondary" onClick={onClose}>Cancel</button>
      <button className="danger" disabled={busy} onClick={remove}>{busy ? 'Deleting…' : 'Delete well'}</button></div>
  </Modal>;
}

function AddWell({at, onClose, onDone}) {
  const {field, notify, set} = useCtx();
  const [v, setV] = useState({code: '', name: '', role: 'producer', easting: at.easting, northing: at.northing, field});
  const [error, setError] = useState(''), [busy, setBusy] = useState(false);
  async function save(e) {
    e.preventDefault(); setBusy(true); setError('');
    try {
      const w = await api('/wells', {method: 'POST', body: {code: v.code, name: v.name || v.code, type: v.role, field: v.field, easting: v.easting, northing: v.northing}});
      notify(`Added ${w.name}`); set({well: w.id, field: w.field}); onDone();
    } catch (err) { setError(err.message); setBusy(false); }
  }
  return <Modal title="Add well here" onClose={onClose}>
    <form className="form" onSubmit={save}>
      <div className="row2"><label>Well ID<input required autoFocus maxLength={40} value={v.code} onChange={e => setV({...v, code: e.target.value})}/></label>
        <label>Name (optional)<input maxLength={80} value={v.name} onChange={e => setV({...v, name: e.target.value})}/></label></div>
      <div className="row2"><label>Role<select value={v.role} onChange={e => setV({...v, role: e.target.value})}><option value="producer">Producer</option><option value="injector">Injector</option></select></label>
        <label>Field<input maxLength={80} value={v.field} onChange={e => setV({...v, field: e.target.value})}/></label></div>
      <div className="row2"><label>Easting<input type="number" step="any" required value={v.easting} onChange={e => setV({...v, easting: e.target.value})}/></label>
        <label>Northing<input type="number" step="any" required value={v.northing} onChange={e => setV({...v, northing: e.target.value})}/></label></div>
      <small>Add ESP or gas lift afterwards by right-clicking the well.</small>
      {error && <p className="formError">{error}</p>}
      <div className="formActions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button disabled={busy}>{busy ? 'Adding…' : 'Add well'}</button></div>
    </form>
  </Modal>;
}

/** Edit one well's column in the Deep-CRM rate dataset (time axis fixed). */
function CrmColumn({well, onClose}) {
  const {field, notify} = useCtx();
  const [data, setData] = useState(undefined);
  useEffect(() => { api('/crm' + qs({field})).then(r => setData(r.data)).catch(e => { notify(e.message, 'error'); onClose(); }); }, [field]); // eslint-disable-line react-hooks/exhaustive-deps
  if (data === undefined) return null;
  const producer = well.role === 'producer';
  if (!data) return <Modal title={`${well.name}: CRM rates`} onClose={onClose}><p>There is no Deep-CRM dataset for this field yet. Save time.csv (and the rate files) in Deep-CRM → Data first; the column needs its time axis.</p>
    <div className="formActions"><button onClick={onClose}>Close</button></div></Modal>;
  const list = producer ? data.producers : data.injectors, matrix = producer ? data.production : data.injection;
  const j = list.indexOf(well.code);
  const initial = data.time.map((t, k) => [t, j >= 0 ? matrix[k][j] : '']);
  return <CsvTable title={`${well.name}: ${producer ? 'production' : 'injection'} rates`} headerless fixedRows minRows={0}
    columns={[{key: 'time', label: 'time', readOnly: true}, {key: 'rate', label: `${well.code} ${producer ? 'production' : 'injection'}`}]} initial={initial}
    note={j >= 0 ? `Column ${j + 1} of ${producer ? 'production.csv' : 'injection.csv'} (dataset v${data.version}). Saving creates a new dataset version.`
      : `${well.code} is not in the dataset yet; saving adds it as a new ${producer ? 'production' : 'injection'} column. Import a one-column CSV with ${data.time.length} rows, or type the rates.`}
    onClose={onClose} saveLabel="Save column"
    onSave={async rows => {
      if (rows.some(r => r[1] === '')) throw new Error('Fill every row (use 0 for shut-in).');
      const r = await api('/crm/column', {method: 'POST', body: {field, well: well.code, values: rows.map(x => Number(x[1]))}});
      notify(`${well.code} column saved (dataset v${r.version})`); onClose();
    }}/>;
}
