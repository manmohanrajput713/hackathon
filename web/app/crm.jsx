'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, upload, download, fmt, when, qs } from './lib';
import { useCtx, useRun } from './ctx';
import { LineChart, Empty, Tabs, Stale, KV, COLORS, useReplay } from './components';
import { RunBar } from './data';
import { CsvTable, toCsv } from './table';

/** Send a table to the same staging endpoint a CSV upload uses, so it is validated identically. */
async function stageTable(field, kind, header, rows) {
  const file = new File([toCsv(header, rows)], `${kind}.csv`, {type: 'text/csv'});
  return upload(`/crm/upload/${kind}` + qs({field}), file);
}
const ids = text => text.split(',').map(s => s.trim()).filter(Boolean);

// Sub-tabs follow the user guide's workflow: data → history match → connectivity → forecast → companion models.
const SUBS = [{id: 'data', label: '1 · Data'}, {id: 'match', label: '2 · History Match'}, {id: 'connectivity', label: '3 · Connectivity'},
              {id: 'forecast', label: '4 · Forecast'}, {id: 'companion', label: '5 · Companion models'}];

function useCrmData() {
  const {field, notify} = useCtx();
  const [state, setState] = useState(null);
  const load = useCallback(() => api('/crm' + qs({field})).then(setState).catch(e => notify(e.message, 'error')), [field, notify]);
  useEffect(() => { load(); }, [load]);
  return [state, load];
}

export function DeepCRM() {
  const {route, set} = useCtx();
  const sub = SUBS.some(s => s.id === route.sub) ? route.sub : 'data';
  const [crm, reload] = useCrmData();
  return <>
    <div className="moduleHead"><div><small>Waterflood connectivity · pywaterflood</small><h1>Deep-CRM</h1></div>
      {crm?.data && <span className="muted">Dataset v{crm.data.version} · {crm.data.producers.length} producers · {crm.data.injectors.length} injectors · {crm.data.time.length} steps</span>}</div>
    <Tabs tabs={SUBS} active={sub} onChange={id => set({sub: id})} small/>
    {!crm ? <section><p>Loading…</p></section> :
      sub === 'data' ? <DataTab crm={crm} reload={reload}/> :
      !crm.data ? <section><Empty title="No CRM dataset" action={<button onClick={() => set({sub: 'data'})}>Go to Data</button>}>Upload production.csv, injection.csv and time.csv first.</Empty></section> :
      sub === 'match' ? <MatchTab crm={crm}/> : sub === 'connectivity' ? <ConnectivityTab crm={crm}/> : sub === 'forecast' ? <ForecastTab crm={crm}/> : <CompanionTab crm={crm} reload={reload}/>}
  </>;
}

// ---------- 1 · Data ----------

const FILES = [
  ['production', 'production.csv', 'Required. Rows = time steps, columns = producers in a fixed order. Reservoir volume per time.'],
  ['injection', 'injection.csv', 'Required. Rows = time steps, columns = injectors in a fixed order.'],
  ['time', 'time.csv', 'Required. One column of elapsed time (same unit as τ), increasing.'],
  ['pressure', 'pressure.csv', 'Optional. Producer bottom-hole pressure, same shape and column order as production.csv. Enables CrmCompensated.'],
  ['locations', 'locations.csv', 'Optional. Header name,easting,northing,kind (Producer/Injector). Enables the geometry seed and MPI.'],
];

function FileSlot({kind, label, help, staged, stored, onStaged, onTable}) {
  const {field, notify} = useCtx();
  const ref = useRef(null);
  const [busy, setBusy] = useState(false);
  async function pick(e) {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    setBusy(true);
    try { await upload(`/crm/upload/${kind}` + qs({field}), f); onStaged(); notify(`${label} checked`); }
    catch (err) { notify(err.message, 'error'); } finally { setBusy(false); }
  }
  return <div className={'fileSlot' + (staged && !staged.error ? ' ok' : staged?.error ? ' bad' : '')}>
    <div><b>{label}</b><small>{help}</small>
      {staged ? (staged.error ? <small className="dangerText">{staged.error}</small> :
        <small className="good">Ready to save: {staged.rows} rows{staged.columns ? ` × ${staged.columns} columns` : ''}{staged.producers != null ? ` (${staged.producers} producers, ${staged.injectors} injectors)` : ''}{staged.min != null ? ` · min ${fmt(staged.min)} · max ${fmt(staged.max)}` : ''}</small>)
        : stored ? <small>In dataset: {stored}</small> : <small className="muted">Not uploaded</small>}</div>
    <div className="actions editOnly"><button className="secondary small" onClick={onTable}>Fill table</button>
      <button className="secondary small" disabled={busy} onClick={() => ref.current.click()}>{busy ? 'Checking…' : 'Import CSV'}</button></div>
    <input ref={ref} type="file" accept=".csv,.txt" hidden onChange={pick}/>
  </div>;
}

function DataTab({crm, reload}) {
  const {field, notify, loadFields} = useCtx();
  const d = crm.data;
  const [prodIds, setProdIds] = useState(''), [injIds, setInjIds] = useState(''), [busy, setBusy] = useState(false);
  const [table, setTable] = useState(null);
  async function openTable(kind) {
    let rows = null;
    try { rows = (await api(`/crm/staged/${kind}` + qs({field}))).rows; } catch {}
    if (!rows && d) rows = kind === 'time' ? d.time.map(t => [t]) : kind === 'locations' ? (d.locations || []).map(l => [l.name, l.easting, l.northing, l.kind])
      : kind === 'pressure' ? d.pressure : d[kind];
    if (rows && kind === 'locations' && rows.length && !Array.isArray(rows[0])) rows = rows.map(l => [l.name, l.easting, l.northing, l.kind]);
    if (rows && kind === 'time' && rows.length && !Array.isArray(rows[0])) rows = rows.map(t => [t]);
    setTable({kind, rows: rows || []});
  }
  function tableSpec(kind) {
    const pIds = ids(prodIds).length ? ids(prodIds) : ['P1'], iIds = ids(injIds).length ? ids(injIds) : ['I1'];
    if (kind === 'production' || kind === 'pressure') return {columns: pIds.map(l => ({key: l, label: l})), headerless: true, matrix: kind === 'production', newColumnLabel: i => `P${i + 1}`};
    if (kind === 'injection') return {columns: iIds.map(l => ({key: l, label: l})), headerless: true, matrix: true, newColumnLabel: i => `I${i + 1}`};
    if (kind === 'time') return {columns: [{key: 'time', label: 'time'}], headerless: true};
    return {columns: [{key: 'name', label: 'name', required: true}, {key: 'easting', label: 'easting', required: true}, {key: 'northing', label: 'northing', required: true},
                      {key: 'kind', label: 'kind', required: true, options: ['Producer', 'Injector']}], headerless: false};
  }
  const stagedProd = crm.staged.production, stagedInj = crm.staged.injection;
  const nProd = stagedProd?.columns ?? d?.producers.length, nInj = stagedInj?.columns ?? d?.injectors.length;
  useEffect(() => {
    setProdIds((stagedProd || !d) ? Array.from({length: nProd || 0}, (_, k) => d?.producers[k] || `P${k + 1}`).join(', ') : d.producers.join(', '));
    setInjIds((stagedInj || !d) ? Array.from({length: nInj || 0}, (_, k) => d?.injectors[k] || `I${k + 1}`).join(', ') : d.injectors.join(', '));
  }, [d?.version, nProd, nInj]); // eslint-disable-line react-hooks/exhaustive-deps
  const stored = kind => {
    if (!d) return null;
    if (kind === 'production') return `${d.time.length} × ${d.producers.length}`;
    if (kind === 'injection') return `${d.time.length} × ${d.injectors.length}`;
    if (kind === 'time') return `${d.time.length} steps, ${fmt(d.time[0])} → ${fmt(d.time[d.time.length - 1])}`;
    if (kind === 'pressure') return d.pressure ? `${d.time.length} × ${d.producers.length}` : null;
    if (kind === 'locations') return d.locations ? `${d.locations.length} wells` : null;
  };
  async function save() {
    setBusy(true);
    try { await api('/crm/commit', {method: 'POST', body: {field, producer_ids: prodIds, injector_ids: injIds}}); await reload(); loadFields(); notify('CRM dataset saved'); }
    catch (e) { notify(e.message, 'error'); } finally { setBusy(false); }
  }
  async function discard() { await api('/crm/staged' + qs({field}), {method: 'DELETE'}).catch(() => {}); reload(); }
  const hasStaged = Object.keys(crm.staged).some(k => FILES.some(f => f[0] === k));
  const rows = d ? d.time.map((t, k) => Object.fromEntries([['t', t], ...d.producers.map((p, j) => [p, d.production[k][j]])])) : [];
  const irows = d ? d.time.map((t, k) => Object.fromEntries([['t', t], ...d.injectors.map((p, j) => [p, d.injection[k][j]])])) : [];
  const brows = d ? d.time.map((t, k) => ({t, production: crm.balance.production[k], injection: crm.balance.injection[k]})) : [];
  return <>
    <section>
      <div className="sectionHead"><div><h2>Rate data</h2><p>Headerless CSV files in the pywaterflood layout. All files share one time axis; column order is the well order. A shut-in well is a column of zeros, not a missing column.</p></div></div>
      <div className="fileSlots">{FILES.map(([kind, label, help]) => <FileSlot key={kind} kind={kind} label={label} help={help} staged={crm.staged[kind]} stored={stored(kind)} onStaged={reload} onTable={() => openTable(kind)}/>)}</div>
      {table && (() => { const spec = tableSpec(table.kind); return <CsvTable key={table.kind} title={`${table.kind}.csv`} {...spec} initial={table.rows}
        minRows={table.rows.length ? 0 : (d?.time.length || 10)}
        note={spec.headerless ? `Same layout as ${table.kind}.csv: one row per time step${spec.columns.length > 1 || spec.matrix ? ', one column per well in order' : ''}, no header row.${spec.matrix ? ' Column labels are the well IDs below.' : ''}` : 'Same layout as locations.csv, including its header row.'}
        onClose={() => setTable(null)} saveLabel="Use table (check it like an upload)"
        onSave={async (rows, cols) => {
          await stageTable(field, table.kind, spec.headerless ? null : ['name', 'easting', 'northing', 'kind'], rows);
          if (table.kind === 'production') setProdIds(cols.map(c => c.label).join(', '));
          if (table.kind === 'injection') setInjIds(cols.map(c => c.label).join(', '));
          setTable(null); reload(); notify(`${table.kind}.csv table checked; save the dataset to keep it`);
        }}/>; })()}
      <div className="row2 editOnly">
        <label>Producer IDs in column order (comma separated)<input value={prodIds} onChange={e => setProdIds(e.target.value)}/></label>
        <label>Injector IDs in column order (comma separated)<input value={injIds} onChange={e => setInjIds(e.target.value)}/></label>
      </div>
      <div className="toolbar editOnly"><button disabled={busy || (!hasStaged && !d)} onClick={save}>{busy ? 'Saving…' : d ? 'Save as new dataset version' : 'Save dataset'}</button>
        {hasStaged && <button className="link dangerText" onClick={discard}>Discard uploaded files</button>}
        <span className="muted small">Files not re-uploaded are kept from the current version.</span></div>
      {crm.versions.length > 0 && <table className="mini"><thead><tr><th>Version</th><th>Saved</th><th>Source</th></tr></thead>
        <tbody>{crm.versions.map(v => <tr key={v.version}><td>v{v.version}</td><td>{when(v.created_at)}</td><td>{v.source}</td></tr>)}</tbody></table>}
    </section>
    {d && <>
      <div className="grid2">
        <section><h2>Producers</h2><LineChart rows={rows} x="t" xLabel="time" height={210} title="producers" series={d.producers.map(p => ({key: p, label: p}))}/></section>
        <section><h2>Injectors</h2><LineChart rows={irows} x="t" xLabel="time" height={210} title="injectors" series={d.injectors.map(p => ({key: p, label: p}))}/></section>
      </div>
      <div className="grid2">
        <section><h2>Field material balance</h2>
          <LineChart rows={brows} x="t" xLabel="time" height={210} title="material balance" series={[{key: 'production', label: 'Field production'}, {key: 'injection', label: 'Field injection', dashed: true}]}/>
          <p>Production / injection ratio: <b>{fmt(crm.balance.ratio, 3)}</b>. {crm.balance.ratio > 1.1 ? 'Production runs well above injection: plan an aquifer pseudo-injector (Companion models).' : 'Field injection tracks production.'}</p>
        </section>
        <section><h2>Well map</h2>{d.locations ? <WellMap locations={d.locations}/> : <p className="muted">Upload locations.csv to see the map, seed gains from geometry and run MPI.</p>}</section>
      </div>
      <section><h3>Before you fit (user guide)</h3><ul className="checklist">
        <li>Convert stock-tank rates to reservoir volume if formation volume factors matter.</li>
        <li>Shut-in wells are zero columns, not missing columns.</li>
        <li>A large persistent surplus of production over injection means aquifer support: add a pseudo-injector.</li>
        <li>Keep well IDs in the same order as the columns.</li></ul></section>
    </>}
  </>;
}

function WellMap({locations, highlight = []}) {
  const xs = locations.map(l => l.easting), ys = locations.map(l => l.northing);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const W = 360, H = 300, P = 30;
  const px = v => P + ((v - x0) / ((x1 - x0) || 1)) * (W - 2 * P), py = v => H - P - ((v - y0) / ((y1 - y0) || 1)) * (H - 2 * P);
  return <svg viewBox={`0 0 ${W} ${H}`} className="wellMap" role="img" aria-label="Well map">
    {locations.map(l => <g key={l.name}>{l.kind === 'Injector'
      ? <path d={`M${px(l.easting) - 6},${py(l.northing) - 6}L${px(l.easting) + 6},${py(l.northing) + 6}M${px(l.easting) + 6},${py(l.northing) - 6}L${px(l.easting) - 6},${py(l.northing) + 6}`} stroke="#b06a1f" strokeWidth="2.5"/>
      : <circle cx={px(l.easting)} cy={py(l.northing)} r="6" fill={highlight.includes(l.name) ? '#c0392b' : '#2f5d9b'}/>}
      <text x={px(l.easting) + 8} y={py(l.northing) - 6}>{l.name}</text></g>)}
    <text x={P} y={H - 6}>● producer  ✕ injector</text>
  </svg>;
}

// ---------- 2 · History Match ----------

const DEFAULTS = {model: 'crm', primary: true, tau_selection: 'per-pair', constraints: 'up-to one', initial_guess: 'uniform', random: false,
  epochs: 500, lr: 0.001, optimizer: 'rmsprop', hidden_layers: 4, hidden_units: 20, data_weight: 1, physics_weight: 1, pressure_weight: 1,
  constraint_weight: 1, collocation_multiplier: 1, seed: 0, num_cores: 1, device: 'cpu', holdout_steps: 12, x_e: '', y_e: ''};

function matchRuns(runs) { return runs.filter(r => r.kind === 'crm_match'); }

function RunPicker({runs, value, onChange, label = 'History match'}) {
  return <label className="inlineLabel">{label}<select value={value || ''} onChange={e => onChange(e.target.value)}>
    {!runs.length && <option value="">No completed runs</option>}
    {runs.map(r => <option key={r.id} value={r.id}>{r.name} · {when(r.finished_at)}{r.stale?.length ? ' · needs review' : ''}</option>)}</select></label>;
}

function MatchTab({crm}) {
  const {runs, route, set} = useCtx();
  const d = crm.data;
  const [o, setO] = useState(() => {
    const ext = d.locations ? (() => { const xs = d.locations.map(l => l.easting), ys = d.locations.map(l => l.northing); return [(Math.max(...xs) - Math.min(...xs)) * 1.2, (Math.max(...ys) - Math.min(...ys)) * 1.2]; })() : ['', ''];
    return {...DEFAULTS, x_e: ext[0] ? Math.round(ext[0]) : '', y_e: ext[1] ? Math.round(ext[1]) : ''};
  });
  const [name, setName] = useState('');
  const setv = (k, v) => setO(s => ({...s, [k]: v}));
  const done = matchRuns(runs).filter(r => r.status === 'done');
  const rid = route.run && done.some(r => r.id === route.run) ? route.run : done[0]?.id;
  const run = useRun(rid);
  const out = run?.outputs;
  const [pick, setPick] = useState(0);
  const j = out ? Math.min(pick, out.producers.length - 1) : 0;
  const series = out ? out.time.map((t, k) => ({t, observed: out.observed[k][j], fit: k < out.train_rows ? out.predicted[k][j] : null,
    held: k >= out.train_rows - 1 && out.train_rows < out.time.length ? out.predicted[k][j] : null, residual: out.residual[k][j]})) : [];
  const [cursor, replay] = useReplay(series.length);
  const num = (k, label, step = 'any', min) => <label key={k}>{label}<input type="number" step={step} min={min} value={o[k]} onChange={e => setv(k, e.target.value)}/></label>;
  const params = () => Object.fromEntries(Object.entries(o).filter(([k]) => o.initial_guess === 'geometry' || !['x_e', 'y_e'].includes(k)));
  const losses = out?.loss_history?.length ? out.loss_history.map((h, k) => Object.fromEntries([['step', k], ...Object.entries(h).filter(([key]) => key !== 'step').map(([key, v]) => [key, v > 0 ? Math.log10(v) : null])])) : [];
  const lossKeys = losses.length ? Object.keys(out.loss_history[0]).filter(k => out.loss_history.some(h => h[k] > 0)) : [];
  return <>
    <section className="editOnly">
      <div className="sectionHead"><div><h2>Fit</h2><p>Default from the guide: CRM-IP (per-pair τ), primary production on, gains “up-to one”. History matching trains the Deep-CRM physics-informed network; predictions use the closed-form CRM with the identified f<sub>ij</sub> and τ.</p></div></div>
      <fieldset className="group"><legend>Model</legend><div className="rowFields">
        <label>Model<select value={o.model} onChange={e => setv('model', e.target.value)}><option value="crm">CRM</option><option value="compensated" disabled={!d.pressure}>CrmCompensated (producer BHP){d.pressure ? '' : ' — needs pressure.csv'}</option></select></label>
        <label>Primary production<select value={o.primary ? 'yes' : 'no'} onChange={e => setv('primary', e.target.value === 'yes')}><option value="yes">On (depletion drive)</option><option value="no">Off (injection-dominated)</option></select></label>
        <label>Time constants<select value={o.tau_selection} onChange={e => setv('tau_selection', e.target.value)}><option value="per-pair">Per pair (CRM-IP)</option><option value="per-producer">Per producer (CRM-P)</option></select></label>
        <label>Gain constraints<select value={o.constraints} onChange={e => setv('constraints', e.target.value)}><option value="up-to one">Up-to one (injector sums ≤ 1)</option><option value="sum-to-one">Sum-to-one (producer sums = 1)</option><option value="positive">Positive (gains ≥ 0)</option></select></label>
      </div></fieldset>
      <fieldset className="group"><legend>Initialization</legend><div className="rowFields">
        <label>Initial guess<select value={o.initial_guess} onChange={e => setv('initial_guess', e.target.value)}><option value="uniform">Uniform</option><option value="random">Random</option><option value="geometry" disabled={!d.locations}>Geometry (MPI){d.locations ? '' : ' — needs locations.csv'}</option></select></label>
        {o.initial_guess === 'geometry' && <>{num('x_e', 'Reservoir length x_e')}{num('y_e', 'Reservoir width y_e')}</>}
        {num('seed', 'Seed', 1, 0)}
        <label className="check"><input type="checkbox" checked={o.random} onChange={e => setv('random', e.target.checked)}/>Random restart (random=True)</label>
      </div></fieldset>
      <fieldset className="group"><legend>Training</legend><div className="rowFields">
        {num('epochs', 'Epochs', 1, 1)}{num('lr', 'Learning rate (lr)')}
        <label>Optimizer<select value={o.optimizer} onChange={e => setv('optimizer', e.target.value)}><option value="rmsprop">rmsprop</option><option value="adam">adam</option></select></label>
        {num('hidden_layers', 'Hidden layers', 1, 1)}{num('hidden_units', 'Hidden units', 1, 1)}
        {num('data_weight', 'Data weight')}{num('physics_weight', 'Physics weight')}{o.model === 'compensated' && num('pressure_weight', 'Pressure weight')}
        {num('constraint_weight', 'Constraint weight')}{num('collocation_multiplier', 'Collocation multiplier', 1, 1)}
        <label>Device<select value={o.device} onChange={e => setv('device', e.target.value)}><option value="cpu">cpu</option><option value="cuda">cuda</option></select></label>
        {num('num_cores', 'CPU threads (num_cores)', 1, 1)}
      </div></fieldset>
      <fieldset className="group"><legend>Validation</legend><div className="rowFields">
        {num('holdout_steps', `Hold out last N steps (of ${d.time.length})`, 1, 0)}
        <label>Run name (optional)<input value={name} onChange={e => setName(e.target.value)} maxLength={100}/></label>
      </div><small>Keep the model with the lowest held-out RMSE, not the lowest training loss. Compare seeds and variants in Results.</small></fieldset>
      <RunBar kind="crm_match" label="Run history match" params={params} opts={{name}} hint={`Dataset v${d.version}. Training runs in the background.`}/>
    </section>
    {!run ? <section><Empty title="No history match yet">Choose options and run a history match.</Empty></section> : <section>
      <div className="sectionHead"><div><h2>Observed vs CRM</h2><small>{run.name} · fit on {out.train_rows} of {out.time.length} steps</small></div>
        <div className="actions"><RunPicker runs={done} value={rid} onChange={id => set({run: id})}/>
          <label className="inlineLabel">Producer<select value={j} onChange={e => setPick(Number(e.target.value))}>{out.producers.map((p, k) => <option key={p} value={k}>{p}</option>)}</select></label></div></div>
      <Stale reasons={run.stale}/>
      <div className="metrics">{[['RMSE fit (all)', fmt(out.rmse_fit_overall)], ['RMSE held out (all)', fmt(out.rmse_holdout_overall)], [`RMSE fit ${out.producers[j]}`, fmt(out.rmse_fit[j])],
        [`RMSE held out ${out.producers[j]}`, out.rmse_holdout ? fmt(out.rmse_holdout[j]) : '—']].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
      {replay}
      <LineChart rows={series} x="t" xLabel="time" cursor={cursor} title={`${out.producers[j]} history match`}
        bands={out.train_rows < out.time.length ? [{from: out.time[out.train_rows], to: out.time[out.time.length - 1], label: 'Held out'}] : []}
        series={[{key: 'observed', label: 'Observed', color: '#aa7551', dashed: true}, {key: 'fit', label: 'CRM (fit)', color: '#007953'}, {key: 'held', label: 'CRM (held out)', color: '#2f5d9b'}]}/>
      <h3>Residual (observed − CRM)</h3>
      <LineChart rows={series} x="t" xLabel="time" height={150} cursor={cursor} title="residual" series={[{key: 'residual', label: 'Residual', color: '#8a4fa3'}]}/>
      <table className="mini"><thead><tr><th>Producer</th>{out.producers.map(p => <th key={p}>{p}</th>)}<th>Overall</th></tr></thead>
        <tbody><tr><td>RMSE fit</td>{out.rmse_fit.map((v, k) => <td key={k}>{fmt(v)}</td>)}<td>{fmt(out.rmse_fit_overall)}</td></tr>
          {out.rmse_holdout && <tr><td>RMSE held out</td>{out.rmse_holdout.map((v, k) => <td key={k}>{fmt(v)}</td>)}<td>{fmt(out.rmse_holdout_overall)}</td></tr>}</tbody></table>
      {losses.length > 0 && <><h3>PINN training (log₁₀ loss)</h3>
        <LineChart rows={losses} x="step" xLabel="logged step" height={170} title="loss history" series={lossKeys.map((k, i) => ({key: k, label: k, color: COLORS[i % COLORS.length]}))}/>
        <p className="muted small">Data and physics losses should fall together. If physics stays high while data falls, raise the physics weight or train longer.</p></>}
    </section>}
  </>;
}

// ---------- 3 · Connectivity ----------

function ConnectivityTab({crm}) {
  const {runs, route, set, select, notify} = useCtx();
  const done = matchRuns(runs).filter(r => r.status === 'done');
  const rid = route.run && done.some(r => r.id === route.run) ? route.run : done[0]?.id;
  const run = useRun(rid);
  const [cell, setCell] = useState(null);
  if (!done.length) return <section><Empty title="No fitted model" action={<button onClick={() => set({sub: 'match'})}>Go to History Match</button>}>Run a history match to read f<sub>ij</sub> and τ.</Empty></section>;
  if (!run?.outputs) return <section><p>Loading…</p></section>;
  const o = run.outputs;
  const pair = o.tau_selection === 'per-pair';
  const tau = (j, i) => pair ? o.tau[j][i] : o.tau[j];
  function open(j, i) {
    setCell([j, i]);
    select({title: `${o.injectors[i]} → ${o.producers[j]}`, kind: 'pair', items: [['f (gain)', fmt(o.gains[j][i], 3)], ['τ', fmt(tau(j, i), 1)], ['Run', run.name]],
      text: `${fmt(o.gains[j][i] * 100, 1)}% of ${o.injectors[i]}'s injection supports ${o.producers[j]}. A useful gain is large, stable across seeds, and geologically plausible.`});
  }
  const pairRows = cell ? o.time.map((t, k) => ({t, injection: o.injection[k][cell[1]], production: o.observed[k][cell[0]]})) : [];
  return <section>
    <div className="sectionHead"><div><h2>Interwell connectivity f<sub>ij</sub></h2><small>{run.name} · gains[j, i] = injector i supporting producer j · producers in rows</small></div>
      <div className="actions"><RunPicker runs={done} value={rid} onChange={id => { setCell(null); set({run: id}); }}/>
        <button className="secondary small" onClick={() => download(`/runs/${rid}/export?format=xlsx`).catch(e => notify(e.message, 'error'))}>Export Excel (gains, taus, primary)</button></div></div>
    <Stale reasons={run.stale}/>
    <div className="scroll"><table className="heat"><thead><tr><th>Producer \ injector</th>{o.injectors.map(i => <th key={i}>{i}</th>)}<th>Producer sum</th></tr></thead>
      <tbody>{o.producers.map((p, j) => <tr key={p}><th>{p}</th>
        {o.injectors.map((inj, i) => { const g = o.gains[j][i]; const on = cell && cell[0] === j && cell[1] === i;
          return <td key={inj} className={'cell' + (on ? ' on' : '')} style={{background: `rgba(192,57,43,${Math.min(1, g) * 0.85})`, color: g > 0.45 ? '#fff' : undefined}} onClick={() => open(j, i)}>{fmt(g, 3)}</td>; })}
        <td className="muted">{fmt(o.producer_sums[j], 3)}</td></tr>)}
        <tr className="total"><th>Injector sum</th>{o.injector_sums.map((s, i) => <td key={i} className={o.constraints === 'up-to one' && s > 1.0001 ? 'warnCell' : ''}>{fmt(s, 3)}</td>)}<td/></tr></tbody></table></div>
    <p className="muted small">{o.constraints === 'up-to one' ? 'Each injector column should sum to ≤ 1; the remainder is injection not seen at these producers (aquifer, unswept rock, measurement error).' : o.constraints === 'sum-to-one' ? 'Each producer row sums to 1.' : 'Positive gains with no sum constraint can over-allocate injection.'}</p>
    <div className="grid2">
      <div><h3>Time constants τ {pair ? '(per pair)' : '(per producer)'}</h3>
        {pair ? <table className="mini"><thead><tr><th>Producer \ injector</th>{o.injectors.map(i => <th key={i}>{i}</th>)}</tr></thead>
          <tbody>{o.producers.map((p, j) => <tr key={p}><th>{p}</th>{o.injectors.map((inj, i) => <td key={inj}>{fmt(o.tau[j][i], 1)}</td>)}</tr>)}</tbody></table>
          : <table className="mini"><tbody>{o.producers.map((p, j) => <tr key={p}><th>{p}</th><td>{fmt(o.tau[j], 1)}</td></tr>)}</tbody></table>}</div>
      <div><h3>Primary production{o.productivity_index ? ' and productivity index' : ''}</h3>
        <table className="mini"><thead><tr><th>Producer</th><th>Producer gain</th><th>Producer τ</th>{o.productivity_index && <th>J</th>}</tr></thead>
          <tbody>{o.producers.map((p, j) => <tr key={p}><td>{p}</td><td>{fmt(o.gains_producer[j], 4)}</td><td>{fmt(o.tau_producer[j], 1)}</td>{o.productivity_index && <td>{fmt(o.productivity_index[j], 4)}</td>}</tr>)}</tbody></table></div>
    </div>
    {cell ? <div className="pairDetail"><h3>{o.injectors[cell[1]]} → {o.producers[cell[0]]}: f = {fmt(o.gains[cell[0]][cell[1]], 3)}, τ = {fmt(tau(cell[0], cell[1]), 1)}</h3>
      <LineChart rows={pairRows} x="t" xLabel="time" height={190} title="pair history" series={[{key: 'injection', label: `${o.injectors[cell[1]]} injection`, color: '#2f5d9b'}, {key: 'production', label: `${o.producers[cell[0]]} production`, color: '#007953'}]}/></div>
      : <p className="muted">Click a cell to see the injection and production histories behind it.</p>}
  </section>;
}

// ---------- 4 · Forecast ----------

function ForecastTab({crm}) {
  const {runs, route, set, field, notify, go} = useCtx();
  const done = matchRuns(runs).filter(r => r.status === 'done');
  const [mid, setMid] = useState(null);
  const matchId = mid && done.some(r => r.id === mid) ? mid : done[0]?.id;
  const match = useRun(matchId);
  const [mode, setMode] = useState('hold'), [steps, setSteps] = useState('12'), [dt, setDt] = useState(''), [name, setName] = useState('');
  const [staged, setStaged] = useState({});
  const futureRef = useRef(null), timeRef = useRef(null);
  const [ftable, setFtable] = useState(null);
  const forecasts = runs.filter(r => r.kind === 'crm_forecast' && r.parent_id === matchId);
  const fid = route.run && forecasts.some(r => r.id === route.run) ? route.run : forecasts.find(r => r.status === 'done')?.id;
  const fc = useRun(fid);
  const [pick, setPick] = useState(0);
  if (!done.length) return <section><Empty title="No fitted model" action={<button onClick={() => set({sub: 'match'})}>Go to History Match</button>}>The forecast uses a fitted model without retraining.</Empty></section>;
  const mo = match?.outputs;
  async function stage(kind, e) {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    try { const s = await upload(`/crm/upload/${kind}` + qs({field}), f); setStaged(x => ({...x, [kind]: s})); notify(`${f.name} checked`); }
    catch (err) { notify(err.message, 'error'); }
  }
  const f = fc?.outputs;
  const j = f ? Math.min(pick, f.producers.length - 1) : 0;
  const rows = f ? [...f.history_time.map((t, k) => ({t, observed: f.observed[k][j], crm: f.history_predicted[k][j]})), ...f.time.map((t, k) => ({t, plan: f.plan[k][j]}))] : [];
  const defaultDt = mo ? mo.time[1] - mo.time[0] : null;
  return <>
    <section className="editOnly">
      <div className="sectionHead"><div><h2>Injection plan</h2><p>predict() is the closed-form CRM: the plan is appended to the history and the tail is read off, with the identified f<sub>ij</sub> and τ frozen.</p></div>
        <RunPicker runs={done} value={matchId} onChange={setMid} label="Fitted model"/></div>
      <div className="inline"><fieldset>
        <label>Plan<select value={mode} onChange={e => setMode(e.target.value)}><option value="hold">Hold last injection</option><option value="table">Upload future injection CSV</option></select></label>
        {mode === 'hold' ? <label>Steps<input type="number" min="1" step="1" value={steps} onChange={e => setSteps(e.target.value)}/></label> : <>
          <div className="fileInline"><button className="secondary small" onClick={() => futureRef.current.click()}>Import future injection CSV</button><button className="secondary small" onClick={() => setFtable('future_injection')}>Fill table</button><input ref={futureRef} type="file" accept=".csv,.txt" hidden onChange={e => stage('future_injection', e)}/>
            <small>{staged.future_injection ? `${staged.future_injection.rows} steps × ${staged.future_injection.columns} injectors` : `Headerless, rows = future steps, ${mo?.injectors.length ?? '?'} columns in injector order`}</small></div>
          <div className="fileInline"><button className="secondary small" onClick={() => timeRef.current.click()}>Import future time CSV (optional)</button><button className="secondary small" onClick={() => setFtable('future_time')}>Fill table</button><input ref={timeRef} type="file" accept=".csv,.txt" hidden onChange={e => stage('future_time', e)}/>
            <small>{staged.future_time ? `${staged.future_time.rows} times` : 'Otherwise last time + n × step'}</small></div></>}
        <label>Time step (default {defaultDt != null ? fmt(defaultDt) : '—'})<input type="number" step="any" min="0" placeholder={defaultDt != null ? String(defaultDt) : ''} value={dt} onChange={e => setDt(e.target.value)}/></label>
        <label>Name (optional)<input value={name} onChange={e => setName(e.target.value)}/></label>
      </fieldset></div>
      {ftable && mo && <CsvTable title={ftable === 'future_injection' ? 'Future injection' : 'Future time'} headerless
        columns={ftable === 'future_injection' ? mo.injectors.map(i => ({key: i, label: i})) : [{key: 'time', label: 'time'}]}
        initial={[]} minRows={12} onClose={() => setFtable(null)} saveLabel="Use table"
        note={ftable === 'future_injection' ? 'One row per future step, one column per injector in order (no header row).' : 'One future time per row, after the last history time (no header row).'}
        onSave={async rows => { const st = await stageTable(field, ftable, null, rows); setStaged(x => ({...x, [ftable]: st})); setFtable(null); notify('Table checked'); }}/>}
      <RunBar kind="crm_forecast" label="Run forecast" filter={{parent_id: matchId}} opts={{name}}
        params={() => ({parent_id: matchId, mode, steps: Number(steps), ...(dt ? {dt: Number(dt)} : {})})}/>
    </section>
    {!f ? <section><Empty title="No forecast yet">Run a forecast for the selected model.</Empty></section> : <section>
      <div className="sectionHead"><div><h2>History and plan</h2><small>{fc.name} · forecast starts at t = {fmt(f.forecast_start)} · {f.time.length} steps</small></div>
        <div className="actions"><RunPicker runs={forecasts.filter(r => r.status === 'done')} value={fid} onChange={id => set({run: id})} label="Forecast"/>
          <label className="inlineLabel">Producer<select value={j} onChange={e => setPick(Number(e.target.value))}>{f.producers.map((p, k) => <option key={p} value={k}>{p}</option>)}</select></label></div></div>
      <Stale reasons={fc.stale}/>
      {f.notes.map(n => <p key={n} className="warning">{n}</p>)}
      <LineChart rows={rows} x="t" xLabel="time" markers={[{x: f.forecast_start, label: 'Plan starts'}]} title={`${f.producers[j]} plan`}
        series={[{key: 'observed', label: 'Observed', color: '#aa7551', dashed: true}, {key: 'crm', label: 'CRM (history)', color: '#007953'}, {key: 'plan', label: 'Plan', color: '#2f5d9b'}]}/>
      <table className="mini"><thead><tr><th>Planned production, last step</th>{f.producers.map(p => <th key={p}>{p}</th>)}</tr></thead>
        <tbody><tr><td>t = {fmt(f.time[f.time.length - 1])}</td>{f.last_step.map((v, k) => <td key={k}>{fmt(v, 1)}</td>)}</tr></tbody></table>
      <div className="toolbar"><button className="secondary small" onClick={() => download(`/runs/${fid}/export?format=xlsx`).catch(e => notify(e.message, 'error'))}>Export Excel</button>
        <button className="secondary small" onClick={() => go('results', null, {run: fid})}>Open in Results</button></div>
    </section>}
  </>;
}

// ---------- 5 · Companion models ----------

const ROCK = [['phi', 'Porosity φ', 0.25], ['viscosity_oil', 'Oil viscosity (Pa·s)', 0.0042], ['viscosity_water', 'Water viscosity (Pa·s)', 0.001], ['sat_oil_r', 'Residual oil saturation', 0.2],
  ['sat_water_c', 'Connate water saturation', 0.1], ['sat_gas_c', 'Critical gas saturation', 0.05], ['n_oil', 'Brooks–Corey oil exponent', 2], ['n_water', 'Brooks–Corey water exponent', 2],
  ['flow_cross_section', 'Flow area (m²)', 1], ['flow_rate', 'Flow rate (m³/d)', 1]];
const AQ = [['pore_volume', 'Reservoir pore volume', 385], ['porosity', 'Reservoir porosity', 0.31], ['thickness', 'Thickness (ft)', 20], ['solid_angle', 'Aquifer open angle (°)', 180],
  ['k_a', 'Aquifer permeability k_a', 64.2], ['mu', 'Water viscosity μ_w', 0.48], ['c_t', 'Total compressibility c_t', 5.88], ['phi_a', 'Aquifer porosity φ_a', 0.27],
  ['r_ed', 'Radius ratio r_eD', 5], ['aquifer_constant', 'Aquifer constant U', 1]];

function CompanionTab({crm, reload}) {
  const {field, notify, runs, mode: appMode} = useCtx();
  const d = crm.data;
  const ext = d.locations ? [(Math.max(...d.locations.map(l => l.easting)) - Math.min(...d.locations.map(l => l.easting))) * 1.2, (Math.max(...d.locations.map(l => l.northing)) - Math.min(...d.locations.map(l => l.northing))) * 1.2] : ['', ''];
  const [xe, setXe] = useState(ext[0] ? String(Math.round(ext[0])) : ''), [ye, setYe] = useState(ext[1] ? String(Math.round(ext[1])) : '');
  const [mpi, setMpi] = useState(null), [bl, setBl] = useState(null), [aq, setAq] = useState(null), [busy, setBusy] = useState('');
  const [rock, setRock] = useState(Object.fromEntries(ROCK.map(([k, , v]) => [k, String(v)])));
  const [spacing, setSpacing] = useState('250');
  const [aqMode, setAqMode] = useState('imbalance'), [aqv, setAqv] = useState(Object.fromEntries(AQ.map(([k, , v]) => [k, String(v)]))), [method, setMethod] = useState('klins'), [unit, setUnit] = useState('days');
  const [dpStaged, setDpStaged] = useState(null);
  const dpRef = useRef(null);
  const [dpTable, setDpTable] = useState(false);
  const fitted = matchRuns(runs).find(r => r.status === 'done');
  const fit = useRun(fitted?.id);
  async function call(path, body, setter, label) {
    setBusy(label);
    try { setter(await api(path, {method: 'POST', body})); } catch (e) { notify(e.message, 'error'); } finally { setBusy(''); }
  }
  const pair = fit?.outputs;
  const [pairSel, setPairSel] = useState('0,0');
  const [pj, pi] = pairSel.split(',').map(Number);
  const tauPair = pair ? (pair.tau_selection === 'per-pair' ? pair.tau[pj]?.[pi] : pair.tau[pj]) : null;
  return <>
    <section>
      <div className="sectionHead"><div><h2>Multiwell productivity index (geometry)</h2><p>Turns well locations into the influence matrix Λ for a rectangular, homogeneous reservoir. −Λ is the geometry seed for CRM gains. Use it as a ranking, then let CRM train.</p></div></div>
      {!d.locations ? <p className="muted">Upload locations.csv on the Data tab.</p> : <>
        <div className="inline"><fieldset><label>Reservoir length x_e<input type="number" step="any" value={xe} onChange={e => setXe(e.target.value)}/></label>
          <label>Reservoir width y_e<input type="number" step="any" value={ye} onChange={e => setYe(e.target.value)}/></label>
          <button disabled={busy === 'mpi'} onClick={() => call('/crm/mpi', {field, x_e: Number(xe), y_e: Number(ye)}, setMpi, 'mpi')}>{busy === 'mpi' ? 'Computing…' : 'Compute Λ'}</button></fieldset></div>
        <small>Wells are translated so the lower-left well sits at the origin; every well must lie inside (0, x_e) × (0, y_e).</small>
        {mpi && <div className="grid2">
          <div><h3>MPI Λ (raw, negative)</h3><Matrix rows={mpi.producers} cols={mpi.injectors} values={mpi.lambda_} tone="blue" digits={3}/></div>
          <div><h3>CRM seed −Λ</h3><Matrix rows={mpi.producers} cols={mpi.injectors} values={mpi.seed} tone="red" digits={3}/>
            {pair && pair.producers.join() === mpi.producers.join() && pair.injectors.join() === mpi.injectors.join() && <><h3>Fitted gains ({fitted.name})</h3><Matrix rows={pair.producers} cols={pair.injectors} values={pair.gains} tone="red" digits={3}/></>}</div>
        </div>}</>}
    </section>
    <section>
      <div className="sectionHead"><div><h2>Buckley-Leverett front</h2><p>1-D waterflood with Brooks–Corey relative permeability. Use it to sanity-check a pair CRM says is strongly connected: travel time ≈ L / v(S<sub>w,bt</sub>) should be the same order as τ.</p></div></div>
      <div className="rowFields">{ROCK.map(([k, label]) => <label key={k}>{label}<input type="number" step="any" value={rock[k]} onChange={e => setRock(r => ({...r, [k]: e.target.value}))}/></label>)}
        <label>Well spacing L (m)<input type="number" step="any" value={spacing} onChange={e => setSpacing(e.target.value)}/></label></div>
      <div className="toolbar"><button disabled={busy === 'bl'} onClick={() => call('/crm/bl', {...Object.fromEntries(Object.entries(rock).map(([k, v]) => [k, Number(v)])), spacing_m: Number(spacing)}, setBl, 'bl')}>{busy === 'bl' ? 'Computing…' : 'Compute front'}</button></div>
      {bl && <>
        <div className="metrics">{[['S_w at breakthrough', fmt(bl.sw_bt, 3)], ['Mobile oil at breakthrough', fmt(bl.mobile_oil, 3)], ['Front speed v(S_w,bt)', fmt(bl.v_bt, 3) + ' m/d'], ['Travel time L / v', bl.travel_days != null ? fmt(bl.travel_days, 1) + ' days' : '—']].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
        <div className="grid2">
          <div><h3>Water-front velocity</h3><LineChart rows={bl.curve} x="Sw" xLabel="S_w" unit="m/d" height={190} title="front velocity" series={[{key: 'v', label: 'v(S_w)'}]} markers={[{x: bl.sw_bt, label: 'S_w,bt'}]}/></div>
          <div><h3>Saturation profiles (distance vs S_w)</h3><LineChart rows={bl.profiles} x="Sw" xLabel="S_w" unit="m" height={190} title="saturation profiles" series={bl.times.slice(1).map((t, k) => ({key: `t${t}`, label: `t = ${t} d`, color: COLORS[k % COLORS.length]}))}/></div>
        </div>
        {pair && <div className="pairDetail"><h3>Compare with a CRM pair ({fitted.name})</h3>
          <label className="inlineLabel">Pair<select value={pairSel} onChange={e => setPairSel(e.target.value)}>{pair.producers.flatMap((p, j) => pair.injectors.map((inj, i) => <option key={`${j},${i}`} value={`${j},${i}`}>{inj} → {p}</option>))}</select></label>
          <KV items={[['f (gain)', fmt(pair.gains[pj]?.[pi], 3)], ['CRM τ', fmt(tauPair, 1)], ['Buckley-Leverett travel time', bl.travel_days != null ? fmt(bl.travel_days, 1) + ' days' : 'enter spacing']]}/>
          <p className="muted small">CRM τ is a tank lag, not a pure delay: expect agreement in magnitude, not equality. A large f with a travel time much longer than τ is a mathematical fit, not a short path.</p></div>}
      </>}
    </section>
    <section>
      <div className="sectionHead"><div><h2>Aquifer influx as a pseudo-injector</h2><p>If field production runs above injection, add the missing support as an extra injector column and refit with “up-to one”. Gains in that column show which producers receive aquifer support.</p></div></div>
      <div className="inline"><fieldset><label>Pseudo-injector rate<select value={aqMode} onChange={e => setAqMode(e.target.value)}><option value="imbalance">Field imbalance max(production − injection, 0)</option><option value="influx">Aquifer influx dW_e/dt (van Everdingen–Hurst)</option></select></label></fieldset></div>
      {aqMode === 'influx' && <>
        <div className="rowFields">{AQ.map(([k, label]) => <label key={k}>{label}<input type="number" step="any" value={aqv[k]} onChange={e => setAqv(r => ({...r, [k]: e.target.value}))}/></label>)}
          <label>Method<select value={method} onChange={e => setMethod(e.target.value)}><option value="klins">klins</option><option value="marsal-walsh">marsal-walsh</option></select></label>
          <label>time.csv unit<select value={unit} onChange={e => setUnit(e.target.value)}><option value="days">days</option><option value="years">years</option></select></label></div>
        <div className="fileInline"><button className="secondary small" onClick={() => dpRef.current.click()}>Import pressure-drop CSV</button><button className="secondary small" onClick={() => setDpTable(true)}>Fill table</button>
          <input ref={dpRef} type="file" accept=".csv,.txt" hidden onChange={async e => { const f = e.target.files[0]; e.target.value = ''; if (!f) return; try { setDpStaged(await upload('/crm/upload/aquifer_dp' + qs({field}), f)); } catch (err) { notify(err.message, 'error'); } }}/>
          <small>{dpStaged ? `${dpStaged.rows} values` : `Headerless, one pressure drop per time step (${d.time.length} rows).`}</small></div>
        {dpTable && <CsvTable title="Pressure drop per time step" headerless columns={[{key: 'dp', label: 'pressure drop'}]} initial={d.time.map(() => [''])} minRows={0}
          note={`One value per time step (${d.time.length} rows, same order as time.csv), no header row.`} onClose={() => setDpTable(false)} saveLabel="Use table"
          onSave={async rows => { setDpStaged(await stageTable(field, 'aquifer_dp', null, rows)); setDpTable(false); notify('Table checked'); }}/>}
        <small>t_D = k_t·t with k_t = 2.309 k_a / (φ_a μ_w c_t r_o²), r_o from pore volume, porosity, thickness and open angle. U = 1.119 h f r_o² φ c_t.</small>
      </>}
      <div className="toolbar"><button disabled={busy === 'aq'} onClick={() => call('/crm/aquifer', {field, mode: aqMode, method, time_unit: unit, ...Object.fromEntries(Object.entries(aqv).map(([k, v]) => [k, Number(v)]))}, setAq, 'aq')}>{busy === 'aq' ? 'Computing…' : 'Compute pseudo-injector'}</button>
        {aq && appMode === 'edit' && <button className="secondary" onClick={async () => { try { const r = await api('/crm/aquifer', {method: 'POST', body: {field, mode: aqMode, method, time_unit: unit, apply: true, name: 'AQUIFER', ...Object.fromEntries(Object.entries(aqv).map(([k, v]) => [k, Number(v)]))}}); await reload(); notify(`AQUIFER added as dataset v${r.applied_version}. Refit the history match.`); } catch (e) { notify(e.message, 'error'); } }}>Add as injector “AQUIFER” (new dataset version)</button>}</div>
      {aq && <><p className="muted">{aq.positive_steps} of {aq.steps} steps have positive support.{aq.params?.r_o != null && ` r_o = ${fmt(aq.params.r_o, 2)}, k_t = ${fmt(aq.params.k_t, 5)}.`}</p>
        <LineChart rows={aq.time.map((t, k) => ({t, rate: aq.rate[k]}))} x="t" xLabel="time" height={180} title="pseudo-injector" series={[{key: 'rate', label: 'Pseudo-injector rate'}]}/></>}
    </section>
  </>;
}

function Matrix({rows, cols, values, tone, digits}) {
  const max = Math.max(...values.flat().map(Math.abs)) || 1;
  const color = v => tone === 'blue' ? `rgba(47,93,155,${Math.abs(v) / max * 0.8})` : `rgba(192,57,43,${Math.abs(v) / max * 0.8})`;
  return <div className="scroll"><table className="heat"><thead><tr><th/>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
    <tbody>{rows.map((r, j) => <tr key={r}><th>{r}</th>{cols.map((c, i) => <td key={c} className="cell static" style={{background: color(values[j][i]), color: Math.abs(values[j][i]) / max > 0.55 ? '#fff' : undefined}}>{fmt(values[j][i], digits)}</td>)}</tr>)}</tbody></table></div>;
}
