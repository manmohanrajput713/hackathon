'use client';
import { useEffect, useState } from 'react';
import { api, download, fmt, pct, when, RUN_ICON } from './lib';
import { useCtx, useRun } from './ctx';
import { LineChart, Empty, Stale, KV, Confirm, COLORS, useReplay, Meter, Status } from './components';

export function Results() {
  const {runs, route, set, scenarios, reloadRuns, notify, getRun, mode, wells} = useCtx();
  const [kind, setKind] = useState(''), [scen, setScen] = useState(''), [status, setStatus] = useState(''), [checked, setChecked] = useState([]);
  const [compare, setCompare] = useState(false);
  const shown = runs.filter(r => (!kind || r.kind === kind) && (!scen || r.scenario_id === scen) && (!status || r.status === status));
  const kinds = [...new Set(runs.map(r => r.kind))];
  const open = route.run && runs.some(r => r.id === route.run) ? route.run : null;
  const toggle = id => setChecked(c => c.includes(id) ? c.filter(x => x !== id) : [...c, id]);
  const wellName = id => wells.find(w => w.id === id)?.name;
  return <>
    <div className="moduleHead"><div><small>Saved runs and scenarios</small><h1>Results</h1></div>
      <div className="actions"><button className="secondary" onClick={() => download('/field/export' + (route.field ? `?field=${encodeURIComponent(route.field)}` : '')).catch(e => notify(e.message, 'error'))}>Field report (Excel)</button></div></div>
    <div className="grid2 results">
      <section className="runList">
        <div className="filters">
          <select value={kind} onChange={e => setKind(e.target.value)} aria-label="Kind"><option value="">All kinds</option>{kinds.map(k => <option key={k} value={k}>{runs.find(r => r.kind === k).kind_label}</option>)}</select>
          <select value={scen} onChange={e => setScen(e.target.value)} aria-label="Scenario"><option value="">All scenarios</option>{scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select>
          <select value={status} onChange={e => setStatus(e.target.value)} aria-label="Status"><option value="">Any status</option>{['done', 'running', 'queued', 'failed'].map(s => <option key={s}>{s}</option>)}</select>
        </div>
        <div className="toolbar"><button className="secondary small" disabled={checked.length < 2} onClick={() => setCompare(true)}>Compare {checked.length || ''}</button>
          {checked.length > 0 && <button className="link small" onClick={() => setChecked([])}>Clear</button>}<span className="muted small">Tick runs of the same kind to compare them on common axes.</span></div>
        {!shown.length ? <Empty title="No runs">Runs from every module are saved here.</Empty> :
          <ul className="runs">{shown.map(r => <li key={r.id} className={(open === r.id ? 'on ' : '') + r.status}>
            <input type="checkbox" checked={checked.includes(r.id)} onChange={() => toggle(r.id)} aria-label={`Select ${r.name}`} disabled={r.status !== 'done'}/>
            <button className="runItem" onClick={() => { setCompare(false); set({run: r.id}); }}>
              <span className={'runIcon ' + r.status}>{RUN_ICON[r.status]}</span>
              <span className="runText"><b>{r.name}</b><small>{r.kind_label}{r.well_id && wellName(r.well_id) ? ` · ${wellName(r.well_id)}` : ''}{r.scenario_id ? ` · ${scenarios.find(s => s.id === r.scenario_id)?.name || 'scenario'}` : ''} · {when(r.created_at)}</small></span>
              {r.stale?.length > 0 && <span className="tag warnTag" title={r.stale.join('\n')}>Review</span>}
            </button></li>)}</ul>}
      </section>
      <div>{compare ? <Compare ids={checked}/> : open ? <RunDetail id={open} onDeleted={() => { set({run: null}); reloadRuns(); }}/> :
        <section><Empty title="Select a run">Open a run to see its outputs, the exact inputs it used, and whether those inputs have changed since.</Empty></section>}</div>
    </div>
  </>;
}

function RunDetail({id, onDeleted}) {
  const {notify, reloadRuns, go, mode} = useCtx();
  const run = useRun(id);
  const [name, setName] = useState(''), [remove, setRemove] = useState(false);
  useEffect(() => { if (run) setName(run.name); }, [run?.id, run?.name]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!run) return <section><p>Loading run…</p></section>;
  async function rename() { try { await api(`/runs/${id}`, {method: 'PUT', body: {name}}); notify('Renamed'); reloadRuns(); } catch (e) { notify(e.message, 'error'); } }
  const pins = run.inputs.datasets || {};
  return <section>
    <div className="sectionHead"><div><small>{run.kind_label} · {run.status}</small>
      {mode === 'edit' ? <div className="renameRow"><input value={name} onChange={e => setName(e.target.value)} aria-label="Run name"/>{name !== run.name && <button className="small" onClick={rename}>Save name</button>}</div> : <h2>{run.name}</h2>}</div>
      <div className="actions"><button className="secondary small" disabled={run.status !== 'done'} onClick={() => download(`/runs/${id}/export?format=xlsx`).catch(e => notify(e.message, 'error'))}>Excel</button>
        <button className="secondary small" disabled={run.status !== 'done'} onClick={() => download(`/runs/${id}/export?format=csv`).catch(e => notify(e.message, 'error'))}>CSV</button>
        <button className="secondary small dangerText editOnly" disabled={['queued', 'running'].includes(run.status)} onClick={() => setRemove(true)}>Delete</button></div></div>
    {run.status === 'failed' && <p className="formError">{run.error}</p>}
    {['queued', 'running'].includes(run.status) && <p className="muted">This run is {run.status}…</p>}
    <Stale reasons={run.stale}/>
    {run.outputs && <RunOutputs run={run}/>}
    <details className="pins"><summary>Pinned inputs</summary>
      <KV items={[['Created', when(run.created_at)], ['Finished', when(run.finished_at)], ['Field', run.field || '—'], ['Parameters', <code key="p">{JSON.stringify(run.params)}</code>],
        ['Dataset versions', Object.keys(pins).length ? Object.entries(pins).map(([w, d]) => `${w}: ${Object.entries(d).map(([k, v]) => `${k} v${v}`).join(', ')}`).join(' · ') : 'none'],
        ['Model versions', Object.keys(run.inputs.models || {}).join(', ') || '—'], ['Upstream runs', (run.inputs.parents || []).join(', ') || 'none'],
        run.inputs.scenario && ['Scenario snapshot', `${run.inputs.scenario.name}: ${Object.entries(run.inputs.scenario.facility).map(([k, v]) => `${k.replace(/_/g, ' ')} ${fmt(v)}`).join(', ')}; policy ${run.inputs.scenario.policy}`]]}/>
    </details>
    {remove && <Confirm title={`Delete ${run.name}?`} message="The run and its outputs are removed. Runs that depend on it are flagged for review." onClose={() => setRemove(false)}
      onConfirm={async () => { try { await api(`/runs/${id}`, {method: 'DELETE'}); notify('Run deleted'); onDeleted(); } catch (e) { notify(e.message, 'error'); } }}/>}
  </section>;
}

/** Outputs view per run kind, with replay for time series. */
function RunOutputs({run}) {
  const o = run.outputs;
  const series = timeSeries(run);
  const [cursor, replay] = useReplay(series?.rows.length || 0);
  const [pick, setPick] = useState(0);
  if (run.kind === 'optimize') return <>
    {!o.feasible ? <p className="warning">{o.message}</p> : <>
      <div className="metrics">{[['Oil now', fmt(o.totals.current_oil, 0)], ['Recommended oil', fmt(o.totals.oil, 0)], ['Lift gas', fmt(o.totals.lift_gas, 0)], ['Energy kWh/d', fmt(o.totals.energy, 0)]].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
      <table className="mini"><thead><tr><th>Well</th><th>Current</th><th>Recommended</th><th>Oil now</th><th>Oil predicted</th><th>Risk</th><th>Rule</th></tr></thead>
        <tbody>{o.wells.map(w => <tr key={w.id}><td>{w.well}</td><td>{fmt(w.current, w.unit === 'Hz' ? 1 : 0)} {w.unit}</td><td>{w.shut_in ? 'Shut in' : `${fmt(w.recommended, w.unit === 'Hz' ? 1 : 0)} ${w.unit}`}</td><td>{fmt(w.current_oil, 0)}</td><td>{fmt(w.oil, 0)}</td><td><Status value={w.risk}/></td><td>{w.policy_applied}</td></tr>)}</tbody></table>
      {o.constraints.map(c => <Meter key={c.name} {...c}/>)}</>}
    {o.excluded?.length > 0 && <p className="muted">Not optimised: {o.excluded.map(e => `${e.well} (${e.reason})`).join('; ')}</p>}
  </>;
  if (run.kind === 'crm_match') {
    const j = Math.min(pick, o.producers.length - 1);
    const rows = o.time.map((t, k) => ({t, observed: o.observed[k][j], crm: o.predicted[k][j]}));
    return <>
      <label className="inlineLabel">Producer<select value={j} onChange={e => setPick(Number(e.target.value))}>{o.producers.map((x, i) => <option key={x} value={i}>{x}</option>)}</select></label>
      <KV items={[['RMSE fit', fmt(o.rmse_fit_overall)], ['RMSE held out', fmt(o.rmse_holdout_overall)], ['Model', `${o.model === 'compensated' ? 'CrmCompensated' : 'CRM'} · ${o.tau_selection} · ${o.constraints}`]]}/>
      {replay}
      <LineChart rows={rows} x="t" xLabel="time" cursor={cursor} title={run.name} bands={o.train_rows < o.time.length ? [{from: o.time[o.train_rows], to: o.time[o.time.length - 1], label: 'Held out'}] : []}
        series={[{key: 'observed', label: 'Observed', color: '#aa7551', dashed: true}, {key: 'crm', label: 'CRM'}]}/>
      <table className="mini"><thead><tr><th>Producer / injector</th>{o.injectors.map(x => <th key={x}>{x}</th>)}</tr></thead>
        <tbody>{o.producers.map((p, pj) => <tr key={p}><th>{p}</th>{o.injectors.map((x, ii) => <td key={x}>{fmt(o.gains[pj][ii], 3)} <small>τ {fmt(o.tau_selection === 'per-pair' ? o.tau[pj][ii] : o.tau[pj], 1)}</small></td>)}</tr>)}</tbody></table>
    </>;
  }
  if (run.kind === 'crm_forecast') {
    const j = Math.min(pick, o.producers.length - 1);
    const rows = [...o.history_time.map((t, k) => ({t, observed: o.observed[k][j], crm: o.history_predicted[k][j]})), ...o.time.map((t, k) => ({t, plan: o.plan[k][j]}))];
    return <>
      <label className="inlineLabel">Producer<select value={j} onChange={e => setPick(Number(e.target.value))}>{o.producers.map((x, i) => <option key={x} value={i}>{x}</option>)}</select></label>
      <KV items={[['Plan starts', fmt(o.forecast_start)], ['Steps', o.time.length]]}/>
      {replay}
      <LineChart rows={rows} x="t" xLabel="time" cursor={cursor} title={run.name} markers={[{x: o.forecast_start, label: 'Plan starts'}]}
        series={[{key: 'observed', label: 'Observed', dashed: true, color: '#aa7551'}, {key: 'crm', label: 'CRM (history)'}, {key: 'plan', label: 'Plan', color: '#2f5d9b'}]}/>
    </>;
  }
  if (run.kind === 'well_curve' && o.nodal) return <LineChart rows={o.nodal.ipr.map((ip, k) => ({q: ip.q, ipr: ip.p,
      ...Object.fromEntries(o.nodal.curves.map((c, i) => [`vlp${i}`, c.points[k].p != null && c.points[k].p <= o.nodal.reservoir_bar * 1.35 ? c.points[k].p : null]))}))}
    x="q" xLabel={`Liquid rate (${o.nodal.rate_unit})`} unit={o.nodal.pressure_unit} title={run.name}
    points={o.points.filter(p => p.feasible).map(p => ({x: p.liquid_m3d, y: p.bhp_bar, color: p.stable === false ? '#c0392b' : '#24302b'}))}
    series={[{key: 'ipr', label: 'Inflow (IPR)', color: '#24302b'}, ...o.nodal.curves.map((c, i) => ({key: `vlp${i}`, label: `VLP ${fmt(c.setpoint)} ${o.nodal.curve_unit}`, dashed: true, color: COLORS[(i + 1) % COLORS.length]}))]}/>;
  if (run.kind === 'well_curve') return <LineChart rows={o.points.filter(p => p.feasible)} x="setpoint" xLabel={o.setpoint_unit} unit="m³/d" title={run.name}
    series={[{key: 'oil_m3d', label: 'Oil'}, {key: 'liquid_m3d', label: 'Liquid', dashed: true}]}/>;
  if (series) return <>
    {o.warnings?.map(w => <p key={w} className="warning">{w}</p>)}
    {replay}
    <LineChart rows={series.rows} x={series.x} cursor={cursor} threshold={o.threshold} title={run.name} series={series.series}/>
  </>;
  return <p className="muted">No chart for this run kind.</p>;
}

function timeSeries(run) {
  const o = run.outputs;
  if (!o) return null;
  if (run.kind === 'vfm') return {rows: o.rows, x: 'timestamp', series: [{key: 'vfm_m3h', label: 'Virtual liquid rate'}, {key: 'test_m3h', label: 'Well test', dots: true, color: '#aa7551'}]};
  if (run.kind === 'esp_risk') return {rows: o.rows, x: 'date', series: [{key: 'risk', label: 'ESP failure risk'}]};
  if (run.kind === 'valve_risk') return {rows: o.rows, x: 'timestamp', series: [{key: 'risk', label: 'Valve failure risk'}]};
  if (run.kind === 'crm_match') return {rows: o.time};
  if (run.kind === 'crm_forecast') return {rows: [...o.history_time, ...o.time]};
  return null;
}

function Compare({ids}) {
  const {getRun} = useCtx();
  const [list, setList] = useState(null);
  useEffect(() => { Promise.all(ids.map(getRun)).then(setList).catch(() => setList([])); }, [ids.join(','), getRun]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!list) return <section><p>Loading runs…</p></section>;
  const kinds = new Set(list.map(r => r.kind));
  if (kinds.size > 1) return <section><p className="warning">Compare runs of the same kind. Selected: {[...kinds].join(', ')}.</p></section>;
  const kind = list[0].kind;
  if (kind === 'optimize') {
    const wellNames = [...new Set(list.flatMap(r => (r.outputs.wells || []).map(w => w.well)))];
    const constraintNames = [...new Set(list.flatMap(r => r.outputs.constraints.map(c => c.name)))];
    return <section>
      <h2>Optimizer comparison</h2>
      <div className="scroll"><table><thead><tr><th>Well</th>{list.map(r => <th key={r.id}>{r.name}<small>{r.inputs.scenario?.name}</small></th>)}</tr></thead>
        <tbody>{wellNames.map(n => <tr key={n}><td><b>{n}</b></td>{list.map(r => { const w = (r.outputs.wells || []).find(x => x.well === n); return <td key={r.id}>{w ? <>{w.shut_in ? 'Shut in' : `${fmt(w.recommended, w.unit === 'Hz' ? 1 : 0)} ${w.unit}`}<small>{fmt(w.oil, 0)} m³/d oil</small></> : '—'}</td>; })}</tr>)}
          <tr className="total"><td>Total oil (m³/d)</td>{list.map(r => <td key={r.id}><b>{r.outputs.feasible ? fmt(r.outputs.totals.oil, 0) : 'infeasible'}</b></td>)}</tr>
          <tr><td>Lift gas (sm³/d)</td>{list.map(r => <td key={r.id}>{r.outputs.feasible ? fmt(r.outputs.totals.lift_gas, 0) : '—'}</td>)}</tr>
          <tr><td>Energy (kWh/d)</td>{list.map(r => <td key={r.id}>{r.outputs.feasible ? fmt(r.outputs.totals.energy, 0) : '—'}</td>)}</tr></tbody></table></div>
      <h3>Constraint margins</h3>
      <div className="scroll"><table><thead><tr><th>Constraint</th>{list.map(r => <th key={r.id}>{r.name}</th>)}</tr></thead>
        <tbody>{constraintNames.map(c => <tr key={c}><td>{c}</td>{list.map(r => { const x = r.outputs.constraints.find(k => k.name === c); return <td key={r.id}>{x ? (x.used != null ? <>{fmt(x.used, 0)} / {fmt(x.capacity, 0)} {x.unit}<small>{x.binding ? 'binding' : `slack ${fmt(x.slack, 0)}`}</small></> : `needs ${fmt(x.minimum_needed, 0)}`) : 'no limit'}</td>; })}</tr>)}</tbody></table></div>
      <LineChart rows={wellNames.map(n => Object.fromEntries([['well', wellNames.indexOf(n)], ...list.map((r, i) => [`r${i}`, (r.outputs.wells || []).find(w => w.well === n)?.oil ?? null])]))}
        x="well" xLabel={wellNames.join(' · ')} unit="m³/d" title="oil by well" series={list.map((r, i) => ({key: `r${i}`, label: r.name, dots: true, color: COLORS[i % COLORS.length]}))}/>
    </section>;
  }
  if (kind === 'crm_forecast' || kind === 'crm_match') {
    const producers = list[0].outputs.producers;
    return <section>
      <h2>{kind === 'crm_forecast' ? 'Forecast comparison' : 'History match comparison'}</h2>
      {producers.map((p, j) => {
        const byT = new Map();
        list.forEach((r, i) => { const o = r.outputs; const k = o.producers.indexOf(p); if (k < 0) return;
          (kind === 'crm_forecast' ? o.time : o.time).forEach((t, n) => { const e = byT.get(t) || {t}; e[`r${i}`] = kind === 'crm_forecast' ? o.plan[n][k] : o.predicted[n][k]; if (kind === 'crm_match') e.obs = o.observed[n][k]; byT.set(t, e); }); });
        const rows = [...byT.values()].sort((a, b) => a.t - b.t);
        return <div key={p}><h3>{p}</h3><LineChart rows={rows} x="t" xLabel="time" height={190} title={`${p} comparison`}
          series={[...(kind === 'crm_match' ? [{key: 'obs', label: 'Observed', dashed: true, color: '#999'}] : []), ...list.map((r, i) => ({key: `r${i}`, label: r.name, color: COLORS[i % COLORS.length]}))]}/></div>;
      })}
      {kind === 'crm_match' && <table className="mini"><thead><tr><th>Run</th><th>Model</th><th>RMSE fit</th><th>RMSE held out</th>{producers.map(p => <th key={p}>{p} held out</th>)}</tr></thead>
        <tbody>{list.map(r => { const o = r.outputs; return <tr key={r.id}><td>{r.name}</td><td>{o.model} · {o.tau_selection} · {o.constraints}</td><td>{fmt(o.rmse_fit_overall)}</td><td>{fmt(o.rmse_holdout_overall)}</td>
          {producers.map(p => <td key={p}>{o.rmse_holdout ? fmt(o.rmse_holdout[o.producers.indexOf(p)]) : '—'}</td>)}</tr>; })}</tbody></table>}
    </section>;
  }
  return <section><h2>Comparison</h2><table className="mini"><thead><tr><th>Run</th><th>Finished</th><th>Summary</th></tr></thead>
    <tbody>{list.map(r => <tr key={r.id}><td>{r.name}</td><td>{when(r.finished_at)}</td><td>{Object.entries(r.summary || {}).map(([k, v]) => `${k}: ${typeof v === 'number' ? fmt(v, 3) : v}`).join(' · ')}</td></tr>)}</tbody></table></section>;
}
