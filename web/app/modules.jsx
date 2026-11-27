'use client';
import { useEffect, useMemo, useState } from 'react';
import { api, fmt, pct, when, typeTone } from './lib';
import { useCtx, useRun, latest } from './ctx';
import { LineChart, Empty, Tabs, Stale, KV, Meter, Status, COLORS, useReplay } from './components';
import { DatasetPanel, RunBar, WellForm } from './data';

/** The active well if it has one of the given types, otherwise the first such well, plus a picker. */
function useTypedWell(types) {
  const {wells, well, set, schema} = useCtx();
  const list = wells.filter(w => types.includes(w.type));
  const current = well && types.includes(well.type) ? well : list[0] || null;
  const picker = list.length > 0 && <label className="inlineLabel">Well<select value={current?.id || ''} onChange={e => set({well: e.target.value})}>
    {list.map(w => <option key={w.id} value={w.id}>{w.name} ({schema.wellTypes[w.type].label})</option>)}</select></label>;
  return [current, picker, list];
}

function NoWells({what, types}) {
  const {go, schema} = useCtx();
  return <section><Empty title={`No ${types.map(t => schema.wellTypes[t].label).join(' or ')} in this field`} action={<button onClick={() => go('crm', 'wells')}>Add a well in Deep-CRM → Wells</button>}>{what}</Empty></section>;
}

// ---------- VFM ----------

const VFM_DATASET = {esp: 'esp_fast', gas_lift: 'gl_fast'};

export function VFM() {
  const {runs, select} = useCtx();
  const [w, picker] = useTypedWell(['esp', 'gas_lift']);
  const last = w && latest(runs, 'vfm', {well_id: w.id});
  const run = useRun(last?.id);
  const o = run?.outputs;
  const [cursor, replay] = useReplay(o?.rows.length || 0);
  const sub = 'Liquid rate from ESP and gas-lift sensors';
  if (!w) return <><Head title="Virtual flow meter" sub={sub}/><NoWells types={['esp', 'gas_lift']} what="The virtual flow meter estimates liquid rate for ESP producers (pinn checkpoint) and gas-lift producers (gas_pinc model) from high-frequency sensor data."/></>;
  const gl = w.type === 'gas_lift';
  const dataset = VFM_DATASET[w.type];
  const cov = o?.coverage;
  const hasData = !!w.counts[dataset]?.rows;
  const covMetrics = gl
    ? [['Samples', fmt(cov?.samples, 0)], ['Estimated', `${fmt(cov?.valid_windows, 0)} (${pct(cov?.valid_windows / cov?.samples)})`],
       ['State from pressures', fmt(cov?.measured, 0)], ['Carried forward', fmt(cov?.propagated, 0)],
       ['Median spacing', fmt(cov?.spacing_s?.median, 2) + ' s'], ['Off-cadence steps', fmt(cov?.spacing_s?.off_cadence, 0)]]
    : [['Samples', fmt(cov?.samples, 0)], ['Samples with full history', `${fmt(cov?.valid_windows, 0)} (${pct(cov?.valid_windows / cov?.samples)})`],
       ['Median spacing', fmt(cov?.spacing_s?.median, 2) + ' s'], ['Off-cadence steps', fmt(cov?.spacing_s?.off_cadence, 0)]];
  const check = o?.model?.pinc_check;
  return <>
    <Head title="Virtual flow meter" sub={sub} right={picker}/>
    <section>
      <div className="sectionHead"><div><h2>{w.name}: model inputs and coverage</h2>
        <p>{gl
          ? 'gas_pinc gas-lift model. Casing and tubing-head pressure (plus bottom-hole pressure when a gauge exists) fix the plant state at each sample; the gas_pinc algebraic network turns that state into liquid rate, lift-gas rate and bottom-hole pressure. Samples without pressures are carried forward one 60 s gas_pinc control step from the previous estimate.'
          : `pinn ESP checkpoint. It needs ${cov?.window ?? 20} consecutive samples spaced ${w.settings.vfm_sample_seconds ?? 1} s apart before each estimate. Gaps or off-cadence samples make inference impossible until the window refills.`}</p></div></div>
      <RunBar kind="vfm" label="Run flow meter" params={{well_id: w.id}} filter={{well_id: w.id}} disabled={!hasData} hint={hasData ? undefined : 'Add high-frequency telemetry below.'}/>
      {cov && <>
        <div className="metrics">{covMetrics.map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
        <div className="coverage"><div className="covBar">{cov.segments.map((s, i) => <i key={i} className={s.valid ? 'ok' : 'gap'} style={{flex: s.samples}} title={`${s.start} → ${s.end}: ${s.valid ? 'estimate available' : 'no estimate'} (${s.samples} samples)`}/>)}</div>
          <div className="legendLine"><span className="swatch ok"/>{gl ? 'estimate available' : 'inference possible'} <span className="swatch gap"/>{gl ? 'no pressures and nothing to carry forward' : 'insufficient sample history'}</div></div>
        <table className="mini"><thead><tr><th>Input</th><th>Present</th></tr></thead><tbody>{Object.entries(cov.channels).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, ' ')}</td><td className={v < 0.99 && !['test_m3h', 'bottomhole_bar'].includes(k) ? 'warnCell' : ''}>{pct(v)}</td></tr>)}</tbody></table>
      </>}
      {gl && o?.model && <KV items={[['Rates from', o.model.rates], ['Carry-forward model', o.model.propagation],
        ['gas_pinc PINC check', check?.error != null ? `${check.usable ? 'passes' : 'fails'}: one-step error ${fmt(check.error, 3)} (tolerance 0.02)` : (check?.reason || '—')],
        ['Network / equation samples', `${fmt(o.model.network_rows, 0)} / ${fmt(o.model.equation_rows, 0)}`]]}/>}
    </section>
    {o && <section>
      <div className="sectionHead"><div><h2>Virtual liquid rate vs well tests</h2><small>{run.name} · {when(run.finished_at)}</small></div></div>
      <Stale reasons={run.stale}/>
      {o.warnings.map(x => <p key={x} className="warning">{x}</p>)}
      <div className="metrics">{[['Latest estimate', fmt(o.summary.value) + ' m³/h'], ['At', o.summary.at], ['Well tests compared', o.metrics.tests ?? 0],
        ['Mean absolute error', o.metrics.mean_abs_error_m3h != null ? fmt(o.metrics.mean_abs_error_m3h) + ' m³/h' : '—'],
        ...(gl && o.metrics.propagated_bhp_mae_bar != null ? [['Carried-forward BHP error', fmt(o.metrics.propagated_bhp_mae_bar, 3) + ' bar']] : [])].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
      {replay}
      <LineChart rows={o.rows} x="timestamp" unit="m³/h" cursor={cursor} title={`${w.name} flow meter`} series={[{key: 'vfm_m3h', label: 'Virtual liquid rate'}, {key: 'test_m3h', label: 'Well test', dots: true, color: '#aa7551'}]}/>
      {gl && <div className="grid2">
        <div><h3>Bottom-hole pressure (bar a)</h3><LineChart rows={o.rows} x="timestamp" unit="bar" height={190} cursor={cursor} title={`${w.name} BHP`} series={[{key: 'bhp_bar', label: 'Estimated'}, {key: 'bhp_measured_bar', label: 'Gauge', dashed: true, color: '#2f5d9b'}]}/></div>
        <div><h3>Lift gas through the gas-lift valve (sm³/d)</h3><LineChart rows={o.rows} x="timestamp" unit="sm³/d" height={190} cursor={cursor} title={`${w.name} lift gas`} series={[{key: 'lift_gas_sm3d', label: 'Lift gas', color: '#8a4fa3'}]}/></div>
      </div>}
      <button className="link" onClick={() => select({title: `${w.name} rate estimate`, kind: 'vfm', items: [['Rate', fmt(o.summary.value) + ' m³/h'], ['Timestamp', o.summary.at], ['Model', gl ? 'gas_pinc gas-lift' : 'pinn ESP'], ['Quality', o.summary.quality === 'good' ? 'Good' : 'Check warnings']], text: 'The Well Models and Results tabs use this estimate with its timestamp and quality flag.'})}>Show estimate in inspector</button>
    </section>}
    <DatasetPanel wellId={w.id} dataset={dataset}/>
  </>;
}

function Head({title, sub, right}) {
  return <div className="moduleHead"><div><small>{sub}</small><h1>{title}</h1></div>{right && <div className="actions">{right}</div>}</div>;
}

// ---------- Well Models (nodal analysis) ----------

const METRICS = {
  esp: [['oil_m3d', 'Oil', 'm³/d'], ['water_m3d', 'Water', 'm³/d'], ['gas_sm3d', 'Formation gas', 'sm³/d'], ['bhp_bar', 'Bottom-hole pressure', 'bar'], ['whp_bar', 'Wellhead pressure', 'bar'], ['power_kw', 'Power', 'kW']],
  gas_lift: [['oil_m3d', 'Oil', 'm³/d'], ['water_m3d', 'Water', 'm³/d'], ['gas_sm3d', 'Formation gas', 'sm³/d'], ['bhp_bar', 'Bottom-hole pressure', 'bar'], ['valve', 'Gas-lift valve opening', 'fraction']],
};

/** IPR plus a readable subset of VLP curves on one rate grid, and the operating points where they cross. */
function NodalChart({o, current, name}) {
  const [all, setAll] = useState(false);
  const n = o.nodal;
  // Curves the lift-gas supply cannot deliver stay in the export but are not drawn.
  const usable = n.curves.map((c, i) => c.feasible === false ? null : i).filter(i => i != null);
  const near = current != null && usable.length
    ? usable.reduce((b, i) => Math.abs(n.curves[i].setpoint - current) < Math.abs(n.curves[b].setpoint - current) ? i : b, usable[0]) : null;
  const pick = all || usable.length <= 7 ? usable
    : [...new Set([...Array.from({length: 6}, (_, k) => usable[Math.round(k * (usable.length - 1) / 5)]), ...(near != null ? [near] : [])])].sort((a, b) => a - b);
  const cap = n.reservoir_bar * 1.35;  // keep the crossing region readable: outflow far above reservoir pressure is off-scale
  const clip = p => p == null || p < 0 || p > cap ? null : p;
  const rows = n.ipr.map((ip, k) => Object.fromEntries([['q', ip.q], ['ipr', clip(ip.p)], ...pick.map(i => [`vlp${i}`, clip(n.curves[i].points[k].p)])]));
  const color = j => COLORS[(j + 1) % COLORS.length];
  const series = [{key: 'ipr', label: 'Inflow (IPR)', color: '#24302b'},
    ...pick.map((i, j) => ({key: `vlp${i}`, label: `VLP ${fmt(n.curves[i].setpoint)} ${n.curve_unit}${i === near ? ' (current)' : ''}`, color: color(j), dashed: i !== near}))];
  const ops = o.points.filter(p => p.feasible).map(p => {
    const j = pick.findIndex(i => Math.abs(n.curves[i].setpoint - p.setpoint) < 1e-6);
    return j < 0 ? null : {x: p.liquid_m3d, y: p.bhp_bar, color: p.stable === false ? '#c0392b' : color(j), label: pick[j] === near ? 'Current' : null};
  }).filter(Boolean);
  return <>
    <LineChart rows={rows} x="q" xLabel={`Liquid rate (${n.rate_unit})`} unit={n.pressure_unit} yLabel={`${n.node} pressure (${n.pressure_unit})`} height={340} points={ops} title={`${name} nodal analysis`} series={series}/>
    <div className="legendLine"><span className="swatch" style={{background: '#24302b'}}/>operating point = IPR/VLP crossing <span className="swatch" style={{background: '#c0392b'}}/>unstable (casing heading)
      {usable.length > 7 && <label className="inlineLabel"><input type="checkbox" checked={all} onChange={e => setAll(e.target.checked)}/>Show all {usable.length} outflow curves</label>}</div>
  </>;
}

export function WellModels() {
  const {runs, schema, reloadWells, select} = useCtx();
  const [w, picker] = useTypedWell(['esp', 'gas_lift']);
  const [edit, setEdit] = useState(false);
  const last = w && latest(runs, 'well_curve', {well_id: w.id});
  const run = useRun(last?.id);
  const [tested, setTested] = useState([]);
  useEffect(() => {
    if (w?.type === 'gas_lift' && w.counts.gl_curve?.rows) api(`/wells/${w.id}/data/gl_curve?limit=500`).then(d => setTested(d.rows)).catch(() => setTested([]));
    else setTested([]);
  }, [w?.id, w?.counts?.gl_curve?.version]); // eslint-disable-line react-hooks/exhaustive-deps
  const sub = 'Nodal analysis at the bottom hole';
  if (!w) return <><Head title="Well models" sub={sub}/><NoWells types={['esp', 'gas_lift']} what="Configure ESP or gas-lift equipment and fluid inputs, then run a nodal analysis: inflow (IPR) against outflow (VLP) curves for each pump frequency or lift-gas rate."/></>;
  const o = run?.outputs;
  const s = w.settings;
  const current = w.type === 'esp' ? s.current_hz : s.current_gas_sm3d;
  const env = w.type === 'esp' ? [s.env_min_hz, s.env_max_hz] : [s.env_min_gas, s.env_max_gas];
  const feasible = o ? o.points.filter(p => p.feasible) : [];
  const spec = schema.settings[w.type].filter(x => x.group === 'Well model');
  const rows = feasible.map(p => ({...p, unstable_oil: p.stable === false ? p.oil_m3d : null}));
  const testedRows = tested.map(t => ({setpoint: t.gas_sm3d, tested: t.oil_m3d}));
  const merged = [...rows, ...testedRows].sort((a, b) => a.setpoint - b.setpoint);
  const markers = [current != null && {x: current, label: 'Current', color: '#c0392b'}, env[0] != null && {x: env[0], label: 'Env min', color: '#8a4fa3'}, env[1] != null && {x: env[1], label: 'Env max', color: '#8a4fa3'}].filter(Boolean);
  return <>
    <Head title="Well models" sub={sub} right={picker}/>
    <div className="grid2 wide1">
      <section>
        <div className="sectionHead"><div><h2>{w.name} · {schema.wellTypes[w.type].label}</h2><p>{w.type === 'esp'
          ? 'Inflow from the productivity index; outflow from the pinn ESP equations (choke, tubing friction, hydrostatic head and pump head) at each frequency.'
          : 'Inflow from the productivity index; outflow from the gas_pinc gas-lift plant at steady state for each lift-gas rate.'}</p></div>
          <button className="secondary editOnly" onClick={() => setEdit(true)}>Edit inputs</button></div>
        <KV items={spec.map(x => [`${x.label}${x.unit ? ` (${x.unit})` : ''}`, fmt(s[x.key] ?? x.default, 4)])}/>
        <KV items={[[w.type === 'esp' ? 'Current frequency (Hz)' : 'Current lift gas (sm³/d)', fmt(current)], ['Approved envelope', env.some(v => v != null) ? `${fmt(env[0])} – ${fmt(env[1])}` : 'not set']]}/>
        <RunBar kind="well_curve" label="Run nodal analysis" params={{well_id: w.id}} filter={{well_id: w.id}} hint="Gas-lift nodal analysis takes about 30 s."/>
      </section>
      <section>
        {!o ? <Empty title="No nodal analysis yet">Run the nodal analysis to plot inflow against outflow.</Empty> : <>
          <div className="sectionHead"><div><h2>Nodal analysis: IPR vs VLP</h2><small>{run.name} · {when(run.finished_at)} · {o.model}</small></div></div>
          <Stale reasons={run.stale}/>
          {o.nodal ? <NodalChart o={o} current={current} name={w.name}/> : <p className="warning">This run predates nodal analysis. Run it again to plot IPR and VLP curves.</p>}
          {o.notes.map(n => <p key={n} className="muted small">{n}</p>)}
        </>}
      </section>
    </div>
    {o && <section>
      <h2>Operating points</h2>
      <p className="muted small">One point per {o.setpoint_label.toLowerCase()}, where its outflow curve meets the inflow curve. The optimizer uses these points.</p>
      <div className="grid3">
        <div><h3>Oil vs {o.setpoint_label.toLowerCase()} (m³/d)</h3><LineChart rows={merged} x="setpoint" xLabel={o.setpoint_unit} unit="m³/d" height={170} markers={markers} title={`${w.name} oil`}
          series={[{key: 'oil_m3d', label: 'Oil (nodal)'}, ...(rows.some(r => r.unstable_oil != null) ? [{key: 'unstable_oil', label: 'Unstable (casing heading)', dots: true, color: '#c0392b'}] : []),
                   ...(testedRows.length ? [{key: 'tested', label: 'Tested points', dots: true, color: '#2f5d9b'}] : [])]}/></div>
        {METRICS[w.type].filter(([k]) => k !== 'oil_m3d' && feasible.some(p => p[k] != null)).map(([k, label, unit], i) =>
          <div key={k}><h3>{label} ({unit})</h3><LineChart rows={feasible} x="setpoint" xLabel={o.setpoint_unit} unit={unit} height={170} markers={markers} title={`${w.name} ${label}`} series={[{key: k, label, color: COLORS[(i + 1) % COLORS.length]}]}/></div>)}</div>
      <div className="scroll"><table className="mini"><thead><tr><th>{o.setpoint_label} ({o.setpoint_unit})</th><th>Liquid</th><th>Oil</th><th>Water</th><th>Gas</th><th>BHP</th>{w.type === 'esp' ? <th>Power kW</th> : <><th>Valve</th><th>Stable</th></>}<th>Note</th></tr></thead>
        <tbody>{o.points.map((p, i) => <tr key={i} className={p.feasible ? (p.stable === false ? 'warnRow' : '') : 'muted'} onClick={() => p.feasible && select({title: `${w.name} at ${fmt(p.setpoint)} ${o.setpoint_unit}`, kind: 'curve-point', items: [['Oil', fmt(p.oil_m3d) + ' m³/d'], ['Liquid', fmt(p.liquid_m3d) + ' m³/d'], ['BHP', fmt(p.bhp_bar) + ' bar'], ['Note', p.note || 'Stable']], text: 'Operating point: the crossing of the inflow (IPR) and outflow (VLP) curves at this setpoint.'})}>
          <td>{p.setpoint != null ? fmt(p.setpoint) : `valve ${p.valve}`}</td><td>{fmt(p.liquid_m3d)}</td><td>{fmt(p.oil_m3d)}</td><td>{fmt(p.water_m3d)}</td><td>{fmt(p.gas_sm3d)}</td><td>{fmt(p.bhp_bar)}</td>
          {w.type === 'esp' ? <td>{fmt(p.power_kw)}</td> : <><td>{p.valve != null ? pct(Math.min(p.valve, 1)) : '—'}</td><td>{p.feasible ? (p.stable === false ? 'No' : 'Yes') : '—'}</td></>}<td>{p.note || ''}</td></tr>)}</tbody></table></div>
    </section>}
    {w.type === 'gas_lift' && <DatasetPanel wellId={w.id} dataset="gl_curve"/>}
    {edit && <WellForm well={w} onClose={() => setEdit(false)} onSaved={() => { setEdit(false); reloadWells(); }}/>}
  </>;
}

// ---------- Failure ----------

export function Failure() {
  const {route, set} = useCtx();
  const sub = route.sub === 'gas_lift' ? 'gas_lift' : 'esp';
  return <>
    <Head title="Failure" sub="Predictive maintenance"/>
    <Tabs tabs={[{id: 'esp', label: 'ESP'}, {id: 'gas_lift', label: 'Gas Lift'}]} active={sub} onChange={id => set({sub: id})} small/>
    <FailurePanel type={sub} key={sub}/>
  </>;
}

function envelopeFor(w, status, policy) {
  const s = w.settings, esp = w.type === 'esp';
  let lo = esp ? s.env_min_hz : s.env_min_gas, hi = esp ? s.env_max_hz : s.env_max_gas;
  const current = esp ? s.current_hz : s.current_gas_sm3d;
  let rule = 'Approved envelope';
  if (status === 'alert' && policy !== 'ignore') {
    if (current == null) return {rule: 'Needs a current setpoint (policy cannot apply)', lo, hi};
    if (policy === 'hold') { lo = current; hi = current; rule = 'Held at current setpoint (alert)'; }
    else { hi = hi != null ? Math.min(hi, current) : current; rule = 'Reduce only (alert)'; }
  }
  return {rule, lo, hi, unit: esp ? 'Hz' : 'sm³/d'};
}

function FailurePanel({type}) {
  const {runs, scenario, schema, select, go} = useCtx();
  const [w, picker] = useTypedWell([type]);
  const kind = type === 'esp' ? 'esp_risk' : 'valve_risk';
  const last = w && latest(runs, kind, {well_id: w.id});
  const run = useRun(last?.id);
  const curveRun = useRun(w && type === 'gas_lift' ? latest(runs, 'well_curve', {well_id: w.id})?.id : null);
  const o = run?.outputs;
  const [cursor, replay] = useReplay(o?.rows.length || 0);
  const [evidence, setEvidence] = useState(null);
  if (!w) return <NoWells types={[type]} what={type === 'esp' ? 'ESP failure risk uses daily pump sensor data.' : 'Gas-lift valve risk uses casing and tubing-head pressures.'}/>;
  const x = type === 'esp' ? 'date' : 'timestamp';
  const env = envelopeFor(w, o?.summary.status, scenario?.policy);
  const unstable = curveRun?.outputs?.points.filter(p => p.feasible && p.stable === false) || [];
  return <>
    <section>
      <div className="sectionHead"><div><h2>{w.name}: {type === 'esp' ? 'ESP failure risk' : 'gas-lift valve risk'}</h2><p>Probability of a {type === 'esp' ? 'pump failure' : 'confirmed valve leak or malfunction'} within the next 7 days.</p></div>{picker}</div>
      <RunBar kind={kind} label="Run risk model" params={{well_id: w.id}} filter={{well_id: w.id}}
        disabled={!w.counts[type === 'esp' ? 'esp_daily' : 'gl_pressure']?.rows} hint={`Uses ${type === 'esp' ? 'ESP daily sensors' : 'gas-lift pressures'} and maintenance events below.`}/>
      {o && <>
        <Stale reasons={run.stale}/>
        <div className="metrics">{[['Latest risk', fmt(o.summary.value, 3)], ['Status', <Status key="s" value={o.summary.status}/>], ['Horizon', `${o.horizon_days} days`],
          ['Alerts (recent)', o.summary.alerts], ...(type === 'gas_lift' ? [['Casing heading', <span key="h" className={'tag ' + (o.casing_heading === 'possible_heading' ? 'warnTag' : '')}>{String(o.casing_heading).replace(/_/g, ' ')}</span>]] : [])].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
        {o.warnings.map(t => <p key={t} className="warning">{t}</p>)}
        {replay}
        <LineChart rows={o.rows} x={x} threshold={o.threshold} cursor={cursor} title={`${w.name} risk`} onCursor={i => { const r = o.rows[i]; const ev = (o.evidence || []).find(e => (e.date || e.timestamp) === r[x]); if (ev) setEvidence(ev); }}
          markers={(o.events || []).map(e => ({x: e.timestamp, label: e.event_type.replace(/_/g, ' '), color: e.event_type.includes('leak') || e.event_type.includes('failure') || e.event_type.includes('malfunction') ? '#b23b3b' : '#8a4fa3'}))}
          series={[{key: 'risk', label: `Failure risk (${o.horizon_days}-day horizon)`}]}/>
      </>}
    </section>
    {o && <div className="grid2">
      <section>
        <h2>Alert evidence</h2>
        {!(o.evidence || []).length ? <p className="muted">No alerts in this history.</p> : <div className="scroll short"><table className="mini"><thead><tr><th>When</th><th>Risk</th><th>{type === 'esp' ? 'Furthest from training fleet' : 'Casing heading'}</th></tr></thead>
          <tbody>{o.evidence.slice().reverse().map((e, i) => <tr key={i} className="clickable" onClick={() => { setEvidence(e); select({title: `${w.name} alert`, kind: 'evidence', items: [['When', e.date || e.timestamp], ['Risk', fmt(e.risk, 3)], ['Threshold', fmt(o.threshold, 3)]],
            text: type === 'esp' ? `Sensors furthest from the training fleet: ${e.levels.map(l => `${l.sensor} (z ${fmt(l.z, 1)})`).join(', ')}. Largest one-day changes: ${e.changes.map(c => `${c.sensor} (${fmt(c.change_sd, 2)} sd)`).join(', ')}.` : `Casing heading screen: ${e.casing_heading || 'unavailable'}. Check the casing–tubing pressure difference around this time.`}); }}>
            <td>{e.date || e.timestamp}</td><td>{fmt(e.risk, 3)}</td><td>{type === 'esp' ? e.levels.map(l => `${l.sensor} ${fmt(l.z, 1)}σ`).join(', ') : (e.casing_heading || '—').replace(/_/g, ' ')}</td></tr>)}</tbody></table></div>}
        {evidence && type === 'esp' && <KV items={[['Selected alert', evidence.date], ...evidence.levels.map(l => [l.sensor, `${fmt(l.value)} (z ${fmt(l.z, 1)})`]), ...evidence.changes.map(c => [`Δ ${c.sensor}`, `${fmt(c.change_sd, 2)} sd/day`])]}/>}
      </section>
      <section>
        <h2>Operating envelope by policy</h2>
        <KV items={[['Scenario', scenario?.name], ['Policy', schema.policies[scenario?.policy] || '—'], ['Risk status', o.summary.status.replace('_', ' ')], ['Envelope applied', env.rule],
          ['Allowed setpoint', env.lo == null && env.hi == null ? 'No limits set' : `${fmt(env.lo)} – ${fmt(env.hi)} ${env.unit}`]]}/>
        <p className="muted">The Optimizer applies this envelope. Change the policy on the Facility tab; set the approved envelope on the well.</p>
        {type === 'gas_lift' && <>
          <h3>Casing heading</h3>
          <p>{`Pressure-cycle screen: ${String(o.casing_heading).replace(/_/g, ' ')}.`} {unstable.length ? `The well model predicts unstable flow (heading) at ${unstable.length} of ${curveRun.outputs.points.filter(p => p.feasible).length} lift-gas rates, ${fmt(Math.min(...unstable.map(p => p.setpoint)), 0)}–${fmt(Math.max(...unstable.map(p => p.setpoint)), 0)} sm³/d.` : curveRun ? 'The well model predicts stable flow at all computed rates.' : 'Run the well curve to see where the model predicts heading.'}</p>
          <button className="link" onClick={() => go('models', null, {well: w.id})}>Open gas-lift curve →</button>
        </>}
      </section>
    </div>}
    <DatasetPanel wellId={w.id} dataset={type === 'esp' ? 'esp_daily' : 'gl_pressure'}/>
    <DatasetPanel wellId={w.id} dataset="events"/>
  </>;
}

// ---------- Facility ----------

export function Facility() {
  const {scenario, scenarios, schema, reloadScenarios, notify, runs, mode} = useCtx();
  const [values, setValues] = useState(null), [policy, setPolicy] = useState('hold'), [notes, setNotes] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => { if (scenario) { setValues(Object.fromEntries(schema.facility.map(f => [f.key, scenario.facility[f.key] ?? '']))); setPolicy(scenario.policy); setNotes(scenario.notes); } }, [scenario?.id, scenario?.updated_at]); // eslint-disable-line react-hooks/exhaustive-deps
  const last = scenario && latest(runs, 'optimize', {scenario_id: scenario.id});
  const run = useRun(last?.id);
  const o = run?.outputs;
  if (!scenario || !values) return <><Head title="Facility" sub="Limits and capacity"/><section><p>Loading scenario…</p></section></>;
  async function save(e) {
    e.preventDefault(); setBusy(true);
    try { await api(`/scenarios/${scenario.id}`, {method: 'PUT', body: {name: scenario.name, facility: values, policy, notes}}); await reloadScenarios(); notify(`Saved ${scenario.name}`); }
    catch (err) { notify(err.message, 'error'); } finally { setBusy(false); }
  }
  const groups = [...new Set(schema.facility.map(f => f.group))];
  const t = o?.totals;
  return <>
    <Head title="Facility" sub={`Scenario: ${scenario.name}`}/>
    <div className="grid2 wide1">
      <section>
        <form onSubmit={save} className="form">
          <fieldset disabled={mode !== 'edit'} className="plain">
            {groups.map(g => <fieldset key={g} className="group"><legend>{g}</legend><div className="rowFields">
              {schema.facility.filter(f => f.group === g).map(f => <label key={f.key}>{f.label} ({f.unit})<input type="number" min="0" step="any" placeholder={f.default != null ? String(f.default) : 'no limit'} value={values[f.key]} onChange={e => setValues(v => ({...v, [f.key]: e.target.value}))}/></label>)}</div></fieldset>)}
            <fieldset className="group"><legend>Failure policy</legend>
              {Object.entries(schema.policies).map(([k, v]) => <label key={k} className="radio"><input type="radio" name="policy" checked={policy === k} onChange={() => setPolicy(k)}/>{v}</label>)}</fieldset>
            <label>Notes<input value={notes} onChange={e => setNotes(e.target.value)} maxLength={1000}/></label>
            <div className="formActions editOnly"><button disabled={busy}>{busy ? 'Saving…' : 'Save facility limits'}</button></div>
          </fieldset>
        </form>
        <p className="muted">Blank limits are unconstrained. Each scenario in this field ({scenarios.length}) keeps its own limits; use the Scenario menu to copy one.</p>
      </section>
      <section>
        <h2>Capacity used vs available</h2>
        {!o ? <Empty title="No optimizer run for this scenario">Run the Optimizer to see capacity use, gas disposition and energy.</Empty> : !o.feasible ? <p className="warning">{o.message}</p> : <>
          <small>From {run.name} · {when(run.finished_at)}</small>
          <Stale reasons={run.stale}/>
          {o.constraints.map(c => <Meter key={c.name} label={c.name} used={c.used} capacity={c.capacity} unit={c.unit} binding={c.binding}/>)}
          <h3>Gas disposition</h3>
          <KV items={[['Formation gas produced', fmt(t.gas, 0) + ' sm³/d'], ['Lift gas circulated (recompressed)', fmt(t.lift_gas, 0) + ' sm³/d'], ['Total gas through separator', fmt(t.total_gas, 0) + ' sm³/d'],
            ['Net gas available to export or fuel', fmt(t.gas, 0) + ' sm³/d']]}/>
          <h3>Energy and emissions</h3>
          <KV items={[['Compression and ESP energy', fmt(t.energy, 0) + ' kWh/d'], ['ESP power', fmt(t.power, 1) + ' kW'], ['Emissions', fmt(t.emissions, 0) + ' kg CO₂e/d'],
            ['Emissions intensity', t.oil ? fmt(t.emissions / t.oil, 2) + ' kg CO₂e per m³ oil' : '—']]}/>
        </>}
      </section>
    </div>
  </>;
}

// ---------- Optimizer ----------

export function Optimizer() {
  const {scenario, runs, set, route, go, select, mode} = useCtx();
  const [unstable, setUnstable] = useState(false), [shutIn, setShutIn] = useState(false);
  const mine = runs.filter(r => r.kind === 'optimize' && r.scenario_id === scenario?.id);
  const done = mine.filter(r => r.status === 'done');
  const rid = route.run && done.some(r => r.id === route.run) ? route.run : done[0]?.id;
  const run = useRun(rid);
  const o = run?.outputs;
  if (!scenario) return <><Head title="Optimizer" sub="Setpoint recommendations"/><section><p>Loading…</p></section></>;
  function explain(w) {
    set({well: w.id});
    const binding = o.constraints.filter(c => c.binding).map(c => c.name);
    select({title: `${w.well}: ${w.shut_in ? 'shut in' : `${sp(w.recommended, w.unit)} ${w.unit}`}`, kind: 'recommendation', wellId: w.id,
      items: [['Current', w.current != null ? `${sp(w.current, w.unit)} ${w.unit}` : 'not set'], ['Recommended', w.shut_in ? 'Shut in' : `${sp(w.recommended, w.unit)} ${w.unit}`],
              ['Oil now → predicted', `${fmt(w.current_oil)} → ${fmt(w.oil)} m³/d`], ['Risk', w.risk ? w.risk.replace('_', ' ') : 'not assessed'], ['Envelope rule', w.policy_applied],
              ['Curve', w.source.label], ['Binding constraints', binding.join(', ') || 'none']],
      text: w.shut_in ? 'Shutting in this well frees lift gas for wells that produce more oil per unit of gas.' :
        `Chosen from ${w.curve.length} allowed curve points to maximise field oil within the facility limits.${binding.length ? ` More oil here is blocked by: ${binding.join(', ')}.` : ''}${w.note ? ' ' + w.note + '.' : ''}`,
      links: [['Open well curve', () => go('models', null, {well: w.id})], ['Failure evidence', () => go('failure', w.lift === 'esp' ? 'esp' : 'gas_lift', {well: w.id})]]});
  }
  return <>
    <Head title="Optimizer" sub={`Scenario: ${scenario.name}`}/>
    <section className="editOnly">
      <div className="inline"><fieldset>
        <label className="check"><input type="checkbox" checked={shutIn} onChange={e => setShutIn(e.target.checked)}/>Allow shut-in</label>
        <label className="check"><input type="checkbox" checked={unstable} onChange={e => setUnstable(e.target.checked)}/>Allow unstable (heading) curve points</label>
      </fieldset></div>
      <RunBar kind="optimize" label="Run optimizer" params={{allow_shut_in: shutIn, allow_unstable: unstable}} filter={{scenario_id: scenario.id}} opts={{scenario_id: scenario.id}}
        hint="Uses each well's latest Well Models curve (or tested curve), its latest risk, and this scenario's facility limits and policy."/>
    </section>
    {!o ? <section><Empty title="No recommendation yet">Run the optimizer for this scenario.</Empty></section> : <>
      <section>
        <div className="sectionHead"><div><h2>{o.feasible ? 'Recommended setpoints' : 'No recommendation possible'}</h2><small>{run.name} · {when(run.finished_at)}</small></div>
          {done.length > 1 && <label className="inlineLabel">Run<select value={rid} onChange={e => set({run: e.target.value})}>{done.map(r => <option key={r.id} value={r.id}>{r.name} · {when(r.finished_at)}</option>)}</select></label>}</div>
        <Stale reasons={run.stale}/>
        {!o.feasible && <p className="warning"><b>Why:</b> {o.message}</p>}
        {o.feasible && <div className="metrics">{[['Field oil now', fmt(o.totals.current_oil, 0) + ' m³/d'], ['Recommended', fmt(o.totals.oil, 0) + ' m³/d'], ['Change', (o.totals.oil - o.totals.current_oil >= 0 ? '+' : '') + fmt(o.totals.oil - o.totals.current_oil, 0) + ' m³/d'],
          ['Lift gas', fmt(o.totals.lift_gas, 0) + ' sm³/d']].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>}
        {o.feasible && <div className="scroll"><table><thead><tr><th>Well</th><th>Risk</th><th>Envelope rule</th><th>Current</th><th>Recommended</th><th>Oil now</th><th>Oil predicted</th><th>Change</th><th>Curve</th><th/></tr></thead>
          <tbody>{o.wells.map(w => <tr key={w.id} className="clickable" onClick={() => explain(w)}>
            <td><b>{w.well}</b> <span className={'tag ' + typeTone[w.lift]}>{w.lift === 'esp' ? 'ESP' : 'GL'}</span></td><td><Status value={w.risk}/></td><td>{w.policy_applied}</td>
            <td>{w.current != null ? `${sp(w.current, w.unit)} ${w.unit}` : '—'}</td><td><b>{w.shut_in ? 'Shut in' : `${sp(w.recommended, w.unit)} ${w.unit}`}</b></td>
            <td>{fmt(w.current_oil, 0)}</td><td>{fmt(w.oil, 0)}</td><td className={w.delta_oil > 0 ? 'good' : w.delta_oil < 0 ? 'dangerText' : ''}>{w.delta_oil != null ? (w.delta_oil > 0 ? '+' : '') + fmt(w.delta_oil, 0) : '—'}</td>
            <td><Spark curve={w.curve} chosen={w.recommended} current={w.current}/></td>
            <td onClick={e => e.stopPropagation()}><button className="link" onClick={() => go('models', null, {well: w.id})}>Curve</button><button className="link" onClick={() => go('failure', w.lift === 'esp' ? 'esp' : 'gas_lift', {well: w.id})}>Failure</button></td></tr>)}</tbody></table></div>}
        {o.excluded.length > 0 && <><h3>Not optimised</h3><table className="mini"><tbody>{o.excluded.map(e => <tr key={e.id}><td><b>{e.well}</b></td><td>{e.reason}</td></tr>)}</tbody></table></>}
      </section>
      <section>
        <h2>Constraints</h2>
        {o.constraints.map(c => c.used != null ? <Meter key={c.name} label={c.name} used={c.used} capacity={c.capacity} unit={c.unit} binding={c.binding}/> :
          <p key={c.name} className="muted">{c.name}: needs at least {fmt(c.minimum_needed, 0)} {c.unit}; limit {fmt(c.capacity, 0)}.</p>)}
        {!o.constraints.length && <p className="muted">No facility limits set in this scenario.</p>}
        <p className="muted">{o.policy} · Unstable points {o.allow_unstable ? 'allowed' : 'excluded'} · Shut-in {o.allow_shut_in ? 'allowed' : 'not allowed'}.</p>
      </section>
    </>}
  </>;
}

const sp = (v, unit) => v == null ? '—' : fmt(v, unit === 'Hz' ? 1 : 0);

function Spark({curve, chosen, current}) {
  if (!curve?.length) return null;
  const xs = curve.map(c => c.setpoint), ys = curve.map(c => c.oil);
  const x0 = Math.min(...xs, chosen ?? Infinity, current ?? Infinity), x1 = Math.max(...xs, chosen ?? -Infinity, current ?? -Infinity), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = v => 2 + ((v - x0) / ((x1 - x0) || 1)) * 96, py = v => 22 - ((v - y0) / ((y1 - y0) || 1)) * 20;
  const pts = curve.slice().sort((a, b) => a.setpoint - b.setpoint);
  const pick = pts.find(p => Math.abs(p.setpoint - chosen) < 1e-6);
  return <svg viewBox="0 0 100 24" className="spark"><polyline points={pts.map(p => `${px(p.setpoint)},${py(p.oil)}`).join(' ')} fill="none" stroke="#007953" strokeWidth="1.5"/>
    {current != null && <line x1={px(current)} x2={px(current)} y1="0" y2="24" stroke="#c0392b" strokeDasharray="2 2"/>}
    {pick && <circle cx={px(pick.setpoint)} cy={py(pick.oil)} r="2.8" fill="#2f5d9b"/>}</svg>;
}
