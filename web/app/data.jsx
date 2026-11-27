'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, download, upload, fmt, when, header, qs } from './lib';
import { useCtx } from './ctx';
import { LineChart, Modal, Confirm, Empty } from './components';
import { CsvTable, toCsv } from './table';

export function Templates({name, label = 'Template'}) {
  const {notify} = useCtx();
  return <span className="templates">{label}: <button className="link" onClick={() => download(`/templates/${name}?format=csv`).catch(e => notify(e.message, 'error'))}>CSV</button>
    <button className="link" onClick={() => download(`/templates/${name}?format=xlsx`).catch(e => notify(e.message, 'error'))}>Excel</button></span>;
}

// ---------- well form ----------

export function WellForm({well, onClose, onSaved, defaults = {}}) {
  const {schema, field, fields, notify} = useCtx();
  const [values, setValues] = useState({code: well?.code || '', name: well?.name || '', type: well?.type || defaults.type || 'esp',
    field: well?.field ?? defaults.field ?? field ?? '', status: well?.status || 'active', notes: well?.notes || '', settings: {...(well?.settings || {})}});
  const [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const set = (k, v) => setValues(s => ({...s, [k]: v}));
  const settings = schema.settings[values.type] || [];
  const groups = [...new Set(settings.map(s => s.group))];
  async function save(e) {
    e.preventDefault(); setBusy(true); setError('');
    try {
      const body = {...values, settings: Object.fromEntries(settings.map(s => [s.key, values.settings[s.key] ?? s.default ?? null]))};
      const saved = await api(well ? `/wells/${well.id}` : '/wells', {method: well ? 'PUT' : 'POST', body});
      notify(well ? `Saved ${saved.name}` : `Added ${saved.name}`);
      onSaved(saved);
    } catch (err) { setError(err.message); setBusy(false); }
  }
  return <Modal title={well ? `Edit ${well.name}` : 'Add well'} onClose={onClose} wide>
    <form onSubmit={save} className="form">
      <div className="row3">
        <label>Well ID<input required maxLength={40} value={values.code} onChange={e => set('code', e.target.value)} placeholder="e.g. P-01" autoFocus/></label>
        <label>Name<input required maxLength={80} value={values.name} onChange={e => set('name', e.target.value)}/></label>
        <label>Role and lift<select value={values.type} disabled={!!well} onChange={e => set('type', e.target.value)}>
          {Object.entries(schema.wellTypes).map(([k, t]) => <option key={k} value={k}>{t.label}</option>)}</select></label>
      </div>
      <div className="row3">
        <label>Field<input list="fieldNames" maxLength={80} value={values.field} onChange={e => set('field', e.target.value)}/>
          <datalist id="fieldNames">{(fields || []).map(f => <option key={f.name} value={f.name}/>)}</datalist></label>
        <label>Status<select value={values.status} onChange={e => set('status', e.target.value)}>{schema.statuses.map(s => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}</select></label>
        <label>Notes<input maxLength={2000} value={values.notes} onChange={e => set('notes', e.target.value)}/></label>
      </div>
      <small>{well ? 'Role and lift type are fixed after creation.' : `Data for this well: ${schema.wellTypes[values.type].datasets.map(d => schema.datasets[d].label).join(', ')}.`}</small>
      {groups.map(g => <fieldset key={g} className="group"><legend>{g}</legend><div className="rowFields">
        {settings.filter(s => s.group === g).map(s => <label key={s.key} title={s.help}>{s.label}{s.unit ? ` (${s.unit})` : ''}{s.default === undefined ? ' — optional' : ''}
          <input type="number" step="any" min={s.min} max={s.max} required={s.default !== undefined} value={values.settings[s.key] ?? (s.default ?? '')}
            onChange={e => setValues(v => ({...v, settings: {...v.settings, [s.key]: e.target.value}}))}/></label>)}</div></fieldset>)}
      {error && <p className="formError" role="alert">{error}</p>}
      <div className="formActions"><button type="button" className="secondary" onClick={onClose}>Cancel</button><button disabled={busy}>{busy ? 'Saving…' : well ? 'Save changes' : 'Add well'}</button></div>
    </form>
  </Modal>;
}

// ---------- staged CSV / Excel import ----------

export function ImportWizard({kind, onClose, onDone}) {
  const {schema, field, notify} = useCtx();
  const [preview, setPreview] = useState(null), [mapping, setMapping] = useState({}), [unit, setUnit] = useState('m3/d');
  const [busy, setBusy] = useState(false), [error, setError] = useState(''), [result, setResult] = useState(null), [table, setTable] = useState(false);
  const file = useRef(null);
  const title = {production: 'Import production history', injection: 'Import reservoir injection history', wells: 'Import wells'}[kind];
  async function pick(e) {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    await previewFile(f);
  }
  async function previewFile(f) {
    setBusy(true); setError(''); setPreview(null);
    try {
      const p = await upload(`/imports/${kind}/preview${qs({field})}`, f);
      setPreview(p); setUnit(p.unit || 'm3/d');
      if (p.columns) setMapping(Object.fromEntries(p.columns.map(c => [c.source, {action: c.action, well_id: c.well_id, type: c.suggested_type || (kind === 'injection' ? 'injector' : 'producer')}])));
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  async function commit() {
    setBusy(true); setError('');
    try {
      const r = await api(`/imports/${preview.token}/commit`, {method: 'POST', body: {mapping, unit, field}});
      setResult(r); onDone?.(r);
      notify('Import complete');
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  const role = kind === 'injection' ? 'injector' : 'producer';
  const types = Object.entries(schema.wellTypes).filter(([, t]) => t.role === role);
  return <Modal title={title} onClose={onClose} wide>
    {!preview && !result && <div className="stack">
      {kind === 'production' && <p>Daily <b>total liquid production</b> per producer. Wide format (a date column, then one column per well ID or name) or long format (well, date, rate). This is not lift gas.</p>}
      {kind === 'injection' && <p><b>Reservoir water injection</b> per injector. Same wide or long layout. Artificial-lift gas belongs to gas-lift pressures and Well Models, not here.</p>}
      {kind === 'wells' && <p>One row per well: <code>well_id, name, role, lift_type, field, status, notes</code>. Role is producer or injector; lift type is esp, gas_lift or natural. Existing wells (same ID or name) are updated.</p>}
      <div className="toolbar"><button disabled={busy} onClick={() => file.current.click()}>{busy ? 'Reading file…' : 'Import CSV'}</button>
        {kind === 'wells' && <button className="secondary" onClick={() => setTable(true)}>Fill table</button>}
        <input ref={file} type="file" accept=".csv,.txt,.tsv,.xlsx,.xlsm,.xls" hidden onChange={pick}/><Templates name={kind}/></div>
      {table && <CsvTable title="Wells" note="Same layout as the well-list CSV, including its header row. Role is producer or injector; lift type is esp, gas_lift or natural."
        columns={[{key: 'well_id', label: 'well_id', required: true}, {key: 'name', label: 'name'}, {key: 'role', label: 'role', options: ['producer', 'injector']},
                  {key: 'lift_type', label: 'lift_type', options: ['esp', 'gas_lift', 'natural']}, {key: 'field', label: 'field'},
                  {key: 'status', label: 'status', options: schema.statuses}, {key: 'notes', label: 'notes'}]}
        onClose={() => setTable(false)} saveLabel="Preview import"
        onSave={async rows => { setTable(false); await previewFile(new File([toCsv(['well_id', 'name', 'role', 'lift_type', 'field', 'status', 'notes'], rows)], 'wells.csv', {type: 'text/csv'})); }}/>}
      {error && <p className="formError">{error}</p>}
    </div>}
    {preview && !result && kind !== 'wells' && <div className="stack">
      <div className="metrics compact">{[['File', preview.filename], ['Layout', preview.layout.format === 'wide' ? 'Wide (column per well)' : 'Long (well, date, rate)'],
        ['Date range', `${preview.date_range[0]} → ${preview.date_range[1]}`], ['Dates', fmt(preview.dates, 0)]].map(([k, v]) => <div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
      <p className="muted">Reading as <b>{preview.label}</b>. Date column: <code>{preview.layout.date_column}</code>{preview.layout.well_column && <> · well column <code>{preview.layout.well_column}</code> · rate column <code>{preview.layout.value_column}</code></>}.
        {preview.unparsed_dates > 0 && <span className="dangerText"> {preview.unparsed_dates} rows have unreadable dates and will be skipped.</span>}</p>
      <label className="inlineLabel">Units in the file<select value={unit} onChange={e => setUnit(e.target.value)}>{preview.units.map(u => <option key={u}>{u}</option>)}</select><small>Stored as m³/d.</small></label>
      <div className="scroll"><table><thead><tr><th>Column</th><th>Action</th><th>Well</th><th>Values</th><th>Missing</th><th>Duplicates</th><th>Negative</th><th>First</th><th>Last</th><th>Step (days)</th></tr></thead>
        <tbody>{preview.columns.map(c => { const m = mapping[c.source] || {}; return <tr key={c.source} className={c.status === 'role_mismatch' ? 'warnRow' : ''}>
          <td><b>{c.source}</b></td>
          <td><select value={m.action} onChange={e => setMapping(x => ({...x, [c.source]: {...m, action: e.target.value}}))}>
            {c.well_id && c.status === 'matched' && <option value="map">Import into well</option>}<option value="create">Create new well</option><option value="ignore">Ignore</option></select></td>
          <td>{m.action === 'create' ? <select value={m.type} onChange={e => setMapping(x => ({...x, [c.source]: {...m, type: e.target.value}}))}>{types.map(([k, t]) => <option key={k} value={k}>{t.label}</option>)}</select>
            : c.well_name ? <>{c.well_name}{c.status === 'role_mismatch' && <small className="dangerText">Not a {role}</small>}</> : '—'}</td>
          <td>{fmt(c.values, 0)}</td><td className={c.missing ? 'warnCell' : ''}>{c.missing}</td><td className={c.duplicates ? 'warnCell' : ''}>{c.duplicates}</td>
          <td className={c.negative ? 'warnCell' : ''}>{c.negative}</td><td>{c.first}</td><td>{c.last}</td><td>{fmt(c.typical_step_days, 1)}</td></tr>; })}</tbody></table></div>
      {preview.gap_count > 0 && <div className="warning"><b>{preview.gap_count} missing interval(s)</b> longer than 1.5× the usual spacing:
        <ul>{preview.gaps.slice(0, 8).map((g, i) => <li key={i}>{g.source}: {g.days} days after {g.start}</li>)}</ul></div>}
      <p className="muted">Duplicate dates keep the last value. Existing rows with the same date are replaced.</p>
      {error && <p className="formError">{error}</p>}
      <div className="formActions"><button className="secondary" onClick={() => setPreview(null)}>Choose another file</button>
        <button disabled={busy || !Object.values(mapping).some(m => m.action !== 'ignore')} onClick={commit}>{busy ? 'Importing…' : 'Import'}</button></div>
    </div>}
    {preview && !result && kind === 'wells' && <div className="stack">
      <div className="scroll"><table><thead><tr><th>Row</th><th>Well ID</th><th>Name</th><th>Type</th><th>Field</th><th>Status</th><th>Action</th></tr></thead>
        <tbody>{preview.rows.map(r => <tr key={r.line} className={r.problem ? 'warnRow' : ''}><td>{r.line}</td><td>{r.code}</td><td>{r.name}</td>
          <td>{r.type ? schema.wellTypes[r.type].label : '—'}</td><td>{r.field || '—'}</td><td>{r.status}</td><td>{r.problem ? <span className="dangerText">{r.problem}</span> : r.action}</td></tr>)}</tbody></table></div>
      {error && <p className="formError">{error}</p>}
      <div className="formActions"><button className="secondary" onClick={() => setPreview(null)}>Choose another file</button>
        <button disabled={busy || !preview.rows.some(r => !r.problem)} onClick={commit}>{busy ? 'Importing…' : `Import ${preview.rows.filter(r => !r.problem).length} wells`}</button></div>
    </div>}
    {result && <div className="stack">
      {result.created?.length > 0 && <p>Created: {result.created.join(', ')}</p>}
      {result.updated?.length > 0 && <p>Updated: {result.updated.join(', ')}</p>}
      {result.wells && <table><thead><tr><th>Column</th><th>Added</th><th>Updated</th><th>Rejected</th><th>Version</th></tr></thead>
        <tbody>{result.wells.map(w => <tr key={w.source}><td>{w.source}</td><td>{w.inserted}</td><td>{w.updated}</td><td>{w.rejected}</td><td>v{w.version}</td></tr>)}</tbody></table>}
      {result.skipped?.length > 0 && <p className="muted">Ignored: {result.skipped.join(', ')}</p>}
      {result.problems?.length > 0 && <div className="warning"><ul>{result.problems.map(p => <li key={p}>{p}</li>)}</ul></div>}
      <div className="formActions"><button onClick={onClose}>Done</button></div>
    </div>}
  </Modal>;
}

// ---------- per-well dataset editor ----------

const PAGE = 50;

export function DatasetPanel({wellId, dataset, compact, onChange}) {
  const {schema, notify, reloadWells} = useCtx();
  const spec = schema.datasets[dataset];
  const [data, setData] = useState(null), [offset, setOffset] = useState(0), [selected, setSelected] = useState(new Set());
  const [adding, setAdding] = useState(false), [editing, setEditing] = useState(null), [report, setReport] = useState(null);
  const [busy, setBusy] = useState(false), [confirm, setConfirm] = useState(null), [history, setHistory] = useState(false), [table, setTable] = useState(false);
  const file = useRef(null);
  const base = `/wells/${wellId}/data/${dataset}`;
  const load = useCallback(() => api(`${base}?offset=${offset}&limit=${PAGE}`).then(setData).catch(e => notify(e.message, 'error')), [base, offset, notify]);
  useEffect(() => { load(); setSelected(new Set()); }, [load]);
  const changed = () => { load(); reloadWells(); onChange?.(); setSelected(new Set()); };
  async function onFile(e) {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    setBusy(true); setReport(null);
    try { const r = await upload(`${base}/upload`, f); setReport({...r, file: f.name}); setOffset(0); changed(); }
    catch (err) { notify(err.message, 'error'); } finally { setBusy(false); }
  }
  async function remove(all) {
    await api(base, {method: 'DELETE', body: all ? {all: true} : {ids: [...selected]}}).then(r => notify(`Deleted ${r.deleted} rows`)).catch(e => notify(e.message, 'error'));
    changed();
  }
  const cols = spec.columns;
  const total = data?.total || 0;
  return <section className={compact ? 'flat' : ''}>
    <div className="sectionHead"><div><h3>{spec.label}</h3><p>{spec.description}</p></div>
      {data?.versions?.[0] && <button className="link small" onClick={() => setHistory(!history)}>v{data.versions[0].version} · {data.versions[0].source} · {when(data.versions[0].created_at)}</button>}</div>
    {history && <table className="mini"><thead><tr><th>Version</th><th>When</th><th>Source</th><th>Rows</th></tr></thead>
      <tbody>{data.versions.map(v => <tr key={v.version}><td>v{v.version}</td><td>{when(v.created_at)}</td><td>{v.source}</td><td>{fmt(v.rows, 0)}</td></tr>)}</tbody></table>}
    <div className="toolbar">
      <button className="editOnly" onClick={() => { setTable(true); setEditing(null); }}>Fill table</button>
      <button className="secondary editOnly" disabled={busy} onClick={() => file.current.click()}>{busy ? 'Importing…' : 'Import CSV'}</button>
      <input ref={file} type="file" accept=".csv,.txt,.tsv,.xlsx,.xlsm,.xls" hidden onChange={onFile}/>
      <Templates name={dataset}/>
      {total > 0 && <span className="templates">Download: <button className="link" onClick={() => download(`${base}/export?format=csv`).catch(e => notify(e.message, 'error'))}>CSV</button>
        <button className="link" onClick={() => download(`${base}/export?format=xlsx`).catch(e => notify(e.message, 'error'))}>Excel</button></span>}
      <span className="spacer"/>
      {selected.size > 0 && <button className="secondary dangerText editOnly" onClick={() => setConfirm('some')}>Delete {selected.size}</button>}
      {total > 0 && <button className="link dangerText editOnly" onClick={() => setConfirm('all')}>Clear all</button>}
    </div>
    {report && <div className={'importReport' + (report.rejected ? ' warn' : '')}>
      <b>{report.file}:</b> {report.inserted} added, {report.updated} updated{report.rejected ? `, ${report.rejected} rejected` : ''}{report.version ? ` → v${report.version}` : ''}.
      {report.ignored_columns?.length > 0 && <> Ignored columns: {report.ignored_columns.join(', ')}.</>}
      {report.problems?.length > 0 && <ul>{report.problems.map(p => <li key={p}>{p}</li>)}</ul>}
      <button className="link" onClick={() => setReport(null)}>Dismiss</button></div>}
    {table && <CsvTable title={spec.label} note={`Same columns as the ${spec.label} CSV template (header row included). Rows with the same ${cols[0].label.toLowerCase()} replace stored rows.`}
      columns={cols.map(c => ({key: c.key, label: header(c), aliases: [c.key, c.label], required: c.required, options: c.kind === 'choice' ? c.options : undefined}))}
      onClose={() => setTable(false)} saveLabel="Save rows"
      onSave={async rows => {
        const r = await api(base, {method: 'POST', body: {rows: rows.map((row, i) => Object.fromEntries([...cols.map((c, j) => [c.key, row[j] === '' ? null : row[j]]), ['_line', i + 1]]))}});
        if (r.rejected && !r.inserted && !r.updated) throw new Error(r.problems.slice(0, 5).join(' · '));
        setReport({...r, file: 'Table'}); setTable(false); changed();
      }}/>}
    {adding && <RowForm cols={cols} title="New row" onCancel={() => setAdding(false)} onSave={async row => {
      const r = await api(base, {method: 'POST', body: {rows: [row]}});
      if (r.rejected) throw new Error(r.problems[0].replace(/^Row 1: /, ''));
      notify(r.updated ? 'Row replaced (same key)' : 'Row added'); setAdding(false); changed();
    }}/>}
    {total > 1 && spec.chart.length > 0 && <LineChart rows={data.series} x={spec.key} height={180} xLabel={spec.key === 'gas_sm3d' ? 'sm³/d lift gas' : undefined}
      series={spec.chart.map(k => ({key: k, label: header(cols.find(c => c.key === k))}))} title={spec.label}/>}
    {!data ? <p>Loading…</p> : !total ? <Empty title="No rows yet">Add rows by hand, or fill the CSV/Excel template and upload it. Rows with the same {cols[0].label.toLowerCase()} replace existing rows.</Empty> :
      <><div className="scroll"><table>
        <thead><tr><th className="check editOnly"><input type="checkbox" aria-label="Select page" checked={data.rows.length > 0 && data.rows.every(r => selected.has(r.id))}
          onChange={e => setSelected(e.target.checked ? new Set(data.rows.map(r => r.id)) : new Set())}/></th>{cols.map(c => <th key={c.key}>{header(c)}</th>)}<th className="editOnly"/></tr></thead>
        <tbody>{data.rows.map(r => editing === r.id ?
          <tr key={r.id}><td colSpan={cols.length + 2}><RowForm cols={cols} initial={r} title="Edit row" onCancel={() => setEditing(null)} onSave={async row => {
            await api(`${base}/${r.id}`, {method: 'PUT', body: {row}}); notify('Row updated'); setEditing(null); changed(); }}/></td></tr> :
          <tr key={r.id} className={selected.has(r.id) ? 'selected' : ''}>
            <td className="check editOnly"><input type="checkbox" aria-label="Select row" checked={selected.has(r.id)} onChange={e => { const s = new Set(selected); e.target.checked ? s.add(r.id) : s.delete(r.id); setSelected(s); }}/></td>
            {cols.map(c => <td key={c.key}>{c.kind === 'number' ? fmt(r[c.key], 4) : (r[c.key] ?? '—')}</td>)}
            <td className="editOnly"><button className="link" onClick={() => { setEditing(r.id); setAdding(false); }}>Edit</button></td>
          </tr>)}</tbody>
      </table></div>
      <div className="pager"><span>Rows {fmt(offset + 1, 0)}–{fmt(Math.min(offset + PAGE, total), 0)} of {fmt(total, 0)}{data.newest_first ? ' (newest first)' : ''}</span>
        <button className="secondary small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
        <button className="secondary small" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next</button></div></>}
    {confirm && <Confirm title={confirm === 'all' ? `Clear all ${spec.label.toLowerCase()}?` : `Delete ${selected.size} rows?`}
      message={confirm === 'all' ? `This deletes all ${fmt(total, 0)} rows. A new dataset version is recorded, and runs that used the old data are flagged for review.` : 'The selected rows will be permanently deleted.'}
      onClose={() => setConfirm(null)} onConfirm={() => remove(confirm === 'all')}/>}
  </section>;
}

function toInput(col, value) {
  if (value == null) return '';
  if (col.kind === 'time') return String(value).replace('Z', '').slice(0, 19);
  return String(value);
}

function RowForm({cols, initial, title, onSave, onCancel}) {
  const [values, setValues] = useState(() => Object.fromEntries(cols.map(c => [c.key, toInput(c, initial?.[c.key])])));
  const [error, setError] = useState(''), [busy, setBusy] = useState(false);
  async function submit(e) {
    e.preventDefault(); setBusy(true); setError('');
    const row = Object.fromEntries(cols.map(c => [c.key, values[c.key] === '' ? null : c.kind === 'time' ? values[c.key] + 'Z' : values[c.key]]));
    try { await onSave(row); } catch (err) { setError(err.message); setBusy(false); }
  }
  const input = c => c.kind === 'choice' ? <select required={c.required} value={values[c.key]} onChange={e => setValues(v => ({...v, [c.key]: e.target.value}))}>
      <option value="">{c.required ? 'Choose…' : '—'}</option>{c.options.map(o => <option key={o} value={o}>{o.replace(/_/g, ' ')}</option>)}</select> :
    <input required={c.required} type={c.kind === 'date' ? 'date' : c.kind === 'time' ? 'datetime-local' : c.kind === 'text' ? 'text' : 'number'} step={c.kind === 'time' ? 1 : 'any'}
      min={c.min} max={c.max} maxLength={c.kind === 'text' ? 500 : undefined} value={values[c.key]} onChange={e => setValues(v => ({...v, [c.key]: e.target.value}))}/>;
  return <form className="rowForm" onSubmit={submit}>
    <b>{title}</b>
    <div className="rowFields">{cols.map(c => <label key={c.key}>{header(c)}{c.required ? '' : ' (optional)'}{input(c)}</label>)}</div>
    {error && <p className="formError" role="alert">{error}</p>}
    <div className="formActions"><button type="button" className="secondary" onClick={onCancel}>Cancel</button><button disabled={busy}>{busy ? 'Saving…' : 'Save row'}</button></div>
  </form>;
}

/** Status line for the latest run of a kind, with a run button. */
export function RunBar({kind, label, params, filter, opts, disabled, hint}) {
  const {runs, startRun, mode} = useCtx();
  const active = runs.find(r => r.kind === kind && ['queued', 'running'].includes(r.status) && Object.entries(filter || {}).every(([k, v]) => r[k] === v || r.params?.[k] === v));
  const last = runs.find(r => r.kind === kind && ['done', 'failed'].includes(r.status) && Object.entries(filter || {}).every(([k, v]) => r[k] === v || r.params?.[k] === v));
  return <div className="runBar">
    {mode === 'edit' && <button disabled={disabled || !!active} onClick={() => startRun(kind, typeof params === 'function' ? params() : params, opts)}>
      {active ? (active.status === 'queued' ? 'Queued…' : 'Running…') : label}</button>}
    <span className="muted">{active ? `${active.name} is ${active.status}. You can keep working; results appear when it finishes.` :
      last ? <>{last.status === 'failed' ? <span className="dangerText">Last run failed: {last.error}</span> : <>Last run {when(last.finished_at)}</>}{last.stale?.length > 0 && <span className="tag warnTag">Needs review</span>}</> : hint || 'Not run yet.'}</span>
  </div>;
}
